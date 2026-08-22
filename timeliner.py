"""
Usage:
  python3 build_timeline.py --acc rec_acc.dat --gyro rec_gyro.dat --model rf
  python3 build_timeline.py --acc rec_acc.dat --gyro rec_gyro.dat --model cnn
"""
import argparse, json, os
import numpy as np
from scipy.signal import butter, filtfilt

TARGET_FS = 40
WIN_SEC   = 3.0
WIN       = int(WIN_SEC * TARGET_FS)
SMOOTH_K  = 5            # median filter size (odd) over window predictions
MIN_INTERVAL_SEC = 3.0   # drop intervals shorter than this (micro-blips)

# ---------------- feature extraction (must MATCH training) ----------------
B_LP, A_LP = butter(3, 0.3/(TARGET_FS/2), btype="low")

def stats(x):
    return [x.mean(), x.std(), x.min(), x.max(),
            np.mean(np.abs(x-x.mean())), np.mean(x**2)]

def spectral(x):
    xz = x - x.mean()
    sp = np.abs(np.fft.rfft(xz))**2
    fr = np.fft.rfftfreq(len(x), 1/TARGET_FS)
    tot = sp.sum() + 1e-12
    bands = []
    for lo, hi in [(0.3,1.0),(1.0,2.0),(2.0,3.5),(3.5,8.0)]:
        m = (fr>=lo)&(fr<hi); bands.append(sp[m].sum()/tot)
    p = sp/tot
    return [fr[int(np.argmax(sp))], sp.max()/tot] + bands + [-np.sum(p*np.log(p+1e-12))]

def autocorr(x):
    xz = x-x.mean()
    ac = np.correlate(xz,xz,mode="full")[len(xz)-1:]
    ac = ac/(ac[0]+1e-12)
    lo,hi = int(TARGET_FS*0.25), int(TARGET_FS*2.0)
    seg = ac[lo:hi]
    if len(seg)==0: return [0.0,0.0]
    k = int(np.argmax(seg)); return [float(seg[k]), float((lo+k)/TARGET_FS)]

def features(w):
    acc, gyr = w[:,:3], w[:,3:]
    grav = filtfilt(B_LP, A_LP, acc, axis=0); body = acc-grav
    f=[]
    gm = grav.mean(0); gn = np.linalg.norm(gm)+1e-12
    f += list(gm) + list(np.arccos(np.clip(gm/gn,-1,1))) + [gn, grav.std(0).mean()]
    chans = [body[:,i] for i in range(3)]+[gyr[:,i] for i in range(3)]
    chans += [np.linalg.norm(body,axis=1), np.linalg.norm(gyr,axis=1)]
    for x in chans: f += stats(x)+spectral(x)
    f += autocorr(np.linalg.norm(body,axis=1)) + autocorr(np.linalg.norm(gyr,axis=1))
    for a,b in [(0,1),(0,2),(1,2)]:
        f.append(float(np.corrcoef(body[:,a],body[:,b])[0,1]))
        f.append(float(np.corrcoef(gyr[:,a],gyr[:,b])[0,1]))
    return np.nan_to_num(np.array(f,dtype=np.float32))

# ---------------- evidence stats (used later for Explanations) -------------
def window_evidence(w):
    acc, gyr = w[:,:3], w[:,3:]
    grav = filtfilt(B_LP, A_LP, acc, axis=0); body = acc - grav
    bm = np.linalg.norm(body, axis=1); gm = np.linalg.norm(gyr, axis=1)
    sp = np.abs(np.fft.rfft(bm - bm.mean()))**2
    fr = np.fft.rfftfreq(len(bm), 1/TARGET_FS)
    return {
        "acc_motion_std":   round(float(bm.std()), 4),
        "gyro_energy":      round(float(np.mean(gm**2)), 4),
        "dominant_freq_hz": round(float(fr[int(np.argmax(sp))]), 2),
        "tilt_z":           round(float(grav[:,2].mean()), 3),
    }

# ---------------- loading + windowing (any input rate) ---------------------
def load_raw(path):
    a = np.loadtxt(path)
    t = a[:,0].astype(np.float64); xyz = a[:,1:4].astype(np.float32)
    o = np.argsort(t); t, xyz = t[o], xyz[o]
    return t - t[0], xyz

def cut_windows(t_acc, acc, t_gyr, gyr):
    """NON-overlapping WIN_SEC windows, each resampled to WIN samples."""
    dur = min(t_acc[-1], t_gyr[-1])
    wins, times = [], []
    start = 0.0
    while start + WIN_SEC <= dur + 1e-9:
        end = start + WIN_SEC
        grid = np.linspace(start, end, WIN, endpoint=False)
        ma = (t_acc>=start)&(t_acc<=end); mg = (t_gyr>=start)&(t_gyr<=end)
        if ma.sum() >= 5 and mg.sum() >= 5:
            w = np.empty((WIN,6), dtype=np.float32)
            for c in range(3):
                w[:,c]   = np.interp(grid, t_acc[ma], acc[ma,c])
                w[:,c+3] = np.interp(grid, t_gyr[mg], gyr[mg,c])
            wins.append(w); times.append((start, end))
        start += WIN_SEC          # NON-overlapping
    return wins, times

# ---------------- prediction ----------------------------------------------
def predict_rf(wins, model_path="rf_model_40hz.joblib"):
    import joblib
    clf = joblib.load(model_path)
    F = np.stack([features(w) for w in wins])
    proba = clf.predict_proba(F)
    labels = clf.classes_[proba.argmax(1)]
    conf = proba.max(1)
    return labels, conf

def predict_cnn(wins, model_path="cnn_model_40hz.pt"):
    import torch, torch.nn as nn
    ck = torch.load(model_path, map_location="cpu")
    classes, mean, std = ck["classes"], ck["mean"], ck["std"]
    class CNN(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(6,64,15,padding=7), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64,128,9,padding=4),nn.BatchNorm1d(128),nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(128,128,5,padding=2),nn.BatchNorm1d(128),nn.ReLU(),
                nn.AdaptiveAvgPool1d(1))
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128,n))
        def forward(self,x): return self.head(self.net(x))
    m = CNN(len(classes)); m.load_state_dict(ck["state_dict"]); m.eval()
    Xb = np.stack([(w-mean)/std for w in wins]).transpose(0,2,1)
    with torch.no_grad():
        p = torch.softmax(m(torch.tensor(Xb, dtype=torch.float32)), dim=1).numpy()
    return np.array([classes[i] for i in p.argmax(1)]), p.max(1)

# ---------------- smoothing + merging --------------------------------------
def smooth(labels, k=SMOOTH_K):
    """Majority vote over a sliding window of k -> removes 1-window flickers."""
    out = list(labels); n = len(labels); h = k//2
    for i in range(n):
        lo, hi = max(0,i-h), min(n,i+h+1)
        seg = list(labels[lo:hi])
        out[i] = max(set(seg), key=seg.count)
    return np.array(out)

def merge(labels, times, conf, evid):
    """Merge consecutive same-label windows into intervals."""
    intervals = []
    i = 0
    while i < len(labels):
        j = i
        while j+1 < len(labels) and labels[j+1] == labels[i]:
            j += 1
        start, end = times[i][0], times[j][1]
        if end - start >= MIN_INTERVAL_SEC:
            intervals.append({
                "activity": str(labels[i]),
                "start": round(float(start),1),
                "end": round(float(end),1),
                "duration": round(float(end-start),1),
                "confidence": round(float(np.mean(conf[i:j+1])),3),
                "evidence": {
                    "acc_motion_std":   round(float(np.mean([e["acc_motion_std"] for e in evid[i:j+1]])),4),
                    "gyro_energy":      round(float(np.mean([e["gyro_energy"] for e in evid[i:j+1]])),4),
                    "dominant_freq_hz": round(float(np.mean([e["dominant_freq_hz"] for e in evid[i:j+1]])),2),
                    "tilt_z":           round(float(np.mean([e["tilt_z"] for e in evid[i:j+1]])),3),
                }
            })
        i = j+1
    return intervals

def summarize(intervals, duration):
    s = {}
    for iv in intervals:
        a = iv["activity"]
        s.setdefault(a, {"total_sec":0.0, "count":0})
        s[a]["total_sec"] += iv["duration"]; s[a]["count"] += 1
    for a in s: s[a]["total_sec"] = round(s[a]["total_sec"],1)
    return s

# ---------------- main ------------------------------------------------------
def collect_user_files(uuid, acc_root="data/raw_acc", gyro_root="data/raw_gyro"):
    """Find every (timestamp, acc_path, gyro_path) for one user, sorted by time."""
    import glob
    acc = {}
    for p in glob.glob(os.path.join(acc_root, "**", uuid, "*"), recursive=True):
        ts = os.path.basename(p).split(".")[0]
        if ts.isdigit(): acc[int(ts)] = p
    gyr = {}
    for p in glob.glob(os.path.join(gyro_root, "**", uuid, "*"), recursive=True):
        ts = os.path.basename(p).split(".")[0]
        if ts.isdigit(): gyr[int(ts)] = p
    common = sorted(set(acc) & set(gyr))
    return [(ts, acc[ts], gyr[ts]) for ts in common]


def build_user_windows(uuid, acc_root, gyro_root, max_minutes=None):
    """Stitch a whole user's recording. Returns windows + absolute times (sec from first minute)."""
    files = collect_user_files(uuid, acc_root, gyro_root)
    if max_minutes: files = files[:max_minutes]
    if not files:
        raise SystemExit(f"No paired acc+gyro files found for user {uuid}")
    t0_abs = files[0][0]                      # first minute's unix timestamp
    all_w, all_t = [], []
    for k, (ts, pa, pg) in enumerate(files):
        try:
            t_acc, acc = load_raw(pa)
            t_gyr, gyr = load_raw(pg)
        except Exception:
            continue
        offset = float(ts - t0_abs)           # where this minute sits on the global clock
        wins, times = cut_windows(t_acc, acc, t_gyr, gyr)
        for w, (s0, e0) in zip(wins, times):
            all_w.append(w); all_t.append((offset + s0, offset + e0))
        if (k+1) % 200 == 0:
            print(f"  processed {k+1}/{len(files)} minutes", flush=True)
    return all_w, all_t, len(files)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", help="UUID -> stitch ALL of this user's recording")
    ap.add_argument("--acc",  help="single raw accelerometer .dat")
    ap.add_argument("--gyro", help="single raw gyroscope .dat")
    ap.add_argument("--acc-root",  default="data/raw_acc")
    ap.add_argument("--gyro-root", default="data/raw_gyro")
    ap.add_argument("--max-minutes", type=int, default=None,
                    help="limit for a quick test, e.g. 200")
    ap.add_argument("--model", default="rf", choices=["rf","cnn"])
    ap.add_argument("--out", default="timeline.json")
    args = ap.parse_args()

    if args.user:
        print(f"Building timeline for user {args.user} ...", flush=True)
        wins, times, n_min = build_user_windows(args.user, args.acc_root,
                                                args.gyro_root, args.max_minutes)
        print(f"  {n_min} minutes -> {len(wins)} windows", flush=True)
    elif args.acc and args.gyro:
        t_acc, acc = load_raw(args.acc)
        t_gyr, gyr = load_raw(args.gyro)
        wins, times = cut_windows(t_acc, acc, t_gyr, gyr)
        print(f"{len(wins)} windows from a single recording", flush=True)
    else:
        raise SystemExit("Give either --user UUID  or  --acc FILE --gyro FILE")

    if not wins:
        raise SystemExit("No usable windows found.")

    print("Classifying windows ...", flush=True)
    labels, conf = (predict_rf(wins) if args.model=="rf" else predict_cnn(wins))
    evid = [window_evidence(w) for w in wins]

    labels = smooth(labels)
    intervals = merge(labels, times, conf, evid)
    duration = round(float(times[-1][1]), 1)

    timeline = {
        "recording_duration_sec": duration,
        "time_base": "seconds from start",
        "sampling_rate_hz": TARGET_FS,
        "window_sec": WIN_SEC,
        "model": args.model,
        "n_windows": len(wins),
        "intervals": intervals,
        "summary": summarize(intervals, duration),
    }
    with open(args.out, "w") as f:
        json.dump(timeline, f, indent=2)

    print(f"\nsaved {args.out}")
    print(f"  duration: {duration/3600:.2f} hours, {len(intervals)} intervals")
    print("\n  summary:")
    for a, v in sorted(timeline["summary"].items(), key=lambda kv: -kv[1]["total_sec"]):
        print(f"    {a:22s} {v['total_sec']:9.1f}s  in {v['count']} bout(s)")
    print("\n  first 10 intervals:")
    for iv in intervals[:10]:
        print(f"    {iv['activity']:22s} {iv['start']:9.1f} - {iv['end']:9.1f}s  ({iv['duration']}s)")