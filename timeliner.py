
import argparse, json, os, glob
import numpy as np
from scipy.signal import butter, filtfilt

# ---- set by --fs at runtime ----
TARGET_FS = 40
WIN_SEC   = 3.0
SMOOTH_K  = 9            # wider than before -> fewer flicker bouts (helps count questions)
MIN_INTERVAL_SEC = 6.0   # drop micro-blips
TZ_OFFSET = -8          # San Diego local hour, overridden by the model file if present

def _setfs(fs):
    global TARGET_FS, WIN, B_LP, A_LP
    TARGET_FS = fs
    WIN = int(WIN_SEC * TARGET_FS)
    B_LP, A_LP = butter(3, 0.3/(TARGET_FS/2), btype="low")

# ---------------- features (IDENTICAL to train_rf_40.py) ----------------
def stats(x):
    return [x.mean(), x.std(), x.min(), x.max(),
            np.mean(np.abs(x-x.mean())), np.mean(x**2),
            np.percentile(x,25), np.percentile(x,75)]

def spectral(x):
    xz = x - x.mean()
    sp = np.abs(np.fft.rfft(xz))**2
    fr = np.fft.rfftfreq(len(x), 1/TARGET_FS)
    tot = sp.sum() + 1e-12
    bands = [sp[(fr>=lo)&(fr<hi)].sum()/tot for lo,hi in [(0.3,1),(1,2),(2,3.5),(3.5,8)]]
    p = sp/tot
    centroid = float((fr*sp).sum()/tot)
    return [fr[int(np.argmax(sp))], sp.max()/tot, centroid] + bands + [-np.sum(p*np.log(p+1e-12))]

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
    bm, gym = np.linalg.norm(body,axis=1), np.linalg.norm(gyr,axis=1)
    chans = [body[:,0],body[:,1],body[:,2],gyr[:,0],gyr[:,1],gyr[:,2],bm,gym]
    for x in chans: f += stats(x)+spectral(x)
    f += autocorr(bm) + autocorr(gym)
    for a,b in [(0,1),(0,2),(1,2)]:
        f.append(float(np.corrcoef(body[:,a],body[:,b])[0,1]))
        f.append(float(np.corrcoef(gyr[:,a],gyr[:,b])[0,1]))
    jerk = np.diff(body, axis=0)*TARGET_FS
    jm = np.linalg.norm(jerk, axis=1)
    f += [jm.mean(), jm.std(), np.mean(jm**2)]
    f += [np.mean(np.abs(body).sum(axis=1)), np.mean(np.abs(gyr).sum(axis=1))]
    return np.nan_to_num(np.array(f,dtype=np.float32))

def time_feats(epoch, tz):
    local_h = ((epoch/3600.0 + tz) % 24)
    ang = 2*np.pi*local_h/24.0
    return np.array([np.sin(ang), np.cos(ang)], dtype=np.float32)

# ---------------- evidence stats ----------------
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

# ---------------- loading + windowing ----------------
def load_raw(path):
    a = np.loadtxt(path)
    t = a[:,0].astype(np.float64); xyz = a[:,1:4].astype(np.float32)
    o = np.argsort(t); t, xyz = t[o], xyz[o]
    return t - t[0], xyz

def cut_windows(t_acc, acc, t_gyr, gyr):
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
        start += WIN_SEC
    return wins, times

# ---------------- prediction (matches train_rf_40.py model format) ----------------
def predict_rf(wins, epochs, model_path):
    import joblib
    obj = joblib.load(model_path)
    # new format is a dict; old format is a bare classifier
    if isinstance(obj, dict):
        clf = obj["clf"]; weights = np.asarray(obj.get("weights"))
        has_ts = obj.get("has_ts", False); tz = obj.get("tz", TZ_OFFSET)
    else:
        clf = obj; weights = None; has_ts = False; tz = TZ_OFFSET

    # base signal features
    F = np.stack([features(w) for w in wins])

    # neighbour-context: motion proxy (feature col 8) of prev/next window in time order
    motion = F[:, 8]
    N = len(F)
    prevm = np.empty(N, np.float32); nextm = np.empty(N, np.float32)
    prevm[0] = motion[0]; nextm[-1] = motion[-1]
    prevm[1:] = motion[:-1]; nextm[:-1] = motion[1:]
    F = np.concatenate([F, prevm[:,None], nextm[:,None]], axis=1)

    # time-of-day, only if the model was trained with it
    if has_ts:
        tf = np.stack([time_feats(e, tz) for e in epochs])
        F = np.concatenate([F, tf], axis=1)

    proba = clf.predict_proba(F)
    if weights is not None and len(weights) == proba.shape[1]:
        proba = proba * weights          # apply tuned class weights
    labels = clf.classes_[proba.argmax(1)]
    conf = proba.max(1) / (proba.sum(1) + 1e-12)
    return labels, conf

# ---------------- smoothing + merging ----------------
def smooth(labels, k=SMOOTH_K):
    out = list(labels); n = len(labels); h = k//2
    for i in range(n):
        lo, hi = max(0,i-h), min(n,i+h+1)
        seg = list(labels[lo:hi]); out[i] = max(set(seg), key=seg.count)
    return np.array(out)

def merge(labels, times, conf, evid):
    intervals = []; i = 0
    while i < len(labels):
        j = i
        while j+1 < len(labels) and labels[j+1] == labels[i]: j += 1
        start, end = times[i][0], times[j][1]
        if end - start >= MIN_INTERVAL_SEC:
            intervals.append({
                "activity": str(labels[i]),
                "start": round(float(start),1), "end": round(float(end),1),
                "duration": round(float(end-start),1),
                "confidence": round(float(np.mean(conf[i:j+1])),3),
                "evidence": {
                    "acc_motion_std":   round(float(np.mean([e["acc_motion_std"] for e in evid[i:j+1]])),4),
                    "gyro_energy":      round(float(np.mean([e["gyro_energy"] for e in evid[i:j+1]])),4),
                    "dominant_freq_hz": round(float(np.mean([e["dominant_freq_hz"] for e in evid[i:j+1]])),2),
                    "tilt_z":           round(float(np.mean([e["tilt_z"] for e in evid[i:j+1]])),3),
                }})
        i = j+1
    return intervals

def summarize(intervals):
    s = {}
    for iv in intervals:
        a = iv["activity"]; s.setdefault(a, {"total_sec":0.0, "count":0})
        s[a]["total_sec"] += iv["duration"]; s[a]["count"] += 1
    for a in s: s[a]["total_sec"] = round(s[a]["total_sec"],1)
    return s

# ---------------- user stitching ----------------
def collect_user_files(uuid, acc_root, gyro_root):
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
    files = collect_user_files(uuid, acc_root, gyro_root)
    if max_minutes: files = files[:max_minutes]
    if not files: raise SystemExit(f"No paired files for {uuid}")
    t0 = files[0][0]
    all_w, all_t, all_e = [], [], []          # windows, (start,end) global, absolute epoch
    for k,(ts,pa,pg) in enumerate(files):
        try:
            t_acc, acc = load_raw(pa); t_gyr, gyr = load_raw(pg)
        except Exception:
            continue
        off = float(ts - t0)
        wins, times = cut_windows(t_acc, acc, t_gyr, gyr)
        for w,(s0,e0) in zip(wins, times):
            all_w.append(w); all_t.append((off+s0, off+e0)); all_e.append(float(ts)+s0)
        if (k+1)%200==0: print(f"  processed {k+1}/{len(files)} minutes", flush=True)
    return all_w, all_t, all_e, len(files)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user")
    ap.add_argument("--acc"); ap.add_argument("--gyro")
    ap.add_argument("--acc-root", default="data/raw_acc")
    ap.add_argument("--gyro-root", default="data/raw_gyro")
    ap.add_argument("--fs", type=int, default=40)
    ap.add_argument("--max-minutes", type=int, default=None)
    ap.add_argument("--model", default="rf", choices=["rf"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--out", default="timeline.json")
    args = ap.parse_args()

    _setfs(args.fs)
    model_path = args.model_path or f"rf_model_{args.fs}hz.joblib"

    if args.user:
        print(f"Building timeline for user {args.user} (fs={args.fs}) ...", flush=True)
        wins, times, epochs, n_min = build_user_windows(args.user, args.acc_root,
                                                         args.gyro_root, args.max_minutes)
        print(f"  {n_min} minutes -> {len(wins)} windows", flush=True)
    elif args.acc and args.gyro:
        t_acc, acc = load_raw(args.acc); t_gyr, gyr = load_raw(args.gyro)
        wins, times = cut_windows(t_acc, acc, t_gyr, gyr)
        base = int(os.path.basename(args.acc).split(".")[0])
        epochs = [float(base)+s for s,_ in times]
        print(f"{len(wins)} windows from a single recording", flush=True)
    else:
        raise SystemExit("Give --user UUID  or  --acc FILE --gyro FILE")

    if not wins: raise SystemExit("No usable windows found.")

    print("Classifying windows ...", flush=True)
    labels, conf = predict_rf(wins, epochs, model_path)
    evid = [window_evidence(w) for w in wins]

    labels = smooth(labels)
    intervals = merge(labels, times, conf, evid)
    duration = round(float(times[-1][1]), 1)

    timeline = {
        "recording_duration_sec": duration, "time_base": "seconds from start",
        "sampling_rate_hz": args.fs, "window_sec": WIN_SEC, "model": args.model,
        "n_windows": len(wins), "intervals": intervals, "summary": summarize(intervals),
    }
    json.dump(timeline, open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}")
    print(f"  duration {duration/3600:.2f} h, {len(intervals)} intervals")
    for a,v in sorted(timeline["summary"].items(), key=lambda kv:-kv[1]["total_sec"]):
        print(f"    {a:22s} {v['total_sec']:9.1f}s in {v['count']} bout(s)")