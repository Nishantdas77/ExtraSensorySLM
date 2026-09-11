# import os, glob, time
# import numpy as np
# from scipy.signal import butter, filtfilt
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GroupKFold
# from sklearn.metrics import classification_report, confusion_matrix, f1_score
# import joblib

# WINF      = "windows_40hz.npz"
# FEAT_CACHE= "feat40.npz"           # features saved here so re-runs are instant
# FS        = 40
# N_CORES   = 16
# N_TREES   = 300
# CAP       = 80_000                 # max windows per class kept for training
# AUG_TO    = 50_000                 # augment rare classes up to this many
# MODEL_OUT = "rf_model_40hz.joblib"

# B_LP, A_LP = butter(3, 0.3/(FS/2), btype="low")   # 0.3 Hz -> gravity component

# def stats(x):
#     return [x.mean(), x.std(), x.min(), x.max(),
#             np.mean(np.abs(x-x.mean())), np.mean(x**2)]

# def spectral(x):
#     xz = x - x.mean()
#     sp = np.abs(np.fft.rfft(xz))**2
#     fr = np.fft.rfftfreq(len(x), 1/FS)
#     tot = sp.sum() + 1e-12
#     bands = []
#     for lo, hi in [(0.3,1.0),(1.0,2.0),(2.0,3.5),(3.5,8.0)]:
#         m = (fr>=lo)&(fr<hi); bands.append(sp[m].sum()/tot)
#     p = sp/tot
#     return [fr[int(np.argmax(sp))], sp.max()/tot] + bands + [-np.sum(p*np.log(p+1e-12))]

# def autocorr(x):
#     xz = x-x.mean()
#     ac = np.correlate(xz,xz,mode="full")[len(xz)-1:]
#     ac = ac/(ac[0]+1e-12)
#     lo,hi = int(FS*0.25), int(FS*2.0)
#     seg = ac[lo:hi]
#     if len(seg)==0: return [0.0,0.0]
#     k = int(np.argmax(seg)); return [float(seg[k]), float((lo+k)/FS)]

# def features(w):
#     acc, gyr = w[:,:3], w[:,3:]
#     grav = filtfilt(B_LP, A_LP, acc, axis=0); body = acc-grav
#     f=[]
#     gm = grav.mean(0); gn = np.linalg.norm(gm)+1e-12
#     f += list(gm) + list(np.arccos(np.clip(gm/gn,-1,1))) + [gn, grav.std(0).mean()]
#     chans = [body[:,i] for i in range(3)]+[gyr[:,i] for i in range(3)]
#     chans += [np.linalg.norm(body,axis=1), np.linalg.norm(gyr,axis=1)]
#     for x in chans: f += stats(x)+spectral(x)
#     f += autocorr(np.linalg.norm(body,axis=1)) + autocorr(np.linalg.norm(gyr,axis=1))
#     for a,b in [(0,1),(0,2),(1,2)]:
#         f.append(float(np.corrcoef(body[:,a],body[:,b])[0,1]))
#         f.append(float(np.corrcoef(gyr[:,a],gyr[:,b])[0,1]))
#     return np.nan_to_num(np.array(f,dtype=np.float32))

# # ---- augmentation for rare classes ----
# def jitter(w): return w + np.random.normal(0,0.05,w.shape).astype(np.float32)
# def scale(w):  return w * np.random.normal(1,0.1,(1,w.shape[1])).astype(np.float32)
# def rot(w):
#     ax=np.random.normal(size=3); ax/=np.linalg.norm(ax)+1e-12; an=np.random.uniform(0,2*np.pi)
#     K=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
#     R=(np.eye(3)+np.sin(an)*K+(1-np.cos(an))*(K@K)).astype(np.float32)
#     o=w.copy(); o[:,:3]=w[:,:3]@R.T; o[:,3:]=w[:,3:]@R.T; return o
# AUG=[jitter,scale,rot]
# def augment(w):
#     o=w.copy()
#     for f in np.random.choice(AUG,np.random.randint(1,3),replace=False): o=f(o)
#     return o.astype(np.float32)

# print("Loading windows ...", flush=True)
# d = np.load(WINF, allow_pickle=True)
# X_raw, y, groups = d["X"], d["y"], d["groups"]
# N = len(y); classes = sorted(set(y.tolist()))
# print(f"  {N} windows {X_raw.shape[1:]}, classes {classes}", flush=True)

# # ---- features (cached) ----
# if os.path.exists(FEAT_CACHE):
#     print("Loading cached features ...", flush=True)
#     F = np.load(FEAT_CACHE)["F"]
# else:
#     print("Computing features in PARALLEL (all cores) ...", flush=True)
#     from joblib import Parallel, delayed
#     t0=time.time()
#     # process in blocks so we see progress
#     BLK = 200000
#     parts = []
#     for s0 in range(0, N, BLK):
#         e0 = min(s0+BLK, N)
#         block = Parallel(n_jobs=-1, batch_size=2000)(
#             delayed(features)(X_raw[i]) for i in range(s0, e0))
#         parts.append(np.stack(block).astype(np.float32))
#         print(f"  {e0}/{N}  ({time.time()-t0:.0f}s)", flush=True)
#     F = np.concatenate(parts, axis=0)
#     np.savez_compressed(FEAT_CACHE, F=F)
#     print(f"  saved {FEAT_CACHE}  (total {time.time()-t0:.0f}s)", flush=True)

# # ---- 5-fold leave-users-out ----
# gkf = GroupKFold(n_splits=5)
# all_true, all_pred, fold_f1 = [], [], []

# for fold,(tr,te) in enumerate(gkf.split(F,y,groups),1):
#     print(f"\n===== FOLD {fold}/5 =====", flush=True)
#     rng = np.random.default_rng(42+fold)

#     # cap huge classes
#     keep=[]
#     for c in classes:
#         ic = tr[y[tr]==c]
#         if len(ic)>CAP: ic = rng.choice(ic,CAP,replace=False)
#         keep.append(ic)
#     keep = np.concatenate(keep)
#     Xtr, Ytr = F[keep], y[keep]

#     # augment rare classes
#     eF,eY=[],[]
#     for c in classes:
#         have=int((Ytr==c).sum()); need=AUG_TO-have
#         if need<=0: continue
#         src=tr[y[tr]==c]; pick=rng.choice(src,need,replace=True)
#         for i in pick: eF.append(features(augment(X_raw[i]))); eY.append(c)
#     if eF:
#         Xtr=np.concatenate([Xtr,np.stack(eF)]); Ytr=np.concatenate([Ytr,np.array(eY)])
#     print(f"  train {Xtr.shape}", flush=True)

#     clf=RandomForestClassifier(n_estimators=N_TREES,class_weight="balanced_subsample",
#                                min_samples_leaf=2,n_jobs=N_CORES,random_state=42)
#     clf.fit(Xtr,Ytr)
#     pred=clf.predict(F[te])
#     f1=f1_score(y[te],pred,average="macro")
#     fold_f1.append(f1); all_true.append(y[te]); all_pred.append(pred)
#     per=f1_score(y[te],pred,average=None,labels=classes,zero_division=0)
#     print("  per-class F1:", {classes[i]:round(float(per[i]),3) for i in range(len(classes))}, flush=True)
#     print(f"  fold macro-F1: {f1:.3f}", flush=True)
#     if fold==1: joblib.dump(clf,MODEL_OUT); print(f"  saved {MODEL_OUT}", flush=True)

# yt=np.concatenate(all_true); yp=np.concatenate(all_pred)
# print("\n==== RF 5-FOLD RESULTS ====", flush=True)
# print(classification_report(yt,yp,digits=3))
# print(f"mean macro-F1: {np.mean(fold_f1):.3f} (+/- {np.std(fold_f1):.3f})")
# print("\nConfusion matrix (rows=true, cols=pred):"); print(classes)
# print(confusion_matrix(yt,yp,labels=classes))
# np.savez("rf_results.npz", cm=confusion_matrix(yt,yp,labels=classes),
#          classes=np.array(classes), y_true=yt, y_pred=yp, fold_f1=np.array(fold_f1))
# print("Saved rf_results.npz")

"""
Random Forest v3 -- higher macro-F1 via:
  (A) richer signal features  (jerk, SMA, cross-axis energy, percentiles)
  (B) time-of-day features    (local-hour sin/cos) -- strongest single lever
  (C) probability threshold tuning per class, fitted on a validation split
  (D) probability calibration is left to the forest; weights do the rebalancing

All gains are chosen on a VALIDATION split inside each training fold, then reported
on the untouched test fold -- so the tuned weights never see test data.

Needs windows_<fs>hz.npz. If the .npz stores per-window timestamps under key "ts"
(epoch seconds), time-of-day features switch on automatically; otherwise they are
skipped and the script still runs.

Run:  python3 train_rf_v3.py --fs 25
"""
import os, glob, time, argparse
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from joblib import Parallel, delayed
import joblib

ap = argparse.ArgumentParser()
ap.add_argument("--fs", type=int, default=40)
ap.add_argument("--trees", type=int, default=400)
ap.add_argument("--cap", type=int, default=80_000)
ap.add_argument("--aug-to", type=int, default=60_000)
ap.add_argument("--tz-offset", type=int, default=-8, help="UTC offset for local hour (San Diego = -8)")
ap.add_argument("--quick", action="store_true", help="fold 1 only")
ap.add_argument("--merge-standing", action="store_true",
                help="train 6 classes (merge the two standing types)")
ap.add_argument("--hp-search", action="store_true",
                help="grid-search forest hyperparameters on fold-1 validation")
args = ap.parse_args()

FS        = args.fs
WINF      = f"windows_{FS}hz.npz"
FEAT_CACHE= f"feat{FS}_v3.npz"
MODEL_OUT = f"rf_model_{FS}hz.joblib"
N_CORES   = 16

B_LP, A_LP = butter(3, 0.3/(FS/2), btype="low")

# ---------------- signal features (richer than v1) ----------------
def stats(x):
    return [x.mean(), x.std(), x.min(), x.max(),
            np.mean(np.abs(x-x.mean())), np.mean(x**2),
            np.percentile(x,25), np.percentile(x,75)]     # NEW: quartiles

def spectral(x):
    xz = x - x.mean()
    sp = np.abs(np.fft.rfft(xz))**2
    fr = np.fft.rfftfreq(len(x), 1/FS)
    tot = sp.sum() + 1e-12
    bands = [sp[(fr>=lo)&(fr<hi)].sum()/tot for lo,hi in [(0.3,1),(1,2),(2,3.5),(3.5,8)]]
    p = sp/tot
    centroid = float((fr*sp).sum()/tot)                   # NEW: spectral centroid
    return [fr[int(np.argmax(sp))], sp.max()/tot, centroid] + bands + [-np.sum(p*np.log(p+1e-12))]

def autocorr(x):
    xz = x-x.mean()
    ac = np.correlate(xz,xz,mode="full")[len(xz)-1:]
    ac = ac/(ac[0]+1e-12)
    lo,hi = int(FS*0.25), int(FS*2.0)
    seg = ac[lo:hi]
    if len(seg)==0: return [0.0,0.0]
    k = int(np.argmax(seg)); return [float(seg[k]), float((lo+k)/FS)]

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
    # NEW: jerk (rate of change of body accel) -- separates smooth cycling from steppy gait
    jerk = np.diff(body, axis=0)*FS
    jm = np.linalg.norm(jerk, axis=1)
    f += [jm.mean(), jm.std(), np.mean(jm**2)]
    # NEW: signal magnitude area -- overall motion intensity
    f += [np.mean(np.abs(body).sum(axis=1)), np.mean(np.abs(gyr).sum(axis=1))]
    return np.nan_to_num(np.array(f,dtype=np.float32))

def time_feats(ts, tz=args.tz_offset):
    """epoch seconds -> [sin(hour), cos(hour)] in local time. Strong for lying vs sitting."""
    local_h = ((ts/3600.0 + tz) % 24)
    ang = 2*np.pi*local_h/24.0
    return np.stack([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)

# ---------------- augmentation ----------------
def jitter(w): return w + np.random.normal(0,0.05,w.shape).astype(np.float32)
def scale(w):  return w * np.random.normal(1,0.1,(1,w.shape[1])).astype(np.float32)
def rot(w):
    ax=np.random.normal(size=3); ax/=np.linalg.norm(ax)+1e-12; an=np.random.uniform(0,2*np.pi)
    K=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    R=(np.eye(3)+np.sin(an)*K+(1-np.cos(an))*(K@K)).astype(np.float32)
    o=w.copy(); o[:,:3]=w[:,:3]@R.T; o[:,3:]=w[:,3:]@R.T; return o
AUG=[jitter,scale,rot]
def augment(w):
    o=w.copy()
    for f in np.random.choice(AUG,np.random.randint(1,3),replace=False): o=f(o)
    return o.astype(np.float32)

# ---------------- load ----------------
print(f"Loading {WINF} ...", flush=True)
d = np.load(WINF, allow_pickle=True)
X_raw, y, groups = d["X"], d["y"], d["groups"]
HAS_TS = "ts" in d.files
ts = d["ts"] if HAS_TS else None
print(f"  {'timestamps present -> time-of-day ON' if HAS_TS else 'no timestamps -> time-of-day OFF'}", flush=True)
if args.merge_standing:
    y = np.array(["standing" if str(v).startswith("standing") else str(v) for v in y])
    print("  merged standing_in_place + standing_and_moving -> standing", flush=True)
N = len(y); classes = sorted(set(y.tolist()))
print(f"  {N} windows, {len(classes)} classes", flush=True)

# ---------------- features (cached) ----------------
if os.path.exists(FEAT_CACHE):
    print("Loading cached features ...", flush=True)
    F = np.load(FEAT_CACHE)["F"]
else:
    print("Computing features in parallel ...", flush=True)
    t0=time.time(); BLK=200_000; parts=[]
    for s0 in range(0,N,BLK):
        e0=min(s0+BLK,N)
        blk=Parallel(n_jobs=-1,batch_size=2000)(delayed(features)(X_raw[i]) for i in range(s0,e0))
        parts.append(np.stack(blk).astype(np.float32))
        print(f"  {e0}/{N} ({time.time()-t0:.0f}s)", flush=True)
    F=np.concatenate(parts,0)
    np.savez_compressed(FEAT_CACHE,F=F)
    print(f"  saved {FEAT_CACHE} ({time.time()-t0:.0f}s)", flush=True)

# neighbour-context: mean acc-energy of the window before and after (same user, time order)
# energy proxy = feature column 0-ish is not reliable; recompute a light motion scalar
print("  adding neighbour-context features ...", flush=True)
motion = F[:, 8]   # first body-channel mean; used only as a smooth motion proxy
prevm = np.zeros(N, np.float32); nextm = np.zeros(N, np.float32)
order = np.lexsort((ts if HAS_TS else np.arange(N), groups))
inv = np.empty(N, np.int64); inv[order] = np.arange(N)
gs = groups[order]
ms = motion[order]
for i in range(len(order)):
    prevm[order[i]] = ms[i-1] if i>0 and gs[i-1]==gs[i] else ms[i]
    nextm[order[i]] = ms[i+1] if i<len(order)-1 and gs[i+1]==gs[i] else ms[i]
F = np.concatenate([F, prevm[:,None], nextm[:,None]], axis=1)

# append time-of-day features if available
if HAS_TS:
    F = np.concatenate([F, time_feats(ts)], axis=1)
print(f"  feature matrix {F.shape}", flush=True)

def tune_weights(clf, Xv, yv, classes):
    """Grid-search a per-class probability multiplier to maximise macro-F1 on val."""
    proba = clf.predict_proba(Xv)
    idx = {c:i for i,c in enumerate(clf.classes_)}
    best_w = np.ones(len(clf.classes_)); best = -1
    # coordinate ascent: adjust one class weight at a time
    grid = [0.15,0.25,0.35,0.5,0.7,1.0,1.4,2.0,3.0,4.0]
    for _ in range(3):
        for ci in range(len(clf.classes_)):
            local_best_w, local_best = best_w[ci], best
            for wv in grid:
                w = best_w.copy(); w[ci]=wv
                pred = clf.classes_[(proba*w).argmax(1)]
                s = f1_score(yv, pred, average="macro")
                if s>local_best: local_best, local_best_w = s, wv
            best_w[ci]=local_best_w; best=local_best
    return best_w

# ---------------- optional hyperparameter search (fold-1 val) ----------------
best_hp = dict(n_estimators=args.trees, min_samples_leaf=2, max_features="sqrt")
if args.hp_search:
    print("\nHYPERPARAMETER SEARCH on fold-1 validation ...", flush=True)
    from itertools import product
    tr0, te0 = next(GroupKFold(n_splits=5).split(F,y,groups))
    a,b = next(GroupShuffleSplit(1, test_size=0.2, random_state=1).split(F[tr0],y[tr0],groups[tr0]))
    tr2,val = tr0[a], tr0[b]
    keep=[]
    rng=np.random.default_rng(1)
    for c in classes:
        ic=tr2[y[tr2]==c]
        if len(ic)>args.cap: ic=rng.choice(ic,args.cap,replace=False)
        keep.append(ic)
    keep=np.concatenate(keep)
    grid = list(product([300,500,800], [1,2,5], ["sqrt",0.3]))
    best=-1
    for nt,leaf,mf in grid:
        c=RandomForestClassifier(n_estimators=nt,min_samples_leaf=leaf,max_features=mf,
                                 class_weight="balanced_subsample",n_jobs=N_CORES,random_state=42)
        c.fit(F[keep],y[keep])
        w=tune_weights(c,F[val],y[val],classes)
        s_=f1_score(y[val],c.classes_[(c.predict_proba(F[val])*w).argmax(1)],average="macro")
        print(f"   trees={nt} leaf={leaf} maxf={mf}: val macro-F1 {s_:.3f}", flush=True)
        if s_>best: best,best_hp=s_,dict(n_estimators=nt,min_samples_leaf=leaf,max_features=mf)
    print(f"  BEST: {best_hp} (val macro-F1 {best:.3f})", flush=True)

# ---------------- 5-fold leave-users-out ----------------
gkf = GroupKFold(n_splits=5)
all_true, all_pred, fold_f1 = [], [], []

for fold,(tr,te) in enumerate(gkf.split(F,y,groups),1):
    print(f"\n===== FOLD {fold}/5 =====", flush=True)
    rng=np.random.default_rng(42+fold)

    # split TRAIN -> train'/val by user, so weight tuning never sees test
    tr_users = np.array(sorted(set(groups[tr])))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=fold)
    a,b = next(gss.split(F[tr], y[tr], groups[tr]))
    tr2, val = tr[a], tr[b]

    # cap majority classes
    keep=[]
    for c in classes:
        ic=tr2[y[tr2]==c]
        if len(ic)>args.cap: ic=rng.choice(ic,args.cap,replace=False)
        keep.append(ic)
    keep=np.concatenate(keep)
    Xtr,Ytr=F[keep],y[keep]

    # augment rare classes (signal only; time features copied from source window)
    eF,eY=[],[]
    for c in classes:
        have=int((Ytr==c).sum()); need=args.aug_to-have
        if need<=0: continue
        src=tr2[y[tr2]==c]
        if len(src)==0: continue
        pick=rng.choice(src,need,replace=True)
        blk=Parallel(n_jobs=-1,batch_size=1000)(delayed(features)(augment(X_raw[i])) for i in pick)
        blk=np.stack(blk)
        # neighbour-context for augmented rows: reuse the source window's own values
        blk=np.concatenate([blk, F[pick, -4:-2] if HAS_TS else F[pick, -2:]], axis=1)
        if HAS_TS:                                       # keep the source window's hour
            blk=np.concatenate([blk, time_feats(ts[pick])],axis=1)
        eF.append(blk); eY.append(np.array([c]*need))
    if eF:
        Xtr=np.concatenate([Xtr]+eF); Ytr=np.concatenate([Ytr]+eY)
    print(f"  train {Xtr.shape}", flush=True)

    clf=RandomForestClassifier(class_weight="balanced_subsample",
                               n_jobs=N_CORES, random_state=42, **best_hp)
    clf.fit(Xtr,Ytr)

    # tune thresholds on val, apply to test
    w = tune_weights(clf, F[val], y[val], classes)
    wmap = {c:round(float(w[i]),2) for i,c in enumerate(clf.classes_)}
    print(f"  tuned weights: {wmap}", flush=True)
    proba_te = clf.predict_proba(F[te])
    pred = clf.classes_[(proba_te*w).argmax(1)]

    f1=f1_score(y[te],pred,average="macro")
    fold_f1.append(f1); all_true.append(y[te]); all_pred.append(pred)
    per=f1_score(y[te],pred,average=None,labels=classes,zero_division=0)
    print("  per-class F1:", {classes[i]:round(float(per[i]),3) for i in range(len(classes))}, flush=True)
    print(f"  fold macro-F1: {f1:.3f}", flush=True)

    if fold==1:
        joblib.dump({"clf":clf, "weights":w, "has_ts":HAS_TS, "tz":args.tz_offset,
                     "classes":list(clf.classes_)}, MODEL_OUT)
        print(f"  saved {MODEL_OUT}", flush=True)
    if args.quick: break

yt=np.concatenate(all_true); yp=np.concatenate(all_pred)
print("\n==== RF v3 5-FOLD RESULTS ====", flush=True)
print(classification_report(yt,yp,digits=3))
print(f"mean macro-F1: {np.mean(fold_f1):.3f} (+/- {np.std(fold_f1):.3f})")
np.savez(f"rf_results_{FS}hz.npz", cm=confusion_matrix(yt,yp,labels=classes),
         classes=np.array(classes), y_true=yt, y_pred=yp, fold_f1=np.array(fold_f1))
print(f"Saved rf_results_{FS}hz.npz")