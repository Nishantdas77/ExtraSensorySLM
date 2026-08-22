"""
Random Forest on 40 Hz / 3s windows -> 5-fold leave-users-out.
Self-contained: reads windows_40hz.npz, builds features, trains, reports.

Features per window (gravity-separated, the key to lying vs sitting vs standing):
  - gravity direction + tilt angles  (posture)      [Anguita 2013]
  - body-motion stats + spectral bands + entropy    [Bao&Intille 2004; Vaizman 2017]
  - autocorrelation cadence, axis correlations
Imbalance: cap huge classes; augment rare ones (jitter/scale/rotate/warp) [Tang 2020].

Run:  nohup python3 train_rf.py > rf.txt 2>&1 &   ;   tail -f rf.txt
"""
import os, glob, time
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib

WINF      = "windows_40hz.npz"
FEAT_CACHE= "feat40.npz"           # features saved here so re-runs are instant
FS        = 40
N_CORES   = 16
N_TREES   = 300
CAP       = 80_000                 # max windows per class kept for training
AUG_TO    = 50_000                 # augment rare classes up to this many
MODEL_OUT = "rf_model_40hz.joblib"

B_LP, A_LP = butter(3, 0.3/(FS/2), btype="low")   # 0.3 Hz -> gravity component

def stats(x):
    return [x.mean(), x.std(), x.min(), x.max(),
            np.mean(np.abs(x-x.mean())), np.mean(x**2)]

def spectral(x):
    xz = x - x.mean()
    sp = np.abs(np.fft.rfft(xz))**2
    fr = np.fft.rfftfreq(len(x), 1/FS)
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
    chans = [body[:,i] for i in range(3)]+[gyr[:,i] for i in range(3)]
    chans += [np.linalg.norm(body,axis=1), np.linalg.norm(gyr,axis=1)]
    for x in chans: f += stats(x)+spectral(x)
    f += autocorr(np.linalg.norm(body,axis=1)) + autocorr(np.linalg.norm(gyr,axis=1))
    for a,b in [(0,1),(0,2),(1,2)]:
        f.append(float(np.corrcoef(body[:,a],body[:,b])[0,1]))
        f.append(float(np.corrcoef(gyr[:,a],gyr[:,b])[0,1]))
    return np.nan_to_num(np.array(f,dtype=np.float32))

# ---- augmentation for rare classes ----
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

print("Loading windows ...", flush=True)
d = np.load(WINF, allow_pickle=True)
X_raw, y, groups = d["X"], d["y"], d["groups"]
N = len(y); classes = sorted(set(y.tolist()))
print(f"  {N} windows {X_raw.shape[1:]}, classes {classes}", flush=True)

# ---- features (cached) ----
if os.path.exists(FEAT_CACHE):
    print("Loading cached features ...", flush=True)
    F = np.load(FEAT_CACHE)["F"]
else:
    print("Computing features in PARALLEL (all cores) ...", flush=True)
    from joblib import Parallel, delayed
    t0=time.time()
    # process in blocks so we see progress
    BLK = 200000
    parts = []
    for s0 in range(0, N, BLK):
        e0 = min(s0+BLK, N)
        block = Parallel(n_jobs=-1, batch_size=2000)(
            delayed(features)(X_raw[i]) for i in range(s0, e0))
        parts.append(np.stack(block).astype(np.float32))
        print(f"  {e0}/{N}  ({time.time()-t0:.0f}s)", flush=True)
    F = np.concatenate(parts, axis=0)
    np.savez_compressed(FEAT_CACHE, F=F)
    print(f"  saved {FEAT_CACHE}  (total {time.time()-t0:.0f}s)", flush=True)

# ---- 5-fold leave-users-out ----
gkf = GroupKFold(n_splits=5)
all_true, all_pred, fold_f1 = [], [], []

for fold,(tr,te) in enumerate(gkf.split(F,y,groups),1):
    print(f"\n===== FOLD {fold}/5 =====", flush=True)
    rng = np.random.default_rng(42+fold)

    # cap huge classes
    keep=[]
    for c in classes:
        ic = tr[y[tr]==c]
        if len(ic)>CAP: ic = rng.choice(ic,CAP,replace=False)
        keep.append(ic)
    keep = np.concatenate(keep)
    Xtr, Ytr = F[keep], y[keep]

    # augment rare classes
    eF,eY=[],[]
    for c in classes:
        have=int((Ytr==c).sum()); need=AUG_TO-have
        if need<=0: continue
        src=tr[y[tr]==c]; pick=rng.choice(src,need,replace=True)
        for i in pick: eF.append(features(augment(X_raw[i]))); eY.append(c)
    if eF:
        Xtr=np.concatenate([Xtr,np.stack(eF)]); Ytr=np.concatenate([Ytr,np.array(eY)])
    print(f"  train {Xtr.shape}", flush=True)

    clf=RandomForestClassifier(n_estimators=N_TREES,class_weight="balanced_subsample",
                               min_samples_leaf=2,n_jobs=N_CORES,random_state=42)
    clf.fit(Xtr,Ytr)
    pred=clf.predict(F[te])
    f1=f1_score(y[te],pred,average="macro")
    fold_f1.append(f1); all_true.append(y[te]); all_pred.append(pred)
    per=f1_score(y[te],pred,average=None,labels=classes,zero_division=0)
    print("  per-class F1:", {classes[i]:round(float(per[i]),3) for i in range(len(classes))}, flush=True)
    print(f"  fold macro-F1: {f1:.3f}", flush=True)
    if fold==1: joblib.dump(clf,MODEL_OUT); print(f"  saved {MODEL_OUT}", flush=True)

yt=np.concatenate(all_true); yp=np.concatenate(all_pred)
print("\n==== RF 5-FOLD RESULTS ====", flush=True)
print(classification_report(yt,yp,digits=3))
print(f"mean macro-F1: {np.mean(fold_f1):.3f} (+/- {np.std(fold_f1):.3f})")
print("\nConfusion matrix (rows=true, cols=pred):"); print(classes)
print(confusion_matrix(yt,yp,labels=classes))
np.savez("rf_results.npz", cm=confusion_matrix(yt,yp,labels=classes),
         classes=np.array(classes), y_true=yt, y_pred=yp, fold_f1=np.array(fold_f1))
print("Saved rf_results.npz")