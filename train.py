"""
STEP 3 (v4) — extra gentle. Avoids the sudden all-core spike that rebooted the server.

Key safety choices:
  - features already cached in feat_chunks/ are REUSED (no recompute)
  - forest trains on a SMALL sample with FEW trees and only 2 cores
  - everything checkpointed: re-run resumes, never starts from zero

Run:
    nohup python step3_train.py > result.txt 2>&1 &
    tail -f result.txt
"""
import os, glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import joblib

FS        = 25
N_CORES   = 2             # only 2 cores for training -> no big spike
N_TREES   = 100           # fewer trees -> lighter
CHUNK     = 100_000
SAMPLE    = 300_000       # train the forest on 300k windows (plenty for a baseline)
FEAT_DIR  = "feat_chunks"
MODEL_OUT = "rf_model.joblib"
os.makedirs(FEAT_DIR, exist_ok=True)

def features(w):
    feats = []
    mags = [np.linalg.norm(w[:, :3], axis=1),
            np.linalg.norm(w[:, 3:], axis=1)]
    chans = [w[:, i] for i in range(6)] + mags
    for x in chans:
        feats += [x.mean(), x.std(), x.min(), x.max(), np.mean(x**2)]
        spec = np.abs(np.fft.rfft(x - x.mean()))
        freqs = np.fft.rfftfreq(len(x), 1 / FS)
        feats.append(freqs[np.argmax(spec)])
    return np.array(feats, dtype=np.float32)

print("Loading windows.npz ...", flush=True)
d = np.load("windows.npz", allow_pickle=True)
X_raw, y, groups = d["X"], d["y"], d["groups"]
N = len(y)
print(f"  {N} windows", flush=True)

# ---------- features in resumable chunks ----------
n_chunks = (N + CHUNK - 1) // CHUNK
for c in range(n_chunks):
    out = os.path.join(FEAT_DIR, f"chunk_{c:04d}.npy")
    if os.path.exists(out):
        continue
    s, e = c * CHUNK, min((c + 1) * CHUNK, N)
    block = np.stack([features(X_raw[i]) for i in range(s, e)])
    np.save(out, block)
    print(f"  features chunk {c+1}/{n_chunks} saved", flush=True)
print("  all feature chunks ready", flush=True)

# ---------- load chunks ----------
files = sorted(glob.glob(os.path.join(FEAT_DIR, "chunk_*.npy")))
X = np.concatenate([np.load(f) for f in files], axis=0)
print(f"  feature matrix {X.shape}", flush=True)

# ---------- sample ----------
rng = np.random.default_rng(42)
idx = rng.choice(N, size=min(SAMPLE, N), replace=False)
Xs, ys, gs = X[idx], y[idx], groups[idx]

# ---------- split by USER ----------
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
              .split(Xs, ys, gs))

# ---------- train (gentle) ----------
print("Training Random Forest (gentle: 100 trees, 2 cores) ...", flush=True)
clf = RandomForestClassifier(n_estimators=N_TREES, class_weight="balanced",
                             n_jobs=N_CORES, random_state=42)
clf.fit(Xs[tr], ys[tr])
joblib.dump(clf, MODEL_OUT)
print(f"  saved model -> {MODEL_OUT}", flush=True)

# ---------- evaluate ----------
pred = clf.predict(Xs[te])
print("\n==== RESULTS ====", flush=True)
print(classification_report(ys[te], pred, digits=3))
labels = sorted(set(ys))
print("Confusion matrix (rows = true, cols = predicted):")
print(labels)
print(confusion_matrix(ys[te], pred, labels=labels))
