"""
YOU SET THESE TWO:
  TARGET_FS  = 40   (or 25)
  WIN_SEC    = 3.0  (or 5.0)
Output: windows_25hz.npz -> X (N, WIN, 6), y (N,), groups (N,)
"""
import os, glob
import numpy as np
import pandas as pd

# ============ YOU CHANGE THESE ============
TARGET_FS = 25          # output rate in Hz  (set 25 or 40)
WIN_SEC   = 3.0         # window length in seconds (set 3.0 or 5.0)
OVERLAP   = 0.5         # 50% overlap between windows
# ==========================================

ACC_DIR, GYRO_DIR = "data/raw_acc", "data/raw_gyro"
WIN      = int(WIN_SEC * TARGET_FS)          # samples per output window
STEP_SEC = WIN_SEC * (1 - OVERLAP)           # hop in seconds
OUT      = f"windows_{TARGET_FS}hz.npz"

def build_index(root):
    idx = {}
    for path in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if not os.path.isfile(path):
            continue
        ts = os.path.basename(path).split("_")[0].split(".")[0]
        if not ts.isdigit():
            continue
        uuid = next((p for p in path.split(os.sep) if "-" in p and len(p) > 30), None)
        if uuid:
            idx[(uuid, int(ts))] = path
    return idx

def load_raw(path):
    """Load [t,X,Y,Z] -> (t, xyz) with t starting at 0. None if unusable."""
    try:
        a = np.loadtxt(path)
    except Exception:
        return None
    if a.ndim != 2 or a.shape[0] < 20 or a.shape[1] < 4:
        return None
    t = a[:, 0].astype(np.float64)
    xyz = a[:, 1:4].astype(np.float32)
    if np.isnan(a).any():
        return None
    # sort by time if needed, normalize to start at 0
    order = np.argsort(t)
    t, xyz = t[order], xyz[order]
    t = t - t[0]
    dur = float(t[-1])
    if not np.isfinite(dur) or dur < WIN_SEC or dur > 120.0:
        return None                      # too short for one window, or glitched clock
    return t, xyz

def windows_from_file(t, xyz):
    dur = t[-1]
    start = 0.0
    out = []
    while start + WIN_SEC <= dur + 1e-9:
        end = start + WIN_SEC
        # samples inside [start, end]
        m = (t >= start) & (t <= end)
        if m.sum() >= max(5, WIN // 4):          # enough real samples to interpolate
            tw, xw = t[m], xyz[m]
            grid = np.linspace(start, end, WIN, endpoint=False)
            win = np.empty((WIN, 3), dtype=np.float32)
            for c in range(3):
                win[:, c] = np.interp(grid, tw, xw[:, c])
            out.append(win)
        else:
            out.append(None)                     # gap -> placeholder (keeps time aligned)
        start += STEP_SEC
    return out

print(f"TARGET_FS={TARGET_FS} Hz, WIN_SEC={WIN_SEC}s -> {WIN} samples/window, "
      f"step {STEP_SEC}s, out={OUT}")
print("Indexing raw files (one-time scan)...")
acc_idx, gyro_idx = build_index(ACC_DIR), build_index(GYRO_DIR)
print(f"acc files: {len(acc_idx)}, gyro files: {len(gyro_idx)}")

table = pd.read_csv("label_table.csv")
X, y, groups = [], [], []

for uuid, ts, act in table.itertuples(index=False):
    pa, pg = acc_idx.get((uuid, ts)), gyro_idx.get((uuid, ts))
    if not pa or not pg:
        continue
    ra, rg = load_raw(pa), load_raw(pg)
    if ra is None or rg is None:
        continue
    ta, xa = ra
    tg, xg = rg
    wa = windows_from_file(ta, xa)
    wg = windows_from_file(tg, xg)
    n = min(len(wa), len(wg))
    for k in range(n):
        if wa[k] is None or wg[k] is None:
            continue                              # skip gap windows
        sig = np.hstack([wa[k], wg[k]])           # (WIN, 6)
        X.append(sig); y.append(act); groups.append(uuid)

X = np.asarray(X, dtype=np.float32)
np.savez_compressed(OUT, X=X, y=np.array(y), groups=np.array(groups))
print(f"Saved {len(X)} windows of shape {X.shape[1:]} -> {OUT}")
print(pd.Series(y).value_counts())