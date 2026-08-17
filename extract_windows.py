"""
STEP 2 — Turn raw signals into labeled training windows.

For every (uuid, timestamp, activity) in label_table.csv:
  1. open the raw accelerometer + gyroscope file for that minute
  2. resample 40 Hz -> 25 Hz  (the challenge's common time base)
  3. cut into 5-second windows (125 samples x 6 channels), 50% overlap
  4. every window gets that minute's activity label

Output: windows.npz  ->  X (N,125,6), y (N,), groups (N,)  [groups = uuid, for per-user CV]
"""
import os, glob
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

ACC_DIR, GYRO_DIR = "data/raw_acc", "data/raw_gyro"
FS_IN, FS_OUT     = 40, 25          # resample 40 Hz -> 25 Hz  (x 5/8)
WIN, STEP         = 125, 62         # 5 s window, 50% overlap at 25 Hz

def build_index(root):
    """Map (uuid, timestamp) -> file path. Works whatever the folder layout is."""
    idx = {}
    for path in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if not os.path.isfile(path):
            continue
        name = os.path.basename(path)
        ts = name.split(".")[0]
        if not ts.isdigit():
            continue
        # uuid = the folder name that looks like a UUID (contains '-')
        uuid = next((p for p in path.split(os.sep) if "-" in p and len(p) > 30), None)
        if uuid:
            idx[(uuid, int(ts))] = path
    return idx

def load_signal(path):
    """Load one minute file -> (samples, 3). Last 3 columns are X,Y,Z."""
    try:
        a = np.loadtxt(path)
    except Exception:
        return None
    if a.ndim != 2 or a.shape[0] < FS_IN * 2:   # need at least ~2 s of data
        return None
    a = a[:, -3:]
    if np.isnan(a).any():
        return None
    return a

print("Indexing raw files (one-time scan)...")
acc_idx, gyro_idx = build_index(ACC_DIR), build_index(GYRO_DIR)
print(f"acc files: {len(acc_idx)}, gyro files: {len(gyro_idx)}")

table = pd.read_csv("label_table.csv")
X, y, groups = [], [], []

for uuid, ts, act in table.itertuples(index=False):
    pa, pg = acc_idx.get((uuid, ts)), gyro_idx.get((uuid, ts))
    if not pa or not pg:
        continue                       # this minute is missing a sensor -> skip
    acc, gyr = load_signal(pa), load_signal(pg)
    if acc is None or gyr is None:
        continue

    acc = resample_poly(acc, FS_OUT, FS_IN, axis=0)   # 40 -> 25 Hz
    gyr = resample_poly(gyr, FS_OUT, FS_IN, axis=0)
    n = min(len(acc), len(gyr))
    sig = np.hstack([acc[:n], gyr[:n]])               # (n, 6)

    for s in range(0, n - WIN + 1, STEP):             # sliding 5-s windows
        X.append(sig[s:s + WIN]); y.append(act); groups.append(uuid)

X = np.asarray(X, dtype=np.float32)
np.savez_compressed("windows.npz", X=X, y=np.array(y), groups=np.array(groups))
print(f"Saved {len(X)} windows of shape {X.shape[1:]} -> windows.npz")
print(pd.Series(y).value_counts())
