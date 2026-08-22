# """
# Verify windows.npz is clean and correct before trusting training.
# Checks: shape, no NaN/inf, all 6 channels present, label set, per-user integrity,
# and that acc & gyro channels actually carry real (non-zero) signal.
# """
# import numpy as np

# d = np.load("windows.npz", allow_pickle=True)
# X, y, groups = d["X"], d["y"], d["groups"]

# print("=== SHAPE ===")
# print("X:", X.shape, "  y:", y.shape, "  groups:", groups.shape)
# assert X.shape[1:] == (125, 6), "each window must be 125 x 6"
# assert len(X) == len(y) == len(groups), "arrays must align row-by-row"

# print("\n=== MISSING VALUES ===")
# print("NaN in X:", np.isnan(X).any(), "   Inf in X:", np.isinf(X).any())

# print("\n=== CHANNELS CARRY SIGNAL (std per channel, averaged) ===")
# names = ["AccX","AccY","AccZ","GyrX","GyrY","GyrZ"]
# stds = X.std(axis=(0,1))
# for n, s in zip(names, stds):
#     flag = "  <-- LOOKS EMPTY!" if s < 1e-6 else ""
#     print(f"  {n}: {s:.4f}{flag}")

# print("\n=== LABELS ===")
# uniq, cnt = np.unique(y, return_counts=True)
# for u, c in zip(uniq, cnt):
#     print(f"  {u:20s} {c}")
# print("  total classes:", len(uniq))

# print("\n=== USERS ===")
# print("  unique users:", len(np.unique(groups)))

# print("\n=== PER-USER: does one user mix activities? (spot check first user) ===")
# u0 = groups[0]
# acts = np.unique(y[groups == u0])
# print(f"  user {u0[:8]}... has activities: {list(acts)}")

# print("\nAll checks passed if you see no 'LOOKS EMPTY' and no NaN/Inf above.")


# """Quick health check of the current windows_40hz.npz."""
# import numpy as np
# d = np.load("windows_40hz.npz", allow_pickle=True)
# X, y, groups = d["X"], d["y"], d["groups"]
# print("shape:", X.shape)
# print("classes + counts:")
# u, c = np.unique(y, return_counts=True)
# for a, b in zip(u, c): print(f"  {a:22s} {b}")
# print("users:", len(np.unique(groups)))
# print("NaN:", np.isnan(X).any(), " Inf:", np.isinf(X).any())
# print("per-channel mean:", np.round(X.mean(axis=(0,1)), 3))
# print("per-channel std :", np.round(X.std(axis=(0,1)), 3))
# print("global min/max:", float(X.min()), float(X.max()))
# # check a walking window has motion, a lying window is still
# for act in ["lying_down", "walking", "running"]:
#     idx = np.where(y == act)[0]
#     if len(idx):
#         w = X[idx[0]]
#         accmag = np.linalg.norm(w[:, :3], axis=1)
#         print(f"  {act:12s} acc-mag std = {accmag.std():.3f} (still~low, moving~high)")




"""
Profile each user: how many labeled minutes of each activity.
Reads label_table.csv (fast, no signal needed).
Prints a per-user table + flags candidates to drop.
"""
import pandas as pd
import numpy as np

t = pd.read_csv("label_table.csv")
acts = sorted(t["activity"].unique())

# pivot: rows=user, cols=activity, values=count
piv = t.pivot_table(index="uuid", columns="activity", aggfunc="size", fill_value=0)
piv = piv[acts]
piv["TOTAL"] = piv.sum(axis=1)

# rare classes we care about protecting
rare = [c for c in ["running", "bicycling", "standing_in_place"] if c in acts]
piv["RARE_SUM"] = piv[rare].sum(axis=1)

piv = piv.sort_values("TOTAL", ascending=False)

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 20)

print("=== PER-USER ACTIVITY COUNTS (minutes) ===\n")
print(piv.to_string())

print("\n=== SUMMARY ===")
print(f"Total users: {len(piv)}")
print(f"Total labeled minutes: {int(piv['TOTAL'].sum())}")

print("\n=== DROP CANDIDATES ===")
print("(a) Users with very little data (<200 minutes):")
few = piv[piv["TOTAL"] < 200]
print("   ", list(few.index) if len(few) else "none")

print("\n(b) Users that are almost ALL sedentary (>=95% sitting+lying, and NO rare classes):")
sed = piv[( (piv.get("sitting",0)+piv.get("lying_down",0)) / piv["TOTAL"] >= 0.95 )
         & (piv["RARE_SUM"] == 0)]
print("   ", list(sed.index) if len(sed) else "none")

print("\n=== USERS WITH THE RARE CLASSES (protect these!) ===")
for c in rare:
    have = piv[piv[c] > 0]
    print(f"  {c}: {len(have)} users have it; top holders:",
          list(have.sort_values(c, ascending=False).head(5).index))