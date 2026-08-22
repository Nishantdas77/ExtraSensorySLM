"""
Output: label_table.csv  with columns: uuid, timestamp, activity
"""
import glob, os
import pandas as pd

LABEL_DIR = "data/labels"
ORIG_DIR  = "data/original"
OUT_CSV   = "label_table.csv"

# the 5 classes that exist directly in the cleaned file
CLEAN_MAP = {
    "label:LYING_DOWN":  "lying_down",
    "label:SITTING":     "sitting",
    "label:FIX_walking": "walking",
    "label:FIX_running": "running",
    "label:BICYCLING":   "bicycling",
}

def find_col(cols, keyword):
    """Find a column whose name contains keyword (case-insensitive)."""
    hits = [c for c in cols if keyword.lower() in c.lower()]
    return hits[0] if hits else None

rows = []
files = sorted(glob.glob(os.path.join(LABEL_DIR, "**", "*.features_labels.csv.gz"), recursive=True))
print(f"Found {len(files)} user label files")

for f in files:
    uuid = os.path.basename(f).split(".")[0]

    # --- cleaned labels: read only the columns we need (saves memory) ---
    header = pd.read_csv(f, nrows=0).columns
    need = ["timestamp"] + [c for c in CLEAN_MAP] + [c for c in header if "OR_standing" in c]
    need = [c for c in need if c in header]
    df = pd.read_csv(f, usecols=need)

    # --- original labels: to split standing into "in place" vs "and moving" ---
    sp_col = sm_col = None
    orig = None
    orig_files = glob.glob(os.path.join(ORIG_DIR, "**", uuid + "*.csv.gz"), recursive=True)
    if orig_files:
        orig = pd.read_csv(orig_files[0])
        ts_col = find_col(orig.columns, "timestamp")
        sp_col = find_col(orig.columns, "STANDING_IN_PLACE")
        sm_col = find_col(orig.columns, "STANDING_AND_MOVING")
        if ts_col and (sp_col or sm_col):
            keep = [ts_col] + [c for c in (sp_col, sm_col) if c]
            orig = orig[keep].rename(columns={ts_col: "timestamp"})
            df = df.merge(orig, on="timestamp", how="left")

    # --- decide ONE activity per minute ---
    std_col = find_col(df.columns, "OR_standing")
    for _, r in df.iterrows():
        active = [name for col, name in CLEAN_MAP.items() if col in df.columns and r.get(col) == 1]

        # standing: cleaned file says "standing", original file says which kind
        if std_col and r.get(std_col) == 1:
            in_place = sp_col and r.get(sp_col) == 1
            moving   = sm_col and r.get(sm_col) == 1
            if in_place and not moving:
                active.append("standing_in_place")
            elif moving and not in_place:
                active.append("standing_and_moving")
            # if original file doesn't tell us which kind -> skip (ambiguous)

        if len(active) == 1:                      # exactly-one rule
            rows.append((uuid, int(r["timestamp"]), active[0]))

out = pd.DataFrame(rows, columns=["uuid", "timestamp", "activity"])
out.to_csv(OUT_CSV, index=False)
print(f"Saved {len(out)} labeled minutes -> {OUT_CSV}")
print(out["activity"].value_counts())
