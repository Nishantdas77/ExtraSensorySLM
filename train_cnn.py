import numpy as np, time, os, collections
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score

WINF     = "windows_40hz.npz"
EPOCHS   = 20
BATCH    = 256
LR       = 1e-3
N_FOLDS  = 5
SAMP_PER_EP = 300_000
DEV      = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42); np.random.seed(42)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

print("device:", DEV, flush=True)
if DEV == "cuda":
    print("gpu:", torch.cuda.get_device_name(0), flush=True)
    try: torch.cuda.set_per_process_memory_fraction(0.35, 0)
    except Exception: pass

d = np.load(WINF, allow_pickle=True)
X, y, groups = d["X"], d["y"], d["groups"]
classes = sorted(set(y.tolist()))
cls2i = {c: i for i, c in enumerate(classes)}
Y = np.array([cls2i[v] for v in y], dtype=np.int64)
print(f"{X.shape} windows, classes: {classes}", flush=True)

class WinDS(Dataset):
    def __init__(self, X, Y, idx, mean, std):
        self.X, self.Y, self.idx, self.mean, self.std = X, Y, idx, mean, std
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        w = (self.X[j].astype(np.float32) - self.mean) / self.std
        return torch.from_numpy(w.T), self.Y[j]

def augment(xb):
    """Light augmentation: jitter + per-channel scaling only (no rotation — safer)."""
    xb = xb + torch.randn_like(xb) * 0.05
    xb = xb * (1 + torch.randn(xb.shape[0], xb.shape[1], 1, device=xb.device) * 0.1)
    return xb

class CNN(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(6, 64, 15, padding=7),  nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 9, padding=4), nn.BatchNorm1d(128),nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128,128, 5, padding=2), nn.BatchNorm1d(128),nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_cls))
    def forward(self, x): return self.head(self.net(x))

gkf = GroupKFold(n_splits=N_FOLDS)
all_true, all_pred, fold_f1 = [], [], []

for fold, (tr, te) in enumerate(gkf.split(X, Y, groups), start=1):
    print(f"\n===== FOLD {fold}/{N_FOLDS} =====", flush=True)
    print(f"  train users {len(set(groups[tr]))}, test users {len(set(groups[te]))}", flush=True)

    samp = tr if len(tr) < 200000 else np.random.default_rng(0).choice(tr, 200000, replace=False)
    flat = X[samp].reshape(-1, 6)
    mean = flat.mean(0).astype(np.float32); std = (flat.std(0) + 1e-6).astype(np.float32)

    counts = np.bincount(Y[tr], minlength=len(classes)).astype(np.float64)
    print("  train counts:", {classes[i]: int(counts[i]) for i in range(len(classes))}, flush=True)
    wpc = 1.0 / np.sqrt(np.maximum(counts, 1))     # sqrt-inverse: gentler than full inverse
    wpc = wpc / wpc.sum()
    sampler = WeightedRandomSampler(torch.DoubleTensor(wpc[Y[tr]]),
                                    num_samples=min(SAMP_PER_EP, len(tr)), replacement=True)

    dl_tr = DataLoader(WinDS(X, Y, tr, mean, std), batch_size=BATCH, sampler=sampler,
                       num_workers=2, pin_memory=True)
    dl_te = DataLoader(WinDS(X, Y, te, mean, std), batch_size=512, shuffle=False,
                       num_workers=2, pin_memory=True)

    model = CNN(len(classes)).to(DEV)
    cw = (1.0 / np.sqrt(np.maximum(counts, 1))).astype(np.float32)
    cw = cw / cw.mean()                    # normalize around 1
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(cw, device=DEV))
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for ep in range(1, EPOCHS + 1):
        model.train(); tot = seen = 0; t0 = time.time()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEV, non_blocking=True), yb.to(DEV, non_blocking=True)
            if torch.rand(1).item() < 0.5: xb = augment(xb)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
            tot += loss.item()*len(yb); seen += len(yb)
        print(f"  epoch {ep:2d}/{EPOCHS}  loss {tot/seen:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    model.eval(); preds = []
    with torch.no_grad():
        for xb, _ in dl_te:
            preds.append(model(xb.to(DEV)).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)

    print("  TRUE dist:", {classes[k]: v for k,v in sorted(collections.Counter(Y[te].tolist()).items())}, flush=True)
    print("  PRED dist:", {classes[k]: v for k,v in sorted(collections.Counter(pred.tolist()).items())}, flush=True)
    per = f1_score(Y[te], pred, average=None, labels=list(range(len(classes))), zero_division=0)
    print("  per-class F1:", {classes[i]: round(float(per[i]),3) for i in range(len(classes))}, flush=True)

    f1 = f1_score(Y[te], pred, average="macro")
    fold_f1.append(f1); all_true.append(Y[te]); all_pred.append(pred)
    print(f"  fold macro-F1: {f1:.3f}", flush=True)

    if fold == 1:
        torch.save({"state_dict": model.state_dict(), "classes": classes,
                    "mean": mean, "std": std}, "cnn_model_40hz.pt")

yt = np.concatenate(all_true); yp = np.concatenate(all_pred)
print("\n==== CNN 5-FOLD RESULTS ====", flush=True)
print(classification_report(yt, yp, target_names=classes, digits=3))
print(f"mean macro-F1: {np.mean(fold_f1):.3f} (+/- {np.std(fold_f1):.3f})")
np.savez("cnn_results.npz", cm=confusion_matrix(yt, yp), classes=np.array(classes),
         y_true=yt, y_pred=yp, fold_f1=np.array(fold_f1))
print("Saved cnn_results.npz")