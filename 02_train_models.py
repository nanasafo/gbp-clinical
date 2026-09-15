"""
02_train_models.py — train risk models on the NHANES cohort, dump penultimate
activations + final-layer weights for the GBP audit.

Run:  python 02_train_models.py
Deps: pip install torch scikit-learn pandas pyarrow
Outputs: models/mlp_seed{k}.pt, data/activations_seed{k}.npz, results/model_perf.csv
"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEEDS = [0,1,2,3,4]
FEATURES = ["BMXBMI","BMXWAIST","SBP","LBXTC","LBDHDD","log_tg","smoker_current"]
os.makedirs("models", exist_ok=True); os.makedirs("results", exist_ok=True)

df = pd.read_parquet("data/nhanes_cohort.parquet")
X = df[FEATURES].values.astype(np.float32)
y = df["diabetes"].values.astype(np.float32)
groups = df[["sex_female","race","age"]].reset_index(drop=True)

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU())
        self.head = nn.Linear(32,1)
    def forward(self, x, return_h=False):
        h = self.body(x); out = self.head(h).squeeze(-1)
        return (out, h) if return_h else out

perf = []
for seed in SEEDS:
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.4, random_state=seed, stratify=y)
    va, te = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=y[tmp])
    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X).astype(np.float32)

    lr = LogisticRegression(max_iter=2000).fit(Xs[tr], y[tr])
    auc_lr = roc_auc_score(y[te], lr.predict_proba(Xs[te])[:,1])

    torch.manual_seed(seed)
    m = MLP(len(FEATURES)); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    Xt, yt = torch.tensor(Xs), torch.tensor(y)
    best, patience = 1e9, 0
    for epoch in range(200):
        m.train(); opt.zero_grad()
        loss = lossf(m(Xt[tr]), yt[tr]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = lossf(m(Xt[va]), yt[va]).item()
        if vl < best - 1e-4: best, state, patience = vl, m.state_dict(), 0
        else:
            patience += 1
            if patience >= 15: break
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        logits, H = m(Xt, return_h=True)
        p = torch.sigmoid(logits).numpy()
    auc_mlp = roc_auc_score(y[te], p[te])
    perf.append({"seed":seed,"auc_logreg":auc_lr,"auc_mlp":auc_mlp})
    print(f"seed {seed}: LogReg AUROC={auc_lr:.3f}  MLP AUROC={auc_mlp:.3f}")

    torch.save(m.state_dict(), f"models/mlp_seed{seed}.pt")
    np.savez(f"data/activations_seed{seed}.npz",
             H=H.numpy(), p=p, y=y, idx_train=tr, idx_val=va, idx_test=te,
             w_task=m.head.weight.detach().numpy().squeeze(),
             X_scaled=Xs)
groups.to_parquet("data/groups.parquet")
pd.DataFrame(perf).to_csv("results/model_perf.csv", index=False)
print("done; mean MLP AUROC:", np.mean([r["auc_mlp"] for r in perf]).round(3))
