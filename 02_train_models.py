"""
02_train_models.py - train risk models on the NHANES cohort, dump penultimate
activations, final-layer weights and the survey design for the GBP audit.

STEP 2 CHANGES:
  1. Reports weighted AUROC (sample_weight=WT_POOLED) alongside unweighted.
     Unweighted is a statistic about the sample; weighted estimates the
     population. Report both; if they agree, the objection is closed.
  2. Carries WT_POOLED, SDMVSTRA and SDMVPSU into the npz so 06 can run a
     design-based cluster bootstrap instead of resampling rows.
  3. --cohort / --no-tg / --tag so the fasting and full-MEC cohorts can be run
     side by side without overwriting each other.
  4. --hidden parameterises the architecture (Step 4 needs this anyway).

COHORT TIEBREAKER - run these three and compare mean weighted AUROC:
  python 02_train_models.py --cohort data/nhanes_cohort.parquet --tag _fast
  python 02_train_models.py --cohort data/nhanes_cohort.parquet --no-tg --tag _fast_notg
  python 02_train_models.py --cohort data/nhanes_cohort_noTG.parquet --tag _mec
The first two differ only in log_tg on identical rows, which is the clean test
of whether triglycerides earn their place. If they are close, the third cohort
(2x the rows, bigger subgroups) is the better primary analysis.

Deps: pip install torch scikit-learn pandas pyarrow
Outputs: models/mlp_seed{k}{tag}.pt, data/activations_seed{k}{tag}.npz,
         results/model_perf{tag}.csv
"""
import os, argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--cohort", default="data/nhanes_cohort.parquet")
ap.add_argument("--no-tg", action="store_true", help="drop log_tg from features")
ap.add_argument("--hidden", default="64,32", help="comma-separated hidden sizes")
ap.add_argument("--tag", default="", help="suffix for output filenames")
args = ap.parse_args()

TAG = args.tag
HIDDEN = [int(x) for x in args.hidden.split(",") if x.strip()]
SEEDS = [0, 1, 2, 3, 4]
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

df = pd.read_parquet(args.cohort)

BASE_FEATURES = ["BMXBMI", "BMXWAIST", "SBP", "LBXTC", "LBDHDD", "log_tg", "smoker_current"]
FEATURES = [f for f in BASE_FEATURES if f in df.columns]
if args.no_tg and "log_tg" in FEATURES:
    FEATURES.remove("log_tg")
missing = [f for f in BASE_FEATURES if f not in df.columns and f != "log_tg"]
if missing:
    raise RuntimeError(f"{args.cohort} is missing {missing}")

for col in ["WT_POOLED", "SDMVSTRA", "SDMVPSU"]:
    if col not in df.columns:
        raise RuntimeError(
            f"{args.cohort} has no {col}. Rebuild it with the current "
            f"01_nhanes_download_prep.py (rev 3 or later)."
        )

X = df[FEATURES].values.astype(np.float32)
OUTCOME = "outcome" if "outcome" in df.columns else "diabetes"
y = df[OUTCOME].values.astype(np.float32)
w = df["WT_POOLED"].values.astype(np.float64)
strata = df["SDMVSTRA"].values
psu = df["SDMVPSU"].values
groups = df[["sex_female", "race", "age"]].reset_index(drop=True)

print(f"cohort   : {args.cohort}  N={len(df)}  outcome={OUTCOME}")
print(f"features : {', '.join(FEATURES)}")
print(f"hidden   : {HIDDEN if HIDDEN else 'none (logistic head only)'}")
print(f"tag      : {TAG or '(none)'}")
print()


class MLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        layers, prev = [], d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.body = nn.Sequential(*layers)   # empty body => linear model
        self.head = nn.Linear(prev, 1)

    def forward(self, x, return_h=False):
        h = self.body(x)
        out = self.head(h).squeeze(-1)
        return (out, h) if return_h else out


perf = []
for seed in SEEDS:
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.4, random_state=seed, stratify=y)
    va, te = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=y[tmp])
    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X).astype(np.float32)

    lr = LogisticRegression(max_iter=2000).fit(Xs[tr], y[tr])
    p_lr = lr.predict_proba(Xs[te])[:, 1]
    auc_lr = roc_auc_score(y[te], p_lr)
    auc_lr_w = roc_auc_score(y[te], p_lr, sample_weight=w[te])

    torch.manual_seed(seed)
    m = MLP(len(FEATURES), HIDDEN)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    Xt, yt = torch.tensor(Xs), torch.tensor(y)
    best, patience, state = 1e9, 0, m.state_dict()
    for epoch in range(200):
        m.train(); opt.zero_grad()
        loss = lossf(m(Xt[tr]), yt[tr]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = lossf(m(Xt[va]), yt[va]).item()
        if vl < best - 1e-4:
            best, state, patience = vl, m.state_dict(), 0
        else:
            patience += 1
            if patience >= 15:
                break
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        logits, H = m(Xt, return_h=True)
        p = torch.sigmoid(logits).numpy()

    auc_mlp = roc_auc_score(y[te], p[te])
    auc_mlp_w = roc_auc_score(y[te], p[te], sample_weight=w[te])
    perf.append({"seed": seed, "n": len(y), "n_features": len(FEATURES),
                 "hidden": args.hidden,
                 "auc_logreg": auc_lr, "auc_logreg_wtd": auc_lr_w,
                 "auc_mlp": auc_mlp, "auc_mlp_wtd": auc_mlp_w})
    print(f"seed {seed}: LogReg {auc_lr:.3f} (wtd {auc_lr_w:.3f})   "
          f"MLP {auc_mlp:.3f} (wtd {auc_mlp_w:.3f})")

    torch.save(m.state_dict(), f"models/mlp_seed{seed}{TAG}.pt")
    np.savez(f"data/activations_seed{seed}{TAG}.npz",
             H=H.numpy(), p=p, y=y, idx_train=tr, idx_val=va, idx_test=te,
             w_task=m.head.weight.detach().numpy().squeeze(),
             X_scaled=Xs, wt=w, strata=strata, psu=psu)

groups.to_parquet(f"data/groups{TAG}.parquet")
pf = pd.DataFrame(perf)
pf.to_csv(f"results/model_perf{TAG}.csv", index=False)

print()
print("=" * 62)
print(f"mean MLP AUROC   unweighted {pf.auc_mlp.mean():.4f} "
      f"+/- {pf.auc_mlp.std(ddof=1):.4f}")
print(f"mean MLP AUROC   WEIGHTED   {pf.auc_mlp_wtd.mean():.4f} "
      f"+/- {pf.auc_mlp_wtd.std(ddof=1):.4f}")
print(f"mean LogReg AUROC WEIGHTED  {pf.auc_logreg_wtd.mean():.4f} "
      f"+/- {pf.auc_logreg_wtd.std(ddof=1):.4f}")
gap = abs(pf.auc_mlp.mean() - pf.auc_mlp_wtd.mean())
print(f"weighted/unweighted gap     {gap:.4f}")
print("A small gap means the unweighted AUROC you published is close to the")
print("population estimate; say so in one sentence and the objection closes.")
print("=" * 62)
print(f"\nwrote results/model_perf{TAG}.csv and "
      f"data/activations_seed*{TAG}.npz")
