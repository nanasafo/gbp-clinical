"""
12_model_justification.py - does the MLP earn its place?

AUROC says the MLP beats logistic regression by under one point, which is not
a justification you can defend. These four tests answer the question properly
instead of asserting an answer.

  1 CALIBRATION   Weighted Brier, calibration slope/intercept, ECE. AUROC only
                  measures ranking and is insensitive to exactly the kind of
                  improvement a nonlinear model usually delivers. For a
                  clinical risk model calibration is what matters at the point
                  of use, and it is the most likely place to find a real and
                  defensible difference.
  2 NONLINEARITY  Logistic regression on splines + pairwise interactions. If
                  that closes the gap, the MLP is learning structure you can
                  write down, so write it down. If it does not, the MLP
                  captures something beyond two-way interactions and THAT is
                  the justification, stated as a result.
  3 LEARNING      AUROC against training-set size for both models. Converging
     CURVE        curves mean more data will not rescue the MLP. A widening
                  gap justifies it specifically at the larger cohort.
                  Needs torch; run with --curve.
  4 DELTA CI      Stratified cluster bootstrap on the AUROC DIFFERENCE, PSUs
                  resampled within strata, survey-weighted. An interval on the
                  difference, not a hand-waved p-value. If it spans zero, say
                  so in the paper.

Run:  python 12_model_justification.py --tag _mec
      python 12_model_justification.py --tag _mec --curve
      python 12_model_justification.py --tag _fast --boot 2000
Deps: pip install scikit-learn pandas numpy scipy  (torch only for --curve)
"""
import argparse, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import SplineTransformer, PolynomialFeatures
from sklearn.pipeline import make_pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="_mec", help="suffix used by 02_train_models.py")
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--curve", action="store_true", help="run the learning curve (needs torch)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

TAG = args.tag
SEEDS = [0, 1, 2, 3, 4]
rng = np.random.default_rng(args.seed)
W = 74


def wauc(y, p, w):
    return roc_auc_score(y, p, sample_weight=w)


def wbrier(y, p, w):
    return float(np.average((p - y) ** 2, weights=w))


def calibration(y, p, w):
    """Weighted calibration slope and intercept from y ~ logit(p)."""
    eps = 1e-6
    lo = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    m = LogisticRegression(max_iter=1000).fit(lo.reshape(-1, 1), y, sample_weight=w)
    return float(m.coef_[0][0]), float(m.intercept_[0])


def ece(y, p, w, bins=10):
    """Weighted expected calibration error."""
    edges = np.linspace(0, 1, bins + 1)
    tot, out = w.sum(), 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if not m.any():
            continue
        ww = w[m].sum()
        out += (ww / tot) * abs(np.average(y[m], weights=w[m])
                                - np.average(p[m], weights=w[m]))
    return float(out)


def cluster_idx(st, ps, rng):
    out = []
    for s in np.unique(st):
        rows = np.where(st == s)[0]
        for p in rng.choice(np.unique(ps[rows]), size=len(np.unique(ps[rows])), replace=True):
            out.append(rows[ps[rows] == p])
    return np.concatenate(out) if out else np.arange(len(st))


def load(seed):
    z = np.load(f"data/activations_seed{seed}{TAG}.npz")
    need = ["X_scaled", "y", "idx_train", "idx_test", "p", "wt", "strata", "psu"]
    miss = [k for k in need if k not in z]
    if miss:
        raise RuntimeError(
            f"data/activations_seed{seed}{TAG}.npz is missing {miss}. "
            f"Re-run 02_train_models.py (Step 2 version)."
        )
    return z


rows_cal, rows_nl, rows_d = [], [], []
for seed in SEEDS:
    z = load(seed)
    X, y = z["X_scaled"], z["y"]
    tr, te = z["idx_train"], z["idx_test"]
    wt, st, ps = z["wt"], z["strata"], z["psu"]
    p_mlp = z["p"]

    lr = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
    p_lr = lr.predict_proba(X)[:, 1]

    for nm, p in [("MLP", p_mlp), ("LogReg", p_lr)]:
        sl, ic = calibration(y[te], p[te], wt[te])
        rows_cal.append({"seed": seed, "model": nm,
                         "auroc": wauc(y[te], p[te], wt[te]),
                         "brier": wbrier(y[te], p[te], wt[te]),
                         "cal_slope": sl, "cal_intercept": ic,
                         "ece": ece(y[te], p[te], wt[te])})

    # --- 2. explicit nonlinearity -------------------------------------
    cont = [i for i in range(X.shape[1]) if len(np.unique(X[:, i])) > 2]
    binr = [i for i in range(X.shape[1]) if i not in cont]
    sp = SplineTransformer(n_knots=5, degree=3).fit(X[tr][:, cont])
    def expand(M):
        return np.hstack([sp.transform(M[:, cont]), M[:, binr]])
    Xs_tr, Xs_te = expand(X[tr]), expand(X[te])
    lr_sp = LogisticRegression(max_iter=4000).fit(Xs_tr, y[tr])
    auc_sp = wauc(y[te], lr_sp.predict_proba(Xs_te)[:, 1], wt[te])

    pf = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    Xi_tr, Xi_te = pf.fit_transform(X[tr]), pf.transform(X[te])
    lr_ix = LogisticRegression(max_iter=4000).fit(Xi_tr, y[tr])
    auc_ix = wauc(y[te], lr_ix.predict_proba(Xi_te)[:, 1], wt[te])

    both_tr = np.hstack([Xs_tr, Xi_tr])
    both_te = np.hstack([Xs_te, Xi_te])
    lr_bo = LogisticRegression(max_iter=6000).fit(both_tr, y[tr])
    auc_bo = wauc(y[te], lr_bo.predict_proba(both_te)[:, 1], wt[te])

    rows_nl.append({"seed": seed,
                    "logreg": wauc(y[te], p_lr[te], wt[te]),
                    "lr_splines": auc_sp, "lr_interactions": auc_ix,
                    "lr_both": auc_bo, "mlp": wauc(y[te], p_mlp[te], wt[te])})

    # --- 4. CI on the AUROC difference --------------------------------
    yte, wte = y[te], wt[te]
    pm, pl = p_mlp[te], p_lr[te]
    ste, pse = st[te], ps[te]
    draws = np.empty(args.boot)
    for b in range(args.boot):
        k = cluster_idx(ste, pse, rng)
        if len(np.unique(yte[k])) < 2:
            draws[b] = np.nan; continue
        try:
            draws[b] = (roc_auc_score(yte[k], pm[k], sample_weight=wte[k])
                        - roc_auc_score(yte[k], pl[k], sample_weight=wte[k]))
        except ValueError:
            draws[b] = np.nan
    draws = draws[~np.isnan(draws)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    rows_d.append({"seed": seed, "delta": wauc(yte, pm, wte) - wauc(yte, pl, wte),
                   "lo": lo, "hi": hi})

cal = pd.DataFrame(rows_cal)
nl = pd.DataFrame(rows_nl)
dd = pd.DataFrame(rows_d)

print("=" * W)
print(f"1. CALIBRATION  (cohort tag {TAG or '(none)'}, survey-weighted, mean over 5 seeds)")
print("=" * W)
g = cal.groupby("model").agg(["mean", "std"])
print(f"  {'model':8s} {'AUROC':>16s} {'Brier':>16s} {'cal slope':>15s} {'ECE':>14s}")
for m in ["LogReg", "MLP"]:
    r = cal[cal.model == m]
    print(f"  {m:8s} {r.auroc.mean():8.4f} +/-{r.auroc.std(ddof=1):6.4f}"
          f" {r.brier.mean():8.4f} +/-{r.brier.std(ddof=1):6.4f}"
          f" {r.cal_slope.mean():7.3f} +/-{r.cal_slope.std(ddof=1):5.3f}"
          f" {r.ece.mean():7.4f} +/-{r.ece.std(ddof=1):5.4f}")
db = cal[cal.model == "LogReg"].brier.mean() - cal[cal.model == "MLP"].brier.mean()
de = cal[cal.model == "LogReg"].ece.mean() - cal[cal.model == "MLP"].ece.mean()
print(f"\n  Brier improvement from the MLP: {db:+.4f}  (positive favours MLP)")
print(f"  ECE   improvement from the MLP: {de:+.4f}  (positive favours MLP)")
print("  Calibration slope of 1.0 is perfect; below 1 means overconfident.")
print("  A clear Brier/ECE gain is a defensible reason to prefer the MLP even")
print("  when AUROC ties, because calibration is what matters at the bedside.")

print()
print("=" * W)
print("2. IS THERE NONLINEAR STRUCTURE A LINEAR MODEL CAN REACH?")
print("=" * W)
order = ["logreg", "lr_splines", "lr_interactions", "lr_both", "mlp"]
names = {"logreg": "logistic regression", "lr_splines": "+ splines",
         "lr_interactions": "+ pairwise interactions", "lr_both": "+ both",
         "mlp": "MLP (64, 32)"}
for c in order:
    print(f"  {names[c]:28s} {nl[c].mean():.4f} +/- {nl[c].std(ddof=1):.4f}")
gap_before = nl.mlp.mean() - nl.logreg.mean()
gap_after = nl.mlp.mean() - nl.lr_both.mean()
print(f"\n  MLP advantage over plain logistic regression : {gap_before:+.4f}")
print(f"  MLP advantage over the expanded linear model : {gap_after:+.4f}")
if gap_after <= 0.002:
    print("  VERDICT the expanded linear model closes the gap. The MLP is")
    print("          learning structure you can write down explicitly, so")
    print("          write it down and drop the architecture argument.")
else:
    print("  VERDICT the gap survives explicit splines and interactions, so the")
    print("          MLP captures structure beyond two-way effects. Report this")
    print("          table; it IS the justification.")

print()
print("=" * W)
print("4. AUROC DIFFERENCE, MLP MINUS LOGREG, WITH DESIGN-BASED 95% CI")
print("=" * W)
for _, r in dd.iterrows():
    star = "" if (r.lo <= 0 <= r.hi) else "  *excludes 0"
    print(f"  seed {int(r.seed)}:  {r.delta:+.4f}   CI [{r.lo:+.4f}, {r.hi:+.4f}]{star}")
print(f"\n  mean difference {dd.delta.mean():+.4f}, "
      f"mean CI [{dd.lo.mean():+.4f}, {dd.hi.mean():+.4f}]")
n_excl = int((~((dd.lo <= 0) & (0 <= dd.hi))).sum())
print(f"  {n_excl} of 5 seed intervals exclude zero.")
if n_excl <= 2:
    print("  Report the difference as not distinguishable from zero and justify")
    print("  the MLP on calibration or on the framework, not on discrimination.")

cal.to_csv(f"results/justify_calibration{TAG}.csv", index=False)
nl.to_csv(f"results/justify_nonlinearity{TAG}.csv", index=False)
dd.to_csv(f"results/justify_delta_auc{TAG}.csv", index=False)
print(f"\nwrote results/justify_*{TAG}.csv")

# ---------------------------------------------------------------- 3. curve
if args.curve:
    import torch, torch.nn as nn
    print()
    print("=" * W)
    print("3. LEARNING CURVE")
    print("=" * W)
    FRACS = [0.10, 0.25, 0.50, 1.00]
    out = []
    for seed in SEEDS:
        z = load(seed)
        X, y = z["X_scaled"].astype(np.float32), z["y"].astype(np.float32)
        tr, te, wt = z["idx_train"], z["idx_test"], z["wt"]
        r = np.random.default_rng(seed)
        for f in FRACS:
            sub = r.permutation(tr)[:max(50, int(f * len(tr)))]
            lr = LogisticRegression(max_iter=2000).fit(X[sub], y[sub])
            a_lr = wauc(y[te], lr.predict_proba(X[te])[:, 1], wt[te])
            torch.manual_seed(seed)
            m = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(),
                              nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            lf = nn.BCEWithLogitsLoss()
            Xt, yt = torch.tensor(X), torch.tensor(y)
            for _ in range(200):
                opt.zero_grad()
                lf(m(Xt[sub]).squeeze(-1), yt[sub]).backward()
                opt.step()
            with torch.no_grad():
                pm = torch.sigmoid(m(Xt).squeeze(-1)).numpy()
            out.append({"seed": seed, "frac": f, "n": len(sub),
                        "logreg": a_lr, "mlp": wauc(y[te], pm[te], wt[te])})
    cv = pd.DataFrame(out)
    print(f"  {'train n':>9s} {'LogReg':>9s} {'MLP':>9s} {'delta':>9s}")
    for f in FRACS:
        s = cv[cv.frac == f]
        print(f"  {int(s.n.mean()):9d} {s.logreg.mean():9.4f} {s.mlp.mean():9.4f} "
              f"{s.mlp.mean()-s.logreg.mean():+9.4f}")
    d0 = cv[cv.frac == FRACS[0]].mlp.mean() - cv[cv.frac == FRACS[0]].logreg.mean()
    d1 = cv[cv.frac == 1.0].mlp.mean() - cv[cv.frac == 1.0].logreg.mean()
    print(f"\n  gap at smallest n {d0:+.4f} -> at full n {d1:+.4f}")
    print("  Widening favours the MLP at scale; converging says more data will")
    print("  not rescue it and the linear model is the honest primary.")
    cv.to_csv(f"results/justify_learning_curve{TAG}.csv", index=False)
    print(f"  wrote results/justify_learning_curve{TAG}.csv")
