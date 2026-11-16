"""
03_gbp_audit.py - utility and fairness under each intervention (the paper's
fairness table): unedited, kappa-gated projection, ungated erasure, INLP, LEACE.

THREE CHANGES FROM THE SUBMITTED VERSION

1. PARAMETERISED. The submitted version had no arguments and read
   data/activations_seed{N}.npz, the untagged files from the original cohort,
   so it kept reproducing the pre-correction table no matter what 01 and 02
   had been re-run with. It now takes --tag, the same suffix 02 and 06 use.

2. SURVEY-WEIGHTED, the same way 06 is. The head is refit with sample_weight,
   the bias direction is a weighted group-mean difference, INLP probes and
   LEACE moments use the weights, and every AUROC, ECE, parity gap and
   equalized-odds gap is a weighted estimate. The whole table is computed
   both ways, and the script prints how far weighting moves each cell.

3. LEACE IN NUMPY. The closed form (Belrose et al. 2023, Thm 4.3) is
   implemented directly, so the script no longer needs concept-erasure or
   torch. --check-leace compares it to the library when the library is
   installed, and the script always verifies the defining property itself:
   zero cross-covariance between the erased representation and the attribute.

Run:  python 03_gbp_audit.py --tag _fast
      python 03_gbp_audit.py --tag _mec
Deps: scikit-learn, pandas, numpy, pyarrow
"""
import argparse, os, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="", help="suffix used by 02_train_models.py")
ap.add_argument("--out", default="results")
ap.add_argument("--check-leace", action="store_true",
                help="compare the numpy LEACE to concept-erasure, if installed")
args = ap.parse_args()

TAG = args.tag
SEEDS = [0, 1, 2, 3, 4]
K_FULL, K_PROTECT = 0.30, 0.50      # the angle-only gate thresholds, GBP-Audit
os.makedirs(args.out, exist_ok=True)
W = 78

gpath = f"data/groups{TAG}.parquet"
if not os.path.exists(gpath):
    sys.exit(f"{gpath} not found. Run 02_train_models.py with --tag {TAG or '(none)'} first.")
groups = pd.read_parquet(gpath)
ATTRS = {"sex_female": (groups.sex_female.values == 1),
         "race_black": (groups.race.values == "black_nh"),
         "race_hisp":  (groups.race.values == "hispanic"),
         "age_60plus": (groups.age.values >= 60)}
REF = {"sex_female": None,
       "age_60plus": (groups.age.values < 45),
       "race_black": (groups.race.values == "white_nh"),
       "race_hisp":  (groups.race.values == "white_nh")}
LABEL = {"age_60plus": "Age >= 60", "sex_female": "Sex (F)",
         "race_black": "Black NH", "race_hisp": "Hispanic"}
METHODS = ["unedited", "gbp_gated", "erase_ungated", "inlp", "leace"]
MCOLS = ["auroc", "ece", "dp_gap", "eo_gap", "auroc_g", "ece_g"]


def wmean(x, w=None):
    """Weighted mean, nan for an empty selection."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    return float(x.mean()) if w is None else float(np.average(x, weights=w))


def safe_auc(y, p, w=None):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p, sample_weight=w)


def ece(y, p, w=None, bins=10):
    """Expected calibration error. Bin mass is the weighted share of the
    sample, and accuracy and confidence are weighted means within the bin."""
    ww = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    tot, e = ww.sum(), 0.0
    for i in range(bins):
        hi = (p <= edges[i + 1]) if i == bins - 1 else (p < edges[i + 1])
        m = (p >= edges[i]) & hi
        if m.any():
            e += ww[m].sum() / tot * abs(np.average(y[m], weights=ww[m])
                                         - np.average(p[m], weights=ww[m]))
    return float(e)


def metrics(y, p, g, ref, w=None):
    yhat = (p >= 0.5).astype(float)
    sel = lambda m: None if w is None else w[m]
    rate = lambda m: wmean(yhat[m], sel(m))
    dp = abs(rate(g) - rate(ref))
    tpr_gap = abs(rate(g & (y == 1)) - rate(ref & (y == 1)))
    fpr_gap = abs(rate(g & (y == 0)) - rate(ref & (y == 0)))
    eo = np.nanmax([tpr_gap, fpr_gap]) if not (np.isnan(tpr_gap) and np.isnan(fpr_gap)) else np.nan
    return {"auroc": safe_auc(y, p, w), "ece": ece(y, p, w),
            "dp_gap": dp, "eo_gap": eo,
            "auroc_g": safe_auc(y[g], p[g], sel(g)),
            "ece_g": ece(y[g], p[g], sel(g))}


def leace_fit(X, z, w=None):
    """Closed-form LEACE. Returns (mu, P) with the eraser x -> x - (x - mu) P^T.
    P = W^+ Q W, where W is the pseudo-inverse square root of Cov(X) and Q is
    the orthogonal projector onto the column space of W Cov(X, z)."""
    X = np.asarray(X, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).reshape(len(X), -1)
    ww = np.ones(len(X)) if w is None else np.asarray(w, dtype=np.float64)
    ww = ww / ww.sum()
    mu, mz = ww @ X, ww @ z
    Xc, zc = X - mu, z - mz
    Sxx = (Xc * ww[:, None]).T @ Xc
    Sxz = (Xc * ww[:, None]).T @ zc
    L, V = np.linalg.eigh(Sxx)
    keep = L > L.max() * Sxx.shape[0] * np.finfo(np.float64).eps
    Wm = (V[:, keep] / np.sqrt(L[keep])) @ V[:, keep].T       # whitening
    Wp = (V[:, keep] * np.sqrt(L[keep])) @ V[:, keep].T       # its pseudo-inverse
    U, s, _ = np.linalg.svd(Wm @ Sxz, full_matrices=False)
    U = U[:, s > s.max() * 1e-10] if s.size and s.max() > 0 else U[:, :0]
    return mu, Wp @ (U @ U.T) @ Wm


def leace_apply(X, mu, P):
    return X - (X - mu) @ P.T


def run(weighted):
    rows, kaps, worst_xcov = [], [], 0.0
    for seed in SEEDS:
        path = f"data/activations_seed{seed}{TAG}.npz"
        if not os.path.exists(path):
            sys.exit(f"{path} not found. Run 02_train_models.py --tag {TAG} first.")
        z = np.load(path)
        if "wt" not in z:
            sys.exit(f"{path} has no survey weights. Re-run 02_train_models.py.")
        H, y = z["H"].astype(np.float64), z["y"]
        tr, te, wt = z["idx_train"], z["idx_test"], z["wt"]
        sw_tr = wt[tr] if weighted else None
        sw_te = wt[te] if weighted else None

        # same head refit as 06, so kappa and impact here match that script
        head = LogisticRegression(max_iter=1000).fit(H[tr], y[tr], sample_weight=sw_tr)
        w, b = head.coef_.squeeze(), head.intercept_[0]
        v_t = w / np.linalg.norm(w)
        logit_std = np.std(H[tr] @ w)
        prob = lambda Hm: 1 / (1 + np.exp(-(Hm @ w + b)))

        for attr, g in ATTRS.items():
            ref = REF[attr] if REF[attr] is not None else ~g
            gtr, rtr = g[tr], ref[tr]
            if weighted:
                mu_g = np.average(H[tr][gtr], axis=0, weights=wt[tr][gtr])
                mu_r = np.average(H[tr][rtr], axis=0, weights=wt[tr][rtr])
            else:
                mu_g, mu_r = H[tr][gtr].mean(0), H[tr][rtr].mean(0)
            v_b = mu_g - mu_r
            v_b /= np.linalg.norm(v_b)
            kappa = abs(float(v_b @ v_t))
            impact = kappa * np.linalg.norm(w) * np.std(H[tr] @ v_b) / logit_std
            policy = "FULL" if kappa < K_FULL else ("PARTIAL" if kappa <= K_PROTECT else "PROTECT")
            s = 1.0 if policy == "FULL" else ((1.0 - kappa) if policy == "PARTIAL" else 0.0)
            kaps.append({"seed": seed, "attr": attr, "kappa": kappa, "impact": impact,
                         "policy": policy, "s": s})

            proj = lambda Hm, sc: Hm - sc * np.outer(Hm @ v_b, v_b)
            variants = {"unedited": H, "gbp_gated": proj(H, s),
                        "erase_ungated": proj(H, 1.0)}

            Hi = H.copy()                       # INLP, five rounds
            for _ in range(5):
                clf = LogisticRegression(max_iter=500).fit(Hi[tr], g[tr], sample_weight=sw_tr)
                u = clf.coef_.squeeze()
                u /= np.linalg.norm(u)
                Hi = Hi - np.outer(Hi @ u, u)
            variants["inlp"] = Hi

            gz = g[tr].astype(np.float64)
            mu, P = leace_fit(H[tr], gz, sw_tr)
            Hl = leace_apply(H, mu, P)
            variants["leace"] = Hl
            # defining property: no linear covariance left with the attribute
            wn = (np.ones(len(tr)) if sw_tr is None else sw_tr)
            wn = wn / wn.sum()
            xc = Hl[tr] - wn @ Hl[tr]
            worst_xcov = max(worst_xcov, float(np.abs((xc * wn[:, None]).T @ (gz - wn @ gz)).max()))

            for name in METHODS:
                r = metrics(y[te], prob(variants[name])[te], g[te], ref[te], sw_te)
                rows.append({"seed": seed, "attr": attr, "method": name,
                             "kappa": round(kappa, 4), "impact": round(impact, 4),
                             "policy": policy,
                             **{k: (np.nan if np.isnan(v) else round(float(v), 4))
                                for k, v in r.items()}})
    return pd.DataFrame(rows), pd.DataFrame(kaps), worst_xcov


if args.check_leace:
    try:
        import torch
        from concept_erasure import LeaceEraser
        z0 = np.load(f"data/activations_seed0{TAG}.npz")
        Htr0 = z0["H"][z0["idx_train"]].astype(np.float64)
        g0 = ATTRS["sex_female"][z0["idx_train"]].astype(np.float64)
        er = LeaceEraser.fit(torch.tensor(Htr0), torch.tensor(g0).unsqueeze(1))
        lib = er(torch.tensor(z0["H"].astype(np.float64))).numpy()
        mine = leace_apply(z0["H"].astype(np.float64), *leace_fit(Htr0, g0))
        print(f"LEACE check, numpy vs concept-erasure: max abs difference "
              f"{np.abs(lib - mine).max():.2e} on activations of scale "
              f"{np.abs(z0['H']).max():.2f}")
    except ImportError:
        print("LEACE check skipped: concept-erasure or torch is not installed.")
    print()

print("computing unweighted (as submitted) ...")
ru, ku, xu = run(weighted=False)
print("computing survey-weighted ...")
rw, kw, xw = run(weighted=True)

rw.to_csv(f"{args.out}/fairness_utility_raw{TAG}.csv", index=False)
ru.to_csv(f"{args.out}/fairness_utility_raw_unweighted{TAG}.csv", index=False)
kw.round(4).to_csv(f"{args.out}/kappa_policies{TAG}.csv", index=False)
summ = rw.groupby(["attr", "method"])[MCOLS].agg(["mean", "std"]).round(4)
summ.to_csv(f"{args.out}/table_fairness_summary{TAG}.csv")

print()
print("=" * W)
print(f"GATE STATISTICS, survey-weighted, mean over {len(SEEDS)} seeds  (tag '{TAG}')")
print("=" * W)
km = kw.groupby("attr")[["kappa", "impact"]].mean()
kp = kw.groupby("attr")["policy"].agg(lambda s_: s_.value_counts().to_dict())
for a in ATTRS:
    print(f"  {LABEL[a]:12s} kappa {km.loc[a, 'kappa']:.3f}   impact {km.loc[a, 'impact']:.3f}"
          f"   angle-only verdicts {kp.loc[a]}")
print("  These should match 06_impact_gate.py on the same tag. If they do not,")
print("  the two scripts are reading different activations.")

print()
print("=" * W)
print("FAIRNESS TABLE, survey-weighted, mean (SD) over seeds")
print("=" * W)
mean = rw.groupby(["attr", "method"])[MCOLS].mean()
sd = rw.groupby(["attr", "method"])[MCOLS].std()
print(f"  {'attribute':12s} {'method':14s} {'AUROC':>13s} {'ECE':>13s} {'DP gap':>13s} {'EO gap':>13s}")
for a in ["age_60plus", "sex_female", "race_black", "race_hisp"]:
    for mth in METHODS:
        cell = lambda c: f"{mean.loc[(a, mth), c]:.3f} ({sd.loc[(a, mth), c]:.3f})"
        print(f"  {LABEL[a]:12s} {mth:14s} {cell('auroc'):>13s} {cell('ece'):>13s}"
              f" {cell('dp_gap'):>13s} {cell('eo_gap'):>13s}")
    print()

print("=" * W)
print("HOW FAR WEIGHTING MOVES THE TABLE  (weighted minus unweighted, seed means)")
print("=" * W)
mu_ = ru.groupby(["attr", "method"])[MCOLS].mean()
delta = (mean - mu_)[["auroc", "ece", "dp_gap", "eo_gap"]]
print(f"  largest absolute shift:  AUROC {delta.auroc.abs().max():.3f}   ECE {delta.ece.abs().max():.3f}"
      f"   DP {delta.dp_gap.abs().max():.3f}   EO {delta.eo_gap.abs().max():.3f}")
a_, m_ = delta.abs().max(axis=1).idxmax()
print(f"  the cell that moves most: {LABEL[a_]}, {m_}")

print()
print(f"LEACE self-check: largest remaining cross-covariance with the attribute, "
      f"{max(xu, xw):.1e} (should be ~0)")
if rw.loc[rw.method == "gbp_gated", "policy"].eq("FULL").all():
    print()
    print("NOTE the angle-only gate returned FULL for every case, so gbp_gated and")
    print("     erase_ungated are the same rows. That is the gate's failure, not a")
    print("     bug: at these kappa values it approves full erasure everywhere.")

print()
print(f"wrote {args.out}/fairness_utility_raw{TAG}.csv              (weighted, the one to quote)")
print(f"      {args.out}/fairness_utility_raw_unweighted{TAG}.csv")
print(f"      {args.out}/kappa_policies{TAG}.csv")
print(f"      {args.out}/table_fairness_summary{TAG}.csv")
