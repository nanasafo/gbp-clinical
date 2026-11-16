"""
06_impact_gate.py - does impact-weighted coherence predict harm better than kappa?

STEP 2 CHANGES:
  1. Survey-weighted throughout: the probe is fit with sample_weight, the bias
     direction is a WEIGHTED group-mean difference, and every AUROC passes
     sample_weight. The whole statistic is computed both ways and printed side
     by side, so "does weighting change the conclusion?" is answered, not
     asserted.
  2. Design-based CIs from a STRATIFIED CLUSTER BOOTSTRAP: PSUs are resampled
     within strata, not rows. Resampling rows assumes an independence a
     clustered sample does not have and gives intervals that are too narrow.
     Expect these to be wider. Wide and defensible beats narrow and wrong.
  3. Both correlation framings. Spearman over all 20 rows treats 5 seeds as
     independent observations; they are re-splits of one cohort. The collapsed
     4-attribute version is what a careful reviewer will compute, so compute
     it first.
  4. Separation and seed-stability, which are the arguments that survive any
     framing of n.

Run:  python 06_impact_gate.py                      # fasting cohort
      python 06_impact_gate.py --tag _mec           # full MEC cohort
      python 06_impact_gate.py --boot 2000
Deps: pip install scikit-learn pandas numpy scipy pyarrow
"""
import argparse, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="", help="suffix used by 02_train_models.py")
ap.add_argument("--boot", type=int, default=1000, help="bootstrap replicates")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

TAG = args.tag
rng = np.random.default_rng(args.seed)
SEEDS = [0, 1, 2, 3, 4]

groups = pd.read_parquet(f"data/groups{TAG}.parquet")
ATTRS = {"sex_female": (groups.sex_female.values == 1),
         "race_black": (groups.race.values == "black_nh"),
         "race_hisp":  (groups.race.values == "hispanic"),
         "age_60plus": (groups.age.values >= 60)}
REF = {"sex_female": None,
       "age_60plus": (groups.age.values < 45),
       "race_black": (groups.race.values == "white_nh"),
       "race_hisp":  (groups.race.values == "white_nh")}
LABEL = {"race_hisp": "Hispanic", "race_black": "Black non-Hispanic",
         "sex_female": "Sex (female)", "age_60plus": "Age >= 60"}


def safe_auc(yy, pp, ww=None):
    """roc_auc_score, or nan when a resample has one class only."""
    if len(np.unique(yy)) < 2:
        return np.nan
    try:
        return roc_auc_score(yy, pp, sample_weight=ww)
    except ValueError:
        return np.nan


def cluster_boot_idx(strata_te, psu_te, rng):
    """Resample PSUs within strata. Returns positions into the test array."""
    out = []
    for s in np.unique(strata_te):
        rows_s = np.where(strata_te == s)[0]
        ps = np.unique(psu_te[rows_s])
        for p in rng.choice(ps, size=len(ps), replace=True):
            out.append(rows_s[psu_te[rows_s] == p])
    return np.concatenate(out) if out else np.arange(len(strata_te))


def run(weighted):
    """Compute the whole gate statistic weighted or unweighted."""
    rows = []
    for seed in SEEDS:
        z = np.load(f"data/activations_seed{seed}{TAG}.npz")
        H, y = z["H"], z["y"]
        tr, te = z["idx_train"], z["idx_test"]
        for col in ("wt", "strata", "psu"):
            if col not in z:
                raise RuntimeError(
                    f"data/activations_seed{seed}{TAG}.npz has no '{col}'. "
                    f"Re-run 02_train_models.py (Step 2 version)."
                )
        wt, strata, psu = z["wt"], z["strata"], z["psu"]
        sw_tr = wt[tr] if weighted else None
        sw_te = wt[te] if weighted else None

        lr = LogisticRegression(max_iter=1000).fit(H[tr], y[tr], sample_weight=sw_tr)
        w, b = lr.coef_.squeeze(), lr.intercept_[0]
        v_t = w / np.linalg.norm(w)
        logit_std = np.std(H[tr] @ w)
        prob = lambda Hm: 1 / (1 + np.exp(-(Hm @ w + b)))

        Htr, wtr = H[tr], wt[tr]
        for attr, g in ATTRS.items():
            ref = REF[attr] if REF[attr] is not None else ~g
            gtr, rtr = g[tr], ref[tr]
            if weighted:
                mu_g = np.average(Htr[gtr], axis=0, weights=wtr[gtr])
                mu_r = np.average(Htr[rtr], axis=0, weights=wtr[rtr])
            else:
                mu_g, mu_r = Htr[gtr].mean(0), Htr[rtr].mean(0)
            v_b = mu_g - mu_r
            v_b /= np.linalg.norm(v_b)

            kappa = abs(float(v_b @ v_t))
            proj_std = np.std(Htr @ v_b)
            impact = kappa * np.linalg.norm(w) * proj_std / logit_std

            He = H - np.outer(H @ v_b, v_b)
            dmg = (safe_auc(y[te], prob(H)[te], sw_te)
                   - safe_auc(y[te], prob(He)[te], sw_te))

            lo = hi = np.nan
            if weighted and args.boot:
                st, ps = strata[te], psu[te]
                yte, pH, pHe = y[te], prob(H)[te], prob(He)[te]
                draws = np.empty(args.boot)
                for bi in range(args.boot):
                    k = cluster_boot_idx(st, ps, rng)
                    draws[bi] = (safe_auc(yte[k], pH[k], wt[te][k])
                                 - safe_auc(yte[k], pHe[k], wt[te][k]))
                draws = draws[~np.isnan(draws)]
                if len(draws) > 10:
                    lo, hi = np.percentile(draws, [2.5, 97.5])

            rows.append({"seed": seed, "attr": attr, "label": LABEL[attr],
                         "kappa": kappa, "impact": impact, "auroc_damage": dmg,
                         "dmg_lo": lo, "dmg_hi": hi})
    return pd.DataFrame(rows)


print("computing unweighted (as published) ...")
du = run(weighted=False)
print(f"computing weighted with {args.boot} cluster-bootstrap replicates ...")
dw = run(weighted=True)

W = 72
print()
print("=" * W)
print("1. DOES WEIGHTING CHANGE THE CONCLUSION?")
print("=" * W)
cmp = pd.DataFrame({
    "attribute": [LABEL[a] for a in ATTRS],
    "kappa_unw": [du[du.attr == a].kappa.mean() for a in ATTRS],
    "kappa_wtd": [dw[dw.attr == a].kappa.mean() for a in ATTRS],
    "impact_unw": [du[du.attr == a].impact.mean() for a in ATTRS],
    "impact_wtd": [dw[dw.attr == a].impact.mean() for a in ATTRS],
    "damage_unw": [du[du.attr == a].auroc_damage.mean() for a in ATTRS],
    "damage_wtd": [dw[dw.attr == a].auroc_damage.mean() for a in ATTRS],
}).sort_values("impact_wtd")
print(cmp.round(3).to_string(index=False))
same = list(cmp.sort_values("impact_unw").attribute) == list(cmp.attribute)
print(f"\nattribute ordering by impact is {'UNCHANGED' if same else 'DIFFERENT'} "
      f"under weighting.")
print("That one word is the sentence the reviewers asked for.")

print()
print("=" * W)
print("2. CORRELATION, BOTH FRAMINGS (weighted)")
print("=" * W)
a = dw.groupby("attr")[["kappa", "impact", "auroc_damage"]].mean()
rk2, pk2 = spearmanr(a.kappa, a.auroc_damage)
ri2, pi2 = spearmanr(a.impact, a.auroc_damage)
print(f"  collapsed to {len(a)} attribute means (seeds are re-splits, not new data)")
print(f"    rho(kappa,  damage) = {rk2:6.3f}   p = {pk2:.3f}")
print(f"    rho(impact, damage) = {ri2:6.3f}   p = {pi2:.3f}")
rk, pk = spearmanr(dw.kappa, dw.auroc_damage)
ri, pi = spearmanr(dw.impact, dw.auroc_damage)
print(f"\n  all {len(dw)} rows (5 seeds x 4 attributes)")
print(f"    rho(kappa,  damage) = {rk:6.3f}   p = {pk:.2e}")
print(f"    rho(impact, damage) = {ri:6.3f}   p = {pi:.2e}")
print("\n  Report both. The first is what a careful reviewer will compute.")

print()
print("=" * W)
print("3. SEPARATION AND STABILITY (the arguments that do not depend on n)")
print("=" * W)
for col, nm in [("kappa", "kappa"), ("impact", "impact I"),
                ("auroc_damage", "observed damage")]:
    lo, hi = a[col].min(), a[col].max()
    print(f"  {nm:18s} {lo:7.3f} to {hi:7.3f}   spread {hi/lo:5.1f}x"
          if lo > 0 else f"  {nm:18s} {lo:7.3f} to {hi:7.3f}")
print()
print(f"  {'attribute':22s} {'kappa relSD':>12s} {'impact relSD':>13s}  stability gain")
for attr in ATTRS:
    g = dw[dw.attr == attr]
    rk_ = g.kappa.std(ddof=1) / g.kappa.mean()
    ri_ = g.impact.std(ddof=1) / g.impact.mean()
    print(f"  {LABEL[attr]:22s} {100*rk_:11.0f}% {100*ri_:12.0f}%"
          f"  {rk_/ri_:6.1f}x steadier")

print()
print("=" * W)
print("4. PER-ATTRIBUTE, WEIGHTED, WITH DESIGN-BASED 95% CI ON DAMAGE")
print("=" * W)
for attr in sorted(ATTRS, key=lambda x: dw[dw.attr == x].impact.mean()):
    g = dw[dw.attr == attr]
    print(f"\n  {LABEL[attr]}")
    print(f"    kappa   {g.kappa.mean():.3f} +/- {g.kappa.std(ddof=1):.3f}")
    print(f"    impact  {g.impact.mean():.3f} +/- {g.impact.std(ddof=1):.3f}")
    print(f"    damage  {g.auroc_damage.mean():.3f} +/- {g.auroc_damage.std(ddof=1):.3f}"
          f"   cluster-bootstrap CI [{g.dmg_lo.mean():.3f}, {g.dmg_hi.mean():.3f}]")
    rel = g.auroc_damage.std(ddof=1) / abs(g.auroc_damage.mean())
    if rel > 0.40:
        print(f"    NOTE seed relative SD {100*rel:.0f}% - state this in the text")

dw.to_csv(f"results/impact_gate{TAG}.csv", index=False)
du.to_csv(f"results/impact_gate{TAG}_unweighted.csv", index=False)
print(f"\nwrote results/impact_gate{TAG}.csv and "
      f"results/impact_gate{TAG}_unweighted.csv")
