"""
13_nonlinear_probe.py - after erasure, is the attribute actually gone?

THE QUESTION THIS ANSWERS
Your framework estimates the bias direction as a mean difference, a linear
object, and erases it by projecting that one direction out. The gate then
reports how much utility that cost. But "we projected out the direction" and
"the attribute is no longer recoverable" are different claims, and only the
first is demonstrated so far.

If a nonlinear probe still recovers the attribute from the erased
representation, then what you measured is the cost of LINEAR erasure. That is
a narrower claim than either manuscript currently makes, and a reviewer will
ask for exactly this experiment.

METHOD
For each seed and attribute, the bias direction is fit on TRAIN only, erasure
is applied to all rows, probes are trained on erased TRAIN rows and scored on
erased TEST rows. Fitting the probe on the rows that produced the direction
would overstate how well erasure worked.

Three probe families, on the original and the erased representation:
  linear  logistic regression   - is the attribute linearly present at all
  mlp     MLPClassifier(64,32)  - the usual nonlinear probe
  gbm     HistGradientBoosting  - a different nonlinear family, because one
                                  architecture failing to find something is
                                  weak evidence that it is not there
Probes train unweighted, deliberately: the question is whether the information
is recoverable AT ALL, so the probe should be as strong as possible. Reported
AUROC is survey-weighted, with a stratified cluster bootstrap CI.

READING THE RESULT
  linear ~0.5 and both nonlinear ~0.5   erasure is genuine. Strongest version
                                        of the paper. Lead with it.
  linear ~0.5, nonlinear well above     linear erasure only. The gate still
                                        works, the result is still publishable,
                                        but every claim gets scoped to linear
                                        recoverability. Abstract, section
                                        headings and conclusion all change.
  both well above 0.5                   erasure is not working. Fix that first.

The middle outcome is the likely one and it is survivable: LEACE and the
concept-erasure literature you already cite guarantee linear guardedness and
claim nothing more, so scoping to linear puts you in line with the standard.
The danger is submitting without this experiment and being asked for it during
review.

Run:  python 13_nonlinear_probe.py --tag _fast
      python 13_nonlinear_probe.py --tag _mec --boot 500
Time: roughly 5 to 10 minutes; 40 MLP fits and 40 boosting fits.
Deps: pip install scikit-learn pandas numpy pyarrow
"""
import argparse, warnings, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="_fast")
ap.add_argument("--boot", type=int, default=300)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--chance", type=float, default=0.55,
                help="AUROC at or below this counts as not recoverable")
args = ap.parse_args()

TAG = args.tag
SEEDS = [0, 1, 2, 3, 4]
rng = np.random.default_rng(args.seed)
W = 76

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


def probes():
    return {
        "linear": LogisticRegression(max_iter=2000),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                             early_stopping=True, n_iter_no_change=10,
                             random_state=0),
        "gbm": HistGradientBoostingClassifier(max_iter=150, random_state=0),
    }


def cluster_idx(st, ps, rng):
    out = []
    for s in np.unique(st):
        rows = np.where(st == s)[0]
        u = np.unique(ps[rows])
        for p in rng.choice(u, size=len(u), replace=True):
            out.append(rows[ps[rows] == p])
    return np.concatenate(out) if out else np.arange(len(st))


def boot_ci(yy, pp, ww, st, ps, rng, n):
    d = np.empty(n)
    for b in range(n):
        k = cluster_idx(st, ps, rng)
        if len(np.unique(yy[k])) < 2:
            d[b] = np.nan; continue
        try:
            d[b] = roc_auc_score(yy[k], pp[k], sample_weight=ww[k])
        except ValueError:
            d[b] = np.nan
    d = d[~np.isnan(d)]
    return (np.percentile(d, 2.5), np.percentile(d, 97.5)) if len(d) > 10 else (np.nan, np.nan)


rows = []
for seed in SEEDS:
    z = np.load(f"data/activations_seed{seed}{TAG}.npz")
    for k in ("H", "y", "idx_train", "idx_test", "wt", "strata", "psu"):
        if k not in z:
            raise RuntimeError(f"activations_seed{seed}{TAG}.npz has no '{k}'. "
                               f"Re-run 02_train_models.py (Step 2 version).")
    H, tr, te = z["H"], z["idx_train"], z["idx_test"]
    wt, st, ps = z["wt"], z["strata"], z["psu"]

    for attr, g in ATTRS.items():
        ref = REF[attr] if REF[attr] is not None else ~g
        # bias direction from TRAIN only, weighted group means
        gtr, rtr = g[tr], ref[tr]
        mu_g = np.average(H[tr][gtr], axis=0, weights=wt[tr][gtr])
        mu_r = np.average(H[tr][rtr], axis=0, weights=wt[tr][rtr])
        v_b = mu_g - mu_r
        v_b /= np.linalg.norm(v_b)
        He = H - np.outer(H @ v_b, v_b)

        # the probe target: group vs reference only, not group vs everyone
        keep = g | ref
        tr_k = tr[keep[tr]]
        te_k = te[keep[te]]
        target = g.astype(int)
        if len(np.unique(target[te_k])) < 2:
            continue

        for rep_name, M in [("H", H), ("H_erased", He)]:
            for pname, clf in probes().items():
                clf.fit(M[tr_k], target[tr_k])
                pp = clf.predict_proba(M[te_k])[:, 1]
                auc = roc_auc_score(target[te_k], pp, sample_weight=wt[te_k])
                lo, hi = (np.nan, np.nan)
                if rep_name == "H_erased" and args.boot:
                    lo, hi = boot_ci(target[te_k], pp, wt[te_k],
                                     st[te_k], ps[te_k], rng, args.boot)
                rows.append({"seed": seed, "attr": attr, "label": LABEL[attr],
                             "rep": rep_name, "probe": pname, "auroc": auc,
                             "lo": lo, "hi": hi})
        print(f"  seed {seed} {LABEL[attr]:20s} done")

d = pd.DataFrame(rows)

print()
print("=" * W)
print(f"RECOVERABILITY OF THE ATTRIBUTE AFTER ERASURE  (cohort {TAG or '(none)'})")
print("=" * W)
print("  weighted AUROC of a probe predicting the attribute; 0.50 is chance")
print()
print(f"  {'attribute':20s} {'probe':7s} {'from H':>14s} {'from H erased':>16s} {'drop':>8s}")
for attr in ATTRS:
    for pname in ["linear", "mlp", "gbm"]:
        a = d[(d.attr == attr) & (d.probe == pname) & (d.rep == "H")].auroc
        e = d[(d.attr == attr) & (d.probe == pname) & (d.rep == "H_erased")].auroc
        if not len(a):
            continue
        lab = LABEL[attr] if pname == "linear" else ""
        print(f"  {lab:20s} {pname:7s} {a.mean():7.3f} +/-{a.std(ddof=1):5.3f}"
              f" {e.mean():9.3f} +/-{e.std(ddof=1):5.3f} {a.mean()-e.mean():8.3f}")
    print()

print("=" * W)
print("VERDICT")
print("=" * W)
er = d[d.rep == "H_erased"]
lin = er[er.probe == "linear"].auroc.mean()
non = er[er.probe.isin(["mlp", "gbm"])].groupby("probe").auroc.mean()
print(f"  linear probe on erased representation : {lin:.3f}")
for p, v in non.items():
    print(f"  {p:5s}  probe on erased representation : {v:.3f}")
best_non = non.max()
print()
if lin <= args.chance and best_non <= args.chance:
    print("  ERASURE IS GENUINE. No probe family recovers the attribute above")
    print("  chance. This is the strongest version of the paper: lead with it,")
    print("  and the claim 'the attribute is removed' is earned.")
elif lin <= args.chance < best_non:
    print("  LINEAR ERASURE ONLY. The linear probe is at chance but a nonlinear")
    print("  probe still recovers the attribute, so what the gate measures is")
    print("  the cost of LINEAR erasure. Scope every claim accordingly: the")
    print("  abstract, the section headings and the conclusion all change from")
    print("  'removing the attribute' to 'linearly removing the attribute'.")
    print("  This is the standard that LEACE and the concept-erasure literature")
    print("  hold themselves to, so you are in line with the field, not behind")
    print("  it. Report it yourself rather than having a reviewer find it.")
else:
    print("  ERASURE IS NOT WORKING. Even the linear probe recovers the")
    print("  attribute from the erased representation. Check the projection")
    print("  before interpreting anything else in the pipeline.")

print()
print("  per-attribute, strongest nonlinear probe on the erased representation,")
print("  with design-based CI (an interval excluding 0.50 means the attribute")
print("  is still there):")
for attr in ATTRS:
    e = er[(er.attr == attr) & (er.probe.isin(["mlp", "gbm"]))]
    if not len(e):
        continue
    best = e.groupby("probe").auroc.mean().idxmax()
    b = e[e.probe == best]
    flag = "" if (b.lo.mean() <= 0.5 <= b.hi.mean()) else "   *above chance"
    print(f"    {LABEL[attr]:20s} {best:4s} {b.auroc.mean():.3f}"
          f"  CI [{b.lo.mean():.3f}, {b.hi.mean():.3f}]{flag}")

d.to_csv(f"results/nonlinear_probe{TAG}.csv", index=False)
print(f"\nwrote results/nonlinear_probe{TAG}.csv")
