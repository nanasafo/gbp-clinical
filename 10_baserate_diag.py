"""
10_baserate_diag.py - the second routing diagnostic: outcome base-rate gap.

Hypothesis: translation (meanmatch) is safe iff the between-group outcome
base-rate gap is small; a large gap means the mean separation IS task signal,
so the case is irrecoverable.

TWO CHANGES FROM THE SUBMITTED VERSION

1. SURVEY-WEIGHTED. The gap is a claim about populations, "diabetes is this
   much more common among Hispanic adults than White adults", not about the
   people who happened to have a complete lab panel. The submitted version
   used a plain sample mean, so the routing rule's two axes were computed on
   different populations: impact survey-weighted, gap not. Both are reported
   here, and the script flags any attribute whose weighted gap crosses the
   routing threshold when the unweighted one did not, because that is the only
   way the correction changes a verdict.

2. PARAMETERISED. --cohort, --tag and a generic outcome column, so it runs on
   the hypertension cohort as well as the diabetes one.

The in-sample threshold sweep the submitted version used to pick its
thresholds is retained below for the record, clearly labelled, but it is NOT
the number to quote. Thresholds are validated out of sample by
16_routing_threshold_validation.py.

Run:  python 10_baserate_diag.py
      python 10_baserate_diag.py --cohort data/nhanes_cohort_htn.parquet --tag _htn
Deps: pandas, numpy, pyarrow
"""
import argparse, glob, os, sys, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--cohort", default="data/nhanes_cohort.parquet")
ap.add_argument("--tag", default="", help="suffix on cases.csv and the output")
ap.add_argument("--gap-thr", type=float, default=0.10,
                help="routing gap threshold, for the crossing check")
ap.add_argument("--out", default="results3")
args = ap.parse_args()

TAG = args.tag
os.makedirs(args.out, exist_ok=True)
W = 74

cases_path = f"{args.out}/cases{TAG}.csv"
if not os.path.exists(cases_path) and os.path.exists(f"{args.out}/cases.csv"):
    cases_path = f"{args.out}/cases.csv"
cases = pd.read_csv(cases_path)

d = pd.read_parquet(args.cohort)
OUTCOME = "outcome" if "outcome" in d.columns else "diabetes"
if OUTCOME not in d.columns:
    sys.exit(f"{args.cohort} has neither 'outcome' nor 'diabetes'")
WCOL = "WT_POOLED" if "WT_POOLED" in d.columns else None
if WCOL is None:
    print("WARNING: no WT_POOLED in the cohort, falling back to unweighted gaps.")
    print("         Rebuild the cohort with 01_nhanes_download_prep.py rev 3+.")

print("=" * W)
print(f"BASE-RATE GAPS  (cohort {args.cohort}, outcome '{OUTCOME}')")
print("=" * W)


def gap(mask_a, mask_b, weighted):
    """Between-group difference in outcome prevalence, group minus reference."""
    ya, yb = d[OUTCOME].values[mask_a], d[OUTCOME].values[mask_b]
    if not weighted or WCOL is None:
        return abs(ya.mean() - yb.mean())
    wa, wb = d[WCOL].values[mask_a], d[WCOL].values[mask_b]
    return abs(np.average(ya, weights=wa) - np.average(yb, weights=wb))


# group and REFERENCE must match the pairs the bias directions are built from
PAIRS = {
    "sex_female": ((d.sex_female == 1).values, (d.sex_female == 0).values),
    "race_black": ((d.race == "black_nh").values, (d.race == "white_nh").values),
    "race_hisp":  ((d.race == "hispanic").values, (d.race == "white_nh").values),
    "age_60plus": ((d.age >= 60).values, (d.age < 45).values),
}

rows = []
for attr, (ga, gb) in PAIRS.items():
    u, wt = gap(ga, gb, False), gap(ga, gb, True)
    rows.append({"attr": attr, "gap_unweighted": u, "gap_weighted": wt,
                 "gap_shift": wt - u, "n_group": int(ga.sum()), "n_ref": int(gb.sum())})
g = pd.DataFrame(rows)

print(f"  {'attribute':14s} {'unweighted':>11s} {'weighted':>10s} {'shift':>8s}"
      f" {'n grp':>7s} {'n ref':>7s}")
for _, r in g.iterrows():
    print(f"  {r.attr:14s} {r.gap_unweighted:11.4f} {r.gap_weighted:10.4f}"
          f" {r.gap_shift:+8.4f} {r.n_group:7d} {r.n_ref:7d}")

print()
print(f"  routing threshold in use: gap < {args.gap_thr}")
crossed = []
for _, r in g.iterrows():
    su, sw = r.gap_unweighted < args.gap_thr, r.gap_weighted < args.gap_thr
    if su != sw:
        crossed.append(r.attr)
        print(f"  CROSSES: {r.attr} moves from "
              f"{'below' if su else 'above'} to {'below' if sw else 'above'} "
              f"the threshold once weighted. Its routing verdict changes.")
if not crossed:
    print("  No attribute crosses the threshold, so weighting does not change any")
    print("  verdict here. Report the weighted gaps anyway: the rule's two axes")
    print("  must be computed on the same population to be defensible.")

# ---- synthetic gaps, unweighted by construction -------------------------
gap_syn = {}
for f in glob.glob("data_syn/case_*.npz"):
    z = np.load(f)
    gs = z["g"].astype(bool); y = z["y"]
    gap_syn[os.path.basename(f)[:-4]] = abs(y[gs].mean() - y[~gs].mean())

gap_w = dict(zip(g.attr, g.gap_weighted))
gap_u = dict(zip(g.attr, g.gap_unweighted))


def gap_for(row, table):
    if row.testbed == "nhanes":
        return table.get(row.case.rsplit("_s", 1)[0], np.nan)
    return gap_syn.get(row.case, np.nan)


cases["baserate_gap"] = cases.apply(lambda r: gap_for(r, gap_w), axis=1)
cases["baserate_gap_unweighted"] = cases.apply(lambda r: gap_for(r, gap_u), axis=1)

# data_syn/ is gitignored like data/. If it is absent, the synthetic gaps would
# silently become NaN and 16_routing_threshold_validation.py would quietly lose
# 75 of 95 cases. Carry forward the gaps from a previous run rather than that.
n_syn = int((cases.testbed == "synthetic").sum())
if n_syn and cases.loc[cases.testbed == "synthetic", "baserate_gap"].isna().all():
    prev = f"{args.out}/baserate_validation{TAG}.csv"
    if os.path.exists(prev):
        old_g = pd.read_csv(prev).set_index("case").baserate_gap
        m = cases.testbed == "synthetic"
        cases.loc[m, "baserate_gap"] = cases.loc[m, "case"].map(old_g)
        cases.loc[m, "baserate_gap_unweighted"] = cases.loc[m, "case"].map(old_g)
        got = int(cases.loc[m, "baserate_gap"].notna().sum())
        print(f"\n  NOTE data_syn/ not found. Carried {got} of {n_syn} synthetic")
        print(f"       gaps forward from {prev}. Regenerate with 07 to recompute.")
    else:
        print(f"\n  WARNING data_syn/ not found and no previous {prev}.")
        print(f"          {n_syn} synthetic cases have no gap, and 16 will drop")
        print(f"          them. Run 07_synthetic_testbed.py before trusting this.")

cases.to_csv(f"{args.out}/baserate_validation{TAG}.csv", index=False)

print()
print("=" * W)
print("GAP BY VERDICT (weighted)")
print("=" * W)
for v, grp in cases.groupby("verdict"):
    print(f"  {v:14s} n={len(grp):3d}  gap={grp.baserate_gap.mean():.3f} "
          f"[{grp.baserate_gap.min():.3f}-{grp.baserate_gap.max():.3f}]")

print()
print("=" * W)
print("IN-SAMPLE SWEEP, FOR THE RECORD ONLY")
print("=" * W)
print("  These thresholds are selected on the same cases they are scored")
print("  against, which is the circularity three reviewers flagged. Do not")
print("  quote this accuracy. Use 16_routing_threshold_validation.py.")
print()
best = (0, None)
for it in [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]:
    for gt in [0.05, 0.10, 0.15, 0.20, 0.25]:
        pred = np.where(cases.impact < it, "meandiff_ok",
                        np.where(cases.baserate_gap < gt, "recovered", "irrecoverable"))
        acc = (pred == cases.verdict).mean()
        if acc > best[0]:
            best = (acc, (it, gt))
print(f"  best in-sample rule: impact<{best[1][0]}, gap<{best[1][1]}, "
      f"accuracy {best[0]:.1%} over {len(cases)} cases")
print(f"  held-out estimate from 16: run it and quote that number instead")

print(f"\nwrote {args.out}/baserate_validation{TAG}.csv "
      f"(carries both weighted and unweighted gaps)")
