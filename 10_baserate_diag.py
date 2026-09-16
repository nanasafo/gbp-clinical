"""
10_baserate_diag.py — validate the second diagnostic: outcome base-rate gap.
Hypothesis: translation (meanmatch) is safe iff the between-group outcome
base-rate gap is small; large gap => the mean separation IS task signal =>
irrecoverable. Computes the gap for every case (NHANES + synthetic), merges
with routing verdicts, reports the separation and a simple threshold.

Run AFTER 09:  python 10_baserate_diag.py
Out: results3/baserate_validation.csv + printed threshold analysis + Fig 3 data
"""
import glob, os, numpy as np, pandas as pd

cases = pd.read_csv("results3/cases.csv")

# ---- NHANES gaps ----
d = pd.read_parquet("data/nhanes_cohort.parquet")
gap_nh = {
    "sex_female": abs(d[d.sex_female==1].diabetes.mean() - d[d.sex_female==0].diabetes.mean()),
    "race_black": abs(d[d.race=="black_nh"].diabetes.mean() - d[d.race=="white_nh"].diabetes.mean()),
    "race_hisp":  abs(d[d.race=="hispanic"].diabetes.mean() - d[d.race=="white_nh"].diabetes.mean()),
    "age_60plus": abs(d[d.age>=60].diabetes.mean() - d[d.age<45].diabetes.mean()),
}

# ---- synthetic gaps (per case, from saved labels) ----
gap_syn = {}
for f in glob.glob("data_syn/case_*.npz"):
    z = np.load(f); g = z["g"].astype(bool); y = z["y"]
    gap_syn[os.path.basename(f)[:-4]] = abs(y[g].mean() - y[~g].mean())

def gap_for(row):
    if row.testbed == "nhanes":
        return gap_nh[row.case.rsplit("_s", 1)[0]]
    return gap_syn[row.case]

cases["baserate_gap"] = cases.apply(gap_for, axis=1)
cases.to_csv("results3/baserate_validation.csv", index=False)

# ---- separation analysis ----
print("Base-rate gap by verdict (mean [min-max]):")
for v, grp in cases.groupby("verdict"):
    print(f"  {v:14s} n={len(grp):3d}  gap={grp.baserate_gap.mean():.3f} "
          f"[{grp.baserate_gap.min():.3f}-{grp.baserate_gap.max():.3f}]")

# threshold sweep: how well does (impact, gap) 2x2 predict the verdict?
print("\n2x2 routing-rule accuracy sweep (impact thr x gap thr):")
best = (0, None)
for it in [0.3, 0.4, 0.5, 0.6, 0.7]:
    for gt in [0.10, 0.15, 0.20, 0.25]:
        pred = np.where(cases.impact < it, "meandiff_ok",
                np.where(cases.baserate_gap < gt, "recovered", "irrecoverable"))
        acc = (pred == cases.verdict).mean()
        if acc > best[0]: best = (acc, (it, gt))
        print(f"  impact<{it:.1f}, gap<{gt:.2f}: acc={acc:.2%}")
print(f"\nBest 2x2 rule: impact thr={best[1][0]}, gap thr={best[1][1]} -> accuracy {best[0]:.2%} over {len(cases)} cases")

# per-testbed accuracy at the best rule
it, gt = best[1]
pred = np.where(cases.impact < it, "meandiff_ok",
        np.where(cases.baserate_gap < gt, "recovered", "irrecoverable"))
cases["pred_2x2"] = pred
for tb, grp in cases.groupby("testbed"):
    print(f"  {tb}: {(grp.pred_2x2==grp.verdict).mean():.2%} ({(grp.pred_2x2==grp.verdict).sum()}/{len(grp)})")
print("\nwrote results3/baserate_validation.csv")
