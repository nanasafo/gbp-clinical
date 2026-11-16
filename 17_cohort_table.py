"""
17_cohort_table.py - survey-weighted cohort characteristics by race/ethnicity,
for the merged paper's cohort table.

Every cell is a weighted estimate using WT_POOLED. Triglycerides are reported
as a weighted geometric mean (exp of the weighted mean of log_tg), because the
distribution is right-skewed and the model uses the log. Proportions are
percentages. n is the unweighted count, which is what a reader needs to judge
precision.

Run:  python 17_cohort_table.py
      python 17_cohort_table.py --cohort data/nhanes_cohort_noTG.parquet --tag _mec
Out:  results/cohort_table{tag}.csv  and  results/cohort_table{tag}.tex
Deps: pandas, numpy, pyarrow
"""
import argparse, os, sys, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--cohort", default="data/nhanes_cohort.parquet")
ap.add_argument("--tag", default="")
ap.add_argument("--out", default="results")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

d = pd.read_parquet(args.cohort)
if "WT_POOLED" not in d.columns:
    sys.exit("no WT_POOLED in the cohort; rebuild it with 01_nhanes_download_prep.py rev 3+")
OUTCOME = "outcome" if "outcome" in d.columns else "diabetes"

GROUPS = [("white_nh", "White NH"), ("hispanic", "Hispanic"), ("black_nh", "Black NH"),
          ("asian_nh", "Asian NH"), ("other", "Other"), (None, "All")]
# (label, column, kind, decimals)
ROWS = [("Age (yr)", "age", "mean", 1),
        ("Female (\\%)", "sex_female", "pct", 1),
        ("BMI (kg/m$^2$)", "BMXBMI", "mean", 1),
        ("Waist (cm)", "BMXWAIST", "mean", 1),
        ("SBP (mmHg)", "SBP", "mean", 1),
        ("Total chol.\\ (mg/dL)", "LBXTC", "mean", 1),
        ("HDL (mg/dL)", "LBDHDD", "mean", 1),
        ("Triglycerides (mg/dL)$^{a}$", "log_tg", "geo", 1),
        ("Current smoker (\\%)", "smoker_current", "pct", 1),
        ("Diabetes (\\%)", OUTCOME, "pct", 1)]
ROWS = [r for r in ROWS if r[1] in d.columns]


def cell(g, col, kind):
    x, w = g[col].values.astype(float), g["WT_POOLED"].values.astype(float)
    ok = ~np.isnan(x)
    m = np.average(x[ok], weights=w[ok])
    return 100 * m if kind == "pct" else (np.exp(m) if kind == "geo" else m)


table = {}
for code, label in GROUPS:
    g = d if code is None else d[d.race == code]
    if len(g) == 0:
        continue
    col = {"n": len(g)}
    for lab, c, kind, _ in ROWS:
        col[lab] = cell(g, c, kind)
    table[label] = col
t = pd.DataFrame(table)
t.to_csv(f"{args.out}/cohort_table{args.tag}.csv")

labels = list(table)
lines = ["Characteristic & " + " & ".join(labels) + " \\\\", "\\midrule",
         "$n$ & " + " & ".join(f"{int(table[l]['n']):,}".replace(",", "{,}") for l in labels) + " \\\\"]
for lab, c, kind, dec in ROWS:
    lines.append(f"{lab} & " + " & ".join(f"{table[l][lab]:.{dec}f}" for l in labels) + " \\\\")
open(f"{args.out}/cohort_table{args.tag}.tex", "w").write("\n".join(lines) + "\n")

pd.set_option("display.width", 160)
print(f"SURVEY-WEIGHTED COHORT CHARACTERISTICS  ({args.cohort}, N={len(d)})")
print(t.rename(index=lambda s: s.replace("\\%", "%").replace("\\ ", " ").replace("$^{a}$", "").replace("$^2$", "2")).round(1).to_string())
print(f"\nwrote {args.out}/cohort_table{args.tag}.csv and .tex")
