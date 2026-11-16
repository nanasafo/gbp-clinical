"""
19_paper_assets.py - build the gate figure and the three LaTeX tables used in
the IEEE Access manuscript from the CSVs the pipeline writes.

Inputs (defaults are the primary, fasting-subsample cohort):
  results/impact_gate_fast.csv          from 06_impact_gate.py --tag _fast
  results/fairness_utility_raw_fast.csv from 03_gbp_audit.py   --tag _fast
  results3/cases.csv                    from 09_routing_eval.py --tag _fast

Outputs (in --out, default build/):
  fig_gate.pdf / fig_gate.png   (a) angle-only coherence vs AUROC damage,
                                (b) impact-weighted coherence vs AUROC damage
  tab_gate.tex                  gate statistics and realized harm by attribute
  tab_fairness.tex              utility and fairness under each intervention
  tab_family.tex                clinical cases each family recovers

Every number is read from the CSVs; nothing is typed in by hand.

Run:  python 19_paper_assets.py
      python 19_paper_assets.py --tag _mec --out build_mec
Deps: pandas, numpy, matplotlib, scipy
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="_fast")
ap.add_argument("--cases", default="results3/cases.csv")
ap.add_argument("--kappa-threshold", type=float, default=0.30,
                help="angle-only gate threshold (GBP-Audit)")
ap.add_argument("--impact-threshold", type=float, default=0.64,
                help="impact gate threshold fitted by 14_threshold_validation.py")
ap.add_argument("--out", default="build")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

ORDER = ["race_hisp", "race_black", "sex_female", "age_60plus"]
LABEL = {"race_hisp": "Hispanic", "race_black": "Black NH",
         "sex_female": "Sex (F)", "age_60plus": r"Age $\geq$ 60"}
PLOT_LABEL = {"race_hisp": "Hispanic", "race_black": "Black NH",
              "sex_female": "Sex (F)", "age_60plus": "Age ≥ 60"}
MARK = {"race_hisp": ("o", "#1b7837"), "race_black": ("s", "#762a83"),
        "sex_female": ("^", "#2166ac"), "age_60plus": ("D", "#b2182b")}

# ---------------------------------------------------------------- gate figure
g = pd.read_csv(f"results/impact_gate{args.tag}.csv")
rk, pk = spearmanr(g.kappa, g.auroc_damage)
ri, pi = spearmanr(g.impact, g.auroc_damage)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, col, thr, name, (r, p) in [
        (axes[0], "kappa", args.kappa_threshold, "angle-only coherence κ", (rk, pk)),
        (axes[1], "impact", args.impact_threshold, "impact-weighted coherence I", (ri, pi))]:
    for a in ORDER:
        s = g[g.attr == a]
        m, c = MARK[a]
        ax.errorbar(s[col], s.auroc_damage,
                    yerr=[s.auroc_damage - s.dmg_lo, s.dmg_hi - s.auroc_damage],
                    fmt=m, color=c, ms=5, lw=0.8, capsize=2, label=PLOT_LABEL[a])
    ax.axvline(thr, color="0.35", ls="--", lw=1)
    ax.set_xlabel(name)
    ax.text(0.03, 0.95, f"ρ = {r:.2f}  (p = {p:.2g})", transform=ax.transAxes,
            va="top", fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
axes[0].set_ylabel("AUROC damage from full erasure")
axes[0].set_title("(a)", fontsize=9, loc="left")
axes[1].set_title("(b)", fontsize=9, loc="left")
axes[1].legend(fontsize=7, loc="center left", frameon=False)
fig.tight_layout()
fig.savefig(f"{args.out}/fig_gate.pdf", bbox_inches="tight")
fig.savefig(f"{args.out}/fig_gate.png", bbox_inches="tight", dpi=200)
plt.close(fig)

# ------------------------------------------------------------- gate table
def cell(s, col, fmt):
    return f"{s[col].mean():{fmt}} ({s[col].min():{fmt}}-{s[col].max():{fmt}})"

lines = [r"\begin{tabular}{lccc}", r"\toprule",
         r"Attribute & $\kappa$ & Impact $I$ & AUROC damage \\", r"\midrule"]
for a in ORDER:
    s = g[g.attr == a]
    k = f"{s.kappa.mean():.3f} ({s.kappa.min():.2f}-{s.kappa.max():.2f})"
    i = f"{s.impact.mean():.2f} ({s.impact.min():.2f}-{s.impact.max():.2f})"
    d = cell(s, "auroc_damage", ".3f")
    lines.append(f"{LABEL[a]} & {k} & {i} & {d} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{args.out}/tab_gate.tex", "w").write("\n".join(lines) + "\n")

# --------------------------------------------------------- fairness table
f = pd.read_csv(f"results/fairness_utility_raw{args.tag}.csv")
f = f[f.method.isin(["unedited", "erase_ungated", "inlp", "leace"])]
mean = f.groupby(["attr", "method"])[["auroc", "ece", "dp_gap", "eo_gap"]].mean()
NAME = {"unedited": "unedited", "erase_ungated": "erasure", "inlp": "INLP", "leace": "LEACE"}
lines = [r"\begin{tabular}{llcccc}", r"\toprule",
         r"Attribute & Method & AUROC & ECE & DP gap & EO gap \\"]
for a in ["age_60plus", "sex_female", "race_black", "race_hisp"]:
    lines.append(r"\midrule")
    for j, mth in enumerate(["unedited", "erase_ungated", "inlp", "leace"]):
        r = mean.loc[(a, mth)]
        lead = rf"\multirow{{4}}{{*}}{{{LABEL[a]}}}" if j == 0 else ""
        lines.append(f"{lead} & {NAME[mth]} & {r.auroc:.3f} & {r.ece:.3f} & "
                     f"{r.dp_gap:.3f} & {r.eo_gap:.3f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{args.out}/tab_fairness.tex", "w").write("\n".join(lines) + "\n")

# ----------------------------------------------------------- family table
c = pd.read_csv(args.cases)
c = c[c.testbed == "nhanes"].copy()
c["attr"] = c.case.str.replace(r"_s\d+$", "", regex=True)
FAM = [("s_meandiff", "mean-diff"), ("s_midpoint", "midpoint"),
       ("s_meanmatch", "mean-match"), ("s_inlp2", "INLP-2"), ("s_leace", "LEACE")]
lines = [r"\begin{tabular}{lccccc}", r"\toprule",
         "Attribute & " + " & ".join(n for _, n in FAM) + r" \\", r"\midrule"]
for a in ["sex_female", "race_black", "race_hisp", "age_60plus"]:
    s = c[c.attr == a]
    lines.append(f"{LABEL[a]} & " + " & ".join(
        f"{int(s[col].notna().sum())}/{len(s)}" for col, _ in FAM) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{args.out}/tab_family.tex", "w").write("\n".join(lines) + "\n")

print(f"Spearman(kappa, damage)  = {rk:.2f} (p = {pk:.2g})")
print(f"Spearman(impact, damage) = {ri:.2f} (p = {pi:.2g})")
print(f"wrote {args.out}/fig_gate.pdf, fig_gate.png, tab_gate.tex, "
      f"tab_fairness.tex, tab_family.tex")
