"""
00_reanalysis.py - rebuild the Table III statistics with variance and both
correlation framings, and emit paste-ready LaTeX.

Answers these reviewer points without touching the pipeline:
  - "statistical reporting in Table IV ... means over 5 seeds without variance"
  - "subgroup sample sizes ... report confidence intervals"
  - the effective-n problem: 20 rows are 5 re-splits of one cohort crossed with
    4 attributes, so seeds are not independent observations. We report the
    correlation under both framings rather than letting a reviewer find it.

Run:  python 00_reanalysis.py --infile results/impact_gate.csv
Deps: pip install pandas numpy scipy
"""
import argparse, numpy as np, pandas as pd
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--infile", default="results/impact_gate.csv")
ap.add_argument("--outdir", default="results")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
d = pd.read_csv(args.infile)
ATTR_LABEL = {"race_hisp": "Hispanic", "race_black": "Black non-Hispanic",
              "sex_female": "Sex (female)", "age_60plus": r"Age $\geq$ 60"}

print("=" * 70)
print("1. CORRELATION UNDER BOTH FRAMINGS")
print("=" * 70)

rk, pk = spearmanr(d.kappa, d.auroc_damage)
ri, pi = spearmanr(d.impact, d.auroc_damage)
n_rows = len(d)
print(f"  all {n_rows} rows ({d.seed.nunique()} seeds x {d.attr.nunique()} attributes)")
print(f"    rho(kappa,  damage) = {rk:6.3f}   p = {pk:.2e}")
print(f"    rho(impact, damage) = {ri:6.3f}   p = {pi:.2e}")

a = d.groupby("attr")[["kappa", "impact", "auroc_damage"]].mean()
rk2, pk2 = spearmanr(a.kappa, a.auroc_damage)
ri2, pi2 = spearmanr(a.impact, a.auroc_damage)
print(f"\n  collapsed to {len(a)} attribute means (seeds are re-splits, not new data)")
print(f"    rho(kappa,  damage) = {rk2:6.3f}   p = {pk2:.3f}")
print(f"    rho(impact, damage) = {ri2:6.3f}   p = {pi2:.3f}")
print("\n  Report both. The second framing is what a careful reviewer will compute.")

print()
print("=" * 70)
print("2. SEPARATION: the argument that does not depend on n")
print("=" * 70)
for col, name in [("kappa", "kappa"), ("impact", "impact I"),
                  ("auroc_damage", "observed damage")]:
    lo, hi = a[col].min(), a[col].max()
    print(f"  {name:18s} {lo:.3f} to {hi:.3f}   spread {hi/lo:5.1f}x")
print("\n  kappa barely moves while the harm it gates moves by a factor of 30.")
print("  This holds at any n and under any framing of the seeds.")

print()
print("=" * 70)
print("3. PER-ATTRIBUTE STATISTICS WITH BOOTSTRAP CIs")
print("=" * 70)


def boot_ci(x, n_boot, rng, alpha=0.05):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan, np.nan
    draws = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])


rows = []
for attr, g in d.groupby("attr"):
    rec = {"attr": attr, "label": ATTR_LABEL.get(attr, attr), "n_seeds": len(g)}
    for col in ["kappa", "impact", "auroc_damage"]:
        lo, hi = boot_ci(g[col], args.boot, rng)
        rec[f"{col}_mean"] = g[col].mean()
        rec[f"{col}_sd"] = g[col].std(ddof=1)
        rec[f"{col}_lo"] = lo
        rec[f"{col}_hi"] = hi
        rec[f"{col}_relsd"] = g[col].std(ddof=1) / g[col].mean() if g[col].mean() else np.nan
    rows.append(rec)

res = pd.DataFrame(rows).sort_values("impact_mean").reset_index(drop=True)
for _, r in res.iterrows():
    print(f"\n  {r.label}  (n_seeds={r.n_seeds})")
    for col, nm in [("kappa", "kappa "), ("impact", "impact"), ("auroc_damage", "damage")]:
        print(f"    {nm}  {r[f'{col}_mean']:.3f} +/- {r[f'{col}_sd']:.3f}"
              f"   95% CI [{r[f'{col}_lo']:.3f}, {r[f'{col}_hi']:.3f}]"
              f"   rel.SD {100*r[f'{col}_relsd']:.0f}%")

unstable = res[res.auroc_damage_relsd > 0.40]
if len(unstable):
    print("\n  SEED INSTABILITY worth stating in the text (relative SD > 40%):")
    for _, r in unstable.iterrows():
        print(f"    {r.label}: damage {r.auroc_damage_mean:.3f} "
              f"+/- {r.auroc_damage_sd:.3f} ({100*r.auroc_damage_relsd:.0f}%)")

res.to_csv(f"{args.outdir}/table3_with_variance.csv", index=False)

print()
print("=" * 70)
print("4. LATEX FOR TABLE III")
print("=" * 70)
tex = [
    r"\begin{table}[t]", r"\centering",
    r"\caption{Per-attribute gate statistics and observed harm. "
    r"Mean $\pm$ SD over 5 seeds; 95\% CIs from "
    f"{args.boot} bootstrap resamples." + r"}",
    r"\label{tab:impact}",
    r"\begin{tabular}{lccc}", r"\hline",
    r"Attribute & $\kappa$ & $I$ & AUROC damage \\", r"\hline",
]
for _, r in res.iterrows():
    tex.append(
        f"{r.label} & "
        f"${r.kappa_mean:.2f} \\pm {r.kappa_sd:.2f}$ & "
        f"${r.impact_mean:.2f} \\pm {r.impact_sd:.2f}$ & "
        f"${r.auroc_damage_mean:.3f} \\pm {r.auroc_damage_sd:.3f}$ \\\\"
    )
tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
tex = "\n".join(tex)
print(tex)
with open(f"{args.outdir}/table3.tex", "w") as f:
    f.write(tex + "\n")

print(f"\nwrote {args.outdir}/table3_with_variance.csv and {args.outdir}/table3.tex")
