"""05_fig_dose_response.py — Figure 2: kappa vs impact as harm forecasts.
Panel (a): angle-only kappa vs AUROC damage (weak).
Panel (b): impact-weighted coherence vs AUROC damage (near-perfect).
Reads results/impact_gate.csv (from 06_impact_gate.py).
"""
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

d = pd.read_csv("results/impact_gate.csv")

rk, pk = spearmanr(d.kappa,  d.auroc_damage)
ri, pi = spearmanr(d.impact, d.auroc_damage)
print(d.sort_values("impact").to_string(index=False))
print(f"\nSpearman(kappa,  damage) = {rk:.3f} (p={pk:.1e})")
print(f"Spearman(impact, damage) = {ri:.3f} (p={pi:.1e})")

MARKERS = {"race_hisp": "o", "race_black": "s", "sex_female": "D", "age_60plus": "^"}
LABELS  = {"race_hisp": "Hispanic", "race_black": "Black (NH)",
           "sex_female": "Sex (female)", "age_60plus": "Age 60+"}

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)

for ax, xcol, title, rho, thr in [
    (axes[0], "kappa",  "(a) Angle-only coherence $\\kappa$",       rk, (0.30, 0.50)),
    (axes[1], "impact", "(b) Impact-weighted coherence (ours)",     ri, (0.60, 0.90)),
]:
    for attr, m in MARKERS.items():
        s = d[d.attr == attr]
        ax.scatter(s[xcol], s.auroc_damage, marker=m, s=46, label=LABELS[attr],
                   edgecolor="black", linewidth=0.4)
    ax.axvline(thr[0], ls="--", c="gray", lw=1)
    ax.axvline(thr[1], ls=":",  c="gray", lw=1)
    ax.set_title(f"{title}\nSpearman $\\rho$ = {rho:.2f}", fontsize=10)
    ax.set_xlabel(f"{'$\\kappa$' if xcol=='kappa' else 'impact score'} (pre-intervention)")

axes[0].set_ylabel("AUROC damage from ungated erasure")
axes[1].legend(fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig("results/fig2_dose_response.pdf", bbox_inches="tight")
fig.savefig("results/fig2_dose_response.png", dpi=200, bbox_inches="tight")
print("\nwrote results/fig2_dose_response.pdf and .png")