"""
11_fig_verdict_map.py - the routing verdict map.

CHANGED: thresholds are now 0.2 and 0.1, re-estimated out of sample by
16_routing_threshold_validation.py. The previous 0.3 and 0.15 were the
smallest values in the grid 10_baserate_diag.py sweeps, which is the standard
signal that the grid was too narrow, and all twelve misclassifications under
them erred in the same direction: routing cases to the aggressive projection
edit that should have been translated or refused.

ADDED: the declared deferral band, gap 0.1 +/- 0.02, which defers 10 of 95
cases and leaves no false-safe routings. It is drawn because the paper now
claims it, and a figure that does not show it would contradict the text.
"""
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv("results3/baserate_validation.csv")
I_THR, G_THR = 0.2, 0.1
BAND = 0.02

COLORS = {"meandiff_ok": "#2166AC", "recovered": "#1B7837", "irrecoverable": "#B2182B"}
LABELS = {"meandiff_ok": "mean-difference safe (project)",
          "recovered": "recovered (translate)",
          "irrecoverable": "irrecoverable (abstain)"}
MARK = {"nhanes": "o", "synthetic": "^"}

fig, ax = plt.subplots(figsize=(5.6, 4.0))
XMAX = 1.25

ax.axvspan(0, I_THR, color="#2166AC", alpha=0.06)
ax.axvline(I_THR, ls="--", c="gray", lw=1.1)
ax.plot([I_THR, XMAX], [G_THR, G_THR], ls=":", c="gray", lw=1.1)

# declared deferral band, only where the gap axis is actually consulted
ax.fill_between([I_THR, XMAX], G_THR - BAND, G_THR + BAND,
                color="#7F7F7F", alpha=0.16, lw=0, zorder=0)

for tb in ["synthetic", "nhanes"]:
    for v, c in COLORS.items():
        s = d[(d.testbed == tb) & (d.verdict == v)]
        if s.empty:
            continue
        ax.scatter(s.impact, s.baserate_gap, c=c, marker=MARK[tb], s=42, alpha=0.85,
                   edgecolor="black", linewidth=0.35,
                   label=LABELS[v] if tb == "nhanes" else None)

ax.set_xlabel("impact-weighted coherence $I$ (pre-intervention)")
ax.set_ylabel("outcome base-rate gap")
ax.set_xlim(0, XMAX)
ax.set_ylim(-0.04, 0.60)
ax.text(0.10, 0.55, "project", fontsize=9, color="#2166AC", ha="center")
ax.text(0.62, 0.045, "translate", fontsize=9, color="#1B7837", ha="center")
ax.text(0.90, 0.55, "abstain", fontsize=9, color="#B2182B", ha="center")
ax.text(XMAX - 0.02, G_THR + BAND + 0.012, "defer", fontsize=8, color="#555555",
        ha="right", va="bottom", style="italic")
ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16),
          ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.0)
fig.savefig("results3/fig_verdict_map.pdf", bbox_inches="tight")
fig.savefig("results3/fig_verdict_map.png", dpi=200, bbox_inches="tight")
print(f"thresholds I<{I_THR}, gap<{G_THR}, deferral band +/-{BAND}")
print("wrote results3/fig_verdict_map.pdf and .png")
