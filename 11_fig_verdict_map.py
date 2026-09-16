import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv("results3/baserate_validation.csv")
I_THR, G_THR = 0.3, 0.15

COLORS = {"meandiff_ok": "#2166AC", "recovered": "#1B7837", "irrecoverable": "#B2182B"}
LABELS = {"meandiff_ok": "mean-difference safe (project)",
          "recovered": "recovered (translate)",
          "irrecoverable": "irrecoverable (abstain)"}
MARK = {"nhanes": "o", "synthetic": "^"}

fig, ax = plt.subplots(figsize=(5.6, 4.0))
ax.axvspan(0, I_THR, color="#2166AC", alpha=0.06)
ax.axvline(I_THR, ls="--", c="gray", lw=1.1)
ax.plot([I_THR, 1.25], [G_THR, G_THR], ls=":", c="gray", lw=1.1)

# synthetic first so clinical circles sit on top
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
ax.set_xlim(0, 1.25)
ax.set_ylim(-0.04, 0.60)
ax.text(0.15, 0.55, "project", fontsize=9, color="#2166AC", ha="center")
ax.text(0.62, 0.095, "translate", fontsize=9, color="#1B7837", ha="center")
ax.text(0.90, 0.55, "abstain", fontsize=9, color="#B2182B", ha="center")
ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16),
          ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.0)
fig.savefig("results3/fig_verdict_map.pdf", bbox_inches="tight")
fig.savefig("results3/fig_verdict_map.png", dpi=200, bbox_inches="tight")
print("wrote results3/fig_verdict_map.pdf and .png")