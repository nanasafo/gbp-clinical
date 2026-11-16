"""
18_ties_demo.py - why a nonlinear probe score after mean-matching cannot be
read without a random-partition control.

Self-contained: needs no data. It builds a 32-dimensional representation two
ways from the same latent draws, once dense (no exact ties) and once through
a ReLU (a large share of exact zeros, as in the audited models). It then draws
a RANDOM partition with no relationship to anything, applies mean-matching
translation with that partition exactly as 08_families.edit_meanmatch does,
and asks a gradient-boosted probe to recover the partition from the edited
representation.

On the dense representation the probe should sit at chance. On the ReLU
representation it should be near perfect, because the edit turns every exact
zero into one of two group-specific constants. A probe that separates a
meaningless partition is reading the edit, not anything the partition means.

Run:  python 18_ties_demo.py
Deps: numpy, scikit-learn
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

N, D, SEEDS, PREV = 6000, 32, [0, 1, 2, 3, 4], 0.4


def meanmatch(H, g, tr):
    """08_families.edit_meanmatch at severity 1, direction from training data."""
    m1, m0 = H[tr][g[tr]].mean(0), H[tr][~g[tr]].mean(0)
    v = (m1 - m0) / np.linalg.norm(m1 - m0)
    c = (m1 + m0) / 2
    out = H.copy()
    out[g] -= ((m1 - c) @ v) * v
    out[~g] -= ((m0 - c) @ v) * v
    return out


rows = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(N, 8))
    W = rng.normal(size=(8, D)) / np.sqrt(8)
    dense = Z @ W + 0.5 * rng.normal(size=(N, D))
    relu = np.maximum(0.0, dense + 0.28)               # about 40% exact zeros, as audited
    g = np.zeros(N, dtype=bool)
    g[rng.permutation(N)[:int(PREV * N)]] = True       # a meaningless partition
    tr, te = train_test_split(np.arange(N), test_size=0.3, random_state=seed)
    for name, H in (("dense", dense), ("relu", relu)):
        He = meanmatch(H, g, tr)
        res = {"seed": seed, "rep": name, "zeros": float((H == 0).mean())}
        for pname, clf in (("linear", LogisticRegression(max_iter=1000)),
                           ("gbm", HistGradientBoostingClassifier(max_iter=150, random_state=0))):
            clf.fit(He[tr], g[tr])
            res[pname] = roc_auc_score(g[te], clf.predict_proba(He[te])[:, 1])
        rows.append(res)

print("RANDOM PARTITION, RECOVERED AFTER MEAN-MATCHING  (AUROC, chance = 0.5)")
print(f"  {'representation':16s} {'exact zeros':>12s} {'linear probe':>13s} {'boosted probe':>14s}")
for name in ("dense", "relu"):
    r = [x for x in rows if x["rep"] == name]
    z = np.mean([x["zeros"] for x in r]); lin = [x["linear"] for x in r]; gb = [x["gbm"] for x in r]
    print(f"  {name:16s} {100*z:11.1f}% {np.mean(lin):8.3f} ({min(lin):.2f}-{max(lin):.2f})"
          f" {np.mean(gb):8.3f} ({min(gb):.2f}-{max(gb):.2f})")
print("\nmean over five seeds, range in parentheses")
