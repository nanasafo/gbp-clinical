"""
07_synthetic_testbed.py — latent testbed with GROUND-TRUTH entanglement control.

Generates cases over a grid of (rho_ent, delta, seed):
  rho_ent : correlation between the attribute factor and the task factor
            (high rho_ent -> truly entangled -> should be IRRECOVERABLE)
  delta   : one-sided offset of the attribute encoding (asymmetry)
            (high delta at low rho_ent -> meandiff fails but centered
             families should RECOVER)
For each case: train a linear head on h, save activations in the same .npz
format as the NHANES pipeline so 08/09 run unchanged.

Run: python 07_synthetic_testbed.py
Out: data_syn/case_r{rho}_d{delta}_s{seed}.npz + data_syn/manifest.csv
"""
import os, itertools, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

os.makedirs("data_syn", exist_ok=True)
D, N = 32, 6000
RHOS   = [0.0, 0.2, 0.4, 0.6, 0.8]
DELTAS = [0.0, 1.0, 2.0]
SEEDS  = [0, 1, 2, 3, 4]

rows = []
for rho, delta, seed in itertools.product(RHOS, DELTAS, SEEDS):
    rng = np.random.default_rng(1000*seed + int(100*rho) + int(10*delta))
    # latent factors
    z_task = rng.normal(size=N)
    z_attr = rho * z_task + np.sqrt(max(1e-9, 1 - rho**2)) * rng.normal(size=N)
    g = (z_attr > 0).astype(int)                       # binary attribute
    y = (z_task + 0.5*rng.normal(size=N) > 0).astype(np.float32)  # task label

    # embed into D dims: orthonormal-ish directions u_task, u_attr
    u_task = rng.normal(size=D); u_task /= np.linalg.norm(u_task)
    u_raw  = rng.normal(size=D); u_raw -= (u_raw @ u_task) * u_task
    u_attr = u_raw / np.linalg.norm(u_raw)

    H = (np.outer(z_task, u_task) + 1.5*np.outer(z_attr, u_attr)
         + 0.8*rng.normal(size=(N, D)))
    # ONE-SIDED encoding: attribute group 1 additionally offset along u_attr
    H += delta * np.outer(g, u_attr)
    H = H.astype(np.float32)

    idx = np.arange(N)
    tr, tmp = train_test_split(idx, test_size=0.4, random_state=seed, stratify=y)
    va, te  = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=y[tmp])

    head = LogisticRegression(max_iter=1000).fit(H[tr], y[tr])
    w, b = head.coef_.squeeze().astype(np.float32), float(head.intercept_[0])

    name = f"case_r{int(rho*10)}_d{int(delta)}_s{seed}"
    np.savez(f"data_syn/{name}.npz", H=H, y=y, g=g,
             idx_train=tr, idx_val=va, idx_test=te, w_task=w, b_task=b)
    # ground-truth label for routing validation (design-based):
    truth = "irrecoverable" if rho >= 0.6 else ("recoverable_by_centering" if (delta >= 1.0 and rho <= 0.2) else "gray")
    rows.append({"case": name, "rho_ent": rho, "delta": delta, "seed": seed, "truth": truth})

pd.DataFrame(rows).to_csv("data_syn/manifest.csv", index=False)
print(f"wrote {len(rows)} synthetic cases + manifest.csv")
