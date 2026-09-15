"""
03_gbp_audit.py — the paper's core experiment.
For each (seed, attribute): compute bias direction in penultimate space,
coherence kappa vs task direction, apply the gated policy, compare against
ungated erasure / INLP / LEACE. Emit the paper's Tables 2-3 and Fig 2 data.

Run:  python 03_gbp_audit.py
Deps: pip install torch scikit-learn pandas concept-erasure
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, os
from sklearn.metrics import roc_auc_score
from concept_erasure import LeaceEraser

SEEDS = [0,1,2,3,4]
K_FULL, K_PROTECT = 0.30, 0.50            # gate thresholds (from GBP-Audit)
os.makedirs("results", exist_ok=True)

groups = pd.read_parquet("data/groups.parquet")
ATTRS = {
    "sex_female":  (groups.sex_female.values == 1),
    "race_black":  (groups.race.values == "black_nh"),
    "race_hisp":   (groups.race.values == "hispanic"),
    "age_60plus":  (groups.age.values >= 60),
}
ATTR_REF = {   # comparison group masks (None = complement)
    "sex_female": None, "age_60plus": (groups.age.values < 45),
    "race_black": (groups.race.values == "white_nh"),
    "race_hisp":  (groups.race.values == "white_nh"),
}

def ece(y, p, bins=10):
    e, edges = 0.0, np.linspace(0,1,bins+1)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i+1])
        if m.sum(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e

def head_prob(H, w, b):
    return 1/(1+np.exp(-(H @ w + b)))

def metrics(y, p, g_mask, ref_mask):
    yhat = (p >= 0.5)
    dp  = abs(yhat[g_mask].mean() - yhat[ref_mask].mean())
    tpr = lambda m: yhat[m & (y==1)].mean() if (m & (y==1)).any() else np.nan
    fpr = lambda m: yhat[m & (y==0)].mean() if (m & (y==0)).any() else np.nan
    eo  = max(abs(tpr(g_mask)-tpr(ref_mask)), abs(fpr(g_mask)-fpr(ref_mask)))
    return {"auroc": roc_auc_score(y, p), "ece": ece(y, p),
            "dp_gap": dp, "eo_gap": eo,
            "auroc_g": roc_auc_score(y[g_mask], p[g_mask]),
            "ece_g": ece(y[g_mask], p[g_mask])}

rows, kappa_rows = [], []
for seed in SEEDS:
    z = np.load(f"data/activations_seed{seed}.npz")
    H, y, te = z["H"], z["y"], z["idx_test"]
    w = z["w_task"]; b = 0.0  # bias absorbed; refit head bias below
    # refit intercept on train to keep probabilities calibrated
    from sklearn.linear_model import LogisticRegression
    lr_head = LogisticRegression(max_iter=1000).fit(H[z["idx_train"]], y[z["idx_train"]])
    w, b = lr_head.coef_.squeeze(), lr_head.intercept_[0]
    v_task = w / np.linalg.norm(w)

    for attr, g_mask in ATTRS.items():
        ref = ATTR_REF[attr] if ATTR_REF[attr] is not None else ~g_mask
        tr = z["idx_train"]
        v_bias = H[tr][g_mask[tr]].mean(0) - H[tr][ref[tr]].mean(0)
        v_bias /= np.linalg.norm(v_bias)
        kappa = abs(float(v_bias @ v_task))
        policy = "FULL" if kappa < K_FULL else ("PARTIAL" if kappa <= K_PROTECT else "PROTECT")
        s = 1.0 if policy=="FULL" else ((1.0-kappa) if policy=="PARTIAL" else 0.0)
        kappa_rows.append({"seed":seed,"attr":attr,"kappa":round(kappa,3),"policy":policy,"s":round(s,2)})

        def proj(Hm, scale): return Hm - scale*np.outer(Hm @ v_bias, v_bias)

        variants = {
            "unedited": H,
            "gbp_gated": proj(H, s),
            "erase_ungated": proj(H, 1.0),
        }
        # INLP (simple 5-round linear nullspace) and LEACE on train stats
        Hi = H.copy()
        for _ in range(5):
            clf = LogisticRegression(max_iter=500).fit(Hi[tr], g_mask[tr])
            u = clf.coef_.squeeze(); u /= np.linalg.norm(u)
            Hi = Hi - np.outer(Hi @ u, u)
        variants["inlp"] = Hi
        er = LeaceEraser.fit(torch.tensor(H[tr]).float(),
                             torch.tensor(g_mask[tr].astype(np.float32)).unsqueeze(1))
        variants["leace"] = er(torch.tensor(H).float()).numpy()

        for name, Hm in variants.items():
            p = head_prob(Hm, w, b)
            r = metrics(y[te], p[te], g_mask[te], ref[te])
            rows.append({"seed":seed,"attr":attr,"method":name,"kappa":round(kappa,3),
                         "policy":policy, **{k:round(v,4) for k,v in r.items()}})

res = pd.DataFrame(rows); kap = pd.DataFrame(kappa_rows)
res.to_csv("results/fairness_utility_raw.csv", index=False)
kap.to_csv("results/kappa_policies.csv", index=False)

summary = (res.groupby(["attr","method"])
              [["auroc","ece","dp_gap","eo_gap","auroc_g","ece_g"]]
              .agg(["mean","std"]).round(4))
summary.to_csv("results/table2_summary.csv")
print(kap.groupby("attr")[["kappa"]].mean().round(3))
print("\nPolicy decisions (mode across seeds):")
print(kap.groupby("attr")["policy"].agg(lambda s: s.mode()[0]))
print("\nresults/ written: table2_summary.csv, kappa_policies.csv, fairness_utility_raw.csv")
