"""06_impact_gate.py — does impact-weighted coherence predict harm better than kappa?"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

groups = pd.read_parquet("data/groups.parquet")
ATTRS = {"sex_female": (groups.sex_female.values==1),
         "race_black": (groups.race.values=="black_nh"),
         "race_hisp":  (groups.race.values=="hispanic"),
         "age_60plus": (groups.age.values>=60)}
REF = {"sex_female": None, "age_60plus": (groups.age.values<45),
       "race_black": (groups.race.values=="white_nh"),
       "race_hisp":  (groups.race.values=="white_nh")}

rows=[]
for seed in [0,1,2,3,4]:
    z=np.load(f"data/activations_seed{seed}.npz")
    H,y,tr,te=z["H"],z["y"],z["idx_train"],z["idx_test"]
    lr=LogisticRegression(max_iter=1000).fit(H[tr],y[tr])
    w,b=lr.coef_.squeeze(),lr.intercept_[0]
    v_t=w/np.linalg.norm(w)
    logit_std=np.std(H[tr]@w)
    for attr,g in ATTRS.items():
        ref=REF[attr] if REF[attr] is not None else ~g
        v_b=H[tr][g[tr]].mean(0)-H[tr][ref[tr]].mean(0); v_b/=np.linalg.norm(v_b)
        kappa=abs(float(v_b@v_t))
        proj_std=np.std(H[tr]@v_b)                      # signal mass along bias dir
        impact=kappa*np.linalg.norm(w)*proj_std/logit_std   # predicted logit disruption (relative)
        prob=lambda Hm: 1/(1+np.exp(-(Hm@w+b)))
        He=H-np.outer(H@v_b,v_b)
        dmg=roc_auc_score(y[te],prob(H)[te])-roc_auc_score(y[te],prob(He)[te])
        rows.append({"seed":seed,"attr":attr,"kappa":round(kappa,3),
                     "impact":round(impact,3),"auroc_damage":round(dmg,3)})
d=pd.DataFrame(rows); print(d.sort_values("impact").to_string(index=False))
rk,pk=spearmanr(d.kappa,d.auroc_damage); ri,pi=spearmanr(d.impact,d.auroc_damage)
print(f"\nSpearman(kappa, damage)  = {rk:.3f} (p={pk:.1e})")
print(f"Spearman(impact, damage) = {ri:.3f} (p={pi:.1e})")
d.to_csv("results/impact_gate.csv",index=False)