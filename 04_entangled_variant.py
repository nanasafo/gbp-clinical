"""04_entangled_variant.py — age-included model: does kappa detect true entanglement?"""
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEEDS=[0,1,2,3,4]
FEATURES=["BMXBMI","BMXWAIST","SBP","LBXTC","LBDHDD","log_tg","smoker_current","age"]  # + age
df=pd.read_parquet("data/nhanes_cohort.parquet")
X=df[FEATURES].values.astype(np.float32); y=df["diabetes"].values.astype(np.float32)
g=(df["age"].values>=60); ref=(df["age"].values<45)

class MLP(nn.Module):
    def __init__(s,d):
        super().__init__(); s.body=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU()); s.head=nn.Linear(32,1)
    def forward(s,x,h=False):
        z=s.body(x); o=s.head(z).squeeze(-1); return (o,z) if h else o

rows=[]
for seed in SEEDS:
    idx=np.arange(len(y)); tr,tmp=train_test_split(idx,test_size=0.4,random_state=seed,stratify=y)
    va,te=train_test_split(tmp,test_size=0.5,random_state=seed,stratify=y[tmp])
    Xs=StandardScaler().fit(X[tr]).transform(X).astype(np.float32)
    torch.manual_seed(seed); m=MLP(len(FEATURES)); opt=torch.optim.Adam(m.parameters(),1e-3); lf=nn.BCEWithLogitsLoss()
    Xt,yt=torch.tensor(Xs),torch.tensor(y); best,pat=1e9,0
    for ep in range(200):
        m.train(); opt.zero_grad(); l=lf(m(Xt[tr]),yt[tr]); l.backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl=lf(m(Xt[va]),yt[va]).item()
        if vl<best-1e-4: best,state,pat=vl,m.state_dict(),0
        else:
            pat+=1
            if pat>=15: break
    m.load_state_dict(state); m.eval()
    with torch.no_grad(): _,H=m(Xt,h=True); H=H.numpy()
    lr=LogisticRegression(max_iter=1000).fit(H[tr],y[tr]); w,b=lr.coef_.squeeze(),lr.intercept_[0]
    v_t=w/np.linalg.norm(w)
    v_b=H[tr][g[tr]].mean(0)-H[tr][ref[tr]].mean(0); v_b/=np.linalg.norm(v_b)
    k=abs(float(v_b@v_t))
    def prob(Hm): return 1/(1+np.exp(-(Hm@w+b)))
    auc0=roc_auc_score(y[te],prob(H)[te])
    He=H-np.outer(H@v_b,v_b)                       # ungated full erasure
    auc1=roc_auc_score(y[te],prob(He)[te])
    pol="PROTECT" if k>0.50 else ("PARTIAL" if k>=0.30 else "FULL")
    rows.append({"seed":seed,"kappa":round(k,3),"policy":pol,
                 "auc_unedited":round(auc0,3),"auc_after_full_erasure":round(auc1,3),
                 "auc_damage":round(auc0-auc1,3)})
    print(rows[-1])
pd.DataFrame(rows).to_csv("results/entangled_variant.csv",index=False)
print("\nmean kappa:",np.mean([r['kappa'] for r in rows]).round(3),
      "| mean AUROC damage from ungated erasure:",np.mean([r['auc_damage'] for r in rows]).round(3))