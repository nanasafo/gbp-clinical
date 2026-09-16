"""
08_families.py — pre-intervention diagnostics + intervention families.
Imported by 09_routing_eval.py. No side effects.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ---------------- diagnostics (all computed on TRAIN only) ----------------
def bias_direction(H, g, ref=None):
    ref = ~g if ref is None else ref
    v = H[g].mean(0) - H[ref].mean(0)
    return v / np.linalg.norm(v)

def diagnostics(H_tr, g_tr, w, n_boot=30, rng=None):
    rng = rng or np.random.default_rng(0)
    v = bias_direction(H_tr, g_tr)
    v_t = w / np.linalg.norm(w)
    kappa = abs(float(v @ v_t))
    impact = kappa * np.linalg.norm(w) * np.std(H_tr @ v) / np.std(H_tr @ w)
    # stability: bootstrap the direction
    vs = []
    n = len(H_tr)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        gi = g_tr[i]
        if gi.sum() < 5 or (~gi).sum() < 5: continue
        vs.append(bias_direction(H_tr[i], gi))
    V = np.stack(vs)
    cos = np.abs(V @ V.T)
    stability = float((cos.sum() - len(V)) / (len(V)**2 - len(V)))
    # asymmetry: one-sidedness of group means about the global mean, along v
    m1, m0, ma = (H_tr[g_tr] @ v).mean(), (H_tr[~g_tr] @ v).mean(), (H_tr @ v).mean()
    asym = abs((m1 + m0)/2 - ma) / (abs(m1 - m0)/2 + 1e-9)
    return {"kappa": kappa, "impact": impact, "stability": stability, "asymmetry": asym, "v": v}

# ---------------- intervention families ----------------
def edit_meandiff(H, v, s, ctx):                 # incumbent
    return H - s * np.outer(H @ v, v)

def edit_midpoint(H, v, s, ctx):                 # centered removal
    c = ctx["c_mid"]
    return H - s * np.outer((H - c) @ v, v)

def edit_meanmatch(H, v, s, ctx):                # bidirectional translation
    g = ctx["g_all"]; c = ctx["c_mid"]
    m1, m0 = ctx["m1"], ctx["m0"]
    out = H.copy()
    out[g]  -= s * ((m1 - c) @ v) * v
    out[~g] -= s * ((m0 - c) @ v) * v
    return out

def edit_inlp2(H, v, s, ctx):                    # 2-round nullspace (subspace)
    Hi = H.copy(); tr, g = ctx["tr"], ctx["g_all"]
    for _ in range(2):
        clf = LogisticRegression(max_iter=500).fit(Hi[tr], g[tr])
        u = clf.coef_.squeeze(); u /= np.linalg.norm(u)
        Hi = Hi - s * np.outer(Hi @ u, u)
    return Hi

def edit_leace(H, v, s, ctx):                    # surgical affine (s=1 only meaningful)
    er = ctx["leace_eraser"]
    import torch
    He = er(torch.tensor(H).float()).numpy()
    return H + s * (He - H)

FAMILIES = {
    "meandiff": edit_meandiff, "midpoint": edit_midpoint,
    "meanmatch": edit_meanmatch, "inlp2": edit_inlp2, "leace": edit_leace,
}

# ---------------- evaluation ----------------
def evaluate_case(Hm, H0, y, g, w, b, te, tr):
    prob = lambda X: 1/(1+np.exp(-(X @ w + b)))
    def ece(yv, p, bins=10):
        e, edges = 0.0, np.linspace(0,1,bins+1)
        for i in range(bins):
            m = (p >= edges[i]) & (p < edges[i+1])
            if m.sum(): e += m.mean() * abs(yv[m].mean() - p[m].mean())
        return e
    p0, pm = prob(H0), prob(Hm)
    auc0, aucm = roc_auc_score(y[te], p0[te]), roc_auc_score(y[te], pm[te])
    d_ece = ece(y[te], pm[te]) - ece(y[te], p0[te])
    # attribute decodability after edit (probe trained on edited TRAIN, scored on TEST)
    probe = LogisticRegression(max_iter=500).fit(Hm[tr], g[tr])
    probe_auc = roc_auc_score(g[te], probe.predict_proba(Hm[te])[:,1])
    return {"auc_damage": auc0 - aucm, "ece_increase": d_ece, "probe_auc": probe_auc}

def success(m, probe_max=0.55, auc_budget=0.01, ece_budget=0.02):
    return (m["probe_auc"] <= probe_max and m["auc_damage"] <= auc_budget
            and m["ece_increase"] <= ece_budget)
