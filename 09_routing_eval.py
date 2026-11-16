"""
09_routing_eval.py - the routing experiment: for every case, run all five
intervention families at increasing severity, keep the smallest severity that
meets the joint budget, and classify the case as MEANDIFF-OK, RECOVERED or
IRRECOVERABLE.

FOUR CHANGES FROM THE SUBMITTED VERSION

1. IT READS THE CORRECTED COHORT. The submitted version had no arguments and
   read data/activations_seed{N}.npz, the untagged files. Every clinical
   routing number (the recovery count, the verdict map, the routing accuracy)
   therefore still described the original cohort, even after 01, 02 and 06
   were corrected. It now takes --tag, the suffix 02 and 06 use.

2. SURVEY-WEIGHTED, and computed both ways. In the weighted run the head is
   refit with sample_weight, the bias direction and the group means the
   operators use are weighted means, and the success budget is checked on
   weighted AUROC, weighted ECE and a weighted probe. The unweighted run goes
   through 08_families.py untouched, so it is the submitted procedure applied
   to the corrected cohort. Both are written; the script prints where they
   disagree.

3. LEACE IN NUMPY, the same closed form 03 uses, so the script no longer
   needs torch or concept-erasure and LEACE is never silently skipped.

4. THE SYNTHETIC ARM IS PROTECTED. If data_syn/ exists the 75 synthetic cases
   are recomputed. If it does not, they are carried forward unchanged from the
   existing cases.csv, since they do not depend on the cohort. The first run
   also keeps the submitted file as results3/cases_submitted.csv.

Outputs: results3/cases.csv              clinical weighted + synthetic
         results3/cases_unweighted.csv   clinical unweighted + synthetic
         results3/recovery_summary.csv, results3/routing_validation.csv

CONVERGENCE CHECK. --max-iter N re-runs the clinical cases with every
logistic fit (head, probes, INLP) allowed at least N iterations, writes
results3/cases_maxiter{N}.csv, and reports any verdict that differs from
cases.csv. It never overwrites cases.csv.

Run:  python 09_routing_eval.py --tag _fast
then: python 10_baserate_diag.py
      python 16_routing_threshold_validation.py
      python 11_fig_verdict_map.py
Deps: scikit-learn, pandas, numpy, pyarrow
"""
import argparse, glob, os, re, shutil, sys, numpy as np, pandas as pd
from importlib.machinery import SourceFileLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="", help="suffix used by 02_train_models.py")
ap.add_argument("--out", default="results3")
ap.add_argument("--max-iter", type=int, default=0,
                help="convergence check: raise every logistic fit to at least N iterations")
args = ap.parse_args()

TAG = args.tag
fam = SourceFileLoader("fam", "08_families.py").load_module()
os.makedirs(args.out, exist_ok=True)
SEVERITIES = [0.25, 0.5, 0.75, 1.0]
FAMILIES = ["meandiff", "midpoint", "meanmatch", "inlp2", "leace"]
W = 78

if args.max_iter:
    _LR = LogisticRegression

    def LogisticRegression(*a, **k):                 # noqa: F811
        k["max_iter"] = max(k.get("max_iter", 100), args.max_iter)
        return _LR(*a, **k)

    fam.LogisticRegression = LogisticRegression     # 08 builds its probes with this


# ------------------------------------------------------------------ helpers
def remap(idx, mask):
    pos = -np.ones(len(mask), dtype=int)
    pos[mask] = np.arange(mask.sum())
    return pos[idx[mask[idx]]]


def wmean0(X, w=None):
    return X.mean(0) if w is None else np.average(X, axis=0, weights=w)


def leace_fit(X, z, w=None):
    """Closed-form LEACE (Belrose et al. 2023). Eraser: x -> x - (x - mu) P^T."""
    X = np.asarray(X, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).reshape(len(X), -1)
    ww = np.ones(len(X)) if w is None else np.asarray(w, dtype=np.float64)
    ww = ww / ww.sum()
    mu, mz = ww @ X, ww @ z
    Xc, zc = X - mu, z - mz
    Sxx = (Xc * ww[:, None]).T @ Xc
    Sxz = (Xc * ww[:, None]).T @ zc
    L, V = np.linalg.eigh(Sxx)
    keep = L > L.max() * Sxx.shape[0] * np.finfo(np.float64).eps
    Wm = (V[:, keep] / np.sqrt(L[keep])) @ V[:, keep].T
    Wp = (V[:, keep] * np.sqrt(L[keep])) @ V[:, keep].T
    U, s, _ = np.linalg.svd(Wm @ Sxz, full_matrices=False)
    U = U[:, s > s.max() * 1e-10] if s.size and s.max() > 0 else U[:, :0]
    return mu, Wp @ (U @ U.T) @ Wm


def diagnostics(H_tr, g_tr, w, sw_tr=None, n_boot=30):
    """08_families.diagnostics, with optionally weighted group means. With
    sw_tr=None this is that function, called directly."""
    if sw_tr is None:
        return fam.diagnostics(H_tr, g_tr, w)
    rng = np.random.default_rng(0)

    def direction(H, g, sw):
        v = wmean0(H[g], sw[g]) - wmean0(H[~g], sw[~g])
        return v / np.linalg.norm(v)

    v = direction(H_tr, g_tr, sw_tr)
    kappa = abs(float(v @ (w / np.linalg.norm(w))))
    impact = kappa * np.linalg.norm(w) * np.std(H_tr @ v) / np.std(H_tr @ w)
    vs, n = [], len(H_tr)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        gi = g_tr[i]
        if gi.sum() < 5 or (~gi).sum() < 5:
            continue
        vs.append(direction(H_tr[i], gi, sw_tr[i]))
    V = np.stack(vs)
    cos = np.abs(V @ V.T)
    stability = float((cos.sum() - len(V)) / (len(V) ** 2 - len(V)))
    pr = H_tr @ v
    m1 = np.average(pr[g_tr], weights=sw_tr[g_tr])
    m0 = np.average(pr[~g_tr], weights=sw_tr[~g_tr])
    ma = np.average(pr, weights=sw_tr)
    asym = abs((m1 + m0) / 2 - ma) / (abs(m1 - m0) / 2 + 1e-9)
    return {"kappa": kappa, "impact": impact, "stability": stability,
            "asymmetry": asym, "v": v}


def edit(name, H, v, s, ctx, sw):
    if name in ("meandiff", "midpoint", "meanmatch"):
        return fam.FAMILIES[name](H, v, s, ctx)          # ctx carries the means
    if name == "inlp2":
        if sw is None:
            return fam.FAMILIES["inlp2"](H, v, s, ctx)
        Hi, tr, g = H.copy(), ctx["tr"], ctx["g_all"]
        for _ in range(2):
            clf = LogisticRegression(max_iter=500).fit(Hi[tr], g[tr], sample_weight=sw[tr])
            u = clf.coef_.squeeze()
            u /= np.linalg.norm(u)
            Hi = Hi - s * np.outer(Hi @ u, u)
        return Hi
    mu, P = ctx["leace"]
    He = H - (H - mu) @ P.T
    return H + s * (He - H)


def wece(y, p, w, bins=10):
    edges, tot, e = np.linspace(0, 1, bins + 1), w.sum(), 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])         # binning as in 08
        if m.any():
            e += w[m].sum() / tot * abs(np.average(y[m], weights=w[m])
                                        - np.average(p[m], weights=w[m]))
    return e


def evaluate(Hm, H0, y, g, w, b, te, tr, sw):
    if sw is None:
        return fam.evaluate_case(Hm, H0, y, g, w, b, te, tr)
    prob = lambda X: 1 / (1 + np.exp(-(X @ w + b)))
    p0, pm, wte = prob(H0)[te], prob(Hm)[te], sw[te]
    dmg = (roc_auc_score(y[te], p0, sample_weight=wte)
           - roc_auc_score(y[te], pm, sample_weight=wte))
    d_ece = wece(y[te], pm, wte) - wece(y[te], p0, wte)
    probe = LogisticRegression(max_iter=500).fit(Hm[tr], g[tr], sample_weight=sw[tr])
    pa = roc_auc_score(g[te], probe.predict_proba(Hm[te])[:, 1], sample_weight=wte)
    return {"auc_damage": dmg, "ece_increase": d_ece, "probe_auc": pa}


def route(case, weighted):
    H, y, g, tr, te = case["H"], case["y"], case["g"], case["tr"], case["te"]
    w, b = (case["w_wtd"], case["b_wtd"]) if weighted else (case["w"], case["b"])
    sw = case["wt"] if weighted else None
    d = diagnostics(H[tr], g[tr], w, None if sw is None else sw[tr])
    v = d.pop("v")
    swt = None if sw is None else sw[tr]
    m1 = wmean0(H[tr][g[tr]], None if swt is None else swt[g[tr]])
    m0 = wmean0(H[tr][~g[tr]], None if swt is None else swt[~g[tr]])
    ctx = {"c_mid": (m1 + m0) / 2, "m1": m1, "m0": m0, "g_all": g, "tr": tr,
           "leace": leace_fit(H[tr], g[tr].astype(np.float64), swt)}
    res = {}
    for name in FAMILIES:
        res[name] = None
        for s in SEVERITIES:
            m = evaluate(edit(name, H, v, s, ctx, sw), H, y, g, w, b, te, tr, sw)
            if fam.success(m):
                res[name] = s                              # smallest safe severity
                break
    verdict = ("meandiff_ok" if res["meandiff"] is not None else
               ("recovered" if any(res[k] is not None for k in FAMILIES if k != "meandiff")
                else "irrecoverable"))
    rec_by = ",".join(k for k in FAMILIES if k != "meandiff" and res[k] is not None)
    return {"testbed": case["testbed"], "case": case["case"], "truth": case["truth"],
            **{k: round(float(x), 3) for k, x in d.items()},
            "verdict": verdict, "recovered_by": rec_by,
            **{f"s_{k}": res[k] for k in FAMILIES}}


# -------------------------------------------------------------------- cases
def nhanes_cases():
    gpath = f"data/groups{TAG}.parquet"
    if not os.path.exists(gpath):
        sys.exit(f"{gpath} not found. Run 02_train_models.py --tag {TAG or '(none)'} first.")
    groups = pd.read_parquet(gpath)
    ATTRS = {"sex_female": (groups.sex_female.values == 1),
             "race_black": (groups.race.values == "black_nh"),
             "race_hisp":  (groups.race.values == "hispanic"),
             "age_60plus": (groups.age.values >= 60)}
    REF = {"sex_female": None, "age_60plus": (groups.age.values < 45),
           "race_black": (groups.race.values == "white_nh"),
           "race_hisp":  (groups.race.values == "white_nh")}
    pat = re.compile(r"activations_seed(\d+)" + re.escape(TAG) + r"\.npz$")
    files = sorted((int(pat.search(f).group(1)), f)
                   for f in glob.glob("data/activations_seed*.npz") if pat.search(f))
    if not files:
        sys.exit(f"no data/activations_seed*{TAG}.npz found. Run 02_train_models.py --tag {TAG}.")
    for seed, f in files:
        z = np.load(f)
        if len(z["y"]) != len(groups):
            sys.exit(f"{f} has {len(z['y'])} rows but {gpath} has {len(groups)}. "
                     f"They come from different cohorts.")
        tr = z["idx_train"]
        wt = z["wt"] if "wt" in z else None
        lr = LogisticRegression(max_iter=1000).fit(z["H"][tr], z["y"][tr])
        w, b = lr.coef_.squeeze(), float(lr.intercept_[0])
        w_w = b_w = None
        if wt is not None:
            lw = LogisticRegression(max_iter=1000).fit(z["H"][tr], z["y"][tr], sample_weight=wt[tr])
            w_w, b_w = lw.coef_.squeeze(), float(lw.intercept_[0])
        for attr, g in ATTRS.items():
            ref = REF[attr] if REF[attr] is not None else ~g
            mask = g | ref                    # restrict to the two groups compared
            yield {"testbed": "nhanes", "case": f"{attr}_s{seed}", "truth": "",
                   "H": z["H"][mask], "y": z["y"][mask], "g": g[mask],
                   "tr": remap(tr, mask), "te": remap(z["idx_test"], mask),
                   "w": w, "b": b, "w_wtd": w_w, "b_wtd": b_w,
                   "wt": None if wt is None else wt[mask], "n": int(len(groups))}


def syn_cases():
    man = pd.read_csv("data_syn/manifest.csv").set_index("case")
    for f in sorted(glob.glob("data_syn/case_*.npz")):
        z = np.load(f)
        name = os.path.basename(f)[:-4]
        yield {"testbed": "synthetic", "case": name, "truth": man.loc[name, "truth"],
               "H": z["H"], "y": z["y"], "g": z["g"].astype(bool),
               "tr": z["idx_train"], "te": z["idx_test"],
               "w": z["w_task"], "b": float(z["b_task"]), "wt": None}


# ------------------------------------------------------- convergence check
if args.max_iter:
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    ref_w, ref_u = f"{args.out}/cases.csv", f"{args.out}/cases_unweighted.csv"
    for f in (ref_w, ref_u):
        if not os.path.exists(f):
            sys.exit(f"{f} not found. Run 09_routing_eval.py --tag {TAG} first.")
    rw, ru_, nconv = [], [], 0
    print(f"convergence check: every logistic fit raised to >= {args.max_iter} iterations")
    for case in nhanes_cases():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            ru_.append(route(case, weighted=False))
            rw.append(route(case, weighted=True) if case["wt"] is not None else ru_[-1])
        nconv += sum(issubclass(x.category, ConvergenceWarning) for x in caught)
        print(f"  {case['case']:16s} done")
    out = pd.DataFrame(rw)
    out.to_csv(f"{args.out}/cases_maxiter{args.max_iter}.csv", index=False)
    print()
    print(f"  convergence warnings at this cap: {nconv}")
    for lab, new, ref in (("weighted", pd.DataFrame(rw), ref_w), ("unweighted", pd.DataFrame(ru_), ref_u)):
        old = pd.read_csv(ref)
        m = new.merge(old[old.testbed == "nhanes"], on="case", suffixes=("_new", "_old"))
        dv = m[m.verdict_new != m.verdict_old]
        ds = sum(int(not ((pd.isna(r[f"s_{k}_new"]) and pd.isna(r[f"s_{k}_old"]))
                          or r[f"s_{k}_new"] == r[f"s_{k}_old"]))
                 for _, r in m.iterrows() for k in FAMILIES)
        print(f"  {lab:10s}: {len(dv)} of {len(m)} verdicts change, {ds} of {5 * len(m)} family severities change")
        for _, r in dv.iterrows():
            print(f"      {r.case}: {r.verdict_old} -> {r.verdict_new}")
    print(f"\nwrote {args.out}/cases_maxiter{args.max_iter}.csv  (cases.csv untouched)")
    sys.exit(0)

# ---------------------------------------------------------------------- run
cases_path = f"{args.out}/cases.csv"
backup = f"{args.out}/cases_submitted.csv"
previous = pd.read_csv(cases_path) if os.path.exists(cases_path) else None
if previous is not None and not os.path.exists(backup):
    shutil.copy(cases_path, backup)
    print(f"kept the existing {cases_path} as {backup}")

rows_w, rows_u, n_cohort, have_wt = [], [], None, True
print(f"clinical cases from data/activations_seed*{TAG}.npz ...")
for case in nhanes_cases():
    n_cohort = case["n"]
    ru = route(case, weighted=False)
    rows_u.append(ru)
    if case["wt"] is None:
        have_wt = False
        rows_w.append(ru)
    else:
        rows_w.append(route(case, weighted=True))
    r = rows_w[-1]
    print(f"  {r['case']:16s} -> {r['verdict']:13s} (I={r['impact']:.2f} "
          f"A={r['asymmetry']:.2f} S={r['stability']:.2f})"
          + ("" if ru["verdict"] == r["verdict"] else f"   unweighted: {ru['verdict']}"))
if not have_wt:
    print("WARNING: no survey weights in the activations; the 'weighted' file is unweighted.")

if os.path.exists("data_syn/manifest.csv"):
    print("synthetic cases from data_syn/ ...")
    syn = pd.DataFrame([route(c, weighted=False) for c in syn_cases()])
    syn_note = "recomputed from data_syn/"
elif previous is not None and (previous.testbed == "synthetic").any():
    syn = previous[previous.testbed == "synthetic"].copy()
    syn_note = f"carried forward unchanged from the existing {cases_path}"
else:
    sys.exit("No data_syn/ and no existing cases.csv to carry the synthetic cases "
             "from. Run 07_synthetic_testbed.py first.")
print(f"synthetic arm: {len(syn)} cases, {syn_note}")

cols = ["testbed", "case", "truth", "kappa", "impact", "stability", "asymmetry",
        "verdict", "recovered_by"] + [f"s_{k}" for k in FAMILIES]
dfw = pd.concat([pd.DataFrame(rows_w), syn], ignore_index=True)[cols]
dfu = pd.concat([pd.DataFrame(rows_u), syn], ignore_index=True)[cols]
dfw.to_csv(cases_path, index=False)
dfu.to_csv(f"{args.out}/cases_unweighted.csv", index=False)
summ = dfw.groupby(["testbed", "verdict"]).size().unstack(fill_value=0)
summ.to_csv(f"{args.out}/recovery_summary.csv")
val = pd.crosstab(dfw[dfw.testbed == "synthetic"].truth, dfw[dfw.testbed == "synthetic"].verdict)
val.to_csv(f"{args.out}/routing_validation.csv")

# ------------------------------------------------------------------- report
nw = dfw[dfw.testbed == "nhanes"].copy()
nu = dfu[dfu.testbed == "nhanes"].copy()
for t in (nw, nu):
    t["attr"] = t.case.str.rsplit("_s", n=1).str[0]
ORDER = ["sex_female", "race_black", "race_hisp", "age_60plus"]

print()
print("=" * W)
print(f"CLINICAL VERDICTS, survey-weighted  (cohort N={n_cohort}, tag '{TAG}')")
print("=" * W)
vc = nw.verdict.value_counts()
n_md, n_rec, n_irr = (int(vc.get(k, 0)) for k in ("meandiff_ok", "recovered", "irrecoverable"))
print(f"  mean-difference safe {n_md}   recovered {n_rec}   irrecoverable {n_irr}")
print(f"  i.e. {n_rec} of {n_rec + n_irr} cases refused under mean-difference projection are recovered")
print()
print(f"  {'attribute':12s} {'impact':>14s}  " + "  ".join(f"{k:>9s}" for k in FAMILIES) + "   verdicts")
for a in ORDER:
    t = nw[nw.attr == a]
    fams = "  ".join(f"{int(t[f's_{k}'].notna().sum())}/{len(t):<7d}" for k in FAMILIES)
    print(f"  {a:12s} {t.impact.min():.2f} to {t.impact.max():.2f}    {fams}  {t.verdict.value_counts().to_dict()}")
sev = nw.loc[nw.verdict == "recovered", "s_meanmatch"].dropna()
if len(sev):
    print(f"\n  mean-matching severities among recovered cases: {sorted(sev.unique().tolist())}")
print(f"  stability S: min {nw.stability.min():.3f}   asymmetry A: {nw.asymmetry.min():.2f} to {nw.asymmetry.max():.2f}")

diff = nw.merge(nu, on="case", suffixes=("_w", "_u"))
diff = diff[diff.verdict_w != diff.verdict_u]
print()
print(f"  weighted and unweighted verdicts disagree on {len(diff)} of {len(nw)} clinical cases")
for _, r in diff.iterrows():
    print(f"    {r.case}: weighted {r.verdict_w}, unweighted {r.verdict_u}")

if os.path.exists(backup):
    sub = pd.read_csv(backup)
    sub = sub[sub.testbed == "nhanes"].copy()
    sub["attr"] = sub.case.str.rsplit("_s", n=1).str[0]
    print()
    print("=" * W)
    print("SUBMITTED COHORT vs CORRECTED COHORT, clinical verdicts by attribute")
    print("=" * W)
    for a in ORDER:
        print(f"  {a:12s} submitted {sub[sub.attr == a].verdict.value_counts().to_dict()}"
              f"   corrected {nw[nw.attr == a].verdict.value_counts().to_dict()}")
    svc = sub.verdict.value_counts()
    print(f"\n  submitted: {int(svc.get('recovered', 0))} recovered of "
          f"{int(svc.get('recovered', 0)) + int(svc.get('irrecoverable', 0))} refused."
          f"   corrected: {n_rec} of {n_rec + n_irr}.")

print()
print("=" * W)
print("SYNTHETIC GROUND TRUTH vs VERDICT")
print("=" * W)
print(val.to_string())

print()
print(f"wrote {cases_path}  and  {args.out}/cases_unweighted.csv")
print("next: python 10_baserate_diag.py, then 16, then 11. The recovery count,")
print("the routing accuracy and the verdict map all read this file.")
