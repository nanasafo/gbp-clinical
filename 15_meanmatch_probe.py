"""
15_meanmatch_probe.py - does the recovery claim survive a nonlinear probe,
and if a probe succeeds, is it reading the ATTRIBUTE or reading the EDIT?

THE REVIEWER POINT
`08_families.py` defines success as `probe_auc <= 0.55` from a
LogisticRegression probe on the edited representation (`evaluate_case`, line
85). Mean-matching translation preserves within-group variance by design, so a
linear probe losing sight of the attribute does not establish that it is gone.
If a boosted or MLP probe still recovers it, "13 of 18 recovered" means the
linear probe stopped seeing it.

THE CONTROL, AND WHY IT IS ESSENTIAL
A naive nonlinear probe overstates this badly. `edit_meanmatch` subtracts a
DIFFERENT constant vector from each group, so the edit is group-conditional.
Wherever the representation holds exact ties, and a ReLU layer holds them in
bulk because every inactive unit is exactly 0, those ties become two distinct
constants, one per group, and a tree separates them perfectly while learning
nothing about the attribute.

Demonstrated: on ReLU activations, mean-matching makes a MEANINGLESS RANDOM
partition recoverable at AUROC 1.000. On the same data without exact zeros,
the same probe sits at 0.47. A high score alone therefore cannot be
interpreted.

Every case is run twice: once with the real attribute, once with a random
partition of the same prevalence, edited by the same family at the same
severity. The random arm measures what the edit reveals about ANY partition.
Attribute minus random is what is actually left of the attribute.

READING THE RESULT, per family:
  real ~ chance                    erasure is genuine
  real high, random EQUALLY high   the operator is self-revealing. The score
                                   is an artifact of a group-conditional edit
                                   meeting exact ties, not residual attribute
                                   content. Still a serious failure of the
                                   operator and a sharper criticism than
                                   linear decodability, but a different claim.
  real high, random at chance      genuine residual recoverability. Scope
                                   every claim to linear guardedness.

Nothing in your pipeline is modified; `08_families.py` is imported.

REVISED TO MATCH 09. The first version estimated every operator unweighted,
while the corrected 09_routing_eval.py is survey-weighted, so the two scripts
disagreed about which cases pass the linear test (14 here, 15 there). This
version uses the same weighted head, weighted directions, weighted group
means and weighted success budget as 09, and the same numpy LEACE, so LEACE
is never skipped. Its linear-criterion verdicts must equal 09's on the same
tag, and it prints them so you can check. --unweighted reproduces the first
version exactly and writes to a separate file.

Run:  python 15_meanmatch_probe.py --tag _fast
      python 15_meanmatch_probe.py --tag _mec
Deps: scikit-learn, pandas, numpy, pyarrow
"""
import os, glob, argparse, warnings, numpy as np, pandas as pd
from importlib.machinery import SourceFileLoader
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="", help="suffix used by 02_train_models.py")
ap.add_argument("--probe-max", type=float, default=0.55,
                help="the erasure threshold used by 08_families.success")
ap.add_argument("--unweighted", action="store_true",
                help="estimate operators unweighted, as the first version did")
ap.add_argument("--no-control", action="store_true",
                help="skip the random-partition control (not recommended)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results3")
args = ap.parse_args()

TAG = args.tag
rng = np.random.default_rng(args.seed)
os.makedirs(args.out, exist_ok=True)
fam = SourceFileLoader("fam", "08_families.py").load_module()
SEVERITIES = [0.25, 0.5, 0.75, 1.0]          # identical to 09_routing_eval.py
W = 84

FAMILIES = ["meandiff", "midpoint", "meanmatch", "inlp2", "leace"]


# ---- the same helpers as 09_routing_eval.py, so the two scripts agree ----
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
    U, sv, _ = np.linalg.svd(Wm @ Sxz, full_matrices=False)
    U = U[:, sv > sv.max() * 1e-10] if sv.size and sv.max() > 0 else U[:, :0]
    return mu, Wp @ (U @ U.T) @ Wm


def direction(H_tr, lab_tr, sw_tr=None):
    if sw_tr is None:
        return fam.bias_direction(H_tr, lab_tr)
    v = wmean0(H_tr[lab_tr], sw_tr[lab_tr]) - wmean0(H_tr[~lab_tr], sw_tr[~lab_tr])
    return v / np.linalg.norm(v)


def edit(name, H, v, s, ctx, sw):
    if name in ("meandiff", "midpoint", "meanmatch"):
        return fam.FAMILIES[name](H, v, s, ctx)          # ctx carries the means
    if name == "inlp2":
        if sw is None:
            return fam.FAMILIES["inlp2"](H, v, s, ctx)
        Hi, tr_, g_ = H.copy(), ctx["tr"], ctx["g_all"]
        for _ in range(2):
            clf = LogisticRegression(max_iter=500).fit(Hi[tr_], g_[tr_], sample_weight=sw[tr_])
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
        m = (p >= edges[i]) & (p < edges[i + 1])
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


def remap(idx, mask):
    """Index remap, identical to _remap in 09_routing_eval.py."""
    pos = -np.ones(len(mask), dtype=int)
    pos[mask] = np.arange(mask.sum())
    return pos[idx[mask[idx]]]


def nhanes_cases(tag):
    groups = pd.read_parquet(f"data/groups{tag}.parquet")
    ATTRS = {"sex_female": (groups.sex_female.values == 1),
             "race_black": (groups.race.values == "black_nh"),
             "race_hisp":  (groups.race.values == "hispanic"),
             "age_60plus": (groups.age.values >= 60)}
    REF = {"sex_female": None,
           "age_60plus": (groups.age.values < 45),
           "race_black": (groups.race.values == "white_nh"),
           "race_hisp":  (groups.race.values == "white_nh")}
    files = sorted(glob.glob(f"data/activations_seed*{tag}.npz")) if tag \
        else sorted(f for f in glob.glob("data/activations_seed*.npz")
                    if f.rstrip(".npz").split("seed")[1].isdigit())
    if not files:
        raise RuntimeError(f"no data/activations_seed*{tag}.npz found")
    for f in files:
        z = np.load(f)
        seed = int(''.join(c for c in f.split("seed")[1] if c.isdigit()))
        lr = LogisticRegression(max_iter=1000).fit(z["H"][z["idx_train"]],
                                                   z["y"][z["idx_train"]])
        w, b = lr.coef_.squeeze(), float(lr.intercept_[0])
        wt = z["wt"] if "wt" in z else None
        if wt is not None and not args.unweighted:      # same head refit as 09
            lw = LogisticRegression(max_iter=1000).fit(
                z["H"][z["idx_train"]], z["y"][z["idx_train"]],
                sample_weight=wt[z["idx_train"]])
            w, b = lw.coef_.squeeze(), float(lw.intercept_[0])
        for attr, g in ATTRS.items():
            ref = REF[attr] if REF[attr] is not None else ~g
            mask = g | ref
            yield {"case": f"{attr}_s{seed}", "attr": attr, "seed": seed,
                   "H": z["H"][mask], "y": z["y"][mask], "g": g[mask],
                   "tr": remap(z["idx_train"], mask), "te": remap(z["idx_test"], mask),
                   "w": w, "b": b,
                   "wt": (wt[mask] if wt is not None else None)}


def probes():
    return {"linear": LogisticRegression(max_iter=500),
            "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                 early_stopping=True, n_iter_no_change=10,
                                 random_state=0),
            "gbm": HistGradientBoostingClassifier(max_iter=150, random_state=0)}


def probe_all(Hm, lab, tr, te, wt, prefix=""):
    out = {}
    for name, clf in probes().items():
        if len(np.unique(lab[tr])) < 2 or len(np.unique(lab[te])) < 2:
            out[prefix + name] = np.nan
            continue
        clf.fit(Hm[tr], lab[tr])
        out[prefix + name] = roc_auc_score(
            lab[te], clf.predict_proba(Hm[te])[:, 1],
            sample_weight=(wt[te] if wt is not None else None))
    return out


def build_ctx(H, lab, tr, sw):
    swt = None if sw is None else sw[tr]
    m1 = wmean0(H[tr][lab[tr]], None if swt is None else swt[lab[tr]])
    m0 = wmean0(H[tr][~lab[tr]], None if swt is None else swt[~lab[tr]])
    return {"c_mid": (m1 + m0) / 2, "m1": m1, "m0": m0, "g_all": lab, "tr": tr,
            "leace": leace_fit(H[tr], lab[tr].astype(np.float64), swt)}


rows = []
weighted_any = False
for case in nhanes_cases(TAG):
    H, y, g = case["H"], case["y"], case["g"]
    tr, te, w, b, wt = case["tr"], case["te"], case["w"], case["b"], case["wt"]
    weighted_any = weighted_any or (wt is not None)
    zero_frac = float((H == 0).mean())          # measured on the ORIGINAL H

    sw = None if (args.unweighted or wt is None) else wt   # operator weights
    v = direction(H[tr], g[tr], None if sw is None else sw[tr])
    ctx = build_ctx(H, g, tr, sw)

    # a partition with the same prevalence and no relationship to anything
    g_rand = np.zeros(len(g), dtype=bool)
    g_rand[rng.permutation(len(g))[:int(g.sum())]] = True
    v_rand = direction(H[tr], g_rand[tr], None if sw is None else sw[tr])
    ctx_rand = build_ctx(H, g_rand, tr, sw)

    for name in FAMILIES:
        chosen_s = chosen_m = chosen_Hm = None
        for s in SEVERITIES:                       # identical first-success rule
            Hm = edit(name, H, v, s, ctx, sw)
            m = evaluate(Hm, H, y, g, w, b, te, tr, sw)
            if fam.success(m, probe_max=args.probe_max):
                chosen_s, chosen_m, chosen_Hm = s, m, Hm
                break
        rec = {"case": case["case"], "attr": case["attr"], "seed": case["seed"],
               "family": name, "severity": chosen_s,
               "succeeded": chosen_s is not None, "zero_frac": zero_frac,
               "auc_damage": chosen_m["auc_damage"] if chosen_m else np.nan,
               "probe_linear_09": chosen_m["probe_auc"] if chosen_m else np.nan}
        if chosen_s is not None:
            rec.update(probe_all(chosen_Hm, g, tr, te, wt))
            if not args.no_control:
                Hr = edit(name, H, v_rand, chosen_s, ctx_rand, sw)
                rec.update(probe_all(Hr, g_rand, tr, te, wt, prefix="rnd_"))
        rows.append(rec)
    print(f"  {case['case']:18s} done")

d = pd.DataFrame(rows)
for c in ["linear", "mlp", "gbm", "rnd_linear", "rnd_mlp", "rnd_gbm"]:
    if c not in d.columns:
        d[c] = np.nan
ok = d[d.succeeded].copy()
ok["best_nl"] = ok[["mlp", "gbm"]].max(axis=1)
ok["rnd_best_nl"] = ok[["rnd_mlp", "rnd_gbm"]].max(axis=1)
# Excess must be the largest PER-PROBE difference, not the difference of the
# maxima. A probe that saturates at 1.000 on both arms, as a tree does against
# a group-conditional edit, would otherwise cancel to zero and mask a real
# signal in another probe.
for _p in ("linear", "mlp", "gbm"):
    ok["ex_" + _p] = ok[_p] - ok["rnd_" + _p]
ok["excess"] = ok[["ex_linear", "ex_mlp", "ex_gbm"]].max(axis=1)

print()
print("=" * W)
print(f"EDITS THAT PASS THE LINEAR TEST, RE-PROBED  (tag {TAG or '(none)'})")
print("=" * W)
print(f"  'erased' means probe AUROC <= {args.probe_max:.2f}, as in 08_families.success")
print(f"  AUROC is {'survey-weighted' if weighted_any else 'UNWEIGHTED (npz has no wt)'}")
print(f"  operators are estimated {'UNWEIGHTED (first version)' if args.unweighted or not weighted_any else 'survey-weighted, as in 09'}")
print(f"  exact zeros in the representation: {100*d.zero_frac.mean():.1f}%")
print()
print(f"  {'family':10s} {'n':>3s} | {'ATTRIBUTE':^27s} | {'RANDOM PARTITION':^27s} | excess")
print(f"  {'':10s} {'':>3s} | {'linear':>8s}{'mlp':>9s}{'gbm':>9s} |"
      f" {'linear':>8s}{'mlp':>9s}{'gbm':>9s} |")
for name in FAMILIES:
    t = ok[ok.family == name]
    if not len(t):
        continue
    ex = {q: t[q].mean() - t["rnd_" + q].mean() for q in ("linear", "mlp", "gbm")}
    top = max(ex, key=ex.get)
    print(f"  {name:10s} {len(t):3d} | {t.linear.mean():8.3f}{t.mlp.mean():9.3f}"
          f"{t.gbm.mean():9.3f} | {t.rnd_linear.mean():8.3f}{t.rnd_mlp.mean():9.3f}"
          f"{t.rnd_gbm.mean():9.3f} | {ex[top]:+7.3f} ({top})")

mm = ok[ok.family == "meanmatch"]
print()
print("=" * W)
print("VERDICT")
print("=" * W)
if not len(mm):
    print("  mean-matching passed on no case; nothing to interpret.")
else:
    n_tot = int((d.family == "meanmatch").sum())
    real, rand = float(mm.best_nl.mean()), float(mm.rnd_best_nl.mean())
    exq = {q: mm[q].mean() - mm["rnd_" + q].mean() for q in ("linear", "mlp", "gbm")}
    top = max(exq, key=exq.get)
    excess = exq[top]
    print(f"  mean-matching passed the linear test on {len(mm)} of {n_tot} cases.")
    print(f"  strongest nonlinear probe: {real:.3f} on the attribute, "
          f"{rand:.3f} on a random partition")
    print(f"  per-probe excess over random: " +
          ", ".join(f"{q} {exq[q]:+.3f}" for q in ("linear", "mlp", "gbm")))
    print(f"  largest excess: {excess:+.3f} ({top})")
    saturated = [q for q in ("linear", "mlp", "gbm")
                 if mm[q].mean() > 0.99 and mm["rnd_" + q].mean() > 0.99]
    if saturated:
        print(f"  NOTE {', '.join(saturated)} saturates at 1.000 on BOTH arms:")
        print("       that probe is reading the edit, not the attribute, and its")
        print("       excess of zero says nothing about what survives.")
    print()
    if real <= args.probe_max:
        print("  RECOVERY IS GENUINE. No nonlinear probe beats the threshold.")
        print("  Report the table as it stands; this answers both reviewers.")
    elif args.no_control:
        print("  Control skipped, so this number cannot be interpreted. Re-run")
        print("  without --no-control.")
    elif excess <= 0.05:
        print("  THE OPERATOR IS SELF-REVEALING. The probe scores just as high on")
        print("  a partition that means nothing, so it is reading the edit, not")
        print("  the attribute. Mean-matching subtracts a different constant from")
        print("  each group; wherever the representation holds exact ties, and a")
        print("  ReLU layer holds them in bulk, those ties become two distinct")
        print("  constants and a tree separates them perfectly.")
        print()
        print("  Do NOT report this as 'the attribute survives erasure'. The")
        print("  correct claim is sharper: an intervention that must know the")
        print("  protected attribute in order to apply itself leaves group")
        print("  membership exactly recoverable, and the 0.55 linear criterion")
        print("  certifies the opposite. Report the mechanism and this control,")
        print("  which is what makes the claim defensible.")
    else:
        print(f"  GENUINE RESIDUAL RECOVERABILITY, excess {excess:+.3f} via the")
        print(f"  {top} probe. Real attribute information survives beyond anything")
        print("  the edit itself reveals, so scope every claim to linear")
        print("  guardedness: these operators hide the attribute from a linear")
        print("  probe rather than removing it.")
        if saturated:
            print()
            print("  BOTH findings hold and they are separate. Report them as two:")
            print("  (1) the operator is self-revealing, shown by the saturated")
            print("      probe scoring identically on a meaningless partition;")
            print("  (2) a genuine residual remains, shown by the excess above.")
            print("  Conflating them would overstate (2) and lose (1) entirely.")

    print()
    print("  by attribute, mean-matching only:")
    for a, t in mm.groupby("attr"):
        print(f"    {a:14s} n={len(t):2d}  attribute {t.best_nl.mean():.3f}"
              f"  random {t.rnd_best_nl.mean():.3f}"
              f"  excess {t.excess.mean():+.3f}")

    print()
    print("  RECOVERED verdict, cases where a family other than meandiff passed:")
    nz = ok[ok.family != "meandiff"]
    n_lin = nz.case.nunique()
    n_nl = sum(1 for _, t in nz.groupby("case")
               if int((t.best_nl <= args.probe_max).sum()) > 0)
    n_ex = sum(1 for _, t in nz.groupby("case")
               if int((t.excess <= 0.05).sum()) > 0)
    print(f"    linear criterion, as published          {n_lin} cases")
    print(f"    nonlinear criterion, uncontrolled       {n_nl} cases")
    print(f"    nonlinear, excess over random <= 0.05   {n_ex} cases")
    print("    The third row is the defensible one.")

print()
print("=" * W)
print("WHAT THE COUNT BECOMES")
print("=" * W)
md_safe = set(ok[ok.family == "meandiff"].case)
passing = set(ok.case)
n_md, n_rec = len(md_safe), len(passing - md_safe)
print(f"  linear criterion: mean-difference safe {n_md}, recovered {n_rec}, "
      f"irrecoverable {d.case.nunique() - len(passing)}")
print("  These three numbers must equal 09_routing_eval.py on the same tag"
      + (" (its unweighted file)." if args.unweighted else "."))
best = ok.groupby("case").excess.min()
surv = sorted(best[best <= 0.05].index.tolist())
print(f"  controlled nonlinear criterion: {len(surv)} of {d.case.nunique()} cases have an "
      f"excess of at most 0.05 under a family that passes")
print(f"    {', '.join(surv) if surv else '(none)'}")
for a, t in ok.groupby("attr"):
    bt = t.groupby("case").excess.min()
    print(f"    {a:12s} {int((bt <= 0.05).sum())} of {t.case.nunique()} passing cases survive"
          f"   (mean-matching excess {t[t.family == 'meanmatch'].excess.mean():+.3f})")

SUFFIX = "_unweighted" if args.unweighted else ""
d.to_csv(f"{args.out}/meanmatch_probe{SUFFIX}{TAG}.csv", index=False)
print(f"\nwrote {args.out}/meanmatch_probe{SUFFIX}{TAG}.csv")
