"""
14_threshold_validation.py - does the gate threshold generalise?

THE PROBLEM THIS FIXES
The gate fires below a threshold you chose after looking at the results. As
written, the paper reports how well a threshold separates the data it was
picked on, which is not a finding. A reviewer will say so.

METHOD
The gate's job: given a detected bias direction, predict whether erasing it is
SAFE, meaning observed AUROC damage below a tolerance d_c.

  1 fit the threshold on development seeds only
  2 apply it, unchanged, to held-out seeds
  3 report counts, not percentages

Two protocols:
  SPLIT    seeds 0-2 develop, seeds 3-4 held out. 4 attributes x 2 seeds =
           8 held-out decisions.
  LOSO     leave-one-seed-out, 5 folds. 20 held-out decisions from the same
           data, so it is the more informative of the two. Note the folds are
           re-splits of one cohort, not independent samples, so treat 20 as an
           upper bound on the evidence.

Both impact and kappa are thresholded by the identical procedure, which makes
this the out-of-sample version of your central claim. If the impact threshold
transfers and the kappa threshold does not, that is a much stronger statement
than a correlation, and it is immune to the effective-n objection.

Counts are small by construction. Report them as counts; a percentage with a
decimal point on 8 decisions reads as false precision.

Run:  python 14_threshold_validation.py --tag _fast
      python 14_threshold_validation.py --tag _mec --dc 0.02
      python 14_threshold_validation.py --tag _fast --sweep
Deps: pip install pandas numpy
"""
import argparse, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="_fast")
ap.add_argument("--dc", type=float, default=0.02,
                help="damage tolerance: below this, erasure counts as safe")
ap.add_argument("--sweep", action="store_true",
                help="repeat across a range of d_c to show sensitivity")
args = ap.parse_args()

TAG = args.tag
W = 74
LABEL = {"race_hisp": "Hispanic", "race_black": "Black non-Hispanic",
         "sex_female": "Sex (female)", "age_60plus": "Age >= 60"}

d = pd.read_csv(f"results/impact_gate{TAG}.csv")
for c in ["seed", "attr", "kappa", "impact", "auroc_damage"]:
    if c not in d.columns:
        raise RuntimeError(f"results/impact_gate{TAG}.csv has no '{c}'. "
                           f"Run 06_impact_gate.py first.")
SEEDS = sorted(d.seed.unique())


def pick_threshold(dev, col, dc):
    """Threshold on `col` that best separates safe from unsafe on dev rows.

    Candidates are midpoints between consecutive observed values, so the
    threshold is never one of the data points itself. Ties break toward the
    smaller threshold, which is the conservative choice: it fires the gate
    less often.
    """
    safe = (dev.auroc_damage < dc).values
    v = np.sort(dev[col].unique())
    if len(v) < 2 or len(np.unique(safe)) < 2:
        return np.nan
    cands = (v[:-1] + v[1:]) / 2
    best, best_t = -1, np.nan
    for t in cands:
        pred = (dev[col].values < t)          # below threshold => predicted safe
        tp = (pred & safe).sum(); tn = (~pred & ~safe).sum()
        fp = (pred & ~safe).sum(); fn = (~pred & safe).sum()
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        bal = (sens + spec) / 2
        if bal > best:
            best, best_t = bal, t
    return best_t


def evaluate(dev, hold, col, dc):
    t = pick_threshold(dev, col, dc)
    if np.isnan(t):
        return None
    safe = (hold.auroc_damage < dc).values
    pred = (hold[col].values < t)
    return {"threshold": t,
            "correct": int((pred == safe).sum()), "n": len(hold),
            "false_safe": int((pred & ~safe).sum()),
            "false_unsafe": int((~pred & safe).sum())}


def protocol(col, dc, mode):
    if mode == "split":
        dev = d[d.seed.isin(SEEDS[:3])]
        hold = d[d.seed.isin(SEEDS[3:])]
        r = evaluate(dev, hold, col, dc)
        return ([r], r) if r else ([], None)
    folds = []
    for s in SEEDS:
        r = evaluate(d[d.seed != s], d[d.seed == s], col, dc)
        if r:
            folds.append(r)
    if not folds:
        return [], None
    agg = {"threshold": np.mean([f["threshold"] for f in folds]),
           "correct": sum(f["correct"] for f in folds),
           "n": sum(f["n"] for f in folds),
           "false_safe": sum(f["false_safe"] for f in folds),
           "false_unsafe": sum(f["false_unsafe"] for f in folds)}
    return folds, agg


def report(dc):
    n_safe = int((d.auroc_damage < dc).sum())
    print(f"  damage tolerance d_c = {dc:.3f}  ->  {n_safe} of {len(d)} "
          f"attribute-seed cases are genuinely safe")
    if n_safe in (0, len(d)):
        print("  every case falls on one side of the tolerance, so no threshold")
        print("  can be validated here. Choose a d_c that splits the cases.")
        return
    for mode, name in [("split", "SPLIT  seeds 0-2 develop, 3-4 held out"),
                       ("loso", "LOSO   leave-one-seed-out, 5 folds")]:
        print(f"\n  {name}")
        print(f"    {'gate on':8s} {'threshold':>10s} {'held-out correct':>18s}"
              f" {'false safe':>12s} {'false unsafe':>14s}")
        for col in ["impact", "kappa"]:
            _, agg = protocol(col, dc, mode)
            if not agg:
                print(f"    {col:8s} {'n/a':>10s}   development set has one class only")
                continue
            print(f"    {col:8s} {agg['threshold']:10.3f} "
                  f"{agg['correct']:>11d} of {agg['n']:<4d}"
                  f" {agg['false_safe']:>12d} {agg['false_unsafe']:>14d}")


print("=" * W)
print(f"HELD-OUT THRESHOLD VALIDATION  (cohort {TAG or '(none)'})")
print("=" * W)
print("  'false safe' is the costly error: the gate said erasure was cheap and")
print("  it was not. Weigh it more heavily than the other kind.")
print()
report(args.dc)

if args.sweep:
    print()
    print("=" * W)
    print("SENSITIVITY TO THE DAMAGE TOLERANCE")
    print("=" * W)
    print("  A conclusion that holds only at one hand-picked d_c is not a")
    print("  conclusion. These should look much the same across the range.")
    print()
    print(f"  {'d_c':>6s} {'n safe':>8s} {'impact correct':>16s} {'kappa correct':>15s}")
    for dc in [0.005, 0.01, 0.02, 0.05, 0.10]:
        ns = int((d.auroc_damage < dc).sum())
        if ns in (0, len(d)):
            print(f"  {dc:6.3f} {ns:8d}   all cases on one side, skipped")
            continue
        _, ai = protocol("impact", dc, "loso")
        _, ak = protocol("kappa", dc, "loso")
        si = f"{ai['correct']} of {ai['n']}" if ai else "n/a"
        sk = f"{ak['correct']} of {ak['n']}" if ak else "n/a"
        print(f"  {dc:6.3f} {ns:8d} {si:>16s} {sk:>15s}")

print()
print("=" * W)
print("HOW TO WRITE THIS UP")
print("=" * W)
print("  Report the counts and the protocol, and say plainly that the folds")
print("  are re-splits of one cohort rather than independent samples. A")
print("  threshold that transfers across re-splits is weaker evidence than one")
print("  that transfers across cohorts, and you now have two cohorts to try it")
print("  on: fit on _fast, apply to _mec, and report that too.")
