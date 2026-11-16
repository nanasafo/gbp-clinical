"""
16_routing_threshold_validation.py - does the 2x2 routing rule generalise?

THE REVIEWER POINT THIS ANSWERS
`10_baserate_diag.py` lines 46 to 54 sweep the impact and gap thresholds over
every case, pick the pair with the highest accuracy, and report that accuracy.
The rule is therefore selected on the data it is scored on, and the published
number is an upper bound rather than an estimate. Three reviewers flagged the
circularity.

This script leaves `10` alone and re-estimates the same rule honestly:

  A  LOSO        leave-one-seed-out over the NHANES cases. Fit the thresholds
                 on four seeds, apply unchanged to the fifth.
  B  SPLIT-HALF  repeated random halves over all cases, fit on one, score the
                 other. Reports the mean held-out accuracy and its spread.
  C  CROSS-BED   fit on synthetic, score on NHANES, and the reverse. The
                 strongest test here: the synthetic cases are the ones with
                 known ground truth, so a rule that transfers from there to
                 clinical cases has earned something.

It also quantifies the boundary instability Reviewer 1 raised, where verdicts
flip between seeds near gap ~ 0.11, and prices a declared defer band, which
turns that weakness into a design choice.

The rule, exactly as `10` defines it:
    impact < I_thr                 -> meandiff_ok
    else baserate_gap < G_thr      -> recovered
    else                           -> irrecoverable

Run:  python 16_routing_threshold_validation.py
      python 16_routing_threshold_validation.py --infile results3/baserate_validation.csv
Deps: pandas, numpy
"""
import argparse, itertools, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--infile", default="results3/baserate_validation.csv")
ap.add_argument("--repeats", type=int, default=200, help="split-half repeats")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
W = 76

d = pd.read_csv(args.infile)
for c in ["testbed", "impact", "baserate_gap", "verdict", "case"]:
    if c not in d.columns:
        raise RuntimeError(f"{args.infile} has no '{c}'. Run 10_baserate_diag.py first.")

# The impact grid must extend well below any value it selects. The submitted
# grid started at 0.3 and selected 0.3; the first correction started at 0.2 and
# selected 0.2. Both were boundary solutions, and a threshold chosen at the edge
# of its search range says only that the range was too narrow. The script now
# warns whenever a fitted threshold lands on either edge of its grid.
I_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8]
G_GRID = [0.05, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30]


def predict(frame, it, gt):
    return np.where(frame.impact < it, "meandiff_ok",
                    np.where(frame.baserate_gap < gt, "recovered", "irrecoverable"))


def accuracy(frame, it, gt):
    return float((predict(frame, it, gt) == frame.verdict).mean())


def fit(frame):
    """Best (I_thr, G_thr) on `frame`. Ties break toward the smaller pair, which
    is the conservative choice: it sends fewer cases to the aggressive edit."""
    best = (-1.0, None)
    for it, gt in itertools.product(I_GRID, G_GRID):
        a = accuracy(frame, it, gt)
        if a > best[0] + 1e-12:
            best = (a, (it, gt))
    return best[1], best[0]


def tied(frame, acc):
    """Every grid pair that reaches the best accuracy on `frame`."""
    return [(it, gt) for it, gt in itertools.product(I_GRID, G_GRID)
            if abs(accuracy(frame, it, gt) - acc) < 1e-12]


def edge_warning(it, gt, label):
    for v, grid, name in ((it, I_GRID, "impact"), (gt, G_GRID, "gap")):
        if v in (grid[0], grid[-1]):
            print(f"  WARNING {label}: the {name} threshold {v} is on the edge of its"
                  f" grid. Extend the grid before reporting it.")


print("=" * W)
print("IN-SAMPLE, THE PUBLISHED PROCEDURE")
print("=" * W)
(it0, gt0), acc0 = fit(d)
print(f"  best rule over all {len(d)} cases: impact < {it0}, gap < {gt0}")
print(f"  in-sample accuracy {acc0:.1%}   <- selected on the data it is scored on")
for tb, g in d.groupby("testbed"):
    print(f"    {tb:10s} {accuracy(g, it0, gt0):.1%}  (n={len(g)})")
edge_warning(it0, gt0, "in-sample")
print("  impact threshold profile at the chosen gap threshold:")
print("    " + "   ".join(f"{it:.2f}: {accuracy(d, it, gt0):.1%}" for it in I_GRID[:7]))

# ---------------------------------------------------------------- A: LOSO
nh = d[d.testbed == "nhanes"].copy()
if len(nh):
    nh["seed"] = nh.case.str.rsplit("_s", n=1).str[-1]
    nh = nh[nh.seed.str.isdigit()]
    nh["seed"] = nh.seed.astype(int)
print()
print("=" * W)
print("A. LEAVE-ONE-SEED-OUT, NHANES CASES")
print("=" * W)
if len(nh) and nh.seed.nunique() > 1:
    corr = tot = 0
    picks = []
    for s in sorted(nh.seed.unique()):
        dev, hold = nh[nh.seed != s], nh[nh.seed == s]
        (it, gt), _ = fit(dev)
        picks.append((it, gt))
        hit = int((predict(hold, it, gt) == hold.verdict).sum())
        corr += hit; tot += len(hold)
        print(f"  seed {s} held out: rule impact<{it}, gap<{gt} -> "
              f"{hit} of {len(hold)} correct")
    print(f"\n  held-out accuracy {corr}/{tot} = {corr/tot:.1%}")
    print(f"  in-sample on the same cases {accuracy(nh, it0, gt0):.1%}")
    uniq = sorted(set(picks))
    print(f"  thresholds chosen across folds: {uniq}")
    if len(uniq) > 1:
        print("  the rule is NOT stable across folds; report the range, not one pair")
else:
    print("  not enough NHANES seeds to run this")

# ---------------------------------------------------- B: repeated split-half
print()
print("=" * W)
print(f"B. REPEATED SPLIT-HALF, ALL CASES, {args.repeats} REPEATS")
print("=" * W)
accs, chosen = [], []
idx = np.arange(len(d))
for _ in range(args.repeats):
    perm = rng.permutation(idx)
    a, b = perm[:len(idx) // 2], perm[len(idx) // 2:]
    dev, hold = d.iloc[a], d.iloc[b]
    if dev.verdict.nunique() < 2:
        continue
    (it, gt), _ = fit(dev)
    chosen.append((it, gt))
    accs.append(accuracy(hold, it, gt))
accs = np.array(accs)
if len(accs):
    print(f"  held-out accuracy {accs.mean():.1%} "
          f"(SD {accs.std(ddof=1):.1%}, 5th-95th pct "
          f"{np.percentile(accs,5):.1%} to {np.percentile(accs,95):.1%})")
    print(f"  optimism from selecting on the evaluation set: "
          f"{acc0 - accs.mean():+.1%}")
    cnt = pd.Series(chosen).value_counts()
    print(f"  most frequently chosen rule: impact<{cnt.index[0][0]}, "
          f"gap<{cnt.index[0][1]} ({cnt.iloc[0]}/{len(chosen)} splits)")
    print("  all rules chosen: " + ", ".join(
        f"({k[0]}, {k[1]}) {v}" for k, v in cnt.head(6).items()))

# ---------------------------------------------------------- C: cross-testbed
print()
print("=" * W)
print("C. CROSS-TESTBED TRANSFER")
print("=" * W)
beds = d.testbed.unique()
if len(beds) >= 2:
    for src, dst in [("synthetic", "nhanes"), ("nhanes", "synthetic")]:
        if src not in beds or dst not in beds:
            continue
        S, D = d[d.testbed == src], d[d.testbed == dst]
        if S.verdict.nunique() < 2:
            print(f"  fit on {src}: only one verdict class present, skipped")
            continue
        (it, gt), a_in = fit(S)
        a_out = accuracy(D, it, gt)
        print(f"  fit on {src} (n={len(S)}): impact<{it}, gap<{gt}, "
              f"in-sample {a_in:.1%}")
        print(f"    applied to {dst} (n={len(D)}): {a_out:.1%}")
        ties = tied(S, a_in)
        if len(ties) > 1:
            outs = sorted({round(accuracy(D, i, g), 3) for i, g in ties})
            its = sorted({i for i, _ in ties})
            print(f"    NOTE {len(ties)} rules tie on {src}; impact thresholds {its}."
                  f" Transfer across the tied rules: {', '.join(f'{o:.1%}' for o in outs)}."
                  f" The thresholds are not identified from {src} alone.")
    print("\n  Transfer from synthetic to clinical is the meaningful direction:")
    print("  the synthetic cases carry known ground truth, so a rule fitted")
    print("  there and holding on real data has earned its thresholds.")
else:
    print("  only one testbed present, skipped")

# ------------------------------------------------- boundary instability
print()
print("=" * W)
print("BOUNDARY INSTABILITY AND A DECLARED DEFER BAND")
print("=" * W)
if len(nh) and nh.seed.nunique() > 1:
    flip = []
    for attr, g in nh.groupby(nh.case.str.rsplit("_s", n=1).str[0]):
        if g.verdict.nunique() > 1:
            flip.append((attr, dict(g.verdict.value_counts()),
                         g.baserate_gap.mean(), g.impact.min(), g.impact.max()))
    if flip:
        print("  attributes whose verdict is not stable across seeds:")
        for a, vc, gap, lo, hi in flip:
            print(f"    {a:14s} {vc}  gap={gap:.3f}  impact {lo:.2f} to {hi:.2f}")
    else:
        print("  no attribute flips verdict across seeds")

else:
    print("  not enough NHANES seeds to assess stability")

# Price the band on whichever axis the errors actually sit on, rather than
# assuming it is the gap. Misreading this is easy: the published rule's errors
# are on the impact axis even though the instability was reported as a gap
# effect.
err = d[predict(d, it0, gt0) != d.verdict]
print()
print(f"  errors at the fitted rule (impact<{it0}, gap<{gt0}): {len(err)} of {len(d)}")
if len(err):
    di = (err.impact - it0).abs().median()
    dg = (err.baserate_gap - gt0).abs().median()
    print(f"  median distance to the impact threshold {di:.3f}, "
          f"to the gap threshold {dg:.3f}")
    axis = "impact" if di <= dg else "gap"
    print(f"  the errors cluster on the {axis.upper()} axis, so band that one")
    print()
    print(f"  {'axis':>8s} {'band':>8s} {'deferred':>11s} {'accuracy outside':>18s}"
          f" {'false-safe left':>16s}")
    for ax, thr, col in [("impact", it0, d.impact), ("gap", gt0, d.baserate_gap)]:
        for half in [0.01, 0.02, 0.05, 0.10]:
            inb = col.between(thr - half, thr + half)
            out = d[~inb]
            if not len(out):
                continue
            pr = predict(out, it0, gt0)
            acc = float((pr == out.verdict).mean())
            fs = int(((pr == "meandiff_ok") & (out.verdict != "meandiff_ok")).sum())
            print(f"  {ax:>8s} +/-{half:5.2f} {int(inb.sum()):7d}/{len(d):<4d}"
                  f" {acc:17.1%} {fs:16d}")
    print()
    print("  'false-safe left' counts cases still routed to the aggressive edit")
    print("  that should not have been. That is the error worth pricing: it is")
    print("  the one that destroys clinical signal. A band that defers a handful")
    print("  of cases and drives it to zero is a design choice you can defend.")

print()
print("=" * W)
print("HOW TO WRITE THIS UP")
print("=" * W)
print("  Report the held-out number, not the swept one, and say which protocol")
print("  produced it. If held-out accuracy is close to in-sample, the rule was")
print("  not overfitted and you have strengthened the claim by checking. If it")
print("  drops, report the drop: a rule with honest accuracy beats a rule with")
print("  a flattering one, and the reviewers already know how to compute both.")
