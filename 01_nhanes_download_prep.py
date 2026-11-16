"""
01_nhanes_download_prep.py - Paper A (journal merge revision, rev 3)

REV 3 RECOVERS THE Aug2021-Aug2023 CYCLE.
NHANES renamed triglycerides from LBXTR to LBXTLG in _L. The original script
asks for LBXTR, so the column merged in as all-NaN and complete-case filtering
then deleted every _L row without a word: the published 4,500-row cohort is
2015-2018 only, not the three cycles it is described as. ALIASES fixes the
rename; a per-cycle column audit now stops the run if it happens again.

Downloads NHANES 2015-2016 (_I), 2017-2018 (_J) and Aug2021-Aug2023 (_L),
builds the analysis cohort, writes nhanes_cohort.parquet.

REV 2 FIXES THE WEIGHT VARIABLE.
Requiring LBXTR (triglycerides) restricts the cohort to the FASTING SUBSAMPLE:
morning-session participants who fasted 8.5 to 24 hours. NHANES documents
WTSAF2YR as the correct weight for that subsample; WTMEC2YR does not account
for morning-session selection or fasting status. Rev 1 reported 12.9% using
WTMEC2YR on a fasting-subsample cohort, which is the wrong weight.

CHANGES vs the MLBD submission:
  1. Pulls SDMVSTRA / SDMVPSU / SDDSRVYR from DEMO, WTMEC2YR from DEMO, and
     WTSAF2YR from TRIGLY.
  2. Selects the weight from the cohort definition:
       default   -> fasting subsample, WTSAF2YR  (log_tg is a feature)
       --no-tg   -> full MEC sample,   WTMEC2YR  (log_tg dropped)
  3. Builds WT_POOLED = <weight> / n_cycles for the pooled estimate.
  4. Drops zero or missing weight, reporting zero and missing separately.
  5. Reports prevalence under every applicable weight so the gap is visible.
  6. --drop-L excludes Aug2021-Aug2023: removes both the pooling gap NCHS
     cautions about and the auscultatory/oscillometric BP change.

Run:  python 01_nhanes_download_prep.py            # main: fasting subsample
      python 01_nhanes_download_prep.py --no-tg    # sensitivity: full MEC
      python 01_nhanes_download_prep.py --drop-L   # sensitivity: no _L cycle
Deps: pip install pandas numpy requests pyarrow
"""
import io, os, sys, argparse, requests, pandas as pd, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--drop-L", action="store_true",
                help="exclude the Aug2021-Aug2023 cycle (sensitivity analysis)")
ap.add_argument("--no-tg", action="store_true",
                help="drop log_tg; keeps the full MEC sample and uses WTMEC2YR")
ap.add_argument("--task", default="diabetes", choices=["diabetes", "hypertension"],
                help="clinical outcome to model")
ap.add_argument("--allow-missing", action="store_true",
                help="continue when a requested column is absent (default: stop)")
ap.add_argument("--out", default="data", help="output directory")
args = ap.parse_args()

OUT = args.out
os.makedirs(OUT, exist_ok=True)
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"

CYCLES = {
    "I": f"{BASE}/2015/DataFiles",   # 2015-2016
    "J": f"{BASE}/2017/DataFiles",   # 2017-2018
    "L": f"{BASE}/2021/DataFiles",   # Aug 2021 - Aug 2023
}
if args.drop_L:
    CYCLES.pop("L")
    print("SENSITIVITY RUN: excluding cycle _L (Aug2021-Aug2023)")

N_CYCLES = len(CYCLES)
TASK = args.task
SUFFIX = (("_htn" if TASK == "hypertension" else "")
          + ("_noTG" if args.no_tg else "") + ("_noL" if args.drop_L else ""))

# The cohort definition picks the weight, not the other way round.
USE_TG = not args.no_tg
WEIGHT = "WTMEC2YR" if args.no_tg else "WTSAF2YR"
WEIGHT_NOTE = ("full MEC sample (log_tg dropped)" if args.no_tg
               else "fasting subsample (log_tg requires LBXTR)")

FILES = ["DEMO", "BMX", "TCHOL", "HDL", "TRIGLY", "GHB", "DIQ", "SMQ", "BPQ"]
BP = {"I": "BPX", "J": "BPX", "L": "BPXO"}   # oscillometric from _L onward

DEMO_DESIGN = ["WTMEC2YR", "SDMVSTRA", "SDMVPSU", "SDDSRVYR"]

# NHANES renames variables between cycles. Map alias -> canonical name.
# LBXTLG is the Aug2021-Aug2023 name for triglycerides. Units verified
# identical to LBXTR: median 85 mg/dL in _I, 96 mg/dL in _L, IQRs overlapping.
# Without this the _L triglyceride column arrives empty and complete-case
# filtering then silently deletes the entire cycle.
ALIASES = {"LBXTLG": "LBXTR"}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/126.0 Safari/537.36",
    "Accept": "*/*",
}


def fetch_xpt(url):
    print("  ", url)
    r = requests.get(url, timeout=120, headers=HEADERS)
    r.raise_for_status()
    head = r.content[:80]
    if not head.startswith(b"HEADER RECORD"):
        snippet = head.decode("utf-8", errors="replace")
        raise RuntimeError(f"Not an XPT file from {url} - server sent: {snippet!r}")
    out = pd.read_sas(io.BytesIO(r.content), format="xport", encoding=None)
    hit = {a: c for a, c in ALIASES.items()
           if a in out.columns and c not in out.columns}
    if hit:
        print("     renamed:", ", ".join(f"{a} -> {c}" for a, c in hit.items()))
        out = out.rename(columns=hit)
    return out


frames = []
for suf, base in CYCLES.items():
    print(f"Cycle _{suf}")
    dfs = {stem: fetch_xpt(f"{base}/{stem}_{suf}.XPT") for stem in FILES}
    dfs["BP"] = fetch_xpt(f"{base}/{BP[suf]}_{suf}.XPT")

    demo = dfs["DEMO"]
    missing = [v for v in DEMO_DESIGN if v not in demo.columns]
    if missing:
        raise RuntimeError(
            f"cycle _{suf}: DEMO is missing {missing}. Check the codebook at "
            f"{base.replace('/DataFiles','')} - weight names change occasionally."
        )
    if USE_TG and "WTSAF2YR" not in dfs["TRIGLY"].columns:
        raise RuntimeError(
            f"cycle _{suf}: TRIGLY is missing WTSAF2YR, the fasting subsample "
            f"weight. Check {base.replace('/DataFiles','')}/TRIGLY_{suf}.htm"
        )

    d = demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3"] + DEMO_DESIGN]

    WANT = [
        ("BMX", ["SEQN", "BMXBMI", "BMXWAIST"]),
        ("TCHOL", ["SEQN", "LBXTC"]),
        ("HDL", ["SEQN", "LBDHDD"]),
        ("TRIGLY", ["SEQN", "LBXTR", "WTSAF2YR"]),
        ("GHB", ["SEQN", "LBXGH"]),
        ("DIQ", ["SEQN", "DIQ010"]),
        ("SMQ", ["SEQN", "SMQ020", "SMQ040"]),
        ("BPQ", ["SEQN", "BPQ050A"]),
    ]

    # A renamed variable merges in as all-NaN, and complete-case filtering then
    # deletes the whole cycle without a word. Fail loudly instead.
    gone = {n: [c for c in cols if c not in dfs[n].columns] for n, cols in WANT}
    gone = {n: c for n, c in gone.items() if c}
    if gone:
        msg = "; ".join(f"{n} is missing {c}" for n, c in gone.items())
        if args.allow_missing:
            print(f"   WARNING cycle _{suf}: {msg} - continuing, these are NaN")
        else:
            raise RuntimeError(
                f"cycle _{suf}: {msg}. Check the codebook at "
                f"{base.replace('/DataFiles','')}; if the variable was renamed, "
                f"add it to ALIASES. To proceed anyway, pass --allow-missing."
            )

    for name, cols in WANT:
        d = d.merge(dfs[name][[c for c in cols if c in dfs[name].columns]],
                    on="SEQN", how="left")

    bp = dfs["BP"]
    sys_cols = [c for c in ["BPXSY1", "BPXSY2", "BPXSY3",
                            "BPXOSY1", "BPXOSY2", "BPXOSY3"] if c in bp.columns]
    bp["SBP"] = bp[sys_cols].mean(axis=1)
    # NHANES records 0 for an inaudible or absent diastolic reading. Averaging
    # those as zeros would invent hypotension, so treat them as missing.
    dia_cols = [c for c in ["BPXDI1", "BPXDI2", "BPXDI3",
                            "BPXODI1", "BPXODI2", "BPXODI3"] if c in bp.columns]
    bp["DBP"] = bp[dia_cols].replace(0, np.nan).mean(axis=1) if dia_cols else np.nan
    d = d.merge(bp[["SEQN", "SBP", "DBP"]], on="SEQN", how="left")
    d["cycle"] = suf
    d["bp_device"] = "oscillometric" if BP[suf] == "BPXO" else "auscultatory"
    frames.append(d)

df = pd.concat(frames, ignore_index=True)
print("raw merged:", df.shape)

# ---- weight diagnostics -------------------------------------------------
print()
print(f"WEIGHT IN USE: {WEIGHT}   ({WEIGHT_NOTE})")
for wv in ["WTMEC2YR", "WTSAF2YR"]:
    if wv in df.columns:
        n_zero = int((df[wv] == 0).sum())
        n_na = int(df[wv].isna().sum())
        print(f"  {wv}: {n_zero} zero, {n_na} missing, "
              f"{len(df) - n_zero - n_na} usable of {len(df)}")

before = len(df)
df = df[df[WEIGHT].notna() & (df[WEIGHT] > 0)]
print(f"dropped {before - len(df)} rows with zero/missing {WEIGHT}")

# Pooled weight: divide the 2-year weight by the number of pooled cycles.
# NOTE: NCHS advises caution combining Aug2021-Aug2023 with earlier cycles
# (1.5-year gap). Use --drop-L for the sensitivity analysis.
df["WT_POOLED"] = df[WEIGHT] / N_CYCLES

# ---- cohort rules -------------------------------------------------------
df = df[df.RIDAGEYR >= 18]
# Outcome. Written to "outcome" and also under the task name, so cohorts built
# by either task are self-describing and downstream scripts need one column.
if TASK == "diabetes":
    df["outcome"] = ((df.DIQ010 == 1) | (df.LBXGH >= 6.5)).astype(int)
else:
    # ACC/AHA 2017 thresholds, plus current antihypertensive use. Treated
    # patients often read normal, so omitting medication would misclassify
    # exactly the people whose hypertension is established.
    df["outcome"] = ((df.SBP >= 130) | (df.DBP >= 80)
                     | (df.BPQ050A == 1)).astype(int)
df[TASK] = df["outcome"]
df["diabetes"] = df["outcome"]   # back-compatible alias for older scripts
df["sex_female"] = (df.RIAGENDR == 2).astype(int)
# RIDRETH3: 1 MexAm, 2 OtherHisp, 3 White NH, 4 Black NH, 6 Asian NH, 7 Other
df["race"] = df.RIDRETH3.map({1: "hispanic", 2: "hispanic", 3: "white_nh",
                              4: "black_nh", 6: "asian_nh", 7: "other"})
df["smoker_current"] = ((df.SMQ020 == 1) & (df.SMQ040.isin([1, 2]))).astype(int)
df["log_tg"] = np.log(df.LBXTR.clip(lower=10))
df["age"] = df.RIDAGEYR

FEATURES = ["BMXBMI", "BMXWAIST", "SBP", "LBXTC", "LBDHDD", "smoker_current"]
if USE_TG:
    FEATURES.insert(5, "log_tg")
if TASK == "hypertension":
    # SBP and DBP define the label. Leaving either in the feature set would
    # put the outcome in the inputs, which is the criticism this second task
    # exists to answer, not to repeat.
    FEATURES = [f for f in FEATURES if f not in ("SBP", "DBP")]

DESIGN_KEEP = ["WT_POOLED", WEIGHT, "SDMVSTRA", "SDMVPSU"]
need = FEATURES + ["outcome", "diabetes", "sex_female", "race", "age"] + DESIGN_KEEP
if TASK == "hypertension":
    need = need + ["DBP"]   # kept for audit, not used as a feature

n_before_cc = len(df)
cohort = df.dropna(subset=[c for c in need if c != "race"]).copy()
cohort = cohort[cohort.race.notna()]

# ---- what the weights change -------------------------------------------
w = cohort["WT_POOLED"].values
unw = cohort.outcome.mean()
wtd = np.average(cohort.outcome.values, weights=w)

print()
print("=" * 66)
print(f"cohort N                      : {len(cohort)}  "
      f"(complete-case from {n_before_cc} adults)")
print(f"task                          : {TASK}")
print(f"features                      : {', '.join(FEATURES)}")
print(f"weight                        : {WEIGHT}  [{WEIGHT_NOTE}]")
print("-" * 66)
print(f"{TASK} prevalence UNWEIGHTED : {unw:.4f}  ({100*unw:.1f}%)")
print(f"{TASK} prevalence {WEIGHT:<10}: {wtd:.4f}  ({100*wtd:.1f}%)")
# Show the wrong-weight figure too, so the correction is auditable.
if USE_TG and "WTMEC2YR" in cohort.columns:
    mec = cohort["WTMEC2YR"].where(cohort["WTMEC2YR"] > 0)
    ok = mec.notna().values
    if ok.any():
        mec_wtd = np.average(cohort.outcome.values[ok], weights=mec.values[ok])
        print(f"  (same cohort, WTMEC2YR      : {mec_wtd:.4f}  "
              f"({100*mec_wtd:.1f}%)  <- wrong weight, for comparison)")
print(f"absolute difference            : {abs(wtd-unw):.4f} "
      f"({100*abs(wtd-unw)/unw:.1f}% relative)")
print("=" * 66)
if TASK == "diabetes":
    print("NCHS benchmark: total diabetes among US adults 20+ was 15.0% crude in")
    print("NHANES 2013-2016, on a BROADER definition (fasting glucose >=126 OR")
    print("HbA1c >=6.5). This cohort uses self-report OR HbA1c only, so it should")
    print("sit somewhat below 15%.")
else:
    print("Benchmark: hypertension under the ACC/AHA 2017 definition runs near")
    print("45% of US adults. A figure far from that points at the blood-pressure")
    print("averaging or the medication variable, so check those before the model.")
print()
print("design: n_strata =", cohort.SDMVSTRA.nunique(),
      "| n_PSU_within_strata =", cohort.groupby("SDMVSTRA").SDMVPSU.nunique().mean().round(2))
print(f"complete-case retention: {100*len(cohort)/n_before_cc:.1f}% of adults; "
      "weights no longer sum to the US population. Report this, do not rescale.")
print()
print(cohort.race.value_counts())

path = f"{OUT}/nhanes_cohort{SUFFIX}.parquet"
cohort[["SEQN", "cycle", "bp_device"] + need].to_parquet(path, index=False)
print(f"\nwrote {path}")

# ---- Table 1, weighted and unweighted ----------------------------------
COLS = FEATURES + ["outcome", "age"]
t1 = cohort.groupby("race")[COLS].mean().round(2)
t1["n"] = cohort.race.value_counts()
t1.to_csv(f"{OUT}/table1_by_race{SUFFIX}.csv")

def wmean(g, col):
    return np.average(g[col].values, weights=g["WT_POOLED"].values)

t1w = cohort.groupby("race").apply(
    lambda g: pd.Series({c: wmean(g, c) for c in COLS}),
    include_groups=False).round(2)
t1w["n"] = cohort.race.value_counts()
t1w.to_csv(f"{OUT}/table1_by_race_weighted{SUFFIX}.csv")

print("\nUNWEIGHTED group means:"); print(t1)
print(f"\nWEIGHTED group means ({WEIGHT}):"); print(t1w)
print(f"\nwrote {OUT}/table1_by_race{SUFFIX}.csv and "
      f"{OUT}/table1_by_race_weighted{SUFFIX}.csv")
