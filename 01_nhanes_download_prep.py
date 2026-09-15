"""
01_nhanes_download_prep.py — Paper A (MLBD 2026)
Downloads NHANES 2017-2018 (_J) and Aug2021-Aug2023 (_L) files, builds the
analysis cohort, writes nhanes_cohort.parquet.

Run:  python 01_nhanes_download_prep.py
Deps: pip install pandas pyreadstat pyarrow requests
"""
import io, os, requests, pandas as pd, numpy as np

OUT = "data"; os.makedirs(OUT, exist_ok=True)
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"

CYCLES = {
    "I": f"{BASE}/2015/DataFiles",   # 2015-2016 cycle
    "J": f"{BASE}/2017/DataFiles",   # 2017-2018 cycle
    "L": f"{BASE}/2021/DataFiles",   # Aug 2021 - Aug 2023 cycle
}

# component file stems (suffix added per cycle)
FILES = ["DEMO", "BMX", "TCHOL", "HDL", "TRIGLY", "GHB", "DIQ", "SMQ"]
BP = {"I": "BPX", "J": "BPX", "L": "BPXO"}    # BP file differs by cycle (auscultatory vs oscillometric)

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
        raise RuntimeError(
            f"Not an XPT file from {url} — server sent: {snippet!r}"
        )
    return pd.read_sas(io.BytesIO(r.content), format="xport")

frames = []
for suf, base in CYCLES.items():
    print(f"Cycle _{suf}")
    dfs = {stem: fetch_xpt(f"{base}/{stem}_{suf}.XPT") for stem in FILES}
    dfs["BP"] = fetch_xpt(f"{base}/{BP[suf]}_{suf}.XPT")

    d = dfs["DEMO"][["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3"]]
    for name, cols in [
        ("BMX", ["SEQN", "BMXBMI", "BMXWAIST"]),
        ("TCHOL", ["SEQN", "LBXTC"]),
        ("HDL", ["SEQN", "LBDHDD"]),
        ("TRIGLY", ["SEQN", "LBXTR"]),
        ("GHB", ["SEQN", "LBXGH"]),
        ("DIQ", ["SEQN", "DIQ010"]),
        ("SMQ", ["SEQN", "SMQ020", "SMQ040"]),
    ]:
        d = d.merge(dfs[name][[c for c in cols if c in dfs[name].columns]], on="SEQN", how="left")

    bp = dfs["BP"]
    sys_cols = [c for c in ["BPXSY1","BPXSY2","BPXSY3","BPXOSY1","BPXOSY2","BPXOSY3"] if c in bp.columns]
    bp["SBP"] = bp[sys_cols].mean(axis=1)
    d = d.merge(bp[["SEQN","SBP"]], on="SEQN", how="left")
    d["cycle"] = suf
    frames.append(d)

df = pd.concat(frames, ignore_index=True)
print("raw merged:", df.shape)

# ---- cohort rules ----
df = df[df.RIDAGEYR >= 18]
df["diabetes"] = ((df.DIQ010 == 1) | (df.LBXGH >= 6.5)).astype(int)
df["sex_female"] = (df.RIAGENDR == 2).astype(int)
# RIDRETH3: 1 MexAm, 2 OtherHisp, 3 White NH, 4 Black NH, 6 Asian NH, 7 Other
df["race"] = df.RIDRETH3.map({1:"hispanic",2:"hispanic",3:"white_nh",4:"black_nh",6:"asian_nh",7:"other"})
df["smoker_current"] = ((df.SMQ020 == 1) & (df.SMQ040.isin([1,2]))).astype(int)
df["log_tg"] = np.log(df.LBXTR.clip(lower=10))
df["age"] = df.RIDAGEYR

FEATURES = ["BMXBMI","BMXWAIST","SBP","LBXTC","LBDHDD","log_tg","smoker_current"]
need = FEATURES + ["diabetes","sex_female","race","age"]
cohort = df.dropna(subset=[c for c in need if c != "race"]).copy()
cohort = cohort[cohort.race.notna()]
print("cohort:", cohort.shape, "| diabetes prevalence:", round(cohort.diabetes.mean(),3))
print(cohort.race.value_counts())

cohort[["SEQN","cycle"]+need].to_parquet(f"{OUT}/nhanes_cohort.parquet", index=False)
print(f"wrote {OUT}/nhanes_cohort.parquet")

# Table 1 snapshot for the paper
t1 = cohort.groupby("race")[FEATURES+["diabetes","age"]].mean().round(2)
t1["n"] = cohort.race.value_counts()
t1.to_csv(f"{OUT}/table1_by_race.csv")
print(t1)
