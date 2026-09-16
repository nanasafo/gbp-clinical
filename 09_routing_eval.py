"""
09_routing_eval.py — the Paper 3 experiment.
For every case (NHANES attr x seed + synthetic grid):
  diagnostics -> run all families x severities -> smallest-s success per family
  -> classify case: MEANDIFF-OK / RECOVERED(by which family) / IRRECOVERABLE
Outputs: results3/cases.csv, results3/recovery_summary.csv, results3/routing_validation.csv

Run AFTER 07 (and with Paper A's data/activations_seed*.npz present):
  python 09_routing_eval.py
Deps: everything from Paper A + concept-erasure
"""
import os, glob, numpy as np, pandas as pd, torch
from concept_erasure import LeaceEraser
from importlib.machinery import SourceFileLoader
fam = SourceFileLoader("fam", "08_families.py").load_module()

os.makedirs("results3", exist_ok=True)
SEVERITIES = [0.25, 0.5, 0.75, 1.0]

def load_nhanes_cases():
    groups = pd.read_parquet("data/groups.parquet")
    ATTRS = {"sex_female": (groups.sex_female.values==1),
             "race_black": (groups.race.values=="black_nh"),
             "race_hisp":  (groups.race.values=="hispanic"),
             "age_60plus": (groups.age.values>=60)}
    REF = {"sex_female": None, "age_60plus": (groups.age.values<45),
           "race_black": (groups.race.values=="white_nh"),
           "race_hisp":  (groups.race.values=="white_nh")}
    for f in sorted(glob.glob("data/activations_seed*.npz")):
        z = np.load(f); seed = int(f.split("seed")[1].split(".")[0])
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=1000).fit(z["H"][z["idx_train"]], z["y"][z["idx_train"]])
        w, b = lr.coef_.squeeze(), float(lr.intercept_[0])
        for attr, g in ATTRS.items():
            ref = REF[attr] if REF[attr] is not None else ~g
            mask = g | ref                     # restrict to the two groups compared
            yield {"testbed":"nhanes", "case": f"{attr}_s{seed}", "truth": "",
                   "H": z["H"][mask], "y": z["y"][mask], "g": g[mask],
                   "tr": _remap(z["idx_train"], mask), "te": _remap(z["idx_test"], mask),
                   "w": w, "b": b}

def _remap(idx, mask):
    pos = -np.ones(len(mask), dtype=int); pos[mask] = np.arange(mask.sum())
    keep = idx[mask[idx]]; return pos[keep]

def load_syn_cases():
    man = pd.read_csv("data_syn/manifest.csv").set_index("case")
    for f in sorted(glob.glob("data_syn/case_*.npz")):
        z = np.load(f); name = os.path.basename(f)[:-4]
        yield {"testbed":"synthetic", "case": name, "truth": man.loc[name, "truth"],
               "H": z["H"], "y": z["y"], "g": z["g"].astype(bool),
               "tr": z["idx_train"], "te": z["idx_test"],
               "w": z["w_task"], "b": float(z["b_task"])}

rows = []
for case in list(load_nhanes_cases()) + list(load_syn_cases()):
    H, y, g, tr, te, w, b = case["H"], case["y"], case["g"], case["tr"], case["te"], case["w"], case["b"]
    d = fam.diagnostics(H[tr], g[tr], w)
    v = d.pop("v")
    m1, m0 = H[tr][g[tr]].mean(0), H[tr][~g[tr]].mean(0)
    ctx = {"c_mid": (m1+m0)/2, "m1": m1, "m0": m0, "g_all": g, "tr": tr,
           "leace_eraser": LeaceEraser.fit(torch.tensor(H[tr]).float(),
                                           torch.tensor(g[tr].astype(np.float32)).unsqueeze(1))}
    fam_result = {}
    for name, editfn in fam.FAMILIES.items():
        best_s = None; best_m = None
        for s in SEVERITIES:
            Hm = editfn(H, v, s, ctx)
            m = fam.evaluate_case(Hm, H, y, g, w, b, te, tr)
            if fam.success(m): best_s, best_m = s, m; break   # smallest safe severity
        fam_result[name] = (best_s, best_m)
    verdict = ("meandiff_ok" if fam_result["meandiff"][0] is not None else
               ("recovered" if any(v[0] is not None for k, v in fam_result.items() if k != "meandiff")
                else "irrecoverable"))
    recovered_by = ",".join(k for k, v in fam_result.items() if k != "meandiff" and v[0] is not None)
    rows.append({"testbed": case["testbed"], "case": case["case"], "truth": case["truth"],
                 **{k: round(x, 3) for k, x in d.items()},
                 "verdict": verdict, "recovered_by": recovered_by,
                 **{f"s_{k}": v[0] for k, v in fam_result.items()}})
    print(rows[-1]["testbed"], rows[-1]["case"], "->", verdict, f"(I={d['impact']:.2f} A={d['asymmetry']:.2f} S={d['stability']:.2f})")

df = pd.DataFrame(rows); df.to_csv("results3/cases.csv", index=False)

summ = df.groupby(["testbed","verdict"]).size().unstack(fill_value=0)
summ.to_csv("results3/recovery_summary.csv"); print("\n", summ)

syn = df[df.testbed=="synthetic"]
val = pd.crosstab(syn.truth, syn.verdict)
val.to_csv("results3/routing_validation.csv"); print("\nSynthetic ground-truth vs verdict:\n", val)
print("\nwrote results3/{cases,recovery_summary,routing_validation}.csv")
