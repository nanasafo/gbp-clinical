# gbp-clinical

Code and results for **"From Gating to Routing: Pre-Intervention Diagnostics for Selective Bias Intervention in Clinical Risk Models"**, submitted to *IEEE Access* (November 2026). The article reports Studies 3 and 4 of the doctoral dissertation *Selective Post-Hoc Intervention in AI Systems: Safety-Gated Editing, Reliability Diagnostics, and Abstention as a First-Class Output* (Howard University).

Nana Yaw A. Osafo, Ronald Doku, John Kwagyan, Danda B. Rawat, William M. Southerland.

## What the code shows

Projection-based debiasing removes a demographic direction from a model's internal representation. Earlier safety gates, including GBP-Audit (IEEE CogMI 2025), decide whether to erase from the angle between the bias direction and the task direction. On diabetes risk models trained on three pooled NHANES cycles (N = 7,487, survey-weighted prevalence 12.5%):

- **Angle-only coherence** correlates with the AUROC loss from linear erasure at Spearman ρ = 0.28 (p = 0.24) and approves erasure in all 20 attribute-seed cases, including losses of up to 0.25 AUROC.
- **Impact-weighted coherence**, a closed-form pre-intervention statistic, correlates at ρ = 0.96. A threshold fitted on development seeds is correct on 20 of 20 held-out decisions, against 9 of 20 for the angle.
- **Routing.** Across five linear intervention families, 12 of the 17 clinical refusals meet a linear erasure criterion, all through mean-matching translation. A rule on impact and the outcome base-rate gap predicts the three-way verdict with 88.7% held-out accuracy (59% majority baseline), and refuses all 30 synthetic cases built to be entangled.
- **Nonlinear re-test.** Against a random-partition control, only 4 of the 12 recoveries survive a nonlinear probe. Mean-matching conditions on the attribute in order to apply itself, so it leaves group membership exactly recoverable through ReLU ties (`18_ties_demo.py`).

All results are survey-weighted with the NHANES fasting-subsample weight, with stratified cluster-bootstrap intervals that resample PSUs within strata.

## Reproduce

NHANES is public. The full pipeline runs on a laptop.

```bash
pip install -r requirements.txt

# Cohort and models
python 01_nhanes_download_prep.py                 # fasting cohort, WTSAF2YR, N = 7,487
python 01_nhanes_download_prep.py --no-tg         # full-examination cohort (sensitivity)
python 02_train_models.py --cohort data/nhanes_cohort.parquet      --tag _fast
python 02_train_models.py --cohort data/nhanes_cohort_noTG.parquet --tag _mec

# Gate (Study 3)
python 06_impact_gate.py --tag _fast              # angle vs impact, cluster bootstrap
python 06_impact_gate.py --tag _mec
python 14_threshold_validation.py --tag _fast     # held-out gate threshold (20 of 20)
python 14_threshold_validation.py --tag _mec
python 03_gbp_audit.py --tag _fast                # utility and fairness table
python 13_nonlinear_probe.py --tag _fast          # erasure is linear erasure
python 17_cohort_table.py                         # weighted cohort table

# Routing (Study 4)
python 07_synthetic_testbed.py                    # 75 synthetic cases with known ground truth
python 09_routing_eval.py --tag _fast             # five families, verdict per case
python 10_baserate_diag.py                        # survey-weighted base-rate gaps
python 16_routing_threshold_validation.py         # out-of-sample routing thresholds
python 11_fig_verdict_map.py                      # verdict map figure
python 15_meanmatch_probe.py --tag _fast          # nonlinear re-test with random-partition control
python 15_meanmatch_probe.py --tag _mec
python 18_ties_demo.py                            # self-contained exact-ties demonstration

# Reviewer checks and manuscript assets
python 12_model_justification.py --tag _mec       # why an MLP
python 09_routing_eval.py --tag _fast --max-iter 5000   # convergence check, never overwrites cases.csv
python 00_reanalysis.py --infile results/impact_gate_fast.csv   # variance and both correlation framings
python 19_paper_assets.py                         # fig_gate.pdf and LaTeX tables, from the CSVs
```

`08_families.py` is a library (diagnostics and the five intervention families) imported by `09`. The `data/`, `data_syn/` and `models/` directories are not committed; scripts 01, 02 and 07 rebuild them.

## Where each manuscript result comes from

| Manuscript item | Script | Output |
|---|---|---|
| Cohort table, prevalence 12.5% | `01`, `17` | `data/table1_by_race_weighted.csv`, `results/cohort_table.csv` and `.tex` |
| Gate statistics table, Fig. (gate) | `06`, `19` | `results/impact_gate_fast.csv`, `build/tab_gate.tex`, `build/fig_gate.pdf` |
| Held-out gate threshold, 20 of 20 vs 9 of 20 | `14` | printed summary (reads `results/impact_gate_fast.csv`) |
| Utility and fairness table | `03`, `19` | `results/fairness_utility_raw_fast.csv`, `build/tab_fairness.tex` |
| Linear vs nonlinear probes after erasure | `13` | `results/nonlinear_probe_fast.csv` |
| Family recovery table | `09`, `19` | `results3/cases.csv`, `build/tab_family.tex` |
| Base-rate gaps, routing rule accuracy | `10`, `16` | `results3/baserate_validation.csv` |
| Verdict map | `11` | `results3/fig_verdict_map.pdf` |
| Nonlinear re-test, 4 of 12 | `15` | `results3/meanmatch_probe_fast.csv`, `results3/meanmatch_probe_mec.csv` |
| Exact-ties artifact | `18` | printed summary |

## Data

NHANES 2015-2016, 2017-2018, and August 2021 to August 2023 (public, de-identified; CDC/NCHS). Adults 18 and over with a complete fasting lipid panel, blood pressure, anthropometry, HbA1c, smoking and diabetes questionnaire data. Outcome: diabetes (self-reported diagnosis or HbA1c of at least 6.5%). Race and ethnicity, sex, and age are **excluded from the model inputs** and used only as audited attributes. Because the cohort requires triglycerides it is the fasting subsample, so estimates use `WTSAF2YR` pooled over the three cycles; `--no-tg` gives the full-examination cohort with `WTMEC2YR`.

## Changes from the conference submission

See [CHANGELOG.md](CHANGELOG.md). In short: the cohort is now survey-weighted; a variable rename (`LBXTR` to `LBXTLG`) that had silently removed the 2021-2023 cycle is fixed, so N is 7,487 rather than 4,500; thresholds are validated out of sample; and the recovery claim is re-tested against a nonlinear criterion. The superseded outputs are kept in `legacy/results_submitted/` for transparency.

## Related work in this line

- Doku, Osafo, Rawat. *GBP-Audit: AI Bias Intervention with Built-In Safety-First Guardrails.* IEEE CogMI 2025, pp. 399-408. doi:10.1109/CogMI67134.2025.00052
- Osafo, Doku, Rawat. *When Not to Edit: Reliability Diagnostics for Activation-Space Interventions in Language Models.* IEEE IRI 2026, Seattle, pp. 80-85. doi:10.1109/IRI69576.2026.00026

## Citation

```bibtex
@unpublished{osafo2026gating,
  title  = {From Gating to Routing: Pre-Intervention Diagnostics for Selective Bias Intervention in Clinical Risk Models},
  author = {Osafo, Nana Yaw A. and Doku, Ronald and Kwagyan, John and Rawat, Danda B. and Southerland, William M.},
  note   = {Submitted to IEEE Access},
  year   = {2026}
}
```

## Funding

Supported in part by the U.S. National Science Foundation and by the National Institute on Minority Health and Health Disparities of the National Institutes of Health (2U54MD007597).

## License

MIT (code). NHANES data are subject to CDC/NCHS data use terms.
