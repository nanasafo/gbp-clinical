# gbp-clinical

Code for two companion studies on **safe post-hoc intervention in clinical risk models**, submitted to the Machine Learning on Big Data special session at IEEE Big Data 2026.

Nana Yaw A. Osafo, Ronald Doku, John Kwagyan, William M. Southerland, Danda B. Rawat — Howard University.

## The two papers

**Paper A — "Knowing When Not to Intervene: Impact-Weighted Coherence Gating for Bias Projection in Clinical Risk Models."**
Projection-based debiasing removes a demographic direction from a model's latent space. Prior work (our GBP-Audit, IEEE CogMI 2025) gates this on coherence κ — the angle between the bias and task directions. We show angle-only coherence **under-predicts intervention harm** (Spearman ρ = 0.45), and introduce an **impact-weighted coherence score** — alignment × signal magnitude along the bias direction, normalized by logit scale — which forecasts erasure damage at **ρ = 0.98**. High-impact directions trigger abstention rather than projection.

**Paper 3 — "Beyond Abstention: Routing Recoverable from Irrecoverable Interventions in Post-Hoc Model Editing."**
A gate's refusal admits two readings: intervention is impossible, or the edit family is too crude. We decompose abstention using closed-form diagnostics and route across five intervention families. On clinical models, **13 of 18 refused cases are recoverable** — exclusively by mean-matching translation, never by re-centered projection — while age refuses every family. A two-diagnostic rule (impact × outcome base-rate gap) predicts the three-way verdict with **87.4% accuracy**.

## Reproduce everything

NHANES is public; the full pipeline runs on a laptop in ~20 minutes.

```bash
pip install pandas pyreadstat pyarrow requests torch scikit-learn concept-erasure matplotlib scipy

# Paper A
python 01_nhanes_download_prep.py   # 3 NHANES cycles -> cohort (N=4,500)
python 02_train_models.py           # LogReg + MLP, 5 seeds; dumps activations
python 03_gbp_audit.py              # kappa per (model, attribute); gated vs INLP/LEACE/ungated
python 04_entangled_variant.py      # age-included model: kappa's failure case
python 06_impact_gate.py            # impact-weighted coherence vs kappa as harm forecast
python 05_fig_dose_response.py      # Figure: kappa-vs-impact comparison

# Paper 3 (reuses Paper A's activations)
python 07_synthetic_testbed.py      # 75 cases with ground-truth entanglement control
python 09_routing_eval.py           # diagnostics -> 5 families x 4 severities -> verdict per case
python 10_baserate_diag.py          # base-rate gap diagnostic + 2x2 routing rule accuracy
python 11_fig_verdict_map.py        # Figure: verdict map on the two-diagnostic plane
```

`08_families.py` is a library (diagnostics + intervention families), imported by `09`.
Outputs land in `results/` (Paper A) and `results3/` (Paper 3). The `data/` and `data_syn/` directories are not committed — scripts 01 and 07 rebuild them.

## Data

NHANES cycles 2015–2016, 2017–2018, and Aug 2021–Aug 2023 (public, de-identified; CDC/NCHS). Cohort: adults 18+ with complete fasting lipid panel, blood pressure, anthropometry, HbA1c, smoking, and diabetes questionnaire data. Target: diabetes (self-reported diagnosis or HbA1c ≥ 6.5%). Race/ethnicity, sex, and age are **excluded from model inputs** and used only as audited attributes. We model the NHANES sample; estimates are not survey-weighted population statistics.

## Prior work in this line

- Doku, Osafo, Rawat. *GBP-Audit: AI Bias Intervention with Built-In Safety-First Guardrails.* IEEE CogMI 2025, pp. 399–408. doi:10.1109/CogMI67134.2025.00052
- Osafo, Doku, Rawat. *When Not to Edit: Reliability Diagnostics for Activation-Space Interventions in Language Models.* IEEE IRI 2026, Seattle, pp. 80–85. doi:10.1109/IRI69576.2026.00026

## License

MIT (code). NHANES data are subject to CDC/NCHS data use terms.
