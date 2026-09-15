# Paper A Sprint Kit — MLBD 2026 (IEEE Big Data 2026 Special Session)

**Paper:** Knowing When Not to Intervene: Coherence-Gated Bias Projection for Clinical Risk Models
**Authors:** Osafo, Doku, Rawat (Howard University)
**Venue:** 11th Special Session on Machine Learning on Big Data (MLBD 2026), IEEE Big Data 2026, Phoenix, Dec 14–17
**Submission:** Sept 30, 2026 (CyberChair, track "SpecialSession on Machine Learning on Big Data") — SUBMIT SEPT 29
**Format:** 6-page short paper, IEEE 2-column. Notification Oct 31. Camera-ready Nov 14.

---

## Day-by-day (Sept 15 → Sept 29)

| Days | Task | Output |
|---|---|---|
| 15–16 | Run `01_nhanes_download_prep.py` (your machine or Leapfrog). Inspect cohort table. | `nhanes_cohort.parquet`, Table 1 stats |
| 17–18 | Run `02_train_models.py`. Check AUROC sanity (LogReg ~0.80+, MLP similar for diabetes target). | trained models + activation dumps |
| 19–22 | Run `03_gbp_audit.py`. Core results: κ per (model, attribute); gated vs ungated vs INLP/LEACE fairness–utility table; abstained-pairs analysis. | `results/*.csv`, 2 figures |
| 20–24 | Write into `mlbd2026_skeleton.tex` (intro/methods are pre-drafted; drop numbers in). | full draft |
| 25–26 | Rawat + Doku review pass. | comments resolved |
| 27–28 | Polish, PDF check (no author info? MLBD inherits Big Data single-blind → authors ON the paper), CyberChair account + upload dry run. | camera-quality PDF |
| **29** | **Submit.** | done |

## Experiment design (frozen — don't expand mid-sprint)

**Task:** predict diabetes (self-reported diagnosis OR HbA1c ≥ 6.5%) from clinical features.
**Data:** NHANES 2017–2018 + Aug 2021–Aug 2023 cycles, adults 18+, MEC-examined.
**Features (NO protected attributes as inputs):** BMI, waist circumference, systolic BP, total cholesterol, HDL, triglycerides (log), smoking status, age *excluded from features* — age is an audited attribute.
  - Rationale: bias directions are learned representations of race/sex/age that leak in via correlates; that's exactly what GBP audits.
**Audited attributes:** race/ethnicity (Black vs White non-Hispanic; Hispanic vs White NH), sex, age (≥60 vs <45).
**Models:** logistic regression (baseline, no audit), 2-layer MLP (64,32) — audit layer = penultimate (32-d).
**GBP pipeline per (model, attribute):**
  1. v_bias = normalized mean-difference of penultimate activations between groups
  2. v_task = normalized direction of final-layer weights
  3. κ = |cos(v_bias, v_task)|
  4. Policy: κ < 0.30 → FULL projection; 0.30–0.50 → PARTIAL (s = 1 − κ); κ > 0.50 → PROTECT (abstain, no intervention; flag for review)
**Baselines:** ungated full erasure (s=1 always), INLP, LEACE (`concept-erasure` pip package).
**Metrics:** AUROC, ECE (10-bin), demographic-parity gap, equalized-odds gap, per-group AUROC & ECE. Report deltas vs unedited model.
**Headline analyses:**
  - Table 2: fairness gains vs utility cost, gated vs ungated vs INLP/LEACE
  - Table 3: κ per (model, attribute) with policy decision — the PROTECT rows ARE the contribution
  - Fig 2: subgroup calibration under full erasure vs gated (show where blind erasure hurts)
**Seeds:** 5 seeds, report mean ± sd. Fixed splits 60/20/20.

## Survey-weight stance (state in Limitations)
We model the NHANES sample, not population estimates; survey weights not applied to model training (standard for prediction-methods papers). One sentence, cite precedent.

## Repo
github.com/nanasafo/gbp-clinical  (code public at submission; NHANES is public data — full reproducibility)
