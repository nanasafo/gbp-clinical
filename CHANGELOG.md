# Changelog

## v2.0.0 (November 2026): journal version

Code and results for the IEEE Access submission, which combines the two
conference papers into one article. Every number in the manuscript is produced
by the scripts in this release.

### Corrections

- **Survey weighting.** The cohort is a complex multistage probability sample and
  was previously treated as a simple random sample. Because the cohort requires
  triglycerides it is the fasting subsample, so all estimates now use `WTSAF2YR`
  pooled over three cycles (`WT_POOLED`). Weighted diabetes prevalence is 12.5%,
  against 17.4% unweighted. Intervals come from a stratified cluster bootstrap
  over PSUs within strata (`01`, `02`, `03`, `06`, `09`, `10`, `15`, `17`).
- **Lost cycle recovered.** NHANES renamed triglycerides from `LBXTR` to `LBXTLG`
  in August 2021 to August 2023. The old pipeline requested the old name, the
  column merged in empty, and complete-case filtering silently removed the whole
  cycle. `01` now maps the alias and stops if a requested column is missing.
  The cohort is N = 7,487 across all three cycles, not N = 4,500 over two.
- **Tagged inputs.** `03` and `09` read untagged activation files from the old
  cohort regardless of what `01` and `02` had produced. Both now take `--tag`.

### Revised analyses

- **Held-out gate threshold** (`14`): fitted on development seeds and applied
  unchanged to held-out seeds. Impact is correct on 20 of 20 decisions, the angle
  on 9 of 20.
- **Routing thresholds out of sample** (`16`): the grid now starts at 0.05, so the
  selected impact threshold (0.2) is an interior optimum. Held-out accuracy 88.7%
  over 200 half-splits; leave-one-seed-out 20 of 20 on the clinical cases.
- **Nonlinear re-test of recoveries** (`15`): mean-matching re-tested against a
  random-partition control. Only 4 of the 12 linear recoveries survive.
- **Exact-ties demonstration** (`18`): why a group-conditional edit is
  self-revealing in a ReLU representation.
- **Linear-erasure scoping** (`13`), **model justification** (`12`), and the
  weighted **cohort table** (`17`).
- **Convergence check**: `09 --max-iter 5000` changes no verdict.
- **LEACE** is implemented in closed form in numpy in `03` and `09`, removing the
  `concept-erasure` and torch dependency from those scripts.
- **Manuscript assets** (`19`): the gate figure and three LaTeX tables are built
  from the CSVs.

### Moved

- `04_entangled_variant.py` and `05_fig_dose_response.py` belong to the
  conference version and are in `legacy/`. `05` is superseded by `19`.
- Outputs from the conference version are in `legacy/results_submitted/`.

## v1.0.0 (2026): conference submission

Initial release of the two companion conference papers.
