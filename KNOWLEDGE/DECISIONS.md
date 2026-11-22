# Decisions and what is frozen

Why things are done the way they are. A decision here is binding until it is changed *here*.

---

## Lineage comes from `helpers/lineage.py`

Nothing may test `cell_type_final == "CD4"`. The old label is an else-branch
(`CAVEATS.md#the-cd4-label-was-a-default`); the benign CD4 arm built from it is only 45.8%
conventional CD4 in skin — 16.1% CD8, 10.9% Treg, 7.7% NK/ILC, 14.9% unresolved.

- `helpers/lineage.py` is the only reader of `data/atlas_joint/nonmal_T_labels_v1.parquet`.
- `cluster_malig_helpers.read_obs()` applies the overlay by default, so every job that reads obs
  through it is correct without doing anything.
- `malignancy.cd4_level()` takes its CD4 set from `lineage.cd4_set(tissue)` and **raises** if no
  cell matches — a silent empty arm is the failure mode this guards against.

**The benign arm is `CD4_conv + CD4_naive` in skin and `CD4_conv` alone in blood.** That
asymmetry is inherited deliberately: v1's blood clean-up dropped naive clusters (they are
transcriptionally far from the malignant cells and would inflate every contrast) and skin kept
them. `Mixed`, `LowQC` and `Suspect_malignant` are never in any arm.

**Malignant cells keep the old `CD4` label.** They are absent from the re-annotation by
construction, so `"CD4"` stays in the CD4 set — remove it and the malignant arm vanishes.

---

## `mal_cluster` is frozen

`mal_cluster` is the headline malignancy call (since 2026-10-22). It is **not** rebuilt on the new
lineage labels, and the jobs that produce it pass `read_obs(..., lineage=False)`.

The reason is circularity: `cluster_malig_helpers.CELL_TYPES` and `MIN_CD4` (skin 0.8, blood 0.5)
define a benign cluster by its CD4 *composition*, and `jobs/run_cluster_malig.py` masks on
`cell_type_T == "CD4"`. The re-annotation was itself derived from `mal_cluster` — it labelled the
cells `mal_cluster` called non-malignant. Feeding the new labels back in would move the call the
labels depend on.

Revisiting this is a real analysis, not a refactor: it needs the clustering re-run, the model
re-fit, and the manual calls re-reviewed. It is listed as pending in `ANALYSES.md`.

## The four malignancy calls are never OR-ed

There are exactly four and they measure different things: `mal_tcr` (ALICE TCR clonality — a
sequence fact), `mal_cnv` (inferCNV arm consensus), `mal_knn` (label propagation on the MrVI
latent), `mal_cluster` (per-sample Leiden, one call per cluster — **the headline**). Each ships
with a `_defined` mask, because a cell with no TCR is *untested*, not benign.

The retired pipeline carried `combined_malignant = tcr | (cnv & ok)`. It was dropped because a
disjunction hides which measurement fired, and the CNV call derives from smoothed regional
expression — partly the same signal a downstream expression analysis then attributes to it.
Keeping them apart is what allows "holds on the TCR call as well", a claim about replication.

**Two floors that are inputs, not results.** `mal_knn` has a ~23% healthy false-positive rate
(it was trained on CTCL anchors and has never seen a healthy repertoire) — report it next to any
figure split on it. `mal_cluster` has a 0% healthy rate because nb26b assigns healthy samples
benign *by rule*. Neither number is evidence about specificity.

---

## Cohort rules for the malignant-vs-benign contrast

Applied in every arm-based analysis, from `helpers/target_cohort.py`:

- **Unit is the patient**, not the sample: `clin_person_key` → `patient_key` → `donor`. `donor` is
  dataset-prefixed and never collides; `real_donor` does (15 values span studies).
- **geskin2026 is dropped entirely** from blood — 100% ex vivo culture (`CAVEATS.md`).
- **Non-CD4 clones are dropped**: `clin_phenotype` in (CD8, gd, DN) or `clin_dx` in
  (CD8_aggr_CTCL, pcAECTCL). This donor-level exclusion was the workaround for the lineage leak
  and is **kept** even though the re-annotation addresses the cell-level part of it — the clone
  itself is still not CD4 in those donors.
- **Off-spectrum diagnoses are dropped**: LyP, SPP, LPP.
- **Eligibility is ≥50 cells in each arm**, per patient per tissue.

Healthy-donor CD4 is its own arm (`CD4_HC`), never folded into `CD4_reactive`: those two mean
different things — benign cells beside a tumour, versus the T cells of a person with no tumour.

---

## The output contract

Every analysis ends with `reports/<name>/for_analysis/` = `INDEX.md` + plain CSVs + `report.md`,
written by `helpers/bundle.py`. No figures, no binaries. The bundle is meant to be pasted whole
into another model, so it has to carry its own orientation: what the columns mean, what the row
counts are, and what would mislead a reader who does not know this project.

Figures are opt-in (`--figures`), because the bundle is for numbers and the user makes their own.

---

## Notebooks are not an execution surface

Analyses live in `jobs/` as headless scripts with bsub wrappers. Notebooks are archived under
`notebooks/_archive/` and are never run. The reason is reproducibility and handover: a notebook's
result depends on the order its cells were executed in, and an agent cannot re-run one without a
display. `HISTORY.md` records what each archived notebook did and which artifact it produced.

---

## Deliberately not changed by the lineage migration

| Thing | Why |
|---|---|
| `mal_cluster` and its parquets | circular — see above |
| `skin_T_cnv_helpers` CD8-based CNV reference | `mal_cnv` is a sensitivity arm computed on old labels; changing the reference would change the call without re-validating it |
| `jobs/run_alice_pgen.py` | feeds ALICE → `mal_cluster`; also uses a retired v1 vocabulary (`CD4_Cm`, `CD4_Th2`, …) |
| `jobs/run_ccc52_mvb.py`, `jobs/run_ccc52_qc.py` | pinned to v1 arms, with a banner saying so — they reproduce published numbers, and the QC script's whole purpose is to characterise the *old* arm |
| v1 CCC (`ccc_helpers`, `ccc_data`, `ccc_utils`, `ccc_data_sub`) | legacy generation, kept alive only by `degradome_helpers`; not migrated |
