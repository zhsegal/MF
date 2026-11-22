# Malignant vs benign CD4 differential cell-cell communication — analysis bundle

Everything here is plain text. Start with `ccc_malignant_vs_benign_cd4_report.md`; the CSVs are
the same numbers in machine-readable form. Generated 2026-11-22 17:51 by
`jobs/run_ccc52_mvb.py`.

## What the analysis is

Paired pseudobulk differential expression of ligand and receptor genes between a patient's
**malignant (M)** and **benign (B)** CD4 T cells, run separately in skin and blood. The logic: in
one donor both arms face the same partner cells, so in any multiplicative LR score the partner
term cancels and the differential interaction reduces to a paired DE on the CD4 side. **The
CD4-side statistics are the inference. The partner assignment is descriptive only** — partner
cells act purely as a gate (is the cognate gene expressed in that cell type), with no test and no
ambient correction.

## Files

| file | size | rows | contents |
|---|---|---|---|
| ccc_malignant_vs_benign_cd4_report.md | 64 KB |  | the full write-up: rationale, data, methods, results per tissue, prior axes, limitations, deviations |
| de_lr_genes_skin.csv | 83 KB | 372 | every tested ligand-receptor gene in SKIN |
| de_lr_genes_blood.csv | 68 KB | 303 | every tested ligand-receptor gene in BLOOD |
| interactions_skin.csv | 32 KB | 174 | SKIN: one row per CD4 gene x cognate gene x partner type that passes the partner gate |
| interactions_blood.csv | 30 KB | 163 | BLOOD: the same |
| prior_axes.csv | 7 KB | 24 | the 12 pre-specified axes plus the TOX positive control, both tissues, reported regardless of significance |
| cell_counts_donor_arm.csv | 11 KB | 162 | cells per donor x tissue x arm, with the eligibility flag |
| cohort_excluded.csv | 2 KB | 37 | every donor removed by a cohort rule, with the reason |
| partner_types.csv | 1 KB | 32 | partner cell types and their pooled cell counts per tissue |
| arm_composition.csv | 3 KB | 59 | v2 arms only: old label x new label x malignancy x arm, per tissue -- exactly which cells the re-annotation moved in and out of the benign arm |
| annotation_shift.csv | 166 KB | 723 | v2 arms only: every LR gene and every gene significant in either run, with its v1 and v2 log2FC side by side and a stable/moved verdict. READ THIS BEFORE reusing any number from reports/ccc52_mvb/ |
| units.csv | 12 KB | 202 | the eligible donor x arm pseudobulk index |
| lr_resource_1422.csv | 19 KB | 1,422 | the filtered LR resource: 1,422 interactions with their support |
| de_all_genes_skin.csv | 2.4 MB | 11,887 | SKIN full-transcriptome fit (no padj outside the LR family) -- LARGE, look genes up rather than reading it whole |
| de_all_genes_blood.csv | 2.6 MB | 12,577 | BLOOD full-transcriptome fit -- LARGE, same caveat |

## How to read the gene tables

- `log2FC` is the **shrunken** estimate from the `~ donor + arm` model, M relative to B; a
  positive value means higher in the malignant arm. `log2FC_raw` is the unshrunken Wald estimate.
- `padj` is BH **over the genes in the filtered LR resource only**, computed per tissue per fit.
  A gene outside that family has a `pvalue` and a blank `padj` — it was still fitted, it is just
  not in the multiple-testing family. Do not compare a blank padj to a significant one.
- `significant` already encodes all three call criteria together: padj < 0.05,
  |log2FC| >= 1.0, and detection in >= 10% of the cells of the
  higher arm. Use that column rather than re-thresholding.
- `pct_M` / `pct_B` are detection fractions (cells with a non-zero count) pooled over eligible
  donors — not expression levels.
- `donors_concordant` / `donors_eligible` is the per-donor sign check: how many donors' own
  CPM-based log2FC has the same sign as the model estimate. It is the main guard against a result
  driven by one or two patients, and it is worth quoting next to any claim.
- `fit` says which model the gene came from. Where a tissue mixes chemistries, genes detected in
  every chemistry are fitted on all donors and the rest on the non-probe-panel donors only.

## Things that will mislead you if you do not know them

- **Significance in one tissue only is not evidence of tissue specificity.** The compartments
  differ in donor count, cells per arm and benign-arm composition, so they differ in power.
  Section 7 of the report gives the effect-size correlation, which is the comparison that counts.
- **The benign arm is the re-annotated lineage** (`nonmal_T_labels_v1` via
  `helpers/lineage.py`): `CD4_conv` + `CD4_naive` in skin, `CD4_conv` in blood. It is NOT
  `cell_type_final == "CD4"`, which is an else-branch only ~46% conventional CD4 in skin.
  `arm_composition.csv` is the old-label x new-label crosstab. The older bundle in
  `reports/ccc52_mvb/` used the v1 arm; do not mix numbers between the two for any gene that
  separates CD4 from CD8/NK/Treg.
- **The two tissues still differ in one way**: skin keeps naive CD4 in the benign arm and blood
  does not. That asymmetry is deliberate and inherited from v1.
- **`label_defining_gene == True` means the contrast for that gene is partly circular.** The
  benign arm is selected on a marker panel (CD4, CD40LG, CD8A/B, FOXP3, CCR7, SELL, NKG7, GZMK,
  KLRF1, TRDC, ...), so those genes separate the arms by construction. They are reported, not
  dropped, because hiding them would hide the arm's composition — but do not quote one as a
  finding about the malignant clone.
- **Blood CD8 and NK partner rows are mixtures.** The upstream annotation called CD8 on `CD8A`,
  which NK cells also express; only 12 clusters were corrected.
- **Blood is far more imbalanced than skin** — the benign arm is small in Sezary patients. Check
  `cell_counts_donor_arm.csv` before reading much into a blood effect size.
- No ambient-RNA correction was applied anywhere, by design. Partner detection fractions are
  inflated by contamination from neighbouring cell types.

- `de_all_genes_skin.csv`, `de_all_genes_blood.csv` are several MB; look genes up in them rather than reading them whole.
