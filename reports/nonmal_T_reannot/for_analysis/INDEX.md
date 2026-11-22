# Re-annotation of the non-malignant T/NK cells

Every non-malignant T/NK cell in the atlas got a lineage label from a scVI embedding built without the malignant cells, because the previous label (`cell_type_final`) was an else-branch whose CD4 arm was ~20% CD8 and ~20% Treg in skin.

Generated 2026-11-22 15:29. Lineage annotation: **nonmal_T_labels_v1 (this analysis produced it)**.

## What it found

- 576,188 cells relabelled (257,297 skin, 318,891 blood) across 114 clusters
- benign-arm CD8A/B+ fraction: skin 0.199 -> 0.089, blood 0.141 -> 0.035
- blood ADT confirms independently: CD4_conv 80% CD4-protein+, CD8 71% CD8-protein+, NK_ILC double-negative
- skin has no MAIT label: SLC4A10 and TRAV1-2 never reach 10% detection in any skin cluster, so the marker is uninformative there

## Files

| file | rows | what it holds |
|---|---|---|
| `reannot_summary.md` | - |  |
| `reannot_adt.csv` | 117 | surface protein by label, buus25 donors only, per-donor gates |
| `reannot_cluster_table.csv` | 114 | one row per cluster: label, size, donor/TRBV dominance, TCR fraction, median UMI, marker detection |
| `reannot_coexpression.csv` | 6 | per-cell CD8A/B+ and FOXP3+ fractions inside each arm |
| `reannot_crosstab.csv` | 7 | old cell_type_final and v1 arm against the new label |
| `reannot_donor_arms.csv` | 186 | per donor: arm sizes and eligibility under each definition |
| `reannot_label_reference.csv` | 54 | per tissue x label, pooled and across-donor median detection; includes the malignant arm as a reference row |
| `reannot_sensitivity.csv` | 410 | per gene, median per-donor log2FC under each arm definition; sets a/b/c |

## How to read the columns

- **`reannot_adt.csv`** — surface protein by label, buus25 donors only, per-donor gates
- **`reannot_cluster_table.csv`** — one row per cluster: label, size, donor/TRBV dominance, TCR fraction, median UMI, marker detection
- **`reannot_coexpression.csv`** — per-cell CD8A/B+ and FOXP3+ fractions inside each arm
- **`reannot_crosstab.csv`** — old cell_type_final and v1 arm against the new label
- **`reannot_donor_arms.csv`** — per donor: arm sizes and eligibility under each definition
- **`reannot_label_reference.csv`** — per tissue x label, pooled and across-donor median detection; includes the malignant arm as a reference row
- **`reannot_sensitivity.csv`** — per gene, median per-donor log2FC under each arm definition; sets a/b/c

## Things that will mislead you if you do not know them

- The benign arm is CD4_conv + CD4_naive in **skin** but CD4_conv alone in **blood** -- the same asymmetry v1 had.
- `Mixed`, `LowQC` and `Suspect_malignant` are not an arm. `Suspect_malignant` is a donor/TRBV dominance heuristic, NOT a malignancy call; it does not override `mal_cluster`.
- Malignant cells are absent from this table by construction and keep their old `CD4` label. Code must treat `CD4` as part of the CD4 set -- `helpers/lineage.py` does.
- Blood TRDC and KLRD1 detection RISES in the new arm. That is composition, not contamination: the new blood arm excludes 37,829 naive CD4, which are its KLRD1-lowest cells.
- Section-4 target 1 is unreachable as stated -- the malignant arm detects those five genes in 0.0000-0.011 of its cells, so any polyclonal arm exceeds 2x it.

## Provenance

- **script**: jobs/run_nonmal_T_reannot.py
- **labels**: data/atlas_joint/nonmal_T_labels_v1.parquet
- **versions**: reports/nonmal_T_reannot/_versions.json

