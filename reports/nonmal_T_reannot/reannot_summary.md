# Re-annotation of the non-malignant T/NK cells

`jobs/run_nonmal_T_reannot.py`, 2026-11-22. Labels: `data/atlas_joint/nonmal_T_labels_v1.parquet` (new file); nothing else was written. scvi-tools 1.3.3, torch 2.9.1, scanpy 1.11.5, python 3.12.12; full parameters in `_versions.json` and the script's CONFIG.

## Method actually used

- Cells: every T-compartment cell (`cell_type_final` in ['CD4', 'CD4_Treg', 'CD8', 'NK']) whose `mal_cluster` is not True -- all donors, healthy controls and ineligible donors included. No cell in the atlas has a null malignancy call, so the "flag the nulls" branch never fires.
- Genes: a gene is dropped when a dataset of that tissue has zero counts for it while >= 10 detections were expected from the tissue-wide rate (this is what catches the Flex panel); TR/IG V-D-J, MT-, ribosomal and haemoglobin genes excluded; candidates must be detected in >= 1% of cells. HVG = top 2000 by binned dispersion of log-normalised counts, union the marker panel.
  - skin: 9,029 candidates (9,945 unmeasured) -> 2019 genes, median detection 0.053, +19 markers forced in; blood: 10,091 candidates (8,206 unmeasured) -> 2026 genes, median detection 0.076, +26 markers forced in.
- Embedding: scVI (n_latent 30, 2 layers, NB, <= 60 epochs, early stopping), batch = sample_id in skin / donor in blood. Leiden 2.0 on 15 neighbours; one round of sub-clustering at 0.5 for Mixed clusters >= 200 cells. Seed 0.
- Labels: every level is relative -- a marker's detection is divided by the highest any cluster of >= 200 cells in that tissue reaches, and a marker whose best cluster stays under 0.1 detection is dropped as uninformative there. Each lineage has a gate it must pass (NK needs a low TCR fraction and low CD3E; MAIT needs SLC4A10/TRAV1-2; CD8 needs CD8A/CD8B; the CD4 labels need CD4/CD40LG and a LOW CD8A/B) and then wins on score by >= 0.1, else `Mixed`. SLC4A10 and TRAV1-2 are uninformative in skin, so skin gets no MAIT label. `Suspect_malignant` overrides all: one donor holds >= 80% of a cluster and one TRBV >= 50% of its TCR-bearing cells.

## Cells per label

| tissue | CD4_Treg | CD4_conv | CD4_naive | CD8 | LowQC | MAIT | Mixed | NK_ILC | Prolif | Suspect_malignant | gdT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| blood | 21,233 | 166,619 | 37,829 | 44,739 | 1,141 | 5,580 | 7,214 | 22,827 | 4,334 | 643 | 6,732 |
| skin | 41,851 | 68,503 | 19,850 | 68,677 | 2,354 | 0 | 31,968 | 16,581 | 4,452 | 1,744 | 1,317 |

## Arms

| tissue | donors eligible v1 | eligible new | median benign v1 | median benign new |
|---|---|---|---|---|
| blood | 38 | 45 | 470 | 624 |
| skin | 71 | 56 | 660 | 298 |

## Section-4 targets (reported, not asserted)

**Target 1 cannot be met as stated**: the malignant arm is a clone, detecting these five genes in 0.0000-0.011 of its cells, so any arm of real polyclonal T cells is many times it; the informative column is `change`. **Target 2 is missed** -- every value stays below -1 -- but all ten move the right way in skin. Detection is the across-donor median with both arms on the SAME donors (eligible under both); lfc is the median per-donor log2FC, (a) v1 arm and (b) new arm on those same donors, (c) new arm on every donor eligible under it. The tcr/auto split of (c) is in `reannot_sensitivity.csv`.

| tissue | gene | det v1 | det new | change | det malig | lfc (a) | lfc (b) | lfc (c) |
|---|---|---|---|---|---|---|---|---|
| skin | CD8A | 0.1405 | 0.0639 | 0.45x | 0.0109 | -4.53 | -3.45 | -3.45 |
| skin | CD8B | 0.0548 | 0.0309 | 0.56x | 0.0005 | -3.15 | -1.67 | -1.67 |
| skin | TRDC | 0.0326 | 0.0061 | 0.19x | 0.0011 | -3.43 | -1.18 | -1.18 |
| skin | KLRD1 | 0.0845 | 0.0153 | 0.18x | 0.0035 | -4.81 | -1.45 | -1.45 |
| skin | FOXP3 | 0.0935 | 0.0445 | 0.48x | 0.0072 | -3.71 | -2.04 | -2.04 |
| blood | CD8A | 0.0402 | 0.0301 | 0.75x | 0.0003 | -2.64 | -2.57 | -2.63 |
| blood | CD8B | 0.0342 | 0.0236 | 0.69x | 0.0003 | -3.16 | -2.11 | -2.85 |
| blood | TRDC | 0.0083 | 0.0111 | 1.33x | 0.0000 | -2.12 | -2.30 | -2.30 |
| blood | KLRD1 | 0.0120 | 0.0215 | 1.78x | 0.0005 | -0.92 | -2.52 | -2.53 |
| blood | FOXP3 | 0.0277 | 0.0092 | 0.33x | 0.0028 | -1.72 | -0.17 | 0.00 |

Positive controls, lfc (a) -> (b): skin TOX +1.11->+1.36, skin CD70 +1.15->+1.96, blood TOX +2.88->+3.36, blood CD70 +2.68->+3.42.

Per-cell co-expression inside each arm, all cells (`reannot_coexpression.csv`):

| tissue | arm | cells | CD8A/B+ | FOXP3+ |
|---|---|---|---|---|
| skin | new_benign | 88,353 | 0.0888 | 0.0430 |
| skin | v1_benign | 180,978 | 0.1989 | 0.0803 |
| blood | new_benign | 166,619 | 0.0355 | 0.0272 |
| blood | v1_benign | 148,752 | 0.1414 | 0.0705 |

## Unsure about

- The new benign arm is CD4_conv + CD4_naive in skin and CD4_conv alone in blood; the brief's wording also admits CD4_conv alone in both, which would drop skin naive cells.
- `Suspect_malignant` is a heuristic on donor and TRBV dominance, not a malignancy call; it does not replace `mal_cluster` and those cells are simply kept out of the benign arm.
- Blood TRDC and KLRD1 detection RISES in the new arm (1.3x, 1.8x) while CD8A, CD8B and FOXP3 fall. That is composition, not contamination: the new blood arm excludes 37,829 naive CD4 cells, which are the KLRD1-lowest cells in it, so removing them raises what remains. The per-cell CD8A/B+ fraction still falls 4x, from 0.141 to 0.036.
- The CD8 and NK_ILC references in `reannot_label_reference.csv` come from this re-annotation, so they are no longer the mixtures the old labels were, but blood NK is small and its separation from CD8 rests on KLRF1/TYROBP and the absence of a TCR.
