# Lineage migration check

Did the switch to nonmal_T_labels_v1 land correctly, and what did it change about the benign CD4 arm?

Generated 2026-11-22 15:23. Lineage annotation: **nonmal_T_labels_v1**.

## What it found

- 7/7 assertions passed
- skin eligible donors 80 -> 63, blood 50 -> 50

## Files

| file | rows | what it holds |
|---|---|---|
| `report.md` | - |  |
| `donor_arm_sizes.csv` | 206 | per patient: arm sizes and eligibility under each definition |
| `lineage_checks.csv` | 7 | one row per assertion; `pass` must be True for all |
| `old_arm_composition.csv` | 21 | what the v1 benign arm's cells are called now; `in_new_arm` marks the ones that remain CD4 |

## How to read the columns

- **`donor_arm_sizes.csv`** — per patient: arm sizes and eligibility under each definition
- **`lineage_checks.csv`** — one row per assertion; `pass` must be True for all
- **`old_arm_composition.csv`** — what the v1 benign arm's cells are called now; `in_new_arm` marks the ones that remain CD4

## Things that will mislead you if you do not know them

- **No cohort rules are applied here.** Every donor in the atlas is counted, including healthy controls, geskin2026's ex vivo wells and non-CD4-clone donors. That is why these donor counts are higher than the CCC cohort's 71 skin / 38 blood -- see KNOWLEDGE/DECISIONS.md#cohort-rules.
- `n_benign_v1` is the arm as v1 built it: cell_type_final == 'CD4' and mal_cluster False. It is NOT a clean CD4 population -- that is the point.
- The benign arm is CD4_conv + CD4_naive in skin but CD4_conv alone in blood.
- Donor here is the patient-level unit (clin_person_key -> patient_key -> donor), not the atlas `donor` column.

## Provenance

- **script**: jobs/run_lineage_check.py
- **labels**: data/atlas_joint/nonmal_T_labels_v1.parquet

