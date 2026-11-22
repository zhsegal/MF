# Lineage migration check (2026-11-22)

7 of 7 assertions passed.

## What the old benign CD4 arm actually was

- **skin**: 180,978 cells in the v1 benign arm; 82,876 (45.8%) are still CD4 under the re-annotation. The rest: CD8 29,038, Mixed 26,882, CD4_Treg 19,757, NK_ILC 13,873, Prolif 4,053, LowQC 2,145, Suspect_malignant 1,486, gdT 868.
- **blood**: 207,361 cells in the v1 benign arm; 127,491 (61.5%) are still CD4 under the re-annotation. The rest: CD4_naive 29,037, CD4_Treg 20,320, CD8 15,942, Mixed 5,716, MAIT 3,498, Prolif 1,948, NK_ILC 1,842, LowQC 1,092, gdT 441, Suspect_malignant 34.

## Donor eligibility at >= 50 cells per arm

_No cohort rules applied: healthy controls, geskin2026 ex vivo wells and non-CD4-clone donors are all counted. The CCC cohort, which applies them, has 71 skin and 38 blood._

| tissue | eligible v1 | eligible new |
|---|---|---|
| blood | 50 | 50 |
| skin | 80 | 63 |

Expression-level evidence that the new arm is cleaner is in `reports/nonmal_T_reannot/reannot_coexpression.csv`: the per-cell CD8A/B+ fraction falls 0.199 -> 0.089 in skin and 0.141 -> 0.036 in blood.
