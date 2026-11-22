# Differential cell–cell communication of malignant vs benign CD4 T cells (MF/SS atlas)

_Generated 2026-11-22 10:07 by `jobs/run_ccc52_mvb.py`. Skin and blood are analysed completely separately._

## 1. Summary

**Skin.** Across **71 eligible donors** (198,620 malignant and 131,330 benign CD4 cells), the paired pseudobulk model tested 12,248 genes and 411 of the 1,043 ligand–receptor genes. **27 LR genes** meet all three call criteria — **7 up in the malignant arm** (PTHLH, IGFL2, PTH2R, CXCL13, CD9, CD70, PDGFA) and **20 down** (CCL4, KLRK1, ITGA4, CCL5, CCR6, CXCR6, AREG, SELL, …). 147 CD4-gene × cognate-gene × partner-type rows pass the 10% partner gate across 12 partner types.

**Blood.** Across **38 eligible donors** (182,329 malignant and 27,634 benign CD4 cells), the paired pseudobulk model tested 12,488 genes and 294 of the 1,043 ligand–receptor genes. **23 LR genes** meet all three call criteria — **13 up in the malignant arm** (PTHLH, KIR3DL2, TNFSF11, CXCL13, IGFL2, TNFSF9, CD70, IGFBP4, …) and **10 down** (DPP4, CCL5, ITGA4, CCR6, CXCR3, SLC40A1, KLRG1, FAS, …). 149 CD4-gene × cognate-gene × partner-type rows pass the 10% partner gate across 13 partner types.

## 2. Rationale and estimand

Within a donor, malignant (M) and benign (B) CD4 cells face the same partner cells. For any multiplicative ligand–receptor score the partner term cancels in the M-vs-B contrast, so the differential interaction is exactly a **paired differential expression of ligand and receptor genes on the CD4 side**. Partner cells only act as a gate: is the cognate gene expressed in that cell type. The CD4-side result is the inference; the partner assignment is descriptive. Nothing here is an ambient correction, a LIANA specificity score, a latent-variable model, or a permutation test.

## 3. Data

### Paths used

| role | path |
|---|---|
| counts (raw integers, full gene space) | data/atlas_joint/joint_annotated.h5ad  [layers/raw_counts] |
| cell metadata | data/atlas_joint/atlas_obs_full_v2.parquet |
| malignancy call, skin | data/atlas_joint/skin_T_malignancy_cluster_v1.parquet |
| malignancy call, blood | data/atlas_joint/blood_T_malignancy_cluster_v1.parquet |
| blood per-donor clusters (benign-arm cleanup) | data/atlas_joint/blood_T_clusters_v1.parquet |
| blood marker expression (10k HVG) | data/atlas_joint/blood_T_annotated.h5ad |
| blood non-T fine labels | data/atlas_joint/blood_subtypes.csv |
| blood CD8→NK correction | data/atlas_joint/blood_T_celltype_corrections_v1.csv |
| LR resource (rebuilt here) | reports/ccc52_mvb/lr_resource_1422.csv |

### obs columns used

| quantity | column |
|---|---|
| dataset / study | `dataset`, `study` |
| donor key | `donor` (dataset-prefixed, e.g. `D8__P78`; no cross-study collisions) |
| patient (unit of analysis) | `clin_person_key`, falling back to `patient_key`, then `donor` |
| sample | `sample_id` |
| tissue | `compartment` (Skin / Blood / LN) |
| malignancy | `mal_cluster` from the cluster-malignancy parquets |
| cell type | `cell_type_final` (19 levels; myeloid and fibroblast sub-clusters already collapsed) |
| chemistry | `tech` (10x_5p, 10x_flex, 10x_3p, ECCITE, 10x) |
| clinical | `clin_dx`, `clin_phenotype`, `clin_ex_vivo` |

### Cells per donor × tissue × arm

All 162 donor × tissue units that have at least one cell in either arm after the cohort rules. `eligible` marks the ones with >= 50 cells in **each** arm.

| tissue | unit | n_malignant | n_benign | eligible | study | tech | dx | n_samples | n_donor_keys |
|---|---|---|---|---|---|---|---|---|---|
| blood | washu:CTCL2 | 30525 | 0 | False | borcherding2023 | 10x_5p | SS | 1 | 1 |
| blood | washu:1084 | 22652 | 420 | True | dorando2026 | 10x_5p | MF_or_SS_unresolved | 3 | 1 |
| blood | other/B7:SS3 | 19592 | 0 | False | harro2023 | 10x_5p | SS | 1 | 1 |
| blood | washu:1126 | 19304 | 827 | True | dorando2026 | 10x_5p | MF_or_SS_unresolved | 4 | 1 |
| blood | other/B7:SS1 | 15119 | 1453 | True | harro2023 | 10x_5p | SS | 1 | 1 |
| blood | other/B6:P6 | 10753 | 518 | True | ren2023 | 10x_5p | SS | 1 | 1 |
| blood | other/B6:P3 | 8978 | 265 | True | ren2023 | 10x_5p | SS | 1 | 1 |
| blood | other/B6:P1 | 8831 | 2 | False | ren2023 | 10x_5p | SS | 1 | 1 |
| blood | other/B6:P8 | 8160 | 0 | False | ren2023 | 10x_5p | SS | 1 | 1 |
| blood | other/B6:P4 | 7986 | 1372 | True | ren2023 | 10x_5p | MF | 1 | 1 |
| blood | other/B6:P2 | 7976 | 2142 | True | ren2023 | 10x_5p | MF | 1 | 1 |
| blood | other/B6:P9 | 7929 | 307 | True | ren2023 | 10x_5p | SS | 1 | 1 |
| blood | other/B7:SS2 | 7667 | 164 | True | harro2023 | 10x_5p | SS | 1 | 1 |
| blood | washu:999 | 7631 | 429 | True | dorando2026 | 10x_5p | MF_or_SS_unresolved | 1 | 1 |
| blood | other/B7:SS4 | 7500 | 593 | True | harro2023 | 10x_5p | SS | 1 | 1 |
| blood | washu:1135 | 7094 | 43 | False | dorando2026 | 10x_5p | MF_or_SS_unresolved | 2 | 1 |
| blood | pittsburgh:MF25 | 6752 | 319 | True | gaydosik2022 | 10x_5p | SS | 1 | 1 |
| blood | other/B6:P7 | 6535 | 1295 | True | ren2023 | 10x_5p | SS | 1 | 1 |
| blood | B1__P4 | 6187 | 3275 | True | borcherding2019 | 10x | SS | 4 | 2 |
| blood | B5__PT14 | 6145 | 444 | True | buus2025 | 10x_5p | SS | 2 | 1 |
| blood | B5__PT02 | 5610 | 42 | False | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | B5__PT06 | 5533 | 8 | False | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | other/B6:P10 | 5426 | 2250 | True | ren2023 | 10x_5p | MF | 1 | 1 |
| blood | other/B6:P5 | 5011 | 1070 | True | ren2023 | 10x_5p | SS | 1 | 1 |
| blood | B5__PT05 | 4827 | 0 | False | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | B5__PT08 | 4408 | 2 | False | buus2025 | 10x_5p | SS | 2 | 1 |
| blood | B5__PT01 | 3294 | 5 | False | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | B5__PT09 | 3239 | 184 | True | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | other/B6:P11 | 3112 | 192 | True | ren2023 | 10x_5p | MF | 1 | 1 |
| blood | B5__PT16 | 2952 | 43 | False | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | B5__PT18 | 2786 | 62 | True | buus2025 | 10x_5p | MF | 1 | 1 |
| blood | B5__PT07 | 2532 | 0 | False | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | B5__PT10 | 1781 | 88 | True | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | H__SS1 | 1650 | 594 | True | herrera2021 | ECCITE | SS | 1 | 1 |
| blood | H__MFIV1 | 1365 | 1175 | True | herrera2021 | ECCITE | MF | 1 | 1 |
| blood | H__SS6 | 1171 | 599 | True | herrera2021 | ECCITE | SS | 1 | 1 |
| blood | H__SS3 | 1042 | 90 | True | herrera2021 | ECCITE | SS | 1 | 1 |
| blood | pittsburgh:MF19 | 885 | 1040 | True | gaydosik2022|herrera2021 | 10x_5p|ECCITE | SS | 2 | 2 |
| blood | H__SS5 | 863 | 497 | True | herrera2021 | ECCITE | SS | 1 | 1 |
| blood | washu:1181 | 781 | 831 | True | dorando2026 | 10x_5p | MF_or_SS_unresolved | 3 | 1 |
| blood | B5__PT21 | 777 | 221 | True | buus2025 | 10x_5p | MF | 2 | 1 |
| blood | washu:CTCL6 | 557 | 341 | True | borcherding2023 | 10x_5p | SS | 1 | 1 |
| blood | B5__PT13 | 534 | 37 | False | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | B5__PT15 | 533 | 53 | True | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | washu:CTCL5 | 524 | 185 | True | borcherding2023 | 10x_5p | SS | 1 | 1 |
| blood | vienna:P303 | 447 | 619 | True | rindler2021_fi | 10x_5p | MF | 1 | 1 |
| blood | B5__PT04 | 434 | 3 | False | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | B5__PT23 | 342 | 343 | True | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | H__SS4 | 284 | 317 | True | herrera2021 | ECCITE | SS | 1 | 1 |
| blood | B5__PT22 | 266 | 643 | True | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | pittsburgh:SZ16 | 260 | 2334 | True | gaydosik2022 | 10x_5p | SS | 1 | 1 |
| blood | washu:CTCL4 | 113 | 83 | True | borcherding2023 | 10x_5p | SS | 1 | 1 |
| blood | H__HC1 | 0 | 784 | False | herrera2021 | ECCITE | HC | 1 | 1 |
| blood | other/B6:N1 | 0 | 5606 | False | ren2023 | 10x_5p | HC | 1 | 1 |
| blood | other/B6:N2 | 0 | 3218 | False | ren2023 | 10x_5p | HC | 1 | 1 |
| blood | other/B6:N3 | 0 | 5921 | False | ren2023 | 10x_5p | HC | 1 | 1 |
| blood | other/B7:MF1 | 0 | 24131 | False | harro2023 | 10x_5p | MF | 1 | 1 |
| blood | other/B7:MF2 | 0 | 10408 | False | harro2023 | 10x_5p | MF | 1 | 1 |
| blood | other/B7:MF3 | 0 | 5515 | False | harro2023 | 10x_5p | MF | 1 | 1 |
| blood | pittsburgh:HB | 0 | 2270 | False | gaydosik2022 | 10x_5p | HC | 1 | 3 |
| blood | washu:1137 | 0 | 3667 | False | dorando2026 | 10x_5p | MF_or_SS_unresolved | 2 | 1 |
| blood | washu:CTCL3 | 0 | 690 | False | borcherding2023 | 10x_5p | SS | 1 | 1 |
| skin | vienna:MF311 | 11646 | 2862 | True | chennareddy2025|lyp2026|rindler2021_mc | 10x_5p | MF | 4 | 3 |
| skin | B5__PT09 | 10427 | 5273 | True | buus2025 | 10x_5p | SS | 4 | 1 |
| skin | mda:MF28 | 10395 | 1904 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | vienna:MF309 | 10083 | 3066 | True | chennareddy2025|rindler2021_mc | 10x_5p | MF | 4 | 2 |
| skin | other/D11:P2 | 9473 | 1063 | True | brentuximab2026 | 10x_5p | MF | 3 | 1 |
| skin | mda:CTCL3 | 8823 | 1778 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:CTCL1 | 7676 | 352 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | other/D11:P3 | 7596 | 1589 | True | brentuximab2026 | 10x_5p | MF | 3 | 1 |
| skin | mda:CTCL5 | 7532 | 4073 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:CTCL8 | 7379 | 2447 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:CTCL18 | 7004 | 127 | True | li2024 | 10x_flex | MF | 1 | 1 |
| skin | other/D11:P5 | 6550 | 1370 | True | brentuximab2026 | 10x_5p | MF | 1 | 1 |
| skin | B5__PT23 | 6371 | 1247 | True | buus2025 | 10x_5p | SS | 8 | 1 |
| skin | mda:CTCL13 | 5962 | 2884 | True | li2024 | 10x_flex | MF | 1 | 1 |
| skin | vienna:P171 | 5216 | 382 | True | chennareddy2025 | 10x_5p | MF | 1 | 1 |
| skin | other/D11:P6 | 4575 | 1376 | True | brentuximab2026 | 10x_5p | MF | 3 | 1 |
| skin | mda:CTCL6 | 3937 | 10298 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:MF17 | 3696 | 56 | True | gaydosik2022 | 10x_5p | MF | 1 | 1 |
| skin | mda:CTCL7 | 3658 | 38432 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:MF48 | 3650 | 709 | True | il4ra2026 | 10x_5p | MF | 1 | 1 |
| skin | mda:PT35 | 3502 | 308 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | vienna:P204 | 3444 | 5741 | True | chennareddy2025 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:MF26 | 3041 | 265 | True | il4ra2026 | 10x_5p | MF | 1 | 1 |
| skin | vienna:P192 | 2874 | 660 | True | brunner2024 | 10x_5p | SS | 1 | 1 |
| skin | mda:CTCL12 | 2784 | 331 | True | li2024 | 10x_flex | MF | 1 | 1 |
| skin | vienna:P150 | 2393 | 912 | True | alkon2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:MF30 | 2342 | 1923 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:CTCL9 | 2216 | 106 | True | li2024 | 10x_flex | MF | 1 | 1 |
| skin | pittsburgh:MF35 | 2187 | 597 | True | il4ra2026 | 10x_5p | MF | 1 | 1 |
| skin | mda:PT55 | 2177 | 71 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | vienna:PGS | 2056 | 459 | True | chennareddy2025 | 10x_5p | MF | 1 | 1 |
| skin | mda:PT11 | 1948 | 282 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | vienna:P84 | 1892 | 3889 | True | rindler2021_mc | 10x_5p | MF | 2 | 1 |
| skin | mda:MF21 | 1761 | 708 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:PT50 | 1464 | 1858 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:PT53 | 1417 | 735 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | vienna:P65 | 1404 | 1452 | True | alkon2024|rindler2021_mc | 10x_5p | MF | 2 | 2 |
| skin | mda:CTCL2 | 1357 | 2200 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:PT56 | 1292 | 2478 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:CTCL4 | 1241 | 7906 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:CTCL10 | 1151 | 157 | True | li2024 | 10x_flex | MF | 1 | 1 |
| skin | vienna:P303 | 1038 | 116 | True | chennareddy2025 | 10x_5p | MF | 1 | 1 |
| skin | mda:CTCL16 | 1037 | 294 | True | li2024 | 10x_flex | MF | 1 | 1 |
| skin | vienna:MF312 | 937 | 648 | True | chennareddy2025|rindler2021_mc | 10x_5p | MF | 2 | 2 |
| skin | vienna:P163 | 881 | 417 | True | alkon2024 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:MF18 | 826 | 935 | True | gaydosik2022 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:SC157 | 791 | 593 | True | gaydosik2019 | 10x_3p | MF | 1 | 1 |
| skin | other/D11:P1 | 678 | 90 | True | brentuximab2026 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:SC67 | 677 | 708 | True | gaydosik2019 | 10x_3p | SS | 1 | 1 |
| skin | pittsburgh:MF24 | 643 | 1958 | True | il4ra2026 | 10x_5p | MF | 1 | 1 |
| skin | vienna:P138 | 627 | 614 | True | alkon2024 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:MF36 | 620 | 2182 | True | il4ra2026 | 10x_5p | MF | 1 | 1 |
| skin | vienna:P196 | 506 | 134 | True | brunner2024 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:MF30 | 497 | 354 | True | il4ra2026 | 10x_5p | MF | 1 | 1 |
| skin | vienna:P87 | 351 | 0 | False | brunner2024 | 10x_5p | SS | 1 | 1 |
| skin | vienna:JONAK1 | 348 | 1920 | True | jonak2021 | 10x_5p | MF | 1 | 1 |
| skin | mda:PT47 | 307 | 165 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:SC158 | 306 | 99 | True | gaydosik2019 | 10x_3p | MF | 1 | 1 |
| skin | mda:MF22 | 300 | 575 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:MF19 | 282 | 339 | True | gaydosik2022|herrera2021 | 10x_5p|ECCITE | SS | 2 | 2 |
| skin | pittsburgh:MF34 | 248 | 692 | True | il4ra2026 | 10x_5p | MF | 1 | 1 |
| skin | H__SS4 | 204 | 46 | False | herrera2021 | ECCITE | SS | 1 | 1 |
| skin | mda:CTCL17 | 203 | 82 | True | li2024 | 10x_flex | MF | 1 | 1 |
| skin | vienna:P73 | 189 | 389 | True | rindler2021_mc | 10x_5p | MF | 2 | 1 |
| skin | other/D11:P4 | 174 | 512 | True | brentuximab2026 | 10x_5p | MF | 2 | 1 |
| skin | mda:PT52 | 160 | 89 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | vienna:P90 | 126 | 283 | True | alkon2024|rindler2021_mc | 10x_5p | MF | 2 | 2 |
| skin | pittsburgh:MF28 | 123 | 691 | True | il4ra2026 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:MF21 | 118 | 547 | True | gaydosik2022 | 10x_5p | MF | 1 | 1 |
| skin | vienna:P105 | 117 | 516 | True | brunner2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:MF27 | 107 | 725 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:CTCL15 | 103 | 0 | False | li2024 | 10x_flex | MF | 1 | 1 |
| skin | pittsburgh:MF25 | 99 | 181 | True | gaydosik2022 | 10x_5p | MF/SS | 1 | 1 |
| skin | mda:CTCL11 | 90 | 0 | False | li2024 | 10x_flex | MF | 1 | 1 |
| skin | mda:CTCL14 | 63 | 156 | True | li2024 | 10x_flex | MF | 1 | 1 |
| skin | H__MFIV1 | 44 | 27 | False | herrera2021 | ECCITE | MF | 1 | 1 |
| skin | mda:MF17 | 37 | 4385 | False | li2024 | 10x_5p | MF | 1 | 1 |
| skin | H__SS1 | 36 | 73 | False | herrera2021 | ECCITE | SS | 1 | 1 |
| skin | H__SS3 | 19 | 52 | False | herrera2021 | ECCITE | SS | 1 | 1 |
| skin | B5__PT22 | 10 | 7637 | False | buus2025 | 10x_5p | SS | 8 | 1 |
| skin | mda:MF15 | 0 | 1147 | False | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:MF18 | 0 | 600 | False | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:MF26 | 0 | 1999 | False | li2024 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:MF40 | 0 | 576 | False | il4ra2026 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:NS13 | 0 | 299 | False | il4ra2026 | 10x_5p | HC | 1 | 1 |
| skin | pittsburgh:NS14 | 0 | 442 | False | il4ra2026 | 10x_5p | HC | 1 | 1 |
| skin | pittsburgh:NS16 | 0 | 88 | False | il4ra2026 | 10x_5p | HC | 1 | 1 |
| skin | pittsburgh:NS17 | 0 | 293 | False | il4ra2026 | 10x_5p | HC | 1 | 1 |
| skin | pittsburgh:PF2 | 0 | 994 | False | gaydosik2022 | 10x_5p | HC | 1 | 2 |
| skin | pittsburgh:SC124 | 0 | 80 | False | gaydosik2019 | 10x_3p | HC | 1 | 1 |
| skin | pittsburgh:SC125 | 0 | 61 | False | gaydosik2019 | 10x_3p | HC | 1 | 1 |
| skin | pittsburgh:SC327 | 0 | 207 | False | gaydosik2022 | 10x_5p | HC | 1 | 2 |
| skin | pittsburgh:SC405 | 0 | 47 | False | gaydosik2023 | 10x_5p | HC | 1 | 1 |
| skin | pittsburgh:SC50 | 0 | 51 | False | gaydosik2019 | 10x_3p | HC | 1 | 1 |
| skin | pittsburgh:SC68 | 0 | 49 | False | gaydosik2019 | 10x_3p | HC | 1 | 1 |
| skin | vienna:P112 | 0 | 241 | False | chennareddy2025 | 10x_5p | HC | 1 | 1 |
| skin | vienna:P115 | 0 | 459 | False | chennareddy2025 | 10x_5p | HC | 1 | 1 |
| skin | vienna:P116 | 0 | 443 | False | chennareddy2025 | 10x_5p | HC | 1 | 1 |
| skin | vienna:P121 | 0 | 64 | False | chennareddy2025 | 10x_5p | HC | 1 | 1 |
| skin | vienna:P94 | 0 | 2701 | False | brunner2024 | 10x_5p | MF | 1 | 1 |

### Eligible donors per dataset and tissue

| tissue | study | n_eligible_donors |
|---|---|---|
| blood | ren2023 | 9 |
| blood | buus2025 | 8 |
| blood | herrera2021 | 6 |
| blood | dorando2026 | 4 |
| blood | borcherding2023 | 3 |
| blood | harro2023 | 3 |
| blood | gaydosik2022 | 2 |
| blood | borcherding2019 | 1 |
| blood | gaydosik2022|herrera2021 | 1 |
| blood | rindler2021_fi | 1 |
| skin | li2024 | 29 |
| skin | il4ra2026 | 8 |
| skin | brentuximab2026 | 6 |
| skin | chennareddy2025 | 4 |
| skin | gaydosik2022 | 4 |
| skin | alkon2024 | 3 |
| skin | brunner2024 | 3 |
| skin | gaydosik2019 | 3 |
| skin | alkon2024|rindler2021_mc | 2 |
| skin | buus2025 | 2 |
| skin | chennareddy2025|rindler2021_mc | 2 |
| skin | rindler2021_mc | 2 |
| skin | chennareddy2025|lyp2026|rindler2021_mc | 1 |
| skin | gaydosik2022|herrera2021 | 1 |
| skin | jonak2021 | 1 |

### Donors excluded, and why

| tissue | drop_reason | n_units | n_cells |
|---|---|---|---|
| blood | ex vivo culture cohort (geskin2026) | 11 | 260455 |
| skin | diagnosis outside MF/SS | 15 | 140974 |
| skin | malignant clone is not CD4 | 11 | 136574 |

Units that survived the cohort rules but failed the 50-cells-per-arm floor: skin 29, blood 24.

### Blood benign-arm cleanup

Blood carries no Treg and no naive-CD4 label (`cell_type_T` is {CD4, CD8, NK}; `cell_type_final` has one flat `CD4` over 554,369 cells), so the brief's fallback applies. The benign blood CD4 cells were scored on the per-donor clusters nb32b had already computed (`blood_T_clusters_v1.parquet`, Leiden res 1.0, keyed `<donor>__<k>`), using marker **detection fractions**: a cluster is Treg-like if FOXP3 is detected in >= 15% of its benign CD4 cells, and naive-like if it is detected above 50% for CCR7, 55% for SELL, 50% for TCF7, 45% for LEF1 and its S100A4 detection sits below the across-cluster mean. A purely relative composite-z rule was tried first and rejected: S100A4 detection saturates (median 0.89 across clusters), so low-depth clusters in which every marker is low scored as naive — it flagged seven clusters whose own naive markers were all under 0.30. The absolute rule flags none of those. Of 353 clusters, 37 were called Treg-like and 59 naive-like; their cells left the benign arm. Per-cluster numbers are in `blood_benign_subclusters.csv`. In skin no such step is needed: `CD4_Treg` is its own `cell_type_final` level, and the script asserts that no Treg-labelled cell reaches either arm.

### Partner cell types

A coarse annotation per tissue, taken from `cell_type_final` (which already collapses the nb22 myeloid and fibroblast sub-clusters to major types), with the malignant and benign CD4 arms added as types of their own. Pooled over eligible donors of that tissue; types below 200 cells are dropped.

| tissue | partner | n_cells | homotypic |
|---|---|---|---|
| blood | CD4_malignant | 182329 | True |
| blood | CD4_benign | 27634 | False |
| blood | CD8 | 27274 | False |
| blood | Mono_CD14 | 19407 | False |
| blood | CD4_naive_like | 16943 | False |
| blood | B | 8207 | False |
| blood | CD4_Treg_like | 6208 | False |
| blood | Mono_CD16 | 3000 | False |
| blood | Platelet | 2185 | False |
| blood | Plasma | 1033 | False |
| blood | NK | 553 | False |
| blood | Erythroid | 462 | False |
| blood | cDC | 453 | False |
| skin | CD4_malignant | 198620 | True |
| skin | Keratinocyte | 193410 | False |
| skin | CD4_benign | 131330 | False |
| skin | Fibroblast | 105212 | False |
| skin | Myeloid | 93856 | False |
| skin | Vascular | 64892 | False |
| skin | CD8 | 43152 | False |
| skin | CD4_Treg | 35736 | False |
| skin | B | 24672 | False |
| skin | Plasma | 11275 | False |
| skin | Melanocyte | 10648 | False |
| skin | Mast | 2920 | False |

## 4. Methods

**Counts.** Raw integer counts were read from `data/atlas_joint/joint_annotated.h5ad`, layer `raw_counts` (2,157,693 × 42,347, CSR). Integer-ness and non-negativity are asserted on the first streamed chunk. The compartment objects were deliberately not used for counts: `blood_T_annotated.h5ad` is a 10,000-HVG subset that holds only 590 of the 1,043 LR genes, and neither compartment object contains non-T partner cells.

**Arms.** Malignancy is `mal_cluster`, the repo headline call since 2026-10-22 (`helpers/malignancy.py`): per-sample Leiden over all T cells with one call per cluster, anchored per sample on the ALICE TCR clone. M = CD4 T cells called malignant; B = CD4 T cells called non-malignant. `cell_type_final == "CD4"` already excludes `CD4_Treg`, which is a separate level. Cells with a null (unassessed) call enter neither arm.

**Unit of analysis.** The donor, defined as `clin_person_key` (the repo's curated person key, which merges the same patient across centres) falling back to `patient_key` and then `donor`. All samples and all timepoints of a patient within a tissue are summed into one pseudobulk per arm.

**Cohort rules.** Applied before summation, following `helpers/target_cohort.py`: skin donors whose malignant clone is not CD4 (`clin_phenotype` in CD8/gd/DN, or `clin_dx` in CD8_aggr_CTCL/pcAECTCL) are dropped, because their CD4 cells sit in the benign arm beside a tumour that is a different lineage; skin donors with a diagnosis outside the MF/SS spectrum (LyP, SPP, LPP) are dropped; the whole geskin2026 blood cohort is dropped. The 'same person under two donor keys' rule is not applied as a drop — patient-level aggregation already handles it by summing.

**Eligibility.** A donor × tissue needs >= 50 cells in each arm.

**Pseudobulk.** Raw counts summed over all cells of a donor × arm within a tissue, in one streamed pass over the atlas in chunks of 20,000 cells. The same pass accumulates per-group detection counts (cells with count > 0) and per-chemistry detection counts.

**LR resource.** LIANA `consensus` intersected with interactions supported by >= 6 of the 15 human LIANA resources = **1,422 interactions over 1,043 unique genes** (asserted). Complexes are underscore-joined and each subunit is treated as a gene. No blacklist was applied — see §10.

**Gene filter.** A gene enters a fit if it has >= 10 counts in >= 25% of that fit's pseudobulks.

**Model.** pydeseq2, paired pseudobulk, exact formula **`~ donor + arm`**, reference level `B`, contrast `arm M vs B`, fitted on the full transcriptome (not only LR genes). `refit_cooks=True`. Shrunken log2 fold changes come from `DeseqStats.lfc_shrink` on the `arm` coefficient; the unshrunken estimate is retained as `log2FC_raw`.

**Mixed chemistries.** A 10x Flex probe panel leaves off-panel genes at exactly zero in every cell of that chemistry. Where a tissue mixes chemistries, fit (i) uses **all** eligible donors over the genes detected in every chemistry present, and fit (ii) uses **non-Flex donors only** for every remaining expressed gene. The two gene sets are disjoint and together cover every gene that passes the count filter, so no gene is stranded by a small chemistry that happens to miss it. The `fit` column records which fit each gene came from. Flex (the only probe-panel chemistry here) is li2024 and skin-only, so in blood fit (ii) keeps every donor and is labelled `chemistry-restricted` rather than `non-Flex`. The chemistries actually present per tissue are: **skin** 10x_3p, 10x_5p, 10x_5p|ECCITE, 10x_flex; **blood** 10x, 10x_5p, 10x_5p|ECCITE, ECCITE.

**Multiple testing.** Benjamini–Hochberg applied to the DESeq2 Wald p-values **only over the genes present in the filtered LR resource**, once per tissue per fit. Genes outside the resource carry a p-value and no padj. DESeq2's own independent-filtering padj is not used, because its family is the whole transcriptome.

**Call.** padj < 0.05, |shrunken log2FC| >= 1.0, and the gene detected in >= 10% of the cells of the higher arm (pooled over eligible donors of that tissue).

**Donor concordance.** Per-donor log2FC from CPM with pseudocount 1, `log2((CPM_M + 1)/(CPM_B + 1))`; the reported count is the number of donors whose sign matches the model estimate.

**Complexes.** Every subunit is evaluated as a gene. A complex is called on the CD4 side through its significant subunit provided every other subunit is detected in >= 10% of the cells of that arm (`cd4_complex_ok`).

**Partner gate.** The cognate gene — all subunits, for a complex — must be detected in >= 10% of the cells of the partner type, pooled over eligible donors of that tissue. No statistical test is applied on the partner side. `outgoing` = CD4 ligand to partner receptor; `incoming` = partner ligand to CD4 receptor.

**Reproducibility.** Seed 0. Versions: numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, pydeseq2 0.5.4, anndata 0.12.7, h5py 3.15.1, matplotlib 3.10.8, polars 1.40.1, python 3.12.12.

## 5. Results, skin

Genes tested: **12,248** over 2 fit(s) (all-chemistry, non-Flex); of those **411** are in the filtered LR resource and carry a BH adjusted p-value.

Significant at padj < 0.05, |shrunken log2FC| >= 1.0, and >= 10% detection in the higher arm: **27** genes — **7 up in malignant**, **20 down**. By role: ligand 12, receptor 15, both 0 (up: L 5 / R 2; down: L 7 / R 13).

**Top 7 up in malignant** (by shrunken log2FC):

| gene | role | baseMean | log2FC | padj | pct_M | pct_B | donors_concordant | donors_eligible | fit |
|---|---|---|---|---|---|---|---|---|---|
| PTHLH | ligand | 128.929 | 2.789 | 8.10e-22 | 0.156 | 0.008 | 57 | 71 | all-chemistry |
| IGFL2 | ligand | 224.293 | 2.775 | 1.28e-13 | 0.178 | 0.017 | 46 | 71 | all-chemistry |
| PTH2R | receptor | 66.099 | 2.517 | 3.39e-06 | 0.157 | 0.005 | 46 | 71 | all-chemistry |
| CXCL13 | ligand | 2063.392 | 1.951 | 2.34e-11 | 0.250 | 0.030 | 49 | 71 | all-chemistry |
| CD9 | receptor | 340.738 | 1.414 | 4.34e-10 | 0.309 | 0.039 | 51 | 71 | all-chemistry |
| CD70 | ligand | 266.448 | 1.198 | 5.23e-15 | 0.224 | 0.046 | 58 | 71 | all-chemistry |
| PDGFA | ligand | 107.396 | 1.048 | 1.84e-07 | 0.112 | 0.019 | 49 | 71 | all-chemistry |

**Top 20 down in malignant**:

| gene | role | baseMean | log2FC | padj | pct_M | pct_B | donors_concordant | donors_eligible | fit |
|---|---|---|---|---|---|---|---|---|---|
| CCL4 | ligand | 2324.050 | -4.962 | 3.63e-59 | 0.042 | 0.104 | 63 | 71 | all-chemistry |
| KLRK1 | receptor | 300.917 | -3.964 | 7.87e-37 | 0.092 | 0.127 | 60 | 71 | all-chemistry |
| ITGA4 | receptor | 181.751 | -3.646 | 6.54e-64 | 0.047 | 0.124 | 69 | 71 | all-chemistry |
| CCL5 | ligand | 1764.056 | -3.501 | 1.39e-52 | 0.152 | 0.216 | 67 | 71 | all-chemistry |
| CCR6 | receptor | 197.697 | -2.563 | 5.61e-21 | 0.090 | 0.111 | 52 | 71 | all-chemistry |
| CXCR6 | receptor | 244.261 | -2.408 | 5.12e-19 | 0.117 | 0.100 | 54 | 71 | all-chemistry |
| AREG | ligand | 940.601 | -2.065 | 1.27e-16 | 0.188 | 0.141 | 57 | 71 | all-chemistry |
| SELL | receptor | 659.831 | -2.036 | 2.41e-14 | 0.237 | 0.300 | 52 | 71 | all-chemistry |
| SORL1 | receptor | 221.271 | -1.669 | 8.65e-11 | 0.176 | 0.188 | 49 | 71 | all-chemistry |
| CXCR3 | receptor | 273.855 | -1.663 | 9.00e-13 | 0.118 | 0.127 | 48 | 71 | all-chemistry |
| CD55 | ligand | 560.077 | -1.476 | 5.74e-21 | 0.281 | 0.321 | 63 | 71 | all-chemistry |
| IL10RA | receptor | 400.070 | -1.475 | 2.59e-28 | 0.233 | 0.288 | 65 | 71 | all-chemistry |
| CXCR4 | receptor | 3089.016 | -1.449 | 5.16e-30 | 0.543 | 0.692 | 68 | 71 | all-chemistry |
| ITGB7 | receptor | 239.020 | -1.313 | 3.50e-13 | 0.160 | 0.120 | 53 | 71 | all-chemistry |
| CD27 | receptor | 620.615 | -1.226 | 4.11e-07 | 0.365 | 0.259 | 41 | 71 | all-chemistry |
| FLT3LG | ligand | 377.817 | -1.167 | 2.07e-24 | 0.236 | 0.235 | 59 | 71 | all-chemistry |
| FAS | receptor | 236.399 | -1.121 | 9.74e-10 | 0.163 | 0.149 | 47 | 71 | all-chemistry |
| LTB | ligand | 2591.540 | -1.075 | 3.75e-10 | 0.553 | 0.588 | 52 | 71 | all-chemistry |
| ICOS | receptor | 485.093 | -1.062 | 3.64e-07 | 0.268 | 0.219 | 48 | 71 | all-chemistry |
| TNFSF8 | ligand | 155.820 | -1.054 | 4.19e-06 | 0.138 | 0.103 | 45 | 71 | all-chemistry |

**Gated interactions for those genes** — 147 CD4-gene x cognate x partner rows pass the 10% partner gate (22 CD4-side genes, 12 partner types). The full set for every significant gene is in `interactions_skin.csv` (147 rows).

| cd4_gene | sign | direction | cognate_gene | partner_type | partner_pct | partner_mean_cp10k |
|---|---|---|---|---|---|---|
| AREG | lost in M | outgoing | ERBB3 | Melanocyte | 0.316 | 0.950 |
| AREG | lost in M | outgoing | ERBB3 | Keratinocyte | 0.216 | 0.709 |
| CCL4 | lost in M | outgoing | CCR8 | CD4_Treg | 0.177 | 0.907 |
| CCL4 | lost in M | outgoing | CCR1 | Myeloid | 0.108 | 0.288 |
| CCL4 | lost in M | outgoing | CCR5 | CD8 | 0.104 | 0.689 |
| CCL5 | lost in M | outgoing | ACKR1 | Vascular | 0.493 | 14.051 |
| CCL5 | lost in M | outgoing | SDC1 | Keratinocyte | 0.387 | 2.004 |
| CCL5 | lost in M | outgoing | CCR4 | CD4_Treg | 0.340 | 2.451 |
| CCL5 | lost in M | outgoing | SDC4 | Keratinocyte | 0.339 | 1.401 |
| CCL5 | lost in M | outgoing | CCR4 | CD4_malignant | 0.277 | 1.421 |
| CCL5 | lost in M | outgoing | SDC1 | Plasma | 0.262 | 2.024 |
| CCL5 | lost in M | outgoing | CXCR3 | CD8 | 0.227 | 1.853 |
| CCL5 | lost in M | outgoing | CCR4 | CD4_benign | 0.209 | 1.571 |
| CCL5 | lost in M | outgoing | CXCR3 | Plasma | 0.180 | 0.763 |
| CCL5 | lost in M | outgoing | ACKR4 | Fibroblast | 0.164 | 0.637 |
| CCL5 | lost in M | outgoing | SDC4 | Myeloid | 0.164 | 0.472 |
| CCL5 | lost in M | outgoing | SDC4 | Fibroblast | 0.163 | 0.513 |
| CCL5 | lost in M | outgoing | CXCR3 | CD4_Treg | 0.145 | 0.861 |
| CCL5 | lost in M | outgoing | CXCR3 | CD4_benign | 0.127 | 0.923 |
| CCL5 | lost in M | outgoing | CXCR3 | CD4_malignant | 0.118 | 0.520 |
| CCL5 | lost in M | outgoing | CCR1 | Myeloid | 0.108 | 0.288 |
| CCL5 | lost in M | outgoing | CCR5 | CD8 | 0.104 | 0.689 |
| CD27 | lost in M | incoming | CD70 | CD4_malignant | 0.224 | 1.096 |
| CD27 | lost in M | incoming | CD70 | CD4_Treg | 0.145 | 0.902 |
| CD27 | lost in M | incoming | CD70 | B | 0.131 | 1.114 |
| CD27 | lost in M | incoming | CD70 | Plasma | 0.102 | 0.487 |
| CD55 | lost in M | outgoing | ADGRE5 | CD4_malignant | 0.351 | 1.777 |
| CD55 | lost in M | outgoing | ADGRE5 | CD8 | 0.297 | 2.586 |
| CD55 | lost in M | outgoing | ADGRE5 | Myeloid | 0.282 | 0.947 |
| CD55 | lost in M | outgoing | ADGRE5 | Plasma | 0.263 | 1.020 |
| CD55 | lost in M | outgoing | ADGRE5 | Fibroblast | 0.248 | 0.844 |
| CD55 | lost in M | outgoing | ADGRE5 | CD4_Treg | 0.232 | 1.259 |
| CD55 | lost in M | outgoing | ADGRE5 | B | 0.230 | 1.692 |
| CD55 | lost in M | outgoing | ADGRE5 | CD4_benign | 0.214 | 1.742 |
| CD55 | lost in M | outgoing | ADGRE5 | Mast | 0.149 | 1.198 |
| CD55 | lost in M | outgoing | ADGRE5 | Melanocyte | 0.116 | 0.290 |
| CD70 | gained in M | outgoing | CD27 | CD4_Treg | 0.574 | 5.289 |
| CD70 | gained in M | outgoing | CD27 | Plasma | 0.433 | 4.075 |
| CD70 | gained in M | outgoing | CD27 | CD8 | 0.417 | 4.507 |
| CD70 | gained in M | outgoing | CD27 | CD4_malignant | 0.365 | 2.456 |
| CD70 | gained in M | outgoing | CD27 | CD4_benign | 0.259 | 2.128 |
| CD70 | gained in M | outgoing | CD27 | B | 0.161 | 1.247 |
| CD9 | gained in M | incoming | HBEGF | Keratinocyte | 0.266 | 1.547 |
| CD9 | gained in M | incoming | HBEGF | Vascular | 0.235 | 1.040 |
| CD9 | gained in M | incoming | HBEGF | Myeloid | 0.140 | 0.613 |
| CXCL13 | gained in M | outgoing | CXCR5 | B | 0.285 | 2.109 |
| CXCL13 | gained in M | outgoing | CXCR3 | CD8 | 0.227 | 1.853 |
| CXCL13 | gained in M | outgoing | CXCR3 | Plasma | 0.180 | 0.763 |
| CXCL13 | gained in M | outgoing | ACKR4 | Fibroblast | 0.164 | 0.637 |
| CXCL13 | gained in M | outgoing | CXCR3 | CD4_Treg | 0.145 | 0.861 |
| CXCL13 | gained in M | outgoing | CXCR3 | CD4_benign | 0.127 | 0.923 |
| CXCL13 | gained in M | outgoing | CXCR3 | CD4_malignant | 0.118 | 0.520 |
| CXCR3 | lost in M | incoming | CCL5 | CD8 | 0.819 | 35.357 |
| CXCR3 | lost in M | incoming | CXCL12 | Fibroblast | 0.623 | 11.646 |
| CXCR3 | lost in M | incoming | CCL19 | Fibroblast | 0.332 | 32.798 |
| CXCR3 | lost in M | incoming | CXCL13 | CD4_malignant | 0.250 | 10.856 |
| CXCR3 | lost in M | incoming | CXCL12 | Vascular | 0.238 | 2.237 |
| CXCR3 | lost in M | incoming | CCL5 | CD4_benign | 0.216 | 5.888 |
| CXCR3 | lost in M | incoming | CXCL13 | CD4_Treg | 0.161 | 12.086 |
| CXCR3 | lost in M | incoming | CCL13 | Myeloid | 0.154 | 2.365 |
| CXCR3 | lost in M | incoming | CCL5 | CD4_malignant | 0.152 | 1.224 |
| CXCR3 | lost in M | incoming | CCL5 | Fibroblast | 0.145 | 1.171 |
| CXCR3 | lost in M | incoming | CCL21 | Vascular | 0.144 | 35.835 |
| CXCR3 | lost in M | incoming | CXCL9 | Myeloid | 0.127 | 2.701 |
| CXCR3 | lost in M | incoming | CXCL9 | Fibroblast | 0.114 | 3.728 |
| CXCR3 | lost in M | incoming | CCL5 | Myeloid | 0.114 | 1.060 |
| CXCR4 | lost in M | incoming | CXCL12 | Fibroblast | 0.623 | 11.646 |
| CXCR4 | lost in M | incoming | CXCL12 | Vascular | 0.238 | 2.237 |
| CXCR6 | lost in M | incoming | CXCL16 | Myeloid | 0.549 | 5.489 |
| CXCR6 | lost in M | incoming | CXCL16 | Keratinocyte | 0.230 | 0.711 |
| CXCR6 | lost in M | incoming | CXCL16 | Fibroblast | 0.113 | 0.298 |
| FAS | lost in M | incoming | TNFSF13 | Myeloid | 0.206 | 0.694 |
| FAS | lost in M | incoming | FASLG | CD8 | 0.131 | 0.977 |
| FAS | lost in M | incoming | TNFSF13 | Plasma | 0.108 | 0.428 |
| FLT3LG | lost in M | outgoing | FLT3 | Myeloid | 0.157 | 0.561 |
| IGFL2 | gained in M | outgoing | IGFLR1 | CD4_Treg | 0.487 | 3.640 |
| IGFL2 | gained in M | outgoing | IGFLR1 | Myeloid | 0.486 | 1.997 |
| IGFL2 | gained in M | outgoing | IGFLR1 | Plasma | 0.430 | 1.803 |
| IGFL2 | gained in M | outgoing | IGFLR1 | CD4_malignant | 0.366 | 1.688 |
| IGFL2 | gained in M | outgoing | IGFLR1 | B | 0.291 | 1.765 |

## 6. Results, blood

Genes tested: **12,488** over 2 fit(s) (all-chemistry, chemistry-restricted); of those **294** are in the filtered LR resource and carry a BH adjusted p-value.

Significant at padj < 0.05, |shrunken log2FC| >= 1.0, and >= 10% detection in the higher arm: **23** genes — **13 up in malignant**, **10 down**. By role: ligand 9, receptor 14, both 0 (up: L 7 / R 6; down: L 2 / R 8).

**Top 13 up in malignant** (by shrunken log2FC):

| gene | role | baseMean | log2FC | padj | pct_M | pct_B | donors_concordant | donors_eligible | fit |
|---|---|---|---|---|---|---|---|---|---|
| PTHLH | ligand | 91.208 | 4.676 | 4.73e-43 | 0.183 | 0.007 | 37 | 38 | all-chemistry |
| KIR3DL2 | receptor | 353.209 | 3.918 | 7.61e-25 | 0.344 | 0.015 | 34 | 38 | all-chemistry |
| TNFSF11 | ligand | 80.725 | 3.886 | 6.16e-36 | 0.129 | 0.011 | 38 | 38 | all-chemistry |
| CXCL13 | ligand | 177.865 | 3.796 | 2.66e-09 | 0.156 | 0.008 | 30 | 38 | all-chemistry |
| IGFL2 | ligand | 185.080 | 3.500 | 1.76e-12 | 0.132 | 0.013 | 32 | 38 | all-chemistry |
| TNFSF9 | ligand | 123.975 | 3.194 | 5.74e-34 | 0.196 | 0.014 | 37 | 38 | all-chemistry |
| CD70 | ligand | 239.630 | 2.622 | 6.20e-21 | 0.224 | 0.033 | 34 | 38 | all-chemistry |
| IGFBP4 | ligand | 247.843 | 2.463 | 2.27e-19 | 0.212 | 0.047 | 34 | 38 | all-chemistry |
| FGFR1 | receptor | 86.017 | 1.756 | 4.43e-11 | 0.111 | 0.025 | 34 | 38 | all-chemistry |
| PDCD1 | receptor | 157.626 | 1.554 | 1.25e-08 | 0.190 | 0.078 | 34 | 38 | all-chemistry |
| ITGB1 | receptor | 2822.970 | 1.410 | 7.72e-19 | 0.841 | 0.533 | 36 | 38 | all-chemistry |
| EPHA4 | receptor | 174.592 | 1.305 | 1.17e-28 | 0.208 | 0.078 | 36 | 38 | all-chemistry |
| CCR4 | receptor | 272.364 | 1.090 | 1.22e-04 | 0.253 | 0.107 | 30 | 38 | all-chemistry |

**Top 10 down in malignant**:

| gene | role | baseMean | log2FC | padj | pct_M | pct_B | donors_concordant | donors_eligible | fit |
|---|---|---|---|---|---|---|---|---|---|
| DPP4 | receptor | 90.803 | -5.637 | 2.08e-56 | 0.008 | 0.146 | 37 | 38 | all-chemistry |
| CCL5 | ligand | 654.328 | -5.426 | 2.96e-57 | 0.021 | 0.154 | 38 | 38 | all-chemistry |
| ITGA4 | receptor | 495.664 | -4.375 | 6.16e-36 | 0.073 | 0.413 | 37 | 38 | all-chemistry |
| CCR6 | receptor | 175.935 | -3.171 | 3.19e-18 | 0.083 | 0.158 | 32 | 38 | all-chemistry |
| CXCR3 | receptor | 213.227 | -3.050 | 5.50e-16 | 0.083 | 0.167 | 31 | 38 | all-chemistry |
| SLC40A1 | receptor | 93.377 | -2.712 | 9.32e-15 | 0.033 | 0.116 | 31 | 38 | all-chemistry |
| KLRG1 | receptor | 133.256 | -1.921 | 9.61e-27 | 0.045 | 0.162 | 35 | 38 | all-chemistry |
| FAS | receptor | 159.639 | -1.176 | 1.62e-09 | 0.086 | 0.180 | 31 | 38 | all-chemistry |
| TIMP1 | ligand | 595.078 | -1.166 | 8.65e-12 | 0.267 | 0.439 | 30 | 38 | all-chemistry |
| CD2 | receptor | 1157.244 | -1.033 | 1.13e-06 | 0.424 | 0.679 | 29 | 38 | all-chemistry |

**Gated interactions for those genes** — 149 CD4-gene x cognate x partner rows pass the 10% partner gate (15 CD4-side genes, 13 partner types). The full set for every significant gene is in `interactions_blood.csv` (149 rows).

| cd4_gene | sign | direction | cognate_gene | partner_type | partner_pct | partner_mean_cp10k |
|---|---|---|---|---|---|---|
| CCL5 | lost in M | outgoing | CXCR3 | CD4_Treg_like | 0.312 | 1.114 |
| CCL5 | lost in M | outgoing | CCR1 | Mono_CD14 | 0.290 | 0.923 |
| CCL5 | lost in M | outgoing | CCR4 | CD4_Treg_like | 0.265 | 0.774 |
| CCL5 | lost in M | outgoing | CCR4 | CD4_malignant | 0.253 | 0.888 |
| CCL5 | lost in M | outgoing | CXCR3 | Plasma | 0.236 | 0.586 |
| CCL5 | lost in M | outgoing | CXCR3 | CD4_benign | 0.167 | 0.533 |
| CCL5 | lost in M | outgoing | CCR1 | Mono_CD16 | 0.164 | 0.296 |
| CCL5 | lost in M | outgoing | CXCR3 | cDC | 0.135 | 0.170 |
| CCL5 | lost in M | outgoing | CXCR3 | CD8 | 0.128 | 0.506 |
| CCL5 | lost in M | outgoing | CCR4 | CD4_benign | 0.107 | 0.273 |
| CCR4 | gained in M | incoming | CCL5 | NK | 0.821 | 24.756 |
| CCR4 | gained in M | incoming | CCL5 | CD8 | 0.677 | 24.229 |
| CCR4 | gained in M | incoming | CCL3 | Mono_CD14 | 0.382 | 13.406 |
| CCR4 | gained in M | incoming | CCL3 | NK | 0.320 | 1.717 |
| CCR4 | gained in M | incoming | CCL5 | Platelet | 0.311 | 4.978 |
| CCR4 | gained in M | incoming | CCL5 | CD4_benign | 0.154 | 1.714 |
| CCR4 | gained in M | incoming | CCL5 | cDC | 0.137 | 0.387 |
| CCR4 | gained in M | incoming | CCL5 | Plasma | 0.135 | 1.123 |
| CCR4 | gained in M | incoming | CCL3 | CD8 | 0.134 | 0.925 |
| CCR4 | gained in M | incoming | CCL2 | Platelet | 0.129 | 1.444 |
| CCR4 | gained in M | incoming | CCL3 | Mono_CD16 | 0.105 | 1.072 |
| CD2 | lost in M | incoming | CD58 | cDC | 0.393 | 0.464 |
| CD2 | lost in M | incoming | CD58 | Mono_CD16 | 0.360 | 0.610 |
| CD2 | lost in M | incoming | CD58 | Mono_CD14 | 0.251 | 0.780 |
| CD2 | lost in M | incoming | CD58 | CD4_Treg_like | 0.227 | 0.573 |
| CD2 | lost in M | incoming | CD58 | NK | 0.192 | 0.566 |
| CD2 | lost in M | incoming | CD58 | Platelet | 0.169 | 0.419 |
| CD2 | lost in M | incoming | CD58 | Plasma | 0.167 | 0.207 |
| CD2 | lost in M | incoming | CD58 | CD8 | 0.151 | 0.433 |
| CD2 | lost in M | incoming | CD58 | CD4_benign | 0.145 | 0.346 |
| CD2 | lost in M | incoming | CD58 | CD4_malignant | 0.127 | 0.343 |
| CD70 | gained in M | outgoing | CD27 | CD4_malignant | 0.684 | 4.570 |
| CD70 | gained in M | outgoing | CD27 | CD4_naive_like | 0.648 | 2.166 |
| CD70 | gained in M | outgoing | CD27 | CD4_Treg_like | 0.612 | 3.059 |
| CD70 | gained in M | outgoing | CD27 | CD4_benign | 0.489 | 1.895 |
| CD70 | gained in M | outgoing | CD27 | Plasma | 0.359 | 1.733 |
| CD70 | gained in M | outgoing | CD27 | CD8 | 0.270 | 1.419 |
| CXCL13 | gained in M | outgoing | CXCR3 | CD4_Treg_like | 0.312 | 1.114 |
| CXCL13 | gained in M | outgoing | CXCR5 | B | 0.242 | 0.784 |
| CXCL13 | gained in M | outgoing | CXCR3 | Plasma | 0.236 | 0.586 |
| CXCL13 | gained in M | outgoing | CXCR3 | CD4_benign | 0.167 | 0.533 |
| CXCL13 | gained in M | outgoing | CXCR3 | cDC | 0.135 | 0.170 |
| CXCL13 | gained in M | outgoing | CXCR3 | CD8 | 0.128 | 0.506 |
| CXCL13 | gained in M | outgoing | CXCR5 | Plasma | 0.123 | 0.147 |
| CXCR3 | lost in M | incoming | CCL5 | NK | 0.821 | 24.756 |
| CXCR3 | lost in M | incoming | CCL5 | CD8 | 0.677 | 24.229 |
| CXCR3 | lost in M | incoming | CCL5 | Platelet | 0.311 | 4.978 |
| CXCR3 | lost in M | incoming | CXCL12 | Platelet | 0.224 | 3.054 |
| CXCR3 | lost in M | incoming | PF4 | Platelet | 0.186 | 2.916 |
| CXCR3 | lost in M | incoming | CXCL13 | CD4_malignant | 0.156 | 0.967 |
| CXCR3 | lost in M | incoming | CCL5 | CD4_benign | 0.154 | 1.714 |
| CXCR3 | lost in M | incoming | CCL5 | cDC | 0.137 | 0.387 |
| CXCR3 | lost in M | incoming | CCL5 | Plasma | 0.135 | 1.123 |
| EPHA4 | gained in M | incoming | EFNB1 | cDC | 0.258 | 0.283 |
| EPHA4 | gained in M | incoming | EFNA5 | Platelet | 0.121 | 0.828 |
| EPHA4 | gained in M | incoming | EFNB2 | Platelet | 0.101 | 0.349 |
| FAS | lost in M | incoming | TNFSF13 | Mono_CD16 | 0.561 | 1.322 |
| FAS | lost in M | incoming | TNFSF13 | cDC | 0.444 | 0.710 |
| FAS | lost in M | incoming | TNFSF13 | Mono_CD14 | 0.283 | 0.836 |
| FAS | lost in M | incoming | FASLG | NK | 0.175 | 0.580 |
| FGFR1 | gained in M | incoming | NCAM1 | NK | 0.165 | 0.534 |
| FGFR1 | gained in M | incoming | FGF7 | Platelet | 0.148 | 1.153 |
| IGFL2 | gained in M | outgoing | IGFLR1 | cDC | 0.607 | 0.917 |
| IGFL2 | gained in M | outgoing | IGFLR1 | Plasma | 0.376 | 0.609 |
| IGFL2 | gained in M | outgoing | IGFLR1 | Mono_CD16 | 0.329 | 0.538 |
| IGFL2 | gained in M | outgoing | IGFLR1 | Mono_CD14 | 0.267 | 0.803 |
| IGFL2 | gained in M | outgoing | IGFLR1 | CD4_Treg_like | 0.237 | 0.624 |
| IGFL2 | gained in M | outgoing | IGFLR1 | B | 0.223 | 0.732 |
| IGFL2 | gained in M | outgoing | IGFLR1 | CD4_naive_like | 0.163 | 0.342 |
| IGFL2 | gained in M | outgoing | IGFLR1 | NK | 0.159 | 0.459 |
| IGFL2 | gained in M | outgoing | IGFLR1 | CD8 | 0.158 | 0.458 |
| IGFL2 | gained in M | outgoing | IGFLR1 | CD4_benign | 0.153 | 0.368 |
| IGFL2 | gained in M | outgoing | IGFLR1 | CD4_malignant | 0.124 | 0.325 |
| IGFL2 | gained in M | outgoing | IGFLR1 | Platelet | 0.124 | 0.363 |
| ITGA4 | lost in M | incoming | VCAN | Mono_CD14 | 0.781 | 13.918 |
| ITGA4 | lost in M | incoming | CD14 | Mono_CD14 | 0.771 | 12.332 |
| ITGA4 | lost in M | incoming | THBS1 | Platelet | 0.366 | 4.992 |
| ITGA4 | lost in M | incoming | ICAM4 | Mono_CD16 | 0.347 | 0.644 |
| ITGA4 | lost in M | incoming | VCAN | cDC | 0.347 | 1.244 |
| ITGA4 | lost in M | incoming | VCAN | Mono_CD16 | 0.287 | 0.913 |

## 7. Skin vs blood

Significant in skin only: **18**; blood only: **14**; both: **9**.

- shared: CCL5, CCR6, CD70, CXCL13, CXCR3, FAS, IGFL2, ITGA4, PTHLH
- skin only: AREG, CCL4, CD27, CD55, CD9, CXCR4, CXCR6, FLT3LG, ICOS, IL10RA, ITGB7, KLRK1, LTB, PDGFA, PTH2R, SELL, SORL1, TNFSF8
- blood only: CCR4, CD2, DPP4, EPHA4, FGFR1, IGFBP4, ITGB1, KIR3DL2, KLRG1, PDCD1, SLC40A1, TIMP1, TNFSF11, TNFSF9

Across the **277 LR genes tested in both tissues**, the shrunken log2FC correlate at Pearson r = **0.711** (Spearman 0.560), and 185 of 277 (67%) agree in sign.

**Significance in one tissue only is not evidence of tissue specificity.** The two compartments differ in eligible donor count, in cell numbers per arm, and in the composition of the benign arm, so they differ in power; a gene can miss the threshold in one tissue while its estimate is the same size and sign in both. Only the effect-size comparison above speaks to tissue specificity, and it does so weakly.

## 8. Prior axes

Reported regardless of significance, both tissues. `padj_family` says whether the gene is inside the BH family (the filtered LR resource); five of the named axes are not, and their cognate pairs are resolved against the unfiltered consensus instead. **TOX is the positive control for the malignant contrast**, not an LR gene.

| gene | tissue | log2FC | padj | padj_family | pct_M | pct_B | donors_concordant | donors_eligible | fit | cognate_genes | gated_partners |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CCR4 | blood | 1.090 | 1.22e-04 | LR genes | 0.253 | 0.107 | 30 | 38 | all-chemistry | CCL17; CCL22 |  |
| CCR4 | skin | -0.182 | 0.312 | LR genes | 0.277 | 0.209 | 36 | 71 | all-chemistry | CCL17; CCL22 | CCL17:Myeloid (23%); CCL22:Myeloid (27%) |
| CD27 | blood | 0.804 | 1.43e-05 | LR genes | 0.684 | 0.489 | 35 | 38 | all-chemistry | CD70 | CD70:CD4_Treg_like (11%); CD70:CD4_malignant (22%); CD70:Plasma (13%) |
| CD27 | skin | -1.226 | 4.11e-07 | LR genes | 0.365 | 0.259 | 41 | 71 | all-chemistry | CD70 | CD70:B (13%); CD70:CD4_Treg (15%); CD70:CD4_malignant (22%); CD70:Plasma (10%) |
| CD28 | blood | 0.553 | 4.97e-10 | LR genes | 0.422 | 0.303 | 35 | 38 | all-chemistry | CD80; CD86 | CD86:Mono_CD14 (30%); CD86:Mono_CD16 (64%); CD86:Plasma (15%); CD86:cDC (67%) |
| CD28 | skin | -0.721 | 4.82e-04 | LR genes | 0.312 | 0.249 | 39 | 71 | all-chemistry | CD80; CD86 | CD80:Myeloid (19%); CD86:Myeloid (44%) |
| CD70 | blood | 2.622 | 6.20e-21 | LR genes | 0.224 | 0.033 | 34 | 38 | all-chemistry | CD27 | CD27:CD4_Treg_like (61%); CD27:CD4_benign (49%); CD27:CD4_malignant (68%); CD27:CD4_naive_like (65%); CD27:CD8 (27%); CD27:Plasma (36%) |
| CD70 | skin | 1.198 | 5.23e-15 | LR genes | 0.224 | 0.046 | 58 | 71 | all-chemistry | CD27 | CD27:B (16%); CD27:CD4_Treg (57%); CD27:CD4_benign (26%); CD27:CD4_malignant (37%); CD27:CD8 (42%); CD27:Plasma (43%) |
| CSF2 | blood |  |  | none (outside the filtered resource) |  |  | 0 | 0 |  | CSF1R; CSF2RA_CSF2RB; CSF3R; IL3RA; ITGA9; ITGB1; SDC2 | CSF1R:Mono_CD14 (32%); CSF1R:Mono_CD16 (88%); CSF1R:cDC (41%); CSF2RA_CSF2RB:Mono_CD14 (21%); CSF2RA_CSF2RB:Mono_CD16 (22%); CSF2RA_CSF2RB:cDC (21%); CSF3R:Mono_CD14 (68%); CSF3R:Mono_CD16 (32%); CSF3R:cDC (57%); IL3RA:Mono_CD16 (20%); IL3RA:Plasma (20%); IL3RA:cDC (14%) |
| CSF2 | skin | -0.149 | 0.307 | LR genes | 0.092 | 0.029 | 35 | 71 | all-chemistry | CSF1R; CSF2RA_CSF2RB; CSF3R; IL3RA; ITGA9; ITGB1; SDC2 | CSF1R:Myeloid (32%); CSF2RA_CSF2RB:Myeloid (28%); CSF2RA_CSF2RB:Plasma (13%); CSF3R:Myeloid (12%); IL3RA:Myeloid (18%); IL3RA:Plasma (28%); IL3RA:Vascular (45%); ITGA9:Vascular (18%); ITGB1:B (12%); ITGB1:CD4_Treg (44%); ITGB1:CD4_benign (35%); ITGB1:CD4_malignant (43%) |
| CTLA4 | blood | 0.889 | 1.39e-05 | LR genes | 0.223 | 0.107 | 30 | 38 | all-chemistry | CD80; CD86 | CD86:Mono_CD14 (30%); CD86:Mono_CD16 (64%); CD86:Plasma (15%); CD86:cDC (67%) |
| CTLA4 | skin | -0.546 | 0.003 | LR genes | 0.300 | 0.180 | 43 | 71 | all-chemistry | CD80; CD86 | CD80:Myeloid (19%); CD86:Myeloid (44%) |
| CXCR4 | blood | -0.561 | 0.001 | LR genes | 0.547 | 0.541 | 25 | 38 | all-chemistry | CXCL12 | CXCL12:Platelet (22%) |
| CXCR4 | skin | -1.449 | 5.16e-30 | LR genes | 0.543 | 0.692 | 68 | 71 | all-chemistry | CXCL12 | CXCL12:Fibroblast (62%); CXCL12:Vascular (24%) |
| IL22 | blood |  |  | none (outside the filtered resource) |  |  | 0 | 0 |  | IL22RA1 |  |
| IL22 | skin | -0.542 | 0.001 | LR genes | 0.050 | 0.007 | 36 | 71 | all-chemistry | IL22RA1 |  |
| JAML | blood | -0.764 |  | none (outside the filtered resource) | 0.126 | 0.152 | 23 | 38 | all-chemistry | CXADR |  |
| JAML | skin | -0.411 |  | none (outside the filtered resource) | 0.284 | 0.212 | 41 | 71 | all-chemistry | CXADR | CXADR:Keratinocyte (42%) |
| KIR3DL2 | blood | 3.918 | 7.61e-25 | LR genes | 0.344 | 0.015 | 34 | 38 | all-chemistry | HLA-A | HLA-A:B (96%); HLA-A:CD4_Treg_like (99%); HLA-A:CD4_benign (98%); HLA-A:CD4_malignant (99%); HLA-A:CD4_naive_like (99%); HLA-A:CD8 (99%); HLA-A:Erythroid (48%); HLA-A:Mono_CD14 (96%); HLA-A:Mono_CD16 (99%); HLA-A:NK (100%); HLA-A:Plasma (92%); HLA-A:Platelet (80%) |
| KIR3DL2 | skin | 0.489 | 0.962 | LR genes | 0.163 | 0.011 | 24 | 71 | non-Flex | HLA-A | HLA-A:B (33%); HLA-A:CD4_Treg (41%); HLA-A:CD4_benign (34%); HLA-A:CD4_malignant (50%); HLA-A:CD8 (36%); HLA-A:Fibroblast (65%); HLA-A:Keratinocyte (61%); HLA-A:Mast (43%); HLA-A:Melanocyte (41%); HLA-A:Myeloid (43%); HLA-A:Plasma (27%); HLA-A:Vascular (63%) |
| SPN | blood | 0.056 |  | none (outside the filtered resource) | 0.305 | 0.323 | 28 | 38 | all-chemistry | SIGLEC1 |  |
| SPN | skin | 0.034 |  | none (outside the filtered resource) | 0.350 | 0.169 | 42 | 71 | all-chemistry | SIGLEC1 | SIGLEC1:Myeloid (12%) |
| TOX | blood | 2.721 |  | none (outside the filtered resource) | 0.448 | 0.087 | 36 | 38 | all-chemistry |  |  |
| TOX | skin | 1.155 |  | none (outside the filtered resource) | 0.476 | 0.122 | 64 | 71 | all-chemistry |  |  |

## 9. Limitations

- **No proximity information.** Dissociated scRNA-seq says nothing about whether two cells were ever in contact. Every interaction reported here is a statement about co-expression within a tissue, not about an engagement.
- **Partner assignments are not corrected for ambient RNA and are suggestive only.** No ambient correction was performed anywhere in this analysis, by design. A partner type that sits next to a highly expressing neighbour will carry that neighbour's transcripts, and the gate is a detection fraction, which ambient contamination inflates directly.
- **Benign arm composition.** The benign arm is whatever the malignancy call left behind. In skin that is conventional CD4 minus the `CD4_Treg` level; in blood it is conventional CD4 after the Treg-like and naive-like clusters were removed by a marker rule, which is a cruder instrument than a curated label. Residual activation-state differences between the arms are not separable from malignancy here.
- **Contact-dependent interactions in blood are weakly interpretable.** Circulating cells are not in stable contact. A gained CD28–CD86 or CTLA4–CD80 edge in blood is a statement about the transcript, not about a synapse.
- **Number of eligible blood donors.** Blood has fewer eligible donors than skin, and the geskin2026 exclusion removed the largest blood cohort outright. Power is correspondingly lower, which is the main reason a skin-only hit is not a tissue-specificity claim.
- **The malignancy call is a cluster call.** `mal_cluster` assigns a whole per-sample cluster, so a benign cell inside a malignant cluster is labelled malignant and vice versa. `helpers/malignancy.py` also notes that healthy-control samples are called benign by construction, so a 0% HC rate is an input, not a result.

## 10. Decisions and deviations; upstream errors noticed

### Decisions and deviations from the brief

- **The filtered LR resource was rebuilt, not loaded — it does not exist on disk.** The brief names it as an artefact of a rebuilt LIANA+ pipeline. A scoped search of `data/`, `tables/`, `docs/` and `helpers/` found no such file; `helpers/ccc_panel.py:12` carries the specification as a run note and nothing implements it. It was reproduced exactly: LIANA `consensus` intersected with support >= 6 of the 15 human resources gives 1,422 interactions over 1,043 genes, and the script asserts both numbers.
- **No blacklist was applied.** No blacklist file exists in this repo, and `docs/CCC_PIPELINE_PORTABLE_SPEC.md:83` states that `LR_CAVEATS` is *commentary only, never a blacklist*, because the v1 version silently deleted the IL4/IL13 pairs the analysis existed to report (`helpers/ccc_utils.py:1157`). Applying one would have contradicted the repo's own standing instruction, so the caveat list is not used as a filter.
- **Arms come from `mal_cluster`, not from the inferCNV/TCR `combined_malignant` flag.** The brief asks for 'the final malignancy calls (inferCNV/TCR pipeline)'. That combined flag was retired: `helpers/malignancy.py` states the four calls are never OR-ed, because the CNV call derives from smoothed regional expression — partly the same signal a downstream expression analysis then attributes. `mal_cluster` has been the headline since 2026-10-22 and is anchored per sample on a TCR sequence fact.
- **The unit is the curated person key.** `clin_person_key` falling back to `patient_key` and then `donor`, so longitudinal and multi-site samples of one patient are summed and the two patients recorded under two donor keys are merged.
- **geskin2026 was dropped from blood in full**, on the user's instruction. All 260,455 of its cells are ex vivo 2-day cultures (213,181 drug-treated, 47,274 vehicle), which are not in-vivo biology.
- **The blood benign arm was cleaned by a cluster marker rule, not a fresh Leiden.** Blood has no Treg and no naive-CD4 label, so the brief's fallback applies. Rather than re-cluster on the 10-dimensional MrVI u latent, the per-donor clusters nb32b had already computed were scored on marker detection fractions — within-donor, so free of batch structure. Thresholds and per-cluster numbers are in §3 and `blood_benign_subclusters.csv`.
- **Counts were taken from `joint_annotated.h5ad`, not the compartment objects**, so that blood is analysed over the full 42,347-gene space rather than a 10,000-gene HVG subset.
- **Five prior axes fall outside the filtered resource** (JAML–CXADR, SPN–SIGLEC1, IL22–IL22RA1, CSF2's receptor, and TOX, which is not an LR gene). They are reported from the unfiltered consensus and flagged, since the brief asks for them regardless of significance.
- **DESeq2's own independent-filtering padj is discarded.** Its family is the whole transcriptome; the brief asks for BH over the LR resource only, so BH is recomputed from the Wald p-values over that family, per tissue per fit.
- **The 'same person under two donor keys' cohort rule is applied as a merge, not a drop**, because patient-level summation is what the brief asks for.
- **Carlito and Calibri are not installed on these compute nodes.** They remain first in the font stack, so the figures render in them anywhere the fonts exist; here they fall back to DejaVu Sans. Everything else in the figure specification — PNG and PDF at 400 dpi, `pdf.fonttype=42`, white background, the single teal accent #0F766E, no decorative elements — is as requested.

### Upstream data errors and inconsistencies noticed

- **The brief's description of geskin2026 does not match the atlas.** It says 5 patients and 28 units. On disk geskin2026 is blood-only, 260,455 cells, 11 `patient_key` values, 65 `donor` keys that are lane × HTO wells, and 11 libraries. `docs/DATA_PROVENANCE.md:61` records that the HTO-to-patient map was never deposited, so the 65 donor keys are pseudo-donors, not people.
- **`blood_T_annotated.h5ad` is a 10,000-gene HVG subset while `skin_T_annotated.h5ad` carries all 42,347.** This asymmetry is not flagged in either object. Using the compartment objects naively would have tested 590 of 1,043 LR genes in blood against 1,028 in skin, and would have silently dropped CXADR and IL22RA1 from the prior-axes table.
- **The blood NK/CD8 labels are mixtures.** The blood object holds 606 NK against 112,609 CD8. `README.md` states that nb31 called CD8 on `CD8A`, which NK cells express as a homodimer, and that `blood_T_celltype_corrections_v1.csv` corrects only the 12 clusters the nb32b review examined while roughly 100 further CD8-labelled clusters carry the same signature. The blood CD8 and NK partner rows should be read as mixtures.
- **`atlas_obs_full_v2.parquet` carries stale columns.** Its `has_tcr` is wrong for li2024 (`helpers/malignancy.py` documents 0.072 there against 0.476 in the V(D)J fold-in cache), and its `is_malignant` is an older atlas-level call that disagrees with every current one. Neither is used here.
- **`_v2`/`_v3` filename suffixes are not recency markers** in this tree — `CLAUDE.md` records that `skin_T_annotated_v2.h5ad` was older than the unsuffixed file. Every input above was pinned by reading the module that owns it, not by its suffix.

## 11. File manifest

| file | contents |
|---|---|
| ccc_malignant_vs_benign_cd4_report.md | this report |
| de_lr_genes_skin.csv | every tested LR gene in skin, with the top-hit columns |
| de_lr_genes_blood.csv | every tested LR gene in blood |
| de_all_genes_skin.csv | the full-transcriptome fit for skin (no BH outside the LR family) |
| de_all_genes_blood.csv | the full-transcriptome fit for blood |
| interactions_skin.csv | CD4 gene × cognate gene × partner type passing the gate, skin |
| interactions_blood.csv | the same for blood |
| cell_counts_donor_arm.csv | cells per donor × tissue × arm, with eligibility |
| prior_axes.csv | the pre-specified axes plus the TOX control, both tissues |
| partner_types.csv | every candidate partner type with its pooled cell count and the keep flag |
| partners.csv | the partner types that passed the cell-count floor, with their indices |
| cohort_excluded.csv | every unit removed by a cohort rule, with the reason |
| blood_benign_subclusters.csv | per-cluster marker detection for the blood benign cleanup |
| units.csv | the eligible donor × arm pseudobulk index |
| lr_resource_1422.csv | the rebuilt filtered resource (1,422 interactions) |
| lr_resource_consensus_full.csv | unfiltered LIANA consensus with per-pair support |
| lr_genes.json | the 1,043 unique genes of the filtered resource |
| _expected_unit_cells.csv | cells per pseudobulk, written by `prep` and asserted by `pass` |
| pseudobulk_skin.parquet | donor × arm summed raw counts, skin (full gene space) |
| pseudobulk_blood.parquet | donor × arm summed raw counts, blood (full gene space) |
| pass_accumulators.npz | the streamed count sums and detection counts |
| row_labels.npz | cell → (donor × arm) and cell → partner-type label vectors |

_Figures are opt-in: the `fit` stage writes them only when run with `--figures`._

Produced by `jobs/run_ccc52_mvb.py` (stages `lr`, `prep`, `pass`, `fit`), submitted with `jobs/run_ccc52_mvb.sh <stage>`. Nothing under `data/` was written.