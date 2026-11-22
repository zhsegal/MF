# What has been found, and which annotation it used

**Read the `annotation` column before quoting a number.** `v1` means the result was built on
`cell_type_final == "CD4"`, whose benign arm is only ~46% conventional CD4 in skin
(`CAVEATS.md#the-cd4-label-was-a-default`). Those results are not wrong about the malignant side,
but any statement about the *benign* side, or any gene that differs between CD4 and CD8/NK/Treg,
is confounded.

| Result | Where | Annotation | Headline |
|---|---|---|---|
| **Non-malignant T/NK re-annotation** | `reports/nonmal_T_reannot/` | — (it *is* the new annotation) | 576,188 cells relabelled from a scVI embedding the malignant cells never entered. Benign-arm CD8A/B+ fraction falls 0.199→0.089 (skin), 0.141→0.036 (blood); FOXP3+ roughly halves. Blood ADT confirms independently: CD4_conv 80% CD4-protein+, CD8 71% CD8-protein+, NK_ILC double-negative. |
| **Lineage migration check** | `reports/lineage_check/` | v2 | The v1 skin benign arm was **45.8% conventional CD4**: 16.1% CD8, 14.9% Mixed, 10.9% Treg, 7.7% NK/ILC. Blood: 61.5% CD4_conv. |
| **QC of the benign CD4 arm** | `reports/ccc52_mvb/qc_benign_arm/` | v1 (by design — it characterises the old arm) | 21.9% of skin benign cells sit in clusters with CD8A/B detection ≥0.25; it is real cells, not ambient (benign arm is *shallower* yet has 14× the CD8A detection) and not doublets (Scrublet no higher in CD8A/B+ cells). |
| **Malignant vs benign CD4 cell–cell communication** | `reports/ccc52_mvb/` | **v1** | skin 27 significant LR genes of 411 tested (7 up, 20 down) over 71 donors; blood 23 of 294 (13 up, 10 down) over 38. TOX +1.16 skin / +2.72 blood as positive control; CD70, KIR3DL2 +3.92, CCR4 recovered. **Re-run on the clean arm is pending** (`ANALYSES.md`). |
| **CD70 payload selectivity** | `reports/51_payload_selectivity.md` | **v1** | TRUCK payload screen, gates 1–3, on `mal_cluster`. |
| **Surface-target screens (tgt41–49)** | `tables/tgt4*_llm_export.md`, `tables/tgt4*.csv` | **v1** | The skin/blood surface screens. Note `tables/` is gitignored — these exist on disk only. `tgt49` is the one with the gaydosik2023 repair applied. |
| **Degradome / TME** | `tables/deg28mc_*` | **v1** | Cluster-call variant of the degradome analysis. Produced by an archived notebook. |

## Calibration anchors

Things that must come out right, or the pipeline is broken rather than interesting:

- **TOX up in malignant** in both tissues. It is the canonical CTCL marker.
- **CD70, KIR3DL2, CCR4** — approved or registrational CTCL targets — recoverable with the right
  sign. KIR3DL2 +4.16 in blood (34/38 donors concordant) is the strongest.
- **DPP4 (CD26) down** in malignant blood. Loss of CD26 is a clinical Sézary criterion.
- **CD8A/CD8B/KLRD1/TRDC/FOXP3 "down in malignant" is an artifact**, not biology — it is the
  benign arm's contamination showing through. Under the clean arm skin KLRD1 moves −4.94 → −1.25
  and TRDC −3.61 → −1.34, while TOX and CD70 are unmoved. Treat any result built on those five
  genes with suspicion.
