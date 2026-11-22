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
| **Malignant vs benign CD4 cell–cell communication** | `reports/ccc52_mvb_v2/` | **v2 — current** | skin 27 significant LR genes of 372 tested (9 up, 18 down) over **56** donors; blood 27 of 303 (17 up, 10 down) over **45**. TOX +1.15 skin / +2.93 blood; CD70 +1.96 / +2.96; KIR3DL2 +1.01 / +4.66. `annotation_shift.csv` gives every gene's v1 and v2 effect side by side. |
| ⤷ the same contrast on the v1 arm | `reports/ccc52_mvb/` | **v1 — superseded** | skin 27 of 411 (7 up, 20 down) over 71 donors; blood 23 of 294 over 38. Regenerable with `--annot v1`. Kept for provenance; do not quote a gene whose `annotation_shift.csv` verdict is not `stable`. |
| **Knowledge-base consistency** | `reports/knowledge_check/` | n/a — reads no cell data | Bookkeeping check of `KNOWLEDGE/` against the file tree. Re-run it after any analysis; the count is only meaningful on the day it ran. |
| **CD70 payload selectivity** | `reports/51_payload_selectivity.md` | **v1** | TRUCK payload screen, gates 1–3, on `mal_cluster`. |
| **Surface-target screens (tgt41–49)** | `tables/tgt4*_llm_export.md`, `tables/tgt4*.csv` | **v1** | The skin/blood surface screens. Note `tables/` is gitignored — these exist on disk only. `tgt49` is the one with the gaydosik2023 repair applied. |
| **Degradome / TME** | `tables/deg28mc_*` | **v1** | Cluster-call variant of the degradome analysis. Produced by an archived notebook. |

## Calibration anchors

Things that must come out right, or the pipeline is broken rather than interesting:

- **TOX up in malignant** in both tissues. It is the canonical CTCL marker.
- **CD70, KIR3DL2, CCR4** — approved or registrational CTCL targets — recoverable with the right
  sign. On the v2 arm KIR3DL2 is **+4.66 in blood, 41/45 donors concordant**, the strongest
  single result in the analysis. In **skin** KIR3DL2 is +1.01 and nominally significant but only
  **22/56 donors concordant** — under half, so it is a pooled effect, not a per-patient one.
  Always quote `donors_concordant` next to a skin effect size.
- **DPP4 (CD26) down** in malignant blood. Loss of CD26 is a clinical Sézary criterion.
- **CD8A/CD8B/KLRD1/TRDC/NKG7/FOXP3 "down in malignant" was an artifact in skin**, not biology.
  Measured v1 → v2 (`reports/ccc52_mvb_v2/annotation_shift.csv`): skin KLRD1 −6.02 → **−2.41**,
  TRDC −5.71 → **−2.41**, NKG7 −3.77 → **−0.87**, CD8A −4.74 → −3.47, FOXP3 −3.94 → −2.55, while
  TOX is unmoved (1.16 → 1.15) and CD70 strengthens (1.20 → 1.96). 23 skin genes shift by ≥ 1.5.
- **In blood the same genes move the OTHER way** — NKG7 −3.55 → −6.52, KLRD1 −3.45 → −5.17 —
  because the v1 blood arm was flat `CD4` with naive/Treg clusters stripped by a marker rule,
  while v2 is `CD4_conv` proper and its benign cells carry *more* NKG7/CCL5. Blood FOXP3 does
  collapse as expected (−1.72 → −0.12). A v1→v2 shift is the benign arm moving, never the clone.
- **`label_defining_gene == True` is circular by construction.** The v2 benign arm is selected on
  a 28-gene marker panel (CD4, CD40LG, CD8A/B, FOXP3, CCR7, SELL, NKG7, GZMK, KLRF1, TRDC, …), so
  those genes separate the arms by definition. Skin `CD4` and `CD40LG` are "significant" under v2
  purely for this reason. Every v2 table carries the flag.
