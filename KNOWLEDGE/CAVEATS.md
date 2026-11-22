# Known defects and traps

Each entry says what is wrong, how big it is, and what it breaks. If your analysis touches the
data named in an entry, that entry applies to you.

---

## The CD4 label was a default

**What.** `cell_type_final` / `cell_type_T` was assigned **per Leiden cluster, by hand, from dot
plots**, separately in nb21 (skin) and nb31 (blood). Every T cluster not positively called
something else became `CD4`. There is no per-cell CD4 test anywhere in that pipeline, and in
blood the `CD4` transcript was not even in the object's 10k HVG when the call was made, so the
marker was never available. About 20% of blood `CD4` (108,677 cells) comes from a second pass
over the `Unk` residue.

**How big.** Two measurements, which answer different questions — quote the right one.

*Cell level* (`reports/lineage_check/`, the re-annotation's verdict on each cell): the skin v1
benign arm of 180,978 cells is **45.8% conventional CD4**. The rest is CD8 16.1%, unresolved
(`Mixed`) 14.9%, Treg 10.9%, NK/ILC 7.7%, Prolif 2.2%, LowQC 1.2%, Suspect_malignant 0.8%,
γδ 0.5%. Blood: 61.5% CD4_conv, plus naive 14.0% (excluded by the blood arm convention),
Treg 9.8%, CD8 7.7%.

*Cluster level* (`reports/ccc52_mvb/qc_benign_arm/`, the diagnostic that found the problem):
161 of 479 skin benign-arm clusters — 21.9% of its cells — have CD8A/CD8B detection ≥ 0.25, and
81 clusters (9.4%) ≥ 0.50, at det_CD8A 0.85–0.97 and det_CD4 ≤ 0.16. Tregs leak in as well: skin
benign FOXP3 detection 0.083 against 0.010 in the malignant arm, and those cells are CD40LG-low,
i.e. real Tregs.

The cluster figure is lower than the cell figure because it only counts cells in clusters that
cross a detection threshold; the cell figure counts every reassigned cell.

**It is real cells, not ambient and not doublets.** Inside the benign arm CD4 is detected in 3.2%
of CD8A/B+ cells against 20.8% of CD8A/B− cells (skin); ambient and doublets would raise both.
The benign arm is also the *shallower* arm (1,982 vs 3,412 median UMI) yet has 14× the CD8A
detection, and Scrublet scores are no higher in its CD8A/B+ cells.

**Fixed, forward.** `data/atlas_joint/nonmal_T_labels_v1.parquet` re-annotates all 576,188
non-malignant T/NK cells from a scVI embedding the malignant cells never entered. Use
`helpers/lineage.py` — never a literal. The new benign arm drops the per-cell CD8A/B+ fraction
from 0.199 → 0.089 (skin) and 0.141 → 0.036 (blood).

**Still broken.** Every result produced before 2026-11-22 used the old arm. See `RESULTS.md` for
which. `mal_cluster` itself was derived from the old labels and is deliberately **not** rebuilt —
see `DECISIONS.md#mal_cluster-is-frozen`.

---

## gaydosik2023 (D14) has empty count rows

**What.** In `data/atlas_joint/skin_T_tcr_annotated_v4.h5ad`, **1,083 of 1,147** gaydosik2023
cells have an empty count row (`indptr[i+1] == indptr[i]`) in both `X` and `layers/raw_counts`,
while `obs` still carries an intact `total_counts` (median 3,774) and `n_genes`. Per donor:
SC320 140/140, SC348 494/494, SC353 253/253, SC354 111/111, SC405 85/149.

**Why it bites.** All five are healthy donors — **24% of the 21-donor CD4_HC arm** of the skin
target screens. A pseudobulk builder divides a row of zeros by a plausible library size and emits
a donor of pure zeros that passes the ≥20-cell floor. That is how `tgt49_all_genes.csv` got an
`hc_median_pct` biased low for essentially every gene, and a `q_vs_hc` that moved for 370 of 388
surface genes once repaired.

**Recoverable exactly.** All 1,147 cells match
`data/D14_gaydosik2023_skin/processed/standardized.h5ad` by cell name; `total_counts` is identical
for 1,147/1,147 and recovered row sums reconcile to 0.9966–1.0000 of it. The working repair is
`repair_d14()` in `jobs/tgt49_export_donor_long.py`. Caveat: D14's own `var` is 27,852 genes, so
205 of the 2,355 screened genes (mostly olfactory receptors) are genuinely unmeasured there and
must be NaN, not 0.

**Still broken.** The atlas object has **not** been fixed — only that one export repairs it. Any
new analysis touching the skin CD4_HC arm has the bug until the atlas is rebuilt.

---

## blood_T_annotated.h5ad is a 10k-HVG subset

`blood_T_annotated.h5ad` is 667,584 × **10,000** genes; `skin_T_annotated.h5ad` is full-gene
(539,916 × 42,347). Of the 1,043 ligand–receptor genes, skin holds 1,028 and blood only 590;
`CXADR` and `IL22RA1` are absent from blood entirely, and so is **`CD4`**.

**34% of blood UMIs sit outside the blood 10k HVG.** For anything gene-wide in blood, read
`joint_annotated.h5ad` instead — it is also the only object containing non-T cells for either
compartment. Nothing in the objects flags this asymmetry; it will silently halve a blood result.

---

## Blood CD8 and NK are mixtures

nb31 called CD8 on `CD8A`, which NK cells express as a homodimer. `README.md` ("Blood" section) records that
roughly **100 blood clusters carry an NK signature and were never corrected**; only 12 clusters
(1,716 cells) were relabelled, via `blood_T_celltype_corrections_v1.csv`, which is applied **on
read only** (`helpers/cluster_malig_helpers.py:apply_celltype_fix`) and never written back.
A blood NK count of 606 out of 667,584 cells is not a credible NK fraction.

In skin the same error runs the other way: `notebooks/_archive/20_skin/22b_unassigned_reannotation.ipynb`
states that **every NK and ILC cell in the atlas was called CD8**, and 55,026 of them still are.

The re-annotation (`nonmal_T_labels_v1`) fixes this for **non-malignant** cells — it separates
NK_ILC from CD8 explicitly. The old columns still carry the mixture.

Also: `blood_T_celltype_corrections_v1.csv` records `cell_type_T_nb31 = CD8` for all 1,716 rows,
but 66 of them are `CD4` in the h5ad — the "from" label is wrong for those rows.

---

## 10x Flex is a probe panel

`tech == "10x_flex"` (li2024, skin) is a probe-panel chemistry: genes outside the panel are
**exactly zero in every cell**, not lowly detected. Any gene filter that treats zero as "not
expressed" will silently drop them for that cohort, and any cross-chemistry comparison on an
off-panel gene is comparing a measurement to a structural zero.

The test that catches it: a gene is unmeasured in a dataset when it has zero counts there while
the tissue-wide detection rate predicts ≥ 10 detections. `jobs/run_nonmal_T_reannot.py` uses
exactly this; it flags 9,945 genes in skin and 8,206 in blood.

---

## geskin2026 is ex vivo, and its sample_id is a lane

geskin2026 (blood only, 260,455 cells, 11 `patient_key`) is **100% ex vivo 2-day culture**
(213,181 drug-treated, 47,274 vehicle). It is not in-vivo biology and is dropped from the CCC
cohort. Its `sample_id` is an **HTO lane pooling 5–6 donors**, so 54 of the 131 blood donor keys
are lane×HTO wells rather than people; the HTO→patient map was never deposited
(`docs/DATA_PROVENANCE.md:61`). Five of its healthy donors are mislabelled SS in the atlas
`disease` column — `helpers/target_cohort.py` carries the corrected truth.

---

## doublet_score exists but predicted_doublet is always False

Per-sample Scrublet ran in `qc_filter` (`helpers/atlas_join_helpers.py:1124-1152`) and
**predicted doublets were deleted before the join** (`:1153`). So `predicted_doublet` is `False`
for all 2,157,693 cells and retained scores top out near 0.25 — a "fraction above threshold" is
empty by construction. `doublet_score` is null for li24 (author-prefiltered) and for samples under
30 cells. No `solo` or `DoubletDetection` scores exist.

---

## ADT exists for exactly one dataset

buus25 only: `joint_annotated.h5ad` `obsm/X_adt`, a dense float32 2,157,693 × 128 matrix that is
**NaN for every other cell**, with names in `uns/adt_panel`. It covers 8 of the 38 CCC-eligible
blood donors. The panel carries **both** `ADT_CD8A` (93.7% of buus25 cells) and `ADT_CD8a` (6.7%)
— they are one marker and must be merged, never intersected.

**The trap:** a marker with no counts at all still yields a CLR value (minus the cell's mean
log1p) that is identical across all such markers and tracks total protein capture. Two-means will
split it cleanly and you will report library size as "TCRgd+". Require real counts in ≥1% of a
donor's cells before gating. In skin, `ADT_TCRgd` and `ADT_CD56` are exactly this case.

ADT for geskin26, herrera21, GSE182861 and GSE293752 was dropped during the atlas join and is not
recoverable from the atlas objects.

---

## v1 and v2 artifacts share filenames

`notebooks/_archive/10_atlas/11_atlas_v2_qc.ipynb` exists because a notebook once reported v1
numbers on a v2 input. Before quoting anything, confirm the input is a v2 object
(`atlas_obs_full_v2.parquet`, `joint_*`, `skin_T_annotated.h5ad`, `blood_T_annotated.h5ad`), and
note that **`_v2`/`_v3` suffixes are not reliable recency markers** — `skin_T_annotated_v2.h5ad`
was older than the unsuffixed file. The v1 freeze is `data/_archive_v1/`, which is read-only.

Malignancy call revisions are versioned in the filename (`skin_T_malignancy_v5.parquet`,
`blood_T_malignancy_v4.parquet`); check which revision a result pinned before comparing.

---

## The LR resource is not a file

The filtered ligand–receptor resource this project calls "≥6 of 15 database support, 1,422
interactions" **does not exist on disk**. It is LIANA `consensus` (4,624 pairs) ∩ pairs supported
by ≥6 of the 15 human LIANA resources = **1,422 interactions, 1,043 genes**, rebuilt on demand by
`jobs/run_ccc52_mvb.py --stage lr` (which asserts both numbers). Support ≥6 without the consensus
intersection gives 1,857, not 1,422. `liana` lives in **`mrvi_env`**, not `neural_nmf_env`.

**Never apply an LR blacklist.** `docs/CCC_PIPELINE_PORTABLE_SPEC.md:83` states `LR_CAVEATS` is
"commentary only, never a blacklist" — the v1 version silently deleted the IL4/IL13 pairs the
analysis existed to report (`helpers/ccc_utils.py:1157`). Carry it as an annotation column.
