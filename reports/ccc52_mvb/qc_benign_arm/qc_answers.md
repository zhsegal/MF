# What the labels actually are (QC of the benign CD4 arm, §2)

> **Annotation vintage: v1, by design.** This report characterises the *old* benign CD4 arm —
> that is its purpose. It is the diagnostic that motivated
> `data/atlas_joint/nonmal_T_labels_v1.parquet`. See `KNOWLEDGE/RESULTS.md`.

Read-only provenance review of the labels the v1 fit (`jobs/run_ccc52_mvb.py`) depends on.
Notebook citations are `notebook / CELL n`, because `.ipynb` JSON has no stable line numbers.

---

## 1. How `cell_type_final == "CD4"` is assigned

**Per cluster, by hand, from dot plots — in both tissues.** There is no per-cell classifier, no
marker-score threshold, no scANVI/kNN label transfer and no majority vote. I found no literal
`np.where(is_cd8, "CD8", "CD4")` or equivalent else-branch anywhere. But **"CD4" is still an
effective default**, because the hand maps assign CD4 to every T cluster that was not positively
called something else.

### Skin — `notebooks/20_skin/21_reannotation.ipynb` → `data/atlas_joint/skin_cell_type_final.csv` (CELL 32), `skin_T_annotated.h5ad` (CELL 36)

1. Broad Leiden on `X_mrvi_u` (CELL 9; exact below a size cap, seeded subsample + kNN vote above
   it). Hand map `cluster2ct` (CELL 15) sends only broad clusters 14, 15, 16 to `"T"`.
2. T-only Leiden, res 1.0 (CELL 19). Hand map `cluster2ct_T` (CELL 23): **every cluster → `CD4`**,
   except cluster 5 → `CD4_Treg` (41,011 cells, which is the final Treg count), cluster 10 →
   `UNK`, cluster 17 → `CD8` (**46 cells**).
3. The UNK block (cluster 10, 58,792 cells) is re-clustered at res 0.5 (CELL 25); `cluster2ct_unk`
   (CELL 28) sends sub-clusters 0,1,2,4,5,6 → `CD8` and 3,7 → `CD4` (+2,915 cells into CD4).
4. Merge (CELL 32) is a straight substitution with an assert that nothing is left NaN — no
   fallback branch.

Final skin vocabulary in `cell_type_T`: `CD4` 442,982, `CD4_Treg` 41,011, `CD8` 55,923. **No gdT,
MAIT or NK label exists in skin.** NK/γδ marker rows (`NCAM1`, `KLRD1`, `KLRF1`) were plotted in
CELL 21 and never given a label.

So skin `CD8` is almost entirely the UNK-derived block, and `22b_unassigned_reannotation.ipynb`
CELL 0 states outright that **"every NK and ILC cell in this atlas is currently called CD8"**
(55,026 of those cells are still labelled `CD8` in `skin_unassigned_subtypes.csv`). The documented
error therefore runs CD8 ← NK/ILC, not CD4 ← CD8.

### Blood — `notebooks/30_blood/31_reannotation.ipynb` → `blood_cell_type_final.csv`, `blood_T_annotated.h5ad` (CELL 44); provenance `blood_annotation_provenance.json`

1. Broad Leiden, hand map in the provenance JSON.
2. T-subset Leiden; `cluster2ct_T` (CELL 31): `CD4` ← clusters 0,5,7,8,9,10,12,13,14,15,16;
   `CD8` ← 1,3; `NK` ← 17 (606 cells); `Unk` ← 2,4,6,11 (cluster 6 alone is 118,262 cells).
3. The 140,954 `Unk` cells are re-clustered at res 0.5; `cluster2ct_unk` (CELL 36) sends
   sub-clusters 0,1,2,3,6,7,8 → `CD4` and 4,5 → `CD8`.

Counts reproduce exactly: CD4 = 445,692 + **108,677** = 554,369; CD8 = 80,332 + 32,277 = 112,609;
NK = 606; total 667,584. **≈20% of blood "CD4" comes from the Unk residue pass.**

`VOCAB_T` (CELL 1) permits `gdT`, `MAIT`, `CD4_Treg`, `Prolif`, `LowQC`, `Doublet`, but the hand
maps never use them — the `{CD4, CD8, NK}` trichotomy is by omission, not by test. This matches
`helpers/cluster_malig_helpers.py:89-94`.

**The decisive detail for this QC: `CD4` is not in the blood 10k-HVG object.** The blood
annotation dot plot's "CD4" row carried only `IL7R` and `CD40LG`; `filter_panel` (CELL 1) silently
drops absent genes. The `CD4` transcript was *never available* when blood CD4 was called
(`CD8A`/`CD8B` were). Skin's 10k HVG does contain `CD4`.

**Can CD8 / NK / γδ / Treg end up under "CD4"?** Nothing prevents it:
- Blood has **no Treg label at all**, so blood Tregs are inside `CD4` by construction (v1 removed
  them by sub-clustering; see `blood_benign_subclusters.csv`).
- No γδ or MAIT label exists in either tissue.
- `blood_T_celltype_corrections_v1.csv` (written by `32b_cluster_malignancy.ipynb`) moves 1,716
  cells from 12 clusters to `NK`; it is applied **on read only**, in
  `helpers/cluster_malig_helpers.py:~198` (`apply_celltype_fix`) — the h5ad and the cluster
  parquet are not rewritten. Its `cell_type_T_nb31` column says `CD8` for all 1,716, but 66 of
  them are `CD4` in the h5ad, so the correction silently moves 66 **CD4** cells to NK. Rule:
  `nk_score − t_score ≥ 1.0 AND tcr_rate < 0.15` (`blood_T_celltype_corrections_v1.json`).

## 2. The per-sample cluster key behind `mal_cluster`

Column **`cluster`**, format `<group>__<k>`; per-group Leiden res 1.0 on 2,000 HVG / 30 PCs / 15
neighbours (`skin_T_clusters_v1_manifest.json`, `blood_T_clusters_v1_manifest.json`;
`helpers/malignancy.py:16`).

| tissue | cluster file | rows | groups | clusters | call file |
|---|---|---|---|---|---|
| skin | `data/atlas_joint/skin_T_clusters_v1.parquet` | 538,808 | 169 `sample_id` | 1,550 | `skin_T_malignancy_cluster_v1.parquet` |
| blood | `data/atlas_joint/blood_T_clusters_v1.parquet` | 667,584 | 131 | 1,438 | `blood_T_malignancy_cluster_v1.parquet` |

Columns, both: `cell_id, sample_id, cluster, umap_1, umap_2, top_trbv, top_trav, cell_type_T,
tcr_ok, family`. Call files: `cell_id, sample_id, cluster, call_source, mal_cluster,
mal_cluster_prob`.

**Grouping differs between tissues**: skin groups by `sample_id`
(`helpers/cluster_malig_helpers.py:52`), **blood groups by `donor`** (`:77`; `README.md:93-94`) —
the blood parquet still *names* the column `sample_id` but it holds donor IDs. Flagged below as an
upstream naming error.

**Availability.** Blood: complete (667,584 / 667,584, no NA `mal_cluster`). Skin: the cluster file
covers 538,808 of the 539,916 cells in `skin_T_annotated.h5ad` (it was built from
`skin_T_tcr_annotated_v4.h5ad`), and **45,496 skin cells have `mal_cluster` NA** — those cells are
in neither arm in v1 and so are outside this QC too. Every eligible donor in `units.csv` has
clusters for both arms by construction: `mal_cluster` is a *per-cluster* call, so each cluster is
wholly malignant or wholly benign and a donor with ≥50 cells per arm necessarily has clusters on
both sides. Call sources — skin: tcr 262,051 / auto 208,947 / unassessed 45,306 / user 17,705 /
hc 4,609 / excluded_user 190; blood: tcr 491,436 / auto 128,856 / hc 44,290 / user 3,002.

## 3. Surface protein and doublet scores

**ADT: yes, for one dataset.** buus25 (GSE284075, 10x 5′, 127-plex ADT; `docs/DATA_PROVENANCE.md:42`)
is the only panel that survived the atlas join. It lives in **`joint_annotated.h5ad`
`obsm/X_adt`** — a 2,157,693 × 128 float32 matrix, NaN for every non-buus25 cell — with names in
`uns/adt_panel` (`helpers/atlas_join_helpers.py:1230-1292, 1599-1630`). The panel includes
`ADT_CD3`, `ADT_CD4`, `ADT_CD8A`, `ADT_CD8a`, `ADT_TCRab`, `ADT_TCRgd`, `ADT_CD56`, `ADT_CD25`,
`ADT_CD127`, `ADT_CD16`. Coverage: 72,378 blood and 55,483 skin cells in the MRVI inputs;
61,215 in `blood_T_annotated.h5ad` (17 donors, all SS, CD4 51,648 / CD8 9,567).
`skin_T_annotated.h5ad` has no `X_adt` — which is why this QC reads ADT from the joint object, the
only place skin ADT survives. Dropped elsewhere: geskin26 19-marker ADT + HTO
(`helpers/atlas_join_helpers.py:1166-1172`), herrera21 ECCITE 49-ADT
(`docs/DATA_PROVENANCE.md:46`), GSE182861 / GSE293752 20-marker panels
(`docs/ATLAS_V2_DEDUP_GATE.md:152-154`). → `qc_adt.csv` is therefore restricted to the eligible
buus25 donors; it exists but it does not cover the cohort.

**Doublet scores: yes, but already acted on.** `doublet_score` and `predicted_doublet` are columns
of `atlas_obs_full_v2.parquet` and of both T h5ads; `blood_cluster_qc.csv` carries `med_doublet`
per cluster. Source is per-sample `sc.pp.scrublet` in `qc_filter`
(`helpers/atlas_join_helpers.py:1124-1152`). **Predicted doublets were deleted before the join**
(`:1153`, `adata = adata[~dflag].copy()`), so `predicted_doublet` is `False` for all 2,157,693
cells and the retained scores top out near 0.25. `doublet_score` is null for li24 (author-
prefiltered; `docs/CCC_PIPELINE_PORTABLE_SPEC.md:257`) and for samples under 30 cells (`:1135`).
No `solo` / `DoubletDetection` scores exist. → `qc_doublet.csv` is informative only through the
*median score* and the CD8A/B+ vs CD8A/B− contrast, not through a threshold.

## 4. Other recorded facts that bear on the question

- **~100 blood CD8-labelled clusters carry an NK signature and were never corrected**
  (`README.md:85-91`); only 12 clusters / 1,716 cells were relabelled. The README states blood CD8
  and NK "are mixtures until nb31 is redone", and that **606 NK of 667,584 cells is not a credible
  NK fraction**. nb31 called CD8 on `CD8A`, which NK cells express as a homodimer
  (`README.md:85-87`; `docs/XENIUM_5K_ADDON_SPEC.md:187-190` estimates ~25% of blood CD8 is
  mislabelled). **Consequence for this QC: the `CD8` and `NK` reference rows in
  `qc_group_reference.csv` are themselves mixtures — read them as "cytotoxic lymphocyte", not as
  pure CD8.**
- The repo records **no** systematic check that `CD4` contains CD8/NK/γδ cells. Every documented
  mislabel runs in the CD8↔NK direction. That is a gap in the record, not evidence of purity.
- **Ambient soup.** `31b_subtype_reannotation.ipynb` labels 25,791 blood non-T cells
  `Ambient_soup`, kept in the object (`README.md:79-80`); v1 drops them from partner types.
  `docs/TCR_costim_gene_lists_and_interpretation.md:186` notes that GZMB and reactive CD8/NK
  transcripts bleed into malignant calls in droplet data.
- **Blood benign arm is intrinsically small**: ~92% of blood CD4 is malignant, and
  `helpers/cluster_malig_helpers.py:102` relaxes `MIN_CD4` to 0.5 for blood against 0.8 for skin.
- **`mal_cluster` is benign-by-rule for HC samples** (`helpers/malignancy.py:35-37`), so apparent
  specificity there is an input, not a measurement.
- `data/atlas_joint/atlas_v2_qc_provenance.json` records verdict `BLOCKED` (PASS 146, WARN 19,
  FAIL 8) dated 2026-09-16, before the annotation work above.

## Upstream errors noticed

1. `blood_T_clusters_v1.parquet` names its grouping column `sample_id` while it holds **donor**
   IDs (`helpers/cluster_malig_helpers.py:77`). Anything that joins blood clusters on `sample_id`
   against the atlas obs will silently mismatch.
2. `blood_T_celltype_corrections_v1.csv` records `cell_type_T_nb31 = CD8` for all 1,716 cells, but
   66 of them are `CD4` in `blood_T_annotated.h5ad` — the "from" label is wrong for those rows.
3. `blood_annotation_provenance.json` names the producing notebook `33_blood_reannotation.ipynb`;
   the repo notebook is `30_blood/31_reannotation.ipynb`.
4. `data/atlas_joint/README.md` is stale (noted in `docs/DATA_PROVENANCE.md:~227`).
