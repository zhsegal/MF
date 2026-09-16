# v1 parameter ledger — freeze before the v2 rebuild

**Purpose.** v2 must reproduce v1's method exactly: the atlas changes because the *cohort set*
changes, not because a knob moved. Every value below is extracted from v1 source, with the
file it lives in. Change nothing here during the rebuild; if a value must change, it is a
separate decision with its own before/after.

**Date:** 2026-09-05.

---

## 0 · Why an exact replay is possible

Everything upstream of the hand-authored cell-type maps is deterministic:

- `scvi.settings.seed = 0` (`jobs/run_mrvi_joint.py:157`), `SEED = 0` in `21_reannotation`/`31_reannotation`.
- `sc.pp.neighbors(..., random_state=SEED)`, `sc.tl.umap(..., random_state=SEED)`,
  `sc.tl.leiden(..., random_state=SEED, flavor="igraph", n_iterations=2, directed=False)`.
- Scrublet is seeded: `sc.pp.scrublet(sub, random_state=0)` (`atlas_join_helpers.py`, in
  `qc_filter`).

**Consequence for the held cohorts:** QC runs per cohort and Scrublet runs per sample, so a
cohort's `standardized.h5ad` depends on nothing outside itself. The ten v1 caches are
therefore already bit-identical to what a rebuild would produce, and v2 **reuses** them
rather than rebuilding. Only the new cohorts are built. This also means the v1↔v2 regression
on held cohorts is an exact identity, except for the one deliberate change in §4.

## 1 · MrVI  (`jobs/run_mrvi_joint.py`)

| knob | value | line |
|---|---|---|
| HVG | `seurat_v3`, `n_top_genes=10000`, `batch_key="study"`, `layer="raw_counts"`, `subset=True` | 112-115 |
| sample / batch key | `sample_key="sample_id"`, `batch_key="study"` | 31-32 |
| counts layer | `raw_counts` | 33 |
| latent | `n_latent=30`, `n_latent_u=10` | 44-45 |
| training | `max_epochs=100`, `batch_size=256`, `early_stopping=True`, `early_stopping_patience=15`, `check_val_every_n_epoch=1`, `train_size=0.9` | 40-42, 171-175 |
| seed | `0` | 53, 157 |
| subsample | `0` = fit on all cells | 47 |
| diagnostics | `batch_size=512`, `get_local_sample_distances(groupby="cell_type", keep_cell=False)`, `differential_abundance(sample_cov_keys=["disease","stage_class"], compute_log_enrichment=True)` | 46, 34, 215, 224-228 |
| runs | untagged (full atlas) · `--tag skin --subset-compartment Skin` · `--tag blood --subset-compartment Blood` | `run_mrvi_joint.sh` |

`--force` is **required** on every v2 run: without it `build_hvg_input` reuses the cached
`joint_mrvi_input*.h5ad` and trains on the old cell set (`run_mrvi_joint.py:92-94`).

v1 measured cost (from `jobs/*.bsub.log`): full 5.56 h / 70 GB · skin 2.66 h / 82 GB ·
blood 1.47 h / 10 GB · DA 2.47 h / 12 GB. **v2 is ~2x the cells — re-provision, do not reuse
these numbers.**

## 2 · Skin re-annotation  (`20_skin/21_reannotation.ipynb`)

| knob | value | cell |
|---|---|---|
| seed | `SEED = 0` | c1 |
| representation | `use_rep="X_mrvi_u"` (sample-unaware) | c7, c20, c26 |
| neighbours | `sc.pp.neighbors(ad, use_rep="X_mrvi_u", random_state=SEED)` — scanpy default `n_neighbors=15` | c7 |
| UMAP | `sc.tl.umap(ad, random_state=SEED)` | c7 |
| Leiden (broad) | `resolution=0.7`, `flavor="igraph"`, `n_iterations=2`, `directed=False` → **21 clusters in v1** | c9 |
| Leiden (T) | `resolution=1.0`, same flags | c20 |
| Leiden (2nd-pass unknown T) | `resolution=0.5`, same flags | c26 |
| unknown labels | `{"UNK","Unknown","unknown",""}` | c26 |

**Broad label vocabulary (v1 `cluster2ct`, c15):**
`Keartinocyte, T, Myeloid, Fibroblast, Mast, Plasma, B, Melanocyte, Vascular, UNK`.

> ⚠️ `"Keartinocyte"` is a **typo in v1** and it propagates into
> `data/atlas_joint/skin_cell_type_final.csv` and every downstream table keyed on that
> string. Preserving it keeps v1↔v2 diffs clean; fixing it is one line but breaks string
> equality with every v1 table. **Decide before authoring the v2 map, not after.**

## 3 · Blood re-annotation  (`30_blood/31_reannotation.ipynb`)

| knob | value | cell |
|---|---|---|
| seed | `SEED = 0` | c1 |
| switches | `LATENT_SCOPE="blood"`, `INCLUDE_LN=False`, `FULL_GENES=False`, `ALLOW_JOINTLAT_LABELS=False` | c1 |
| scope rule | blood arm is **T-cell restricted** (`cell_type_broad == "T"`) | §1b of `DATA_PROVENANCE.md` |
| neighbours / UMAP / Leiden | as skin (`X_mrvi_u`, seeded, igraph, `n_iterations=2`) | c10, c12, c27, c34 |
| broad clusters in v1 | **27** | c19 |
| map guard | `apply_hand_map` validates `cluster2ct` against a pasted `CLUSTER_SIZES_BROAD`, so v1's map cannot be silently reused | c19 |

**`VOCAB_BROAD`:** `T, NK, B, Plasma, Mono_CD14, Mono_CD16, cDC, pDC, Mast, Platelet,
Erythroid, HSPC, Prolif, LowQC, Doublet, UNK`.

> ⚠️ `31_reannotation` c1 hardcodes `N_TOTAL, N_BLOOD, N_LN, N_SKIN = 1_173_694, 423_042, 1_142, 749_510`.
> These are assertions about the atlas, not parameters — they **must** be updated for v2.

`21_reannotation` has no equivalent size guard, only an assert on the key set. Add the
`CLUSTER_SIZES_BROAD` pattern there so a stale map fails loudly.

## 4 · Deliberate departures from v1

Three, all decided explicitly on 2026-09-05. Everything else is an exact replay.

### 4a · A duplicate library removed

`D5__MFIVB_skin` is dropped. The barcode gate proves it is the same library as
`D1__P303_skin` (containment 1.000, n=6,327) and **both are in the v1 atlas** — one patient's
biopsy counted twice under two donor IDs (`docs/ATLAS_V2_DEDUP_GATE.md` §2).
`D5__MFIVB_PBMC` and `D5__MFIVB_LN` are unique and are kept.

### 4b · Dominant-clone call is now per donor, not per lane

`build_clone_table` grouped by `sample_id`. On a hashtag-multiplexed lane that gives several
donors one denominator and one `top` clone, so **at most one patient per lane can ever be
called malignant** and the others are silently `False`. It now groups by donor, using a
`cell_id -> donor` map read from the cohort's standardized obs (lane-level grouping remains
the fallback when no map exists, i.e. for every unhashed cohort).

`clone_id` is namespaced by that same group, so two patients in one lane can no longer share
a clone through an identical CDR3.

**Measured effect: none on v1's numbers.** Regression against
`data/_archive_v1/atlas_joint/tcr_clones.parquet` gives **100.0000 % cell-level agreement on
`is_malignant` for all seven v1 TCR cohorts (B2, B4, B5, D1, D3, D5, H), delta 0**. B4 is the
only v1 cohort with hashed lanes, and its per-lane and per-donor calls coincide because each
lane is overwhelmingly dominated by one patient's clone (SZ38: 35,901 of 37,262 cells). The
mechanism still matters for v2 — D13's lanes hold genuinely different patients — but it is
not a v1 deviation, so the regression stays a clean identity.

Two subtleties this exposed, both fixed:
- Grouping on donor *alone* would **merge** a donor's separate biopsies (D1 keys `P318_thin`
  and `P318_thick` as one patient), changing the per-sample dominance the 0.981/0.834
  benchmark was measured on. The key is therefore `(namespaced sample_id, donor)`, and a
  sample is subdivided only when it actually holds more than one donor.
- The `cell_id -> donor` map resolves only partially for B4 (some contig barcodes have no
  expression cell). Unresolved cells must fall back to the **namespaced** sample id, not the
  bare one, or their `clone_id` silently loses its dataset prefix.

### 4c · Ex-vivo treatment arms excluded from GSE293752

Seven D15 lanes (`MF22 MF28 MF30 MF31 MF34 MF35 MF40`) are one biopsy split into a
vehicle-control and an IL4Ra-blocker arm, hashed together — stated only in the deposit's
free-text description (`"Treatments hashed together, HTO1-ctr,HTO2-regn"`). The control arm
is kept as the baseline and the drug arm dropped, consistent with the exclusion of B5's
cultured patients and all of D12. Without demux these lanes would have entered the atlas
with treated and untreated cells pooled under one donor.

The per-hashtag facts live in `data/<LABEL>/meta/hto_map.tsv` (donor, compartment, tissue,
disease, keep), generated from the deposit descriptions by `data/finalize_samples.py`. The
same mechanism fixes GSE182861's `SZ29` lane, which the deposit says holds **one blood and
one skin-tumour sample from the same patient** — lane-level metadata would have labelled the
skin cells as PBMC.

These are the only expected differences in held-cohort content between v1 and v2. The
regression check in §5 must allow for them and for nothing else.

## 5 · Version wiring — reproduce, do not "fix"

v1 is internally inconsistent about the malignant call, and v2 keeps the inconsistency so the
two atlases stay comparable. Unifying it is a separate change with its own before/after.

| notebook | reads | keep as-is in v2 |
|---|---|---|
| `23_malignancy_tcr_cnv` (writes) | `skin_T_malignancy_v4`, `skin_T_arm_cnv_v4` | yes |
| `24_subclone_tcr_signaling` | v4 | yes |
| `12_atlas_descriptive` | `skin_T_malignancy_**v3**` | yes — do not repoint |
| `old/37_semantic_clonality_transfer` | `skin_T_malignancy_**v3**` | yes — do not repoint |
| `15_spatial_resolvi_deconvolution` | `subclone_signatures_**v2**.json` | yes — do not repoint |

**One real bug must still be fixed**, because it blocks execution rather than changing results:
`20_skin/24_subclone_tcr_signaling.ipynb` c20 references `ARM_CNV_V3`, which c2 never binds (c2 binds
`ARM = skin_T_arm_cnv_v4.parquet`). It resolves only from a warm kernel and `NameError`s on a
fresh run. Bind it to the value v1's kernel held (`skin_T_arm_cnv_v3.parquet`).
