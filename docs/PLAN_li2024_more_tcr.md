# Plan — TCR for CTCL1/5/6/7/8 (Route A: processed WebAtlas object)

## Context / what recon established (read-only, login-safe)

Goal: add per-cell TCR for the fresh donors **CTCL1, CTCL5, CTCL6, CTCL7, CTCL8** to the
`11_li2024_tcr_malignancy.ipynb` comparison (currently CTCL2/3/4 only).

- **Route B is a dead end for these 5 donors.** The only single-cell accession is `E-MTAB-12303`
  and its raw V(D)J covers **only CTCL2/3/4**. `E-MTAB-13614` and `E-MTAB-14559` are both **Visium
  spatial** (confirmed via study JSON titles), not VDJ. The 45 `WSSS_SKN*` tarballs in 12303 are
  GEX-sized (median 49 MB vs VDJ 2–38 MB) — GEX, not VDJ. So raw VDJ FASTQ/contigs for CTCL1/5/6/7/8
  were not deposited in ArrayExpress.
- **Route A is the path.** The authors' TCR is a scirpy-style per-cell table
  `tcr_meta_CTCL.csv` (`IR_VJ_1/2_junction_aa` = TRA, `IR_VDJ_1/2_junction_aa` = TRB, `clone`,
  `TCR_pair`) merged into `CTCL_Tcell_processed_final.h5ad` (see authors'
  `code_final/CTCL_Tcell_TCR.ipynb`). This covers all 14 TCR patients incl. the 5 targets. It is
  served from the WebAtlas <https://collections.cellatlas.io/ctcl> (originals on Sanger lustre,
  not directly reachable).
- Local processed objects (`CTCL_all_final_portal_tags.h5ad`, `Integrated_CTCL_…h5ad`) carry **no**
  clonotype obs — cannot shortcut from disk.

## Guardrails

- **Login kernel: light only** — `curl`/BioStudies+ENA API, GitHub API, reading small CSVs, and
  `h5py` reads of an h5ad's `obs` group (cheap). **No** multi-GB h5ad loads, no `cellranger`, no
  full-object downloads on login. All downloads + any whole-object read → **bsub** (queue `long`,
  the only CPU queue this user can submit to; `transfer`/`gsla-cpu` reject us).
- **Additive, non-breaking.** Do not modify the working CTCL2/3/4 contig path. New data enters
  through a *second* loader that normalises into the **same** `per_cell` schema
  (`sample_id, barcode, tra_cdr3, trb_cdr3`) consumed by `H.build_clone_table`. Notebook 11's
  existing cells stay; we extend Step 2/3 to concatenate the two sources.

## Step 1 — locate the WebAtlas object URLs (login, light)

The site is a JS SPA; object URLs come from its runtime config, not static HTML.
1. `scripts/find_li2024_webatlas_objects.sh` (new): fetch the app's `manifest.json` / config and any
   `*.json` it references (cellgeni/`*.cog.sanger.ac.uk` S3-style host), extract URLs ending in
   `.h5ad`/`.zarr`/`.csv`. Cross-check the GitHub repo `ruoyan-li/Cutaneous-T-cell-lymphoma-study`
   `code_final/` notebooks for any hard-coded public paths.
2. Targets, smallest-first: `tcr_meta_CTCL.csv` (ideal — tiny, has the TCR columns) → the **T-cell**
   object `CTCL_Tcell_processed_final.h5ad` (or the `atlas_subset-tcells` object) → full atlas object.
3. Write `data/Li2024_atlas/tcr/webatlas_objects.txt` (URL, size, kind). **Gate:** if no public URL
   resolves, escalate (the object may be request-only) — do not guess FTP roots.

## Step 2 — fetch + verify (bsub for the download; obs check is light)

`scripts/download_li2024_webatlas_tcr.sh` (new, bsub queue `long`, idempotent, md5 if provided):
download the chosen object into `data/Li2024_atlas/tcr/webatlas/`.
Then a light verifier (login `h5py`/pandas):
- **A2a:** all of CTCL1/5/6/7/8 present in the object's donor column.
- **A2b:** non-trivial `has_ir`/non-null `clone`/`IR_VDJ_1_junction_aa` per target donor.
- Record the authoritative crosswalk `obs[[donor, sample_id, IR_* , clone]]` →
  `data/Li2024_atlas/tcr/crosswalk_CTCL_to_WSSS_to_VDJ.csv`.

## Step 3 — barcode/cell mapping to the atlas

The T-cell object is a subset of the **same** atlas, so try the **direct `obs_name` join** first
(its index should equal atlas `obs_names`); fall back to the `(donor, bc16)` overlap used for
CTCL2/3/4 if indices differ. Produce a `per_cell_processed` frame:
`sample_id = atlas donor`, `barcode = atlas obs_name`, `tra_cdr3 = IR_VJ_1_junction_aa`,
`trb_cdr3 = IR_VDJ_1_junction_aa` (drop `"None"`/NaN). TRB-primary matches `H.clone_id_from_cdr3`.

## Step 4 — fold into notebook 11 (additive)

In Step 2/3 of `11_li2024_tcr_malignancy.ipynb`:
- After building the contig-based `per_cell` (CTCL2/3/4), `pd.concat` the `per_cell_processed`
  (CTCL1/5/6/7/8) — guarded by `if (TCR_DIR/"webatlas"/…).exists()` so the notebook still runs with
  just CTCL2/3/4 if Route A hasn't been fetched.
- Feed the combined frame to the **same** `H.build_clone_table(...)` (dominant-clone rule, frac≥0.05
  & ratio≥2.0). `TCR_DONORS` becomes the 8 fresh donors; everything downstream (Step 4 comparison,
  Step 5 inferCNV `CNV_DONORS`, persist, UMAP) generalises unchanged.
- **Optional cross-check:** the authors already provide a `clone` column — compare our recomputed
  dominant clone to theirs per donor (should match the near-monoclonal malignant compartment).
- **Assertion A3:** each donor has one dominant expanded TRB clone (sanity vs paper Fig 2); flag any
  donor with no dominant clone.

inferCNV note: `CNV_DONORS` can extend to all 8 fresh donors — but CTCL1/5/6/7/8 are GEX-deposited
under `WSSS_SKN*`; they're already in the atlas h5ad (`X`/`raw_counts`), so inferCNV needs no extra
download. Keep `CNV_DONORS = TCR_DONORS` so CNV runs wherever we have a TCR call to compare against.

## Outputs (to `data/Li2024_atlas/tcr/`)

- `webatlas_objects.txt`, `crosswalk_CTCL_to_WSSS_to_VDJ.csv`, `vdj_accession_resolution.txt`
  (records that 13614/14559 = Visium, 12303 = VDJ only for CTCL2/3/4).
- `webatlas/<object>` (+ `md5_check.log` if checksums available).
- updated `li2024_tcr_malignancy.parquet` covering 8 fresh donors.
- `RUN_LOG.md` — each gate/assertion outcome.

## Stop / escalate

- A target donor absent from the processed object's TCR obs → report "TCR not deposited for this
  donor" (only 14/45 patients have TCR); do not fabricate.
- No public WebAtlas object URL resolvable from config/GitHub → escalate (possible request-only
  access); do not scrape blind FTP roots.
