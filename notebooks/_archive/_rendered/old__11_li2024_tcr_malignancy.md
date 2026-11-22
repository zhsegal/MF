# old/11_li2024_tcr_malignancy.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/11_li2024_tcr_malignancy.ipynb` and is never run._

# Li 2024 atlas — re-derive malignancy from TCR clonality vs. cached `tumor_cell`

Takes the **Li 2024 skin atlas** (`CTCL_all_final_portal_tags.h5ad`, the Li-2024 slice of the
joint atlas built in `10_build_joint_atlas.ipynb`) and re-calls malignant T cells **ourselves**
using the dominant-TCR-clone rule from `07_d1_semantic_geom_cd4.ipynb` (generalized in
`atlas_join_helpers.build_clone_table`): per patient, the top clonotype is malignant iff it is
≥ 5 % of V(D)J cells **and** ≥ 2× the 2nd clone. We then compare our calls to the atlas's
published `cell_type == "tumor_cell"` annotation.

**Prereq:** run `scripts/download_li2024_tcr.sh` (via bsub) first. It pulls the CellRanger `vdj`
runs from ArrayExpress **E-MTAB-12303** into `data/Li2024_atlas/tcr/` and writes `manifest.csv`
(`file, accession_sample, vdj_index`).

**TCR coverage.** *Dedicated 10x V(D)J* — the 8 fresh `Sanger_Ncl_Fresh` donors (CTCL1–CTCL8); their
V(D)J is in E-MTAB-12303 in two namings: `CTCL{N}_VDJ_{k}` (CTCL2/3/4) and the *small* `WSSS_SKN*`
vdj runs (the others); the *large* `WSSS_SKN*` are 5′ GEX (skipped). *TRUST4 5′-mined (partial,
TRB-primary)* — the MDA `PT*` cohort (Song 2022, PRJNA754592) has no dedicated V(D)J deposited, but
CDR3s are read-mined from its 5′ GEX BAMs by `scripts/recover_pt_tcr_trust4.sh` (loaded additively in
Step 2, **provenance-tagged** `tcr_source`). Still no TCR: the FFPE-Flex donors (CTCL9–18; Flex
chemistry has no V(D)J) and the PKU `MF*` cohort (V(D)J in GSA HRA000166, controlled-access). Each
library's atlas donor + lane is resolved **empirically by 16-bp barcode overlap** — no manual mapping.
Run on the GPU kernel (`neural_nmf_env`); steps 1–4 light, inferCNV (5) and the UMAP (8) are heavy.

```python  # cell 1
import sys, re, gc
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc


def _resolve_nb_dir() -> Path:
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("notebooks/MF"), Path("scvi-tools-neural-nmf/notebooks/MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(start)


NB_DIR = _resolve_nb_dir()
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))
import importlib
import atlas_join_helpers as H
importlib.reload(H)

SEED = 0
np.random.seed(SEED)
sc.settings.verbosity = 1

ATLAS_H5 = NB_DIR / "data" / "CTCL_all_final_portal_tags.h5ad"
TCR_DIR  = NB_DIR / "data" / "Li2024_atlas" / "tcr"
FIG_DIR  = NB_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)
OUT_PARQUET = NB_DIR / "data" / "Li2024_atlas" / "li2024_tcr_malignancy.parquet"
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

# dominance thresholds — canonical 07_d1 / build_tcr_table values
FRAC_THRESH, RATIO_THRESH = 0.05, 2.0

adata = sc.read_h5ad(ATLAS_H5)
adata.obs["cached_malignant"] = (adata.obs["cell_type"] == "tumor_cell")
print(adata)
print("cached tumor_cell:", int(adata.obs["cached_malignant"].sum()), "/", adata.n_obs)
```

```python  # cell 2
# T-cell UMAP on the paper's pre-computed scVI embedding (sanity view before re-calling malignancy).
# HEAVY (neighbors + UMAP) — GPU kernel; cached to .npy and reloaded on re-run.
# T_CELLTYPES = ["tumor_cell", "Tc", "Th", "Treg", "Tc17_Th17", "Tc_IL13_IL22"]
# assert "X_scVI" in adata.obsm, "expected the paper's scVI embedding in adata.obsm['X_scVI']"

# at = adata[adata.obs["cell_type"].astype(str).isin(T_CELLTYPES)].copy()
# TUMAP_NPY = NB_DIR / "data" / "Li2024_atlas" / "li2024_tcells_umap.npy"
# if TUMAP_NPY.exists():
#     at.obsm["X_umap"] = np.load(TUMAP_NPY)
#     print("loaded cached T-cell UMAP", at.obsm["X_umap"].shape)
# else:
#     sc.pp.neighbors(at, use_rep="X_scVI", random_state=SEED)
#     sc.tl.umap(at, random_state=SEED)
#     np.save(TUMAP_NPY, at.obsm["X_umap"])
#     print("computed + cached T-cell UMAP ->", TUMAP_NPY)

# sc.pl.umap(at, color="cell_type", frameon=False, size=4,
#            title=f"Li2024 T cells (n={at.n_obs}) — paper scVI embedding",
#            save="_li2024_tcells_by_celltype.png")
```

## Step 1 — barcode key

Atlas `obs_names` are `<16bp>-<lane>_<donor>×3`, e.g. `AAACCTGAGAAGCCCA-0_CTCL1_CTCL1_CTCL1`.
We derive `bc16` (the raw cell barcode, matches the contig barcode after its `-1` suffix is
stripped) and `lane` (the atlas integration index = the library within a donor). Each fresh donor
spans 4–6 lanes and `bc16` collides across them, so each V(D)J library is matched to a single
`(donor, lane)` in step 2.

```python  # cell 4
names = adata.obs_names.to_series()
adata.obs["bc16"] = names.str.extract(r"^([ACGT]{16})", expand=False).values
adata.obs["lane"] = pd.to_numeric(names.str.extract(r"-(\d+)_", expand=False), errors="coerce").astype("Int64").values
assert adata.obs["bc16"].notna().all(), "some obs_names lack a 16bp barcode prefix"

lane_ranges = (adata.obs.groupby("donor", observed=True)["lane"]
               .agg(n_lanes=lambda s: s.nunique(),
                    lanes=lambda s: sorted(set(int(x) for x in s.dropna()))))
print(lane_ranges.to_string())
```

## Step 2 — parse contigs + match barcodes to atlas cells

Reads each `filtered_contig_annotations.csv` listed in the manifest via `H.airr_from_10x_contig`
(keeps productive chains, collapses to per-cell TRA/TRB CDR3, strips the `-N` barcode suffix).
Each V(D)J library is mapped to its atlas `(donor, lane)` **by 16-bp barcode overlap** — the true
library lights up one `(donor, lane)` strongly while cross-library 16-bp collisions are random
noise (`purity` = best / (best+2nd)). The matched atlas `obs_name` is the unique per-cell key.

```python  # cell 6
MANIFEST = TCR_DIR / "manifest.csv"
assert MANIFEST.exists(), f"run scripts/download_li2024_tcr.sh first ({MANIFEST} missing)"
man = pd.read_csv(MANIFEST, dtype=str)
print(f"manifest: {len(man)} contig files | samples={sorted(man['accession_sample'].unique())}")

atlas_key = (adata.obs.reset_index()
             .rename(columns={"index": "obs_name"})[["obs_name", "donor", "lane", "bc16"]])

per_cell_rows, map_rows = [], []
for _, row in man.iterrows():
    f = Path(row["file"])
    if not f.is_absolute():
        f = TCR_DIR / f
    if not f.exists():
        print(f"  MISSING {f}");  continue
    air = H.airr_from_10x_contig(f).rename(columns={"barcode": "bc16"})
    # resolve (donor, lane) by 16bp barcode overlap with the atlas
    hit = atlas_key.merge(air[["bc16"]].drop_duplicates(), on="bc16", how="inner")
    if not len(hit):
        print(f"  {f.name}: no barcode overlap with atlas — skipped");  continue
    counts = hit.groupby(["donor", "lane"], observed=True).size().sort_values(ascending=False)
    (best_donor, best_lane), best_n = counts.index[0], int(counts.iloc[0])
    second_n = int(counts.iloc[1]) if len(counts) > 1 else 0
    map_rows.append({"file": f.name, "sample": row["accession_sample"], "vdj_index": row["vdj_index"],
                     "donor": best_donor, "lane": int(best_lane),
                     "n_contig": int(air["bc16"].nunique()), "n_matched": best_n,
                     "purity": round(best_n / max(1, best_n + second_n), 3)})
    lib = atlas_key[(atlas_key["donor"] == best_donor) & (atlas_key["lane"] == int(best_lane))]
    merged = lib.merge(air, on="bc16", how="inner")
    for _, m in merged.iterrows():
        per_cell_rows.append({"sample_id": best_donor, "barcode": m["obs_name"],
                              "tra_cdr3": m["tra_cdr3"], "trb_cdr3": m["trb_cdr3"]})

per_cell = pd.DataFrame(per_cell_rows)
map_report = pd.DataFrame(map_rows)
print("\nresolved V(D)J library -> atlas (donor, lane) by barcode overlap:")
print(map_report.to_string(index=False))
assert len(map_report) and (map_report["purity"] > 0.8).all(), \
    "a V(D)J library did not map cleanly to one (donor, lane) — inspect map_report"
print("\nper_cell TCR rows:", len(per_cell), "| donors:", sorted(per_cell["sample_id"].unique()))
assert len(per_cell), "no TCR cells matched to atlas"
assert not per_cell["barcode"].duplicated().any(), "a cell got two contig assignments"
```

```python  # cell 7
# --- PT (MDA / Song 2022) TCR via TRUST4 — additive, provenance-tagged ---
# PRJNA754592 deposited per channel both a 5' GEX BAM and a dedicated CellRanger-vdj all_contig.bam.
# The V(D)J BAM is contig-aligned from an old pipeline (bamtofastq/TRUST4 can't use it without
# CellRanger), so scripts/recover_pt_tcr_trust4.sh mines the 5' GEX BAM: it streams only the
# TCR-locus reads remotely (~MB, not the 40-150 GB full BAM), reheaders to chr-naming, and runs
# TRUST4 (BAM mode, CB/UB). Partial / TRB-primary — enough for the dominant (malignant) clone.
# Each channel (one SRS) maps to its atlas (donor, lane): donor known from the accession map, lane
# by barcode overlap. NB: some PT donors' two channels share many 16bp barcodes (well above random),
# so the lane argmax is sometimes ambiguous (low "purity"). That only affects WHICH atlas cell of the
# (correct) donor receives the TCR — clone calling is per-donor, so the malignant call is unaffected.
PT_DIR = TCR_DIR.parent / "tcr_pt"          # data/Li2024_atlas/tcr_pt (sibling of tcr/)
PT_MANIFEST = PT_DIR / "manifest_trust4.csv"
SRC_LABEL = {"vdj": "TRUST4_vdj_bam", "gex": "TRUST4_5p_GEX_mined"}
per_cell["tcr_source"] = "10x_vdj"
if PT_MANIFEST.exists():
    ptman = pd.read_csv(PT_MANIFEST, dtype=str)
    pt_rows, pt_map = [], []
    for _, row in ptman.iterrows():
        f = PT_DIR / row["airr_file"]
        if not f.exists():
            print(f"  MISSING {f}");  continue
        src = SRC_LABEL.get(row.get("source", "gex"), "TRUST4_5p_GEX_mined")
        air = H.airr_from_trust4(f).rename(columns={"barcode": "bc16"})
        donor = row["donor"]
        cand = atlas_key[atlas_key["donor"] == donor]
        hit = cand.merge(air[["bc16"]].drop_duplicates(), on="bc16", how="inner")
        if not len(hit):
            print(f"  {row['airr_file']}: no barcode overlap with {donor} — skipped");  continue
        counts = hit.groupby("lane", observed=True).size().sort_values(ascending=False)
        best_lane, best_n = int(counts.index[0]), int(counts.iloc[0])
        second_n = int(counts.iloc[1]) if len(counts) > 1 else 0
        pt_map.append({"airr": row["airr_file"], "srs": row["srs"], "donor": donor, "source": src,
                       "lane": best_lane, "n_contig": int(air["bc16"].nunique()),
                       "n_matched": best_n, "purity": round(best_n / max(1, best_n + second_n), 3)})
        lib = atlas_key[(atlas_key["donor"] == donor) & (atlas_key["lane"] == best_lane)]
        merged = lib.merge(air, on="bc16", how="inner")
        for _, m in merged.iterrows():
            pt_rows.append({"sample_id": donor, "barcode": m["obs_name"],
                            "tra_cdr3": m["tra_cdr3"], "trb_cdr3": m["trb_cdr3"], "tcr_source": src})
    if pt_rows:
        pt_map = pd.DataFrame(pt_map)
        print("PT TRUST4 channel -> atlas (donor, lane) by barcode overlap:")
        print(pt_map.to_string(index=False))
        lowp = pt_map[pt_map["purity"] <= 0.8]
        if len(lowp):
            print(f"\n  NOTE: {len(lowp)} channel(s) with ambiguous lane (purity<=0.8) — shared "
                  f"barcodes across the donor's channels; per-donor clone call is unaffected:")
            print("  " + ", ".join(f"{r.srs}({r.donor},p={r.purity})" for r in lowp.itertuples()))
        assert (pt_map["purity"] > 0.4).all(), "a PT channel mapped to no clear lane (purity<=0.4)"
        per_cell_pt = pd.DataFrame(pt_rows)
        per_cell = pd.concat([per_cell, per_cell_pt], ignore_index=True)
        # disjoint donor sets -> a barcode collision would be coincidental; keep the dedicated-VDJ row
        per_cell = per_cell.drop_duplicates("barcode", keep="first").reset_index(drop=True)
        print(f"\n+ PT: {len(per_cell_pt)} TCR cells over {sorted(per_cell_pt['sample_id'].unique())}"
              f"  sources={per_cell_pt['tcr_source'].value_counts().to_dict()}")
        print("combined per_cell rows:", len(per_cell), "| donors:", sorted(per_cell["sample_id"].unique()))
    else:
        print(f"(PT manifest present but no rows mapped — check {PT_DIR})")
else:
    print(f"(no PT TRUST4 manifest at {PT_MANIFEST} — running CTCL donors only)")
```

## Step 3 — dominant-clone malignancy (07_d1 rule) → atlas obs

`H.build_clone_table` builds a TRB-primary exact clone key per cell, sizes clones per patient
(`sample_id = donor`, aggregating that donor's libraries), and flags the dominant clone malignant
when it clears `FRAC_THRESH` / `RATIO_THRESH`. We map `is_malignant` back onto the atlas by the
`obs_name` embedded in its `cell_id`.

```python  # cell 9
clone = H.build_clone_table(
    per_cell, dataset="Li2024_atlas",
    sample_to_donor={d: d for d in per_cell["sample_id"].unique()},
    frac_thresh=FRAC_THRESH, ratio_thresh=RATIO_THRESH,
)
clone["obs_name"] = clone["cell_id"].str.split("|", n=1).str[1]   # "Li2024_atlas__{donor}|{obs_name}"
clone = clone.set_index("obs_name")

idx = adata.obs_names
adata.obs["has_tcr"]               = idx.isin(clone.index)
adata.obs["tcr_clone_id"]          = clone["clone_id"].reindex(idx).fillna("").values
adata.obs["tcr_clone_size"]        = clone["clone_size"].reindex(idx).fillna(0).astype(int).values
adata.obs["tcr_is_malignant"]      = clone["is_malignant"].reindex(idx).fillna(False).astype(bool).values
adata.obs["tcr_is_dominant_clone"] = clone["is_dominant_clone"].reindex(idx).fillna(False).astype(bool).values
# provenance per cell (10x_vdj for CTCL; TRUST4_vdj_bam / TRUST4_5p_GEX_mined for PT), from per_cell
src = per_cell.drop_duplicates("barcode").set_index("barcode")["tcr_source"]
adata.obs["tcr_source"] = src.reindex(idx).fillna("").values
adata.obs.loc[~adata.obs["has_tcr"], "tcr_source"] = ""   # only meaningful where a clone was called

TCR_DONORS = sorted(per_cell["sample_id"].unique())
print("TCR-covered donors:", TCR_DONORS)
print("tcr_source:", adata.obs.loc[adata.obs["has_tcr"], "tcr_source"].value_counts().to_dict())
dom = (adata.obs[adata.obs["has_tcr"]].groupby("donor", observed=True)
       .agg(n_tcr=("has_tcr", "size"),
            n_malignant=("tcr_is_malignant", "sum"),
            dom_clone_frac=("tcr_is_dominant_clone", "mean")))
print(dom.to_string())
```

```python  # cell 10
# descriptive table: per-donor largest clone, fold-change to 2nd, threshold, malignant call
_rows = []
_pc = per_cell.copy()
_pc["clone_id"] = [H.clone_id_from_cdr3(a, b) for a, b in zip(_pc["tra_cdr3"], _pc["trb_cdr3"])]
_pc = _pc[_pc["clone_id"] != ""]
for sid, sub in _pc.groupby("sample_id"):
    sizes = sub["clone_id"].value_counts()
    n = len(sub)
    top_n = int(sizes.iloc[0])
    second_n = int(sizes.iloc[1]) if len(sizes) > 1 else 0
    dom_frac = top_n / max(1, n)
    ratio = top_n / second_n if second_n else np.inf
    _rows.append({
        "donor": sid,
        "n_tcr_cells": n,
        "n_clones": int(len(sizes)),
        "largest_clone": top_n,
        "second_clone": second_n,
        "fold_change": ratio,
        "dom_frac": dom_frac,
        "pass_frac (>=%.2f)" % FRAC_THRESH: dom_frac >= FRAC_THRESH,
        "pass_ratio (>=%.1f)" % RATIO_THRESH: ratio >= RATIO_THRESH,
        "malignant": (dom_frac >= FRAC_THRESH) and (ratio >= RATIO_THRESH),
    })
clone_summary = (pd.DataFrame(_rows).sort_values("largest_clone", ascending=False)
                 .reset_index(drop=True))
print(f"thresholds: dom_frac >= {FRAC_THRESH}  AND  fold_change >= {RATIO_THRESH}")
print(f"malignant donors: {int(clone_summary['malignant'].sum())}/{len(clone_summary)}")
clone_summary.style.format({"fold_change": "{:.2f}", "dom_frac": "{:.3f}"})
```

## Step 4 — compare to cached `tumor_cell`

Restricted to TCR-covered donors. Confusion matrix + precision/recall/Jaccard of our TCR calls
against the published label; per-donor agreement; and the `cell_type` distribution of both the
cells we call malignant and the cached `tumor_cell` cells we **miss** (TRA/B dropout, subclones,
or non-T tumor_cell).

```python  # cell 12
cov = adata.obs[adata.obs["donor"].isin(TCR_DONORS)].copy()
y_cached = cov["cached_malignant"].to_numpy()
y_tcr    = cov["tcr_is_malignant"].to_numpy()

cm = pd.crosstab(pd.Series(y_tcr, name="tcr_malignant"),
                 pd.Series(y_cached, name="cached_tumor_cell"))
print("confusion (all cells in TCR-covered donors):\n", cm, "\n")
tp = int((y_tcr & y_cached).sum()); fp = int((y_tcr & ~y_cached).sum()); fn = int((~y_tcr & y_cached).sum())
prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn); jac = tp / max(1, tp + fp + fn)
print(f"TCR-vs-cached   precision={prec:.3f}  recall={rec:.3f}  jaccard={jac:.3f}")
print("(precision = fraction of TCR-malignant that are cached tumor_cell;"
      " recall = fraction of tumor_cell recovered by TCR)")
```

```python  # cell 13
# --- stratify TCR-vs-cached by provenance (dedicated 10x VDJ / TRUST4 VDJ-BAM / TRUST4 GEX-mined) ---
# Keeps the headline numbers honest: GEX-mined calls are partial/TRB-only and must not be pooled with
# dedicated-VDJ donors. TRUST4_vdj_bam (PT) is near-dedicated quality but still a re-assembly.
for src in adata.obs.loc[adata.obs["has_tcr"], "tcr_source"].value_counts().index:
    donors = sorted(adata.obs.loc[adata.obs["tcr_source"] == src, "donor"].unique())
    sub = adata.obs[adata.obs["donor"].isin(donors)]
    yt = sub["tcr_is_malignant"].to_numpy(); yc = sub["cached_malignant"].to_numpy()
    tp = int((yt & yc).sum()); fp = int((yt & ~yc).sum()); fn = int((~yt & yc).sum())
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn); j = tp / max(1, tp + fp + fn)
    print(f"[{src:20s}] donors={donors}")
    print(f"    n_tcr_cells={int(sub['has_tcr'].sum()):>7}  precision={p:.3f}  recall={r:.3f}  jaccard={j:.3f}")
```

```python  # cell 14
def _agg(g):
    c = g["cached_malignant"].to_numpy(); t = g["tcr_is_malignant"].to_numpy()
    return pd.Series({
        "n_cells": len(g), "n_tcr": int(g["has_tcr"].sum()),
        "n_cached_tumor": int(c.sum()), "n_tcr_malignant": int(t.sum()),
        "overlap": int((c & t).sum()),
        "jaccard": round(int((c & t).sum()) / max(1, int((c | t).sum())), 3),
    })

donor_tab = cov.groupby("donor", observed=True).apply(_agg)
print(donor_tab.to_string())
```

```python  # cell 15
print("cell_type of TCR-malignant cells:")
print(cov.loc[cov["tcr_is_malignant"], "cell_type"].value_counts().head(15).to_string())

missed = cov[cov["cached_malignant"] & ~cov["tcr_is_malignant"]]
print(f"\ncached tumor_cell NOT called malignant by TCR: n={len(missed)} "
      f"(of these, {int(missed['has_tcr'].sum())} have a recovered TCR)")
print(missed["cell_type"].value_counts().head().to_string())

extra = cov[cov["tcr_is_malignant"] & ~cov["cached_malignant"]]
print(f"\nTCR-malignant but NOT cached tumor_cell: n={len(extra)}")
print(extra["cell_type"].value_counts().head().to_string())
```

## Step 5 — inferCNV on the lymphoid compartment, with a **same-lineage T-cell reference**

The previous run used a stromal/epithelial reference against an all-cell query. That is the wrong
reference *lineage* for a T-cell tumour: inferCNV subtracts the reference per-gene mean, so every T
cell (benign **and** malignant) reads as "aneuploid" purely from T-vs-stroma expression — that
lineage signal swamps CTCL's subtle, focal CNVs and the malignant/benign T cells end up
indistinguishable (mean `cnv_score` 0.009 vs 0.012). Deep literature review (NotebookLM notebook
*inferCNV methodology for CTCL / T-cell lymphoma*) and the replication spec (`CTCL_atlas_notebook_replication_spec.md`
Cell 3) both prescribe a **lineage-matched normal-T-cell reference**.

We therefore:

1. **Restrict the query to the lymphoid compartment** (`tumor_cell` + benign T/NK/ILC subsets) per
   donor — inferCNV is then T-on-T.
2. Use a **combined diploid reference of same-lineage normal T cells**:
   - **within-donor non-clonal T cells** (`has_tcr & ~tcr_is_dominant_clone`, excluding `tumor_cell`)
     — a per-patient reference independent of the published `tumor_cell` label; and
   - **healthy-control T cells** — benign T subsets from `healthy_skin / AD / psoriasis` donors in
     `Integrated_CTCL_skincellatlas_final_portal_tags.h5ad` (the CTCL + skin-cell-atlas integration),
     subsampled and shared across donors.
   Passing both as `reference_cat` triggers infercnvpy's **multi-category bounded subtraction**,
   which zeroes log-fold-changes inside the per-category mean envelope (suppresses HLA/Ig artefacts).
3. **`window_size=250`** (was 100) to smooth 10x dropout; infercnvpy `step=10` default.
4. Genomic positions from the cached Ensembl GRCh38.110 GTF, on the gene-symbol intersection of the
   two atlas files (the integrated file is keyed by Ensembl ID with a `genes` symbol column).

The malignant call (next cell) thresholds each donor's `cnv_score` at the **upper Gaussian mode**
above the T-cell reference floor.

**HEAVY (CPU/RAM-bound)** — run on the GPU kernel. Reads the integrated atlas (backed) for the
healthy reference, then runs inferCNV per donor over the lymphoid cells.

```python  # cell 17
import infercnvpy as cnv
import anndata as ad

GTF           = NB_DIR / "data" / "cache" / "Homo_sapiens.GRCh38.110.chr.gtf.gz"
INTEGRATED_H5 = NB_DIR / "data" / "Integrated_CTCL_skincellatlas_final_portal_tags.h5ad"
CNV_DONORS    = TCR_DONORS                         # all fresh donors with a recovered TCR call

# lymphoid compartment (query) and the benign-T subsets used for the reference / floor
LYMPHOID = ["tumor_cell", "Tc", "Th", "Treg", "Tc17_Th17", "Tc_IL13_IL22",
            "NK", "ILC1_3", "ILC1_NK", "ILC2"]
BENIGN_T = ["Tc", "Th", "Treg", "Tc17_Th17", "Tc_IL13_IL22"]
N_HEALTHY_REF = 3000                              # subsample of healthy/AD/pso benign-T reference cells

# ---- query: lymphoid cells of the CNV donors, log-normalised ----
qmask = adata.obs["donor"].isin(CNV_DONORS) & adata.obs["cell_type"].astype(str).isin(LYMPHOID)
acnv = adata[qmask].copy()
acnv.X = acnv.layers["raw_counts"].copy()
sc.pp.normalize_total(acnv, target_sum=1e4)
sc.pp.log1p(acnv)

# within-donor non-clonal T cells = lineage-matched diploid reference (independent of tumor_cell label)
acnv.obs["cnv_ref"] = "query"
nonclonal = (acnv.obs["has_tcr"] & ~acnv.obs["tcr_is_dominant_clone"]
             & (acnv.obs["cell_type"].astype(str) != "tumor_cell"))
acnv.obs.loc[nonclonal, "cnv_ref"] = "nonclonal"

# ---- healthy-control reference: benign T cells from healthy/AD/psoriasis skin (integrated atlas) ----
# integrated file is keyed by Ensembl gene_ids but carries gene symbols in var['genes']; read backed,
# pull only the subsampled reference rows, then rebuild a clean in-memory AnnData (avoids backed-view copy).
rf = sc.read_h5ad(INTEGRATED_H5, backed="r")
rmask = (rf.obs["sample_type"].astype(str).isin(["healthy_skin", "AD", "psoriasis"])
         & rf.obs["cell_type"].astype(str).isin(BENIGN_T)).to_numpy()
sel = np.where(rmask)[0]
rng = np.random.default_rng(SEED)
if sel.size > N_HEALTHY_REF:
    sel = np.sort(rng.choice(sel, N_HEALTHY_REF, replace=False))
rsub = rf[sel].to_memory()
ref = ad.AnnData(
    X=rsub.layers["raw_counts"].copy(),
    obs=pd.DataFrame({"cnv_ref": "healthy", "donor": "HEALTHY_REF",
                      "cell_type": rsub.obs["cell_type"].astype(str).values},
                     index=rsub.obs_names.astype(str)),
    var=pd.DataFrame(index=pd.Index(rsub.var["genes"].astype(str).values)))
rf.file.close()
ref.var_names_make_unique()
sc.pp.normalize_total(ref, target_sum=1e4)
sc.pp.log1p(ref)

# ---- common gene space (symbols) + genomic positions ----
common = acnv.var_names.intersection(ref.var_names)
acnv = acnv[:, common].copy()
ref  = ref[:, common].copy()
cnv.io.genomic_position_from_gtf(GTF, adata=acnv, gtf_gene_id="gene_name")
ref.var = acnv.var.copy()                          # same gene order -> share genomic positions
n_annot = int(acnv.var[["chromosome", "start", "end"]].notna().all(axis=1).sum())
print(f"common genes: {len(common)} | with genomic position: {n_annot} / {acnv.n_vars}")
assert n_annot > 8000, "too few genes annotated — check GTF gene_name / symbol intersection"

HEALTHY_REF = ref                                  # shared reference, concatenated per donor in next cell
print("CNV donors:", CNV_DONORS)
print(f"query lymphoid cells: {acnv.n_obs}  |  within-donor non-clonal ref: "
      f"{int((acnv.obs['cnv_ref'] == 'nonclonal').sum())}  |  healthy ref: {HEALTHY_REF.n_obs}")
```

```
INFO:root:Extracted GTF attributes: ['gene_id', 'gene_version', 'gene_name', 'gene_source', 'gene_biotype', 'transcript_id', 'transcript_version', 'transcript_name', 'transcript_source', 'transcript_biotype', 'tag', 'transcript_support_level', 'exon_number', 'exon_id', 'exon_version', 'protein_id', 'protein_version', 'ccds_id']

WARNING: GTF file misses annotation for 301 genes in adata.
WARNING: Skipped 28 genes because of duplicate identifiers in GTF file.
common genes: 15777 | with genomic position: 15449 / 15777
CNV donors: ['CTCL1', 'CTCL2', 'CTCL3', 'CTCL4', 'CTCL5', 'CTCL6', 'CTCL7', 'CTCL8', 'PT11', 'PT35', 'PT47', 'PT50', 'PT52', 'PT53', 'PT55', 'PT56']
query lymphoid cells: 165467  |  within-donor non-clonal ref: 69971  |  healthy ref: 3000
```

```python  # cell 18
# Per-donor inferCNV on the lymphoid query + combined same-lineage reference (non-clonal T + healthy).
# Caches the per-cluster cnv_score (heatmaps/diagnostics), the genome-wide cnv_cell_score (mean|X_cnv|,
# kept for comparison) AND a FOCAL per-CELL score cnv_focal_score = mean of each cell's top-K% strongest
# |X_cnv| windows. CTCL CNVs are focal (7q/8q/17q gains, 9p/10q/17p/13q losses); the genome-wide mean
# dilutes them under benign background (median tumor/ref sep ~1.5, AUROC ~0.78) -> recall collapsed.
# The top-K% summary concentrates on the focal events the heatmap shows. The call uses cnv_focal_score.
# FORCE_CNV=True recomputes (cache bumped to v4 for the new column).
WINDOW_SIZE = 250                                  # smooth 10x dropout (was 100)
FORCE_CNV   = False
TOPK_FRAC   = 0.10                                 # focal fraction of windows per cell; tune 0.05-0.15
# infercnvpy's cnv.tl.infercnv defaults to n_jobs=os.cpu_count() (=256 here) and forks one
# ProcessPoolExecutor worker per chunk -> the fork fan-out + per-chunk densification overran the
# kernel's memory cap and killed it on the larger CTCL3 donor. Cap workers + shrink the dense chunk.
CNV_N_JOBS  = 8
CNV_CHUNK   = 2500
CNV_CACHE   = NB_DIR / "data" / "Li2024_atlas" / "li2024_cnv_per_cell_v4_lymphoid.parquet"

acnv.obs["cnv_score"]       = np.nan               # per cnv_leiden CLUSTER (kept for heatmaps/diagnostics)
acnv.obs["cnv_cell_score"]  = np.nan               # per CELL: genome-wide mean |X_cnv| (comparison)
acnv.obs["cnv_focal_score"] = np.nan               # per CELL: mean of top-K% |X_cnv| windows (drives the call)
acnv.obs["cnv_leiden"]      = ""

# ---- heavy step: inferCNV per donor (cached) ----
use_cache = (not FORCE_CNV) and CNV_CACHE.exists()
if use_cache:
    cc = pd.read_parquet(CNV_CACHE).set_index("obs_name")
    use_cache = ({"cnv_cell_score", "cnv_focal_score"}.issubset(cc.columns)
                 and set(CNV_DONORS).issubset(set(cc["donor"].astype(str).unique())))
if use_cache:
    acnv.obs["cnv_score"]       = cc["cnv_score"].reindex(acnv.obs_names).to_numpy()
    acnv.obs["cnv_cell_score"]  = cc["cnv_cell_score"].reindex(acnv.obs_names).to_numpy()
    acnv.obs["cnv_focal_score"] = cc["cnv_focal_score"].reindex(acnv.obs_names).to_numpy()
    acnv.obs["cnv_leiden"]      = cc["cnv_leiden"].reindex(acnv.obs_names).fillna("").to_numpy()
    print(f"loaded cached inferCNV ({CNV_CACHE.name}) for {sorted(CNV_DONORS)}")
else:
    for d in CNV_DONORS:
        q = acnv[acnv.obs["donor"] == d]
        # combined reference = within-donor non-clonal T cells + shared healthy benign-T cells
        sub = ad.concat([q, HEALTHY_REF], join="inner", index_unique=None)
        sub.var = acnv.var.loc[sub.var_names].copy()            # restore genomic positions after concat
        ref_cats = [c for c in ["nonclonal", "healthy"] if int((sub.obs["cnv_ref"] == c).sum()) >= 20]
        n_nc = int((q.obs["cnv_ref"] == "nonclonal").sum())
        print(f"[{d}] query={q.n_obs:>6}  nonclonal={n_nc:>5}  healthy={HEALTHY_REF.n_obs}  ref={ref_cats}")
        cnv.tl.infercnv(sub, reference_key="cnv_ref", reference_cat=ref_cats,
                        window_size=WINDOW_SIZE, n_jobs=CNV_N_JOBS, chunksize=CNV_CHUNK)
        cnv.tl.pca(sub)
        cnv.pp.neighbors(sub)
        cnv.tl.leiden(sub)
        cnv.tl.cnv_score(sub)                                   # per-cluster score -> obs['cnv_score']
        # per-cell magnitudes while X_cnv is in memory: genome-wide mean + focal top-K% mean
        Xc = sub.obsm["X_cnv"]
        Xc = np.abs(Xc.toarray() if hasattr(Xc, "toarray") else np.asarray(Xc))
        k  = max(1, int(round(TOPK_FRAC * Xc.shape[1])))
        sub.obs["cnv_cell_score"]  = Xc.mean(axis=1)
        sub.obs["cnv_focal_score"] = np.partition(Xc, Xc.shape[1] - k, axis=1)[:, -k:].mean(axis=1)
        del Xc
        keep = (sub.obs["donor"] == d).to_numpy()               # drop the shared healthy ref before stitching
        acnv.obs.loc[sub.obs_names[keep], "cnv_score"]       = sub.obs.loc[keep, "cnv_score"].to_numpy()
        acnv.obs.loc[sub.obs_names[keep], "cnv_cell_score"]  = sub.obs.loc[keep, "cnv_cell_score"].to_numpy()
        acnv.obs.loc[sub.obs_names[keep], "cnv_focal_score"] = sub.obs.loc[keep, "cnv_focal_score"].to_numpy()
        acnv.obs.loc[sub.obs_names[keep], "cnv_leiden"]      = (d + "_" + sub.obs.loc[keep, "cnv_leiden"].astype(str)).to_numpy()
        del sub, q                                              # release the per-donor dense CNV matrix before the next donor
        gc.collect()
    cache = acnv.obs[["donor", "cnv_score", "cnv_cell_score", "cnv_focal_score", "cnv_leiden"]].copy()
    cache.index.name = "obs_name"
    cache.reset_index().to_parquet(CNV_CACHE)
    print(f"computed inferCNV; cached -> {CNV_CACHE}")

# ---- cheap step: per-donor malignant call on the FOCAL per-cell score (recomputed each run) ----
# cnv_focal_score is genuinely continuous (the old "no GMM" note was about the spiky per-CLUSTER score),
# so fit a 2-component GMM per donor on log(focal) and cut at the benign/tumour posterior=0.5 crossover,
# floored at the diploid reference median (never cut into pure noise). Fallback to ref-p90 if the GMM is
# degenerate. The orthogonal TCR-clonal label gives an F1-optimal cut per donor for validation only.
from sklearn.mixture import GaussianMixture

SCORE_COL = "cnv_focal_score"
acnv.obs["cnv_malignant"]   = False
acnv.obs["cnv_call_method"] = ""
diploid = acnv.obs["cnv_ref"].eq("nonclonal") | acnv.obs["cell_type"].astype(str).isin(BENIGN_T)


def _best_f1_thr(score, y):                                    # F1-optimal cut over a 99-quantile grid
    ok = np.isfinite(score)
    if ok.sum() < 50 or not (0 < y[ok].sum() < ok.sum()):
        return np.nan, np.nan, int(ok.sum())
    s, yy = score[ok], y[ok].astype(bool)
    best_f1, best_t = 0.0, np.nan
    for t in np.unique(np.percentile(s, np.linspace(1, 99, 99))):
        yp = s >= t
        tp = int((yp & yy).sum()); fp = int((yp & ~yy).sum()); fn = int((~yp & yy).sum())
        pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn); f1 = 2 * pr * rc / max(1e-9, pr + rc)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1, int(ok.sum())


for d in CNV_DONORS:
    m = (acnv.obs["donor"] == d).to_numpy()
    s = acnv.obs.loc[m, SCORE_COL].to_numpy(dtype=float)
    finite = np.isfinite(s) & (s > 0)
    # diploid reference floor: trim the upper tail (mislabeled-malignant benign-T) before the median
    ref_s = acnv.obs.loc[m & diploid.to_numpy(), SCORE_COL].to_numpy(dtype=float)
    ref_s = ref_s[np.isfinite(ref_s) & (ref_s > 0)]
    floor = np.nan
    if ref_s.size >= 20:
        ref_s = ref_s[ref_s <= np.percentile(ref_s, 90)]
        floor = float(np.median(ref_s))
    # 2-component GMM crossover on log(focal)
    thr, method = np.nan, "gmm_xover"
    if finite.sum() >= 50:
        x = np.log(s[finite]).reshape(-1, 1)
        gm = GaussianMixture(2, random_state=SEED, n_init=3).fit(x)
        mu = gm.means_.ravel(); lo, hi = np.argsort(mu)
        n_hi = int((gm.predict(x) == hi).sum())
        if (np.exp(mu[hi] - mu[lo]) > 1.1) and n_hi >= 20:     # two genuinely separated modes
            grid = np.linspace(mu[lo], mu[hi], 2001).reshape(-1, 1)
            post = gm.predict_proba(grid)[:, hi]
            thr = float(np.exp(grid[np.argmin(np.abs(post - 0.5)), 0]))
    if not np.isfinite(thr):
        thr = float(np.percentile(ref_s, 90)) if ref_s.size >= 20 else float(np.nanpercentile(s, 90))
        method = "ref_p90_fallback" if ref_s.size >= 20 else "global_p90_fallback"
    if np.isfinite(floor):
        thr = max(thr, floor)
    call = np.where(np.isfinite(s), s > thr, False)
    acnv.obs.loc[m, "cnv_malignant"]   = call
    acnv.obs.loc[m, "cnv_call_method"] = method
    # validation against the orthogonal TCR-clonal label (TCR-covered cells only)
    tcr_m = m & acnv.obs["has_tcr"].to_numpy()
    yt = acnv.obs.loc[tcr_m, "tcr_is_malignant"].to_numpy().astype(bool)
    f1t, f1v, n_tcr = _best_f1_thr(acnv.obs.loc[tcr_m, SCORE_COL].to_numpy(dtype=float), yt)
    msg = f"  | TCR-F1opt thr={f1t:.4f} f1={f1v:.2f} n={n_tcr}" if np.isfinite(f1t) else ""
    print(f"[{d}] {method:18s} thr={thr:.4f}  cnv_malignant={int(call.sum())}/{int(m.sum())}{msg}")

# propagate onto the full atlas obs (NaN / "" / False outside the lymphoid CNV query)
adata.obs["cnv_score"]       = acnv.obs["cnv_score"].reindex(adata.obs_names)
adata.obs["cnv_cell_score"]  = acnv.obs["cnv_cell_score"].reindex(adata.obs_names)
adata.obs["cnv_focal_score"] = acnv.obs["cnv_focal_score"].reindex(adata.obs_names)
adata.obs["cnv_leiden"]      = acnv.obs["cnv_leiden"].reindex(adata.obs_names).fillna("")
adata.obs["cnv_call_method"] = acnv.obs["cnv_call_method"].reindex(adata.obs_names).fillna("")
adata.obs["cnv_malignant"]   = acnv.obs["cnv_malignant"].reindex(adata.obs_names).fillna(False).astype(bool)
print("\ncnv_malignant cells:", int(adata.obs["cnv_malignant"].sum()))
```

```
loaded cached inferCNV (li2024_cnv_per_cell_v4_lymphoid.parquet) for ['CTCL1', 'CTCL2', 'CTCL3', 'CTCL4', 'CTCL5', 'CTCL6', 'CTCL7', 'CTCL8', 'PT11', 'PT35', 'PT47', 'PT50', 'PT52', 'PT53', 'PT55', 'PT56']
[CTCL1] gmm_xover          thr=0.0325  cnv_malignant=7275/9269  | TCR-F1opt thr=0.0087 f1=0.93 n=8181
[CTCL2] gmm_xover          thr=0.0691  cnv_malignant=2226/5357  | TCR-F1opt thr=0.0762 f1=0.85 n=4692
[CTCL3] gmm_xover          thr=0.0352  cnv_malignant=8965/13376  | TCR-F1opt thr=0.0077 f1=0.83 n=11557
[CTCL4] gmm_xover          thr=0.0593  cnv_malignant=7247/12940  | TCR-F1opt thr=0.0862 f1=0.37 n=9910
[CTCL5] gmm_xover          thr=0.0413  cnv_malignant=10511/15057  | TCR-F1opt thr=0.0380 f1=0.72 n=12514
[CTCL6] gmm_xover          thr=0.0427  cnv_malignant=11183/18012  | TCR-F1opt thr=0.0657 f1=0.55 n=14053
[CTCL7] gmm_xover          thr=0.0511  cnv_malignant=31408/56515  | TCR-F1opt thr=0.0563 f1=0.19 n=32901
[CTCL8] gmm_xover          thr=0.0360  cnv_malignant=8265/12229  | TCR-F1opt thr=0.0254 f1=0.77 n=8553
[PT11] gmm_xover          thr=0.0528  cnv_malignant=1659/2442  | TCR-F1opt thr=0.0341 f1=0.96 n=1836
[PT35] gmm_xover          thr=0.0612  cnv_malignant=3501/5085
[PT47] gmm_xover          thr=0.0357  cnv_malignant=391/561  | TCR-F1opt thr=0.0281 f1=0.86 n=379
[PT50] gmm_xover          thr=0.0315  cnv_malignant=2314/3751  | TCR-F1opt thr=0.0343 f1=0.67 n=971
[PT52] gmm_xover          thr=0.0354  cnv_malignant=171/263  | TCR-F1opt thr=0.0371 f1=0.88 n=216
[PT53] gmm_xover          thr=0.0508  cnv_malignant=1714/2564  | TCR-F1opt thr=0.0456 f1=0.75 n=2191
[PT55] gmm_xover          thr=0.0617  cnv_malignant=2057/2480
[PT56] gmm_xover          thr=0.0450  cnv_malignant=3506/5566  | TCR-F1opt thr=0.0649 f1=0.64 n=3774

cnv_malignant cells: 102393

/tmp/ipykernel_2181621/1607207046.py:140: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  adata.obs["cnv_malignant"]   = acnv.obs["cnv_malignant"].reindex(adata.obs_names).fillna(False).astype(bool)
```

```python  # cell 19
# Step 5 (eval) — inferCNV malignancy vs the preannotated `tumor_cell`, by calling method.
# Binary call (cnv_malignant) -> precision/recall/F1/Jaccard; the FOCAL per-CELL cnv_focal_score ->
# threshold-free AUROC/AP (separability before any cutoff). Per donor, then pooled per calling method.
from sklearn.metrics import roc_auc_score, average_precision_score

ev = adata.obs[adata.obs["donor"].isin(CNV_DONORS) & adata.obs["cnv_focal_score"].notna()].copy()


def _perf(g):
    yt = g["cached_malignant"].to_numpy().astype(bool)
    yp = g["cnv_malignant"].to_numpy().astype(bool)
    sc_ = g["cnv_focal_score"].to_numpy()                      # focal per-cell score (drives the call)
    tp = int((yp & yt).sum()); fp = int((yp & ~yt).sum()); fn = int((~yp & yt).sum())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    out = dict(n=len(g), n_tumor=int(yt.sum()), n_cnv=int(yp.sum()),
               precision=round(prec, 3), recall=round(rec, 3), f1=round(f1, 3),
               jaccard=round(tp / max(1, tp + fp + fn), 3))
    out["auroc"] = round(roc_auc_score(yt, sc_), 3) if 0 < yt.sum() < len(yt) else np.nan
    out["ap"]    = round(average_precision_score(yt, sc_), 3) if 0 < yt.sum() < len(yt) else np.nan
    return pd.Series(out)


per_donor = ev.groupby("donor", observed=True).apply(_perf)
per_donor.insert(0, "method", ev.groupby("donor", observed=True)["cnv_call_method"].first())
print("inferCNV vs preannotated tumor_cell — per donor:")
print(per_donor.to_string())

by_method = ev.groupby("cnv_call_method", observed=True).apply(_perf)
overall = _perf(ev).to_frame("ALL").T
perf_table = pd.concat([by_method, overall])
perf_table.insert(0, "donors", list(ev.groupby("cnv_call_method", observed=True)["donor"].nunique())
                  + [ev["donor"].nunique()])
print("\npooled by calling method (+ overall):")
print(perf_table.to_string())
perf_table
```

```
/tmp/ipykernel_2181621/2752331132.py:24: FutureWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  per_donor = ev.groupby("donor", observed=True).apply(_perf)
/tmp/ipykernel_2181621/2752331132.py:29: FutureWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  by_method = ev.groupby("cnv_call_method", observed=True).apply(_perf)

inferCNV vs preannotated tumor_cell — per donor:
          method        n  n_tumor    n_cnv  precision  recall     f1  jaccard  auroc     ap
donor                                                                                       
CTCL1  gmm_xover   9269.0   8069.0   7275.0      0.926   0.835  0.879    0.783  0.790  0.955
CTCL2  gmm_xover   5357.0   1907.0   2226.0      0.755   0.881  0.813    0.685  0.931  0.876
CTCL3  gmm_xover  13376.0   9524.0   8965.0      0.763   0.718  0.740    0.587  0.624  0.796
CTCL4  gmm_xover  12940.0   1073.0   7247.0      0.096   0.651  0.168    0.092  0.632  0.296
CTCL5  gmm_xover  15057.0   7724.0  10511.0      0.617   0.840  0.712    0.552  0.729  0.742
CTCL6  gmm_xover  18012.0   3773.0  11183.0      0.296   0.878  0.443    0.285  0.775  0.533
CTCL7  gmm_xover  56515.0   3673.0  31408.0      0.073   0.623  0.131    0.070  0.555  0.079
CTCL8  gmm_xover  12229.0   7139.0   8265.0      0.667   0.772  0.716    0.557  0.694  0.763
PT11   gmm_xover   2442.0   1965.0   1659.0      0.984   0.831  0.901    0.819  0.960  0.989
PT35   gmm_xover   5085.0   4092.0   3501.0      0.946   0.809  0.872    0.773  0.888  0.969
PT47   gmm_xover    561.0    279.0    391.0      0.619   0.867  0.722    0.565  0.716  0.633
PT50   gmm_xover   3751.0    603.0   2314.0      0.248   0.952  0.394    0.245  0.818  0.426
PT52   gmm_xover    263.0    144.0    171.0      0.784   0.931  0.851    0.740  0.875  0.882
PT53   gmm_xover   2564.0   1160.0   1714.0      0.550   0.813  0.656    0.488  0.682  0.585
PT55   gmm_xover   2480.0   2215.0   2057.0      0.992   0.921  0.955    0.914  0.977  0.997
PT56   gmm_xover   5566.0   1287.0   3506.0      0.326   0.888  0.477    0.313  0.794  0.561

pooled by calling method (+ overall):
           donors         n  n_tumor     n_cnv  precision  recall     f1  jaccard  auroc     ap
gmm_xover      16  165467.0  54627.0  102393.0      0.426   0.798  0.555    0.384  0.639  0.518
ALL            16  165467.0  54627.0  102393.0      0.426   0.798  0.555    0.384  0.639  0.518

           donors         n  n_tumor     n_cnv  precision  recall     f1  \
gmm_xover      16  165467.0  54627.0  102393.0      0.426   0.798  0.555   
ALL            16  165467.0  54627.0  102393.0      0.426   0.798  0.555   

           jaccard  auroc     ap  
gmm_xover    0.384  0.639  0.518  
ALL          0.384  0.639  0.518
```

```python  # cell 20
# Step 5 (diagnose) — WHY cnv_malignant disagrees with the preannotated tumor_cell.
# Cheap (cached scores + obs, no recompute). D1 separates "is the per-cell CNV signal good"
# (AUROC of cnv_focal_score) from "is the threshold good" (precision/recall). D2 still inspects the
# per-CLUSTER cnv_score to show cluster mixing (the reason the old cluster-level call failed).
# Protocol from infercnvpy_atlas_per_sample_fix.md (Steps 2/6/9).
from sklearn.metrics import roc_auc_score, average_precision_score

dx = adata.obs[adata.obs["donor"].isin(CNV_DONORS) & adata.obs["cnv_focal_score"].notna()].copy()
dx["is_tumor"]   = dx["cached_malignant"].astype(bool)
dx["is_cnvcall"] = dx["cnv_malignant"].astype(bool)
# diploid reference cells (same definition feeding the threshold): within-donor non-clonal T + benign-T
dx["is_diploid"] = ((dx["has_tcr"] & ~dx["tcr_is_dominant_clone"]
                     & (dx["cell_type"].astype(str) != "tumor_cell"))
                    | dx["cell_type"].astype(str).isin(BENIGN_T))


def _au(y, s):
    y = y.to_numpy().astype(bool); s = s.to_numpy()
    if not (0 < y.sum() < len(y)):
        return (np.nan, np.nan)
    return (round(roc_auc_score(y, s), 3), round(average_precision_score(y, s), 3))


# ---- D1 (signal vs threshold) + D3 (reference separation), per donor ----
rows = []
for d, g in dx.groupby("donor", observed=True):
    yt = g["is_tumor"].to_numpy(); yp = g["is_cnvcall"].to_numpy()
    tp = int((yt & yp).sum()); fp = int((~yt & yp).sum()); fn = int((yt & ~yp).sum())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn); f1 = 2 * prec * rec / max(1e-9, prec + rec)
    auroc, ap = _au(g["is_tumor"], g["cnv_focal_score"])       # focal per-CELL score
    refm = g.loc[g["is_diploid"], "cnv_focal_score"]; tumm = g.loc[g["is_tumor"], "cnv_focal_score"]
    rows.append(dict(donor=d, method=g["cnv_call_method"].iloc[0], n=len(g), n_tumor=int(yt.sum()),
                     n_clust=int(g["cnv_score"].round(6).nunique()),
                     precision=round(prec, 3), recall=round(rec, 3), f1=round(f1, 3),
                     auroc=auroc, ap=ap, ref_mean=round(refm.mean(), 4), tum_mean=round(tumm.mean(), 4),
                     sep=round(tumm.mean() / refm.mean(), 2) if refm.mean() > 0 else np.nan))
diag = pd.DataFrame(rows).set_index("donor")
print("D1/D3 — per donor (per-cell signal AUROC vs the call's precision/recall, + reference separation):")
print(diag.to_string())

med_auroc = diag["auroc"].dropna().median(); med_f1 = diag["f1"].median()
print(f"\nVERDICT: median AUROC(cnv_focal_score vs tumor)={med_auroc:.3f}  median F1={med_f1:.3f}")
if med_auroc >= 0.8 and med_f1 < 0.6:
    print("  -> per-cell signal OK, THRESHOLD still off -> tune the GMM cut / floor in the compute cell.")
elif med_auroc < 0.7:
    print("  -> cnv_focal_score lacks malignant signal -> REFERENCE/SCORE (tune TOPK_FRAC, fix F2/F3).")
else:
    print("  -> mixed; inspect per-donor rows + heatmaps.")

# ---- D2 — per-CLUSTER cnv_score purity (why the OLD cluster-level call lost precision/recall) ----
cl = (dx.groupby("cnv_leiden", observed=True)
        .agg(donor=("donor", "first"), n=("is_tumor", "size"),
             tumor_frac=("is_tumor", "mean"), cluster_score=("cnv_score", "first"),
             called_frac=("is_cnvcall", "mean")))
mixed = cl[(cl["tumor_frac"] > 0.2) & (cl["tumor_frac"] < 0.8)]
print(f"\nD2 — {len(cl)} cnv_leiden clusters; {len(mixed)} 'mixed' (tumor_frac 0.2-0.8) holding "
      f"{int(mixed['n'].sum())} cells (the per-cell call no longer pays this cluster-mixing penalty).")
print(cl.sort_values("cluster_score", ascending=False).head(15).round(3).to_string())

# ---- D4 — where the errors live ----
fp_ = dx[dx["is_cnvcall"] & ~dx["is_tumor"]]; fn_ = dx[dx["is_tumor"] & ~dx["is_cnvcall"]]
print(f"\nD4 — FP={len(fp_)} (cnv-malignant, not tumor_cell) | FN={len(fn_)} (tumor_cell, missed)")
print("  FP cell_type:", fp_["cell_type"].value_counts().head(5).to_dict())
print("  FN cell_type:", fn_["cell_type"].value_counts().head(5).to_dict())
print(f"  FP has_tcr/tcr_malignant: {int(fp_['has_tcr'].sum())}/{int(fp_['tcr_is_malignant'].sum())}"
      f"   FN has_tcr/tcr_malignant: {int(fn_['has_tcr'].sum())}/{int(fn_['tcr_is_malignant'].sum())}")


# ---- D5 — realistic ceiling: inferCNV vs an independent label ----
def _jac(a, b):
    a = a.to_numpy().astype(bool); b = b.to_numpy().astype(bool)
    return round(int((a & b).sum()) / max(1, int((a | b).sum())), 3)


print("\nD5 — agreement ceiling (Jaccard):")
print("  cnv_malignant vs tcr_is_malignant:", _jac(dx["is_cnvcall"], dx["tcr_is_malignant"]))
print("  tumor_cell    vs tcr_is_malignant:", _jac(dx["is_tumor"], dx["tcr_is_malignant"]))
```

```
D1/D3 — per donor (per-cell signal AUROC vs the call's precision/recall, + reference separation):
          method      n  n_tumor  n_clust  precision  recall     f1  auroc     ap  ref_mean  tum_mean   sep
donor                                                                                                      
CTCL1  gmm_xover   9269     8069       15      0.926   0.835  0.879  0.790  0.955    0.0328    0.0546  1.66
CTCL2  gmm_xover   5357     1907       15      0.755   0.881  0.813  0.931  0.876    0.0469    0.0882  1.88
CTCL3  gmm_xover  13376     9524       15      0.763   0.718  0.740  0.624  0.796    0.0396    0.0488  1.23
CTCL4  gmm_xover  12940     1073       17      0.096   0.651  0.168  0.632  0.296    0.0598    0.0691  1.15
CTCL5  gmm_xover  15057     7724       16      0.617   0.840  0.712  0.729  0.742    0.0462    0.0661  1.43
CTCL6  gmm_xover  18012     3773       17      0.296   0.878  0.443  0.775  0.533    0.0476    0.0720  1.51
CTCL7  gmm_xover  56515     3673       27      0.073   0.623  0.131  0.555  0.079    0.0545    0.0583  1.07
CTCL8  gmm_xover  12229     7139       14      0.667   0.772  0.716  0.694  0.763    0.0406    0.0557  1.37
PT11   gmm_xover   2442     1965        9      0.984   0.831  0.901  0.960  0.989    0.0249    0.0776  3.11
PT35   gmm_xover   5085     4092       12      0.946   0.809  0.872  0.888  0.969    0.0431    0.0740  1.72
PT47   gmm_xover    561      279       13      0.619   0.867  0.722  0.716  0.633    0.0431    0.0622  1.44
PT50   gmm_xover   3751      603       14      0.248   0.952  0.394  0.818  0.426    0.0390    0.0683  1.75
PT52   gmm_xover    263      144       13      0.784   0.931  0.851  0.875  0.882    0.0319    0.0668  2.09
PT53   gmm_xover   2564     1160       15      0.550   0.813  0.656  0.682  0.585    0.0541    0.0662  1.22
PT55   gmm_xover   2480     2215       12      0.992   0.921  0.955  0.977  0.997    0.0266    0.0942  3.53
PT56   gmm_xover   5566     1287       16      0.326   0.888  0.477  0.794  0.561    0.0481    0.0686  1.43

VERDICT: median AUROC(cnv_focal_score vs tumor)=0.782  median F1=0.719
  -> mixed; inspect per-donor rows + heatmaps.

D2 — 242 cnv_leiden clusters; 73 'mixed' (tumor_frac 0.2-0.8) holding 30087 cells (the per-cell call no longer pays this cluster-mixing penalty).
            donor     n  tumor_frac  cluster_score  called_frac
cnv_leiden                                                     
CTCL2_9     CTCL2   239       0.987          0.013        0.987
CTCL2_0     CTCL2  1374       0.989          0.012        0.933
PT55_2       PT55   678       0.990          0.011        0.935
PT35_0       PT35  1492       0.999          0.011        0.962
PT55_0       PT55  1422       0.999          0.011        0.942
PT11_8       PT11    51       0.980          0.011        0.980
CTCL4_12    CTCL4   385       0.519          0.010        0.777
PT11_0       PT11  1203       0.992          0.009        0.929
PT35_5       PT35   660       0.968          0.009        0.811
CTCL6_3     CTCL6  1838       0.897          0.008        0.931
PT56_4       PT56   835       0.901          0.008        0.901
PT35_1       PT35   996       0.977          0.008        0.753
CTCL5_7     CTCL5  1301       0.969          0.008        0.939
PT53_0       PT53   853       0.943          0.007        0.852
PT50_9       PT50   332       0.816          0.007        0.973

D4 — FP=58808 (cnv-malignant, not tumor_cell) | FN=11042 (tumor_cell, missed)
  FP cell_type: {'Th': 23884, 'Tc': 17794, 'Treg': 12445, 'Tc17_Th17': 2212, 'Tc_IL13_IL22': 825}
  FN cell_type: {'tumor_cell': 11042, 'Basal': 0, 'DC1': 0, 'DC2': 0, 'Differentiated_KC': 0}
  FP has_tcr/tcr_malignant: 36390/369   FN has_tcr/tcr_malignant: 8808/8034

D5 — agreement ceiling (Jaccard):
  cnv_malignant vs tcr_is_malignant: 0.295
  tumor_cell    vs tcr_is_malignant: 0.73
```

```python  # cell 21
# Step 5 (heatmap) — confirm the CNV signal is real. Recompute 1-2 representative donors keeping
# X_cnv, then chromosome heatmaps by status and by cell_type (infercnvpy_atlas_per_sample_fix.md
# Step 7). MF-typical: gains 7q/8q/17q, losses 9p21/10q/17p/13q in tumor T; reference T -> grey.
# HEAVY (re-runs infercnv per picked donor) -> GPU kernel. Reuses acnv/HEALTHY_REF/CNV_N_JOBS/
# CNV_CHUNK from the Step-5 compute cell.
#
# Why the first heatmaps looked empty: cnv.pl.chromosome_heatmap auto-scales colors to the global
# min/max (a few outlier cells wash out the bulk), and the calling run used dynamic_threshold=1.5
# (zeroes most windows) + window_size=250 (heavy smoothing). For *visualization* we drop the
# denoising, use a smaller window, and set explicit symmetric color limits from the 99th pct of
# |X_cnv|. (The calling run in cell 8bef3c2e is unchanged.)
RUN_HEATMAP = True
HM_WINDOW   = 100        # less smoothing than the calling run (250) -> visible bands
if RUN_HEATMAP:
    tum_by_donor = dx.groupby("donor", observed=True)["is_tumor"].sum().sort_values()
    low_pool = tum_by_donor[tum_by_donor > 20]
    pick = [tum_by_donor.index[-1]] + ([low_pool.index[0]] if len(low_pool) else [])
    pick = list(dict.fromkeys(pick))                    # high-burden first, then low-burden; dedup
    print("heatmap donors (high-burden, low-burden):", pick)
    for d in pick:
        q = acnv[acnv.obs["donor"] == d]
        sub = ad.concat([q, HEALTHY_REF], join="inner", index_unique=None)
        sub.var = acnv.var.loc[sub.var_names].copy()    # restore genomic positions after concat
        ref_cats = [c for c in ["nonclonal", "healthy"] if int((sub.obs["cnv_ref"] == c).sum()) >= 20]
        cnv.tl.infercnv(sub, reference_key="cnv_ref", reference_cat=ref_cats,
                        window_size=HM_WINDOW, dynamic_threshold=None,   # keep signal visible
                        n_jobs=CNV_N_JOBS, chunksize=CNV_CHUNK)
        sub.obs["status"] = np.where(sub.obs["cnv_ref"] != "query", "reference",
                            np.where(sub.obs["cell_type"].astype(str) == "tumor_cell", "tumor_cell", "other_T"))
        # symmetric color limits from the bulk (99th pct of |X_cnv|), not the global max
        Xc = sub.obsm["X_cnv"]
        Xc = Xc.toarray() if hasattr(Xc, "toarray") else np.asarray(Xc)
        vlim = float(np.nanpercentile(np.abs(Xc), 99)) or 0.05
        print(f"[{d}] n={sub.n_obs}  color vlim=±{vlim:.4f}")
        cnv.pl.chromosome_heatmap(sub, groupby="status",    vmin=-vlim, vmax=vlim, save=f"_cnv_{d}_by_status.png")
        cnv.pl.chromosome_heatmap(sub, groupby="cell_type", vmin=-vlim, vmax=vlim, save=f"_cnv_{d}_by_celltype.png")
        del sub, q, Xc
        gc.collect()
        print(f"[{d}] chromosome heatmaps saved (chromosome_heatmap_cnv_{d}_*.png)")
else:
    print("RUN_HEATMAP=False — skipping the heavy chromosome-heatmap recompute")
```

## Step 6 — agreement: cached `tumor_cell` vs TCR vs inferCNV

Three independent malignancy calls on the same cells (over `CNV_DONORS`): the atlas's published
`tumor_cell`, our dominant-TCR-clone `tcr_is_malignant`, and the inferCNV `cnv_malignant`. Pairwise
Jaccard + crosstabs + per-donor counts. The paper used CNV to *confirm* that the dominant TCR clone
is the malignant population, so all three should agree strongly on the high-burden donors.

```python  # cell 23
import itertools

covc = adata.obs[adata.obs["donor"].isin(CNV_DONORS)].copy()
labels = {"cached_tumor_cell": covc["cached_malignant"].to_numpy(),
          "tcr_malignant":     covc["tcr_is_malignant"].to_numpy(),
          "cnv_malignant":     covc["cnv_malignant"].to_numpy()}

print("pairwise agreement over CTCL2/3/4:")
for a_, b_ in itertools.combinations(labels, 2):
    A, B = labels[a_], labels[b_]
    inter, uni = int((A & B).sum()), int((A | B).sum())
    print(f"  {a_:18s} vs {b_:16s}  jaccard={inter / max(1, uni):.3f}  "
          f"overlap={inter:6d}  (sizes {int(A.sum())} vs {int(B.sum())})")

print("\ncnv_malignant × cached tumor_cell:\n",
      pd.crosstab(covc["cnv_malignant"], covc["cached_malignant"]))
print("\ncnv_malignant × tcr_malignant:\n",
      pd.crosstab(covc["cnv_malignant"], covc["tcr_is_malignant"]))

print("\nper-donor malignant counts (3 methods):")
print(covc.groupby("donor", observed=True).agg(
    n_cells=("cnv_malignant", "size"),
    cached_tumor=("cached_malignant", "sum"),
    tcr=("tcr_is_malignant", "sum"),
    cnv=("cnv_malignant", "sum")).to_string())

print("\nmean cnv_score by cached tumor_cell (sanity — tumor should be higher):")
print(covc.groupby("cached_malignant")["cnv_score"].mean().to_string())
print("\ntriple-positive (tumor_cell & TCR & CNV):",
      int((labels["cached_tumor_cell"] & labels["tcr_malignant"] & labels["cnv_malignant"]).sum()))
```

```
pairwise agreement over CTCL2/3/4:
  cached_tumor_cell  vs tcr_malignant     jaccard=0.722  overlap= 40266  (sizes 54627 vs 41373)
  cached_tumor_cell  vs cnv_malignant     jaccard=0.384  overlap= 43585  (sizes 54627 vs 102393)
  tcr_malignant      vs cnv_malignant     jaccard=0.293  overlap= 32601  (sizes 41373 vs 102393)

cnv_malignant × cached tumor_cell:
 cached_malignant   False  True 
cnv_malignant                  
False             196346  11042
True               58808  43585

cnv_malignant × tcr_malignant:
 tcr_is_malignant   False  True 
cnv_malignant                  
False             198616   8772
True               69792  32601

per-donor malignant counts (3 methods):
       n_cells  cached_tumor   tcr    cnv
donor                                    
CTCL1    21684          8069  7229   7275
CTCL2    13570          1907  1852   2226
CTCL3    26623          9524  8229   8965
CTCL4    25187          1073   768   7247
CTCL5    41201          7724  6707  10511
CTCL6    37073          3773  3168  11183
CTCL7    77712          3673  3159  31408
CTCL8    34108          7139  5372   8265
PT11      4288          1965  1640   1659
PT35      6522          4092     0   3501
PT47       908           279   257    391
PT50      5161           603   410   2314
PT52       479           144   132    171
PT53      3142          1160  1257   1714
PT55      3297          2215     0   2057
PT56      8826          1287  1193   3506

mean cnv_score by cached tumor_cell (sanity — tumor should be higher):
cached_malignant
False    0.005172
True     0.006356

triple-positive (tumor_cell & TCR & CNV): 32232
```

```python  # cell 24
import matplotlib.pyplot as plt

# cnv_score separation: reference (diploid) vs cached tumor_cell vs TCR-malignant, per donor
fig, ax = plt.subplots(figsize=(5, 3))
order = CNV_DONORS
data, labels = [], []
for d in order:
    g = covc[covc["donor"] == d]
    data.append(g.loc[~g["cached_malignant"], "cnv_score"].dropna())
    data.append(g.loc[g["cached_malignant"], "cnv_score"].dropna())
pos = np.arange(len(order) * 2)
bp = ax.boxplot(data, positions=pos, widths=0.7, showfliers=False, patch_artist=True)
for i, p in enumerate(bp["boxes"]):
    p.set_facecolor("#cccccc" if i % 2 == 0 else "#d6604d")
ax.set_xticks(pos[::2] + 0.5); ax.set_xticklabels(order)
ax.set_ylabel("cnv_score"); ax.set_title("inferCNV score higher in cached tumor_cell")
ax.legend([bp["boxes"][0], bp["boxes"][1]], ["non-tumor_cell", "tumor_cell"], fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(FIG_DIR / "li2024_tcr_cnv_score_by_status.png", dpi=150, bbox_inches="tight")
plt.show()
```

```
<Figure size 500x300 with 1 Axes>
```

## Step 7 — persist obs annotations

TCR + inferCNV calls together. `cnv_*` columns are populated only for `CNV_DONORS` (the fresh donors
with TCR); elsewhere `cnv_score` is NaN and `cnv_malignant` is False.

```python  # cell 26
keep = ["donor", "study", "stage", "cell_type", "cached_malignant", "has_tcr", "tcr_source",
        "tcr_is_malignant", "tcr_is_dominant_clone", "tcr_clone_id", "tcr_clone_size",
        "cnv_score", "cnv_cell_score", "cnv_leiden", "cnv_call_method", "cnv_malignant", "bc16", "lane"]
out = adata.obs[keep].copy()
out.to_parquet(OUT_PARQUET)
print("wrote", OUT_PARQUET, out.shape)
```

```
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/Li2024_atlas/li2024_tcr_malignancy.parquet (419579, 18)
```

## Step 8 — UMAP overlay (HEAVY — GPU kernel; cached)

Neighbors + UMAP on the atlas `X_scVI` latent (cached to `.npy`, reloaded on re-run), colored by
published cell type, cached `tumor_cell`, our TCR-malignant call, inferCNV call, and donor. `cnv_*`
overlays are only meaningful on `CNV_DONORS`. Run on the GPU kernel.

```python  # cell 28
# UMAP on the atlas X_scVI latent — CACHED to .npy (aligned to obs order). Re-runs reload.
UMAP_NPY = NB_DIR / "data" / "Li2024_atlas" / "li2024_atlas_umap.npy"
FORCE_UMAP = False
if (not FORCE_UMAP) and UMAP_NPY.exists():
    adata.obsm["X_umap"] = np.load(UMAP_NPY)
    print("loaded cached UMAP", adata.obsm["X_umap"].shape)
else:
    sc.pp.neighbors(adata, use_rep="X_scVI", random_state=SEED)
    sc.tl.umap(adata, random_state=SEED)
    np.save(UMAP_NPY, adata.obsm["X_umap"])
    print("computed + cached UMAP ->", UMAP_NPY)

adata.obs["tcr_malignant_lbl"] = adata.obs["tcr_is_malignant"].map(
    {True: "TCR_malignant", False: "other"}).astype("category")
adata.obs["cached_malignant_lbl"] = adata.obs["cached_malignant"].map(
    {True: "tumor_cell", False: "other"}).astype("category")
adata.obs["cnv_malignant_lbl"] = adata.obs["cnv_malignant"].map(
    {True: "CNV_malignant", False: "other"}).astype("category")

sc.pl.umap(
    adata, color=["cell_type", "cached_malignant_lbl", "tcr_malignant_lbl",
                  "cnv_malignant_lbl", "cnv_score", "donor"],
    ncols=2, frameon=False, size=3, wspace=0.25,
    save="_li2024_tcr_cnv_malignancy.png",
)

# zoom: the CNV/TCR donors where all three calls are defined
ad_cnv = adata[adata.obs["donor"].isin(CNV_DONORS)].copy()
sc.pl.umap(
    ad_cnv, color=["cached_malignant_lbl", "tcr_malignant_lbl", "cnv_malignant_lbl", "cnv_score"],
    ncols=2, frameon=False, size=5, wspace=0.25,
    save="_li2024_ctcl_three_way.png",
)
```

```
loaded cached UMAP (419579, 2)

... storing 'bc16' as categorical
... storing 'tcr_clone_id' as categorical
... storing 'tcr_source' as categorical
... storing 'cnv_leiden' as categorical
... storing 'cnv_call_method' as categorical

WARNING: saving figure to file figures/umap_li2024_tcr_cnv_malignancy.png

<Figure size 1600x1440 with 7 Axes>
WARNING: saving figure to file figures/umap_li2024_ctcl_three_way.png

<Figure size 1600x960 with 5 Axes>
```

```python  # cell 29
# Step 8c — cluster-level malignancy: high-res Leiden on the scVI latent (T compartment), label
# each cluster from our per-cell CNV call, then compare cluster-vs-cached at MATCHED (cluster)
# resolution. This is the conventional inferCNV workflow and mirrors how the paper's tumor_cell
# label was made (a cluster annotation), so it sidesteps the per-cell-score vs cluster-label
# granularity mismatch that caps the per-cell F1. HEAVY (neighbors+Leiden) -> GPU kernel.
from sklearn.metrics import roc_auc_score

LEIDEN_RES = 2.0       # high resolution -> fine, clone-pure clusters
CLUST_FRAC = 0.7       # cluster called malignant if >= this fraction of its cells are cnv_malignant

# paper's pre-computed scVI T-cell UMAP for the overlay (li2024_tcells_umap.npy; no recompute)
TUMAP_NPY = NB_DIR / "data" / "Li2024_atlas" / "li2024_tcells_umap.npy"
T_TYPES = ["tumor_cell"] + BENIGN_T
if "X_umap_tcell" not in adata.obsm:
    _full_t = adata.obs["cell_type"].astype(str).isin(T_TYPES).to_numpy()
    _um = np.load(TUMAP_NPY)
    assert _um.shape[0] == int(_full_t.sum()), f"cached T-UMAP {_um.shape} != {int(_full_t.sum())} T cells"
    _xt = np.full((adata.n_obs, 2), np.nan); _xt[_full_t] = _um
    adata.obsm["X_umap_tcell"] = _xt

tmask = (adata.obs["donor"].isin(CNV_DONORS)
         & adata.obs["cell_type"].astype(str).isin(T_TYPES)
         & adata.obs["cnv_focal_score"].notna())
at = adata[tmask].copy()                                               # carries X_umap_tcell + *_lbl cols

# high-res Leiden on the paper's scVI latent
sc.pp.neighbors(at, use_rep="X_scVI", random_state=SEED)
sc.tl.leiden(at, resolution=LEIDEN_RES, random_state=SEED, key_added="scvi_leiden",
             flavor="igraph", n_iterations=2, directed=False)

# per-cluster malignancy from OUR per-cell CNV call: majority vote of cnv_malignant
g = at.obs.groupby("scvi_leiden", observed=True)
frac_cnv   = g["cnv_malignant"].mean()                                 # fraction cnv-malignant
clust_malig = frac_cnv >= CLUST_FRAC
at.obs["leiden_cnv_malignant"] = at.obs["scvi_leiden"].map(clust_malig).astype(bool)
at.obs["leiden_cnv_lbl"] = at.obs["leiden_cnv_malignant"].map(
    {True: "malignant", False: "benign"}).astype("category")
adata.obs["leiden_cnv_malignant"] = (at.obs["leiden_cnv_malignant"]
                                     .reindex(adata.obs_names).fillna(False).astype(bool))
print(f"{at.obs['scvi_leiden'].nunique()} clusters @res={LEIDEN_RES} | "
      f"malignant clusters: {int(clust_malig.sum())} | "
      f"cells called: {int(at.obs['leiden_cnv_malignant'].sum())}/{at.n_obs}")

# ---- UMAP (paper scVI T-cell embedding) overlay ----
sc.pl.embedding(
    at, basis="X_umap_tcell",
    color=["scvi_leiden", "leiden_cnv_lbl", "cached_malignant_lbl", "cnv_focal_score"],
    ncols=2, frameon=False, size=6, wspace=0.25, legend_loc="on data",
    save="_li2024_leiden_cnv_malignancy.png",
)

# ---- comparison: cluster-level leiden-CNV annotation vs cached tumor_cell ----
yt = at.obs["cached_malignant"].to_numpy(bool); yp = at.obs["leiden_cnv_malignant"].to_numpy(bool)
tp = int((yp & yt).sum()); fp = int((yp & ~yt).sum()); fn = int((~yp & yt).sum())
prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
f1 = 2 * prec * rec / max(1e-9, prec + rec); jac = tp / max(1, tp + fp + fn)
print("\nleiden-CNV vs cached tumor_cell (cluster-level, per-cell tally):")
print(f"  precision={prec:.3f} recall={rec:.3f} f1={f1:.3f} jaccard={jac:.3f}  "
      f"(n_tumor={int(yt.sum())}, n_called={int(yp.sum())})")

# cluster as the unit: predict 'tumor cluster' (tumor_frac>=0.5) from cluster mean focal score
cl = g.agg(n=("cached_malignant", "size"), tumor_frac=("cached_malignant", "mean"),
           mean_focal=("cnv_focal_score", "mean"), frac_cnv=("cnv_malignant", "mean"))
cl_truth = (cl["tumor_frac"] >= 0.5).to_numpy()
cl_auroc = (roc_auc_score(cl_truth, cl["mean_focal"].to_numpy())
            if 0 < cl_truth.sum() < len(cl_truth) else np.nan)
print(f"  cluster-level AUROC (cluster mean focal -> tumor cluster): {cl_auroc:.3f}  "
      f"({len(cl)} clusters, {int(cl_truth.sum())} tumor)")

# per-donor F1 of the cluster-level call
print("\n  per-donor leiden-CNV vs cached:")
for d, gd in at.obs.groupby("donor", observed=True):
    a = gd["cached_malignant"].to_numpy(bool); b = gd["leiden_cnv_malignant"].to_numpy(bool)
    tpd = int((a & b).sum()); fpd = int((~a & b).sum()); fnd = int((a & ~b).sum())
    pr = tpd / max(1, tpd + fpd); rc = tpd / max(1, tpd + fnd); f = 2 * pr * rc / max(1e-9, pr + rc)
    print(f"    {d:6s} P={pr:.2f} R={rc:.2f} F1={f:.2f}  (n_tumor={int(a.sum())}/{len(gd)})")
```

```
/tmp/ipykernel_2181621/3109110204.py:39: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  .reindex(adata.obs_names).fillna(False).astype(bool))

52 clusters @res=2.0 | malignant clusters: 19 | cells called: 51879/162357
WARNING: saving figure to file figures/X_umap_tcell_li2024_leiden_cnv_malignancy.png

<Figure size 1600x960 with 5 Axes>

leiden-CNV vs cached tumor_cell (cluster-level, per-cell tally):
  precision=0.924 recall=0.877 f1=0.900 jaccard=0.818  (n_tumor=54627, n_called=51879)
  cluster-level AUROC (cluster mean focal -> tumor cluster): 0.923  (52 clusters, 21 tumor)

  per-donor leiden-CNV vs cached:
    CTCL1  P=0.99 R=0.70 F1=0.82  (n_tumor=8069/9254)
    CTCL2  P=0.97 R=0.97 F1=0.97  (n_tumor=1907/5270)
    CTCL3  P=0.99 R=0.99 F1=0.99  (n_tumor=9524/13210)
    CTCL4  P=0.71 R=0.80 F1=0.75  (n_tumor=1073/12427)
    CTCL5  P=0.98 R=0.88 F1=0.93  (n_tumor=7724/14919)
    CTCL6  P=0.93 R=0.95 F1=0.94  (n_tumor=3773/17494)
    CTCL7  P=0.73 R=0.94 F1=0.83  (n_tumor=3673/55751)
    CTCL8  P=0.98 R=0.78 F1=0.87  (n_tumor=7139/11953)
    PT11   P=0.99 R=0.94 F1=0.97  (n_tumor=1965/2325)
    PT35   P=0.99 R=0.91 F1=0.95  (n_tumor=4092/4883)
    PT47   P=0.86 R=0.94 F1=0.90  (n_tumor=279/548)
    PT50   P=0.42 R=0.88 F1=0.56  (n_tumor=603/3610)
    PT52   P=0.73 R=0.74 F1=0.74  (n_tumor=144/259)
    PT53   P=0.85 R=0.93 F1=0.89  (n_tumor=1160/2493)
    PT55   P=1.00 R=0.96 F1=0.98  (n_tumor=2215/2461)
    PT56   P=0.64 R=0.80 F1=0.71  (n_tumor=1287/5500)
```

```python  # cell 30
# Step 8c-mrvi (HEAVY) — load MRVI latent, build the T/CNV subset, neighbors, then RECOMPUTE
# the UMAP on the MRVI latent for these T cells only (cached to .npy), and Leiden. Threshold knob
# + plotting are in the NEXT cell so you can re-call them without this recompute. GPU kernel.
import anndata as _ad
MRVI_CACHE     = NB_DIR / "data" / "cache" / "mrvi_ctcl_cache.h5ad"
MRVI_TUMAP_NPY = NB_DIR / "data" / "Li2024_atlas" / "li2024_mrvi_tcell_umap.npy"
FORCE_MRVI_UMAP = False
if "X_mrvi" not in adata.obsm:
    _m = _ad.read_h5ad(MRVI_CACHE, backed="r")               # backed: skip loading X
    _df = pd.DataFrame(np.asarray(_m.obsm["X_mrvi"]), index=_m.obs_names)
    adata.obsm["X_mrvi"] = _df.reindex(adata.obs_names).to_numpy()
    print("X_mrvi:", adata.obsm["X_mrvi"].shape,
          "| rows with NaN:", int(np.isnan(adata.obsm["X_mrvi"]).any(1).sum()))

tmask = (adata.obs["donor"].isin(CNV_DONORS)
         & adata.obs["cell_type"].astype(str).isin(T_TYPES)
         & adata.obs["cnv_focal_score"].notna())
atm = adata[tmask].copy()
print("atm (T/CNV subset):", atm.n_obs, "cells")

# neighbors on the MRVI latent — shared by both the UMAP and the Leiden below
sc.pp.neighbors(atm, use_rep="X_mrvi", random_state=SEED)

# T-cell-only MRVI UMAP, cached (aligned to atm/tmask order; delete the .npy or set FORCE to redo)
if (not FORCE_MRVI_UMAP) and MRVI_TUMAP_NPY.exists():
    _u = np.load(MRVI_TUMAP_NPY)
    assert _u.shape[0] == atm.n_obs, f"cached MRVI T-UMAP {_u.shape} != {atm.n_obs} cells (subset changed)"
    atm.obsm["X_umap_mrvi_t"] = _u
    print("loaded cached MRVI T-cell UMAP", _u.shape)
else:
    sc.tl.umap(atm, random_state=SEED)
    atm.obsm["X_umap_mrvi_t"] = atm.obsm["X_umap"]
    np.save(MRVI_TUMAP_NPY, atm.obsm["X_umap_mrvi_t"])
    print("computed + cached MRVI T-cell UMAP ->", MRVI_TUMAP_NPY)

# high-res Leiden on the MRVI latent (same neighbors graph)
sc.tl.leiden(atm, resolution=LEIDEN_RES, random_state=SEED, key_added="mrvi_leiden",
             flavor="igraph", n_iterations=2, directed=False)
print(f"{atm.obs['mrvi_leiden'].nunique()} MRVI clusters @res={LEIDEN_RES}")
```

```
atm (T/CNV subset): 162357 cells
computed + cached MRVI T-cell UMAP -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/Li2024_atlas/li2024_mrvi_tcell_umap.npy
33 MRVI clusters @res=2.0
```

```python  # cell 31
# Step 8c-mrvi (call + plot) — LIGHT: change CLUST_FRAC_MRVI or the plot and re-run THIS cell
# only (no neighbors/Leiden recompute). Writes the calls onto adata.obs for the cache cell.
CLUST_FRAC_MRVI = 0.65        # cluster called malignant if >= this fraction of cells are cnv_malignant

def call_mrvi_clusters(clust_frac):
    frac = atm.obs.groupby("mrvi_leiden", observed=True)["cnv_malignant"].mean()
    malig = frac >= clust_frac
    atm.obs["leiden_cnv_malignant_mrvi"] = atm.obs["mrvi_leiden"].map(malig).astype(bool)
    atm.obs["leiden_cnv_mrvi_lbl"] = atm.obs["leiden_cnv_malignant_mrvi"].map(
        {True: "malignant", False: "benign"}).astype("category")
    adata.obs["leiden_cnv_malignant_mrvi"] = (atm.obs["leiden_cnv_malignant_mrvi"]
        .reindex(adata.obs_names).fillna(False).astype(bool))
    adata.obs["leiden_cnv_mrvi_lbl"] = adata.obs["leiden_cnv_malignant_mrvi"].map(
        {True: "malignant", False: "benign"}).astype("category")
    print(f"clust_frac={clust_frac}: malignant clusters {int(malig.sum())}/{len(malig)} | "
          f"cells called {int(atm.obs['leiden_cnv_malignant_mrvi'].sum())}/{atm.n_obs}")
    return malig

def plot_mrvi_calls():
    sc.pl.embedding(
        atm, basis="X_umap_mrvi_t",
        color=["mrvi_leiden", "leiden_cnv_mrvi_lbl", "cached_malignant_lbl",
               "tcr_malignant_lbl", "cnv_focal_score"],
        ncols=2, frameon=False, size=6, wspace=0.25, legend_loc="on data",
        save="_li2024_leiden_cnv_malignancy_mrvi.png",
    )

def eval_mrvi():
    yt = atm.obs["cached_malignant"].to_numpy(bool)
    yp = atm.obs["leiden_cnv_malignant_mrvi"].to_numpy(bool)
    tp = int((yp & yt).sum()); fp = int((yp & ~yt).sum()); fn = int((~yp & yt).sum())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec); jac = tp / max(1, tp + fp + fn)
    print("MRVI leiden-CNV vs cached tumor_cell (cluster-level, per-cell tally):")
    print(f"  precision={prec:.3f} recall={rec:.3f} f1={f1:.3f} jaccard={jac:.3f}  "
          f"(n_tumor={int(yt.sum())}, n_called={int(yp.sum())})")
    ys = atm.obs["leiden_cnv_malignant"].to_numpy(bool)        # scVI-leiden call from 8c
    print(f"  agreement scVI-leiden vs MRVI-leiden: {(ys == yp).mean():.3f}  (n={len(yp)})")
    print("  per-donor MRVI leiden-CNV vs cached:")
    for d, gd in atm.obs.groupby("donor", observed=True):
        a = gd["cached_malignant"].to_numpy(bool); b = gd["leiden_cnv_malignant_mrvi"].to_numpy(bool)
        tpd = int((a & b).sum()); fpd = int((~a & b).sum()); fnd = int((a & ~b).sum())
        pr = tpd / max(1, tpd + fpd); rc = tpd / max(1, tpd + fnd); f = 2 * pr * rc / max(1e-9, pr + rc)
        print(f"    {d:6s} P={pr:.2f} R={rc:.2f} F1={f:.2f}  (n_tumor={int(a.sum())}/{len(gd)})")

call_mrvi_clusters(CLUST_FRAC_MRVI)
plot_mrvi_calls()
eval_mrvi()
```

```
clust_frac=0.65: malignant clusters 13/33 | cells called 66006/162357

/tmp/ipykernel_2181621/3238726940.py:12: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  .reindex(adata.obs_names).fillna(False).astype(bool))

WARNING: saving figure to file figures/X_umap_mrvi_t_li2024_leiden_cnv_malignancy_mrvi.png

<Figure size 1600x1440 with 6 Axes>
MRVI leiden-CNV vs cached tumor_cell (cluster-level, per-cell tally):
  precision=0.706 recall=0.853 f1=0.773 jaccard=0.630  (n_tumor=54627, n_called=66006)
  agreement scVI-leiden vs MRVI-leiden: 0.855  (n=162357)
  per-donor MRVI leiden-CNV vs cached:
    CTCL1  P=0.97 R=0.63 F1=0.77  (n_tumor=8069/9254)
    CTCL2  P=0.87 R=0.98 F1=0.92  (n_tumor=1907/5270)
    CTCL3  P=0.97 R=0.97 F1=0.97  (n_tumor=9524/13210)
    CTCL4  P=0.31 R=0.88 F1=0.45  (n_tumor=1073/12427)
    CTCL5  P=0.91 R=0.82 F1=0.86  (n_tumor=7724/14919)
    CTCL6  P=0.52 R=0.92 F1=0.66  (n_tumor=3773/17494)
    CTCL7  P=0.28 R=0.98 F1=0.44  (n_tumor=3673/55751)
    CTCL8  P=0.92 R=0.83 F1=0.87  (n_tumor=7139/11953)
    PT11   P=0.98 R=0.93 F1=0.96  (n_tumor=1965/2325)
    PT35   P=0.99 R=0.88 F1=0.93  (n_tumor=4092/4883)
    PT47   P=0.76 R=1.00 F1=0.86  (n_tumor=279/548)
    PT50   P=0.24 R=0.79 F1=0.37  (n_tumor=603/3610)
    PT52   P=0.62 R=0.98 F1=0.76  (n_tumor=144/259)
    PT53   P=0.80 R=0.98 F1=0.88  (n_tumor=1160/2493)
    PT55   P=0.98 R=0.81 F1=0.89  (n_tumor=2215/2461)
    PT56   P=0.52 R=0.73 F1=0.61  (n_tumor=1287/5500)
```

```python  # cell 32
# Step 8b/c cache — persist the malignancy calls (obs, atlas order). Reload to skip the
# heavy leiden recompute in 8c. cnv_malignant = per-cell CNV call BEFORE the cluster rule;
# leiden_cnv_malignant = cluster-level call AFTER the rule (Step 8c).
CALLS_PARQUET = NB_DIR / "data" / "Li2024_atlas" / "li2024_malignancy_calls.parquet"
calls = adata.obs[[
    "donor", "study", "cell_type", "cached_malignant",
    "tcr_is_malignant",          # TCR malignancy
    "cnv_focal_score",           # continuous focal-CNV score
    "cnv_malignant",             # per-cell CNV call (pre clustering rule)
    "leiden_cnv_malignant",      # leiden cluster-level CNV call (scVI latent, 8c)
    "leiden_cnv_malignant_mrvi", # leiden cluster-level CNV call (MRVI latent, 8c-mrvi)
    "leiden_cnv_mrvi_lbl",        # MRVI cluster-level CNV label (malignant/benign)
]].copy()
calls.to_parquet(CALLS_PARQUET)
print("wrote", CALLS_PARQUET, calls.shape)
```

```
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/Li2024_atlas/li2024_malignancy_calls.parquet (419579, 10)
```

```python  # cell 33
# Step 8d — CTCL7 T cells only, on the paper's scVI T-cell UMAP: the four per-cell calls side by side
DONOR = "PT35"
T_TYPES = ["tumor_cell"] + BENIGN_T
if "X_umap_tcell" not in adata.obsm:
    _full_t = adata.obs["cell_type"].astype(str).isin(T_TYPES).to_numpy()
    _um = np.load(NB_DIR / "data" / "Li2024_atlas" / "li2024_tcells_umap.npy")
    assert _um.shape[0] == int(_full_t.sum()), f"cached T-UMAP {_um.shape} != {int(_full_t.sum())} T cells"
    _xt = np.full((adata.n_obs, 2), np.nan); _xt[_full_t] = _um
    adata.obsm["X_umap_tcell"] = _xt

adata.obs["has_tcr_lbl"] = adata.obs["has_tcr"].map(
    {True: "has_TCR", False: "no_TCR"}).astype("category")
adata.obs["leiden_cnv_lbl"] = adata.obs["leiden_cnv_malignant"].map(   # Step 8c cluster-level CNV call
    {True: "malignant", False: "benign"}).astype("category")

m7 = (adata.obs["donor"] == DONOR) & adata.obs["cell_type"].astype(str).isin(T_TYPES)
at7 = adata[m7].copy()
print(f"{DONOR} T cells: {at7.n_obs} | {at7.obs['cell_type'].value_counts().to_dict()}")

sc.pl.embedding(
    at7, basis="X_umap_tcell",
    color=["cached_malignant_lbl", "has_tcr_lbl", "tcr_malignant_lbl", "cnv_malignant_lbl",
           "leiden_cnv_lbl"],
    ncols=2, frameon=False, size=8, wspace=0.25,
    save="_li2024_ctcl7_tcell_four_calls.png",
)
```

```
PT35 T cells: 4883 | {'tumor_cell': 4092, 'Tc': 715, 'Th': 44, 'Treg': 20, 'Tc_IL13_IL22': 9, 'Tc17_Th17': 3}
WARNING: saving figure to file figures/X_umap_tcell_li2024_ctcl7_tcell_four_calls.png

<Figure size 1600x1440 with 5 Axes>
```
