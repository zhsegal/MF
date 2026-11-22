# old/22_final_figures.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/22_final_figures.ipynb` and is never run._

# nb22 — Final figures (MF / CTCL atlas)

Re-renders the chosen panels from nb14 (CNV heatmaps), nb16/17 (per-cell track heatmap) and
nb18 (latent UMAP, factor-separation heatmap, OOF F1/ROC) as **editable hybrid SVGs**:
`svg.fonttype='none'` (vector text) + `rasterized=True` on the heavy data layers + `dpi=300`.

Cache-first. The only GPU steps are **Step 1** (model latent `z`, cached) and **Figure a**
(inferCNV for the 2 chosen donors). Everything else reads cached `.npy`/`.parquet`/`.csv`.
Plotting lives in `final_figures_helpers.py` (imported as `F`).

```python  # cell 1
# ============================================================
# config, paths, SVG style  (knobs mirror nb18 so the cached model loads)
# ============================================================
import sys, gc, hashlib, json, importlib
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import scvi


def _resolve_nb_dir() -> Path:
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("notebooks/MF"), Path("scvi-tools-neural-nmf/notebooks/MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data starting from {start}")


NB_DIR = _resolve_nb_dir()
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))
if str(NB_DIR.parent) not in sys.path:
    sys.path.insert(0, str(NB_DIR.parent))

import atlas_join_helpers as H
import skin_T_cnv_helpers as C
import semantic_malig_helpers as M
import final_figures_helpers as F
importlib.reload(H); importlib.reload(C); importlib.reload(M); importlib.reload(F)
from benchmark_helpers import get_or_build_geneformer_map, train_or_load_semantic_scvi

scvi.settings.seed = 0
np.random.seed(0)
sc.settings.verbosity = 1
F.svg_style()

# ---- shared input paths ----
V2              = NB_DIR / "data" / "atlas_joint" / "skin_T_tcr_malig_v2.h5ad"
GENE_ID_SOURCE  = NB_DIR / "data" / "cache" / "cnmf_malignant_counts.h5ad"
SEMANTIC_CACHE_GENEFORMER = NB_DIR / "data" / "mf_cd4_atlas_geneformer.pt"
MODEL_CACHE_DIR = NB_DIR / "models" / ".model_cache_semantic_geom_cd4_atlas"
EMB_UMAP_NPY    = NB_DIR / "data" / "atlas_joint" / "semantic_geom_cd4_atlas_emb_umap.npy"
CALLS_CSV       = NB_DIR / "data" / "atlas_joint" / "mrvi_malignancy" / "calls.csv"
CNV_CD4_PARQUET = NB_DIR / "data" / "atlas_joint" / "skin_cd4_cd8ref_malignancy.parquet"
GTF             = NB_DIR / "data" / "cache" / "Homo_sapiens.GRCh38.110.chr.gtf.gz"
INTEGRATED_H5   = NB_DIR / "data" / "Integrated_CTCL_skincellatlas_final_portal_tags.h5ad"

# ---- nb22 figure cache (built once on GPU in Step 1) ----
Z_NPY       = NB_DIR / "data" / "atlas_joint" / "semantic_geom_cd4_atlas_z.npy"
OBS_PARQUET = NB_DIR / "data" / "atlas_joint" / "semantic_geom_cd4_atlas_obs.parquet"

# ---- output dir for the final SVGs ----
FIG_DIR = NB_DIR / "figures" / "final"; FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- SemanticSCVI knobs (must match nb18 to resolve + load the cached model) ----
HVG_TOP_N, HVG_FLAVOR = 2500, "seurat_v3"
N_LATENT, BATCH_KEY = 10, "sample_id"
SEMANTIC_GENEFORMER_MAX_EPOCHS = 200
SEMANTIC_GENEFORMER_WARMUP_EPOCHS = 20
SEMANTIC_GENEFORMER_N_EPOCHS_KL_WARMUP = 100
SEMANTIC_GENEFORMER_KWARGS = dict(
    loss_mode="geometric", coherence_weight=1000.0, n_gene_sample=1024,
    n_latent=N_LATENT, n_layers=1, n_hidden=128, dropout_rate=0.1,
    gene_likelihood="nb", weights_positive=True, use_batch_norm=False,
)


def _semantic_cache_slug(kwargs, max_epochs, warmup_epochs, n_epochs_kl_warmup, hvg_top_n, n=10):
    blob = json.dumps(
        {"kwargs": dict(sorted(kwargs.items())), "max_epochs": max_epochs,
         "warmup_epochs": warmup_epochs, "n_epochs_kl_warmup": n_epochs_kl_warmup,
         "hvg_top_n": hvg_top_n},
        default=str, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:n]


SEMANTIC_GENEFORMER_CACHE_DIR = MODEL_CACHE_DIR / "semantic_scvi" / _semantic_cache_slug(
    {**SEMANTIC_GENEFORMER_KWARGS, "batch_key": BATCH_KEY},
    SEMANTIC_GENEFORMER_MAX_EPOCHS, SEMANTIC_GENEFORMER_WARMUP_EPOCHS,
    SEMANTIC_GENEFORMER_N_EPOCHS_KL_WARMUP, HVG_TOP_N)
print("NB_DIR =", NB_DIR)
print("model cache_dir exists:", SEMANTIC_GENEFORMER_CACHE_DIR.exists())
print("CUDA:", torch.cuda.is_available())
```

```
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
Seed set to 0

NB_DIR = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
model cache_dir exists: True
CUDA: True
```

## Step 1 — latent `z` + obs cache  *(GPU; run once)*

Rebuilds the CD4 cohort exactly as nb18, loads the cached SemanticSCVI model, takes the
latent representation, and caches `z` + `obs`. Skips entirely once the cache exists, so every
figure below runs on CPU.

```python  # cell 3
# Skips if cache present. Set FORCE_REBUILD_Z=True to recompute.
FORCE_REBUILD_Z = False
if (not FORCE_REBUILD_Z) and Z_NPY.exists() and OBS_PARQUET.exists():
    print("z + obs cache present — skipping model rebuild:", Z_NPY.name, "|", OBS_PARQUET.name)
else:
    adata = sc.read_h5ad(V2)
    adata = M.load_malignancy_calls(adata, NB_DIR)        # CNV/transcriptomic/TCR calls -> bool obs

    # sample-selection rules (nb18) -> training cohort
    clone_summary = C.clone_summary_table(adata, 0.05, 1.33)
    meta = (adata.obs[["donor", "study", "entity", "disease"]].astype(str)
            .groupby("donor", observed=True)
            .agg(lambda s: ", ".join(sorted(s[s != "nan"].unique()))))
    clone_summary = clone_summary.merge(meta, on="donor", how="left")
    drop = {}
    if {"D5__MFIVB", "D1__P303"} <= set(clone_summary["donor"]):
        drop["D1__P303"] = "duplicate of D5__MFIVB"
    DROP_ENTITIES = {"MF_gamma_delta", "CD8_aggressive_epidermotropic_CTCL"}
    for d in clone_summary.loc[clone_summary["entity"].isin(DROP_ENTITIES), "donor"]:
        drop.setdefault(d, "non-ab-CD4 entity")
    small = (clone_summary["n_tcr_cells"] < 300) & (clone_summary["study"] != "herrera2021")
    for d in clone_summary.loc[small, "donor"]:
        drop.setdefault(d, "n_tcr_cells < 300")
    keep_donors = clone_summary.loc[~clone_summary["donor"].isin(drop), "donor"].tolist()
    adata = adata[adata.obs["donor"].isin(keep_donors)].copy()

    # CD4 non-Treg; drop ribosomal genes; raw counts for the NB likelihood
    adata_cd4 = adata[adata.obs["cell_type_T2"].astype(str) == "CD4"].copy()
    ribo = adata_cd4.var_names.str.upper().str.startswith(("RPS", "RPL"))
    adata_cd4 = adata_cd4[:, ~ribo].copy()
    adata_cd4.X = adata_cd4.layers["raw_counts"].copy()
    del adata; gc.collect()

    # gene_id (Ensembl) for the Geneformer map
    src = sc.read_h5ad(GENE_ID_SOURCE, backed="r")
    sym2ens = dict(zip(src.var["gene_name"].astype(str), src.var["gene_id"].astype(str)))
    adata_cd4.var["gene_id"] = [sym2ens.get(s, s) for s in adata_cd4.var_names.astype(str)]
    del src; gc.collect()

    # Geneformer map (cached) -> drop out-of-vocab -> HVG (identical to nb18)
    semantic_map = get_or_build_geneformer_map(adata_cd4, SEMANTIC_CACHE_GENEFORMER, var_id_key="gene_id")
    if semantic_map.shape[0] != adata_cd4.n_vars:
        SEMANTIC_CACHE_GENEFORMER.unlink()
        semantic_map = get_or_build_geneformer_map(adata_cd4, SEMANTIC_CACHE_GENEFORMER, var_id_key="gene_id")
    in_vocab = (semantic_map.norm(dim=1) > 0).cpu().numpy()
    adata_cd4 = adata_cd4[:, in_vocab].copy()
    semantic_map = semantic_map[torch.as_tensor(in_vocab)]
    sc.pp.highly_variable_genes(adata_cd4, n_top_genes=HVG_TOP_N, flavor=HVG_FLAVOR, subset=False)
    keep = adata_cd4.var["highly_variable"].values
    adata_cd4 = adata_cd4[:, keep].copy()
    semantic_map = semantic_map[torch.as_tensor(keep)]
    print("after HVG:", adata_cd4.shape)

    model = train_or_load_semantic_scvi(
        adata_cd4, semantic_map, cache_dir=SEMANTIC_GENEFORMER_CACHE_DIR, force_train=False,
        max_epochs=SEMANTIC_GENEFORMER_MAX_EPOCHS, warmup_epochs=SEMANTIC_GENEFORMER_WARMUP_EPOCHS,
        n_epochs_kl_warmup=SEMANTIC_GENEFORMER_N_EPOCHS_KL_WARMUP, batch_key=BATCH_KEY,
        **SEMANTIC_GENEFORMER_KWARGS)
    assert model.is_trained, "model failed to load/train"

    z = np.asarray(model.get_latent_representation())
    np.save(Z_NPY, z)
    model.adata.obs.to_parquet(OBS_PARQUET)
    print("saved z ->", Z_NPY.name, z.shape, "| obs ->", OBS_PARQUET.name, model.adata.obs.shape)
    del model, semantic_map, adata_cd4; gc.collect()
```

```
z + obs cache present — skipping model rebuild: semantic_geom_cd4_atlas_z.npy | semantic_geom_cd4_atlas_obs.parquet
```

```python  # cell 4
# assemble ad_emb from cache (z + obs + cached latent UMAP)
z = np.load(Z_NPY)
ad_emb = sc.AnnData(z, obs=pd.read_parquet(OBS_PARQUET))
for c in [c for c in M.MALIGNANCY_SOURCES if c in ad_emb.obs]:
    ad_emb.obs[c] = ad_emb.obs[c].astype(bool)

umap = np.load(EMB_UMAP_NPY)
assert umap.shape[0] == ad_emb.n_obs, f"UMAP {umap.shape} vs ad_emb {ad_emb.n_obs}"
ad_emb.obsm["X_umap"] = umap
print("ad_emb:", ad_emb.shape, "| z:", z.shape, "| UMAP:", umap.shape)
```

```
ad_emb: (169014, 10) | z: (169014, 10) | UMAP: (169014, 2)
```

## Figure c — latent UMAP

```python  # cell 6
color = ["study", "cell_type_T", "tcr_malignant_alice", "cnvT_cnvcluster", "mrvi_m2_pseudosample"]
F.plot_latent_umap(ad_emb, color, FIG_DIR / "latent_umap.svg", ncols=2);
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/final/latent_umap.svg

<Figure size 1664x1440 with 5 Axes>
```

## Figure d1 — per-factor separation heatmap (`tcr_malignant_alice`)

```python  # cell 8
F.plot_factor_sep_heatmap(z, ad_emb, "tcr_malignant_alice",
                          FIG_DIR / "factor_sep_tcr_malignant_alice.svg");
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/final/factor_sep_tcr_malignant_alice.svg

<Figure size 1100x600 with 2 Axes>
```

### Per-factor separation heatmaps of the MRVI M2 call (`mrvi_m2_pseudosample`)

```python  # cell 10
F.plot_factor_sep_heatmap(z, ad_emb, "mrvi_m2_pseudosample",
                          FIG_DIR / "factor_sep_mrvi_m2_pseudosample.svg");
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/final/factor_sep_mrvi_m2_pseudosample.svg

<Figure size 1100x600 with 2 Axes>
```

## Figure d2 — OOF F1 bar + ROC overlay

Factors vs the two published calls (CNV nb15, M2 nb17), evaluated out-of-fold against the TCR
ALICE truth. Light CPU. `res` should match `tables/factor_vs_cnv_m2_malignancy.csv`.

```python  # cell 12
res, oof_scores, y, SHOW_FACTORS, res_full = F.evaluate_factors(
    z, ad_emb, CNV_CD4_PARQUET, CALLS_CSV, show_factors=(4, 5, 7, 1))
print(res[["form", "f1", "auc"]])
F.plot_f1_bar(res, FIG_DIR / "factor_vs_cnv_m2_f1.svg")
F.plot_roc(oof_scores, y, res, SHOW_FACTORS, FIG_DIR / "factor_vs_cnv_m2_roc.svg");
```

```
evaluable anchor cells: 94,030 | malignant 68,605 (73.0%) | donors 32
                           form        f1       auc
Z_4                          gt  0.847564   0.71119
Z_5                          lt  0.839012  0.669883
Z_7                    band_out  0.843064   0.66833
Z_1                          lt  0.846938  0.626526
semantic_all_factors   logistic  0.928092  0.947719
cnv_cd4_cd8ref        published  0.699625  0.633396
mrvi_m2_pseudosample  published  0.933613   0.95632
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/final/factor_vs_cnv_m2_f1.svg
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/final/factor_vs_cnv_m2_roc.svg

<Figure size 500x300 with 1 Axes>
<Figure size 500x300 with 1 Axes>
```

## Figure b — per-cell track heatmap (prob + TCR status)

```python  # cell 14
calls = pd.read_csv(CALLS_CSV, index_col=0)
F.plot_cell_track_heatmap(calls, FIG_DIR / "cell_track_heatmap.svg", seed=0);
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/final/cell_track_heatmap.svg

<Figure size 1400x400 with 8 Axes>
```

## Figure a — 2 CNV heatmaps (`Li2024_atlas__CTCL1`, `H__SS1`)  *(GPU first run only)*

Loads cached per-cell CNV scores (no recompute) and computes the per-cell GMM call. The
heatmap `X_cnv` matrix is **not** a pre-existing cache, so the first run reruns inferCNV for the
2 donors and caches the result to `data/atlas_joint/cnv_heatmap_cache/`; later runs load it and
skip inferCNV (set `FORCE_CNV_HEATMAP=True` to invalidate). Saved as SVG via the
`fmt="svg"`/`dpi=300` path in `run_cnv_heatmaps`.

```python  # cell 16
HEATMAP_DONORS = ["Li2024_atlas__CTCL1", "H__SS1"]
CNV_LEIDEN_RES, CNV_N_JOBS, CNV_CHUNK = 0.5, 8, 2500
CNV_CACHE = NB_DIR / "data" / "atlas_joint" / f"skin_T_cnv_per_cell_res{CNV_LEIDEN_RES:g}.parquet"
CNV_HM_CACHE = NB_DIR / "data" / "atlas_joint" / "cnv_heatmap_cache"   # per-donor X_cnv cache
FORCE_CNV_HEATMAP = False   # True to re-run inferCNV (e.g. after changing window/order_by/cohort)

# Memory-lean load: building acnv over all ~240k query cells OOM-kills the kernel. We only need
# the 2 heatmap donors + the HC donors (HC = the diploid reference), so subset V2 backed first.
adata = sc.read_h5ad(V2, backed="r")
donor = adata.obs["donor"].astype(str)
hc_donors = sorted(donor[adata.obs["disease"].astype(str) == "HC"].unique())
keep = donor.isin(set(HEATMAP_DONORS) | set(hc_donors)).to_numpy()
adata = adata[keep].to_memory()
print("subset for heatmap:", adata.n_obs, "cells |", len(hc_donors), "HC donors +", HEATMAP_DONORS)

alice = adata.obs["tcr_malignant_alice"].astype(bool).to_numpy()
adata.obs["tcr_is_malignant"] = alice
adata.obs["tcr_is_dominant_clone"] = alice

acnv, SHARED_REF, CNV_DONORS, HC_DONORS = C.prepare_infercnv_inputs(
    adata, GTF, INTEGRATED_H5, 0, n_hc_ref=5000, n_healthy_ref=3000, use_external=True)
del adata; gc.collect()

# cached per-cell CNV scores (force=False -> load, no inferCNV recompute) + per-cell GMM call
C.run_per_donor_infercnv(acnv, SHARED_REF, CNV_DONORS, CNV_CACHE, leiden_res=CNV_LEIDEN_RES,
                         n_jobs=CNV_N_JOBS, chunk=CNV_CHUNK, force=False)
dip = C.diploid_mask(acnv)
call1, _ = C.call_per_donor("cnv_focal_score", dip, CNV_DONORS, acnv.obs, seed=0)
acnv.obs["cnv_malignant"] = call1   # strat1 per-cell GMM = the heatmap call_col

missing = [d for d in HEATMAP_DONORS if d not in set(acnv.obs["donor"].astype(str))]
assert not missing, f"donors absent from acnv: {missing}"
acnv_sub = acnv[acnv.obs["donor"].astype(str).isin(HEATMAP_DONORS)].copy()
# cnv_cache_dir caches each donor's X_cnv; later re-runs load it and skip the inferCNV recompute.
C.run_cnv_heatmaps(acnv_sub, SHARED_REF, call_col="cnv_malignant", fig_dir=FIG_DIR,
                   n_per_study=2, window=150, vlim_scale=3.0, cmap="RdBu_r",
                   order_by="cnv_leiden", n_jobs=CNV_N_JOBS, chunk=CNV_CHUNK,
                   fmt="svg", dpi=300, cnv_cache_dir=CNV_HM_CACHE, force_cnv=FORCE_CNV_HEATMAP)
```

```
subset for heatmap: 10954 cells | 3 HC donors + ['Li2024_atlas__CTCL1', 'H__SS1']
HC donors held out: 3 | CNV (query) donors: 2 | query cells: 9375
WARNING: adata.X seems to be already log-transformed.
WARNING: adata.X seems to be already log-transformed.
hc_atlas reference: 1559 cells
external healthy reference: 3000 cells

INFO:root:Extracted GTF attributes: ['gene_id', 'gene_version', 'gene_name', 'gene_source', 'gene_biotype', 'transcript_id', 'transcript_version', 'transcript_name', 'transcript_source', 'transcript_biotype', 'tag', 'transcript_support_level', 'exon_number', 'exon_id', 'exon_version', 'protein_id', 'protein_version', 'ccds_id']

WARNING: GTF file misses annotation for 301 genes in adata.
WARNING: Skipped 28 genes because of duplicate identifiers in GTF file.
common genes: 15777 | with genomic position: 15449 / 15777
query cells: 9375 | within-donor nonclonal: 1020 | shared ref: 4559 {'healthy': np.int64(3000), 'hc_atlas': np.int64(1559)}
loaded cached inferCNV (skin_T_cnv_per_cell_res0.5.parquet) for 2 donors
[H__SS1] gmm_xover          thr=0.0230  malignant=93/152  | TCR-F1opt thr=0.0220 f1=0.84 n=117
[Li2024_atlas__CTCL1] gmm_xover          thr=0.0381  malignant=7180/9223  | TCR-F1opt thr=0.0110 f1=0.92 n=8155
heatmap donors (2/study, by TCR burden): ['Li2024_atlas__CTCL1', 'H__SS1'] | call_col = cnv_malignant | ORDER_BY = cnv_leiden

100%|██████████| 6/6 [00:20<00:00,  3.43s/it]

[Li2024_atlas__CTCL1] n=13782  ref=['nonclonal', 'hc_atlas', 'healthy']
[Li2024_atlas__CTCL1] cached heatmap inputs -> skin_T_cnv_heatmap_Li2024_atlas__CTCL1.npz
[li2024/Li2024_atlas__CTCL1] color vlim=±0.3759

<Figure size 1000x500 with 6 Axes>
100%|██████████| 2/2 [00:07<00:00,  3.56s/it]

[H__SS1] n=4711  ref=['nonclonal', 'hc_atlas', 'healthy']
[H__SS1] cached heatmap inputs -> skin_T_cnv_heatmap_H__SS1.npz
[herrera2021/H__SS1] color vlim=±0.2916

<Figure size 1000x500 with 6 Axes>
```
