# old/14_skin_T_tcr_cnv_malignancy.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/14_skin_T_tcr_cnv_malignancy.ipynb` and is never run._

# Skin T cells — TCR clonality + inferCNV malignancy (combined atlas)

Per-cell malignancy on the combined skin-T object `skin_T_annotated.h5ad` (studies: li2024, chennareddy, brunner, gaydosik, herrera, rindler) from two orthogonal signals:

- **TCR clonality** — recompute the dominant-clone malignancy uniformly (top clone ≥5% of the donor's TCR cells **and** ≥2× the 2nd clone).
- **inferCNV** — per-donor copy-number score vs a same-lineage T-cell reference, called **four ways** (2×2 grid below).

Heavy helpers live in `skin_T_cnv_helpers.py` (imported as `C`); this notebook just orchestrates.

### The 4 inferCNV methods — `{base call} × {smoothing}`

| key | obs column | base call | smoothing |
|---|---|---|---|
| `strat1_cell_gmm` | `cnv_malignant` | per-cell GMM on the focal score | none |
| `strat2_cell_mrvi` | `cnv_malig_leiden_mrvi` | per-cell GMM | high-res MRVI-Leiden majority vote |
| `strat3_cnv_cluster` | `cnv_malig_cluster` | per-cnv-cluster GMM | none |
| `strat4_cnv_cluster_mrvi` | `cnv_malig_cluster_mrvi` | per-cnv-cluster GMM | high-res MRVI-Leiden majority vote |

> Cells marked **HEAVY** (`normalize/log`, per-donor inferCNV, MRVI Leiden, heatmaps) run on the GPU kernel.

```python  # cell 1
import sys, gc
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
import skin_T_cnv_helpers as C
importlib.reload(H); importlib.reload(C)

SEED = 0
np.random.seed(SEED)
sc.settings.verbosity = 1

V2            = NB_DIR / "data" / "atlas_joint" / "skin_T_tcr_malig_v2.h5ad"  # curated TCR-complete object
GTF           = NB_DIR / "data" / "cache" / "Homo_sapiens.GRCh38.110.chr.gtf.gz"
INTEGRATED_H5 = NB_DIR / "data" / "Integrated_CTCL_skincellatlas_final_portal_tags.h5ad"
UMAP_NPY      = NB_DIR / "data" / "atlas_joint" / "skin_T_umap_u.npy"
OUT_PARQUET   = NB_DIR / "data" / "atlas_joint" / "skin_T_malignancy.parquet"
FIG_DIR       = NB_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)
```

## Step 1 — load the curated TCR-complete skin-T object

Loads `skin_T_tcr_malig_v2.h5ad` (242,959 cells, 34 kept donors): the re-annotated T object whose
TCR malignancy was settled upstream. The ALICE tumor call `tcr_malignant_alice` is carried in as
`tcr_is_malignant` (and `tcr_is_dominant_clone`, so it also drives the benign/diploid reference);
`tcr_is_expanded` / `has_tcr` / `tcr_clone_*` / `cached_malignant` come straight from the file.

```python  # cell 3
adata = sc.read_h5ad(V2)                       # curated TCR-complete T object (242,959 cells)
# ALICE tumor call is the malignant truth; it also drives benign/reference exclusion downstream
alice = adata.obs["tcr_malignant_alice"].astype(bool).to_numpy()
adata.obs["tcr_is_malignant"]      = alice
adata.obs["tcr_is_dominant_clone"] = alice
# tcr_is_expanded / has_tcr / tcr_clone_size / tcr_clone_id / cached_malignant come from the file
print(adata.shape, "| alice malignant:", int(alice.sum()))
print(adata.obs.groupby("study", observed=True)["has_tcr"].agg(["size", "sum"]))
```

```
(242959, 49683) | alice malignant: 68620
                   size     sum
study                          
brunner2024       25778   12392
chennareddy2025   52315   36685
herrera2021         919     659
li2024           161813  114252
rindler2021_fi     2134    1371
```

## Step 2 — carried TCR malignancy (ALICE)

The malignant call is the upstream ALICE tumor annotation (`tcr_malignant_alice`), carried in as
`tcr_is_malignant`. Sanity-check its distribution by `cell_type_T2` (it is CD4-restricted) and by
study.

```python  # cell 5
# carried ALICE malignancy — distribution by cell_type_T2 and by study
print("tcr_is_malignant by cell_type_T2:\n",
      adata.obs.groupby("cell_type_T2", observed=True)["tcr_is_malignant"].agg(["size", "sum"]))
print("\ntcr_is_malignant by study:\n",
      adata.obs.groupby("study", observed=True)["tcr_is_malignant"].agg(["size", "sum"]))
```

```
tcr_is_malignant by cell_type_T2:
                 size    sum
cell_type_T2               
CD4           169014  68620
CD4_Treg       15667      0
CD8            37207      0
NK             21071      0

tcr_is_malignant by study:
                    size    sum
study                         
brunner2024       25778   5091
chennareddy2025   52315  20335
herrera2021         919    295
li2024           161813  41818
rindler2021_fi     2134   1081
```

## Step 3 — inferCNV input prep & reference  · HEAVY (GPU kernel)

The object is **all T cells** (`cell_type_broad == "T"`); the 49-type `cell_type` is sparse, so query / HC-reference are driven off `cell_type_T`. `C.prepare_infercnv_inputs` builds the query and a shared diploid reference, adds genomic positions from the GTF, and returns `acnv, SHARED_REF, CNV_DONORS, HC_DONORS`.

**Query** = every annotated T cell (`cell_type_T != "UNK"`) of the disease-bearing (non-HC) donors; HC donors are held out of the query.

**Reference** (the diploid baseline inferCNV subtracts) — three pooled groups, each used **per donor only if ≥20 cells** are present:

1. **`nonclonal`** — within-donor non-clonal T: `has_tcr` AND not in the donor's dominant TCR clone AND not cached `tumor_cell`. Per-donor internal baseline.
2. **`hc_atlas`** — atlas-internal healthy-control T cells (`disease=="HC"`, `cell_type_T != "UNK"`), sampled up to `N_HC_REF=5000`. Same platform → **preferred**.
3. **`healthy`** — external benign-T from the Integrated CTCL skin cell atlas (`sample_type ∈ {healthy_skin, AD, psoriasis}` AND `cell_type ∈ {Tc, Th, Treg, Tc17_Th17, Tc_IL13_IL22}`), up to `N_HEALTHY_REF=3000`. Cross-platform fallback.

```python  # cell 7
acnv, SHARED_REF, CNV_DONORS, HC_DONORS = C.prepare_infercnv_inputs(
    adata, GTF, INTEGRATED_H5, SEED, n_hc_ref=5000, n_healthy_ref=3000, use_external=True)
```

```
HC donors held out: 3 | CNV (query) donors: 31 | query cells: 240720
WARNING: adata.X seems to be already log-transformed.
WARNING: adata.X seems to be already log-transformed.
hc_atlas reference: 1559 cells
external healthy reference: 3000 cells

INFO:root:Extracted GTF attributes: ['gene_id', 'gene_version', 'gene_name', 'gene_source', 'gene_biotype', 'transcript_id', 'transcript_version', 'transcript_name', 'transcript_source', 'transcript_biotype', 'tag', 'transcript_support_level', 'exon_number', 'exon_id', 'exon_version', 'protein_id', 'protein_version', 'ccds_id']

WARNING: GTF file misses annotation for 301 genes in adata.
WARNING: Skipped 28 genes because of duplicate identifiers in GTF file.
common genes: 15777 | with genomic position: 15449 / 15777
query cells: 240720 | within-donor nonclonal: 92789 | shared ref: 4559 {'healthy': np.int64(3000), 'hc_atlas': np.int64(1559)}
```

## Step 4 — knobs (resolution & thresholds)

Everything the inferCNV methods are sensitive to, in one place. Change a knob and rerun from Step 5:
- **`CNV_LEIDEN_RES`** — per-cnv-cluster Leiden resolution (the unit methods 3/4 call on). The cache is tagged by it.
- **`CLUSTER_THR_SCALE`** — per-cnv-cluster call threshold (< 1 → more borderline clusters called malignant).
- **`MRVI_LEIDEN_RES` / `MRVI_CLUST_FRAC`** — high-res MRVI-Leiden smoothing shared by methods 2 & 4.

```python  # cell 9
# >>> KNOBS — change & rerun from Step 5 <<<
# per-cnv-cluster inferCNV build (Step 5):
WINDOW_SIZE       = 250    # window size — smooth 10x dropout
TOPK_FRAC         = 0.10   # focal fraction of windows per cell (CTCL CNVs are focal)
CNV_LEIDEN_RES    = 0.5    # CNV-cluster Leiden resolution (cache is tagged by this)
# per-cnv-cluster call threshold (methods 3 & 4):
CLUSTER_THR_SCALE = 0.8   # < 1 -> more borderline clusters called malignant
# high-res MRVI-Leiden smoothing (methods 2 & 4):
MRVI_LEIDEN_RES   = 2.0    # high resolution -> fine, clone-pure clusters
MRVI_CLUST_FRAC   = 0.6    # cluster malignant if >= this frac of its cells are called
# infercnvpy runtime:
CNV_N_JOBS = 8             # cap workers (one fork per chunk -> OOM if uncapped)
CNV_CHUNK  = 2500
FORCE_CNV  = True          # recompute on the v2 cell set (old cache is 401k); set False after 1st run
# resolution-tagged cache (changing CNV_LEIDEN_RES recomputes into its own file)
CNV_CACHE = NB_DIR / "data" / "atlas_joint" / f"skin_T_cnv_per_cell_res{CNV_LEIDEN_RES:g}.parquet"
```

## Step 5 — per-donor inferCNV  · HEAVY (GPU kernel, long-running)

Per-donor `cnv.tl.infercnv` vs the combined reference, then a focal per-cell score (mean of each cell's top-`TOPK_FRAC` strongest `|X_cnv|` windows) plus a per-cnv-cluster `cnv_score`. Cached to `CNV_CACHE`; set `FORCE_CNV=True` to recompute.

```python  # cell 11
C.run_per_donor_infercnv(acnv, SHARED_REF, CNV_DONORS, CNV_CACHE, window=WINDOW_SIZE,
                         topk_frac=TOPK_FRAC, leiden_res=CNV_LEIDEN_RES,
                         n_jobs=CNV_N_JOBS, chunk=CNV_CHUNK, force=FORCE_CNV)
```

```
[D1__P171] query=  8644  nonclonal= 1236  ref=['nonclonal', 'hc_atlas', 'healthy']

100%|██████████| 6/6 [00:16<00:00,  2.77s/it]

[D1__P204] query= 15420  nonclonal= 6331  ref=['nonclonal', 'hc_atlas', 'healthy']

100%|██████████| 8/8 [00:24<00:00,  3.12s/it]

[D1__P311] query=  6821  nonclonal= 2045  ref=['nonclonal', 'hc_atlas', 'healthy']

100%|██████████| 5/5 [00:14<00:00,  2.83s/it]

[D1__P312] query=  2728  nonclonal= 1499  ref=['nonclonal', 'hc_atlas', 'healthy']

100%|██████████| 3/3 [00:09<00:00,  3.07s/it]

[D1__P76] query= 10581  nonclonal= 2394  ref=['nonclonal', 'hc_atlas', 'healthy']

100%|██████████| 7/7 [00:19<00:00,  2.74s/it]

[D1__PGS] query=  6465  nonclonal= 1747  ref=['nonclonal', 'hc_atlas', 'healthy']

100%|██████████| 5/5 [00:14<00:00,  2.84s/it]

[D3__P105] query=  1241  nonclonal=  532  ref=['nonclonal', 'hc_atlas', 'healthy']

100%|██████████| 3/3 [00:07<00:00,  2.44s/it]
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/scipy/sparse/_index.py:210: SparseEfficiencyWarning: Changing the sparsity structure of a csr_matrix is expensive. lil and dok are more efficient.
  self._set_arrayXarray(i, j, x)

[D3__P192] query=  6828  nonclonal= 1476  ref=['nonclonal', 'hc_atlas', 'healthy']

100%|██████████| 5/5 [00:14<00:00,  2.93s/it]

[D3__P196] query=  1432  nonclonal=  342  ref=['nonclonal', 'hc_atlas', 'healthy']

100%|██████████| 3/3 [00:07<00:00,  2.53s/it]
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/scipy/sparse/_index.py:210: SparseEfficiencyWarning: Changing the sparsity structure of a csr_matrix is expensive. lil and dok are more efficient.
  self._set_arrayXarray(i, j, x)

[D3__P87] query= 10414  nonclonal=  237  ref=['nonclonal', 'hc_atlas', 'healthy']

100%|██████████| 6/6 [00:18<00:00,  3.07s/it]

[D3__P94] query=  5788  nonclonal= 4711  ref=['nonclonal', 'hc_atlas', 'healthy']

100%|██████████| 5/5 [00:12<00:00,  2.56s/it]

[D5__MFIVB] query=  2133  nonclonal=  289  ref=['nonclonal', 'hc_atlas', 'healthy']

100%|██████████| 3/3 [00:08<00:00,  2.86s/it]
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/scipy/sparse/_index.py:210: SparseEfficiencyWarning: Changing the sparsity structure of a csr_matrix is expensive. lil and dok are more efficient.
  self._set_arrayXarray(i, j, x)

[H__MFIV1] query=   105  nonclonal=   47  ref=['nonclonal', 'hc_atlas', 'healthy']

100%|██████████| 2/2 [00:06<00:00,  3.02s/it]
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/scipy/sparse/_index.py:210: SparseEfficiencyWarning: Changing the sparsity structure of a csr_matrix is expensive. lil and dok are more efficient.
  self._set_arrayXarray(i, j, x)

[H__SS1] query=   152  nonclonal=   55  ref=['nonclonal', 'hc_atlas', 'healthy']

100%|██████████| 2/2 [00:06<00:00,  3.00s/it]
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/scipy/sparse/_index.py:210: SparseEfficiencyWarning: Changing the sparsity structure of a csr_matrix is expensive. lil and dok are more efficient.
  self._set_arrayXarray(i, j, x)

[H__SS2] query=   217  nonclonal=  135  ref=['nonclonal', 'hc_atlas', 'healthy']

100%|██████████| 2/2 [00:06<00:00,  3.07s/it]

[H__SS3] query=    95  nonclonal=   54  ref=['nonclonal', 'hc_atlas', 'healthy']

100%|██████████| 2/2 [00:05<00:00,  2.95s/it]

[H__SS4] query=   323  nonclonal=   54  ref=['nonclonal', 'hc_atlas', 'healthy']

100%|██████████| 2/2 [00:06<00:00,  3.11s/it]

[Li2024_atlas__CTCL1] query=  9223  nonclonal=  965  ref=['nonclonal', 'hc_atlas', 'healthy']

100%|██████████| 6/6 [00:17<00:00,  2.99s/it]

[Li2024_atlas__CTCL2] query=  5341  nonclonal= 2836  ref=['nonclonal', 'hc_atlas', 'healthy']

100%|██████████| 4/4 [00:12<00:00,  3.22s/it]

[Li2024_atlas__CTCL3] query= 13272  nonclonal= 3000  ref=['nonclonal', 'hc_atlas', 'healthy']

100%|██████████| 8/8 [00:
... [2275 more chars truncated]
```

## Step 6 — four malignancy methods (2×2)

`{per-cell GMM, per-cnv-cluster GMM} × {no smoothing, high-res MRVI-Leiden vote}`. The two smoothed methods (2 & 4) reuse **one** MRVI-Leiden clustering. All four calls are propagated onto `adata.obs` and collected in `METHODS`.  · HEAVY (MRVI neighbors + Leiden → GPU kernel)

```python  # cell 13
# base calls (per-donor GMM crossover + diploid floor)
dip = C.diploid_mask(acnv)
call1, meth1 = C.call_per_donor("cnv_focal_score", dip, CNV_DONORS, acnv.obs, seed=SEED)
acnv.obs["cnv_malignant"]     = call1                 # method 1: per-cell
acnv.obs["cnv_call_method"]   = meth1
call3, _ = C.call_per_donor("cnv_score", dip, CNV_DONORS, acnv.obs, verbose=False,
                            seed=SEED, thr_scale=CLUSTER_THR_SCALE)
acnv.obs["cnv_malig_cluster"] = call3                 # method 3: per-cnv-cluster

# high-res MRVI-Leiden smoothing — one clustering shared by methods 2 & 4
C.compute_mrvi_leiden(acnv, MRVI_LEIDEN_RES, SEED)
acnv.obs["cnv_malig_leiden_mrvi"]  = C.vote_cluster(acnv, "cnv_malignant", MRVI_CLUST_FRAC)
acnv.obs["cnv_malig_cluster_mrvi"] = C.vote_cluster(acnv, "cnv_malig_cluster", MRVI_CLUST_FRAC)

METHODS = {"strat1_cell_gmm":         "cnv_malignant",
           "strat2_cell_mrvi":        "cnv_malig_leiden_mrvi",
           "strat3_cnv_cluster":      "cnv_malig_cluster",
           "strat4_cnv_cluster_mrvi": "cnv_malig_cluster_mrvi"}

# propagate per-cell scores + the 4 calls onto the full object (NaN / "" / False outside the query)
for col in ["cnv_score", "cnv_cell_score", "cnv_focal_score"]:
    adata.obs[col] = acnv.obs[col].reindex(adata.obs_names)
adata.obs["cnv_leiden"]      = acnv.obs["cnv_leiden"].reindex(adata.obs_names).fillna("")
adata.obs["cnv_call_method"] = acnv.obs["cnv_call_method"].reindex(adata.obs_names).fillna("")
for col in METHODS.values():
    adata.obs[col] = acnv.obs[col].reindex(adata.obs_names).fillna(False).astype(bool)

for name, col in METHODS.items():
    print(f"{name:24s} {col:22s} malignant={int(adata.obs[col].sum())}/{adata.n_obs}")
```

```
[D1__P171] gmm_xover          thr=0.0583  malignant=3970/8644  | TCR-F1opt thr=0.0080 f1=0.90 n=6788
[D1__P204] gmm_xover          thr=0.0517  malignant=8630/15420  | TCR-F1opt thr=0.0500 f1=0.29 n=7401
[D1__P311] gmm_xover          thr=0.0469  malignant=4115/6821  | TCR-F1opt thr=0.0165 f1=0.79 n=5821
[D1__P312] gmm_xover          thr=0.0450  malignant=1681/2728  | TCR-F1opt thr=0.0358 f1=0.57 n=2347
[D1__P76] gmm_xover          thr=0.0398  malignant=4559/10581  | TCR-F1opt thr=0.0020 f1=0.82 n=8203
[D1__PGS] gmm_xover          thr=0.0458  malignant=3938/6465  | TCR-F1opt thr=0.0195 f1=0.80 n=5014
[D3__P105] gmm_xover          thr=0.0590  malignant=690/1241  | TCR-F1opt thr=0.0618 f1=0.25 n=598
[D3__P192] gmm_xover          thr=0.0485  malignant=2827/6828  | TCR-F1opt thr=0.0035 f1=0.81 n=4837
[D3__P196] gmm_xover          thr=0.0516  malignant=739/1432  | TCR-F1opt thr=0.0130 f1=0.75 n=857
[D3__P87] gmm_xover          thr=0.0361  malignant=5904/10414  | TCR-F1opt thr=0.0116 f1=0.90 n=1384
[D3__P94] gmm_xover          thr=0.0313  malignant=3182/5788
[D5__MFIVB] gmm_xover          thr=0.0321  malignant=832/2133  | TCR-F1opt thr=0.0006 f1=0.88 n=1370
[H__MFIV1] gmm_xover          thr=0.0577  malignant=53/105  | TCR-F1opt thr=0.0422 f1=0.78 n=67
[H__SS1] gmm_xover          thr=0.0230  malignant=93/152  | TCR-F1opt thr=0.0220 f1=0.84 n=117
[H__SS2] gmm_xover          thr=0.0185  malignant=126/217  | TCR-F1opt thr=0.0399 f1=0.63 n=157
[H__SS3] gmm_xover          thr=0.0169  malignant=53/95  | TCR-F1opt thr=0.0189 f1=0.62 n=69
[H__SS4] gmm_xover          thr=0.0493  malignant=196/323  | TCR-F1opt thr=0.0289 f1=0.97 n=230
[Li2024_atlas__CTCL1] gmm_xover          thr=0.0381  malignant=7180/9223  | TCR-F1opt thr=0.0110 f1=0.92 n=8155
[Li2024_atlas__CTCL2] gmm_xover          thr=0.0621  malignant=2622/5341  | TCR-F1opt thr=0.0732 f1=0.86 n=4698
[Li2024_atlas__CTCL3] gmm_xover          thr=0.0332  malignant=8593/13272  | TCR-F1opt thr=0.0061 f1=0.82 n=11489
[Li2024_atlas__CTCL4] gmm_xover          thr=0.0603  malignant=7113/12613  | TCR-F1opt thr=0.0807 f1=0.36 n=9875
[Li2024_atlas__CTCL5] gmm_xover          thr=0.0426  malignant=10138/14785  | TCR-F1opt thr=0.0343 f1=0.71 n=12373
[Li2024_atlas__CTCL6] gmm_xover          thr=0.0459  malignant=10985/17835  | TCR-F1opt thr=0.0671 f1=0.55 n=13978
[Li2024_atlas__CTCL7] gmm_xover          thr=0.0547  malignant=30846/56200  | TCR-F1opt thr=0.0173 f1=0.17 n=32790
[Li2024_atlas__CTCL8] gmm_xover          thr=0.0426  malignant=8228/12125  | TCR-F1opt thr=0.0235 f1=0.77 n=8525
[Li2024_atlas__PT11] gmm_xover          thr=0.0705  malignant=1438/2409  | TCR-F1opt thr=0.0389 f1=0.81 n=1814
[Li2024_atlas__PT35] gmm_xover          thr=0.0585  malignant=3708/5076  | TCR-F1opt thr=0.0416 f1=0.89 n=3183
[Li2024_atlas__PT47] gmm_xover          thr=0.0658  malignant=401/570  | TCR-F1opt thr=0.0609 f1=0.81 n=385
[Li2024_atlas__PT50] gmm_xover          thr=0.0397  malignant=2278/3746  | TCR-F1opt thr=0.0519 f1=0.62 n=975
[Li2024_atlas__PT53] gmm_xover          thr=0.0511  malignant=1707/2558  | TCR-F1opt thr=0.0458 f1=0.74 n=2189
[Li2024_atlas__PT56] gmm_xover          thr=0.0535  malignant=3546/5580  | TCR-F1opt thr=0.0659 f1=0.64 n=3783
29 MRVI clusters @res=2.0
strat1_cell_gmm          cnv_malignant          malignant=140371/242959
strat2_cell_mrvi         cnv_malig_leiden_mrvi  malignant=94391/242959
strat3_cnv_cluster       cnv_malig_cluster      malignant=155115/242959
strat4_cnv_cluster_mrvi  cnv_malig_cluster_mrvi malignant=161188/242959

/tmp/ipykernel_3649704/3014991033.py:26: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  adata.obs[col] = acnv.obs[col].reindex(adata.obs_names).fillna(False).astype(bool)
/tmp/ipykernel_3649704/3014991033.py:26: FutureWarning: Downcasting object dtype 
... [1088 more chars truncated]
```

## Step 7 — compare the 4 methods vs TCR

Each inferCNV method benchmarked against the orthogonal TCR-clonality call (`tcr_is_malignant`): precision / recall / F1 / Jaccard on TCR+ query cells, then malignant fraction by study and by donor.

```python  # cell 15
import matplotlib.pyplot as plt

avail = adata.obs["has_tcr"].to_numpy() & adata.obs["cnv_focal_score"].notna().to_numpy()
agree = C.strategy_agreement(adata, METHODS, "tcr_is_malignant", avail)
print(f"agreement vs TCR on {int(avail.sum())} TCR+ query cells:")
print(agree.to_string())

rename = {"tcr_is_malignant": "TCR", "cnv_malignant": "S1 cell-GMM",
          "cnv_malig_leiden_mrvi": "S2 cell-MRVI", "cnv_malig_cluster": "S3 cnv-cluster",
          "cnv_malig_cluster_mrvi": "S4 cnv-cluster-MRVI"}
frac_cols = ["tcr_is_malignant", *METHODS.values()]
q = adata.obs[adata.obs["donor"].isin(CNV_DONORS)]
for by, fs in [("study", (6, 3)), ("donor", (12, 3))]:
    tbl = q.groupby(by, observed=True)[frac_cols].mean().rename(columns=rename)
    ax = tbl.plot.bar(figsize=fs)
    ax.set_ylabel("malignant fraction"); ax.set_xlabel("")
    ax.set_title(f"CNV methods vs TCR — malignant fraction by {by}")
    plt.tight_layout(); plt.savefig(FIG_DIR / f"skin_T_cnv_methods_frac_by_{by}.png", dpi=150); plt.show()
```

```
agreement vs TCR on 164183 TCR+ query cells:
                         n_called  precision  recall     f1  jaccard
strategy                                                            
strat1_cell_gmm             95337      0.479   0.665  0.557    0.386
strat2_cell_mrvi            68565      0.680   0.679  0.679    0.515
strat3_cnv_cluster         103738      0.500   0.756  0.602    0.430
strat4_cnv_cluster_mrvi    109008      0.606   0.964  0.744    0.593

<Figure size 600x300 with 1 Axes>
<Figure size 1200x300 with 1 Axes>
```

## Step 8 — UMAP: TCR + the 4 methods

One panel per call on the MRVI u-UMAP. The TCR panel uses `tcr_label` — the **same malignant-clone
scheme as nb16/17** (malignant = ALICE clone, benign = TCR⁺ non-malignant unexpanded, else
unlabeled) — so the malignant set and the benign reference match across notebooks.

```python  # cell 17
if UMAP_NPY.exists() and np.load(UMAP_NPY).shape[0] == adata.n_obs:
    adata.obsm["X_umap"] = np.load(UMAP_NPY)
else:                                          # stale / missing (cell set changed) -> recompute
    sc.pp.neighbors(adata, use_rep="X_mrvi_u", random_state=SEED)
    sc.tl.umap(adata, random_state=SEED)
    np.save(UMAP_NPY, adata.obsm["X_umap"])
    print("recomputed u-UMAP ->", UMAP_NPY.name)
# tcr_label = identical malignant-clone scheme as nb16/17: malignant = ALICE clone,
# benign = TCR+ non-malignant unexpanded, else unlabeled (so the reference set matches too).
has = adata.obs["has_tcr"].astype(bool).to_numpy()
mal = adata.obs["tcr_is_malignant"].astype(bool).to_numpy()
exp = adata.obs["tcr_is_expanded"].astype(bool).to_numpy()
adata.obs["tcr_label"] = pd.Categorical(
    np.where(mal, "malignant", np.where(has & ~mal & ~exp, "benign", "unlabeled")),
    categories=["malignant", "benign", "unlabeled"])
adata.uns["tcr_label_colors"] = ["#d62728", "#1f77b4", "#dddddd"]   # malignant / benign / unlabeled
for c in METHODS.values():
    adata.obs[c + "_lbl"] = (adata.obs[c].astype(bool)
                             .map({True: "malignant", False: "benign"}).astype("category"))
sc.pl.umap(adata, color=["tcr_label", *[c + "_lbl" for c in METHODS.values()]],
           ncols=2, size=2, wspace=0.3, save="_skin_T_cnv_methods.png")
```

```
... storing 'cnv_leiden' as categorical
... storing 'cnv_call_method' as categorical

WARNING: saving figure to file figures/umap_skin_T_cnv_methods.png

<Figure size 1664x1440 with 5 Axes>
```

## Step 9 — chromosome heatmaps for ONE chosen method  · HEAVY (GPU)

Pick a method in `HEATMAP_METHOD`; `C.run_cnv_heatmaps` re-runs inferCNV (keeping `X_cnv`) for the top TCR-burden donors per study and draws a chromosome heatmap with three per-cell strips: the **chosen method's call**, malignant-TCR-clone membership, and clone size.

`ORDER_BY` sets the row grouping: `"cnv_leiden"` (cached Step-5 cluster — the per-cnv-cluster unit, so a cluster-based strip is constant within each block) or `"recompute"` (fresh Leiden on this run's `X_cnv` — visually tighter blocks).

```python  # cell 19
# >>> pick ONE method to draw chromosome heatmaps for <<<
HEATMAP_METHOD  = "strat1_cell_gmm"   # any key of METHODS
ORDER_BY        = "cnv_leiden"           # "cnv_leiden" (strip uniform per block) | "recompute"
RUN_CNV_HEATMAP = True

if RUN_CNV_HEATMAP:
    C.run_cnv_heatmaps(acnv, SHARED_REF, call_col=METHODS[HEATMAP_METHOD], fig_dir=FIG_DIR,
                       n_per_study=2, window=150, vlim_scale=3.0, cmap="RdBu_r",
                       order_by=ORDER_BY, n_jobs=CNV_N_JOBS, chunk=CNV_CHUNK)
else:
    print("RUN_CNV_HEATMAP=False — skipping the heavy per-donor heatmap recompute")
```

```
heatmap donors (2/study, by TCR burden): ['Li2024_atlas__CTCL3', 'Li2024_atlas__CTCL1', 'D1__P76', 'D1__P171', 'D3__P192', 'D3__P87', 'D5__MFIVB', 'H__SS4', 'H__SS1'] | call_col = cnv_malignant | ORDER_BY = cnv_leiden

100%|██████████| 8/8 [00:23<00:00,  2.94s/it]

[li2024/Li2024_atlas__CTCL3] n=17831  ref=['nonclonal', 'hc_atlas', 'healthy']  color vlim=±0.3355

<Figure size 1000x500 with 6 Axes>
100%|██████████| 6/6 [00:17<00:00,  2.92s/it]

[li2024/Li2024_atlas__CTCL1] n=13782  ref=['nonclonal', 'hc_atlas', 'healthy']  color vlim=±0.3530

<Figure size 1000x500 with 6 Axes>
100%|██████████| 7/7 [00:19<00:00,  2.72s/it]

[chennareddy2025/D1__P76] n=15140  ref=['nonclonal', 'hc_atlas', 'healthy']  color vlim=±0.2931

<Figure size 1000x500 with 6 Axes>
100%|██████████| 6/6 [00:16<00:00,  2.79s/it]

[chennareddy2025/D1__P171] n=13203  ref=['nonclonal', 'hc_atlas', 'healthy']  color vlim=±0.3168

<Figure size 1000x500 with 6 Axes>
100%|██████████| 5/5 [00:14<00:00,  3.00s/it]

[brunner2024/D3__P192] n=11387  ref=['nonclonal', 'hc_atlas', 'healthy']  color vlim=±0.2955

<Figure size 1000x500 with 6 Axes>
100%|██████████| 6/6 [00:19<00:00,  3.19s/it]

[brunner2024/D3__P87] n=14973  ref=['nonclonal', 'hc_atlas', 'healthy']  color vlim=±0.2816

<Figure size 1000x500 with 6 Axes>
100%|██████████| 3/3 [00:08<00:00,  2.99s/it]

[rindler2021_fi/D5__MFIVB] n=6692  ref=['nonclonal', 'hc_atlas', 'healthy']  color vlim=±0.3009

<Figure size 1000x500 with 6 Axes>
100%|██████████| 2/2 [00:06<00:00,  3.15s/it]

[herrera2021/H__SS4] n=4882  ref=['nonclonal', 'hc_atlas', 'healthy']  color vlim=±0.3085

<Figure size 1000x500 with 6 Axes>
100%|██████████| 2/2 [00:05<00:00,  2.94s/it]

[herrera2021/H__SS1] n=4711  ref=['nonclonal', 'hc_atlas', 'healthy']  color vlim=±0.3088

<Figure size 1000x500 with 6 Axes>
```

## Step 10 — combine TCR + CNV & persist

Final CNV call = strategy 3 (`cnv_malig_cluster`). `combined_malignant = TCR ∨ CNV`; `malignant_evidence ∈ {both, tcr_only, cnv_only, none}`. Persist the per-cell annotations (incl. all four method columns) to `OUT_PARQUET`.

```python  # cell 21
# Final CNV call = strategy 3 (per-cnv-cluster, cnv_malig_cluster).
FINAL_CNV = "cnv_malig_cluster"
tcr_m = adata.obs["tcr_is_malignant"].to_numpy()
cnv_m = adata.obs[FINAL_CNV].to_numpy().astype(bool)
adata.obs["combined_malignant"] = tcr_m | cnv_m
adata.obs["malignant_evidence"] = np.select(
    [tcr_m & cnv_m, tcr_m & ~cnv_m, ~tcr_m & cnv_m],
    ["both", "tcr_only", "cnv_only"], default="none")
print("combined malignant:", int(adata.obs["combined_malignant"].sum()), "/", adata.n_obs)
print("\nevidence:\n", adata.obs["malignant_evidence"].value_counts())

both_avail = adata.obs["has_tcr"].to_numpy() & adata.obs["cnv_focal_score"].notna().to_numpy()
print("\nTCR x CNV crosstab (cells with both signals):")
print(pd.crosstab(adata.obs.loc[both_avail, "tcr_is_malignant"],
                  adata.obs.loc[both_avail, FINAL_CNV], rownames=["tcr"], colnames=["cnv"]))
```

```
combined malignant: 171881 / 242959

evidence:
 malignant_evidence
cnv_only    103261
none         71078
both         51854
tcr_only     16766
Name: count, dtype: int64

TCR x CNV crosstab (cells with both signals):
cnv    False  True 
tcr                
False  43695  51884
True   16750  51854
```

```python  # cell 22
keep = ["donor", "study", "disease", "cell_type", "cached_malignant",
        "has_tcr", "tcr_clone_id", "tcr_clone_size", "tcr_is_expanded",
        "tcr_is_dominant_clone", "tcr_is_malignant",
        "cnv_score", "cnv_cell_score", "cnv_focal_score", "cnv_leiden",
        "cnv_call_method", "cnv_malignant",
        "cnv_malig_leiden_mrvi", "cnv_malig_cluster", "cnv_malig_cluster_mrvi",
        "combined_malignant", "malignant_evidence"]
out = adata.obs[keep].copy()
out.index.name = "obs_name"
out.to_parquet(OUT_PARQUET)
print("wrote", OUT_PARQUET, out.shape)
out.head()
```

```
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/skin_T_malignancy.parquet (242959, 22)

                                                                 donor  \
obs_name                                                                 
Li2024_atlas|AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   
Li2024_atlas|AACTTTCTCTATGTGG-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   
Li2024_atlas|AAGCCGCAGAAACGAG-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   
Li2024_atlas|AAGGAGCAGACAGGCT-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   
Li2024_atlas|AATCCAGTCACGAAGG-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   

                                                    study disease   cell_type  \
obs_name                                                                        
Li2024_atlas|AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1  li2024      MF  tumor_cell   
Li2024_atlas|AACTTTCTCTATGTGG-0_CTCL1_CTCL1_CTCL1  li2024      MF  tumor_cell   
Li2024_atlas|AAGCCGCAGAAACGAG-0_CTCL1_CTCL1_CTCL1  li2024      MF  tumor_cell   
Li2024_atlas|AAGGAGCAGACAGGCT-0_CTCL1_CTCL1_CTCL1  li2024      MF  tumor_cell   
Li2024_atlas|AATCCAGTCACGAAGG-0_CTCL1_CTCL1_CTCL1  li2024      MF  tumor_cell   

                                                   cached_malignant  has_tcr  \
obs_name                                                                       
Li2024_atlas|AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1              True    False   
Li2024_atlas|AACTTTCTCTATGTGG-0_CTCL1_CTCL1_CTCL1              True    False   
Li2024_atlas|AAGCCGCAGAAACGAG-0_CTCL1_CTCL1_CTCL1              True    False   
Li2024_atlas|AAGGAGCAGACAGGCT-0_CTCL1_CTCL1_CTCL1              True    False   
Li2024_atlas|AATCCAGTCACGAAGG-0_CTCL1_CTCL1_CTCL1              True    False   

                                                  tcr_clone_id  \
obs_name                                                         
Li2024_atlas|AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1                
Li2024_atlas|AACTTTCTCTATGTGG-0_CTCL1_CTCL1_CTCL1                
Li2024_atlas|AAGCCGCAGAAACGAG-0_CTCL1_CTCL1_CTCL1                
Li2024_atlas|AAGGAGCAGACAGGCT-0_CTCL1_CTCL1_CTCL1                
Li2024_atlas|AATCCAGTCACGAAGG-0_CTCL1_CTCL1_CTCL1                

                                                   tcr_clone_size  \
obs_name                                                            
Li2024_atlas|AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1               0   
Li2024_atlas|AACTTTCTCTATGTGG-0_CTCL1_CTCL1_CTCL1               0   
Li2024_atlas|AAGCCGCAGAAACGAG-0_CTCL1_CTCL1_CTCL1               0   
Li2024_atlas|AAGGAGCAGACAGGCT-0_CTCL1_CTCL1_CTCL1               0   
Li2024_atlas|AATCCAGTCACGAAGG-0_CTCL1_CTCL1_CTCL1               0   

                                                   tcr_is_expanded  \
obs_name                                                             
Li2024_atlas|AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1            False   
Li2024_atlas|AACTTTCTCTATGTGG-0_CTCL1_CTCL1_CTCL1            False   
Li2024_atlas|AAGCCGCAGAAACGAG-0_CTCL1_CTCL1_CTCL1            False   
Li2024_atlas|AAGGAGCAGACAGGCT-0_CTCL1_CTCL1_CTCL1            False   
Li2024_atlas|AATCCAGTCACGAAGG-0_CTCL1_CTCL1_CTCL1            False   

                                                   tcr_is_dominant_clone  ...  \
obs_name                                                                  ...   
Li2024_atlas|AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1                  False  ...   
Li2024_atlas|AACTTTCTCTATGTGG-0_CTCL1_CTCL1_CTCL1                  False  ...   
Li2024_atlas|AAGCCGCAGAAACGAG-0_CTCL1_CTCL1_CTCL1                  False  ...   
Li2024_atlas|AAGGAGCAGACAGGCT-0_CTCL1_CTCL1_CTCL1                  False  ...   
Li2024_atlas|AATCCAGTCACGAAGG-0_CTCL1_CTCL1_CTCL1                  False  ...   

                                                   cnv_cell_score  \
obs_name                                                            
Li2024_atlas|AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1        0.002908   
Li2024_atlas|
... [4858 more chars truncated]
```

```python  # cell 23
# Cache the 4 per-method CNV malignancy calls (slim, source-prefixed) for cross-notebook comparison (nb18).
METHOD_COLS = {
    "cnv_malignant":          "cnvT_cell_gmm",
    "cnv_malig_leiden_mrvi":  "cnvT_cell_mrvi",
    "cnv_malig_cluster":      "cnvT_cnvcluster",
    "cnv_malig_cluster_mrvi": "cnvT_cnvcluster_mrvi",
}
meth = adata.obs[list(METHOD_COLS)].rename(columns=METHOD_COLS).astype(bool)
meth.index.name = "obs_name"
METHODS_PARQUET = NB_DIR / "data" / "atlas_joint" / "skin_T_malignancy_methods.parquet"
meth.to_parquet(METHODS_PARQUET)
print("wrote", METHODS_PARQUET, meth.shape, {c: int(meth[c].sum()) for c in meth.columns})
```

```
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/skin_T_malignancy_methods.parquet (242959, 4) {'cnvT_cell_gmm': 140371, 'cnvT_cell_mrvi': 94391, 'cnvT_cnvcluster': 155115, 'cnvT_cnvcluster_mrvi': 161188}
```

## Step 11 — TCR↔CNV quality metrics

For each method, restricted to TCR+ cells, pooled agreement with the TCR call:
**sensitivity** = of TCR-malignant cells, fraction called CNV-malignant; **specificity** =
of TCR-non-malignant cells, fraction called CNV-non-malignant (with raw counts).

```python  # cell 25
# Step 11 — TCR<->CNV quality metrics (TCR+ cells, per method)
avail = adata.obs["has_tcr"].to_numpy() & adata.obs["cnv_focal_score"].notna().to_numpy()
quality = C.tcr_cnv_quality(adata, METHODS, "tcr_is_malignant", avail)
print(f"TCR<->CNV quality on {int(avail.sum())} TCR+ cells:")
print(quality.to_string())
```

```
TCR<->CNV quality on 164183 TCR+ cells:
                         n_tcr_malig  n_tcr_benign     tp     tn  sensitivity  specificity
strategy                                                                                  
strat1_cell_gmm                68604         95579  45638  45880        0.665        0.480
strat2_cell_mrvi               68604         95579  46601  73615        0.679        0.770
strat3_cnv_cluster             68604         95579  51854  43695        0.756        0.457
strat4_cnv_cluster_mrvi        68604         95579  66101  52672        0.964        0.551
```
