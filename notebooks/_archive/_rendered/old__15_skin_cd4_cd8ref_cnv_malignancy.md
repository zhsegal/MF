# old/15_skin_cd4_cd8ref_cnv_malignancy.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/15_skin_cd4_cd8ref_cnv_malignancy.ipynb` and is never run._

# Skin CD4 malignancy via same-sample CD8 inferCNV reference

Benchmark variant of `14_skin_T_tcr_cnv_malignancy.ipynb`. CTCL/MF is overwhelmingly a **CD4** malignancy, so same-sample **CD8** T cells are a clean, lineage-matched benign diploid reference.

- **Exclude** samples whose malignant `lineage` is `CD8` or `gamma_delta`; keep `CD4` / `not_applicable` / `unresolved_provisional`.
- Per kept sample: inferCNV with **same-sample CD8 as reference**, **CD4 as query**; call malignancy on CD4.
- Sweep `(CNV_LEIDEN_RES × CLUSTER_THR_SCALE)` to maximize F1 vs the orthogonal TCR dominant-clone call — same machinery as nb14.

> Heavy cells (per-sample inferCNV) marked **HEAVY — run on GPU kernel**.

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
import skin_T_cnv_helpers as C            # shared with nb14
importlib.reload(H); importlib.reload(C)

SEED = 0
np.random.seed(SEED)
sc.settings.verbosity = 1

V2          = NB_DIR / "data" / "atlas_joint" / "skin_T_tcr_malig_v2.h5ad"  # curated TCR-complete object
GTF         = NB_DIR / "data" / "cache" / "Homo_sapiens.GRCh38.110.chr.gtf.gz"
UMAP_NPY    = NB_DIR / "data" / "atlas_joint" / "skin_T_umap_u.npy"         # shared MRVI u-UMAP (Step 9)
OUT_PARQUET = NB_DIR / "data" / "atlas_joint" / "skin_cd4_cd8ref_malignancy.parquet"
FIG_DIR     = NB_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)
```

## Step 1–2 — load the curated TCR-complete skin-T object

Loads `skin_T_tcr_malig_v2.h5ad` (242,959 cells, 34 kept donors). The upstream ALICE tumor call
`tcr_malignant_alice` is carried in as `tcr_is_malignant` (and `tcr_is_dominant_clone`, so it also
defines the benign/diploid floor in Step 6); `cell_type_T2` drives the CD4-query / CD8-reference
split in Step 4. `tcr_is_expanded` / `has_tcr` / `tcr_clone_*` / `cached_malignant` come from the file.

```python  # cell 3
adata = sc.read_h5ad(V2)                       # curated TCR-complete T object (242,959 cells)
# ALICE tumor call is the malignant truth; it also drives benign/reference exclusion downstream
alice = adata.obs["tcr_malignant_alice"].astype(bool).to_numpy()
adata.obs["tcr_is_malignant"]      = alice
adata.obs["tcr_is_dominant_clone"] = alice
# tcr_is_expanded / has_tcr / tcr_clone_size / tcr_clone_id / cached_malignant come from the file
print(adata.shape, "| alice malignant:", int(alice.sum()))
```

```
(242959, 49683) | alice malignant: 68620
```

## Step 3 — carried TCR malignancy (ALICE)

The malignant call is the upstream ALICE tumor annotation (`tcr_malignant_alice`), carried in as
`tcr_is_malignant`. Sanity-check its distribution by `cell_type_T2` (CD4-restricted) and by study.

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

## Step 4 — sample filter (drop CD8/γδ) + CD8-reference inferCNV input prep  · HEAVY (GPU kernel)

Keep samples with `lineage ∈ {CD4, not_applicable, unresolved_provisional}` (drop CD8 / gamma_delta MF, where CD8 is not a benign reference). Within kept samples split cells by `cell_type_T` prefix: **CD4_\*** = query, **CD8_\*** = reference (`UNK` dropped). Samples with `< MIN_CD8_REF` CD8 cells are dropped (no usable reference).

```python  # cell 7
import infercnvpy as cnv
import anndata as ad

KEPT_LINEAGES = ["CD4", "not_applicable", "unresolved_provisional"]
MIN_CD8_REF   = 20
STD_CHR       = [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]]

# CD4 query + same-sample CD8 reference, log-normalised, positioned, std-chr only (drops chrnan/chrM)
acnv, CNV_DONORS = C.prepare_cd4_cd8ref_inputs(
    adata, GTF, kept_lineages=KEPT_LINEAGES, min_cd8_ref=MIN_CD8_REF, std_chr=STD_CHR)
```

```
kept samples (lineage CD4/NA/unresolved): 34 / 34 | excluded (CD8/gamma_delta): 0
WARNING: adata.X seems to be already log-transformed.
usable samples (CD8 ref >= 20): 32 | dropped (too few CD8): 2
  dropped: ['H__HC1_Skin', 'H__SS3_Skin']

INFO:root:Extracted GTF attributes: ['gene_id', 'gene_version', 'gene_name', 'gene_source', 'gene_biotype', 'transcript_id', 'transcript_version', 'transcript_name', 'transcript_source', 'transcript_biotype', 'tag', 'transcript_support_level', 'exon_number', 'exon_id', 'exon_version', 'protein_id', 'protein_version', 'ccds_id']

WARNING: GTF file misses annotation for 26036 genes in adata.
WARNING: Skipped 72 genes because of duplicate identifiers in GTF file.
query (CD4) cells: 184600 | reference (CD8) cells: 37180 | genes on chr1-22/X/Y: 23563
```

## Step 5 — per-sample inferCNV with CD8 reference  · HEAVY (GPU kernel, long-running)

Per sample, `cnv.tl.infercnv` with `reference_cat=["cd8_ref"]`, then cluster + score the **CD4 query** cells only at every `RES_GRID` resolution (one neighbor graph, Leiden per resolution). `cnv_score` is the per-cluster mean of the per-cell `cnv_cell_score`, recomputed analytically so the Step-6f sweep needs no inferCNV recompute. Cached to `skin_cd4_cd8ref_cnv_per_cell_multires.parquet`.

```python  # cell 9
WINDOW_SIZE    = 250          # inferCNV smoothing window (fixed default)
FORCE_CNV      = True         # recompute on the v2 cell set (old cache is 401k); set False after 1st run
CNV_LEIDEN_RES = 1.0          # >>> KNOB 1: CNV-cluster Leiden resolution — change & rerun Step 5 <<<
CNV_N_JOBS     = 8            # cap workers (infercnvpy forks one worker per chunk -> OOM if uncapped)
CNV_CHUNK      = 2500
# 4-strategy knobs (Step 6): high-res MRVI-Leiden smoothing (strat2 & strat4)
MRVI_LEIDEN_RES = 2.0         # high resolution -> fine, clone-pure clusters
MRVI_CLUST_FRAC = 0.8         # MRVI cluster malignant if >= this frac of its cells are called
# resolution-tagged cache: changing CNV_LEIDEN_RES recomputes into its own file (old kept)
CNV_CACHE = NB_DIR / "data" / "atlas_joint" / f"skin_cd4_cd8ref_cnv_per_cell_res{CNV_LEIDEN_RES:g}.parquet"

C.run_per_sample_cd8ref_infercnv(acnv, CNV_DONORS, CNV_CACHE, window=WINDOW_SIZE,
                                 leiden_res=CNV_LEIDEN_RES, n_jobs=CNV_N_JOBS, chunk=CNV_CHUNK,
                                 std_chr=STD_CHR, force=FORCE_CNV)
```

```
[D1__P115] CD4 query=   533  CD8 ref=  172

100%|██████████| 1/1 [00:00<00:00,  1.04it/s]

[D1__P116] CD4 query=   258  CD8 ref=  331

100%|██████████| 1/1 [00:00<00:00,  1.24it/s]

[D1__P171] CD4 query=  6621  CD8 ref=  606

100%|██████████| 3/3 [00:09<00:00,  3.12s/it]

[D1__P204] CD4 query=  7490  CD8 ref= 5943

100%|██████████| 6/6 [00:17<00:00,  2.86s/it]

[D1__P311_thick] CD4 query=  5653  CD8 ref=  632

100%|██████████| 3/3 [00:08<00:00,  2.85s/it]

[D1__P312_thick] CD4 query=  2227  CD8 ref=  204

100%|██████████| 1/1 [00:03<00:00,  3.23s/it]

[D1__P76] CD4 query=  8666  CD8 ref=  559

100%|██████████| 4/4 [00:12<00:00,  3.11s/it]

[D1__P_GS] CD4 query=  4026  CD8 ref=  788

100%|██████████| 2/2 [00:06<00:00,  3.24s/it]

[D3__P105_Skin] CD4 query=   943  CD8 ref=   65

100%|██████████| 1/1 [00:01<00:00,  1.36s/it]

[D3__P192_Skin] CD4 query=  4659  CD8 ref=  787

100%|██████████| 3/3 [00:07<00:00,  2.36s/it]

[D3__P196_Skin] CD4 query=  1111  CD8 ref=   32

100%|██████████| 1/1 [00:01<00:00,  1.51s/it]

[D3__P87_Skin] CD4 query=  9252  CD8 ref=   36

100%|██████████| 4/4 [00:12<00:00,  3.05s/it]

[D3__P94_Skin] CD4 query=  4788  CD8 ref=  350

100%|██████████| 3/3 [00:06<00:00,  2.27s/it]

[D5__MFIVB_skin] CD4 query=  1465  CD8 ref=  289

100%|██████████| 1/1 [00:02<00:00,  2.39s/it]

[H__MFIV1_Skin] CD4 query=    55  CD8 ref=   31

100%|██████████| 1/1 [00:00<00:00,  4.24it/s]

[H__SS1_Skin] CD4 query=   116  CD8 ref=   21

100%|██████████| 1/1 [00:00<00:00,  3.42it/s]

[H__SS2_Skin] CD4 query=    99  CD8 ref=   75

100%|██████████| 1/1 [00:00<00:00,  3.02it/s]

[H__SS4_Skin] CD4 query=   245  CD8 ref=   41

100%|██████████| 1/1 [00:00<00:00,  2.15it/s]

[Li2024_atlas__CTCL1] CD4 query=  8351  CD8 ref=  685

100%|██████████| 4/4 [00:11<00:00,  2.94s/it]

[Li2024_atlas__CTCL2] CD4 query=  3850  CD8 ref= 1348

100%|██████████| 3/3 [00:06<00:00,  2.31s/it]

[Li2024_atlas__CTCL3] CD4 query= 11048  CD8 ref= 1869

100%|██████████| 6/6 [00:16<00:00,  2.83s/it]

[Li2024_atlas__CTCL4] CD4 query=  8296  CD8 ref= 3648

100%|██████████| 5/5 [00:15<00:00,  3.10s/it]

[Li2024_atlas__CTCL5] CD4 query= 13209  CD8 ref=  793

100%|██████████| 6/6 [00:18<00:00,  3.07s/it]

[Li2024_atlas__CTCL6] CD4 query= 15880  CD8 ref= 1194

100%|██████████| 7/7 [00:21<00:00,  3.10s/it]

[Li2024_atlas__CTCL7] CD4 query= 40762  CD8 ref=13158

100%|██████████| 22/22 [01:09<00:00,  3.16s/it]

[Li2024_atlas__CTCL8] CD4 query= 10311  CD8 ref=  938

100%|██████████| 5/5 [00:14<00:00,  2.86s/it]

[Li2024_atlas__PT11] CD4 query=  1624  CD8 ref=  458

100%|██████████| 1/1 [00:02<00:00,  2.65s/it]

[Li2024_atlas__PT35] CD4 query=  3602  CD8 ref=  423

100%|██████████| 2/2 [00:05<00:00,  2.55s/it]

[Li2024_atlas__PT47] CD4 query=   474  CD8 ref=   31

100%|██████████| 1/1 [00:00<00:00,  1.34it/s]

[Li2024_atlas__PT50] CD4 query=  2753  CD8 ref=  403

100%|██████████| 2/2 [00:04<00:00,  2.11s/it]

[Li2024_atlas__PT53] CD4 query=  2106  CD8 ref=  239

100%|██████████| 1/1 [00:03<00:00,  3.20s/it]

[Li2024_atlas__PT56] CD4 query=  4127  CD8 ref= 1031

100%|██████████| 3/3 [00:06<00:00,  2.23s/it]

computed inferCNV; cached -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/skin_cd4_cd8ref_cnv_per_cell_res1.parquet
```

## Step 6 — base per-cell call (strat1) + shared MRVI clustering  · HEAVY (GPU)

Computes the pieces the 4 strategies share (same machinery as nb14, via `skin_T_cnv_helpers`):
- **diploid floor** = benign CD4 query cells (non-dominant-clone, non-tumor).
- **strategy 1** (`cnv_malignant`) — per-cell 2-component GMM crossover on `log(cnv_cell_score)` with the diploid-median floor (`C.call_per_donor`). CD8 reference cells carry no score → excluded automatically.
- **MRVI-Leiden clustering** of the CD4 query on the `X_mrvi_u` latent — computed **once** here (heavy) and reused by strat2 / strat4 in Step 6b, so retuning the threshold stays cheap.

```python  # cell 11
# benign/diploid floor = benign CD4 query cells (non-dominant-clone, non-tumor). The CD8 cells
# define the inferCNV baseline upstream (Step 5), not this floor.
diploid = ((acnv.obs["cnv_ref"] == "query")
           & ~acnv.obs["tcr_is_dominant_clone"].to_numpy()
           & (acnv.obs["cell_type"].astype(str) != "tumor_cell"))

# strategy 1 — per-cell GMM crossover on the genome-wide cnv_cell_score (C.call_per_donor, reused from nb14)
call1, meth1 = C.call_per_donor("cnv_cell_score", diploid, CNV_DONORS, acnv.obs, seed=SEED)
acnv.obs["cnv_malignant"]   = call1
acnv.obs["cnv_call_method"] = meth1

# high-res MRVI-Leiden on the CD4 query only (CD8 ref excluded) — shared by strat2 & strat4.  HEAVY (GPU)
acnvq = acnv[acnv.obs["cnv_ref"] == "query"].copy()
C.compute_mrvi_leiden(acnvq, MRVI_LEIDEN_RES, SEED)
acnv.obs["mrvi_leiden"] = "ref"                                  # CD8 ref cells -> own group (never voted malignant)
acnv.obs.loc[acnvq.obs_names, "mrvi_leiden"] = acnvq.obs["mrvi_leiden"].astype(str).values
del acnvq; gc.collect()
```

```
[D1__P115] gmm_xover          thr=0.0077  malignant=162/705
[D1__P116] gmm_xover          thr=0.0056  malignant=142/589
[D1__P171] gmm_xover          thr=0.0086  malignant=1925/7227  | TCR-F1opt thr=0.0023 f1=0.98 n=5765
[D1__P204] gmm_xover          thr=0.0063  malignant=4074/13433  | TCR-F1opt thr=0.0022 f1=0.46 n=3592
[D1__P311_thick] gmm_xover          thr=0.0062  malignant=3265/6285  | TCR-F1opt thr=0.0033 f1=0.85 n=5097
[D1__P312_thick] gmm_xover          thr=0.0058  malignant=1515/2431  | TCR-F1opt thr=0.0060 f1=0.68 n=2011
[D1__P76] gmm_xover          thr=0.0073  malignant=2408/9225  | TCR-F1opt thr=0.0013 f1=0.88 n=7249
[D1__P_GS] gmm_xover          thr=0.0064  malignant=2335/4814  | TCR-F1opt thr=0.0026 f1=0.96 n=3551
[D3__P105_Skin] gmm_xover          thr=0.0069  malignant=525/1008  | TCR-F1opt thr=0.0085 f1=0.32 n=486
[D3__P192_Skin] gmm_xover          thr=0.0073  malignant=1565/5446  | TCR-F1opt thr=0.0018 f1=0.93 n=3819
[D3__P196_Skin] gmm_xover          thr=0.0066  malignant=580/1143  | TCR-F1opt thr=0.0033 f1=0.81 n=745
[D3__P87_Skin] gmm_xover          thr=0.0054  malignant=5274/9288  | TCR-F1opt thr=0.0029 f1=0.93 n=1311
[D3__P94_Skin] gmm_xover          thr=0.0068  malignant=1349/5138
[D5__MFIVB_skin] gmm_xover          thr=0.0087  malignant=338/1754  | TCR-F1opt thr=0.0031 f1=0.97 n=1145
[H__MFIV1_Skin] gmm_xover          thr=0.0083  malignant=33/86
[H__SS1_Skin] gmm_xover          thr=0.0054  malignant=76/137  | TCR-F1opt thr=0.0053 f1=0.91 n=96
[H__SS2_Skin] gmm_xover          thr=0.0057  malignant=42/174  | TCR-F1opt thr=0.0072 f1=0.83 n=88
[H__SS4_Skin] gmm_xover          thr=0.0059  malignant=173/286  | TCR-F1opt thr=0.0029 f1=0.99 n=199
[Li2024_atlas__CTCL1] gmm_xover          thr=0.0070  malignant=5921/9036  | TCR-F1opt thr=0.0038 f1=0.97 n=7438
[Li2024_atlas__CTCL2] gmm_xover          thr=0.0080  malignant=2015/5198  | TCR-F1opt thr=0.0078 f1=0.90 n=3445
[Li2024_atlas__CTCL3] gmm_xover          thr=0.0065  malignant=8339/12917  | TCR-F1opt thr=0.0046 f1=0.91 n=9731
[Li2024_atlas__CTCL4] gmm_xover          thr=0.0072  malignant=4080/11944  | TCR-F1opt thr=0.0102 f1=0.37 n=6590
[Li2024_atlas__CTCL5] gmm_xover          thr=0.0065  malignant=9208/14002  | TCR-F1opt thr=0.0059 f1=0.73 n=11447
[Li2024_atlas__CTCL6] gmm_xover          thr=0.0071  malignant=9130/17074  | TCR-F1opt thr=0.0088 f1=0.58 n=12782
[Li2024_atlas__CTCL7] gmm_xover          thr=0.0070  malignant=22887/53920  | TCR-F1opt thr=0.0076 f1=0.26 n=24212
[Li2024_atlas__CTCL8] gmm_xover          thr=0.0066  malignant=6167/11249  | TCR-F1opt thr=0.0047 f1=0.83 n=7503
[Li2024_atlas__PT11] gmm_xover          thr=0.0121  malignant=159/2082  | TCR-F1opt thr=0.0020 f1=0.92 n=1312
[Li2024_atlas__PT35] gmm_xover          thr=0.0074  malignant=2065/4025  | TCR-F1opt thr=0.0035 f1=0.98 n=2556
[Li2024_atlas__PT47] gmm_xover          thr=0.0075  malignant=271/505  | TCR-F1opt thr=0.0071 f1=0.86 n=334
[Li2024_atlas__PT50] gmm_xover          thr=0.0069  malignant=1506/3156  | TCR-F1opt thr=0.0069 f1=0.72 n=830
[Li2024_atlas__PT53] gmm_xover          thr=0.0065  malignant=1367/2345  | TCR-F1opt thr=0.0054 f1=0.81 n=1890
[Li2024_atlas__PT56] gmm_xover          thr=0.0067  malignant=2251/5158  | TCR-F1opt thr=0.0072 f1=0.69 n=2966
30 MRVI clusters @res=2.0

754
```

```python  # cell 12
# propagate CD4 CNV scores + cluster labels onto the full object obs (for the benchmark / Step 8)
adata.obs["cnv_ref"]        = acnv.obs["cnv_ref"].reindex(adata.obs_names).fillna("")
adata.obs["cnv_leiden"]     = acnv.obs["cnv_leiden"].reindex(adata.obs_names).fillna("")
adata.obs["cnv_cell_score"] = acnv.obs["cnv_cell_score"].reindex(adata.obs_names)
adata.obs["cnv_score"]      = acnv.obs["cnv_score"].reindex(adata.obs_names)
```

## Step 6b — assemble the 4 CNV strategies + threshold knob

Builds the same 4 methods as nb14 and exposes the cheap threshold knob:
- **strat1 cell-GMM** (`cnv_malignant`) — per-cell GMM on `cnv_cell_score` (Step 6).
- **strat2 cell-MRVI** (`cnv_malig_leiden_mrvi`) — strat1 majority-voted over MRVI-Leiden clusters.
- **strat3 cnv-cluster** (`cnv_malig_cluster`) — per-cnv-cluster GMM on `cnv_score`.
- **strat4 cnv-cluster-MRVI** (`cnv_malig_cluster_mrvi`) — strat3 majority-voted over MRVI clusters.

`CLUSTER_THR_SCALE` (`< 1` → more cells malignant) is cheap to retune — rerun this cell only (the heavy MRVI clustering from Step 6 is reused).

```python  # cell 14
# >>> KNOB 2: per-sample threshold scale (smaller -> more cells malignant). Rerun this cell to change. <<<
CLUSTER_THR_SCALE = 0.85

# strategy 3 — per-cnv-cluster GMM on cnv_score
call3, _ = C.call_per_donor("cnv_score", diploid, CNV_DONORS, acnv.obs,
                            verbose=False, seed=SEED, thr_scale=CLUSTER_THR_SCALE)
acnv.obs["cnv_malig_cluster"] = call3

# strategies 2 & 4 — MRVI-cluster majority vote of strat1 / strat3 (cheap; reuses cached mrvi_leiden)
acnv.obs["cnv_malig_leiden_mrvi"]  = C.vote_cluster(acnv, "cnv_malignant", MRVI_CLUST_FRAC)
acnv.obs["cnv_malig_cluster_mrvi"] = C.vote_cluster(acnv, "cnv_malig_cluster", MRVI_CLUST_FRAC)

METHODS = {"strat1_cell_gmm":         "cnv_malignant",
           "strat2_cell_mrvi":        "cnv_malig_leiden_mrvi",
           "strat3_cnv_cluster":      "cnv_malig_cluster",
           "strat4_cnv_cluster_mrvi": "cnv_malig_cluster_mrvi"}

# propagate the call method + 4 calls onto the full object (False outside the CD4 query)
adata.obs["cnv_call_method"] = acnv.obs["cnv_call_method"].reindex(adata.obs_names).fillna("")
for col in METHODS.values():
    adata.obs[col] = acnv.obs[col].reindex(adata.obs_names).fillna(False).astype(bool)

n_q = int((adata.obs["cnv_ref"] == "query").sum())
print(f"strategy calls (res={CNV_LEIDEN_RES:g}, thr_scale={CLUSTER_THR_SCALE}) on {n_q} CD4 query cells:")
for name, col in METHODS.items():
    print(f"  {name:24s} {col:24s} malignant={int(adata.obs[col].sum())}/{n_q}")
```

```
strategy calls (res=1, thr_scale=0.85) on 184600 CD4 query cells:
  strat1_cell_gmm          cnv_malignant            malignant=101147/184600
  strat2_cell_mrvi         cnv_malig_leiden_mrvi    malignant=10729/184600
  strat3_cnv_cluster       cnv_malig_cluster        malignant=119099/184600
  strat4_cnv_cluster_mrvi  cnv_malig_cluster_mrvi   malignant=10555/184600

/tmp/ipykernel_2269120/2490500788.py:21: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  adata.obs[col] = acnv.obs[col].reindex(adata.obs_names).fillna(False).astype(bool)
/tmp/ipykernel_2269120/2490500788.py:21: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  adata.obs[col] = acnv.obs[col].reindex(adata.obs_names).fillna(False).astype(bool)
/tmp/ipykernel_2269120/2490500788.py:21: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  adata.obs[col] = acnv.obs[col].reindex(adata.obs_names).fillna(False).astype(bool)
/tmp/ipykernel_2269120/2490500788.py:21: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  adata.obs[col] = acnv.obs[col].reindex(adata.obs_names).fillna(False).astype(bool)
```

## Step 6d — benchmark the 4 CD4 CNV strategies vs TCR

Precision / recall / F1 / Jaccard of each CNV strategy (S1 cell-GMM, S2 cell-MRVI, S3 cnv-cluster, S4 cnv-cluster-MRVI) against the orthogonal TCR dominant-clone call, on TCR+ CD4 cells — same `C.strategy_agreement` machinery as nb14.

```python  # cell 16
import matplotlib.pyplot as plt

# agreement of each CNV strategy with the orthogonal TCR dominant-clone call on TCR+ CD4 cells
avail = adata.obs["has_tcr"].to_numpy() & (adata.obs["cnv_ref"].to_numpy() == "query")
agree = C.strategy_agreement(adata, METHODS, "tcr_is_malignant", avail)
print(f"agreement vs TCR on {int(avail.sum())} TCR+ CD4 cells:")
print(agree.to_string())

print("\nstrat3_cnv_cluster × TCR:\n",
      pd.crosstab(adata.obs.loc[avail, "tcr_is_malignant"],
                  adata.obs.loc[avail, "cnv_malig_cluster"], rownames=["tcr"], colnames=["cnv_cluster"]))

# malignant fraction by study / sample: TCR vs the 4 methods
rename = {"tcr_is_malignant": "TCR", "cnv_malignant": "S1 cell-GMM",
          "cnv_malig_leiden_mrvi": "S2 cell-MRVI", "cnv_malig_cluster": "S3 cnv-cluster",
          "cnv_malig_cluster_mrvi": "S4 cnv-cluster-MRVI"}
frac_cols = ["tcr_is_malignant", *METHODS.values()]
q = adata.obs[adata.obs["sample_id"].isin(CNV_DONORS) & (adata.obs["cnv_ref"] == "query")]
for by, fs in [("study", (6, 3)), ("sample_id", (12, 3))]:
    tbl = q.groupby(by, observed=True)[frac_cols].mean().rename(columns=rename)
    ax = tbl.plot.bar(figsize=fs)
    ax.set_ylabel("malignant fraction"); ax.set_xlabel("")
    ax.set_title(f"CD4 CNV methods vs TCR — malignant fraction by {by}")
    plt.tight_layout(); plt.savefig(FIG_DIR / f"skin_cd4_cd8ref_frac_by_{by}.png", dpi=150); plt.show()
```

```
agreement vs TCR on 132883 TCR+ CD4 cells:
                         n_called  precision  recall     f1  jaccard
strategy                                                            
strat1_cell_gmm             71945      0.583   0.612  0.597    0.426
strat2_cell_mrvi             5544      0.707   0.057  0.106    0.056
strat3_cnv_cluster          79609      0.549   0.637  0.589    0.418
strat4_cnv_cluster_mrvi      8945      0.880   0.115  0.203    0.113

strat3_cnv_cluster × TCR:
 cnv_cluster  False  True 
tcr                      
False        28348  35930
True         24926  43679

<Figure size 720x360 with 1 Axes>
<Figure size 1440x360 with 1 Axes>
```

## Step 6e — inferCNV chromosome heatmaps (annotated) for representative samples  · HEAVY (GPU)

`C.run_cnv_heatmaps` (shared with nb14, `shared_ref=None` since the same-sample CD8 reference already lives in `acnv`) re-runs inferCNV for a few donors keeping `X_cnv`, then draws a chromosome heatmap with three per-cell color strips on the right: the chosen strategy's call (`HEATMAP_METHOD`), **malignant TCR clone** membership, and **TCR clone size** (log).

`ORDER_BY` sets the row grouping — two different clusterings:
- `"cnv_leiden"` (default) — the cached Step-5 cluster the per-cnv-cluster strategy calls on, so the call strip is **constant within each block**.
- `"recompute"` — a fresh Leiden on this run's (`window=150`) `X_cnv`: visually tighter blocks, but the cached-clustering call strip will be mixed within them.

Color knobs: `HM_VLIM_SCALE` (higher → paler), `HM_CMAP`, `HM_WINDOW`.

```python  # cell 18
# inferCNV chromosome heatmaps for representative samples (N_PER_STUDY per study, top TCR burden),
# annotated with the chosen strategy's call + malignant-TCR-clone membership + clone size.
# Reuses C.run_cnv_heatmaps with shared_ref=None (same-sample CD8 reference is already inside acnv)
# and reference_cat=("cd8_ref",).  HEAVY (GPU kernel).
#
# ORDER_BY sets the row grouping (two DIFFERENT clusterings):
#   "cnv_leiden" (default) — cached Step-5 cnv_leiden, the unit strategy 3 calls on, so the
#                            chosen-call strip is CONSTANT within every block.
#   "recompute"            — fresh leiden on THIS run's X_cnv: tighter blocks, but the strat strip
#                            (made on the cached clustering) will be mixed within them.
RUN_CNV_HEATMAP = True
HEATMAP_METHOD  = "strat3_cnv_cluster"   # any key of METHODS
ORDER_BY        = "cnv_leiden"           # "cnv_leiden" (strip uniform per block) | "recompute"
HM_WINDOW       = 150                    # smooth out per-cell speckle so blocks read as blocks
HM_VLIM_SCALE   = 3.0                    # widen color range: diploid -> near-white, only real CNV shows color
HM_CMAP         = "RdBu_r"
N_PER_STUDY     = 3

if RUN_CNV_HEATMAP:
    C.run_cnv_heatmaps(acnv, None, call_col=METHODS[HEATMAP_METHOD], fig_dir=FIG_DIR,
                       reference_cat=("cd8_ref",), fig_prefix="skin_cd4_cd8ref_cnv_heatmap",
                       n_per_study=N_PER_STUDY, window=HM_WINDOW, vlim_scale=HM_VLIM_SCALE,
                       cmap=HM_CMAP, order_by=ORDER_BY, n_jobs=CNV_N_JOBS, chunk=CNV_CHUNK)
else:
    print("RUN_CNV_HEATMAP=False — skipping the heavy per-donor heatmap recompute")
```

```
heatmap donors (3/study, by TCR burden): ['Li2024_atlas__CTCL3', 'Li2024_atlas__CTCL1', 'Li2024_atlas__CTCL5', 'D1__P76', 'D1__P171', 'D1__P311_thick', 'D3__P192_Skin', 'D3__P87_Skin', 'D5__MFIVB_skin', 'D3__P196_Skin', 'H__SS4_Skin', 'H__SS1_Skin', 'H__SS2_Skin'] | call_col = cnv_malig_cluster | ORDER_BY = cnv_leiden

100%|██████████| 6/6 [00:16<00:00,  2.80s/it]

[li2024/Li2024_atlas__CTCL3] n=12917  ref=['cd8_ref']  color vlim=±0.3166

<Figure size 1200x600 with 6 Axes>
100%|██████████| 4/4 [00:11<00:00,  2.90s/it]

[li2024/Li2024_atlas__CTCL1] n=9036  ref=['cd8_ref']  color vlim=±0.3402

<Figure size 1200x600 with 6 Axes>
100%|██████████| 6/6 [00:17<00:00,  3.00s/it]

[li2024/Li2024_atlas__CTCL5] n=14002  ref=['cd8_ref']  color vlim=±0.3255

<Figure size 1200x600 with 6 Axes>
100%|██████████| 4/4 [00:13<00:00,  3.43s/it]

[chennareddy2025/D1__P76] n=9225  ref=['cd8_ref']  color vlim=±0.2757

<Figure size 1200x600 with 6 Axes>
100%|██████████| 3/3 [00:09<00:00,  3.11s/it]

[chennareddy2025/D1__P171] n=7227  ref=['cd8_ref']  color vlim=±0.3103

<Figure size 1200x600 with 6 Axes>
100%|██████████| 3/3 [00:08<00:00,  2.74s/it]

[chennareddy2025/D1__P311_thick] n=6285  ref=['cd8_ref']  color vlim=±0.2850

<Figure size 1200x600 with 6 Axes>
100%|██████████| 3/3 [00:07<00:00,  2.36s/it]

[brunner2024/D3__P192_Skin] n=5446  ref=['cd8_ref']  color vlim=±0.2804

<Figure size 1200x600 with 6 Axes>
100%|██████████| 4/4 [00:11<00:00,  2.88s/it]

[brunner2024/D3__P87_Skin] n=9288  ref=['cd8_ref']  color vlim=±0.2668

<Figure size 1200x600 with 6 Axes>
100%|██████████| 1/1 [00:02<00:00,  2.25s/it]

[rindler2021_fi/D5__MFIVB_skin] n=1754  ref=['cd8_ref']  color vlim=±0.2833

<Figure size 1200x600 with 6 Axes>
100%|██████████| 1/1 [00:01<00:00,  1.49s/it]

[brunner2024/D3__P196_Skin] n=1143  ref=['cd8_ref']  color vlim=±0.3072

<Figure size 1200x600 with 6 Axes>
100%|██████████| 1/1 [00:00<00:00,  2.20it/s]

[herrera2021/H__SS4_Skin] n=286  ref=['cd8_ref']  color vlim=±0.2820

<Figure size 1200x600 with 6 Axes>
100%|██████████| 1/1 [00:00<00:00,  3.63it/s]

[herrera2021/H__SS1_Skin] n=137  ref=['cd8_ref']  color vlim=±0.2753

<Figure size 1200x600 with 6 Axes>
100%|██████████| 1/1 [00:00<00:00,  3.17it/s]

[herrera2021/H__SS2_Skin] n=174  ref=['cd8_ref']  color vlim=±0.2559

<Figure size 1200x600 with 6 Axes>
```

## Step 7 — combine TCR + CD4 CNV call & report

```python  # cell 20
# Final CD4 CNV call = per-cnv-cluster strategy 3. Combine with TCR.
tcr_m = adata.obs["tcr_is_malignant"].to_numpy()
cnv_m = adata.obs["cnv_malig_cluster"].to_numpy().astype(bool)
adata.obs["combined_malignant"] = tcr_m | cnv_m
adata.obs["malignant_evidence"] = np.select(
    [tcr_m & cnv_m, tcr_m & ~cnv_m, ~tcr_m & cnv_m],
    ["both", "tcr_only", "cnv_only"], default="none")
print("combined malignant:", int(adata.obs["combined_malignant"].sum()), "/", adata.n_obs)
print("\nevidence:\n", adata.obs["malignant_evidence"].value_counts())

both_avail = adata.obs["has_tcr"].to_numpy() & (adata.obs["cnv_ref"].to_numpy() == "query")
print("\nTCR x CNV crosstab (TCR+ CD4 cells):")
print(pd.crosstab(adata.obs.loc[both_avail, "tcr_is_malignant"],
                  adata.obs.loc[both_avail, "cnv_malig_cluster"],
                  rownames=["tcr"], colnames=["cnv_cluster"]))
```

```
combined malignant: 144040 / 242959

evidence:
 malignant_evidence
none        98919
cnv_only    75420
both        43679
tcr_only    24941
Name: count, dtype: int64

TCR x CNV crosstab (TCR+ CD4 cells):
cnv_cluster  False  True 
tcr                      
False        28348  35930
True         24926  43679
```

## Step 8 — persist CD4 obs annotations

```python  # cell 22
keep = ["donor", "sample_id", "study", "disease", "lineage", "cell_type", "cell_type_T",
        "cnv_ref", "cached_malignant", "has_tcr", "tcr_clone_id", "tcr_clone_size",
        "tcr_is_expanded", "tcr_is_dominant_clone", "tcr_is_malignant",
        "cnv_score", "cnv_cell_score", "cnv_leiden", "cnv_call_method",
        "cnv_malignant", "cnv_malig_leiden_mrvi", "cnv_malig_cluster", "cnv_malig_cluster_mrvi",
        "combined_malignant", "malignant_evidence"]
out = adata.obs.loc[adata.obs["cnv_ref"] == "query", [c for c in keep if c in adata.obs.columns]].copy()
out.index.name = "obs_name"
out.to_parquet(OUT_PARQUET)
print("wrote", OUT_PARQUET, out.shape)
out.head()
```

```
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/skin_cd4_cd8ref_malignancy.parquet (184600, 25)

                                                                 donor  \
obs_name                                                                 
Li2024_atlas|AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   
Li2024_atlas|AAGCCGCAGAAACGAG-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   
Li2024_atlas|AAGGAGCAGACAGGCT-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   
Li2024_atlas|AATCCAGTCACGAAGG-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   
Li2024_atlas|ACACCGGCAAGCGTAG-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   

                                                             sample_id  \
obs_name                                                                 
Li2024_atlas|AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   
Li2024_atlas|AAGCCGCAGAAACGAG-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   
Li2024_atlas|AAGGAGCAGACAGGCT-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   
Li2024_atlas|AATCCAGTCACGAAGG-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   
Li2024_atlas|ACACCGGCAAGCGTAG-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   

                                                    study disease lineage  \
obs_name                                                                    
Li2024_atlas|AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1  li2024      MF     CD4   
Li2024_atlas|AAGCCGCAGAAACGAG-0_CTCL1_CTCL1_CTCL1  li2024      MF     CD4   
Li2024_atlas|AAGGAGCAGACAGGCT-0_CTCL1_CTCL1_CTCL1  li2024      MF     CD4   
Li2024_atlas|AATCCAGTCACGAAGG-0_CTCL1_CTCL1_CTCL1  li2024      MF     CD4   
Li2024_atlas|ACACCGGCAAGCGTAG-0_CTCL1_CTCL1_CTCL1  li2024      MF     CD4   

                                                    cell_type cell_type_T  \
obs_name                                                                    
Li2024_atlas|AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1  tumor_cell     CD4_mem   
Li2024_atlas|AAGCCGCAGAAACGAG-0_CTCL1_CTCL1_CTCL1  tumor_cell     CD4_mem   
Li2024_atlas|AAGGAGCAGACAGGCT-0_CTCL1_CTCL1_CTCL1  tumor_cell     CD4_Th2   
Li2024_atlas|AATCCAGTCACGAAGG-0_CTCL1_CTCL1_CTCL1  tumor_cell     CD4_mem   
Li2024_atlas|ACACCGGCAAGCGTAG-0_CTCL1_CTCL1_CTCL1  tumor_cell     CD4_Th2   

                                                  cnv_ref  cached_malignant  \
obs_name                                                                      
Li2024_atlas|AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1   query              True   
Li2024_atlas|AAGCCGCAGAAACGAG-0_CTCL1_CTCL1_CTCL1   query              True   
Li2024_atlas|AAGGAGCAGACAGGCT-0_CTCL1_CTCL1_CTCL1   query              True   
Li2024_atlas|AATCCAGTCACGAAGG-0_CTCL1_CTCL1_CTCL1   query              True   
Li2024_atlas|ACACCGGCAAGCGTAG-0_CTCL1_CTCL1_CTCL1   query              True   

                                                   has_tcr  ... cnv_score  \
obs_name                                                    ...             
Li2024_atlas|AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1    False  ...  0.008917   
Li2024_atlas|AAGCCGCAGAAACGAG-0_CTCL1_CTCL1_CTCL1    False  ...  0.008413   
Li2024_atlas|AAGGAGCAGACAGGCT-0_CTCL1_CTCL1_CTCL1    False  ...  0.008670   
Li2024_atlas|AATCCAGTCACGAAGG-0_CTCL1_CTCL1_CTCL1    False  ...  0.008917   
Li2024_atlas|ACACCGGCAAGCGTAG-0_CTCL1_CTCL1_CTCL1    False  ...  0.007733   

                                                   cnv_cell_score  \
obs_name                                                            
Li2024_atlas|AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1        0.006043   
Li2024_atlas|AAGCCGCAGAAACGAG-0_CTCL1_CTCL1_CTCL1        0.008257   
Li2024_atlas|AAGGAGCAGACAGGCT-0_CTCL1_CTCL1_CTCL1        0.007048   
Li2024_atlas|AATCCAGTCACGAAGG-0_CTCL1_CTCL1_CTCL1        0.010773   
Li2024_atlas|ACACCGGCAAGCGTAG-0_CTCL1_CTCL1_CTCL1        0.009462   

                                                              cnv_leiden  \
obs_name                                                                   
Li2024_atlas|AAAC
... [3934 more chars truncated]
```

## Step 9 — UMAP: TCR + the 4 CD4 CNV methods

One panel per call on the shared MRVI u-UMAP (`X_mrvi_u`-derived), same as nb14 Step 8. The TCR
panel uses `tcr_label` — the **same malignant-clone scheme as nb16/17** (malignant = ALICE clone,
benign = TCR⁺ non-malignant unexpanded, else unlabeled) — so the malignant set and the benign
reference match across notebooks. CNV calls are defined only on the CD4 query (CD8 / NK appear benign).

```python  # cell 24
if UMAP_NPY.exists() and np.load(UMAP_NPY).shape[0] == adata.n_obs:
    adata.obsm["X_umap"] = np.load(UMAP_NPY)
else:                                          # stale / missing (cell set changed) -> recompute on X_mrvi_u
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
           ncols=2, size=2, wspace=0.3, save="_skin_cd4_cd8ref_cnv_methods.png")
```

```
... storing 'lin5' as categorical
... storing 'cnv_ref' as categorical
... storing 'cnv_leiden' as categorical
... storing 'cnv_call_method' as categorical
... storing 'malignant_evidence' as categorical

WARNING: saving figure to file figures/umap_skin_cd4_cd8ref_cnv_methods.png

<Figure size 1996.8x1728 with 5 Axes>
```

## Step 10 — TCR↔CNV quality metrics

For each method, restricted to TCR+ CD4 cells, pooled agreement with the TCR call:
**sensitivity** = of TCR-malignant cells, fraction called CNV-malignant; **specificity** =
of TCR-non-malignant cells, fraction called CNV-non-malignant (with raw counts). Same
`C.tcr_cnv_quality` machinery as nb14.

```python  # cell 26
# Step 10 — TCR<->CNV quality metrics (TCR+ CD4 cells, per method)
avail = adata.obs["has_tcr"].to_numpy() & (adata.obs["cnv_ref"].to_numpy() == "query")
quality = C.tcr_cnv_quality(adata, METHODS, "tcr_is_malignant", avail)
print(f"TCR<->CNV quality on {int(avail.sum())} TCR+ CD4 cells:")
print(quality.to_string())
```

```
TCR<->CNV quality on 132883 TCR+ CD4 cells:
                         n_tcr_malig  n_tcr_benign     tp     tn  sensitivity  specificity
strategy                                                                                  
strat1_cell_gmm                68605         64278  41974  34307        0.612        0.534
strat2_cell_mrvi               68605         64278   3922  62656        0.057        0.975
strat3_cnv_cluster             68605         64278  43679  28348        0.637        0.441
strat4_cnv_cluster_mrvi        68605         64278   7872  63205        0.115        0.983
```
