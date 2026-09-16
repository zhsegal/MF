rere# `old/03_mrvi_replication.ipynb` — results only
Numerical results / printed outputs from the re-run notebook (code, figures, and log/warning noise stripped). Section headers retained for context. scvi 1.3.3 · scanpy 1.11.5.

---

# CTCL atlas replication — stage + inferCNV cascade (MrVI-centric)

Reads the cached MrVI run from `old/02_mrvi.ipynb` and implements the slice of `archive/CTCL_atlas_notebook_replication_spec.md` that works on the CTCL-only atlas:

- §1 MrVI sample-level diagnostics (donor distances, DA)
- §2 inferCNV → malignant T-cell labels (spec Cell 3, Fig 3a)
- §3 Pseudobulk DESeq2 DE malignant vs benign (spec Cell 4a, Fig 3b)
- §4 Second MrVI fit with pseudo-samples → MrVI DE 4b
- §5 Concordance scatter pseudobulk-DESeq2 vs MrVI for malignant
- §6 cNMF metaprograms on malignant cells (spec Cell 5, Fig 3d)
- §7 Stage analyses on malignant cells (spec Cell 6a, Fig 4a/b — paper p≈3e-4)
- §8 MrVI stage DE restricted to malignant (spec Cell 6b)
- §9 Concordance scatter for stage

Deferred to a later notebook: CTCL-vs-healthy/AD/Pso (needs Integrated atlas + 2nd MrVI fit), LIANA, drug2cell, cell2location.

Run once if missing: `pip install infercnvpy==0.6.1` (not in env by default).

```
jax 0.8.2 | jaxlib 0.8.2 | devices [CudaDevice(id=0)] | backend gpu
scvi 1.3.3 | scanpy 1.11.5
CACHE_H5AD = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/cache/mrvi_ctcl_cache.h5ad
```

## Load cache + reload MrVI

```
AnnData object with n_obs × n_vars = 419579 × 10000
    obs: 'tissue', 'sex', 'study', 'donor', 'tech', 'cell_type', 'stage', 'n_counts', '_indices', '_scvi_sample', '_scvi_batch', '_scvi_labels'
    var: 'highly_variable', 'highly_variable_rank', 'means', 'variances', 'variances_norm', 'highly_variable_nbatches'
    uns: '_scvi_manager_uuid', '_scvi_uuid', 'broad_ct_colors', 'cell_type_colors', 'donor_colors', 'groups1_colors', 'hvg', 'leiden', 'log1p', 'mrvi', 'mrvi_artifacts', 'neighbors', 'scvi', 'stage_colors', 'study_colors', 'tech_colors', 'umap'
    obsm: 'X_mrvi', 'X_scVI', 'X_scVI_MDE', 'X_umap', 'X_umap_mrvi', 'X_umap_scvi'
    layers: 'raw_counts'
    obsp: 'connectivities', 'distances', 'mrvi_connectivities', 'mrvi_distances', 'scvi_connectivities', 'scvi_distances'
artifacts: {'batch_key': 'study', 'de_stage_tcells_nc': '/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/mrvi_de_stage_tcells.nc', 'hvg_n_top_genes': 10000, 'model_dir': '/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/models/mrvi_ctcl', 'sample_key': 'donor', 'umap_keys': array(['X_umap_mrvi', 'X_umap_scvi'], dtype=object)}
reloaded model. trained: True
donors per stage_group:
stage_group
early       17
advanced    19
Name: donor, dtype: int64
```

## §1 MrVI sample-level diagnostics

Donor-distance heatmap (averaged across cell types) and differential abundance over stage_group.

```
set SUBMIT=True to fire bsub.
DIST_NC exists: True
DA_NC   exists: True
```

```
tissue sex             study  donor  \
AAACCTGAGAAGCCCA-0_CTCL1_CTCL1_CTCL1  Epidermis   F  Sanger_Ncl_Fresh  CTCL1   
AAACCTGAGAATGTTG-0_CTCL1_CTCL1_CTCL1  Epidermis   F  Sanger_Ncl_Fresh  CTCL1   
AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1  Epidermis   F  Sanger_Ncl_Fresh  CTCL1   
AAACCTGAGCGTTCCG-0_CTCL1_CTCL1_CTCL1  Epidermis   F  Sanger_Ncl_Fresh  CTCL1   
AAACCTGAGTACGTTC-0_CTCL1_CTCL1_CTCL1  Epidermis   F  Sanger_Ncl_Fresh  CTCL1   
                                     tech            cell_type stage  \
AAACCTGAGAAGCCCA-0_CTCL1_CTCL1_CTCL1  10x   Differentiated_KC*   IIB   
AAACCTGAGAATGTTG-0_CTCL1_CTCL1_CTCL1  10x   Differentiated_KC*   IIB   
AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1  10x           tumor_cell   IIB   
AAACCTGAGCGTTCCG-0_CTCL1_CTCL1_CTCL1  10x  Undifferentiated_KC   IIB   
AAACCTGAGTACGTTC-0_CTCL1_CTCL1_CTCL1  10x    Differentiated_KC   IIB   
                                      n_counts  _indices  _scvi_sample  \
AAACCTGAGAAGCCCA-0_CTCL1_CTCL1_CTCL1    6708.0         0             0   
AAACCTGAGAATGTTG-0_CTCL1_CTCL1_CTCL1    5956.0         1             0   
AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1     851.0         2             0   
AAACCTGAGCGTTCCG-0_CTCL1_CTCL1_CTCL1    1092.0         3             0   
AAACCTGAGTACGTTC-0_CTCL1_CTCL1_CTCL1   21186.0         4             0   
                                      _scvi_batch  _scvi_labels stage_group  
AAACCTGAGAAGCCCA-0_CTCL1_CTCL1_CTCL1            3             0    advanced  
AAACCTGAGAATGTTG-0_CTCL1_CTCL1_CTCL1            3             0    advanced  
AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1            3             0    advanced  
AAACCTGAGCGTTCCG-0_CTCL1_CTCL1_CTCL1            3             0    advanced  
AAACCTGAGTACGTTC-0_CTCL1_CTCL1_CTCL1            3             0    advanced
```

```
<xarray.Dataset> Size: 259kB
Dimensions:         (cell_type_name: 49, sample_x: 36, sample_y: 36)
Coordinates:
  * cell_type_name  (cell_type_name) <U19 4kB 'tumor_cell' 'Th' ... 'ILC2'
  * sample_x        (sample_x) <U6 864B 'CTCL1' 'CTCL2' ... 'PT55' 'PT56'
  * sample_y        (sample_y) <U6 864B 'CTCL1' 'CTCL2' ... 'PT55' 'PT56'
Data variables:
    cell_type       (cell_type_name, sample_x, sample_y) float32 254kB ...
```

```
<xarray.Dataset> Size: 156MB
Dimensions:                  (cell_name: 419579, sample: 36, stage_group: 2)
Coordinates:
  * cell_name                (cell_name) <U49 82MB 'AAACCTGAGAAGCCCA-0_CTCL1_...
  * sample                   (sample) <U6 864B 'CTCL1' 'CTCL5' ... 'CTCL18'
  * stage_group              (stage_group) <U8 64B 'advanced' 'early'
Data variables:
    log_probs                (cell_name, sample) float32 60MB ...
    stage_group_log_probs    (cell_name, stage_group) float64 7MB ...
    stage_group_log_enrichs  (cell_name, stage_group) float64 7MB ...
```

## §2 Load T-cell subset with inferCNV labels

`ad_t` (T cells + `is_malignant`) is built in `04_infercnv.ipynb` and cached. Load it here.

```
AnnData object with n_obs × n_vars = 229263 × 10000
    obs: 'tissue', 'sex', 'study', 'donor', 'tech', 'cell_type', 'stage', 'n_counts', '_indices', '_scvi_sample', '_scvi_batch', '_scvi_labels', 'stage_group', 'cnv_leiden', 'cnv_score', 'is_T', 'is_malignant'
    var: 'highly_variable', 'highly_variable_rank', 'means', 'variances', 'variances_norm', 'highly_variable_nbatches', 'chromosome', 'start', 'end', 'gene_id', 'gene_name'
    uns: '_scvi_manager_uuid', '_scvi_uuid', 'broad_ct_colors', 'cell_type_colors', 'cnv', 'cnv_leiden', 'cnv_neighbors', 'donor_colors', 'groups1_colors', 'hvg', 'is_malignant_colors', 'leiden', 'log1p', 'mrvi', 'mrvi_artifacts', 'neighbors', 'scvi', 'stage_colors', 'study_colors', 'tech_colors', 'umap'
    obsm: 'X_cnv', 'X_cnv_pca', 'X_mrvi', 'X_scVI', 'X_scVI_MDE', 'X_umap', 'X_umap_mrvi', 'X_umap_scvi'
    layers: 'raw_counts'
    obsp: 'cnv_neighbors_connectivities', 'cnv_neighbors_distances', 'connectivities', 'distances', 'mrvi_connectivities', 'mrvi_distances', 'scvi_connectivities', 'scvi_distances'
malignant: 124141 / 229263 (54.1%)
== malignant by stage_group ==
             n_cells  n_malignant  frac_malignant
stage_group                                      
early         134444        92515           0.688
advanced       94819        31626           0.334
== per-donor malignant fraction ==
donors: 36  |  with >=200 malignant: 28
count    36.000
mean      0.379
std       0.270
min       0.036
25%       0.116
50%       0.318
75%       0.639
max       0.872
Name: frac_malignant, dtype: float64
== cell_type composition ==
cell_type
tumor_cell      102367
Th               52824
Tc               39889
Treg             27614
Tc17_Th17         4233
Tc_IL13_IL22      2336
Name: count, dtype: int64
== cnv_score by malignant status ==
                 count   mean    std    min    25%    50%    75%    max
is_malignant                                                           
False         105122.0  0.003  0.002  0.000  0.002  0.003  0.005  0.011
True          124141.0  0.006  0.002  0.003  0.005  0.006  0.006  0.011
```

## §3 Pseudobulk DESeq2 DE — malignant vs benign T cells (spec Cell 4a, Fig 3b)

Paper method: Fig 3b legend states P-values from a **quasi-likelihood F-test + Benjamini–Hochberg** — a pseudobulk GLM on donor-aggregated counts, not cell-level Wilcoxon (which inflates significance via pseudoreplication; Squair 2021, Crowell 2022). We aggregate raw counts per **donor × malignant-status** and run **pydeseq2** (DESeq2-equivalent NB GLM, Wald + BH) with a paired design `~ donor + status`.

```
paired pseudobulk: 68 samples across 34 donors (dropped 2 unpaired donors)
significant genes (padj<0.05, |log2FC|>1): 64
          log2FoldChange  padj  baseMean
CDKN3              1.023   0.0   136.131
RRM2               1.440   0.0   450.267
TK1                1.234   0.0   237.726
NCAPG              1.054   0.0   165.990
MYBL2              1.520   0.0   187.218
CDC6               1.205   0.0    97.527
TPX2               1.045   0.0   236.868
TYMS               1.336   0.0   605.516
ASF1B              1.112   0.0   183.599
UBE2C              1.456   0.0   350.415
CDK1               1.019   0.0   272.061
PKMYT1             1.432   0.0   177.355
AURKB              1.253   0.0   136.740
DLGAP5             1.368   0.0   121.376
PCLAF              1.226   0.0   391.316
PBK                1.417   0.0    47.019
APOBEC3B           1.278   0.0    32.837
TOP2A              1.135   0.0   492.577
UHRF1              1.140   0.0   207.711
CENPA              1.199   0.0    69.694
```

## §4 Second MrVI fit with pseudo-samples (spec Cell 4b)

MrVI requires sample-level covariates. Malignant status varies within donor, so we create per-donor × status pseudo-samples (`donor_mal`, `donor_ben`). Pseudo-samples with <200 cells are dropped.

```
keeping 60/72 pseudo-samples (>=200 cells)
sample_mb -> target_status mapping (constant per pseudo-sample):
target_status
1    60
Name: count, dtype: int64
```

```
set SUBMIT=True to fire bsub.
MB_LATENT exists: True
DE_MB_NC  exists: True
```

```
X_mrvi_mb: (228370, 10)
```

```
<xarray.Dataset> Size: 28GB
Dimensions:        (cell_name: 228370, covariate: 5, latent_dim: 30,
                    covariate_sub: 1, gene: 10000)
Coordinates:
  * cell_name      (cell_name) <U49 45MB 'AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL...
  * covariate      (covariate) <U23 460B 'offset_batch_0' ... 'target_status_...
  * latent_dim     (latent_dim) int64 240B 0 1 2 3 4 5 6 ... 24 25 26 27 28 29
  * covariate_sub  (covariate_sub) <U23 92B 'target_status_malignant'
  * gene           (gene) <U11 440kB 'SAMD11' 'KLHL17' ... 'MT-ND5' 'MT-ND6'
Data variables:
    beta           (cell_name, covariate, latent_dim) float32 137MB ...
    effect_size    (cell_name, covariate) float32 5MB ...
    pvalue         (cell_name, covariate) float32 5MB ...
    padj           (cell_name, covariate) float32 5MB ...
    lfc            (covariate_sub, cell_name, gene) float32 9GB ...
    lfc_std        (covariate_sub, cell_name, gene) float32 9GB ...
    pde            (covariate_sub, cell_name, gene) float32 9GB ...
covariate: target_status_malignant
```

## §5 Concordance pseudobulk DESeq2 ↔ MrVI for malignant vs benign

## §6 cNMF metaprograms on malignant cells (spec Cell 5, Fig 3d)

Plain per-donor sklearn NMF is dominated by a depth/mito noise axis. The paper's antidote: exclude mito/stress genes and use **cNMF** (Kotliar et al. 2019), which runs many NMF replicates and uses density-based clustering of components to recover robust programs and discard the noise axis. Here we run **one pooled cNMF** on all malignant cells over K = 8–10.

**Note on confound regression:** explicit `regress_out(n_counts, pct_mito)` is *not* applied — it yields negative values, incompatible with NMF's non-negativity. cNMF's internal TPM + overdispersed-gene variance normalisation, combined with the mito/stress-gene exclusion below, is the standard realisation of the paper's confound-control step.

```
malignant T cells: (124141, 10000)
excluding 26 mito/stress genes from cNMF panel
cNMF input: (124141, 9974) -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/cache/cnmf_malignant_counts.h5ad
[Worker 0]. Starting task 0.
[Worker 0]. Starting task 1.
[Worker 0]. Starting task 16.
[Worker 0]. Starting task 17.
[Worker 0]. Starting task 18.
[Worker 0]. Starting task 19.
[Worker 0]. Starting task 20.
[Worker 0]. Starting task 21.
[Worker 0]. Starting task 22.
[Worker 0]. Starting task 23.
[Worker 0]. Starting task 24.
[Worker 0]. Starting task 25.
[Worker 0]. Starting task 26.
[Worker 0]. Starting task 27.
[Worker 0]. Starting task 28.
[Worker 0]. Starting task 29.
[Worker 0]. Starting task 30.
[Worker 0]. Starting task 31.
[Worker 0]. Starting task 32.
[Worker 0]. Starting task 33.
[Worker 0]. Starting task 34.
[Worker 0]. Starting task 35.
[Worker 0]. Starting task 36.
[Worker 0]. Starting task 37.
[Worker 0]. Starting task 38.
[Worker 0]. Starting task 39.
[Worker 0]. Starting task 40.
[Worker 0]. Starting task 41.
[Worker 0]. Starting task 42.
[Worker 0]. Starting task 43.
[Worker 0]. Starting task 44.
[Worker 0]. Starting task 45.
[Worker 0]. Starting task 46.
[Worker 0]. Starting task 47.
[Worker 0]. Starting task 48.
[Worker 0]. Starting task 49.
[Worker 0]. Starting task 50.
[Worker 0]. Starting task 51.
[Worker 0]. Starting task 52.
[Worker 0]. Starting task 53.
[Worker 0]. Starting task 54.
[Worker 0]. Starting task 55.
[Worker 0]. Starting task 56.
[Worker 0]. Starting task 57.
[Worker 0]. Starting task 58.
[Worker 0]. Starting task 59.
Combining factorizations for k=8.
Combining factorizations for k=9.
Combining factorizations for k=10.
```

```
program 1: ['ZFP36L2', 'FTH1', 'PNRC1', 'SYTL3', 'CREM', 'PIK3R1', 'FYN', 'SERP1', 'HSP90B1', 'IL7R', 'EMD', 'XBP1', 'LEPROTL1', 'HERPUD1', 'DCXR']
program 2: ['LGALS3', 'S100A6', 'TMIGD2', 'NBAS', 'TSPAN13', 'S100A4', 'IL32', 'JAML', 'NME2', 'CTSH', 'PPIA', 'KRT86', 'LMO4', 'RCBTB2', 'KRT7']
program 3: ['NR4A1', 'DNAJB1', 'UBC', 'BAG3', 'IL13', 'HSP90AA1', 'PPP1R15A', 'UBB', 'JUN', 'DNAJA1', 'RGS2', 'DUSP1', 'DUSP2', 'NFKBIA', 'ZFAND2A']
program 4: ['IGFL2', 'ACTB', 'SPINT2', 'CYP7B1', 'ICA1', 'PDCD1', 'FZD3', 'CXCL13', 'EPCAM', 'UCP2', 'MME', 'RAC2', 'NMB', 'PON3', 'APBB2']
program 5: ['SYNGR1', 'FCER1G', 'CTSW', 'RARRES2', 'TYROBP', 'GZMB', 'ZNF683', 'LGALS1', 'TRGC1', 'KLRC1', 'CAPG', 'LAT2', 'FAM9C', 'ENPP1', 'SLC12A8']
program 6: ['USP6', 'CYP4F22', 'ADGRB1', 'CXCL12', 'SEC31B', 'KLRF2', 'LRP1B', 'TRBC2', 'ITGA1', 'ITGBL1', 'DDX4', 'CES4A', 'IGFBP5', 'DUSP6', 'NSMF']
program 7: ['AEBP1', 'MRC2', 'HACD1', 'PPP1R9A', 'SHISA2', 'IGFBP4', 'ATP9A', 'TSPOAP1', 'MYB', 'WNT5B', 'PCDHGA5', 'PHF19', 'GPC4', 'FARP1', 'TRAC']
program 8: ['MAL', 'MACC1', 'TNFRSF8', 'FAM30A', 'PEX13', 'WNT10A', 'CIITA', 'CTLA4', 'CNTNAP1', 'CYP20A1', 'ONECUT2', 'KCNMA1', 'IL26', 'TFRC', 'TP63']
program 9: ['SOHLH1', 'RGS12', 'MSANTD1', 'GNG4', 'EVX1', 'SPINT2', 'ZFR2', 'NPTX2', 'DLGAP2', 'NKX2-6', 'PRDX1', 'STRA6', 'KLHL35', 'TDRD10', 'STMND1']
program 10: ['PLTP', 'CD163', 'CD14', 'RNASE1', 'MMP12', 'COL4A1', 'LYZ', 'STAB1', 'COL4A2', 'SPARC', 'C1QA', 'IGFBP7', 'MMP9', 'DAB2', 'C1QB']
```

## §7 Stage analyses on malignant cells (spec Cell 6a)

Pseudobulk DESeq2 advanced vs early on malignant T cells (per-donor aggregation; same QLF-equivalent paper method as §3) + per-donor TH2 fraction violin (paper p ≈ 3e-4).

```
stage pseudobulk: 34 donors (15 early / 19 advanced)
significant genes (padj<0.05, |log2FC|>1): 73
        log2FoldChange   padj  baseMean
KRT86           -4.335  0.000   261.232
SPINT2           3.734  0.000   679.300
HES1             4.357  0.000    19.700
FAM71B           5.878  0.000    53.680
SVOPL            4.550  0.000    11.265
CYP7B1           4.485  0.001    29.448
IGFL2            5.398  0.001    49.696
HEYL             5.481  0.001    25.408
IGFBP2           3.811  0.001    21.600
ICA1             3.382  0.002   123.597
ASCL1            4.837  0.002     9.669
IFNGR2           2.396  0.002   124.373
LEMD1           -5.271  0.003    14.201
GPAT3           -3.224  0.003    43.745
KCNK1            4.102  0.004    21.592
KIF26B           4.260  0.004    20.767
ZBTB46          -3.094  0.004    45.535
ZBED2           -2.526  0.004    67.880
NPW              3.862  0.005    59.716
EPCAM            4.968  0.005    17.060
```

```
TH2 (GATA3) fraction Mann-Whitney one-sided early<advanced: U=92  p=1.44e-02  (paper: 3e-4)
```

## §8 MrVI stage DE on malignant T cells (spec Cell 6b)

Loads the full-malignant DE generated by `jobs/run_mrvi_de.py --malignant-only` (no per-cell-type cap; 124,141 cells × 10k genes). Earlier runs used a 7,500-cell subsample of all T cells; this version covers every inferCNV-malignant T cell.

```
<xarray.Dataset> Size: 15GB
Dimensions:        (cell_name: 124141, covariate: 5, latent_dim: 30,
                    covariate_sub: 1, gene: 10000)
Coordinates:
  * cell_name      (cell_name) <U47 23MB 'AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL...
  * covariate      (covariate) <U20 400B 'offset_batch_0' ... 'stage_group_ad...
  * latent_dim     (latent_dim) int64 240B 0 1 2 3 4 5 6 ... 24 25 26 27 28 29
  * covariate_sub  (covariate_sub) <U20 80B 'stage_group_advanced'
  * gene           (gene) <U11 440kB 'SAMD11' 'KLHL17' ... 'MT-ND5' 'MT-ND6'
Data variables:
    beta           (cell_name, covariate, latent_dim) float32 74MB ...
    effect_size    (cell_name, covariate) float32 2MB ...
    pvalue         (cell_name, covariate) float32 2MB ...
    padj           (cell_name, covariate) float32 2MB ...
    lfc            (covariate_sub, cell_name, gene) float32 5GB ...
    lfc_std        (covariate_sub, cell_name, gene) float32 5GB ...
    pde            (covariate_sub, cell_name, gene) float32 5GB ...
DE cells (all malignant): 124141
top up in advanced (malignant cells):
 CCR7        0.534398
PASK        0.467676
CXCR4       0.434871
DUSP2       0.375661
PIM3        0.343128
JUNB        0.338202
IGFBP4      0.318678
MARCKSL1    0.298848
SPOCK2      0.297559
CD69        0.295523
CYTIP       0.292780
PIM2        0.271429
TRAF4       0.269712
RAC2        0.268355
SPINT2      0.257931
ZFP36       0.256085
FCMR        0.252776
TESPA1      0.251425
IFNGR2      0.250576
CORO1A      0.249547
Name: lfc_mrvi_stage_malignant, dtype: float32
top down in advanced (malignant cells):
 GNLY       -0.890561
CCL5       -0.829536
LGALS3     -0.448793
XCL1       -0.436718
LGALS1     -0.403910
GZMA       -0.379693
CTSW       -0.378820
CXCR3      -0.377055
GZMB       -0.362150
XCL2       -0.349933
IFNG       -0.342617
JAML       -0.330633
S100A6     -0.319590
FUT7       -0.319465
CLU        -0.315058
HOPX       -0.314208
CD63       -0.310673
TNFRSF18   -0.307895
MT-ND1     -0.307719
ADGRG1     -0.292905
Name: lfc_mrvi_stage_malignant, dtype: float32
```

## §9 Concordance pseudobulk DESeq2 ↔ MrVI for stage (malignant cells)
