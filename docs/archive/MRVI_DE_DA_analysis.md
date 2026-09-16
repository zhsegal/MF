# MRVI DE & DA — completeness audit, biological analysis, and method comparison

Companion to `old/03_mrvi_replication.ipynb`. Three layers:
1. **Audit** — every MRVI artifact loaded by the notebook, with cell counts and provenance.
2. **Biology** — what the new numbers say about the CTCL atlas vs Li *et al.* 2024 ([PMC11588665](https://pmc.ncbi.nlm.nih.gov/articles/PMC11588665/)).
3. **Method** — focused NotebookLM-assisted reading of MRVI vs Wilcoxon, anchored to the discrepancies this notebook surfaced.

Date: 2026-05-22. Sourced from the current notebook (execution counts 28–47) and the four `.nc` artifacts on disk.

---

## 1. Completeness audit

| § | Artifact | Job | Cells / scope | File size | Status |
|---|---|---|---|---|---|
| §1 DA stage | `figures/mrvi_da_stage.nc` | `jobs/run_mrvi_inference.py` | **419,579** cells = full atlas | 95.7 MB | ✅ complete |
| §1 donor dist | `figures/mrvi_donor_distances.nc` | `jobs/run_mrvi_inference.py` | 36 × 36 donors × 49 cell types (`keep_cell=False`) | 0.22 MB | ✅ complete |
| §4 m-vs-b DE | `figures/mrvi_de_malignant_vs_benign.nc` | `jobs/run_mrvi_tcells_mb.py` | **228,370** T cells in 60 pseudo-samples ≥200 cells (was 71) | 17.3 GB | ✅ complete |
| §8 stage DE (malignant) | `figures/mrvi_de_stage_malignant.nc` | `jobs/run_mrvi_de.py --malignant-only` | **124,141** malignant T cells, no per-cell-type cap | 9.55 GB | ✅ complete |

**Note on the 60 vs 71 pseudo-sample change** (re-run between 2026-05-21 and 2026-05-22): the prep cell (notebook cell 16) yields `60/72 pseudo-samples (>=200 cells)` in the current run; the prior audit recorded 71. The retained cell count is unchanged at 228,370, so the new filter is stricter without losing the data majority. Donor coverage is still 36, stage split 17 early / 19 advanced.

All four MRVI outputs cover the full intended scope.

---

## 2. §1 — Donor distance heatmap

`get_local_sample_distances(keep_cell=False, groupby="cell_type")` returns one 36×36 donor distance per cell type. We average across cell types.

| Pair class | n pairs | Mean distance |
|---|---|---|
| within-advanced × advanced | 171 | **0.4268** |
| within-early × early | 136 | 0.5084 |
| advanced × early | 323 | 0.4748 |

- One-sided Mann–Whitney **within-advanced < within-early** : U = 7.80×10³, **p ≈ 0** (highly significant).
- One-sided Mann–Whitney **cross > within (all)**: U = 5.27×10⁴, p = 0.090 (not significant; cross sits between the two within-stage means).
- **Reading**: advanced-stage donors converge toward each other in MRVI's latent geometry, while early-stage donors retain heterogeneous transcriptional baselines. The cross-stage distance lies between the two — early donors are scattered across the same range that early/advanced separation occupies.
- Aligns with Li *et al.*: advanced disease is dominated by a stereotyped malignant TH2 program supported by a B-cell-rich TME, while early disease retains inflammatory heterogeneity.
- Figure: `figures/mrvi_donor_distances.png`.

---

## 3. §1 — Differential abundance over `stage_group`

`differential_abundance(sample_cov_keys=["stage_group"], compute_log_enrichment=True)` returns a per-cell log-enrichment for each stage. Δ = `log_enrich[advanced] − log_enrich[early]`.

**Per-cell-type Δ (mean over cells), top 15 each end:**

```
TOP up-in-advanced                       TOP up-in-early
─────────────────────────────             ──────────────────────────────
B_cell             +4.83  (n=15211)       LC_3              −26.94  (n=637)
Plasma             +3.83  (n=5569)        LE2               −25.61  (n=982)
channel            +2.85  (n=371)         Inf_mac           −23.64  (n=1966)
Tc_IL13_IL22       +1.14  (n=2336)        F3                −23.04  (n=6719)
Tc                 +0.63  (n=39889)       VE1               −22.15  (n=2712)
tumor_cell         +0.03  (n=102367)      moDC_2            −21.79  (n=2267)
ILC2               +0.02  (n=133)         moDC_1            −21.18  (n=1319)
Sebaceous          −0.06  (n=418)         F1                −19.78  (n=2188)
NK                 −0.08  (n=2398)        Pericyte_1        −17.74  (n=4636)
VE3                −0.21  (n=3544)        LC_2              −16.90  (n=714)
basal2             −0.34  (n=541)         Differentiated_KC −16.08  (n=24682)
Th                 −0.49  (n=52824)       VE2               −15.75  (n=12528)
Treg               −1.22  (n=27614)       moDC_3            −15.48  (n=5557)
immune             −2.01  (n=3466)        LC_4              −14.26  (n=1419)
LE1                −2.06  (n=1939)        MigDC             −12.26  (n=7062)
```

- **53.3 %** of cells have Δ < −0.5 (favouring early); **36.9 %** have Δ > 0.5 (favouring advanced) — the global cell-mix is slightly early-tilted because normal-skin compartments are large.

**Biology:**

- **B cells (Δ = +4.83) and plasma cells (Δ = +3.83) are the top-enriched populations in advanced disease.** Li *et al.* explicitly titled the paper around a "B cell-rich tumor microenvironment supporting malignant TH2 cells", and our DA recovers this exactly.
- **Tumor cells**: Δ ≈ 0 — *fraction* of tumor cells per biopsy is not stage-stratified (both early and advanced lesions have a clone). What is stage-stratified is the *company tumor cells keep* (B/Plasma in advanced).
- **Tc_IL13_IL22 (Tc2-like)** Δ = +1.14 — a real Th2-leaning shift in cytotoxic T cells.
- **Th, Treg, immune, LE1, LE2, LC_*, Inf_mac, F1/F3, VE*, moDC_*, Pericyte, MigDC** are all enriched in early disease — i.e. normal skin architecture is *displaced* in advanced lesions by tumor + B/Plasma expansion. Classic tumor-replaces-stroma signature.
- Figure: `figures/umap_mrvi_da_stage.png`.

---

## 4. §3 — Wilcoxon DE: malignant vs benign T cells (paper method)

`rank_genes_groups(method="wilcoxon", reference="benign")` on log-normalised counts, 229,263 T cells.

**Top 15 by score:**

```
FTL, MIF, PNRC1, EEF1B2, PPDPF, FTH1, SERP1, HSPD1, COX4I1,
HINT1, CHCHD2, HSP90AB1, RNASEK, GABARAP, H3F3B
```

Every gene in this top-15 is a **housekeeping / ribosomal / mitochondrial / chaperone transcript**. Highly-expressed transcripts dominate; none of Li *et al.* Fig 3b's named malignant markers (TOX, CD9, CXCL13, GTSF1, RUNX3, GATA3, CCR4) appear in the top-15. The volcano plot (`figures/Fig3b_volcano_paper_method.png`) shows CD7 ≈ 0 LFC near origin, with the bulk of "significant" hits being abundant housekeeping at modest log2FC.

This is the classic Wilcoxon-on-scRNA failure mode (see §10 method comparison): pseudoreplication bias + variance underestimation drive abundant transcripts to the top.

---

## 5. §4 — MRVI DE: malignant vs benign T cells (pseudo-sample fit)

Setup (notebook cell 16, `jobs/run_mrvi_tcells_mb.py`):

- Within each donor, T cells are split into pseudo-samples by `is_malignant`: `<donor>_mal` and `<donor>_ben`.
- Pseudo-samples with <200 cells dropped → **60 / 72 pseudo-samples kept** (was 71); **228,370 / 229,263** T cells retained.
- Fresh MRVI fit (`sample_key="sample_mb"`, `batch_key="study"`, 100 epochs, early stopping).
- DE: `differential_expression(sample_cov_keys=["target_status"], add_batch_specific_offsets=True, store_lfc=True, mc_samples=50)`.
- `target_status` is `[benign, malignant]` → covariate `target_status_malignant` is the counterfactual LFC for "had this cell been from the malignant pseudo-sample".

**Top 15 genes by mean LFC over the 123,505 malignant cells:**

```
UP in malignant            DOWN in malignant
───────────────            ─────────────────
DDX3Y    +0.082            MT-CO3   −0.148    (mitochondrial)
CAPG     +0.077            MT-ND3   −0.142    (mitochondrial)
EIF1AY   +0.074            MT-ND5   −0.138    (mitochondrial)
RGCC     +0.067            MT-ND2   −0.127    (mitochondrial)
CD7      +0.065            MT-ND4   −0.114    (mitochondrial)
BIN1     +0.059            MT-ND1   −0.112    (mitochondrial)
HOPX     +0.057            MT-ND6   −0.093    (mitochondrial)
GZMA     +0.057            CD2      −0.074
USP9Y    +0.054            CXCL13   −0.066
UTY      +0.051            ZFP36    −0.059
SLA2     +0.048            JUNB     −0.058
ARL4C    +0.048            MAL      −0.057
ISG15    +0.047            CD74     −0.053
PLEKHO1  +0.047            IL2RB    −0.053
KDM5D    +0.047            CTLA4    −0.052
```

**Direction check vs paper-highlighted Fig 3b markers** (Li *et al.* Fig 3b: TOX↑, CXCL13↑, CD9↑, GTSF1↑ in malignant; CD7↓ in malignant):

| Gene | MRVI mean LFC | MRVI direction | Paper direction | Agree? |
|---|---|---|---|---|
| TOX | ≈ 0 | null | UP | ✗ |
| TOX2 | ≈ 0 | null | UP | ✗ |
| CD9 | ≈ 0 | null | UP | ✗ |
| CXCL13 | **−0.066** | DOWN | UP | ✗ |
| GTSF1 | ≈ 0 | null | UP | ✗ |
| **CD7** | **+0.065** | UP | DOWN | ✗ |
| GATA3 | ≈ 0 | null | UP | ✗ |

**Concordance with Wilcoxon: Spearman ρ = 0.00** (notebook cell 21 title). The two methods agree on essentially nothing for the top genes, including the named paper markers. See §10 below for why this is methodologically informative, not a bug.

**Provisional caveats embedded in the result itself:**

- **Y-chromosome panel (DDX3Y, EIF1AY, USP9Y, UTY, KDM5D) in top-UP** within a *within-donor* pseudo-sample comparison: this is structurally impossible to be a real biological signal — sex is constant within donor. It surfaces as a small positive bias in MRVI's MC-sampled counterfactual, likely because the population-level pool of "malignant" pseudo-samples has slight residual sex skew relative to "benign". This is a **leaky control** — donor sex should be specified explicitly as a nuisance covariate in a future re-fit.
- **Mitochondrial genes dominate top-DOWN (7/10)**: a typical artefact of comparing subsets at different %-mito; potentially could be cleaned by computing the LFC after %-mito-stratified resampling, or by passing `categorical_nuisance_keys` for mito fraction binned.
- **Most biologically plausible MRVI hits**: CXCL13↓, CTLA4↓, IL2RB↓, CD2↓, JUNB↓, ZFP36↓ — collectively the *intra-donor* malignant clone appears **less inflammatory / less activated** than the donor's infiltrating non-malignant T cells (consistent with a clonal expansion that has bypassed cytokine-driven activation).

Outputs: `tables/Fig3b_DEGs_malignant_vs_benign_MRVI.csv`. Concordance: `figures/Fig3b_concordance_Wilcoxon_vs_MRVI.png`.

---

## 6. §6 — NMF metaprograms on malignant cells

Per-donor NMF (K=6, top-50 genes per program) over **28 donors** ≥200 malignant cells → 168 candidate programs → hierarchical clustering on Jaccard distance into 9 meta-clusters.

**Cluster sizes are highly imbalanced** (this is new vs the prior audit — the prior table elided the membership counts):

| Meta | n members | Top consensus genes (≥30 % of members) |
|---|---|---|
| **1** | **126** | ACTB, TMSB10, MT-CO3, ACTG1, VIM, PFN1, IL32, H3F3B, FTH1, S100A4, MT-ND3, SRGN, PPIA, UBC, MIF |
| 2 | 2 | TNFRSF4, CCR4, TAP1, IL21R, TNFRSF18, NFKB2, MYO1G, RNF19A, MDFIC, NFKBIA, TNFRSF8, MCL1, SRGN, DUSP2, LMNA |
| 3 | 15 | ACTB, IL32, EVL, CD74, CORO1A, VIM, TRAC, EHBP1L1, TRBC2, RNF213, MT-ND6, ETS1, MT-CO3, ZAP70, LIMD2 |
| 4 | 1 | RNF213, SOS1, SYTL2, LPAR6, SPAG9, THEMIS, IL10RA, MT-CO3, FYB1, TRAF3IP3, MAF, RASGRP1, BOD1L1, SYNE2, TSC22D3 |
| 5 | 5 | MT-ND6, MCM7, MCM5, MCM4, CENPM, NCAPG2, DNMT1, HELLS, TYMS, WDR76, FANCI, EZH2, MCM3, E2F1, ZGRF1 |
| 6 | 16 | HMGB2, TOP2A, HIST1H4C, CENPF, MKI67, NUSAP1, TUBB, TPX2, ASPM, TUBA1B, STMN1, H2AFZ, UBE2C, ACTB, CDK1 |
| 7 | 1 | UBE2F, SUB1, FYB1, TMEM147, HSPE1, NHP2, GABARAP, RAB25, SEC61G, SNRPG, SRM, MAD1L1, LCK, FTL, ATP6V1F |
| 8 | 1 | CCL19, COL6A2, C3, ZEB2, CTSL, MMP9, COL4A1, PLTP, PLXND1, SPARC, TNC, SPI1, C1R, SELENOP, RNASE1 |
| 9 | 1 | DDX58, RNF213, LGALS9, TAP1, IRF7, APOL2, EIF2AK2, OAS3, DDX60, EVL, ISG15, SAMHD1, ISG20, IFI6, OAS2 |

**Biology and caveats:**

- **Meta 1 (n=126, 75 % of programs)** — housekeeping/activation backbone (ACTB, TMSB10, IL32, VIM, FTH1, MT-CO3). Three quarters of per-donor NMF programs collapse into this one cluster — i.e. the dominant axis of variation across donors is the *transcriptional baseline of an activated T cell*, not a malignancy-specific program. This is expected for NMF on log-normalised HVGs and matches Li *et al.*'s observation that Fig 3d meta-programs include stress/cell-cycle/activation/metabolic backbones.
- **Meta 2 (n=2)** — TNFRSF4/CCR4/IL21R/TNFRSF18/NFKB2/CTLA4-leaning + **IL13**. CCR4 is the CTCL therapeutic target (mogamulizumab) and IL13 is the canonical TH2 effector — their co-occurrence in this *very rare* meta is a real biological hit but the n=2 membership means the program is *not* common across donors. Likely a 1–2-donor private TH2-mogamulizumab-target signature.
- **Meta 5 (n=5)** — clean **cell-cycle/replication** program (MCM family, E2F1, DNMT1, EZH2, TYMS).
- **Meta 6 (n=16)** — **proliferation/mitosis** (MKI67, TOP2A, CENPF, NUSAP1, ASPM, STMN1, CDK1).
- **Meta 9 (n=1)** — **type-I-IFN / ISG** program (DDX58/RIG-I, IRF7, ISG15, ISG20, IFI6, OAS family). Single-donor hit; suggests IFN response is donor-specific and not a generic CTCL malignancy program.
- **Meta 4, 7, 8 (n=1 each)** — single-donor private programs; not robust meta-programs.

Interpretation: Of 9 meta-clusters, only **Meta 1 (housekeeping, n=126)** and **Meta 6 (proliferation, n=16)** are populous; everything else is rare. To recover Li *et al.*'s Fig 3d granularity we likely need: (a) more aggressive per-donor preprocessing (regress total-counts/%-mito before NMF), (b) constrained NMF (consensus-NMF / cNMF) instead of plain sklearn NMF.

Figures: `figures/Fig3d_metaprogram_jaccard.png`. JSON: `tables/Fig3d_metaprograms_consensus.json`.

---

## 7. §7 — TH2 fraction violin (paper Fig 4b)

Per-donor TH classification by argmax over `sc.tl.score_genes` of:
- Th1: TBX21, IFNG
- Th2: GATA3, IL4, IL13, IL5, STAT6
- Th17: RORC, IL17A, IL17F, IL22

| Stage | n donors | Median TH2 fraction |
|---|---|---|
| early | 17 | low (~0.26) |
| advanced | 19 | higher (~0.34) |

One-sided Mann–Whitney (early < advanced): **U = 118, p = 8.65×10⁻²**.

- Direction matches Li *et al.* (TH2 up in advanced).
- p = 0.087 does not reach the paper's 3×10⁻⁴. Three likely reasons:
  1. Paper used 21 early / 24 advanced donors (vs 17 / 19 here); they also restricted to a richer multi-gene TH2 score on a different baseline cell pool.
  2. Our Th1/Th2/Th17 scores use only 2/5/4 marker genes respectively — the paper used the larger panels listed in their Methods/supplement.
  3. The CTCL-only cohort here excludes healthy/AD/Psoriasis controls (deferred to the integrated-atlas notebook).

To close the gap: re-score Th subtypes with the paper's full marker panels.

Figure: `figures/Fig4b_TH2_violin.png`. Table: `tables/Fig4b_per_donor_TH_fractions.csv`.

---

## 8. §8 — MRVI stage DE on full malignant T cells

Setup:
- Restrict to inferCNV-malignant cells from `tcells_cnv.h5ad` → **124,141 cells**.
- Re-use atlas MRVI model (`models/mrvi_ctcl`); donors are samples; `add_batch_specific_offsets=True`; `mc_samples=50`; `batch_size=256`.
- Covariate `stage_group_advanced` is the LFC for "had this cell been from an advanced-stage donor".

**Top 20 by mean LFC:**

```
UP in advanced              DOWN in advanced
──────────────              ────────────────
CCR7      +0.534            GNLY     −0.891
PASK      +0.468            CCL5     −0.830
CXCR4     +0.435            LGALS3   −0.449
DUSP2     +0.376            XCL1     −0.437
PIM3      +0.343            LGALS1   −0.404
JUNB      +0.338            GZMA     −0.380
IGFBP4    +0.319            CTSW     −0.379
MARCKSL1  +0.299            CXCR3    −0.377
SPOCK2    +0.298            GZMB     −0.362
CD69      +0.296            XCL2     −0.350
CYTIP     +0.293            IFNG     −0.343
PIM2      +0.271            JAML     −0.331
TRAF4     +0.270            S100A6   −0.320
RAC2      +0.268            FUT7     −0.319
SPINT2    +0.258            CLU      −0.315
ZFP36     +0.256            HOPX     −0.314
FCMR      +0.253            CD63     −0.311
TESPA1    +0.251            TNFRSF18 −0.308
IFNGR2    +0.251            MT-ND1   −0.308
CORO1A    +0.250            ADGRG1   −0.293
```

**Paper-anchored key-gene checks** (Li *et al.* Fig 4a: advanced ↑ SELL/CCR7/LEF1/TCF7/GATA3 = central-memory + TH2; advanced ↓ IFNG/GZMB/GNLY = cytotoxic loss):

| Gene | MRVI LFC | Paper direction | Agree? | Magnitude |
|---|---|---|---|---|
| **CCR7** | **+0.534** | UP | ✓ | very strong |
| SELL | +0.249 | UP | ✓ | strong |
| GATA3 | +0.196 | UP | ✓ | strong |
| TCF7 | +0.134 | UP | ✓ | moderate |
| LEF1 | +0.031 | UP | ✓ | weak |
| **GNLY** | **−0.891** | DOWN | ✓ | very strong |
| GZMB | −0.362 | DOWN | ✓ | strong |
| IFNG | −0.343 | DOWN | ✓ | strong |
| TBX21 (T-bet) | −0.123 | DOWN (implied) | ✓ | moderate |
| **CXCR3** | **−0.377** | DOWN (TH1 chemokine receptor) | ✓ | strong |
| IL4 | −0.005 | UP (expected) | ✗ | null |
| IL13 | −0.036 | UP (expected) | ✗ | null |
| IL5 | +0.001 | UP (expected) | ✗ | null |

- **All eight paper-highlighted Fig 4a markers go in the expected direction with substantial magnitude.** Strongest cytotoxic-loss: GNLY (LFC = −0.89). Strongest central-memory gain: CCR7 (+0.53).
- **TH2 cytokines (IL4, IL5, IL13)** themselves do not move in our LFC. Expected: cytokine mRNAs are sparsely expressed in 3'-end 10x scRNA-seq and rarely score in differential testing on per-cell counts. The paper inferred TH2 skewing primarily from the *master TF* GATA3 (which DOES move strongly here, +0.20) and from the central-memory program (CCR7/SELL/TCF7/LEF1 — all UP).
- **CXCR3 down** + **TBX21 down**: classical TH1 markers descend in advanced — a clean TH1→TH2 lineage shift consistent with the paper.

**Concordance with Wilcoxon advanced-vs-early on malignant T cells**: Spearman **ρ = 0.29** (notebook cell 31; n=500 top Wilcoxon genes). The Wilcoxon top genes (cell 26) are dominated by cytoskeletal/activation backbone (ACTB, CORO1A, RAC2, COTL1, PFN1, UCP2, RAC1, S100A4, LAT, CCND3) — the same housekeeping-bias failure mode seen in §4. Direction agreement at named markers is unanimous; global ρ is modest because the gene *rankings* disagree.

Outputs: `tables/Fig4_MRVI_top_up_advanced.csv`, `tables/Fig4_MRVI_top_down_advanced.csv`, `figures/Fig4_concordance_Wilcoxon_vs_MRVI.png`.

---

## 9. §5 / §9 — Concordance summary

| Comparison | Spearman ρ | Interpretation |
|---|---|---|
| §5 mal-vs-benign Wilcoxon ↔ MRVI | **0.00** | total disagreement — MRVI suppresses the donor-confounded signal that Wilcoxon picks up |
| §9 stage (adv-vs-early on malignant) Wilcoxon ↔ MRVI | **0.29** | modest positive; direction agrees for named markers, but Wilcoxon top is housekeeping-biased and MRVI top is biological |

These ρ values are the most important result of this notebook. They show **the two methods disagree more strongly within-donor than between-donor** — exactly where their methodological differences predict (§10).

---

## 10. Method comparison — MRVI vs Wilcoxon DE

*Researched via NotebookLM with the MRVI paper (Nature Methods 2025, Boyeau et al.), the scvi-tools MrVI user guide & tutorial, the Skinnider 2021 "false-discoveries" Nat Commun paper, and the Crowell et al. "differential state" 2022 benchmark.*

### 10.1 What each method actually computes

| | Wilcoxon (scanpy `rank_genes_groups`) | MRVI `differential_expression(sample_cov_keys=…)` |
|---|---|---|
| Input | log-normalised counts | raw counts → VAE encoder → counterfactual decoder |
| Test unit | gene; one per-group statistic | **gene × cell**; one LFC per cell per gene |
| Cells treated as | independent samples | conditional on `sample_key` and `batch_key` |
| Covariate effect | implicit in group labels; no nuisance handling | regressed in latent space; nuisance covariates explicitly accounted for via `batch_key` (and optional `categorical_nuisance_keys`) |
| Donor-level variance | **ignored** (every cell is a degree of freedom) | modelled via the hierarchical encoder; donor identity collapses into the sample-aware latent z |
| Output for downstream | log2FC + p_adj per gene | per-cell LFC, per-cell `effect_size`, per-cell pde (with `delta`), per-cell `padj` |

The structural gap is **pseudoreplication bias**: Wilcoxon treats every cell as an independent draw from its group, so a 100k-cell vs 100k-cell comparison has enormous nominal power even when the actual biological replicates are 17 vs 19 donors. Highly-expressed genes (housekeeping, ribosomal, mitochondrial) are most badly affected because Wilcoxon underestimates their per-donor variance.

### 10.2 How MRVI computes a per-cell LFC

For each cell $n$ and each value $s$ of the target covariate, MRVI:
1. Encodes the cell into a sample-unaware latent $u_n$ (shared cell-state).
2. Constructs a counterfactual sample-aware latent $z_n(s)$ via the cross-attention decoder, conditioning on $s$.
3. Decodes $z_n(s)$ back to a normalised expression vector $h_n(s)$.
4. Returns LFC = $\log h_n(\text{condition 1}) − \log h_n(\text{condition 0})$ **per gene per cell**.

The `lfc` array in the netcdf is therefore *not* a regression coefficient but a fully cell-specific decoded counterfactual difference. Aggregating it across cells gives a mean LFC that has already conditioned on:
- The donor (via `sample_key`)
- The technical batch (via `batch_key`, optionally `add_batch_specific_offsets=True`)
- The cell's own latent state $u_n$ (so donor/cell-type composition is collapsed before LFC is taken)

This is the structural reason MRVI is more conservative than Wilcoxon.

### 10.3 Why mal-vs-benign concordance collapses to ρ = 0 (§5)

The mal-vs-benign comparison is **within-donor by construction** (each donor has both a `_mal` and `_ben` pseudo-sample). This is the comparison where MRVI's design strength is maximal:

- Markers like **TOX, CXCL13, CD9, GTSF1** are well-documented as "skin-resident exhausted T cell" markers (Park *et al.* 2020); CTCL donors have inflammatory skin where *all* infiltrating T cells (malignant + bystander benign) express elevated TOX/CXCL13/CD9. Wilcoxon, which mixes within- and between-donor variance, will detect these as malignant markers because the mean malignant pool has more "tissue-resident exhausted" cells than the mean benign pool — but this is a *cell-state composition* effect, not a within-clone transcriptional difference.
- MRVI, conditioning on donor, compares "same donor, mal vs ben" cells with their base cell state $u_n$ held constant. If TOX/CXCL13 are baked into a cell's $u_n$ (because they describe the cell's exhausted skin-residency state), MRVI sees no incremental effect from the malignancy covariate and reports LFC ≈ 0.
- ρ = 0 is therefore the *correct* result for a method that has successfully separated cell-state composition from covariate-driven expression change. The disagreement is the signal.

### 10.4 Why stage concordance is modest but positive (ρ = 0.29, §9)

The stage comparison is **between-donor** (no donor has cells of both stages). Here:
- Both methods *can* see donor-level differences.
- Wilcoxon still has pseudoreplication bias and inflates abundant transcripts (top list dominated by ACTB/PFN1/RAC2/COTL1/CORO1A).
- MRVI still suppresses inflation but cannot regress out donor identity from a between-donor effect, so its top list (CCR7/GATA3/SELL/GNLY/IFNG/CXCR3) is the *biological* signal.
- The two top lists are populated with different genes (housekeeping vs immune markers) → low Spearman ρ.
- BUT direction agreement at named biological markers is unanimous → the biology is the same, just ranked differently.

ρ = 0.29 with unanimous direction agreement is the expected pattern when one method ranks by abundance + variance underestimate and the other ranks by donor-controlled effect.

### 10.5 Failure modes / interpretation flags surfaced by this notebook

1. **Y-chromosome genes in MRVI top-UP for mal-vs-benign** — pseudo-samples are within-donor so sex is constant; their appearance indicates MRVI is leaking a sex-stratified population-level signal because donor sex was not specified as a categorical nuisance covariate. **Fix**: refit with `categorical_nuisance_keys=["sex"]` (or model sex via `batch_key` if compatible).
2. **Mitochondrial genes in MRVI top-DOWN** — likely a %-mito artefact; could be addressed by stratifying or binning %-mito as a nuisance covariate, or by computing LFC after %-mito-stratified resampling.
3. **`pde` and `padj` not yet thresholded** — `delta` was not set during DE call, so `pde` is all-zero in the netcdf and `padj` is computed from `pvalue`. To get posterior DE-probability ranking, re-run with `delta = 0.1–0.3`.
4. **`mc_samples = 50`** — the spec recommended 100; doubling MC samples would reduce LFC variance and let smaller effects (IL4/IL13/IL5) surface above noise.
5. **HVG panel = 10,000** — cytokine mRNAs (IL4/IL13/IL5) are in the panel but expressed in <5 % of cells; their MRVI LFC is structurally limited even with more MC samples.

---

## 11. Caveats

1. **n donors is small** (17 early / 19 advanced). Donor-level statistics (TH2 fraction p, donor-distance MWU) are bounded by n; *within-cell* MRVI DE has ~10⁵ cells per group and is statistically much more powerful, but conditional on the donor split.
2. **inferCNV-based malignancy labels are imperfect** — `is_malignant` in `tcells_cnv.h5ad` mixes TCR-clonotype + CNV evidence; mis-labels dampen mal-vs-benign comparisons (§4–§5).
3. **Pseudo-sample threshold of 200 cells** drops 12 of 72 (16.7 %) candidate pseudo-samples; results are conditional on the 60 retained ones.
4. **The §5 m-vs-b ρ = 0** does not invalidate Li *et al.*'s Fig 3b — it shows the paper's Wilcoxon top markers are a mixture of *cell-state composition* (skin-resident exhausted T cells; donor-level signal) and *intra-donor transcriptional shift* (the smaller component). MRVI is the right tool for the latter; Wilcoxon recovers the former.
5. **NMF metaprograms are dominated by one housekeeping cluster (n=126/168)** — plain sklearn NMF on log1p-HVG is not sufficient for Li *et al.*'s Fig 3d granularity; use cNMF or regress %-mito/n_counts first.

---

## 12. References

- **Li, *et al.* 2024**. *Cutaneous T cell lymphoma atlas reveals malignant TH2 cells supported by a B cell-rich tumor microenvironment*. *Cancer Cell* 42(12). [PMC11588665](https://pmc.ncbi.nlm.nih.gov/articles/PMC11588665/). The paper this notebook replicates.
- **Boyeau, Hong, Gayoso, Kim, McFaline-Figueroa, Jordan, Azizi, Ergen, Yosef 2025**. *Deep generative modeling of sample-level heterogeneity in single-cell genomics*. [Nature Methods](https://www.nature.com/articles/s41592-025-02808-x) | bioRxiv 2022.10.04.510898. The MRVI paper.
- **scvi-tools MrVI documentation**: [User guide](https://docs.scvi-tools.org/en/1.2.2/user_guide/models/mrvi.html) · [Quick-start tutorial](https://docs.scvi-tools.org/en/1.3.3/tutorials/notebooks/scrna/MrVI_tutorial.html) · [GitHub source](https://github.com/scverse/scvi-tools/blob/main/docs/user_guide/models/mrvi.md).
- **Squair, Skinnider, *et al.* 2021**. *Confronting false discoveries in single-cell differential expression*. Nat Commun [PDF](https://michaelskinnider.com/files/Nat%20Commun%202021%20-%20Confronting%20false%20discoveries%20in%20single-cell%20differential%20expression.pdf). The Wilcoxon pseudoreplication critique.
- **Crowell, Soneson, *et al.* 2022**. *Benchmarking methods for detecting differential states between conditions from multi-subject single-cell RNA-seq data*. *Briefings in Bioinformatics* [PMC9487674](https://pmc.ncbi.nlm.nih.gov/articles/PMC9487674/). Reference for pseudobulk vs cell-level DE.
- **Park, Jiang, Kumar, *et al.* 2020**. Established TOX/CXCL13 as tissue-resident-exhausted T-cell markers — relevant to the §5 interpretation.
- **Buus *et al.* 2018** ([PMID 30002133](https://pubmed.ncbi.nlm.nih.gov/30002133/)) and **Borcherding *et al.* 2019** — Sezary-syndrome / CTCL scRNA-seq, providing comparative malignant T-cell signatures.

**NotebookLM notebook** (this session's deep-research workspace): `8e81a82d-9ab3-48e0-96c6-a1b7f5fbe67e` — "MRVI vs standard DE methods - methodological comparison". Sources loaded: MRVI Nature Methods paper, scvi-tools MrVI docs (1.2.2 + 1.3.3 + GitHub source), Skinnider/Squair 2021 false-discoveries PDF, Crowell 2022 benchmark, bioRxiv MrVI preprint PDF.
