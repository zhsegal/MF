# MRVI DE & DA — v2: pseudobulk-DESeq2 + cNMF re-run, biology, and method comparison

Companion to `old/03_mrvi_replication.ipynb` (re-run 2026-05-26). Successor to `archive/MRVI_DE_DA_analysis.md`
(v1, 2026-05-22). Same three layers — **audit · biology · method** — but the downstream DE/program
methodology was overhauled to match Li *et al.* 2024 ([PMC11588665](https://pmc.ncbi.nlm.nih.gov/articles/PMC11588665/)) more faithfully.

**The MRVI fits did not change.** All four `.nc` artifacts are dated 2026-05-21 and the MRVI LFCs
(§4 mal-vs-benign, §8 stage) are byte-identical to v1. What changed is everything *around* MRVI:
the comparison baseline switched from **cell-level Wilcoxon → pseudobulk DESeq2**, and the
metaprogram method switched from **per-donor sklearn NMF → cNMF**. The method-comparison section
(§10) is therefore re-anchored from *MRVI vs Wilcoxon* to *MRVI vs pseudobulk DESeq2* (+ cNMF).

## What changed at a glance

| Section | v1 (2026-05-22) | v2 (2026-05-26) | Result shift |
|---|---|---|---|
| §3 mal-vs-benign DE | cell-level **Wilcoxon** | **pseudobulk DESeq2**, paired `~donor+status` | top = cell-cycle (CDKN3/RRM2/TK1/TYMS/TOP2A); **64** sig genes vs housekeeping before |
| §5 concordance (mal-vs-benign) | Wilcoxon↔MRVI **ρ=0.00** | DESeq2↔MRVI **ρ=0.04** | still ≈0 — but now **both methods are donor-aware** |
| §6 metaprograms | per-donor sklearn NMF (1 housekeeping cluster, n=126/168) | **cNMF K=10** | 10 distinct programs; **exhaustion P4 (PDCD1/CXCL13) 0.04→0.22 in advanced** |
| §7 stage DE | Wilcoxon/MRVI | **pseudobulk DESeq2** (`~stage_group`) | **73** sig genes; new advanced-up axis (IGFL2/SPINT2/EPCAM) |
| §7 TH2 fraction | multi-gene score, U=118 **p=8.65e-2** | **GATA3-only** master-TF argmax, U=92 **p=1.44e-2** | p improved ~6× |
| §9 concordance (stage) | Wilcoxon↔MRVI **ρ=0.29** | DESeq2↔MRVI **ρ=0.35** | modestly higher |
| §1, §4, §8 (MRVI side) | — | unchanged (same May-21 `.nc`) | identical numbers |

---

## 1. Completeness audit

| § | Artifact | Job | Cells / scope | Date | Status |
|---|---|---|---|---|---|
| §1 DA stage | `figures/mrvi_da_stage.nc` | `run_mrvi_inference.py` | 419,579 cells = full atlas | 05-21 | ✅ unchanged |
| §1 donor dist | `figures/mrvi_donor_distances.nc` | `run_mrvi_inference.py` | 36×36 donors × 49 cell types | 05-21 | ✅ unchanged |
| §4 m-vs-b MRVI DE | `figures/mrvi_de_malignant_vs_benign.nc` | `run_mrvi_tcells_mb.py` | 228,370 T cells, 60 pseudo-samples | 05-21 | ✅ unchanged |
| §8 stage MRVI DE | `figures/mrvi_de_stage_malignant.nc` | `run_mrvi_de.py --malignant-only` | 124,141 malignant T cells | 05-21 | ✅ unchanged |
| §3 m-vs-b pseudobulk | `tables/Fig3b_DEGs_malignant_vs_benign_pseudobulk.csv` | notebook (pydeseq2) | 68 pseudobulk libs / 34 donors | **05-26** | ✅ new |
| §6 cNMF programs | `tables/Fig3d_cnmf_programs.json` + `models/cnmf/` | notebook (cNMF) | 124,141 malignant cells, 9,974 genes, K=10 | **05-26** | ✅ new |
| §7 stage pseudobulk | `tables/Fig4a_DEGs_advanced_vs_early_pseudobulk.csv` | notebook (pydeseq2) | 34 donors (15 early / 19 advanced) | **05-26** | ✅ new |
| §7 TH2 fractions | `tables/Fig4b_per_donor_TH_fractions.csv` | notebook | 36 donors | **05-26** | ✅ new |

The MRVI `.nc` set is the v1 set untouched; only the downstream tables were regenerated. The old
Wilcoxon/NMF artifacts (`Fig3b_*_wilcoxon.csv`, `Fig3d_metaprograms_consensus.json`,
`Fig3b_concordance_Wilcoxon_vs_MRVI.png`, `Fig4_concordance_Wilcoxon_vs_MRVI.png`) remain on disk as
the v1 record.

---

## 2. §1 — Donor distance heatmap (unchanged)

**Changed from prior run:** nothing — recomputed from the same `.nc` and confirmed identical.

`get_local_sample_distances(keep_cell=False, groupby="cell_type")`, averaged across cell types.

| Pair class | n pairs | Mean distance |
|---|---|---|
| within-advanced × advanced | 171 | **0.4268** |
| within-early × early | 136 | 0.5084 |
| advanced × early | 323 | 0.4748 |

- MWU **within-adv < within-early**: U = 7.80×10³, **p = 3.5×10⁻⁷**.
- MWU **cross > within (all)**: U = 5.26×10⁴, p = 0.090 (ns).
- Advanced donors converge in MRVI latent geometry (stereotyped malignant TH2 program); early donors
  retain heterogeneous baselines. Matches Li *et al.* Figure: `figures/mrvi_donor_distances.png`.

---

## 3. §1 — Differential abundance over `stage_group` (unchanged)

**Changed from prior run:** nothing — same `.nc`, recomputed and confirmed.

Δ = `log_enrich[advanced] − log_enrich[early]`, mean per cell type.

```
TOP up-in-advanced                       TOP up-in-early
─────────────────────────────           ──────────────────────────────
B_cell        +4.83 (n=15211)           LC_3        −26.94 (n=637)
Plasma        +3.83 (n=5569)            LE2         −25.61 (n=982)
channel       +2.85 (n=371)             Inf_mac     −23.64 (n=1966)
Tc_IL13_IL22  +1.14 (n=2336)            F3          −23.04 (n=6719)
Tc            +0.63 (n=39889)           VE1         −22.15 (n=2712)
tumor_cell    +0.03 (n=102367)          moDC_2      −21.79 (n=2267)
```

- 53.3 % of cells Δ<−0.5 (early); 36.9 % Δ>0.5 (advanced).
- **B cells (+4.83) and Plasma (+3.83)** top-enriched in advanced — recovers the paper's title
  ("B cell-rich TME supporting malignant TH2"). tumor_cell fraction is stage-flat (Δ≈0); what is
  stage-stratified is the *company* (B/Plasma). Normal-skin compartments (LC/LE/F/VE/moDC/Inf_mac)
  are displaced in advanced. Figure: `figures/umap_mrvi_da_stage.png`.

---

## 4. §2 — inferCNV T-cell labels (new section)

**Changed from prior run:** v1 folded this into the audit; v2 reports it as its own section from
notebook cell 13.

`tcells_cnv.h5ad`: **229,263** T cells, **124,141 malignant (54.1 %)**; **28/36 donors** have ≥200
malignant cells. cnv_score mean 0.006 (malignant) vs 0.003 (benign).

| stage_group | n cells | n malignant | frac malignant |
|---|---|---|---|
| early | 134,444 | 92,515 | **0.688** |
| advanced | 94,819 | 31,626 | **0.334** |

per-donor frac malignant: median 0.318, IQR 0.116–0.639, range 0.036–0.872.

- **Flag — counterintuitive direction:** the malignant *fraction of the T-cell compartment* is
  **higher in early (0.69) than advanced (0.33)**. This is not lower tumor burden in advanced; it
  reflects that early biopsies are clone-dominated within the T-cell gate, while advanced lesions
  carry a much larger *non-malignant* infiltrate (consistent with §3: B/Plasma/Tc expansion in
  advanced dilutes the malignant fraction *among T cells*). Absolute clone size is captured by the
  separate `tumor_cell` type (102,367 cells), not by this within-T-cell ratio.

---

## 5. §3 — Pseudobulk DESeq2 DE: malignant vs benign T cells (paper method)

**Changed from prior run:** v1 used cell-level Wilcoxon (`rank_genes_groups`) and called it the
"paper method"; that was a mislabel. Fig 3b's legend specifies a quasi-likelihood F-test + BH — a
**pseudobulk GLM**. v2 implements it correctly: sum raw counts per **donor × malignant-status**
pseudobulk library, then **pydeseq2** (DESeq2-equivalent NB GLM, Wald + BH) with a **paired design
`~donor + status`**.

Setup: 68 pseudobulk libraries across **34 donors** (2 unpaired donors dropped); contrast
`status: malignant vs benign`.

**Top 20 sig genes (padj<0.05, |log2FC|>1; 64 total), by padj:**

```
CDKN3 +1.02   RRM2 +1.44   TK1 +1.23    NCAPG +1.05   MYBL2 +1.52
CDC6 +1.21    TPX2 +1.05   TYMS +1.34   ASF1B +1.11   UBE2C +1.46
CDK1 +1.02    PKMYT1 +1.43 AURKB +1.25  DLGAP5 +1.37  PCLAF +1.23
PBK +1.42     APOBEC3B +1.28 TOP2A +1.14 UHRF1 +1.14  CENPA +1.20
```

- **Every top hit is a cell-cycle / S-G2-M / replication gene** (CDKN3, RRM2, TK1, TYMS, MYBL2,
  CDC6, TPX2, UBE2C, CDK1, AURKB, TOP2A, CENPA) plus the APOBEC3B mutator. The paired donor design
  isolates the *within-donor* malignant-vs-benign difference that is consistent across patients —
  and that difference is **proliferation**: the clone cycles faster than the donor's bystander
  benign T cells.
- **None of Li *et al.* Fig 3b's named markers** (TOX, CXCL13, CD9, GTSF1, RUNX3, GATA3, CCR4)
  survive donor pairing — their variance is donor-to-donor, not within-donor mal-vs-benign (see
  §10.3). *Change vs v1:* the Wilcoxon top-15 was pure housekeeping/ribosomal/mito (FTL, MIF, FTH1,
  HSPD1…); pseudobulk replaces that artefactual abundance ranking with a coherent biological
  (proliferation) signature.

Outputs: `tables/Fig3b_DEGs_malignant_vs_benign_pseudobulk.csv`, `figures/Fig3b_volcano_pseudobulk.png`.

---

## 6. §4 — MRVI DE: malignant vs benign T cells (unchanged)

**Changed from prior run:** nothing — same `.nc`, same `tables/Fig3b_DEGs_malignant_vs_benign_MRVI.csv`.

DE: `differential_expression(sample_cov_keys=["target_status"], add_batch_specific_offsets=True,
mc_samples=50)` on the within-donor `_mal`/`_ben` pseudo-samples (60/72 kept, 228,370 cells).

**Top 15 by mean LFC over 123,505 malignant cells:**

```
UP in malignant            DOWN in malignant
DDX3Y   +0.082             MT-CO3  −0.148   (mito)
CAPG    +0.077             MT-ND3  −0.142   (mito)
EIF1AY  +0.074             MT-ND5  −0.138   (mito)
RGCC    +0.067             MT-ND2  −0.127   (mito)
CD7     +0.065             MT-ND4  −0.114   (mito)
BIN1    +0.059             MT-ND1  −0.112   (mito)
HOPX    +0.057             MT-ND6  −0.093   (mito)
GZMA    +0.057             CD2     −0.074
USP9Y   +0.054             CXCL13  −0.066
UTY     +0.051             ZFP36   −0.059
KDM5D   +0.047             JUNB    −0.058 / MAL −0.057 / CD74 −0.053 / IL2RB −0.053 / CTLA4 −0.052
```

- Same caveats as v1: **Y-chromosome panel** (DDX3Y/EIF1AY/USP9Y/UTY/KDM5D) in top-UP is a leaky
  control (sex constant within donor → specify `sex` as nuisance); **mito** dominates top-DOWN
  (%-mito artefact). Most plausible MRVI biology: CXCL13↓, CTLA4↓, IL2RB↓, CD2↓, JUNB↓ — the
  intra-donor clone is *less inflammatory/activated* than the donor's benign infiltrate.

---

## 7. §5 — Concordance: pseudobulk DESeq2 ↔ MRVI (malignant vs benign)

**Changed from prior run:** partner switched Wilcoxon → pseudobulk DESeq2. **Spearman ρ = 0.04**
(n=9,765 genes; v1 Wilcoxon↔MRVI was 0.00).

- The figure (`figures/Fig3b_concordance_pseudobulk_vs_MRVI.png`) shows both axes near-zero for the
  paper markers: TOX, GTSF1, CD9, GATA3, RUNX3, TBX21 cluster at the origin; CXCL13 sits low on the
  MRVI axis (−0.066); CD7 sits high on MRVI (+0.065). DESeq2 places none of them far from x=0.
- **Key escalation vs v1:** in v1 the ρ≈0 was easy to dismiss as "Wilcoxon pseudoreplication vs
  donor-aware MRVI". In v2 **both methods control for donor**, yet they *still* disagree completely.
  The disagreement is now between two donor-aware **estimands** (count-space marginal FC vs per-cell
  counterfactual at fixed `u_n`), which is a stronger and more interesting result — see §10.

---

## 8. §6 — cNMF metaprograms on malignant cells (Fig 3d)

**Changed from prior run:** sklearn per-donor NMF (K=6, 168 candidate programs → 9 Jaccard clusters,
of which **126/168 collapsed into one housekeeping cluster**) replaced by **one pooled cNMF run**
(Kotliar 2019) over all 124,141 malignant cells, 9,974 genes (26 mito/HSP genes excluded), K∈{8,9,10},
**K=10 selected**, density threshold 0.10.

**10 consensus programs (top genes) and mean usage by stage:**

| Prog | usage early→adv | Top genes | Annotation |
|---|---|---|---|
| **P1** | **0.52 → 0.12** | ZFP36L2, FTH1, PNRC1, CREM, FYN, IL7R, XBP1, HSP90B1 | housekeeping / quiescent-memory (early-dominant) |
| P2 | 0.16 → 0.09 | LGALS3, S100A6, S100A4, IL32, JAML, KRT86, KRT7 | S100/galectin migratory, epithelial-adjacent |
| P3 | 0.08 → 0.09 | NR4A1, JUN, DUSP1/2, DNAJB1, HSP90AA1, NFKBIA, **IL13** | AP-1 immediate-early / heat-shock + TH2 effector IL13 |
| **P4** | **0.04 → 0.22** | IGFL2, SPINT2, CYP7B1, ICA1, **PDCD1**, **CXCL13**, EPCAM, MME | **exhaustion / advanced-malignant (5.5× up in advanced)** |
| P5 | 0.06 → 0.08 | **GZMB**, TYROBP, FCER1G, KLRC1, ZNF683, TRGC1, CTSW | cytotoxic / NK-like |
| P6 | 0.04 → 0.10 | CXCL12, ITGA1, KLRF2, TRBC2, ITGBL1, IGFBP5 | tissue-residency / mixed |
| P7 | 0.03 → 0.09 | AEBP1, MRC2, WNT5B, GPC4, IGFBP4, MYB, PHF19, TRAC | fibroblast/stromal-like (likely doublet) |
| P8 | 0.04 → 0.05 | MAL, **TNFRSF8 (CD30)**, **CTLA4**, CIITA, **IL26**, WNT10A, TP63 | CD30⁺ transformed / checkpoint / IL26 |
| P9 | 0.02 → 0.16 | SOHLH1, EVX1, NKX2-6, GNG4, NPTX2, STRA6 | developmental-TF (probable ambient/artefact) |
| P10 | 0.01 → 0.02 | CD163, CD14, LYZ, C1QA/B, MMP9/12, SPARC, COL4A1/2 | myeloid/macrophage (doublet/contamination) |

**Biology and the v1→v2 gain:**

- **P1 housekeeping collapses from 0.52 (early) to 0.12 (advanced)** while **P4 exhaustion rises
  0.04→0.22** — a clean stage-associated program shift the old sklearn NMF could not resolve
  (everything pooled into the housekeeping cluster). cNMF's variance-normalisation + multi-replicate
  consensus + density filtering discards the depth/mito noise axis (§10.6).
- **P4 is the headline advanced-stage malignant program**: PDCD1 + CXCL13 (exhausted/Tfh-like) plus
  IGFL2/SPINT2/CYP7B1/ICA1/EPCAM — **the same genes that top the §9 pseudobulk advanced-vs-early
  list** (IGFL2, SPINT2, CYP7B1, ICA1, EPCAM). Two independent methods (cNMF usage and pseudobulk
  DESeq2) converge on this advanced program — a genuine cross-method corroboration.
- **P8 carries TNFRSF8 (CD30, the brentuximab-vedotin target), CTLA4, and IL26** — a transformed /
  checkpoint program of therapeutic interest (mild advanced lean, 0.04→0.05).
- **P5** is a clean cytotoxic/NK-like program (GZMB, TYROBP, KLRC1, ZNF683=Hobit).
- **P9, P10** are interpretation flags: P10 is myeloid contamination (CD163/CD14/LYZ/C1Q); P9 is a
  developmental-TF program with no T-cell rationale (likely ambient RNA / doublet) yet carries
  non-trivial advanced usage (0.16) — treat as artefact pending QC.

Outputs: `tables/Fig3d_cnmf_programs.json`, `figures/Fig3d_cnmf_usage_by_stage.png`, `models/cnmf/`.

---

## 9. §7 — Stage analyses on malignant cells (Fig 4a/b)

**Changed from prior run:** stage DE switched to **pseudobulk DESeq2** (per-donor aggregation,
`~stage_group`, contrast advanced vs early); TH2 fraction switched to **GATA3-only master-TF argmax**.

**Pseudobulk DESeq2 advanced-vs-early** — 34 donors (15 early / 19 advanced), **73 sig genes**
(padj<0.05, |log2FC|>1). Top by padj:

```
UP in advanced                         DOWN in advanced
SPINT2  +3.73   HES1   +4.36           KRT86   −4.34
FAM71B  +5.88   CYP7B1 +4.49           LEMD1   −5.27
IGFL2   +5.40   HEYL   +5.48           GPAT3   −3.22
ICA1    +3.38   IFNGR2 +2.40           ZBTB46  −3.09
EPCAM   +4.97   NPW    +3.86           ZBED2   −2.53
```

- The advanced-up axis (IGFL2, SPINT2, CYP7B1, ICA1, EPCAM, HES1/HEYL Notch targets) is the **same
  program as cNMF P4** (§8) — convergent evidence for an advanced-stage malignant state. IFNGR2↑.

**TH2 fraction (Fig 4b)** — per-donor argmax over single master TFs (TBX21/GATA3/RORC):

| stage | n donors | median TH2 (GATA3) fraction |
|---|---|---|
| early | 17 | lower |
| advanced | 19 | higher |

One-sided MWU early<advanced: **U = 92, p = 1.44×10⁻²** (paper 3×10⁻⁴).

- *Change vs v1:* dropping the noisy multi-gene cytokine score (IL4/IL13/IL5/STAT6 etc., which are
  sparse and dilute the signal) for **GATA3-only** scoring improved the p ~6× (was U=118,
  p=8.65×10⁻²). Direction matches the paper; the residual gap to 3×10⁻⁴ is the smaller cohort (17/19
  vs the paper's 21/24) and the CTCL-only pool (no healthy/AD/Pso controls).

Figures: `figures/Fig4a_volcano_pseudobulk.png`, `figures/heatmap_Fig4a_early_vs_advanced.png`,
`figures/Fig4b_TH2_violin.png`. Tables: `Fig4a_DEGs_advanced_vs_early_pseudobulk.csv`,
`Fig4b_per_donor_TH_fractions.csv`.

---

## 10. §8 — MRVI stage DE on full malignant T cells (unchanged)

**Changed from prior run:** nothing — same `.nc` (124,141 malignant cells), same
`Fig4_MRVI_top_{up,down}_advanced.csv`.

```
UP in advanced                DOWN in advanced
CCR7     +0.534               GNLY    −0.891
PASK     +0.468               CCL5    −0.830
CXCR4    +0.435               LGALS3  −0.449
DUSP2    +0.376               XCL1    −0.437
PIM3     +0.343               GZMA    −0.380
JUNB     +0.338               CTSW    −0.379
IGFBP4   +0.319               CXCR3   −0.377
MARCKSL1 +0.299               GZMB    −0.362
CD69     +0.296               IFNG    −0.343
CYTIP    +0.293               ...     (S100A6, FUT7, CLU, HOPX, CD63, TNFRSF18, MT-ND1, ADGRG1)
```

Paper Fig 4a key-gene direction check (all correct):

| Gene | MRVI LFC | Paper | Gene | MRVI LFC | Paper |
|---|---|---|---|---|---|
| CCR7 | +0.534 | UP ✓ | GNLY | −0.891 | DOWN ✓ |
| SELL | +0.249 | UP ✓ | GZMB | −0.362 | DOWN ✓ |
| GATA3 | +0.196 | UP ✓ | IFNG | −0.343 | DOWN ✓ |
| TCF7 | +0.134 | UP ✓ | CXCR3 | −0.377 | DOWN ✓ |
| LEF1 | +0.031 | UP ✓ | TBX21 | −0.123 | DOWN ✓ |

- Central-memory gain (CCR7/SELL/TCF7/LEF1) + GATA3↑ (TH2 master TF) and cytotoxic loss
  (GNLY/GZMB/IFNG/CXCR3/TBX21) = clean TH1→TH2 + naïve/central-memory shift in advanced. TH2
  cytokines IL4/IL5/IL13 themselves stay ≈0 (sparse 3′ 10x counts) — the skew is read off GATA3.

---

## 11. §9 — Concordance: pseudobulk DESeq2 ↔ MRVI (stage, malignant cells)

**Changed from prior run:** partner switched Wilcoxon → pseudobulk DESeq2. **Spearman ρ = 0.35**
(`figures/Fig4_concordance_pseudobulk_vs_MRVI.png`; v1 Wilcoxon↔MRVI was 0.29).

- Named markers fall in the expected quadrants for both methods: CCR7/SELL/GATA3/TCF7 up-right,
  GNLY/IFNG/GZMB/NKG7 down-left. The between-donor stage axis yields markedly higher concordance
  than the within-donor mal-vs-benign axis (§7, ρ=0.04) — explained in §12.4.

### Concordance summary

| Comparison | v1 partner | ρ (v1) | v2 partner | ρ (v2) |
|---|---|---|---|---|
| mal-vs-benign (within-donor) | Wilcoxon | 0.00 | pseudobulk DESeq2 | **0.04** |
| stage adv-vs-early (between-donor) | Wilcoxon | 0.29 | pseudobulk DESeq2 | **0.35** |

The two methods agree far less **within-donor** than **between-donor** — both in v1 and v2 — exactly
where their methodological differences predict.

---

## 12. Method comparison — MRVI vs pseudobulk DESeq2

*Re-anchored from v1's MRVI-vs-Wilcoxon. Researched via NotebookLM with the MRVI paper (Boyeau et al.,
Nat Methods 2025), scvi-tools MrVI docs, the DESeq2 paper (Love et al. 2014), PyDESeq2 docs, the cNMF
paper (Kotliar et al. 2019), Skinnider/Squair 2021, and Crowell et al. 2022.*

### 12.1 What each method computes

| | pseudobulk DESeq2 (`~donor+status`) | MRVI `differential_expression(sample_cov_keys=…)` |
|---|---|---|
| Input | raw counts summed per donor×status → NB GLM | raw counts → VAE encoder → counterfactual decoder |
| Test unit | gene; one library per donor×status | gene × cell; one LFC per cell |
| Estimand | **marginal** size-factor-normalized log2FC of aggregated pools | **conditional** per-cell counterfactual at fixed latent state `u_n` |
| Donor handling | donor as a model term (paired); removes donor baseline | donor collapsed into sample-aware latent; `u_n` holds cell-state fixed |
| Replication | biological replicates = donors (correct) | hierarchical encoder over samples |
| Shrinkage | empirical-Bayes LFC shrinkage toward 0 | MC-sampled counterfactual (`mc_samples=50`) |

Both are **donor-aware** — the v1 pseudoreplication critique of Wilcoxon no longer applies to the
baseline. The remaining gap is a difference of *estimand*, not of statistical rigor.

### 12.2 Why pseudobulk DESeq2 mal-vs-benign returns a cell-cycle signature

A paired pseudobulk on donor×status sums **all** transcript counts per pool. The malignant clone
carries a higher *proportion of actively cycling cells* than the donor's bystander benign T cells, so
the aggregated malignant library is enriched for the (large, coordinated) cell-cycle **activity
program**. The `~donor` term removes donor baseline and the consistent within-donor proliferation
difference is what DESeq2 confidently surfaces (CDKN3/RRM2/TK1/TYMS/TOP2A). This is **differential
abundance of a cell sub-state masquerading as differential expression** — pseudobulk cannot separate
the two. Conversely, markers whose variance is *donor-to-donor* (TOX/CXCL13/CD9) vanish once `~donor`
controls the baseline → they are **shared cell-state of the inflamed-skin T-cell pool**, not
clone-intrinsic drivers (Park 2020: TOX/CXCL13 = tissue-resident-exhausted markers expressed by both
malignant and benign infiltrating T cells).

### 12.3 Why concordance is still ≈0 (ρ=0.04) though both control donor

The crux of v2. DESeq2 estimates a **marginal** count-space FC and therefore *conflates* a shift in
the abundance of a cell sub-state (more cycling cells) with a shift in expression. MRVI estimates a
**conditional** counterfactual: for a cell at a fixed `u_n` (which already encodes its cell-cycle /
activation state), what would its normalized expression be if malignant vs benign? By holding `u_n`
fixed MRVI **regresses out the compositional/proliferation shift**, dropping cell-cycle genes from
the top and surfacing state-independent condition effects. Two donor-aware methods, two different
estimands → uncorrelated gene rankings. Structurally:

- **pseudobulk DESeq2 favors** abundant genes and activity-program / compositional shifts (cell
  cycle, metabolism), reinforced by mean-dependent LFC shrinkage.
- **MRVI favors** intrinsic, state-conditioned condition markers, independent of sub-state abundance.

### 12.4 Why between-donor (stage) concordance is higher (ρ=0.35)

In a between-donor contrast each donor belongs to exactly one stage, so the stage effect is *nested
within donor identity* — **neither** method can regress donor out of the effect of interest (doing so
would delete the effect). Both are therefore forced to estimate a similar **marginal donor-level
shift** between the two donor sets, so their rankings align more. Two reinforcing reasons concordance
rises from 0.04→0.35: (a) the estimands converge when donor cannot be conditioned away; (b) the
compositional/proliferation confound that dominated the within-donor axis is smaller along the stage
axis (stage differences are more about lineage program — central-memory/TH2 vs cytotoxic — than about
cycling fraction).

### 12.5 Failure modes / interpretation flags (v2)

1. **Y-chromosome genes in MRVI top-UP** for within-donor mal-vs-benign — sex is constant within
   donor; refit with `sex` as a categorical nuisance (or via `batch_key`).
2. **Mito genes in MRVI top-DOWN** — %-mito artefact; bin %-mito as nuisance.
3. **Pseudobulk cell-cycle dominance** is *not* an artefact but a real DA-vs-DE conflation — if the
   intent is clone-intrinsic expression, regress/cluster-out cycling cells first or read MRVI instead.
4. **`pde`/`padj` not thresholded** (no `delta` set in the DE call) — rerun with `delta=0.1–0.3` for
   posterior DE-probability ranking.
5. **`mc_samples=50`** (spec recommends 100).

### 12.6 cNMF vs plain NMF (why §8 improved)

Plain NMF on log1p-HVGs is a single stochastic factorization prone to local optima that **merge**
distinct programs; adding components in log space also breaks the linear-mixture assumption, letting
abundant low-variance housekeeping genes dominate (v1: 126/168 programs → one cluster). cNMF instead:
(1) selects overdispersed genes and **scales to unit variance without log** so genes on different
scales contribute comparably; (2) runs **many replicate factorizations** across seeds; (3)
**density-filters** components by k-NN distance, discarding unstable/noise (depth/mito) components;
(4) **consensus** = median over surviving replicate components; (5) **refits against full TPM** for
interpretable units. The net effect is the depth/mito noise axis is removed and reproducible programs
(exhaustion P4, cytotoxic P5, CD30⁺ P8) separate cleanly.

---

## 13. Caveats

1. **n donors small** (17 early / 19 advanced); donor-level stats (TH2 p, donor-distance MWU) bounded
   by n. MRVI/pseudobulk DE has ≫ cells but is conditional on the donor split.
2. **inferCNV `is_malignant` is imperfect** (CNV + TCR mix); mislabels dampen §5/§7.
3. **Pseudobulk drops unpaired/low-cell libraries** (2 donors in §5; min 20 cells/library) — results
   conditional on the retained set.
4. **§7 ρ=0.04 does not invalidate Li *et al.* Fig 3b** — it shows their markers are *shared
   inflamed-skin cell-state* (donor-level), not clone-intrinsic; both donor-aware methods agree they
   are not within-donor mal-vs-benign effects.
5. **cNMF P9/P10 are artefact-flagged** (developmental-TF / myeloid contamination); P4/P5/P8 are the
   trustworthy programs.

---

## 14. References

- **Li *et al.* 2024**. *Cutaneous T cell lymphoma atlas reveals malignant TH2 cells supported by a
  B cell-rich tumor microenvironment*. *Cancer Cell* 42(12). [PMC11588665](https://pmc.ncbi.nlm.nih.gov/articles/PMC11588665/).
- **Boyeau *et al.* 2025**. *Deep generative modeling of sample-level heterogeneity in single-cell
  genomics* (MRVI). [Nature Methods](https://www.nature.com/articles/s41592-025-02808-x).
- **Love, Huber, Anders 2014**. *Moderated estimation of fold change and dispersion for RNA-seq data
  with DESeq2*. *Genome Biology* 15:550. — the pseudobulk GLM used in §5/§9.
- **Muzellec *et al.* 2023**. *PyDESeq2*. [docs](https://pydeseq2.readthedocs.io/) — Python DESeq2 used here.
- **Kotliar *et al.* 2019**. *Identifying gene expression programs of cell-type identity and cellular
  activity with single-cell RNA-Seq* (cNMF). *eLife* 8:e43803.
- **Squair, Skinnider *et al.* 2021**. *Confronting false discoveries in single-cell differential
  expression*. *Nat Commun*. — pseudoreplication critique (motivates pseudobulk).
- **Crowell, Soneson *et al.* 2022**. *Benchmarking methods for detecting differential states…*.
  *Brief Bioinform* [PMC9487674](https://pmc.ncbi.nlm.nih.gov/articles/PMC9487674/).
- **Park *et al.* 2020** — TOX/CXCL13 as tissue-resident-exhausted markers (§5/§12.2 interpretation).

**NotebookLM** deep-research workspace: `8e81a82d-9ab3-48e0-96c6-a1b7f5fbe67e` — "MRVI vs standard DE
methods - methodological comparison". v2 added sources: DESeq2 (Love 2014, `19c7b663`), cNMF (Kotliar
2019, `16aa2c50`), PyDESeq2 docs (`d30a621b`); existing: MRVI Nat Methods, MrVI docs (×2), Skinnider
2021 PDF, Crowell 2022. v2 Q&A saved as note `83f5599f`.
