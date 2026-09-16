# `old/23_subclonal_evolution` — Subclonal (divergent-evolution) analysis of MF malignant clones

Companion doc to `23_subclonal_evolution.ipynb`. Explains the question, inputs, per-section
method, and the actual results (numbers quoted from the notebook's cached cell outputs). Method
details reference `subclone_helpers.py` (imported as `S`).

> **⚠️ Refresh pending.** §0 was revised to use the canonical annotations: malignancy = **TCR-ALICE
> only** (`tcr_malignant_alice`); the **§2 transcriptional latent is now `X_mrvi_u`** (was `X_pca`);
> the **§3 diploid baseline is cleaned with `old/14_skin_T_tcr_cnv_malignancy`'s CNV call** (`cnv_malig_cluster`); and donor
> eligibility is tabulated with an explicit exclusion reason. Quantitative results in §4 (§2
> major/minor), §5 (§3 baseline count), and downstream sections below still reflect the **prior
> `X_pca` / uncleaned-baseline run** and will change on the next GPU re-run — treat those numbers as
> stale until refreshed.

---

## 1. Overview & question

**Divergent evolution** = a single malignant clone diversifies into genomically distinct
**subclones** that then follow different transcriptional / functional trajectories. This notebook
operationalizes the framework of **Herrmann / Iyer et al., *Cancer Discovery* 2025** —
*"Divergent Evolution of Malignant Subclones … in T-cell Cancer"* (L-CTCL / Sézary; PMC12498100) —
and applies it to the **MF (Mycosis Fungoides) / CTCL atlas**.

It **scales `old/20_skin_clone_cnv_relationship`'s Q1b prototype** (CNV subclones for the top-3 donors) to the **whole eligible
cohort** and asks, per patient:

1. Does the malignant clone carry **≥2 CNV subclones**? (paper: 84% of L-CTCL patients do)
2. Are subclones **major** (own transcriptional cluster) or **minor** (CNV-distinct but intermixed)?
3. Which copy-number events are **trunk** (shared ancestor) vs **branch** (private divergence edges)?
4. **The MF-specific centerpiece (§4):** L-CTCL subclones share ONE TCR, but MF *can* carry
   different TCRs. Do our CNV subclones **split by TCR-β variant** (MF branched-TCR evolution) or
   **share the founder TCR** (the paper's L-CTCL model)?
5. How do subclones **diverge** — DE/GSEA, functional programs (metabolism, proliferation,
   cytokine, Th, homing), and epidermis-vs-dermis tropism?
6. **Cohort landscape (§8):** prevalence, #subclones, recurrent trunk/branch arms, stage association.

Runs **CPU-light off the cached arm-CNV matrix** — no inferCNV recompute (all eligible donors are
already cached). **Out of scope** (no data in this object): CITE-seq surface gating, WGS/WES,
drug / *S. aureus* assays, Numbat haplotype phasing.

---

## 2. Data & inputs (§0)

| Input | File | Content |
|---|---|---|
| Malignant compartment | `data/atlas_joint/skin_T_tcr_malig_v2.h5ad` | `old/21_tcr_alice_neighborhood` object, **242,959 cells × 49,683 genes**; carries `tcr_malignant_alice`, `X_mrvi_u` |
| Arm-CNV matrix | `skin_T_arm_cnv_res0.5.parquet` | cached per-cell **41-arm** inferCNV matrix (`old/20_skin_clone_cnv_relationship`) |
| CNV malignancy call | `skin_T_malignancy.parquet` | `old/14_skin_T_tcr_cnv_malignancy` final per-cell CNV call `cnv_malig_cluster` (bool); attached in §0 for the §3 baseline + §1 QC |
| Gene sets | `../lib1_immune.gmt` | hallmark + C2 (for GSEA) |

- **Malignant compartment** = the `tcr_malignant_alice` cells (`old/21_tcr_alice_neighborhood`: dominant TCR founder + ALICE
  ≤1-aa β-variant family) — i.e. cells sharing one clonal TCR, matching the paper's definition.
  **Malignancy is assigned by TCR-ALICE only**; `old/14_skin_T_tcr_cnv_malignancy`'s `cnv_malig_cluster` is attached but is used
  *only* for CNV-side work (the §3 diploid-baseline hygiene and a §1 TCR/CNV concordance QC), never to
  define the compartment.
- On load, `X` and the `counts` layer are freed (cut ~6 GB → ~2 GB; `X` is rebuilt from
  `raw_counts` in §5). ⚠️ ~6 GB load — GPU/compute kernel only, never the login node.
- **Sample selection (requirement a).** The loaded object is already the `old/21_tcr_alice_neighborhood` canonical cohort — the
  full atlas was reduced to **34 donors** upstream (`old/21_tcr_alice_neighborhood` dropped `D1__P303` as a duplicate of
  `D5__MFIVB`, the `MF_gamma_delta` / `CD8_aggressive_epidermotropic_CTCL` non-αβ-CD4 entities, and
  samples with `<300` TCR cells, herrera exempt). `old/23_subclonal_evolution` adds one gate: **≥ `MIN_MAL=200` malignant
  cells/donor** → **24 eligible / 10 excluded**. Every donor is written to
  `subclone_donor_eligibility_v2.csv` with a per-donor `reason_excluded`:
  - **3 HC** (`D1__P115`, `D1__P116`, `H__HC1`) — healthy controls, no malignant compartment;
  - **`D3__P94`** — 0 ALICE-malignant cells (no dominant TCR clone);
  - **`D3__P105` (66)** + **5 herrera** (`H__MFIV1`/`SS1`/`SS2`/`SS3`/`SS4`, 15–176) — `<200`
    malignant cells, too few for subclone KMeans (`MIN_SUB=50`, up to `K_MAX=4`).

  > **Herrera coverage (not a truncation).** The tiny Herrera per-donor counts here are *skin-only
  > αβ-CD4-T* counts, **not** a failed import. Herrera (GSE171811) is ingested in full — 37,819 raw
  > → 32,002 after QC (84.6%), all 8 donors — but it is a **blood-dominant leukemic MF/SS** ECCITE
  > cohort with small skin biopsies. The skin atlas (`compartment=="Skin"`, `old/13_skin_lowres_annotation`) intentionally drops
  > its ~32k blood cells, which is where its thousands of malignant cells live (SS1 1624, MFIV1 1248,
  > SS6 1023, SS3 1007, SS5 749 in blood — see `sample_metadata.csv`). Two facts are irreducible:
  > **SS5/SS6 are blood-only** (never enter a skin atlas) and **HC1's skin biopsy is 50 cells at
  > source**. So these donors are correctly small for a skin-restricted analysis; recovering them
  > would require folding in Herrera's blood compartment (a scientific choice, deferred). Also note
  > Herrera SS donors are Sézary (**disease=`SS`**), corrected from an upstream `MF` mislabel.

  All 24 eligible donors are present in the arm cache (no heavy recompute).

**Key knobs**

| Knob | Value | Meaning |
|---|---|---|
| `MIN_MAL` | 200 | min malignant cells/donor to attempt subclone detection |
| `K_MAX` | 4 | max subclones/donor (paper median 3) |
| `MIN_SUB` | 50 | min cells per subclone |
| `SIL_MIN` | 0.15 | min silhouette to accept a k>1 split |
| `MIN_ARM_DELTA` | 0.03 | a split must differ by ≥ this on ≥1 arm (CNA-scale, not state noise) |
| `Z_THR`, `EPS` | 2.5, 0.01 | arm gain/loss call vs benign baseline (MAD-z + magnitude floor) |
| `BRANCH_DELTA` | 0.03 | arm counts as branch (private) if subclone centroids differ by ≥ this |
| `AUC_THR`, `SIL_THR` | 0.65, 0.10 | major-vs-minor transcriptional-separation thresholds |
| `LATENT` | `X_mrvi_u` | trained MRVI latent (mrvi_joint_skin, sample-unaware) for §2 transcriptional separation |
| `SEED` | 0 | reproducibility |

**Pre-step — TCR founder + ALICE family (Cell 4).** For each donor, `alice_helpers.clonotype_table`
gives the per-donor TRB clonotypes with an `is_founder` flag (dominant clone); `A.founder_family`
grows the **≤1-aa-substitution connected component** around the founder. This yields, per donor, a
`founder_sets` (seed TRBs) and `family_sets` (founder + ≤1-aa variants) used in §4. To keep these
sets consistent with the `tcr_malignant_alice` call that defined the compartment, the notebook mirrors
`old/21_tcr_alice_neighborhood`'s **`Li2024_atlas__PT35` exception** (seed on the top-2 largest CD4 clones). This step is
**TCR-only** (no CNV call). **~8/30 malignant donors carry ≥1 ALICE ≤1-aa TCR-β variant** *(refresh
pending; PT35 seeding may shift this by one).*

---

## 3. §1 — Per-donor CNV-subclone detection

**Method (`S.detect_subclones`).** For each eligible donor, cluster its malignant cells on the
41-arm CNV matrix with **KMeans**, sweeping **k = 2…4** (`n_init=10`). A k>1 partition is accepted
only if it passes **all three guards**:

- **(a) size** — every cluster ≥ `MIN_SUB` (50) cells;
- **(b) cohesion** — silhouette ≥ `SIL_MIN` (0.15), computed on a ≤2,000-cell subsample (O(n²));
- **(c) CNA-scale separation** — subclone centroids differ on ≥1 arm by ≥ `MIN_ARM_DELTA` (0.03).

Guard (c) is the crucial one: arm-level inferCNV also picks up activation/cycling programs, so a
split whose centroids differ by <0.03 on **every** arm is transcriptional-state noise, not a genuine
copy-number subclone. If no k qualifies → **k=1** ("genomically uniform clone"), which avoids
noise-driven over-splitting. Among passing k's, the **highest-silhouette** k wins.

**Result.**

- **≥2 CNV subclones in 23/24 = 96% of donors** (paper: 84%).
- **Median k among multi-subclone donors = 2** (paper: 3) — arm-level CNV is coarser than the
  paper's focal calls, so it resolves fewer distinct subclones.
- Only **1 donor (Li2024 CTCL6)** is k=1 (genomically uniform).

Per-donor highlights (`n_malig`, k, silhouette, largest-arm centroid delta):

| donor | n_malig | k | silhouette | sizes |
|---|---|---|---|---|
| D1__P204 | 1,070 | 4 | 0.20 | 414,52,449,155 |
| D1__P76 | 5,815 | 4 | 0.33 | 3624,1662,186,343 |
| Li2024 CTCL3 | 8,068 | 4 | 0.30 | 260,5940,638,1230 |
| D3__P192 | 3,363 | 3 | 0.30 | 467,2652,244 |
| D5__MFIVB | 1,081 | 3 | 0.46 | (3 subclones) |
| … most others | — | 2 | 0.16–0.33 | — |
| Li2024 CTCL6 | 3,077 | 1 | NaN | (uniform) |

**Cohort/per-donor figures.** `figures/subclone_arm_profiles_cohort.png` (all subclone × arm mean
profiles, one heatmap) and `figures/subclone_cells_<donor>.png` (per-donor cell × arm heatmap, cells
ordered by subclone, black lines at subclone boundaries) for the **23 multi-subclone donors**.

The largest subclone per donor is tagged `is_dominant_subclone` (paper: dominant ≈75% of the
malignant population).

---

## 4. §2 — Major vs minor (transcriptional separation)

**Method (`S.major_minor`).** In the **`X_mrvi_u`** latent (trained MRVI, sample-unaware), each
subclone is **major** if it separates from the rest — either **1-vs-rest LogisticRegression AUC ≥
0.65** (direction-agnostic: `max(auc, 1-auc)`, class-balanced) **or** mean **silhouette ≥ 0.10**.
Otherwise **minor** (CNV-distinct but transcriptionally intermixed). k=1 → `single`. Subclones are
already CNV-derived (§1), so this step uses no CNV malignancy call.

**Result (prior `X_pca` run — refresh pending): 54 major / 3 minor / 1 single.** Transcriptional
separation was **weak** in practice — AUCs cluster around 0.6–0.9 with mostly near-zero silhouettes —
so the AUC≥0.65 gate labels most subclones "major", but the true signal is faint. The switch to
`X_mrvi_u` (batch/sample-corrected) may sharpen or shift these labels; re-run to refresh.
**Consequence (unchanged):** the §5 cohort DE contrast is deliberately **dominant-vs-minor** (by
size), **not** this major/minor label, because the label is too noisy to define a stable cross-donor
contrast.

---

## 5. §3 — Trunk vs branch CNAs + divergent tree

**Baseline.** A pooled **benign-T diploid baseline** = all non-malignant, non-`tumor_cell` T cells
with an arm-CNV row **that are also not flagged malignant by `old/14_skin_T_tcr_cnv_malignancy`'s CNV call** (`cnv_malig_cluster`).
Excluding the CNV-aberrant-but-TCR-negative cells keeps the diploid reference honest — a `cnv_only`
cell would otherwise inflate the per-arm MAD noise scale and mask real trunk/branch events. Its
per-arm medians sit at ~0 (the inferCNV diploid reference), giving a stable per-arm MAD noise scale.
*(Cell count was 68,836 in the prior run; it drops slightly after the CNV exclusion — refresh
pending.)*

**Method.**
- `S.call_arm_events` — per arm, a robust **MAD-z** of the subclone-mean CNV vs the benign
  distribution (`old/20_skin_clone_cnv_relationship` cell-15 logic); calls **gain (+1) / loss (−1)** only if **|z| > 2.5 AND**
  |subclone mean| > 0.01 (magnitude floor).
- `S.trunk_branch` — **trunk** = arms altered in the **clone as a whole** (shared ancestral events);
  **branch** = arms whose **subclone centroids diverge by ≥ `BRANCH_DELTA`** (0.03) = the private /
  variably-amplified divergence edges. An arm can be both (shared event further amplified in one
  subclone). Rooting trunk in the whole-clone call is robust to a near-diploid "transitional"
  subclone that would otherwise erase an all-subclones-unanimous trunk.
- `S.subclone_linkage` — average-linkage Euclidean hierarchy on arm centroids, **rooted at a diploid
  (all-zero) ancestor**, for a per-donor dendrogram.

**Result.** Trunk arms are dominated by the recurrent CTCL **chr7p+/chr7q+** gain (shared ancestral
event in most donors), with donor-specific additions (e.g. chr17q+, chr21q+, chr18p/q+). Examples:

| donor | k | trunk arms |
|---|---|---|
| D1__P204 | 4 | chr7p+, chr7q+, chr21q+ |
| D1__P76 | 4 | chr7p+, chr7q+ |
| D1__P171 | 2 | chr7p+, chr7q+, chr8q+, chr17p−, chr17q+ |
| Li2024 CTCL1 | 2 | chr7p+, chr7q+, chr13q− |
| D5__MFIVB | 3 | chr7p+ |

**Figures.** `figures/subclone_trees.png` (per-donor dendrograms, first 6 multi-subclone donors,
rooted at diploid, annotated with trunk / branch arms). Known CTCL loci are mapped for annotation:
chr8q→MYC, chr17q→STAT3/5, chr7p/q→chr7-MF, chr17p→TP53, chr10q→PTEN, chr9p→CDKN2A, chr5q→RB1,
chr6q→A20.

---

## 6. §4 — TCR-β variant reconciliation (centerpiece)

**Question.** Do CNV subclones **split by TCR-β variant** (MF branched-TCR evolution) or **share the
founder TCR** (paper's L-CTCL model)?

**Method.** Per malignant cell, `S.extract_trb` reads the TRB CDR3 (from `tcr_clone_id` `TRB:<cdr3>`,
falling back to `trb_cdr3`); `S.tcr_variant_class` labels it **founder** / **alice_variant**
(in the ≤1-aa family, not the founder) / **other**. For each multi-subclone donor, cross-tab
`tcr_variant_class` × `cnv_subclone` and quantify association with **normalized mutual information
(NMI)** and a χ² p-value. If NMI ≈ 0 → subclones share the founder TCR; NMI > 0 → CNV subclones
track TCR variants (branched-TCR).

**Result.**
- **8/23 multi-subclone donors carry TCR-β variants.**
- Of those, **CNV subclones align with TCR variant (NMI > 0.05): 0.**
- **All NMI ≈ 0** (max = 0.023 for Li2024 PT50). → **The CNV subclones share the founder TCR** in
  every donor, **supporting the paper's shared-TCR (single-founder) model**. No MF branched-TCR
  signal is detected: divergence is genomic/transcriptional *within* one TCR clone, not TCR-defined.

**Anchor check (Cell 16).** The Rindler **D5__MFIVB single-aa pair** `CASSQDRALENTIYF` /
`CASSQDRTLENTIYF` — both variants populate **all three** CNV subclones (s0/s1/s2), i.e. the ≤1-aa
TCR variant is **not** segregated into a distinct subclone:

```
cnv_subclone     s0    s1   s2
CASSQDRALENTIYF  177  661   55
CASSQDRTLENTIYF   22  153   13
```

---

## 7. §5 — Subclone-vs-subclone DE + GSEA

Within a patient there are **no biological replicates**, so two contrasts are used:

**(a) Within-donor single-cell Wilcoxon (`S.subclone_markers_wilcoxon`).** The paper's
`presto::wilcoxauc` baseline: `sc.tl.rank_genes_groups(method="wilcoxon")` per subclone, top 30
genes → `data/atlas_joint/subclone_de/subclone_de_<donor>.csv` for each multi-subclone donor.
Example (D1__P171 top markers): s0 = PPIA, CD2, JPT1, FAM107B, ATP5F1C; s1 = TPT1, RPS27, RPL30,
RPS14, B2M (ribosomal/translation-skewed — typical of within-clone state, not CNA-driven, DE).

**(b) Cohort dominant-vs-minor pseudobulk DESeq2.** The always-runnable transcriptional-divergence
contrast: largest subclone per donor (**dominant**, ~75% of the malignant pool) vs the pooled
**minor** subclones. `S.pseudobulk_counts` sums `raw_counts` per `donor|dm_class` pseudo-sample
(≥20 cells); `S.run_pydeseq2` fits **`~ donor + kind`** so **donor blocking absorbs patient
identity** (NB-GLM + Wald + BH). Runs on a memory-frugal view (not a full copy) to avoid the OOM
that previously killed the kernel.

- **23 donors** have dominant + ≥1 minor. **149 DEGs at padj < 0.05.**
- **Minor subclones are ↑ in a stress / AP-1 immediate-early program** (negative log2FC = higher in
  minor): **DNAJB1 (−0.67), HSPA1A (−0.66), FOS (−0.56), FOSB (−0.55), DUSP1 (−0.51), HSPH1, RGS1/2,
  TPT1**; dominant subclones are relatively ↑ in **RANBP1** (+0.39). Interpretation: the dominant
  subclone is transcriptionally "cleaner"; minor subclones carry a heat-shock/AP-1 stress signature.

**GSEA (prerank, `gseapy` on the DESeq2 log2FC).** Strongly **negative NES across the board**
(enriched in the *minor* direction), FDR q = 0.005 throughout — top terms: LEE_EARLY_T_LYMPHOCYTE_DN
(NES −3.86), FOXP3 targets, KEGG antigen processing/presentation, IL6/7, TCR-signaling modulators,
IL12/STAT4, NFκB signaling. I.e. minor subclones up-regulate T-activation / antigen-presentation /
cytokine-signaling programs relative to the dominant subclone.

---

## 8. §6 — Functional-program divergence

**Method (`S.score_programs`).** `sc.tl.score_genes` on the paper's Fig 4/5 programs (from
`PROGRAM_SETS`) plus `sc.tl.score_genes_cell_cycle` (Tirosh/Regev S & G2M lists):

- **prolif** (MKI67, TOP2A, PCNA, CCNB1, CDK1, TYMS, BIRC5)
- **oxphos** / **glycolysis** (metabolism)
- **cytokine** (IL2/4/10/13/21/22, IFNG, TNF)
- **th1 / th2 / th17** (master TFs + signature cytokines/receptors)
- **skin_homing** (CCR4/6, CD69, ITGAE, CCR10, SELPLG) / **recirc_homing** (SELL, CCR7, S1PR1, KLF2, TCF7)

Per-subclone mean scores (centered) → `figures/subclone_programs.png`. Within-donor divergence is
tested per program with a **Kruskal–Wallis** across subclones.

**Result: 129/207 (donor × program) tests significant at p<0.05.** Fraction of donors significant,
by program (most→least divergent):

| program | frac donors significant |
|---|---|
| glycolysis | 0.83 |
| oxphos | 0.74 |
| th17 | 0.70 |
| cytokine | 0.65 |
| skin_homing | 0.65 |
| prolif | 0.57 |
| recirc_homing | 0.52 |
| th2 | 0.52 |
| th1 | 0.43 |

**Metabolism (glycolysis, oxphos) is the most consistently divergent axis between subclones**,
followed by Th17 and cytokine programs — echoing the paper's finding that subclones diverge on
metabolic and effector-program state.

---

## 9. §7 — Epidermis vs dermis tropism

**Method.** The paper's niche comparison; our closest skin-only analog (no Visium/blood in this
object). Map `tissue` → **epidermis** / **dermis** / other; for donors with both layers and ≥2
subclones, a per-subclone **Fisher exact** test on (epidermis, dermis) counts vs the donor total.

**Result.** **7 donors testable; 11/18 subclones show a significant layer skew (p<0.05).** Subclones
within the same patient partition by skin layer — e.g. Li2024 CTCL1 s0 is epidermis-enriched
(OR 4.8) while s1 is dermis-enriched (OR 0.21); CTCL8 s0 epidermis-enriched (OR 10.9) vs s1
dermis-enriched (OR 0.09). → **Subclones diverge in tissue tropism**, a spatial correlate of
divergent evolution.

---

## 10. §8 — Cohort summary

- **(a) Prevalence / #subclones** — `figures/subclone_prevalence.png`: histogram of k;
  **≥2 subclones in 23/24 = 96%** (paper 84%).
- **(b) Trunk vs branch landscape** — `figures/subclone_trunk_branch_landscape.png`: per-arm
  fraction of multi-subclone donors where the arm is trunk (shared, up) vs branch (private, down);
  known CTCL loci starred. **chr7p/7q dominate the trunk** (recurrent shared ancestral gain).
- **(c) Subclonality vs disease stage** — **no association**: mean #subclones = **2.40** (early,
  n=10) vs **2.43** (advanced, n=14); mean #branch arms = 1.89 vs 2.00. Subclonal diversity is
  established early and does not scale with stage in this cohort.

---

## 11. §9 — Outputs

Written to `data/atlas_joint/`:

| File | Content |
|---|---|
| `subclone_donor_eligibility_v2.csv` | per-donor include/exclude for subclone detection + `reason_excluded` (§0) |
| `subclones_v2.parquet` | per-cell (**68,259 × 8**): donor, study, disease, stage, `cnv_subclone`, `is_dominant_subclone`, `subclone_kind`, `tcr_variant_class` |
| `subclone_summary_v2.csv` | per-donor: k, silhouette, arm deltas, sizes, #TCR variants, `frac_cnv_confirmed` (TCR/CNV concordance QC), #major/#minor, metadata |
| `subclone_trunk_branch_v2.csv` | per-donor trunk / branch arm lists + counts |
| `subclone_tcr_reconcile_v2.csv` | per-donor NMI / χ² (TCR-variant vs subclone) |
| `subclone_de/subclone_de_<donor>.csv` | within-donor Wilcoxon markers |
| `subclone_de_dominant_vs_minor.csv` | cohort pseudobulk-DESeq2 result |

Figures → `figures/subclone_*.png`.

---

## 12. Interpretation & limitations

**What the notebook establishes.**
1. **Subclonality is near-universal** in the MF/CTCL malignant compartment (96% of donors ≥2 CNV
   subclones), on par with / exceeding the paper's 84% — though at **fewer subclones per patient**
   (median 2 vs 3), a resolution ceiling of arm-level CNV.
2. **MF follows the paper's shared-TCR (single-founder) model** — CNV subclones do **not** segregate
   by TCR-β variant (NMI≈0 in all 8 variant-carrying donors; the D5__MFIVB anchor pair is spread
   across all subclones). Divergence is genomic/transcriptional *within* one TCR clone.
3. **Divergence axes:** trunk chr7p/7q gains anchor a shared ancestor; branches are private arm
   events; subclones diverge most on **metabolism (glycolysis/oxphos)**, **Th17/cytokine** programs,
   and **skin-layer tropism** (epidermis vs dermis). The dominant-vs-minor DE separates a clean
   dominant clone from stress/AP-1-high minor subclones.

**Limitations.**
- **Arm-level `infercnvpy` is coarser than Numbat** — focal drivers (9p21/CDKN2A, 17p/TP53) are
  **blunted**, so subclone count is a lower bound and some branch events are missed.
- **Transcriptional major/minor separation is weak** (§2), hence the size-based dominant-vs-minor
  contrast in §5.
- **No CITE-seq / WGS-WES / Numbat phasing / drug / *S. aureus* data** in this object — several of
  the paper's axes cannot be replicated.
- **Skin-only** — the paper's blood/skin niche comparison reduces here to epidermis vs dermis.
- Within-patient DE has **no biological replicates** (single-cell Wilcoxon only); the pseudobulk
  DESeq2 contrast borrows strength across donors with donor as the block.
