# The TCR clone ↔ copy-number relationship in cutaneous T-cell lymphoma — a detailed review

**Scope.** This is the long-form companion to [`CLONE_CNV_RESEARCH.md`](CLONE_CNV_RESEARCH.md)
(the concise, prediction-focused synthesis) and [`CLONE_CNV_LITERATURE.md`](CLONE_CNV_LITERATURE.md)
(the raw NotebookLM briefing). It explains the biology, the open problems, and the current state of
the field in depth, and connects each strand to the analysis in
[`20_skin_clone_cnv_relationship.ipynb`](20_skin_clone_cnv_relationship.ipynb).

**Provenance & caveat on citations.** The literature here was assembled through a NotebookLM deep-web
research pass (notebook id `d6ebb47b-45e4-4bf5-bb03-11f9d079c387`, ~40–50 unique sources). Author /
year / journal attributions are reproduced as the model reported them and are reliable at the level of
*who showed what*, but **exact bibliographic details (journal, year, volume) should be verified
against PubMed before any formal/manuscript use**. Where the corpus was ambiguous I flag it.

---

## 1. The problem, in one paragraph

Mycosis fungoides (MF) and its leukemic variant Sézary syndrome (SS) are the dominant cutaneous
T-cell lymphomas (CTCL): malignancies of skin-homing CD4⁺ T cells. Two molecular features define a
malignant T cell, and they are fundamentally different in kind. The first is the **T-cell receptor
(TCR) rearrangement** — a single, irreversible DNA recombination event that every descendant of the
transformed cell inherits unchanged. It is a perfect *lineage barcode* but a *static* one. The second
is the cell's **copy-number landscape** — the gains and losses of chromosomal material that accumulate
as the tumor evolves. It is a *dynamic, time-stamped* record of the tumor's history. The scientific
and clinical question that motivates this project is the relationship between the two: **what does
knowing a cell's clone tell us about its copy-number state, and what does the copy-number state tell us
about the clone?** The short answer from the literature is that the two are *coupled but not
identical*, and the places where they disagree are biologically the most interesting — they expose
clonal evolution, pre-malignant transitional states, and the limits of TCR-based diagnosis.

---

## 2. Background: what a malignant T-cell clone is, and how we read it

**The TCR as a clonal mark.** During thymic development a T cell rearranges its TCR loci in a fixed
order — **TCRγ and TCRδ first, then TCRβ, then TCRα** — each via RAG-1/RAG-2–mediated V(D)J
recombination, producing a junction (the CDR3) effectively unique to that cell. Mature peripheral T
cells do *not* re-rearrange or somatically hypermutate their TCR (unlike B cells with their
immunoglobulin loci), so an identical TCR sequence shared by many cells is strong evidence they
descend from one ancestor. This is why a dominant monoclonal TCRβ/TCRγ rearrangement, detected by
multiplex PCR/GeneScan or high-throughput sequencing, has been the diagnostic "gold standard" for
distinguishing CTCL from benign reactive inflammation (the classical view; see §4).

**In our atlas**, the clone is operationalised in `skin_T_cnv_helpers.recompute_dominant_clone`: a
TRB-primary CDR3 key per cell, and a per-donor *dominant-clone* rule (top clone ≥5% of the donor's
TCR⁺ cells **and** ≥2× the second clone). Cells in that clone get `tcr_is_malignant = True`. This is a
deliberately conservative, classical-model definition — and §4 explains exactly where that definition
is expected to under-call.

**Copy-number as the second axis.** Solid-tumor-style aneuploidy (whole-chromosome and arm-level gains
and losses) is pervasive in CTCL, especially in advanced/leukemic disease. Because we have scRNA-seq
rather than scDNA-seq, we read copy number *indirectly* through inferCNV (§8): large structural
changes shift the average expression of all genes in the affected segment, which a smoothing model
recovers as a per-cell CNV profile.

---

## 3. Why this matters clinically

Three practical stakes ride on the clone↔CNV relationship:

1. **Diagnosis.** Early-patch MF is notoriously hard to separate from benign dermatitis. If the
   TCR-monoclonality criterion systematically misses part of the tumor (§4), CNV evidence (e.g. chr7
   gain detectable even at stage IA/IB) becomes a complementary diagnostic axis.
2. **Prognosis.** Genomic instability — high CNV burden, and specific events such as 17p/TP53 and
   10q/PTEN loss — tracks with stage, large-cell transformation, therapy resistance, and shorter
   overall survival. In SS, carrying **>3 distinct recurrent alterations** correlates with worse
   prognosis.
3. **Minimal residual disease & therapy.** If a single TCRβ clonotype is tracked as the disease
   marker, but the tumor is a branching family of subclones from an upstream progenitor, then a
   "molecular remission" of that one clonotype can be illusory while sibling subclones persist.

---

## 4. Main issue #1 — the cell of origin, and why the TCR barcode can mislead

This is the deepest conceptual problem in the field, because it determines how much we can trust the
TCR as a complete description of "the clone."

**The classical mature-cell model.** MF was thought to transform from a *mature, post-thymic,
skin-resident effector-memory* CD4⁺ T cell (TRM phenotype: CLA⁺ CCR4⁺, CCR7⁻ L-selectin⁻); SS from a
*central-memory* T cell (TCM: CCR7⁺ L-selectin⁺) that recirculates between skin, blood and nodes. The
evidence was precisely the detection of one uniform TCR rearrangement: since mature T cells don't
re-rearrange, one TCR sequence implied one mature cell of origin.

**The progenitor / immature-precursor model.** High-throughput TCR sequencing and single-cell
multi-omics have seriously challenged this. The key lines of evidence:

- **Discordant TCR chains (Hamrouni et al., 2019, *Clinical Cancer Research*; Iyer et al., 2020,
  *Blood Advances*; work from the Gniadecki group).** Within a single MF lesion, malignant cells often
  share *one* monoclonal **TCRγ** rearrangement but carry *discordant, apparently polyclonal*
  **TCRβ/TCRα** sequences. Because TCRγ rearranges *before* TCRβ/α in ontogeny, this places the
  transforming hit in an **immature precursor**, after TCRγ assembly but before β/α completion —
  subclones then finish β/α rearrangement independently and diverge.
- **Early aneuploidy preceding TCR rearrangement.** In trisomy-7 cases (chr7 carries the TCRβ locus),
  malignant cells can carry two or three distinct TCRβ transcripts. Since RAG is off in mature cells,
  the chromosome must have duplicated *before* TCRβ rearrangement finished — CNV literally predating
  the barcode.
- **Shared mutations with bone-marrow progenitors (Harro et al., 2023, *Blood Advances* —
  "Sézary syndrome originates from heavily mutated hematopoietic progenitors").** Paired multi-omics
  found **>200 somatic mutations shared between CD34⁺ marrow HSPCs and circulating Sézary cells**, and
  retained sjTRECs (recent-thymic-egress marker) in malignant cells — i.e. a continuously replenished
  upstream reservoir.
- **Extreme phenotypic plasticity** spanning naïve, central-memory, effector-memory and TEMRA states
  within one clone/patient — hard to reconcile with a single committed mature founder.
- **Clinical behaviour:** multifocal lesions, relapse after total-skin electron-beam therapy that
  should clear all skin lymphocytes, and CTCL transfer via T-cell-depleted bone-marrow transplants —
  all consistent with a hidden upstream reservoir.

**Why this matters for the clone↔CNV question (and for our analysis).** If the true malignant
population is a TCRγ-defined family with multiple TCRβ subclones, then **a TCRβ-based dominant-clone
call (ours) under-counts the tumor** — in some lesions the dominant TCRβ clone is only ~15% of true
tumor cells. The corollary: **CNV can recover malignant cells the TCRβ barcode misses.** This is the
single most important reason the TCR call and the CNV call are *expected* to disagree, and it
reframes our concordance map: `cnv_only` cells (aneuploid, not in the dominant TCRβ clone) are
candidate sibling subclones, not false positives.

---

## 5. Main issue #2 — the recurrent CNV landscape of MF/SS

CTCL genomes are characterised by extensive arm-level imbalance and a high focal-deletion load
(reported ~7.5 focal deletions per sample), plausibly driven by aberrant RAG activity in precursors
and/or UV mutational signatures. The recurrent events converge on a few pathways: **TCR signalling,
JAK-STAT, NF-κB, cell-cycle/DNA-damage control, and immune evasion.**

| Arm | Event | Target gene(s) | Frequency / clinical note | Subtype skew |
|---|---|---|---|---|
| **8q** (8q24) | gain | *MYC*, *TOX* | ~45–70% of SS; high-level 8q24 → poor prognosis, tumor stage, ↓5-yr OS (esp. with *TP53* loss) | SS (trunk) |
| **17q** (17q22-25) | gain | *STAT3*, *STAT5B* | ~70% of SS; constitutive JAK-STAT | SS (trunk) |
| **7 / 7q** (7q21-36) | gain | *FASTK*, *SKAP1*; locus also carries TCRβ | **early** event, detectable at stage IA/IB; enhanced TCR-driven survival | **MF** |
| **1p36.2 / 1p** | gain | *ENO1*, *TNFRSF1B* | seen with chr7 in MF; metabolic adaptation | MF |
| **10p15-12** | gain | — | >30% of SS | SS |
| **17p** (17p13) | loss | *TP53*, *NCOR1* | ~40–66% of SS; aggressive subtype, genomic instability, poor prognosis | SS (trunk) |
| **10q** (10q23-25)/10p | loss | *PTEN*, *FAS*, *ZEB1*, *NFKB2* | >50% of SS (FAS/ZEB1); with 17p drives instability; PI3K/AKT activation, apoptosis resistance | SS / leukemic |
| **9p21** | loss | *CDKN2A/B* | high weight in MF; correlates with **large-cell transformation** + drug resistance | MF / tCTCL |
| **6 / 6q** (6q23) | loss | *TNFAIP3/A20*; also MHC-I/II genes | ~50% of SS; **late** event (absent in new MF); NF-κB activation; MHC loss → immune evasion | SS (late) |
| **5q** (5q13) | loss | *RB1*, *DLEU1* | hallmark of MF; defective cell-cycle arrest | MF |
| **11q23** | loss | *USP28* | ~33% of SS; ubiquitin-proteasome | SS |
| **2p23** | loss | *DNMT3A* | epigenetic reprogramming | SS |

(Attributions in the corpus: Srinivas et al., 2024; Hamrouni et al., 2019; Crespi et al., 2026; and
the SS/MF reviews.)

**The MF-vs-SS distinction matters for interpretation.** SS (leukemic) is dominated by 8q/17q gain
and 17p/10q loss; skin-limited MF leans on chr7 gain and 9p21/5q loss. Our atlas mixes studies and
stages, so **Q2 recurrence should be read stratified by `study`/`disease`**, not pooled into a single
"CTCL landscape."

---

## 6. Main issue #3 — branched clonal evolution and subclonal CNV within one TCR clone

A single TCR-defined clone is **not a genomically uniform mass**. The consensus picture:

- **Trunk vs branch CNVs.** All malignant cells share *pan-clonal* "trunk" events (commonly 8q & 17q
  gain, 10q & 17p loss); subclones then diverge by acquiring *private* "branch" events (variable gains
  on chr 1, 3, 4, 5, 7; focal losses such as 1p36, 12p13). Branch CNVs co-occur with subclonal SNVs and
  indels — i.e. genuine, irreversible genetic divergence, not transcriptional state.
- **Prevalence.** Distinct malignant subclones are found in **~84% of leukemic CTCL patients, median
  ~3 subclones** per patient.
- **Branched (divergent), not linear, evolution (Iyer et al., 2020; Herrera et al., 2021;
  Srinivas et al., 2024).** Subclones intermix phylogenetically across skin and blood rather than
  forming a clean linear chain. Srinivas et al. found two dominant clones differing by a *single*
  TCRβ amino acid co-expanded across skin/blood/node — branching from one ancestral cell, spreading
  systemically.
- **Functional specialisation.** Subclones with the same TCR diverge in metabolism (e.g. OXPHOS),
  proliferation, and tissue tropism, "dividing labour" across compartments rather than directly
  competing.
- **Transitional / pre-malignant cells (Ren et al., 2023, *Blood Advances*).** A population carries
  the *identical* malignant TCR barcode but clusters transcriptionally with normal CD4 T and shows
  only **intermediate** CNV — a way-station on a diploid → transitional → complex-aneuploid trajectory.

**For our analysis:** this is the literature basis for **Q1b** (cluster the dominant clone's cells on
the arm matrix → trunk arms shared across subclones, branch arms private) and for reading **TCR⁺ but
CNV-low** cells in the concordance map as transitional rather than as caller error.

---

## 7. Main issue #4 — does CNV burden scale with clonal expansion and stage?

The literature is fairly consistent that **burden accumulates with time, stage, and treatment**:

- **Stage.** CNV/aneuploidy burden rises from indolent stage-I plaques to advanced (≥IIB) tumors, and
  genomic variability correlates with the *number* of coexisting subclones.
- **Trajectory.** Pseudotime runs diploid normal → transitional (intermediate CNV) → complex
  aneuploid; *within* the malignant pool, trajectories run fewest-CNV → most-CNV subclones.
- **Disease duration & therapy (Srinivas et al., 2024).** Longer disease and more therapy lines —
  **especially ionizing radiation** — associate with more alterations; treatment can *accelerate*
  clonal divergence. A cautionary note for interpreting heavily-pretreated samples.
- **Expansion.** Progressive enrichment of the dominant clone is accompanied by accumulation of
  secondary genetic hits (Hamrouni et al., 2019).
- **Outcome.** The highest-burden subclones (e.g. concurrent *PTEN*/*PDCD4* loss + *P70S6K* gain)
  drive hyper-proliferation, higher blood tumor burden, worse OS.

**For our analysis (Q3):** expect a monotone burden gradient dominant > expanded-non-dominant >
singleton, and a positive `cnv_cell_score`–`log(clone_size)` correlation that is likely **stronger
per-donor than pooled** (donor/stage is the larger axis of variation). Because our cohort spans stages
and treatments we cannot control, treat the pooled correlation as suggestive, not definitive.

---

## 8. Methods deep-dive — inferring CNV from scRNA-seq in T-cell lymphoma

This section is load-bearing for trusting our own results, because **CNV-from-RNA is an inference, not
a measurement**, and T cells are an adversarial case.

**How inferCNV works.** It computes a moving average of relative expression across sliding genomic
windows (~50–100 genes), referenced to a user-defined diploid population, optionally denoised and
passed through a Hidden Markov Model to call discrete states (loss / neutral / gain). Our pipeline
uses infercnvpy with a 250-gene window; the new `compute_arm_cnv_per_cell` helper additionally turns
on `calculate_gene_values=True` and averages the per-gene CNV within each chromosome **arm** (p/q via
an hg38 centromere table) to get a compact, interpretable cells × ~39-arm matrix.

**CopyKAT** is the common alternative: similar smoothing but with internal diploid self-detection via
hierarchical clustering and Gaussian-mixture segregation of aneuploid vs diploid — useful when no
clean reference exists, at high compute cost.

**The reference is the single most important choice.** Accuracy hinges on a good diploid reference:
- *Gold standard:* matched, **non-malignant polyclonal CD4⁺ T cells** from the same patient, selected
  via scTCR as those *lacking* the dominant clonotype (the true lineage counterpart).
- *Robust alternatives:* intra-patient **CD8⁺ T cells**, monocytes, or NK cells.
- *Reference-free:* pseudo-reference from the most transcriptionally stable clusters.

**Our design uses same-sample CD8⁺ T cells as the diploid reference** (`cnv_ref = "cd8_ref"`,
CD4 = query), a literature-sanctioned lineage-matched choice — appropriate because CTCL is
overwhelmingly a CD4 malignancy. The within-donor benign CD4 (non-dominant, non-tumor) provides the
*floor* for thresholding. (Caveat: clonally expanded reactive CD8 would blunt contrast.)

**Validation.** Inferred CNAs have been corroborated by bulk WES/WGS of FACS-sorted populations, FISH,
and methylation arrays; scRNA inference recovers both trunk CNVs and subclonal events present in
>15% of malignant cells, and shows **strict concordance with TCR clonality** — dominant-clonotype
cells carry complex inferCNV profiles while polyclonal background T cells read flat/diploid (Herrera
et al., 2021; Song et al., 2022).

**Limitations / pitfalls — directly relevant to our caveats.**
1. **Transcriptional-burst false positives.** T-cell activation, TCR signalling and cytokine exposure
   coordinately upregulate large gene blocks that can masquerade as gains. CD4 query cells in inflamed
   skin are exactly the worst case → interpret modest single-arm signals cautiously.
2. **Cell-cycle artifacts.** Proliferating (S/G2M) malignant cells can produce clustered expression
   changes read as spurious subclones — relevant to Q1b; consider regressing or flagging cycling cells.
3. **Blind to copy-neutral LOH and to SNV/indels** → inferCNV *under*-counts subclones; our subclone
   numbers are lower bounds.
4. **Smoothing dilutes focal events.** Arm-level averaging (our approach) makes broad events robust but
   weakens small but important loci (9p21/CDKN2A, 17p13/TP53) — they may read weaker than their true
   frequency.
5. **Compute cost** (CopyKAT especially) — why we cache aggressively and run per-sample.

---

## 9. Landmark single-cell studies (annotated)

- **Buus et al. (2018)** — SS peripheral blood. Foundational demonstration of single-cell
  heterogeneity within the malignant population; diagnostic biomarkers expressed *heterogeneously*
  across neoplastic cells, undercutting the "uniform clone" assumption.
- **Gaydosik et al. (2019)** — advanced CTCL skin. Diverse intratumoral lymphocyte populations;
  multiple high-frequency TCR clonotypes co-existing in single samples (hint of multiple clones).
- **Borcherding et al. (2019, 2023)** — SS blood, ~50k T cells, longitudinal. Five malignant
  transcriptional clusters; **directional clonal evolution** via *FOXP3*/*GATA3*/*IKZF2*; on therapy
  (HDACi + photopheresis), new *FOXP3*-high clusters emerge at progression (Treg-like immune evasion).
- **Herrera et al. (2021), *Blood*** — matched skin+blood, leukemic MF/SS; scRNA+scTCR+**inferCNV**.
  Same dominant TCR seeds skin and blood, but resolves into **multiple subclones with distinct CNV
  profiles** that intermix across compartments; **tissue drives plasticity** (skin = activated/
  proliferative, blood = quiescent). A direct template for our analysis.
- **Rindler et al. (2021)** — matched skin/node/blood MF. One clone tracked across three compartments;
  TRM signature in skin vs TCM in node/blood — compartment-specific plasticity over a stable CNV core.
- **Song et al. (2022), *Cancer Discovery*** — transformed CTCL (tCTCL), 5′ scRNA + scV(D)J. Combined
  TCR clonotyping with inferCNV to **rescue TCR-dropout cells** and cleanly separate malignant from
  benign; described recurrent large-scale CNAs driving large-cell transformation; minor TCRβ
  subclonal variation within the dominant clonotype.
- **Iyer et al. (2020), *Blood Advances*** — WES; **branched evolution and intratumor genomic
  heterogeneity** as the organising principle of CTCL.
- **Hamrouni et al. (2019), *Clinical Cancer Research*** — TCR clonotypic diversity → **immature
  precursor origin** (shared TCRγ, discordant α/β).
- **Harro et al. (2023), *Blood Advances*** — **Sézary originates from heavily mutated HSPCs**
  (>200 shared mutations marrow↔blood).
- **Ren et al. (2023), *Blood Advances*** — trajectory analysis → **transitional pre-cancer
  populations** (malignant TCR, intermediate CNV).
- **Srinivas et al. (2024)** — MF scRNA+scTCR; robust malignant-vs-reactive signature; branching
  clonal evolution from single TCRβ-amino-acid variants spreading skin/blood/node; CNV burden scales
  with disease duration and therapy.
- **Liu et al. (2022) / Li et al. (2024)** — CTCL TME atlases; malignant TH2 cells supported by a
  B-cell-rich immunosuppressive microenvironment. (Li 2024 is the source of our own atlas.)

---

## 10. Synthesis — learning the clone from the CNV, and the CNV from the clone

- **Clone → CNV.** A cell's clone predicts *which CNV program* it should carry: the trunk arms shared
  across the whole clone, on top of which evolutionary state can be read from where the cell sits on
  the diploid → transitional → complex-aneuploid axis. Non-dominant *expanded* clones are the negative
  control — they should be near-diploid; if they aren't, either they are sibling malignant subclones
  (precursor model) or the CNV signal is contaminated.
- **CNV → clone.** Arm-level CNV can (i) **assign malignancy to TCR-dropout / TCRβ-discordant cells**
  the barcode misses, expanding the true tumor fraction, and (ii) **substructure one TCR clone** into
  trunk-sharing, branch-differing subclones that the TCR alone cannot resolve. The principled reason
  the two callers won't agree perfectly is that **some CNAs predate TCR rearrangement** and the TCRβ
  barcode is an incomplete label of the malignant family.

The disagreement cells are therefore the scientific payload: `tcr_only` ≈ transitional/pre-malignant
or low-burden malignant; `cnv_only` ≈ sibling subclone or TCRβ-discordant malignant cell.

---

## 11. Open questions / gaps the field has not settled

1. **How complete is the progenitor model?** It is strongly supported in SS (Harro 2023) and in
   TCRγ/αβ-discordant MF, but the fraction of CTCL that is genuinely precursor-derived vs mature-cell
   derived is unresolved.
2. **Directionality of trunk vs branch** — which arm events are obligate-early vs contingent-late is
   only partially mapped; chr7-early and 6q-late are among the clearer calls.
3. **Disentangling CNV from transcriptional state** in inferCNV for activated/cycling T cells remains
   a methodological soft spot (no consensus correction).
4. **Whether CNV burden is cause or correlate** of aggressiveness vs simply a clock of disease
   duration/therapy.
5. **CNLoH and focal drivers** are invisible to RNA-based inference, so the true subclonal complexity
   is systematically under-counted.

---

## 12. What we test in `20_skin_clone_cnv_relationship.ipynb`

| Literature claim | Prediction in our atlas | Notebook output |
|---|---|---|
| Dominant clone is aneuploid; benign CD4 diploid; coherent trunk profile | Dominant-clone arm profile distinct from benign & non-dominant-expanded | Q1a `clone_cnv_dominant_arm_profiles.png` |
| Trunk + branch subclones within one TCR clone (~3, in ~84%) | Dominant-clone cells split into ≈2–3 CNV subclones; shared trunk arms, private branch arms | Q1b `clone_cnv_subclones_<donor>.png`, `subclone_summary` |
| Recurrent CTCL landscape (8q/17q gain; 17p/10q/9p loss; chr7 in MF) | Per-donor dominant-clone calls concentrate on the known arms, stratified by study/disease | Q2 `clone_cnv_recurrent_landscape.png`, `freq` table |
| Burden scales with expansion/stage | `cnv_cell_score` gradient dominant>expanded>singleton; positive per-donor Spearman | Q3 `clone_cnv_burden_by_expansion.png` |
| TCRβ under-counts tumor; CNV recovers it; transitional cells exist | `cnv_only` = sibling subclones / TCRβ-dropout; `tcr_only` = transitional/low-burden | reverse concordance crosstab + `skin_clone_cnv_relationship.parquet` |

**Carried caveats:** same-sample CD8 reference (controls T-lineage trunk well, but clonal CD8 would
blunt signal); arm smoothing weakens focal 9p21/17p; activation/cell-cycle can inflate apparent CNV;
MF/SS and treatment heterogeneity in the cohort means Q2/Q3 should be read stratified, and the pooled
burden–expansion correlation is suggestive rather than causal.

---

## 13. Annotated reference list

*Bibliographic details as compiled via NotebookLM — verify against PubMed before formal use.*

1. **Iyer A., Hennessey D., O'Keefe S., et al. (2020), *Blood Advances*.** "Branched evolution and
   genomic intratumor heterogeneity in the pathogenesis of CTCL." WES evidence that one malignant
   clone undergoes branched evolution, acquiring distinct subclonal CNVs/mutations → intratumoral
   genomic heterogeneity.
2. **Herrera A., Cheng A., Mimitou E.P., et al. (2021), *Blood*.** "Multimodal single-cell analysis of
   CTCL reveals distinct subclonal tissue-dependent signatures." scTCR + inferCNV: identical dominant
   TCR diverges into subclones with unique CNV profiles intermixing across blood/skin.
3. **Song X., Chang S., Seminario-Vidal L., et al. (2022), *Cancer Discovery*.** "Genomic and
   single-cell landscape … of transformed CTCL." scTCR + single-cell CNV inference cleanly separates
   malignant from benign; recurrent large-scale CNAs drive large-cell transformation.
4. **Hamrouni A., Fogh H., Zak Z., et al. (2019), *Clinical Cancer Research*.** "Clonotypic Diversity
   of the TCR Corroborates the Immature Precursor Origin of CTCL." Shared TCRγ, discordant α/β →
   transformation begins in an immature precursor.
5. **Harro C.M., Sprenger K.B., Chaurio R.A., et al. (2023), *Blood Advances*.** "Sézary syndrome
   originates from heavily mutated hematopoietic progenitors." >200 mutations shared between CD34⁺
   marrow HSPCs and circulating Sézary cells.
6. **Rindler K., Bauer W.M., Jonak C., et al. (2021).** "scRNA-seq reveals tissue compartment-specific
   plasticity of MF tumor cells." One TCR-defined clone tracked across blood/node/skin; stable CNVs,
   compartment-specific transcriptomes. *(Journal ambiguous in corpus — verify.)*
7. **Ren J., Qu R., Rahman N.T., et al. (2023), *Blood Advances*.** "Integrated transcriptome and
   trajectory analysis of CTCL identifies putative precancer populations." Transitional cells: malignant
   TCR barcode, intermediate CNV.
8. **Buus T.B., et al. (2018).** Single-cell heterogeneity of SS peripheral blood; heterogeneous
   biomarker expression within the malignant clone.
9. **Gaydosik A.M., et al. (2019).** scRNA-seq of advanced CTCL skin; diverse TME, multiple
   high-frequency clonotypes.
10. **Borcherding N., et al. (2019, 2023).** Longitudinal scRNA/scTCR of SS blood; directional clonal
    evolution (FOXP3/GATA3/IKZF2); therapy-associated FOXP3-high clusters at progression.
11. **Srinivas N., et al. (2024).** MF scRNA + scTCR; malignant-vs-reactive signature; branching
    evolution from single-amino-acid TCRβ variants; CNV burden scales with duration/therapy.
12. **Liu X. / Li et al. (2022, 2024).** CTCL TME atlases; TH2 malignant cells supported by a
    B-cell-rich immunosuppressive niche. (Li 2024 = source atlas for this project.)
13. **Crespi et al. (2026).** Focal-deletion landscape (CDKN2A, TP53); genomic instability mechanisms
    (RAG / UV signatures). *(Recent — verify details.)*

*(Methodology for inferCNV/CopyKAT, reference selection, and validation/limitations in §8 is drawn
from the same corpus plus the tools' primary descriptions.)*
