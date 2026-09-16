# TCR clone ↔ CNV in CTCL/MF — what the literature says, and what we can test in our atlas

Synthesis of a NotebookLM deep-web review (notebook `TCR clone ↔ CNV relationship in CTCL/MF`,
id `d6ebb47b-45e4-4bf5-bb03-11f9d079c387`; ~40–50 unique sources). The full cited briefing is in
[`CLONE_CNV_LITERATURE.md`](CLONE_CNV_LITERATURE.md). This doc distills it into (a) the established
biology, (b) concrete predictions for **our** skin-T atlas, and (c) the figure in
`20_skin_clone_cnv_relationship.ipynb` that tests each.

The motivating question — *can the specific clone teach us about the CNV, or the CNV about the
clone?* — has a clear literature answer: **the two are coupled but not identical**. The TCR
rearrangement is a single, frozen "barcode" of the transformation event; CNVs are a *dynamic,
accumulating* record laid down on top of that barcode. So the clone defines the **set of cells**,
and the CNV profile reveals the **evolutionary state and aggressiveness** within that set.

---

## Q1 — Does the dominant TCR clone carry a coherent CNV signature? (+ subclonal evolution)

**Literature.**
- The dominant TCR clone is genomically **distinct from benign T cells**: in paired scRNA+TCR with
  inferCNV, benign polyclonal CD4 T cells are **flat/diploid**, while cells carrying the dominant
  clonotype show complex aneuploidy and patient-specific large-scale CNVs.
- But the clone is **not genomically uniform**. It carries shared **"trunk"** CNVs plus cell-to-cell
  **"branch"** CNVs = branched/divergent clonal evolution under one identical TCR. Subclones are
  found in **~84% of leukemic CTCL patients, median ~3 subclones each**.
- A **"transitional"** population exists: cells with the *identical* malignant TCR barcode that still
  cluster transcriptionally with normal CD4 T and show only **intermediate** CNV — a pre-malignant
  state on the trajectory from diploid → aneuploid.
- Caveat on directionality: some CNAs **predate** TCR rearrangement (trisomy 7 spans the TCRβ locus
  and is seen across different TCRβ transcripts; shared TCRγ with divergent α/β implies a precursor
  origin). So a single TCR marker can *under*-resolve genetically distinct subclones.

**Predictions for our atlas.**
1. Dominant-clone (`tcr_is_malignant`) CD4 cells have a **coherent, sample-specific arm-CNV profile**
   that is **near-zero in benign CD4** and in **non-dominant expanded** clones (specificity).
2. *Within* a single dominant clone there is **measurable arm-CNV heterogeneity** resolving into a
   small number (≈2–3) of CNV subclones sharing trunk arms, differing on branch arms.
3. A subset of TCR-malignant cells will be **CNV-low** (transitional) — they appear as `tcr_only`
   disagreement in the concordance map, not as method failure.

**Notebook figures.** Q1a `clone_cnv_dominant_arm_profiles.png` (dominant vs benign profiles) tests
prediction 1; Q1b `clone_cnv_subclones_<donor>.png` + `subclone_summary` (trunk/branch arms,
centroid spread, k≈2–3) tests 2; the reverse concordance crosstab tests 3.

---

## Q2 — Recurrent CTCL arm-level CNV landscape

**Literature (arm → gene → frequency / stage).** Pan-clonal "trunk" events unless noted.

| Arm | Direction | Gene target(s) | Frequency / stage note |
|---|---|---|---|
| **8q** (8q24) | gain | *MYC*, *TOX* | ~45–70% SS; high-level 8q24 → poor prognosis, tumor stage, ↓5-yr OS (esp. with *TP53* loss) |
| **17q** (17q22-25) | gain | *STAT3*, *STAT5B* | ~70% SS; drives JAK-STAT |
| **7 / 7q** (7q21-36) | gain | *FASTK*, *SKAP1* | characteristic of **MF**; trisomy 7 is an **early** event (stage IA/IB) |
| **1p36.2**, **10p15-12** | gain | — | 1p36 with chr7 in MF; 10p in >30% SS |
| **17p** (17p13) | loss | *TP53*, *NCOR1* | ~40–66% SS; aggressive, unfavorable prognosis |
| **10q** (10q23-25) / 10p | loss | *PTEN*, *FAS*, *ZEB1* | >50% SS; with 17p → genomic instability in advanced SS |
| **9p** (9p21) | loss | *CDKN2A/B* | high weight in **MF**; correlates with **large-cell transformation** + drug resistance |
| **6q** (6q23) | loss | *TNFAIP3/A20* | ~50% SS; **late** event (absent in new MF) → NF-κB |
| **5q** (5q13) | loss | *RB1*, *DLEU1* | hallmark of **MF**; progressive disease |
| **11q23**, **2p23** | loss | *USP28*; *DNMT3A* | ~33% SS; epigenetic reprogramming |

General: **>3 distinct recurrent alterations** in SS, or high overall CNV burden, → significantly
**shorter overall survival**.

**Predictions for our atlas.** Across donors' dominant clones, recurrence should concentrate on
**8q/17q gains and 17p/10q losses** (the SS-skewed trunk set), with **chr7 gain and 9p/5q loss**
appearing in the MF-leaning samples. The recurrence bars should land on the starred known-event arms,
not scatter uniformly.

**Notebook figure.** Q2 `clone_cnv_recurrent_landscape.png` + the `freq` table (per-arm gain/loss
fraction across donors; known events starred). This is the direct recapitulation check.

---

## Q3 — Does CNV burden scale with clonal expansion / stage?

**Literature.**
- CNV/aneuploidy burden **increases with stage** (indolent plaque → tumor stage ≥IIB), and genomic
  variability **correlates with the number of coexisting subclones**.
- Pseudotime: diploid normal → transitional (intermediate CNV) → complex-aneuploid malignant; within
  the malignant pool, trajectories run **fewest-CNV → most-CNV** subclones.
- The highest-CNV-burden subclones drive hyper-proliferation, higher blood tumor burden, and worse OS.

**Predictions for our atlas.**
1. Per-cell `cnv_cell_score` is **highest in the dominant clone**, intermediate/low in non-dominant
   **expanded** clones, lowest in singletons — a monotone gradient.
2. A **positive (if modest) correlation** between `cnv_cell_score` and `log(clone_size)`, expected
   to be stronger as a **per-donor** effect than pooled (donor/stage is the dominant axis).

**Notebook figure.** Q3 `clone_cnv_burden_by_expansion.png` (box by clone category) + pooled and
per-donor Spearman tests both predictions.

---

## What this buys us (the "learn one from the other" answer)

- **Clone → CNV:** knowing a cell's clone tells us *which* CNV program to expect (trunk arms shared
  across the clone) and lets us read **evolutionary state** from where the cell sits on the
  diploid→transitional→aneuploid axis. The non-dominant expanded clones are the negative control.
- **CNV → clone:** arm-level CNV can **assign malignancy to TCR-dropout cells** (recover clone
  membership where the TCR is unrecovered), and the **trunk/branch split** can substructure a single
  TCR clone into subclones the TCR alone cannot see. The literature's directional caveat (CNAs that
  predate TCR rearrangement) is the principled reason the two calls won't be identical — and the
  `tcr_only` (transitional) / `cnv_only` (TCR-dropout) cells in our concordance map are biology, not
  noise.

**Caveats carried into the analysis.** (1) Our reference is same-sample **CD8**, so trunk events on
shared T-lineage arms are well controlled but truly clonal CD8 contamination would blunt signal.
(2) inferCNV arm means are smoothed — focal events (9p21/CDKN2A, 17p13) may read weaker than their
true frequency. (3) MF-vs-SS skew in our cohort will shift which trunk arms dominate; interpret Q2
recurrence stratified by study/disease.
