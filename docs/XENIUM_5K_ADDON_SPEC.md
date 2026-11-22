# Xenium Prime 5K add-on — 100-gene design for new CTCL/MF skin samples

New samples go on the **Xenium Prime 5K panel plus a ≤100-gene custom add-on**. This is the design:
which genes the add-on must carry, why each one, and what to do when 10x rejects some of them.

Roster: **`tables/xen5k_addon_v1.csv`** — 167 rows (100 ordered, 10 reserve, 57 parked). `in_100 = 1` is what to order; `reserve_rank`
is the fill order for rejections; the rest is parked with the reason recorded.

Scope decisions taken up front: **TCR constants only, skin FFPE only.** Blood-only leads are parked, not lost.

Revision history: 2026-11-03 the TRBV block was dropped (§2.1); 2026-11-04 the 10x designer rejected
eight genes and the list was backfilled (§2.2).

## 1. How the list was built

1. Diffed the 5K panel (`docs/XeniumPrimeHuman5Kpan_tissue_pathways_metadata.csv`, 5,001 genes, exact
   match on column 1) against every curated roster in `helpers/`: `surface_targets`, `degradome_data`,
   `ccc_panel`, `cluster_malig_helpers`.
2. Scanned all of `helpers/`, `docs/` and `tables/*.md` for gene symbols that are in the atlas vocabulary
   but off-panel — 745 candidates.
3. Triaged those by **result, not by roster membership**: a gene stays only if a number in
   `tables/tgt4*_llm_export.md`, `tables/deg28mc_*`, `tables/cluster_malig_*`, `tables/cmx28f_*` or a
   `docs/` analysis note carries it. That cut 745 to an inclusive pool of **151 genes + TCR**.
4. Weighted the pool for skin FFPE and cut to 100.

Every gene in the CSV was verified twice: absent from the 5K panel, and present in the
`skin_T_annotated.h5ad` var index (so it is a symbol we actually measure, not a stale alias).
`gen_addon` asserts all of this and refuses to write the file otherwise.

**Alias traps caught.** `TMEM2` is on the panel under its current symbol **`CEMIP2`**
(ENSG00000135048, same gene) and `WISP2` as **`CCN5`**. Both were in our rosters as "missing" and are not.
Check the `full_name` column, not just the symbol, before adding anything to this list.

## 2. Two facts that shaped the design

### 2.1 A partial TRBV panel cannot be read on an unknown clone

`cluster_malig_helpers.vskew` scores clone V-gene skew, and the 12 most-used TRBV genes cover 74% of
our skin clones at gene level (`tables/cluster_malig_clusters_v1.csv`, 107 donor x clone-V pairs; the
88% figure quoted elsewhere is *family*-level, and 10x probes are gene-specific).

**That coverage number is the wrong test.** The clone's V gene is unknown a priori for every new
sample, so on a partial V panel a flat skew reading is ambiguous: it means either "polyclonal" or
"this clone's V is not on the panel", and nothing in the image distinguishes them. The readout is
uninterpretable per sample, not merely blind on 26% of them.

The 12-probe block was therefore **dropped** (2026-11-03) and the slots reallocated. What replaces it:

- **TRBC1/TRBC2 restriction** is clone-agnostic. Every alpha-beta T cell expresses one constant region or
  the other, so a clonal patch skews the local TRBC1:TRBC2 ratio **whatever its V gene**. This is the
  clonality readout for the panel. Both probes must sit in the UTRs - the coding sequences are
  near-identical.
- **The learned program** becomes the sole in-situ malignancy caller (AUC .971 by cluster / .982 by cell,
  versus .989 for program+vskew). Five more program genes were added to compensate; 65 of the 100
  program genes are now readable from base panel + add-on, up from 60.

All 16 TRBV genes are parked in the CSV with this reason recorded. **Revive them only if 10x supports a
conserved-region family probe** - that would make the readout family-level and clone-independent, which
is a different and much stronger instrument. It remains the open question in section 7.

### 2.2 The designer's criterion is breadth, not level

Absent from the base panel: every HLA gene, B2M, PTPRC, all keratins but KRT20, COL1/3/6, DCN, LUM,
SPARC, FN1, S100A8/9, LYZ, C1Q, CCL5, GNLY, NKG7, TIMP1, ACTA2, VWF, TPSAB1, CPA3, JCHAIN, and every
TR gene except TRDC. The first draft of this section read that as an optical-crowding rule on
**abundance**. The designer's own output says otherwise.

**2026-11-04 — eight rejections came back:**

| gene | flag |
|---|---|
| JUNB NFKBIA ZFP36 B2M DUSP1 FOS JUN | Broadly expressed |
| HLA-DRA | Potentially problematic |

Every one of the seven is transcribed in *most* cell types. Meanwhile the list's most abundant
transcripts passed — S100A9, DCN, C1QB, ACTA2, APOE, TIMP1, GNLY — and each of those is confined to one
lineage. So the constraint is **how many cell types a probe fires in, not how high it goes in the cells
that express it.** Lineage-restricted-and-very-high is fine; ubiquitous-and-moderate is not.

`abundance_risk` in the CSV now carries that meaning. Under it, the exposed genes are **ENO1, CSTB,
LGALS1, S100A4, MIF, CD63, BSG, CTSB, CD74, ITGB1, ARPC5, LMNA**, plus **TRAC, TRBC1, TRBC2** (T-restricted
but very high). That is the set to expect in the next rejection round — not S100A9 or DCN.

**What the eight cost.** The signaling readouts are multi-gene z-scores over
`tcr_signaling_helpers.MODULES`, not single-gene calls, so they still compute:

| module | size | readable before | after |
|---|---|---|---|
| `L3a_nfkb_activity` | 19 | 15 | 14 |
| `L3c_ap1_mapk_activity` | 13 | 11 | 7 |
| `L4_ieg_acute` | 19 | 16 | 12 |

The AP-1 module takes the real damage: FOS, JUN, JUNB and DUSP1 go at once, leaving FOSB, EGR1, DUSP2
and IER2 from the add-on against ATF3, DUSP5 and DUSP6 on the base panel. The module still scores, but
any per-gene AP-1 statement now rests on FOSB. **One result does not survive at all:** JUNB carried both
the 19p13 gain and MRVI stage +0.34, and nothing else on the panel reports it.

Two losses are covered by the base panel rather than by a replacement probe. Class II survives as
**CD74 (add-on) + CIITA (panel)**, which is what apCAF identity actually needs. MHC-I falls back to
**NLRC5, TAP1 and TAP2**, all three on the panel — but note the asymmetry: those report transactivator
and transporter loss, whereas B2M loss in CTCL is usually a B2M-specific mutation or deletion. The
escape axis is narrowed, not preserved.

## 3. The 100

| group | n | purpose |
|---|---|---|
| A TCR constant | 5 | clone-agnostic clonality (TRBC1/2 ratio) + γδ identity |
| B malignancy program | 21 | malignant vs reactive CD4 — now the sole in-situ caller |
| C surface targets | 12 | CAR-T / bispecific candidates |
| D subclone signaling | 11 | AP-1 / NF-κB / NFAT, in situ |
| E motility / CosMx | 6 | amoeboid phenotype + CosMx replication |
| F degradome / CCC | 25 | protease–matrix axis, specific ligand–receptor edges |
| G TME annotation | 11 | identity genes with no on-panel substitute |
| H clone secreted | 4 | clone-associated secreted/surface |
| I stage / CNV | 5 | stage axis + immune escape |

### A. TCR constant / clonality — 5
`TRBC1 TRBC2 TRAC TRGC1 TRGC2`

**TRBC1/TRBC2 is the clonality instrument** (§2.1): clone-agnostic, works on an unknown clone, and
unobtainable from the base panel, which carries no TR gene except TRDC. Both probes must be designed in
the UTRs. TRAC is the denominator for that ratio and is the weakest member of this block — CD3E and CD2
on the base panel partly cover αβ identity, so TRAC is the first thing to give up if 10x rejects a
high-abundance gene. TRGC1/2 rescue the γδ entity that `mal_cluster` currently drops entirely.

### B. Malignancy program — 21
Up: `KIR3DL2 DNM3 GTSF1 KRT7 ZNF503 HMSD TMPRSS3 CD9 IGFBP4`
Down: `PLAC8 C1orf162 LGALS3BP GRAP2 GLIPR1 PCED1B IL7R`
Added when the TRBV block was dropped: `TP53I3 LHFPL6 PYCR1 SCCPDH PTPRN2` (skin, 8–9 folds each,
up in 79–90% of donors)

Only 44 of the 100 learned program genes are on the base panel; this block brings it to 65.
**With vskew gone, the program is the sole in-situ malignancy caller**, which is why it was the first
claim on the freed slots. **KIR3DL2 is the only missing positive
control** (blood paired rank 1/1030, lfc 2.79; skin clone pct 1.0, q8e-5) — KIR3DL1 *is* on the panel, so
flag the pair for cross-hybridisation. DNM3 is up in 100% of blood donors and 1.54 in skin.

### C. Surface targets — 12
`ITGB1 CPM TNFRSF25 CD52 LGALS3 COL6A3 SLC39A6 SLC4A2 TNFRSF4 TSPAN2 EVC2`
Added 2026-11-04: `SLC1A4` (skin donor-cluster separator, 0.690 paired).

ITGB1 has the highest pooled AUROC of any gene tested (.739) and is ADT-confirmed. CD52 is a
**protein-only hit** (ADT q.028, transcript ns) — the transcript is on the list to test whether in situ
mRNA tracks it at all. COL6A3 doubles as a segmentation-bleed monitor. **TNFRSF4 is an anti-target**:
clone up q4e-5, but its worst off-tumour arm is Treg at 55.6%, so it is here to be ruled out.

### D. Subclone signaling — 11
`TNFAIP3 NFKBIZ BIRC3 FOSB EGR1 DUSP2 IER2 RCAN1 CCR10 ENO1`
Added 2026-11-04: `NR4A2` — NR4A1 and NR4A3 are already on the base panel, so one probe completes the
trio and rebuilds part of both `L4_ieg_acute` and `L5_chronicity_exhaustion`.
Rejected 2026-11-04: `NFKBIA FOS JUN JUNB DUSP1 ZFP36` (§2.2).

No further AP-1 or IEG gene was promoted, because there is none left that is both off-panel and
restricted: the module is *made of* immediate-early genes, which is exactly the class the designer
rejects. Adding DUSP4 or JUND would have meant inventing evidence, and §5's rule is that a gene with no
result behind it does not get a slot.

AP-1 δ −0.23 and immediate-early δ −0.22; "AP-1 low vs NFAT" is the dominant mode in 98 of 178 subclones.
`docs/24_subclone_tcr_signaling_vs_old_verification.md` marks these genes **dissociation-labile and asks
for validation on non-dissociated tissue** — FFPE in situ is exactly that experiment, which is why this
block is large. TNFAIP3 is both the 6q23 driver and the strongest CNV×signaling link (ρ .119); JUNB adds
19p13 gain and MRVI stage +0.34; ENO1 is the most divergent subclone axis (83% of donors) plus 1p36 gain.

### E. Motility / CosMx — 6
`LMNA ARPC1B ARPC5 KLF2 CD69`
Added 2026-11-04: `MYL9` (CosMx q.022) — it doubles as a smooth-muscle identity gene, which matters if
ACTA2 is rejected in a later round.

All survive the depth-matched rerun (LMNA +.71 q5e-10; ARPC1B +.35 q1e-11). LMNB1 is already on-panel, so
LMNA gives the lamina ratio. KLF2 and CD69 replicate the CosMx epidermal/dermal split on a new platform.

### F. Degradome / CCC — 25
`TIMP1 CD63 BSG MMP2 TIMP3 PRSS23 CTSB CTSS MMP1 MMP3 MMP19 CEMIP FBLN1 CXCL14 OSMR C3 DCN CD74`
Added when the TRBV block was dropped: `MMP25 SLPI OGN MFAP4`
Added 2026-11-04: `FBN1` (elastic fibre, up q≤2e-3), `CCL2` (F_structural q3e-4, Mono_Mac down q8e-5),
`ADAM19` (DC q2e-5, clone q.03). These took three of the six signaling slots because they are
lineage-restricted and carry numbers, which the remaining AP-1 candidates do not.

MMP2 is 12.2× higher in fibroblasts than in T cells in CosMx (q9e-12 over 38 pieces) — a cell-type
attribution that only in situ can settle. CXCL14 is the largest fibroblast shift in the study (Δ −2.43,
q2e-4). OSMR carries IL11→OSMR and CD74 carries 5 of the **8 specific CCC edges**. TIMP1 + CD63 + ITGB1
are the survival axis, and all three are in the 100.

### G. TME annotation — 11
`S100A9 LYVE1 C1QB SPP1 APOE SERPINA1 ACTA2 MYH11`
Rejected 2026-11-04: `HLA-DRA`. apCAF class II is now read as **CD74 (group F) + CIITA (base panel)**.
Added when the TRBV block was dropped: `TYROBP FCER1G GNLY` — the NK score is down to 4 of 8 genes on
the base panel, and NK cells express CD8A as a homodimer, so cytotoxic NK arrive labelled CD8. In blood
that mislabels ~25% of CD8 cells; in skin it contaminates the CD8 arm of every TME comparison.

These are identity genes with **no on-panel substitute**: myofibroblast/SMC loses ACTA2, TAGLN, MYH11, DES, CNN1, ACTG2 and LMOD1 at once, leaving
only MYLK and SYNPO2; Mono_CD14 is down to FCN1+CD14; Mac_FOLR2 has no C1Q gene; Mac_SPP1_TREM2 keeps
only TREM2 and ACP5. Everything else in `22_myeloid_fibro_reannotation` has a working substitute and
was left off — see §5.

### H. Clone secreted/surface — 4
`CSTB MIF LGALS1 S100A4`

### I. Stage / CNV — 5
`IGFL2 SPINT2 GPAT3`
Added 2026-11-04: `CYP7B1` (+4.49, advanced program P4), `HEYL` (+5.48, Notch — HES1 is on the panel).
Rejected 2026-11-04: `B2M` (§2.2) — the MHC-I axis falls back to NLRC5/TAP1/TAP2 on the base panel.

IGFL2 is stage +5.40 (padj .001) and the top gene of the advanced-stage program P4 (usage .04 → .22).

## 4. Reserve policy

`reserve_rank` 1–10, re-ranked 2026-11-04 **by breadth rather than by evidence strength**, because a
reserve gene that trips the same filter is not a reserve: `KRT86 LEMD1 S100A8 CCL5 RPS6KB1 ARHGDIB
CHMP4B LMNB2 FN1 EZR`. Ranks 5–10 are broadly expressed and will probably be rejected too; they are
kept because a rejection costs a round-trip and a missing gene costs an experiment. Fill in rank order,
with one exception: if the rejection is in group G, prefer the reserve gene covering the same cell type
(S100A8 for Mono_CD14).

**If more than four further rejections come back, do not work down the tail of that list.** The better
source is the 11 parked blood-only leads — GPR15, PDE7B, NPAS3, CA8, GNG4, APBA2, CDO1, TPO, HACD1, ID3,
GSTP1 — which are lineage-restricted and carry real blood-screen numbers. Using them means reopening the
skin-only scope decision, so that is a call to make deliberately, not a default.

Parked separately, with reasons in the CSV: the 16 TRBV genes (§2.1), 11 blood-only leads (TPO, GPR15,
HACD1, PDE7B, ID3 …) that become relevant only if a blood or marrow section is added, and the
CNV/MAIT/γδ-subset genes that lost to higher-evidence competitors.

## 5. What was deliberately excluded

- **Roster padding — no result behind them:** MMP7/10/15/16/17/23B/24/26/28, CTSE/F/O/V, all eight KLKs,
  TMPRSS4, TMPRSS11D, PRSS8/21/22, PCSK5/6, ADAM15, ADAMTS3/7/8/9/13/14/20, CST6, CSTA, SPINK5, PI3, A2M,
  SERPINB8, SERPING1, HAS1/3, HYAL1/3, LOXL3/4, COL4A6/5A3/13A1/16A1/24A1, FBN2, THBS4, EFEMP2, LAMA5,
  ITGA9/11, SDC1, PRELP.
- **On-panel substitute exists:** CPA3/TPSAB1/TPSB2 → KIT, MS4A2, CMA1, HDC · CCL21 → PROX1 ·
  RAP1B → RAP1A · TLN1 → FERMT3, ILK · WNK1 → OXSR1, STK39, SLC12A2 · keratins → DSG3, TP63, COL17A1,
  CDH1, DMKN (that combination builds the epidermis mask) · CD3D → CD3E, CD2 · HLA-DRB1 and HLA-DPA1 →
  polymorphic and redundant with HLA-DRA.
- **Artefact-shaped or too low to detect:** RAB25 (epithelial gene topping a blood T-cell screen),
  the EPCAM/CDH1/CLDN set (segmentation-bleed magnets in skin), MBP (worst off-tumour arm NK 78%),
  STAM (cytosolic ESCRT-0 adaptor, no surfaceome evidence), VDAC1, and every COPA/bimodality hit with
  `median_pct_expr < 0.03` — GRIK2, OPCML, PTCHD1, SCN2A, ADAM29, GALNT9, the PCDH cluster. NB47 itself
  calls bimodality "a weak instrument at this donor count."
- **Fails the depth-matched rerun:** ACTR2/3, ARPC2/3/4, CHMP2B, VPS4B, WIPF1, AQP3.

## 6. Caveats that travel with this panel

1. **Both artefact nulls refused to fire.** NB42 §2a and NB43 §2a report `null_calibrated = False`
   (lineage controls median log2FC +0.00), so **no gene was excluded on contamination grounds in either
   screen** — contamination-shaped hits survived by default, not by passing a test. Treat group C and F
   ranks as provisional until the in situ data can separate the cell types directly. That separation is
   a large part of what this panel is for.
2. **IL2RA fails as a positive control in both compartments** (skin rank 2180/2115, blood 1003/1030, both
   wrong sign) because sIL-2R is shed and ADT_CD25 sits at isotype level. **Do not use IL2RA to validate
   the add-on.** Use KIR3DL2 (in the 100) and CCR4/TNFRSF8 (already on the base panel).
3. Only 25 of 124 ADT markers cleared isotype in NB45, and **ADT_CD158k (KIR3DL2) was one of the failures** —
   its flat protein delta is a non-binding clone, not protein contradicting the transcript. TIGIT is on
   the base panel but is the largest transcript→protein overstatement we measured (transcript 1.97,
   protein q .582); do not read it as a surface readout.
4. Skin cell-level separation is weak — every pooled AUROC in NB48 is .51–.60 — and no stratification term
   is significant anywhere in NB47 or NB44. **This add-on buys spatial context and clone identity, not a
   better per-cell classifier.** Expect to retrain any classifier on the panel's gene set.

## 7. Before ordering

- [ ] Submit the revised 100 and expect a second rejection round. The predicted casualties, under the
      breadth criterion established in §2.2, are ENO1, CSTB, LGALS1, S100A4, MIF, CD63, BSG, CTSB, CD74,
      ITGB1, ARPC5, LMNA, TRAC, TRBC1, TRBC2.
- [ ] **TRBC1/TRBC2 are the ones to fight for.** If the designer flags them, the panel loses its only
      clone-agnostic clonality readout and there is no substitute at any price — escalate rather than
      backfill. TRAC is the opposite case: give it up first, CD3E and CD2 on the base panel partly cover
      αβ identity.
- [ ] Ask whether a **conserved-region TRBV family probe** is supported (§2.1). The 16 parked TRBV genes
      are revived only on a yes — a family-level probe is clone-independent, which per-gene probes are not.
- [ ] Flag for cross-hybridisation: **TRBC1 vs TRBC2** (UTR probes — the clonality readout depends on
      telling them apart), KIR3DL2 vs KIR3DL1, C1QB vs C1QA.
- [ ] Gate: the final list still contains KIR3DL2, and CCR4/TNFRSF8 are left to read off the base panel.
- [ ] Optional, strengthens the CNV slots: per-arm panel density via `delta_axis_helpers.arm_annotate`
      against `data/cache/Homo_sapiens.GRCh38.110.chr.gtf.gz`. **Submit as a bsub job — not the login node.**
      Recurrent skin arms are 5p− 62, 7q+ 49, 16p+ 48, 17q+ 47, 10q− 47, 7p+ 44, 6q− 41 of 101 donors;
      the subclone trunk is 7p/7q gain.
