# TCR / co-stimulation gene programs — lists and interpretation

Two data types are in play, and they see different things:

- **RNA (expression)** — reports *pathway activity and cell state*. This is where you read which arm of the TCR/costim cascade is firing.
- **inferCNV (arm/chromosome-level)** — reports *gene dosage over a whole chromosome arm*. It cannot resolve a single gene; a gain of an arm covers hundreds of genes, so attributing an arm event to one pathway node is *suggestive*, not proof. Attribution strengthens when (i) the arm is recurrently altered across donors, (ii) the matched downstream activity co-segregates, and (iii) it is one of the known MF recurrent arms.

**The one circularity to keep in mind.** inferCNV derives the arm call from smoothed regional expression. So a component gene's own RNA level and the inferCNV call for its arm are largely the *same measurement*, not two independent confirmations. The genuinely independent cross is: **arm-level CNV (dosage) × downstream activity-module score (function)** — because the activity-module target genes sit on *other* chromosomes and report pathway output, not local dosage. Lean on that cross; treat "component" lists as context, not as a second vote.

Because of this, each node below is given as two kinds of list:

- **components (dosage)** — the machinery itself. Overlaps with inferCNV for its own arm; use mainly to know *which arm to look at*.
- **activity (function)** — targets + feedback regulators, scattered across the genome. This is the load-bearing readout: a point-mutation driver (e.g. PLCG1 S345F) is invisible to dosage but shows up here as an arm's output raised *out of proportion to its inputs*.

Score each list as a module (per malignant cell), then aggregate to subclone **within donor** (trunk/branch; MrVI major/minor). All comparisons should be malignant-vs-malignant across subclones, or malignant-vs-reactive-CD4 — never malignant-CD4 vs the CD8 inferCNV reference (lineage confound, see caveats).

---

## 1. The gene lists

Each block: **the genes**, then what a high score means, then the chromosome arm(s) of any driver/dose-relevant genes so you can cross with inferCNV.

### Level 0 — receptor complex & T-cell identity

**L0_tcr_cd3_components**
`CD3D, CD3E, CD3G, CD247, TRAC, TRBC2`
Dose of the receptor complex. Mostly context. (CD247 = CD3ζ; TRAC/TRBC are clonotype-linked.)

**L0_pan_t_identity_loss**
`CD2, CD5, CD7, CD6, CD52`
The MF immunophenotype. Here a **drop** is the signal, not a rise — CD2/CD5/CD7 loss is the classic malignant phenotype; test as reduced score in malignant vs reactive-CD4. CD5 is also a negative-feedback rheostat of TCR strength.

**L0_coreceptor (LINEAGE — context only)**
`CD4, CD8A, CD8B`
Do not use for signaling contrasts; CD4/CD8 differences are lineage, not activation.

### Level 1 — proximal kinases/phosphatases (LINEAGE-confounded)

**L1_proximal_kinases (context only)**
`LCK, FYN, ZAP70, ITK, TXK, CSK, PTPN6, PTPN22, PTPRC, UBASH3A, UBASH3B`
ZAP70/LCK are conduits, not drivers, and ZAP70 is intrinsically lower in CD4 than CD8 — so these read as "differential signaling" against a CD8 reference for lineage reasons. Report but do not interpret as malignancy signal. (FYN sits on **6q** — a recurrent loss arm — so its dosage is incidentally informative about 6q status, see §3.)

### Level 2 — LAT signalosome, GEFs, and the PLCG1 pivot

**L2_lat_signalosome_components**
`LAT, LCP2, GRAP2, GRB2, THEMIS, NCK1, WAS, WIPF1`
Scaffold assembly. Dose/context.

**L2_gef_gtpase_components**
`VAV1, VAV2, VAV3, SOS1, RASGRP1, RASGRP2, RHOA, RAC2, CDC42`
GEF/GTPase layer. Recurrent drivers here: **VAV1 (19p)**, **RHOA (3p)**, **RAC2 (22q)**.

**L2_plcg1_node_components**
`PLCG1, PRKCB, PRKCQ, DGKA, DGKZ`
The pivot into both downstream arms. **PLCG1 (20q)** is the most recurrent TCR-effector lesion (activating S345F/S520F); **PRKCB (16p)**. DGKA/DGKZ are DAG-consuming brakes. Read PLCG1-node status through its *outputs* (NFAT and NF-κB activity), since GOF mutations don't raise PLCG1 RNA.

### Level 3a — DAG → PKCθ → CBM → canonical NF-κB

**L3a_cbm_nfkb_components**
`PRKCQ, CARD11, BCL10, MALT1, TRAF6, TRAF2, MAP3K7, TAB1, TAB2, CHUK, IKBKB, IKBKG, NFKB1, NFKB2, RELA, RELB, REL`
CBM signalosome + IKK/NF-κB machinery. Drivers: **CARD11 (7p)**, **BCL10 (1p)**, **MALT1 (18q)**, **NFKB2 (10q)**, **REL (2p)**.

**L3a_nfkb_activity** — *load-bearing*
`NFKBIA, NFKBIE, NFKBIZ, TNFAIP3, TNFAIP8, BIRC3, BIRC2, TRAF1, CFLAR, BCL2A1, CD83, ICAM1, CD40, TNF, LTB, CCL3, CCL4, NFKB2, RELB`
High = canonical NF-κB is firing (survival/proliferation arm). NFKBIA, TNFAIP3(A20), NFKBIZ are feedback inhibitors *induced by* NF-κB, so their presence confirms active (not absent) signaling. A subclone high here **with low acute-IEG (L4)** points to a distal, antigen-independent CBM lesion.

### Level 3b — IP3 → Ca²⁺ → calcineurin → NFAT

**L3b_calcium_nfat_components**
`PLCG1, ITPR1, ITPR2, ITPR3, STIM1, STIM2, ORAI1, ORAI2, ORAI3, PPP3CA, PPP3CB, PPP3R1, CAMK4, NFATC1, NFATC2, NFATC3`
CRAC channel + calcineurin machinery. **PPP3CB (calcineurin) sits on 10q** (recurrent loss arm).

**L3b_nfat_activity** — *load-bearing*
`RCAN1, NFATC1, EGR2, EGR3, IRF4, BATF, IL2RA, CD40LG, PDCD1, CTLA4, TNFRSF9, TNFRSF18`
High = calcium/calcineurin/NFAT arm firing. **RCAN1 is the cleanest calcineurin readout** (direct NFAT target/feedback); anchor on the set, not RCAN1 alone (scRNA can't resolve the NFAT-responsive RCAN1.4 isoform). Note PDCD1/CTLA4/TNFRSF9/18 are NFAT-driven — they double as costim/exhaustion genes, so attribute deliberately. A subclone high here relative to NF-κB activity is calcium/PLCG1-arm-biased.

### Level 3c — RAS → MAPK → AP-1

**L3c_ras_mapk_components**
`RASGRP1, SOS1, KRAS, NRAS, BRAF, RAF1, MAP2K1, MAP2K2, MAPK1, MAPK3, MAPK8, MAPK9, MAPK14, MAP3K8`
MAPK machinery. Dose/context.

**L3c_ap1_mapk_activity**
`FOS, FOSB, JUN, JUNB, JUND, ATF3, EGR1, DUSP1, DUSP2, DUSP4, DUSP5, DUSP6, IER2`
AP-1 output; DUSPs are MAPK-induced feedback phosphatases (good specific readout). **Several of these are dissociation-labile** (FOS/FOSB/JUN/JUNB/JUND/EGR1/DUSP1/DUSP2/IER2) — see caveats.

### Level 4 — integrated acute trigger (immediate-early)

**L4_ieg_acute** — *dissociation-labile; validate on FFPE Flex arm*
`FOS, FOSB, JUN, JUNB, JUND, EGR1, EGR2, EGR3, NR4A1, NR4A2, NR4A3, DUSP1, DUSP2, IER2, IER3, ZFP36, CD69, TNF, MYC`
Reads *how recently/strongly* the cell was triggered. The single best discriminator of acute vs chronic signaling — but also the canonical warm-ischemia/dissociation artifact. In this atlas the FFPE Flex cohorts (no dissociation) are the negative control. CD69/NR4A1 also overlap your TRM markers.

### Level 5 — sustained state & chronicity

**L5_sustained_tf_layer**
`IRF4, BATF, BATF3, TBX21, GATA3, RUNX3, PRDM1, ID2, ID3, TOX, TOX2, ZEB1, IKZF1, IKZF2, IKZF3, MYB`
Late transcription-factor layer. **ZEB1 (10p)** loss (~56–65%) de-represses Th2 cytokines; **GATA3 (10p)** anchors Th2 (overlaps your Th2 program — score once); **TOX (8q)** is an MF marker and chronicity driver.

**L5_chronicity_exhaustion** — the antigen-experience axis
`TOX, NR4A1, NR4A2, NR4A3, PDCD1, TIGIT, HAVCR2, LAG3, ENTPD1, CTLA4, BATF, IRF4`
The partnerless-NFAT exhaustion program. High relative to acute-IEG = chronic antigen-experienced rather than acutely triggered — the molecular form of the "PD-1 as chronic TCR engagement" reading. Overlaps the Childs immune-evasion set (KIRs handled separately below).

### Co-stimulation

**COSTIM_receptors_activating** (on malignant cell)
`CD28, ICOS, TNFRSF4, TNFRSF9, TNFRSF18, CD27, TNFRSF1B, TNFRSF14, CD2, SLAMF1, CD226, CD40LG, TNFRSF8`
Signal-2 receptors. **CD28/ICOS (2q)** — CD28 GOF; **TNFRSF1B/TNFR2 (1p)** — gains/mutations in >1/3; TNFRSF4/9/18 (OX40/4-1BB/GITR, **1p**); TNFRSF8 = CD30 (transformation/brentuximab).

**COSTIM_coinhibitory_receptors** (brakes, on malignant cell)
`CTLA4, PDCD1, BTLA, LAG3, HAVCR2, TIGIT, CD160, VSIR, ENTPD1, KIR3DL2, KIR2DL3, KIR3DL1, KLRG1`
**CTLA4 (2q)** — the CTLA4–CD28 fusion class; **PDCD1 (2q)** — deletion in ~1/3 removes a brake; KIR3DL2/KIR2DL3/KIR3DL1 (**19q**) are the CTCL evasion receptors / CAR-T interest.

**COSTIM_ligands_on_TME_partners** (score on DC/fibroblast/B/macrophage, NOT malignant)
`CD80, CD86, ICOSLG, TNFSF4, TNFSF9, TNFSF18, CD70, TNF, CD58, PVR, NECTIN2, TNFSF14, TNFSF8`
Use for ligand–receptor scoring by tropism/stage. **TNF → TNFR2** (notably B-cell-derived TNF onto malignant TNFRSF1B) is the highest-value pair — it links costimulation to the atlas's B-cell/TLS finding. CD58 → CD2 (CD58 deletions are an evasion event).

**COSTIM_pi3k_akt_components**
`PIK3CD, PIK3R1, PIK3CA, PTEN, AKT1, AKT2, PDPK1, MTOR, RICTOR, RPTOR, RHEB, TSC1, TSC2`
Signal-2 amplifier machinery. **PTEN (10q)** loss is the key recurrent event here.

**COSTIM_pi3k_akt_activity**
`FOXO1, RPS6KB1, EIF4EBP1, MYC, CCND2, SLC2A1, BCL2L1`
PI3K/AKT/mTOR output. **FOXO1 targets fall as AKT rises** (so read FOXO1 inversely); mTOR targets (RPS6KB1, EIF4EBP1) and BCL2L1(BCL-XL) rise. High here + high costim receptors + available ligand = costimulation-amplified. Also the readout for **10q/PTEN loss**.

### Signal 3 (JAK-STAT — adjacent)

**SIGNAL3_jak_stat**
`IL2RA, IL2RB, IL2RG, JAK1, JAK3, TYK2, STAT3, STAT5A, STAT5B, IL4R, IL7R, IL15RA, SOCS1, SOCS3, CISH, PIM1`
**STAT3, STAT5A, STAT5B all sit on 17q** — a recurrent **gain** arm — so 17q gain is the cleanest arm-CNV↔pathway pairing in the whole map (see §3). SOCS1/3, CISH are STAT-induced feedback (confirm active signaling).

### Negative regulators (score alongside everything)

**NEGREG_rheostat**
`CD5, PTPN6, PTPN22, DGKA, DGKZ, CBL, CBLB, ITCH, TNFAIP3, CYLD, RC3H1, ZC3H12A, TNIP1, NFKBIA, RCAN1, UBASH3A, DUSP1, DUSP2, SOCS1, SOCS3, CISH`
High feedback-regulator expression means high-but-braked signaling, not low. Always read this next to the activity modules so induced feedback isn't misread as low activation. **TNFAIP3/A20 and CYLD sit on 6q** (recurrent loss) — their loss releases NF-κB.

---

## 2. Localizing a subclone across the pathway (RNA logic)

Compute these ratios/contrasts per subclone (within donor) from the activity modules above; the pattern *across sibling subclones* is the localization signal, not any absolute score.

- **calcium vs CBM bias** = L3b_nfat_activity ÷ L3a_nfkb_activity. High → calcium/PLCG1-arm-biased output. Low → CBM/NF-κB-biased.
- **autonomous NF-κB** = L3a_nfkb_activity relative to L4_ieg_acute. High NF-κB with **low** IEG → distal antigen-independent CBM lesion (CARD11/BCL10/MALT1). High NF-κB **with** high IEG → proximal/costim-driven (antigen/CD28-dependent, niche-addicted).
- **costim amplification** = COSTIM_pi3k_akt_activity relative to L4_ieg_acute, read with COSTIM_receptors_activating and TME ligands. High → CD28/TNFR2-amplified.
- **chronic vs acute** = L5_chronicity_exhaustion ÷ L4_ieg_acute. High → chronic antigen-experienced (partnerless NFAT), not acutely triggered.
- **AP-1 vs NFAT** = L3c_ap1_mapk_activity ÷ L3b_nfat_activity. Low (NFAT high, AP-1 low) is the partnerless-NFAT chronic/anergic signature.

Cluster subclones by this contrast vector to see whether the atlas contains distinct *signaling-mode classes* (calcium-biased / autonomous-NF-κB / costim-amplified / chronic-exhausted).

---

## 3. Arm-level inferCNV integration (this is where dosage meets function)

Cross each subclone's arm gains/losses against the matched **activity** module. The strongest, most independent inferences are the recurrent MF arms that happen to carry a pathway node. Direction matters — some events are gains that add a driver, some are losses that remove a brake.

| Arm event (recurrence in MF) | Pathway node on that arm | Direction of effect | Cross against (RNA activity) | Interpretation if concordant |
|---|---|---|---|---|
| **17q gain** (recurrent) | STAT3, STAT5A, STAT5B (17q21) | gain adds STAT dose | `SIGNAL3_jak_stat` | signal-3 / JAK-STAT amplification — the cleanest pairing in the map |
| **10q loss** (recurrent) | PTEN (10q23); also PPP3CB, NFKB2, CHUK | loss removes the PI3K brake | `COSTIM_pi3k_akt_activity` (FOXO1 down, mTOR up) | PI3K/AKT amplification via PTEN loss |
| **6q loss** (recurrent) | TNFAIP3/A20 (6q23), CYLD-adjacent; FYN, PRDM1, MAP3K7 | loss removes an NF-κB brake | `L3a_nfkb_activity` | de-repressed NF-κB out of proportion to input (A20 loss) |
| **8q gain** (recurrent) | TOX (8q12), MYC (8q24) | gain adds dose | `L5_chronicity_exhaustion` + proliferation | chronicity/exhaustion (TOX) and proliferative drive (MYC) |
| **2q (status, non-recurrent direction)** | CD28/ICOS/CTLA4 (2q33), PDCD1 (2q37) | focal gain → costim; the fusion class arises here | `COSTIM_receptors_activating`, `COSTIM_pi3k_akt_activity` | costimulation amplified / CTLA4–CD28 shadow |
| **20q (status)** | PLCG1 (20q12) | gain adds PLCG1 dose (GOF invisible to CNV) | `L3b_nfat_activity` | calcium/NFAT-biased output |
| **1p loss (recurrent)** | TNFRSF1B/TNFR2, TNFRSF4/9/18 (1p36); BCL10 (1p22); CD58 (1p13) | loss *removes* costim receptors and CD58 | `COSTIM_receptors_activating`; CD58→CD2 LR | **tension:** literature reports TNFR2 *gains* in a subset, yet 1p is a recurrent *loss* arm — resolve per subclone, don't assume |

Not TCR/costim but worth carrying as context because they sit on the same recurrent arms: **9p21 loss → CDKN2A** (cell cycle), **13q14 loss → RB1** (cell cycle — note RB1 is 13q14.2, *not* 5q, the recurring project mislabel), **17p loss → TP53** (DNA damage).

**How to use the table.** For each subclone: (1) list its arm gains/losses from inferCNV; (2) for any arm carrying a node, check whether the matched activity module is elevated *in the concordant direction*; (3) an event is credible when it recurs across donors and the downstream activity co-segregates. Because activity-module genes sit on other arms, "17q gain + high JAK-STAT activity" is two independent measurements agreeing — that is the inference you can stand on. "20q gain + high PLCG1 RNA" is *not* two independent measurements (inferCNV read the 20q gain partly from PLCG1's own expression), so validate that one through NFAT *activity* instead.

---

## 4. For the coding agent — order of operations & caveats

Order:
1. Score all lists as modules on the malignant compartment (rank-based, e.g. UCell/AUCell, for robustness across the 5′ and FFPE Flex chemistries). Include NEGREG_rheostat.
2. Aggregate to subclone × module means **within donor** (trunk/branch; MrVI major/minor).
3. Compute the §2 contrasts per subclone; cluster subclones into signaling-mode classes.
4. Pull each subclone's arm gains/losses from inferCNV; cross against §3; test whether mode class co-segregates with the matched arm event across donors.
5. Sensitivity: repeat 1–3 with the dissociation-labile genes removed and confirm the mode structure survives (FFPE Flex arm as the check).

Caveats that must be respected:
- **Dissociation artifact.** L4_ieg_acute and the IEG genes in L3c are induced by handling; validate against the FFPE Flex (no-dissociation) cohorts before interpreting; don't let an IEG-based conclusion carry weight. Dissociation-labile set: FOS, FOSB, JUN, JUNB, JUND, EGR1, NR4A1, NR4A2, NR4A3, DUSP1, DUSP2, ZFP36, IER2, CD69.
- **Lineage confound.** CD4/CD8A/CD8B and ZAP70/LCK differ CD4-vs-CD8 for lineage reasons; never contrast malignant-CD4 against the CD8 inferCNV reference for signaling — use malignant-vs-malignant across subclones or malignant-vs-reactive-CD4.
- **inferCNV circularity.** Component/dose modules and the arm call are partly the same measurement; the independent cross is arm CNV × downstream *activity*.
- **Ambient contamination.** GZMB and reactive-CD8/NK transcripts bleed into malignant calls in droplet data; any GZMB↔NFAT test runs on CNV+/clonotype-confirmed malignant cells only.
- **Program overlaps** (attribute once, deliberately): GATA3 ↔ your Th2 program; CD69/NR4A1 ↔ Rindler TRM markers; KIRs/PDCD1/TIGIT/LAG3/HAVCR2 ↔ Childs evasion set.
- **RCAN1** anchors NFAT activity but scRNA can't resolve its NFAT-responsive isoform — read the set, not the single gene.

The end product is a per-subclone signaling-mode label plus its concordant arm-CNV event — i.e. a direct answer to whether subclones concentrate dysregulation at different levels of the TCR/costimulation cascade, and whether that co-varies with copy number at the corresponding arm.
