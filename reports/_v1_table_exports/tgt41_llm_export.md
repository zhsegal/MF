> **Annotation vintage: v1.** Built on `cell_type_final == "CD4"`, whose benign CD4 arm is only
> ~46% conventional CD4 in skin — 16% CD8, 11% Treg, 8% NK/ILC, 15% unresolved
> (`reports/lineage_check/`). Statements about the malignant side stand; statements about the
> *benign* side, and any gene that separates CD4 from CD8/NK/Treg, are confounded. The current
> lineage call is `data/atlas_joint/nonmal_T_labels_v1.parquet`, read via `helpers/lineage.py`.
> See `KNOWLEDGE/RESULTS.md`.
======================== COPY FROM HERE ========================

# 41 — The surface-target panel and the surfaceome reference

This block is a complete, self-contained numerical report from one notebook of the MF/CTCL
surface-target project. Everything another model needs in order to reason about these numbers is
inside it: what was measured, on which cohort, with which test, and the result tables themselves.
No file outside this block has to be read. Numbers are given to 3 significant figures; tables are
CSV with a header row. A column named `q` is a Benjamini-Hochberg adjusted p-value unless the
section says otherwise.

## Context

This notebook builds the ROSTER for the MF/CTCL surface-target screen -- which genes are
eligible to be screened and what is already known about them. It touches no atlas counts; the
screening itself is notebooks 42/47/48 (skin) and 43/44/45/46 (blood).

The screen gene set is the curated a-priori panel UNION the confident surfaceome, intersected
with the genes each compartment's object actually carries. `surface_confident` means two
independent sources, or one source that is experimental evidence rather than prediction -- a
plain `n_evidence >= 2` gate would drop KIR3DL2, the most Sezary-restricted antigen known.

Sources: SURFY (ML over UniProt), CSPA (mass spectrometry of labelled surface glycoproteins on
intact cells -- the only direct evidence of reachability), HPA immunofluorescence, and GO
cell-surface. GO plasma membrane is deliberately excluded.

The governing caveat for everything downstream: mRNA is not surface protein.

## §1 Panel by functional class

**What:** the a-priori panel: how many genes in each of the 12 classes.

**How:** hand-curated in helpers/surface_targets.py; the class carries half the interpretation (a pan-T hit has an aplasia cost a CTCL-associated one may not).

**Full table on disk:** `tables/tgt41_panel.csv`

```csv
key,n_genes
ctcl_associated,23
integrin_adhesion,18
chemokine_receptor,16
cytokine_receptor,15
kir_lectin_nk,14
tcr_core,13
checkpoint,12
tnfrsf_costim,12
coreceptor,8
pan_t,8
cla_biosynthesis,7
transporter,5
```

## §2 Surfaceome evidence

**What:** how many genes each source supports, and how many clear the confidence gate.

**How:** counts over the merged reference.

```csv
key,n_genes
in_surfy,2799
in_cspa,1443
in_hpa_pm,2263
in_go_cs,1164
```

## §2 Genes per number of independent sources

**What:** the distribution of evidence depth.

**How:** value counts of n_evidence.

**Full table on disk:** `tables/tgt41_surfaceome_summary.csv`

```csv
n_evidence,n_genes,n_cspa,n_surfy_curated,n_confident
1,3890,471,594,783
2,990,491,343,990
3,409,338,217,409
4,143,143,100,143
```

## §3 Coverage per compartment

**What:** how many genes each compartment's object carries, how many are screened, and which panel genes or positive controls are missing from it.

**How:** intersection of the screen set with the object's var names. A gene absent here is absent from that compartment's result, which is not the same as unexpressed.

**Full table on disk:** `tables/tgt41_coverage.csv`

```csv
tissue,genes_in_object,n_screened,panel_present,panel_total,panel_absent,controls_absent
skin,42347,2115,142,143,KIR2DL2,
blood,10000,1030,116,143,BTLA;CD164;CD3G;CD4;CD44;CD53;FUT4;GCNT1;ICOSLG;IL12RB1;IL15RA;IL31RA;ITGA2;ITGA5;ITGA6;ITGAE;ITGB7;KIR2DL2;LY75;PDCD1LG2;SLC2A1;SLC3A2;ST3GAL4;ST3GAL6;TNFRSF14;TNFRSF1A;TNFRSF8,TNFRSF8
```

## §4 Trafficking machinery

**What:** the 11 phases between a transcript and a displayed epitope, and their size.

**How:** curated gene sets; scored per arm in the screens as context, never as a per-gene correction factor.

```csv
key,n_genes
er_translocation,17
er_folding_qc,26
er_to_golgi,28
glycosylation,36
endocytosis,18
recycling,20
retrograde,23
ubiquitin_proteostasis,27
snare_fusion,18
shedding,15
escrt_exosome,20
```

## §5 Clinical precedent

**What:** every panel gene with an agent in CTCL or mature T-cell neoplasms.

**How:** verified against ClinicalTrials.gov and PubMed, 2026-10-22. The four positive controls are ['CCR4', 'TNFRSF8', 'KIR3DL2', 'IL2RA']: a ranking that does not put them near the top is miscalibrated.

**Full table on disk:** `tables/tgt41_clinical_precedent.csv`

```csv
gene,cd_name,agent,stage,indication,evidence,off_tumor_concern,positive_control
BSG,CD147,CD147 CAR-T,phase 1 (early),T-NHL,NCT05013372,"ubiquitous (RBC, endothelium) -- broad off-tumour",False
CCR4,CCR4,mogamulizumab,approved,MF/SS (MAVORIC),NCT04128072,"Treg depletion -> rash, autoimmunity, steroid-refractory GvHD post-allo",True
CCR8,CCR8,DT-7012,phase 1 (first-in-human),"R/R MF/SS -- CTCL-dedicated, started 2026-01",NCT07213882,depletes CCR8+ Tregs -> autoimmunity (also the intended mechanism),False
CD37,CD37,AGS67E (ADC); CAR-37 T,phase 1,lymphoid malignancies -- no CTCL-specific cohort,NCT02175433,"B/myeloid depletion, neutropenia",False
CD38,CD38,daratumumab; isatuximab,phase 2/3,"T-ALL, PTCL -- NOT MF",NCT07072585,"NK/plasma-cell loss, hypogammaglobulinaemia, blood-typing interference",False
CD4,CD4,zanolimumab (HuMax-CD4); LB1901 CD4 CAR-T,phase 3 (terminated),MF/SS,NCT00127881,profound CD4 aplasia -> opportunistic infection,False
CD47,CD47,"TTI-621 (SIRPaFc, intratumoral); magrolimab + mogamulizumab",phase 1b/2 -- both terminated,MF/SS,NCT02890368,RBC antigen sink -> anaemia; ubiquitous expression,False
CD5,CD5,CD5 CAR-T; MT-101 (CAR-monocyte),phase 1/2,PTCL and CTCL,NCT07022964,"fratricide, normal T depletion",False
CD52,CD52,alemtuzumab,phase 2 (off-label),SS / erythrodermic MF,NCT00057967,"pan-lymphocyte + monocyte depletion, CMV reactivation",False
CD7,CD7,WU-CART-007; BEAM-201 (base-edited),phase 1/2,T-ALL; MF-transformed / SS cohort,NCT05377827,fratricide + T/NK aplasia; CD7 is often already LOST on SS cells,False
CD70,CD70,INCA036873 (CD70xCD3); CTX130/CTX131,phase 1,MF/SS >= IIB cohort,NCT07195916,activated T/B + Treg loss; renal tubule CD70,False
CD84,CD84 / SLAMF5,CART84,phase 1/2,AML and T-ALL -- not yet CTCL,NCT07471789,platelet / B / monocyte expression,False
IL2RA,CD25,denileukin diftitox-cxdl (Lymphir); camidanlumab tesirine,approved (2024-08),R/R CTCL,NCT00050999,"capillary leak, Treg loss, visual impairment",True
IL31RA,IL-31RA,nemolizumab,case reports in CTCL,CTCL pruritus -- NOT cytoreductive,PMID 42662382,symptom control only,False
KIR3DL2,CD158k,lacutamab (IPH4102),phase 2,SS > MF (TELLOMAK),NCT03902184,narrow -- NK / CD8 subset depletion,True
PDCD1,PD-1,pembrolizumab,phase 2,MF/SS,NCT02243579,irAEs; PARADOXICAL rapid progression / flare reported in MF-SS and ATLL,False
TNFRSF1B,TNFR2,BITR2101; SIM1811-03; BI-1808,phase 1/2a,NHL incl. CTCL/PTCL,NCT06385522,Treg + activated-T depletion -> autoimmunity,False
TNFRSF8,CD30,brentuximab vedotin; AFM13 (CD30xCD16A),approved,CD30+ MF / pcALCL (ALCANZA),NCT01396070,MMAE neuropathy; CD30 on activated lymphocytes,True
TRBC1,TCR beta constant 1,AUTO4 (terminated); FG-M131 ADC; TRBC1 CAR-NK,phase 1/2,"PTCL, CTCL cohort",NCT07758491,spares the TRBC2 repertoire -- only TRBC1+ normal T are lost,False
```

======================== COPY TO HERE ==========================