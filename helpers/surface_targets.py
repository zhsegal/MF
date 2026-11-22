"""Surface-target discovery for CAR-T / bispecific antibodies in CTCL (nb41-45).

The question: which cell-surface protein separates the **malignant** T clone from the **reactive**
T cells sitting next to it, in skin and in blood. Not "what is high in tumour" -- a CAR against a
pan-T antigen works in the dish and causes T-cell aplasia in the patient, so the reactive arm is
the comparator that matters and the healthy compartments are the cost.

Three things this module holds, and one it refuses to hold:

  SURFACE_PANEL       the a-priori roster: T-cell surface molecules by functional class.
  TRAFFIC_MODULES     the intracellular machinery that decides whether a transcript becomes
                      surface protein at all -- translocation, folding, COPII, glycosylation,
                      endocytosis, recycling, retrieval, ubiquitination, fusion, shedding, ESCRT.
  CLINICAL_PRECEDENT  what has actually been given to a CTCL patient, with an NCT/PMID. The
                      positive controls of the whole screen: CCR4, TNFRSF8, KIR3DL2 and IL2RA are
                      approved or in registrational trials, so a screen that does not rank them
                      highly is broken, and that is the first thing to check.

  It does NOT hold a surfaceome. `data/external_ref/surfaceome/surfaceome.parquet`
  (SURFY + CSPA + HPA + GO) is built by `jobs/build_surfaceome_ref.py` and read with
  `load_surfaceome()`, because a genome-wide membership list is reference data, not curation.

**The caveat that governs every number downstream.** mRNA abundance is not surface density.
CTLA4 and PDCD1 are transcribed and then internalised by AP-2; SELL/CD62L is shed by ADAM17
within minutes of activation; MGAT5/ST3GAL1 glycans lock receptors into a galectin lattice that
changes their accessibility without changing their transcript. Nothing in a scRNA screen measures
epitope density, accessibility, turnover or shedding. The output of this pipeline is a ranked
hypothesis list for flow/CITE-seq validation, and it is never to be reported as a target call.

The blood compartment is the one place we can check. `blood_T_annotated.h5ad` carries a 128-marker
CITE-seq ADT panel in `obsm["X_adt"]` (names in `uns["adt_panel"]`), covering CD158k/KIR3DL2, CD30,
CD194/CCR4, CD25, CD7, CD26, CD70, CD5, CD52, CLA and more. The values are **measured integer
counts, not imputed**, and `ADT_TO_GENE` maps the panel to gene symbols so a transcript rank can be
scored against protein on the same cells.

Its scope is narrow and must be stated wherever it is used (measured 2026-10-22):

  * one study only -- **buus2025** -- 17 donors, **all Sezary syndrome**, ~9% of blood T cells.
  * **no healthy-control and no MF donor carries ADT**, so this validates malignant-vs-reactive
    *within SS patients* and says nothing about a healthy baseline or about skin.
  * every other cell is **NaN**, not zero. Mask on `np.isfinite` before any statistic; a mean
    over the column silently becomes a buus2025-only mean.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

NB_DIR = Path(__file__).resolve().parent.parent
DATA = NB_DIR / "data"
EXT_REF = DATA / "external_ref"
SURFACEOME_DIR = EXT_REF / "surfaceome"
SURFACEOME_PQ = SURFACEOME_DIR / "surfaceome.parquet"

TAB_DIR = NB_DIR / "tables"
FIG_DIR = NB_DIR / "figures" / "targets"
TAB_PREFIX = "tgt41_"
FIG_PREFIX = "tgt41_"


def tab(name: str) -> Path:
    """tables/tgt41_<name>.csv"""
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    return TAB_DIR / f"{TAB_PREFIX}{name}.csv"


def fig(name: str, ext: str = "png") -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR / f"{FIG_PREFIX}{name}.{ext}"


# ====================================================================== the a-priori panel
# Functional classes, not a flat list: a candidate's class is half its interpretation. A hit in
# `checkpoint` is a different claim from a hit in `trafficking_receptor`, and a hit in `pan_t`
# carries an aplasia cost that a hit in `ctcl_associated` may not.
SURFACE_PANEL = {
    # ---- TCR complex and lineage co-receptors.
    # TRBC1/TRBC2 are here for a specific reason: every ab-T cell expresses exactly one of the two,
    # so a TRBC1-directed agent spares the TRBC2 half of the normal repertoire. That is the only
    # clonality-based way to be tumour-selective with a pan-T antigen.
    "tcr_core": [
        "TRAC", "TRBC1", "TRBC2", "TRDC", "TRGC1", "TRGC2",
        "CD3D", "CD3E", "CD3G", "CD247", "CD4", "CD8A", "CD8B",
    ],
    # ---- Pan-T markers. High prevalence, low selectivity: the aplasia arm of the screen.
    # CD7 and CD5 are both in CTCL trials despite that, because CTCL cells often LOSE CD7 -- the
    # loss is the diagnostic, which makes CD7 a marker of the reactive arm as often as the tumour.
    "pan_t": ["CD2", "CD5", "CD7", "CD6", "CD28", "CD52", "CD96", "CD160"],
    # ---- Phosphatases, sialomucins and tetraspanins.
    "coreceptor": ["PTPRC", "CD27", "SPN", "CD47", "CD53", "CD81", "CD82", "ADGRE5"],
    # ---- Chemokine receptors: in CTCL these are the homing code, not decoration. CCR4 is an
    # approved target (mogamulizumab) and CCR8 entered a CTCL-dedicated phase 1 in 2026.
    "chemokine_receptor": [
        "CCR1", "CCR2", "CCR3", "CCR4", "CCR5", "CCR6", "CCR7", "CCR8", "CCR9", "CCR10",
        "CXCR3", "CXCR4", "CXCR5", "CXCR6", "XCR1", "CX3CR1",
    ],
    "integrin_adhesion": [
        "ITGA1", "ITGA2", "ITGA4", "ITGA5", "ITGA6", "ITGAE", "ITGAL", "ITGAM", "ITGAX",
        "ITGB1", "ITGB2", "ITGB7", "PECAM1", "SELL", "CD44", "ICAM1", "ICAM2", "VCAM1",
    ],
    "cytokine_receptor": [
        "IL2RA", "IL2RB", "IL2RG", "IL7R", "IL15RA", "IL18R1", "IL21R", "IL12RB1", "IL12RB2",
        "IL23R", "IL17RA", "IFNGR1", "IFNGR2", "TNFRSF1A", "TNFRSF1B",
    ],
    "tnfrsf_costim": [
        "TNFRSF4", "TNFRSF9", "TNFRSF8", "TNFRSF18", "TNFRSF14", "CD40LG", "ICOS", "CD226",
        "SLAMF1", "CD244", "SLAMF6", "SLAMF7",
    ],
    "checkpoint": [
        "CTLA4", "PDCD1", "HAVCR2", "LAG3", "TIGIT", "VSIR", "BTLA",
        "CD274", "PDCD1LG2", "ICOSLG", "CD80", "CD86",
    ],
    "kir_lectin_nk": [
        "KIR2DL1", "KIR2DL2", "KIR2DL3", "KIR3DL1", "KIR3DL2",
        "KLRB1", "KLRC1", "KLRG1", "KLRK1", "KLRD1",
        "SIGLEC7", "SIGLEC9", "ADGRG1", "CD101",
    ],
    # ---- Added here, not in the source roster. CTCL-specific biology and recent trial targets.
    "ctcl_associated": [
        "KIR3DL2",      # CD158k; lacutamab. The most Sezary-restricted antigen known
        "TNFRSF8",      # CD30; brentuximab vedotin, approved
        "CCR4",         # mogamulizumab, approved
        "CCR8",         # 2026 CTCL-dedicated phase 1; marks skin-resident memory T, the MF origin
        "TNFRSF1B",     # TNFR2; three phase-1 agents with CTCL/PTCL cohorts
        "IL2RA",        # CD25; denileukin diftitox
        "CADM1",        # the classic Sezary/ATLL surface gain
        "CD70", "CD38", "SLAMF7", "CD84",
        "DPP4",         # CD26 -- LOST on Sezary cells. A negative marker, never a target
        "GPR15", "ITGB7", "SELPLG",     # skin/gut homing
        "IL31RA", "IL4R",               # the pruritus axis; symptom biology, not cytoreduction
        "ENTPD1", "NT5E",               # CD39 / CD73 adenosine axis
        "CD200", "CD164", "LY75", "BSG",
    ],
    # ---- Metabolite transporters: surface, abundant on proliferating cells, and druggable.
    # Expect them to be high on any dividing cell -- they are a specificity test for the screen.
    "transporter": ["SLC3A2", "SLC7A5", "SLC1A5", "SLC2A1", "TFRC"],
    # ---- Skin-homing glycan biosynthesis. Not surface proteins themselves: they BUILD the CLA
    # epitope on SELPLG. Carried so a SELPLG result can be read against its machinery.
    "cla_biosynthesis": ["FUT7", "FUT4", "ST3GAL4", "ST3GAL6", "CHST2", "B3GNT2", "GCNT1"],
}

# Flat, de-duplicated, order preserved.
PANEL_GENES = list(dict.fromkeys(g for genes in SURFACE_PANEL.values() for g in genes))
GENE_TO_CLASS = {}
for _cls, _genes in SURFACE_PANEL.items():
    for _g in _genes:
        GENE_TO_CLASS.setdefault(_g, []).append(_cls)

# Negative markers: present in the panel, but a LOSS is the CTCL phenotype. Ranking these by
# "high in malignant" is backwards, and a screen that reports DPP4 as a target has a sign error.
LOSS_MARKERS = ["CD7", "DPP4", "CD5", "CD6", "CD27", "CD2", "CD26"]


# ====================================================================== clinical precedent
# Verified 2026-10-22 against ClinicalTrials.gov and PubMed. `stage` is the highest reached for
# that antigen in CTCL unless `indication` says otherwise. These are the screen's positive
# controls -- see the module docstring.
CLINICAL_PRECEDENT = {
    "CCR4": dict(cd="CCR4", agent="mogamulizumab", modality="mAb", stage="approved",
                 indication="MF/SS (MAVORIC)", evidence="NCT04128072",
                 tox="Treg depletion -> rash, autoimmunity, steroid-refractory GvHD post-allo"),
    "TNFRSF8": dict(cd="CD30", agent="brentuximab vedotin; AFM13 (CD30xCD16A)",
                    modality="ADC; bispecific NK-engager", stage="approved",
                    indication="CD30+ MF / pcALCL (ALCANZA)", evidence="NCT01396070",
                    tox="MMAE neuropathy; CD30 on activated lymphocytes"),
    "KIR3DL2": dict(cd="CD158k", agent="lacutamab (IPH4102)", modality="mAb", stage="phase 2",
                    indication="SS > MF (TELLOMAK)", evidence="NCT03902184",
                    tox="narrow -- NK / CD8 subset depletion"),
    "IL2RA": dict(cd="CD25", agent="denileukin diftitox-cxdl (Lymphir); camidanlumab tesirine",
                  modality="immunotoxin; ADC", stage="approved (2024-08)",
                  indication="R/R CTCL", evidence="NCT00050999",
                  tox="capillary leak, Treg loss, visual impairment"),
    "TRBC1": dict(cd="TCR beta constant 1", agent="AUTO4 (terminated); FG-M131 ADC; TRBC1 CAR-NK",
                  modality="CAR-T / ADC / CAR-NK", stage="phase 1/2",
                  indication="PTCL, CTCL cohort", evidence="NCT07758491",
                  tox="spares the TRBC2 repertoire -- only TRBC1+ normal T are lost"),
    "CD7": dict(cd="CD7", agent="WU-CART-007; BEAM-201 (base-edited)", modality="allo CAR-T",
                stage="phase 1/2", indication="T-ALL; MF-transformed / SS cohort",
                evidence="NCT05377827",
                tox="fratricide + T/NK aplasia; CD7 is often already LOST on SS cells"),
    "CD5": dict(cd="CD5", agent="CD5 CAR-T; MT-101 (CAR-monocyte)", modality="CAR-T / CAR-M",
                stage="phase 1/2", indication="PTCL and CTCL", evidence="NCT07022964",
                tox="fratricide, normal T depletion"),
    "CD70": dict(cd="CD70", agent="INCA036873 (CD70xCD3); CTX130/CTX131",
                 modality="bispecific / allo CAR-T", stage="phase 1",
                 indication="MF/SS >= IIB cohort", evidence="NCT07195916",
                 tox="activated T/B + Treg loss; renal tubule CD70"),
    "CCR8": dict(cd="CCR8", agent="DT-7012", modality="mAb", stage="phase 1 (first-in-human)",
                 indication="R/R MF/SS -- CTCL-dedicated, started 2026-01",
                 evidence="NCT07213882",
                 tox="depletes CCR8+ Tregs -> autoimmunity (also the intended mechanism)"),
    "TNFRSF1B": dict(cd="TNFR2", agent="BITR2101; SIM1811-03; BI-1808", modality="mAb",
                     modality_note="three independent phase-1 programmes with CTCL/PTCL cohorts",
                     stage="phase 1/2a", indication="NHL incl. CTCL/PTCL",
                     evidence="NCT06385522", tox="Treg + activated-T depletion -> autoimmunity"),
    "CD38": dict(cd="CD38", agent="daratumumab; isatuximab", modality="mAb", stage="phase 2/3",
                 indication="T-ALL, PTCL -- NOT MF", evidence="NCT07072585",
                 tox="NK/plasma-cell loss, hypogammaglobulinaemia, blood-typing interference"),
    "CD47": dict(cd="CD47", agent="TTI-621 (SIRPaFc, intratumoral); magrolimab + mogamulizumab",
                 modality="fusion / mAb", stage="phase 1b/2 -- both terminated",
                 indication="MF/SS", evidence="NCT02890368",
                 tox="RBC antigen sink -> anaemia; ubiquitous expression"),
    "CD4": dict(cd="CD4", agent="zanolimumab (HuMax-CD4); LB1901 CD4 CAR-T",
                modality="mAb / CAR-T", stage="phase 3 (terminated)", indication="MF/SS",
                evidence="NCT00127881", tox="profound CD4 aplasia -> opportunistic infection"),
    "CD52": dict(cd="CD52", agent="alemtuzumab", modality="mAb", stage="phase 2 (off-label)",
                 indication="SS / erythrodermic MF", evidence="NCT00057967",
                 tox="pan-lymphocyte + monocyte depletion, CMV reactivation"),
    "CD37": dict(cd="CD37", agent="AGS67E (ADC); CAR-37 T", modality="ADC / CAR-T",
                 stage="phase 1", indication="lymphoid malignancies -- no CTCL-specific cohort",
                 evidence="NCT02175433", tox="B/myeloid depletion, neutropenia"),
    "BSG": dict(cd="CD147", agent="CD147 CAR-T", modality="CAR-T", stage="phase 1 (early)",
                indication="T-NHL", evidence="NCT05013372",
                tox="ubiquitous (RBC, endothelium) -- broad off-tumour"),
    "CD84": dict(cd="CD84 / SLAMF5", agent="CART84", modality="CAR-T", stage="phase 1/2",
                 indication="AML and T-ALL -- not yet CTCL", evidence="NCT07471789",
                 tox="platelet / B / monocyte expression"),
    "PDCD1": dict(cd="PD-1", agent="pembrolizumab", modality="checkpoint mAb", stage="phase 2",
                  indication="MF/SS", evidence="NCT02243579",
                  tox="irAEs; PARADOXICAL rapid progression / flare reported in MF-SS and ATLL"),
    "IL31RA": dict(cd="IL-31RA", agent="nemolizumab", modality="mAb",
                   stage="case reports in CTCL", indication="CTCL pruritus -- NOT cytoreductive",
                   evidence="PMID 42662382", tox="symptom control only"),
}

# Checked and found to have NO agent in CTCL / mature T-cell neoplasms as of 2026-10-22. Kept so a
# later search does not have to repeat the work, and so the panel does not imply a precedent.
NO_PRECEDENT_CHECKED = {
    "SLAMF7": "no CTCL/T-NHL trial found (myeloma only)",
    "DPP4": "CD26 is LOST on Sezary cells -- diagnostic, never a target; begelomab/YS110 not in CTCL",
}

POSITIVE_CONTROLS = ["CCR4", "TNFRSF8", "KIR3DL2", "IL2RA"]   # approved or registrational in CTCL


# ====================================================================== trafficking machinery
# The eleven phases between transcription and a displayed epitope. Scored per arm so a candidate's
# transcript rank can be read against the cell's capacity to put it on the surface and keep it
# there. A module score is a cell-state readout, not a per-gene correction factor: treat it as
# context for a candidate, not as a multiplier on its expression.
TRAFFIC_MODULES = {
    # 1. Co-translational translocation and membrane insertion.
    "er_translocation": [
        "SEC61A1", "SEC61B", "SEC61G", "SSR1", "SSR2", "SSR3", "SSR4", "TRAM1", "SEC62", "SEC63",
        "SRPRA", "SRPRB",            # SRPR was renamed SRPRA
        "SRP9", "SRP14", "SRP54", "SRP68", "SRP72",     # added: the particle itself
    ],
    # 2. Folding, chaperones, ER quality control.
    "er_folding_qc": [
        "CANX", "CALR", "HSPA5", "PDIA3", "UGGT1", "UGGT2", "STT3A", "STT3B", "P4HB", "ERO1A",
        "DNAJC10",
        "PDIA4", "PDIA6", "HSP90B1", "DNAJB11", "EDEM1", "EDEM2", "EDEM3", "SEL1L", "HRD1",
        "SYVN1", "OS9", "MLEC", "RPN1", "RPN2", "DDOST",   # added: OST subunits + ERAD
    ],
    # 3. COPII export and cis-Golgi retrieval.
    "er_to_golgi": [
        "LMAN1", "SEC23A", "SEC23B", "SEC24A", "SEC24B", "SEC24C", "SEC24D", "SEC13", "SEC31A",
        "TRIP11", "ERP44", "KDELR1", "KDELR2", "KDELR3",
        "SAR1A", "SAR1B", "PREB", "LMAN2", "ERGIC1", "ERGIC2", "ERGIC3", "SURF4",
        "TMED2", "TMED9", "TMED10", "GOLPH3", "ARFGEF1", "ARFGEF2",   # added: cargo receptors
    ],
    # 4. N-glycan branching and sialylation -- the galectin lattice that physically retains
    #    receptors at the surface and changes epitope accessibility without changing transcript.
    "glycosylation": [
        "MGAT1", "MGAT2", "MGAT3", "MGAT4A", "MGAT4B", "MGAT5",
        "ST3GAL1", "ST3GAL2", "ST6GAL1", "ST6GALNAC1", "B3GNT3", "GCNT1",
        *[f"GALNT{i}" for i in range(1, 15)],
        "B4GALT1", "ST3GAL4", "ST3GAL6", "FUT7", "FUT8", "CHST2",     # added: CLA / sLeX arm
        "NEU1", "NEU3", "GLG1", "SPPL3",                              # added: sialidases + SPPL3
    ],
    # 5. Clathrin-mediated endocytosis. AP2M1 reads the YVKM motif of CTLA-4: this module is why
    #    a high CTLA4 transcript does not mean a CTLA-4 epitope.
    "endocytosis": [
        "CLTC", "CLTA", "AP2A1", "AP2A2", "AP2B1", "AP2M1", "AP2S1", "DNM1", "DNM2",
        "RAB5A", "RAB5B", "RAB5C",
        "EPS15", "EPN1", "SNX9", "SH3GL1", "AAK1", "BIN1",            # added
    ],
    # 6. Endosomal recycling -- what rescues an internalised receptor and puts it back.
    "recycling": [
        "LRBA", "DEF6", "RAB11A", "RAB11B", "RAB4A", "RAB4B", "RAB14", "ARF1", "ARF6",
        "NBEAL2", "RAB8A", "RAB10",
        "RAB35", "EHD1", "EHD3", "RAB25", "VPS51", "VPS52", "VPS53", "VPS54",   # added: GARP
    ],
    # 7. Retromer / ESCPE-1 retrieval away from the lysosome.
    "retrograde": [
        "VPS35", "VPS26A", "VPS26B", "VPS29", "SNX1", "SNX2", "SNX5", "SNX6", "SNX3", "SNX27",
        "SNX17", "SNX31", "SORT1", "RAB7A", "LAMP1",
        *[f"COG{i}" for i in range(1, 9)],                             # added
    ],
    # 8. Deubiquitination vs E3 ligases: how long a receptor survives once internalised.
    "ubiquitin_proteostasis": [
        "USP8", "USP48", "PTPN23", "FBXO38", "CBL", "CBLB", "RNF128", "TRAF6", "MARCHF1",
        "ITCH", "NEDD4",
        "WWP1", "WWP2", "SMURF1", "SMURF2", "MARCHF5", "MARCHF8", "DTX1", "PELI1",
        "OTUB1", "OTUD7B", "USP4", "USP9X", "USP12", "USP33", "RNF13", "RNF125",    # added
    ],
    # 9. SNARE-mediated docking and fusion: the final display step.
    "snare_fusion": [
        "VAMP2", "VAMP3", "VAMP7", "STX4", "STX6", "STX7", "STX11", "SNAP23", "VTI1B",
        "RAB27A", "RAB27B", "PLD1", "PLD2",
        "STXBP1", "STXBP2", "STXBP3", "NSF", "NAPA",                   # added
    ],
    # 10. Ectodomain shedding -- the fastest way to lose an epitope. ADAM17 cleaves SELL/CD62L
    #     within minutes of activation; report it next to any SELL result.
    "shedding": [
        "ADAM17", "ADAM10", "MMP2", "MMP9", "BACE1",
        "ADAM8", "ADAM9", "ADAM15", "ADAM19", "MMP14", "FURIN", "PCSK5", "PCSK7",   # added
        "TIMP1", "TIMP3",                                              # added: the brakes
    ],
    # 11. ESCRT / multivesicular bodies: exosomal release of surface microdomains.
    "escrt_exosome": [
        "TSG101", "UBAP1", "CHMP4A", "CHMP4B", "CHMP4C", "VPS4A", "VPS4B", "PDCD6IP", "SMPD3",
        "HGS", "STAM", "STAM2", "TOM1",                                # added: ESCRT-0
        "CHMP2A", "CHMP2B", "CHMP3", "CHMP5", "CHMP6", "IST1", "VTA1",  # added: ESCRT-III
    ],
}

TRAFFIC_GENES = list(dict.fromkeys(g for genes in TRAFFIC_MODULES.values() for g in genes))
GENE_TO_TRAFFIC = {}
for _mod, _genes in TRAFFIC_MODULES.items():
    for _g in _genes:
        GENE_TO_TRAFFIC.setdefault(_g, []).append(_mod)

# Candidate -> the machinery that specifically governs ITS surface density. Printed beside the
# candidate in the ranking tables, so a reader sees why a transcript rank may mislead.
SURFACE_CAVEAT = {
    "CTLA4": "constitutively internalised by AP2M1 via its YVKM motif; surface pool is a few % "
             "of total. LRBA/DEF6 recycling sets what is displayed",
    "PDCD1": "rapid internalisation and FBXO38-mediated degradation; transcript overstates surface",
    "SELL": "shed by ADAM17 within minutes of activation -- read SELL beside ADAM17",
    "CD4": "down-modulated from the surface by Nef-independent AP-2 routes on activation",
    "TNFRSF8": "the soluble CD30 ectodomain is shed by ADAM17; serum sCD30 is a CTCL biomarker",
    "KIR3DL2": "stable surface receptor, internalises on ligand binding -- the basis of lacutamab ADCC",
    "CCR4": "ligand-induced internalisation (CCL17/CCL22); a high-CCL17 lesion lowers surface CCR4",
    "IL2RA": "shed as sIL-2R; serum sIL-2R tracks CTCL burden, so surface and transcript diverge",
}


# ====================================================================== CITE-seq bridge (blood)
# blood_T_annotated.h5ad obsm["X_adt"], 128 markers, uns["adt_panel"]. NaN for cells from studies
# without CITE-seq -- always mask on np.isfinite before using it. Only entries with an unambiguous
# gene are mapped; isotype/fluorophore controls (ADT_PE, ADT_FITC, ADT_APC, ADT_Biotin) are not.
ADT_TO_GENE = {
    "ADT_CD1a": "CD1A", "ADT_CD1c": "CD1C", "ADT_CD1d": "CD1D", "ADT_CD2": "CD2",
    "ADT_CD3": "CD3E", "ADT_CD4": "CD4", "ADT_CD5": "CD5", "ADT_CD7": "CD7",
    "ADT_CD8A": "CD8A", "ADT_CD8a": "CD8A", "ADT_CD11a": "ITGAL", "ADT_CD11b": "ITGAM",
    "ADT_CD11c": "ITGAX", "ADT_CD14": "CD14", "ADT_CD16": "FCGR3A", "ADT_CD18": "ITGB2",
    "ADT_CD19": "CD19", "ADT_CD20": "MS4A1", "ADT_CD21": "CR2", "ADT_CD22": "CD22",
    "ADT_CD23": "FCER2", "ADT_CD25": "IL2RA", "ADT_CD26": "DPP4", "ADT_CD27": "CD27",
    "ADT_CD28": "CD28", "ADT_CD29": "ITGB1", "ADT_CD30": "TNFRSF8", "ADT_CD31": "PECAM1",
    "ADT_CD35": "CR1", "ADT_CD38": "CD38", "ADT_CD39": "ENTPD1", "ADT_CD43": "SPN",
    "ADT_CD44": "CD44", "ADT_CD45RA": "PTPRC", "ADT_CD45RO": "PTPRC", "ADT_CD48": "CD48",
    "ADT_CD49a": "ITGA1", "ADT_CD49d": "ITGA4", "ADT_CD52": "CD52", "ADT_CD55": "CD55",
    "ADT_CD56": "NCAM1", "ADT_CD56_QA17A16": "NCAM1", "ADT_CD57": "B3GAT1", "ADT_CD62L": "SELL",
    "ADT_CD69": "CD69", "ADT_CD70": "CD70", "ADT_CD71": "TFRC", "ADT_CD73": "NT5E",
    "ADT_CD80": "CD80", "ADT_CD84": "CD84", "ADT_CD86": "CD86", "ADT_CD95": "FAS",
    "ADT_CD99": "CD99", "ADT_CD101": "CD101", "ADT_CD103": "ITGAE", "ADT_CD107a": "LAMP1",
    "ADT_CD117": "KIT", "ADT_CD120b": "TNFRSF1B", "ADT_CD122": "IL2RB", "ADT_CD124": "IL4R",
    "ADT_CD126": "IL6R", "ADT_CD127": "IL7R", "ADT_CD134": "TNFRSF4", "ADT_CD137": "TNFRSF9",
    "ADT_CD137L": "TNFSF9", "ADT_CD138": "SDC1", "ADT_CD141": "THBD", "ADT_CD150": "SLAMF1",
    "ADT_CD152": "CTLA4", "ADT_CD154": "CD40LG", "ADT_CD158b": "KIR2DL3",
    "ADT_CD158e1": "KIR3DL1", "ADT_CD158f": "KIR2DL5A", "ADT_CD158k": "KIR3DL2",
    "ADT_CD161": "KLRB1", "ADT_CD163": "CD163", "ADT_CD164": "CD164", "ADT_CD183": "CXCR3",
    "ADT_CD184": "CXCR4", "ADT_CD185": "CXCR5", "ADT_CD193": "CCR3", "ADT_CD194": "CCR4",
    "ADT_CD195": "CCR5", "ADT_CD196": "CCR6", "ADT_CD197": "CCR7", "ADT_CD207": "CD207",
    "ADT_CD223": "LAG3", "ADT_CD224": "GGT1", "ADT_CD244": "CD244", "ADT_CD254": "TNFSF11",
    "ADT_CD272": "BTLA", "ADT_CD273": "PDCD1LG2", "ADT_CD274": "CD274", "ADT_CD275": "ICOSLG",
    "ADT_CD278": "ICOS", "ADT_CD279": "PDCD1", "ADT_CD294": "PTGDR2", "ADT_CD307c": "FCRL3",
    "ADT_CD314": "KLRK1", "ADT_CD319": "SLAMF7", "ADT_CD326": "EPCAM", "ADT_CD357": "TNFRSF18",
    "ADT_CD360": "IL21R", "ADT_CD366": "HAVCR2", "ADT_CCR10": "CCR10", "ADT_CX3CR1": "CX3CR1",
    "ADT_FceRIa": "FCER1A", "ADT_GARP": "LRRC32", "ADT_GP130": "IL6ST", "ADT_GPR56": "ADGRG1",
    "ADT_HLA-DR": "HLA-DRA", "ADT_ITGB7": "ITGB7", "ADT_KLRG1": "KLRG1", "ADT_LAP": "TGFB1",
    "ADT_TIGIT": "TIGIT", "ADT_TSLPR": "CRLF2", "ADT_CD213a1": "IL13RA1",
    "ADT_CD213a2": "IL13RA2", "ADT_CD217": "IL17RA", "ADT_CLA": "SELPLG",
    # ADT_CD158 is a pan-KIR clone and ADT_TCRab / ADT_TCRgd read a multi-gene complex: both are
    # deliberately left unmapped rather than assigned to an arbitrary member.
}
GENE_TO_ADT = {}
for _adt, _gene in ADT_TO_GENE.items():
    GENE_TO_ADT.setdefault(_gene, []).append(_adt)

ADT_CONTROLS = ["ADT_PE", "ADT_FITC", "ADT_APC", "ADT_Biotin"]   # background / isotype


# ====================================================================== accessors
def panel_genes(classes=None, include_traffic=False):
    """The a-priori gene list, optionally restricted to `classes` or extended with the machinery."""
    if classes is None:
        genes = list(PANEL_GENES)
    else:
        missing = set(classes) - set(SURFACE_PANEL)
        if missing:
            raise KeyError(f"unknown panel classes: {sorted(missing)}")
        genes = list(dict.fromkeys(g for c in classes for g in SURFACE_PANEL[c]))
    if include_traffic:
        genes = list(dict.fromkeys([*genes, *TRAFFIC_GENES]))
    return genes


def load_surfaceome(confident_only=True, strict=True):
    """The built surfaceome reference (`jobs/build_surfaceome_ref.py`), or None if not built.

    `surface_confident` is the screening gate: two independent sources, or one source that is
    evidence rather than prediction (CSPA mass spec on intact cells, or a SURFY curated/GPI
    entry). 2,325 genes as built on 2026-10-22.

    **Never use this as the only inclusion criterion.** A naive `n_evidence >= 2` filter drops
    KIR3DL2 -- a phase-2 CTCL antibody target -- because no CSPA, HPA or GO cell-surface record
    exists for it, and SURFY alone carries it. It also drops TRAC/TRBC1/TRBC2, which SURFY labels
    `nonsurface` as standalone chains although the TCR complex they form is the most displayed
    structure on a T cell. The screen takes the **union** of `SURFACE_PANEL` and this reference,
    and uses the reference to annotate and to find candidates the panel never listed.
    """
    if not SURFACEOME_PQ.exists():
        msg = (f"{SURFACEOME_PQ} not found -- run jobs/download_surfaceome.sh, then "
               "jobs/build_surfaceome_ref.py")
        if strict:
            raise FileNotFoundError(msg)
        import warnings
        warnings.warn(msg)
        return None
    df = pd.read_parquet(SURFACEOME_PQ)
    return df[df["surface_confident"]].copy() if confident_only else df


def screen_genes(var_names, surfaceome=None):
    """Genes to screen: the a-priori panel UNION the confident surfaceome, restricted to `var_names`.

    Returns (genes, report). The report says how each gene got in -- `panel`, `surfaceome` or
    `both` -- and which panel genes the object does not carry, which is the number to quote when
    a known target does not appear in a result.
    """
    present = set(map(str, var_names))
    if surfaceome is None:
        surfaceome = load_surfaceome(strict=False)
    surf = set(surfaceome["gene"]) if surfaceome is not None else set()
    panel = set(PANEL_GENES)

    genes = sorted((panel | surf) & present)
    src = {g: ("both" if g in panel and g in surf else "panel" if g in panel else "surfaceome")
           for g in genes}
    report = {
        "n_screened": len(genes),
        "n_panel": len(panel), "n_panel_present": len(panel & present),
        "panel_absent_from_object": sorted(panel - present),
        "n_surfaceome": len(surf), "n_surfaceome_present": len(surf & present),
        "by_source": pd.Series(src).value_counts().to_dict(),
    }
    return genes, report


def annotate(genes):
    """One row per gene: panel class, trafficking module, precedent, surface caveat, loss marker."""
    genes = list(dict.fromkeys(map(str, genes)))
    rows = []
    for g in genes:
        pre = CLINICAL_PRECEDENT.get(g)
        rows.append({
            "gene": g,
            "panel_class": "|".join(GENE_TO_CLASS.get(g, [])),
            "traffic_module": "|".join(GENE_TO_TRAFFIC.get(g, [])),
            "cd_name": (pre or {}).get("cd", ""),
            "agent": (pre or {}).get("agent", ""),
            "stage": (pre or {}).get("stage", ""),
            "indication": (pre or {}).get("indication", ""),
            "evidence": (pre or {}).get("evidence", ""),
            "off_tumor_concern": (pre or {}).get("tox", ""),
            "no_precedent_note": NO_PRECEDENT_CHECKED.get(g, ""),
            "positive_control": g in POSITIVE_CONTROLS,
            "loss_marker": g in LOSS_MARKERS,
            "surface_caveat": SURFACE_CAVEAT.get(g, ""),
            "adt": "|".join(GENE_TO_ADT.get(g, [])),
        })
    return pd.DataFrame(rows)
