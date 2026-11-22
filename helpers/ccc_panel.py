"""
CCC panel for CTCL MF atlas — CD4 <-> fibroblast / myeloid.
Derived from the NB28 degradome analysis (see nb28_degradome_results_analysis.md).

[NB28] = measured in the degradome run; direction noted is malignant/clonal.
Trimmed from the full list: tier-3 coverage genes dropped, kept what a result or a
named hypothesis depends on.

RUN NOTES
  - Primary sender = TCR-clonal CD4 (33 donors). mal_knn imports stromal ambient into the
    malignant arm (COL6A3, PLAT, HTRA1, SDC4, COL27A1) -> fake autocrine edges. Secondary only.
  - Donor-level pseudobulk, study as blocking factor, >=6/15 interaction-resource support.
  - Score TIMP1 and TIMP3 as SEPARATE readouts, never a composite inhibitor score.
  - Report HC-vs-externalHC edge count as the technical null (same calibration as the DE run).
"""

# =============================================================================
# CD4 — SENDER
# =============================================================================

CD4_SENDER = {
 # measured, clonal-enriched
 "measured":  ["BSG", "TIMP1", "LTBP1", "LAMA3", "CSTB", "CTSH", "SERPINB8", "SPP1", "THBS1"],
 # Z_2 polyfunctional program; IL13/IL4 -> POSTN, TNC, HAS2, and myeloid MMP12
 "th2":       ["IL13", "IL4", "IL5", "IL31", "IL9", "IL22", "IL21", "IL17A", "IL17F", "IL10",
               "CCL1", "EBI3", "CXCL13"],
 # the only axis that explains TIMP1 up WITH TIMP3 down
 "gp130":     ["OSM", "IL6", "IL11", "LIF"],
 # ITGB8 is the informative gene, not TGFB1 (ubiquitous in T cells, uninformative)
 "tgfb":      ["ITGB8", "ITGB6", "ITGAV", "TGFB1", "LTBP1", "THBS1"],
 # CD70-CD27 is the project's strongest existing LIANA hit
 "tnfsf":     ["TNFSF14", "CD40LG", "TNFSF11", "CD70", "TNFSF12", "TNFSF13B", "LTA", "LTB", "TNF"],
 "il1":       ["IL1A", "IL1B", "IL18"],
 "gf":        ["PDGFA", "PDGFB", "VEGFA", "HBEGF", "AREG", "EREG", "CSF1", "CSF2", "IL34"],
 # myeloid cathepsin block; PPIA/PPIB are the BSG ligands
 "innate":    ["S100A8", "S100A9", "S100A4", "MIF", "LGALS1", "LGALS3", "LGALS9", "HMGB1",
               "PPIA", "PPIB"],
 "chemo":     ["CCL3", "CCL4", "CCL5", "CCL20", "XCL1", "CXCL13"],
 # ADAM17 sheds IL6R -> trans-signalling on IL6R-low fibroblasts (route to gp130 at low IL6)
 "sheddase":  ["ADAM10", "ADAM17", "FURIN", "PLAU", "PLAUR"],
}

# =============================================================================
# CD4 — RECEIVER
# =============================================================================

CD4_RECEIVER = {
 # [NB28] CD44 / ITGB1 trend LOWER, ITGA6 6x LOWER (q=2.5e-06), DPP4 7.8x LOWER (CD26 loss).
 # Do NOT fix the receptor side to beta1 or CD44 in advance.
 "measured":  ["CD44", "ITGB1", "ITGA6", "BSG", "DPP4", "SDC4", "ITGAV", "ITGAL", "ITGB2"],
 "tnc_r":     ["ITGA9", "ITGAV", "ITGB3", "ITGB6", "ANXA2", "TLR4", "EGFR", "SDC4"],
 "postn_r":   ["ITGB5", "ITGB4", "ITGAM"],
 # TIMP1 -> CD63/beta1 survival signalling; also autocrine on the clone
 "timp1_r":   ["CD63", "LRP1"],
 "matrix":    ["ITGA1", "ITGA2", "ITGA4", "ITGA5", "ITGAE", "ITGB7", "CD47", "DDR1", "HMMR", "CD36"],
 # IL1RL1 (ST2) closes the fibroblast IL33 -> malignant Th2 loop (Li 2024 F2 / your apCAF)
 "cytokine":  ["IL4R", "IL13RA1", "IL13RA2", "IL2RG", "IL1RL1", "IL1R1", "IL6R", "IL6ST",
               "OSMR", "IL11RA", "IL7R", "CRLF2", "IL15RA", "IL21R", "IL17RB", "IL10RA",
               "TGFBR1", "TGFBR2", "TGFBR3"],
 "tnfrsf":    ["TNFRSF14", "TNFRSF1A", "TNFRSF1B", "TNFRSF4", "TNFRSF9", "CD27", "CD40", "LTBR",
               "TNFRSF11A"],
 "checkpoint":["PDCD1", "CTLA4", "TIGIT", "LAG3", "HAVCR2", "CD28", "ICOS",
               "KIR3DL2", "KIR2DL3", "KIR3DL1"],
 "traffic":   ["CCR4", "CCR7", "CCR10", "CXCR4", "CXCR5", "S1PR1", "SELPLG", "PIEZO1"],
 # PAR1/PAR2: the only transcriptional read on protease ACTIVITY rather than abundance
 "protease_r":["F2R", "F2RL1", "PLAUR"],
}

# =============================================================================
# FIBROBLAST
# =============================================================================

FIB_RECEIVER = {
 "cytokine":  ["IL4R", "IL13RA1", "IL13RA2", "IL6ST", "IL6R", "OSMR", "IL11RA", "LIFR",
               "IL1R1", "IL1RAP", "IL17RA", "IL17RC", "IFNGR1"],
 "tgfb":      ["TGFBR1", "TGFBR2", "TGFBR3", "ACVRL1", "ENG"],
 "tnfrsf":    ["TNFRSF1A", "TNFRSF1B", "LTBR", "TNFRSF14", "CD40", "TNFRSF11A", "TNFRSF12A"],
 "gf":        ["PDGFRA", "PDGFRB", "EGFR", "FGFR1", "MET", "NOTCH1", "NOTCH3", "JAG1", "NRP1"],
 "matrix":    ["BSG", "CD44", "SDC1", "SDC4", "ITGB1", "ITGAV", "ITGA11", "ITGA5", "ITGB3",
               "LRP1", "CD63", "DDR2", "PIEZO1"],
 "innate":    ["TLR2", "TLR4", "CD14", "AGER", "CD74", "CXCR4"],
 # apCAF expanded 4.1x; CD74/HLA-DR <-> CD4 is what defines the population
 "apcaf":     ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "CD74", "CIITA", "CD80", "CD86", "LGALS9"],
}

FIB_SENDER = {
 # tier A: MF > both healthy arms, healthy arms agree. These ARE ligands, not just readouts.
 "tierA":     ["TNC", "POSTN", "PRSS23", "CTSB", "CTSA", "CTSH", "CTSZ", "FAP", "HMCN1",
               "NID2", "EFEMP1", "FBN1", "EMILIN1", "LTBP2", "FN1", "SERPINE1", "A2M", "TIMP1"],
 # tier B: internally supported, external arm diverges
 "tierB":     ["COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL6A1", "COL6A2", "COL6A3", "COL12A1",
               "COL15A1", "COL18A1", "COL4A1", "COL4A2", "MMP2", "MMP14", "MMP1", "MMP3",
               "MMP19", "ADAMTS1", "ADAMTS2", "ADAMTS4", "ADAMTS5", "ADAM10", "ADAM12",
               "HAS2", "LOX", "LOXL1", "LOXL2"],
 # tier C: the reciprocal LOSS. A lost ligand is an interaction change; DCN/LUM sequester
 # TGF-beta, DCN inhibits EGFR/MET, OGN/ASPN modulate BMP. Score direction explicitly.
 "lost":      ["ELN", "MFAP4", "MFAP5", "FBLN1", "FBLN5", "OGN", "ASPN", "PRELP", "DPT",
               "COL14A1", "COL21A1", "THBS4", "HTRA1", "SLPI", "TIMP3", "DCN", "LUM"],
 "cytokine":  ["IL33", "IL15", "IL6", "IL7", "IL11", "LIF", "CSF1", "TNFSF13B", "CXCL12",
               "CXCL9", "CXCL10", "CXCL13", "CXCL14", "CCL2", "CCL5", "CCL19", "CCL21",
               "VEGFA", "HGF", "TGFB1", "TGFB3", "GAS6"],
 "adhesion":  ["VCAM1", "ICAM1", "PDPN", "THY1", "PVR", "LGALS1", "LGALS3", "C3", "VCAN",
               "HSPG2", "NID1", "BGN", "SPARC", "THBS1", "THBS2", "LAMA4", "LAMB1", "LAMC1"],
}

# =============================================================================
# MYELOID
# =============================================================================

MYE_RECEIVER = ["IL4R", "IL13RA1", "CSF1R", "CSF2RA", "IL10RA", "IL6ST", "OSMR", "TLR2", "TLR4",
                "CD14", "AGER", "CD74", "CXCR4", "SIRPA", "TREM2", "TYROBP", "MRC1", "CD163",
                "MERTK", "AXL", "LRP1", "CD36", "MSR1", "FCGR1A", "FCGR3A", "ITGAM", "ITGAX",
                "ITGB2", "CD40", "LTBR", "TNFRSF14", "TNFRSF1A", "NRP1", "LILRB2", "CLEC10A"]

# MMP12 is the priority output: largest MF-up myeloid effect (cDC2, delta 0.86, q=4e-05),
# principal elastolytic MMP, enzymatic complement to the fibroblast elastic-fibre loss.
MYE_SENDER   = ["MMP12", "MMP9", "MMP7", "MMP19", "CTSK", "CTSS", "CTSL", "CTSB", "CTSD", "LGMN",
                "SPP1", "S100A8", "S100A9", "MIF", "IL1B", "TNF", "IL6", "IL10", "OSM", "IL15",
                "IL18", "IL23A", "EBI3", "TGFB1", "VEGFA", "APOE", "GPNMB", "C1QA", "CCL13",
                "CCL17", "CCL18", "CCL22", "CXCL9", "CXCL10", "LGALS9", "PVR", "CD80", "CD86",
                "CD274", "IDO1", "SERPINA1", "CST7"]

# =============================================================================
# PRIORITISED DIRECTED PAIRS — score these before any unbiased sweep
# (sender_cell, ligand, receiver_cell, receptor, predicted readout / basis)
# =============================================================================

PRIORITY_PAIRS = [
 ("malignant_CD4", "IL13|IL4",        "fibroblast", "IL4R|IL13RA1",
  "POSTN+, TNC+, HAS2+, collagen+ | POSTN is a canonical STAT6 target"),
 ("malignant_CD4", "OSM|IL6|IL11",    "fibroblast", "OSMR|IL6ST|IL11RA",
  "TIMP1+ WITH TIMP3- | only axis explaining both signs"),
 ("malignant_CD4", "ADAM17->sIL6R",   "fibroblast", "IL6ST",
  "same as #2 at low IL6 transcript | fibroblasts are IL6R-low"),
 ("malignant_CD4", "LTBP1|ITGB8.ITGAV","fibroblast","TGFBR1|TGFBR2",
  "FAP+, PRSS23+, SERPINE1+, LOXL+, ADAMTS2+ | LTBP1 [NB28] clonal-up; ITGAV flat"),
 ("malignant_CD4", "BSG|PPIA|PPIB",   "fibroblast", "BSG (homophilic)",
  "MMP2+, MMP14+, MMP1+ | Emmprin model; BSG is the top clonal gene"),
 ("fibroblast",    "TIMP1",           "malignant_CD4","CD63|ITGB1",
  "malignant survival / anti-apoptosis | TIMP1 up in BOTH compartments"),
 ("fibroblast",    "TNC",             "malignant_CD4","ITGA9.ITGB1|ITGAV.ITGB3|ANXA2|TLR4|SDC4",
  "low-adhesion migration + NF-kB via TLR4 | strongest stromal finding, 41-51x floor"),
 ("fibroblast",    "POSTN",           "malignant_CD4","ITGAV.ITGB3|ITGAV.ITGB5",
  "survival, adhesion | tier A"),
 ("fibroblast_apCAF","IL33",          "malignant_CD4","IL1RL1",
  "Th2 amplification loop closure | Li 2024 F2 IL33; apCAF 4.1x expanded"),
 ("fibroblast_apCAF","HLA-DR|CD74|LGALS9","CD4",    "TCR|HAVCR2",
  "antigen presentation without CD80/CD86 | apCAF definition"),
 ("fibroblast",    "CXCL12",          "malignant_CD4","CXCR4",
  "retention; gradient altered by CD26 loss | DPP4 7.8x lower [NB28]"),
 ("malignant_CD4", "TNFSF14",         "fibroblast", "LTBR|TNFRSF14",
  "juxtacrine ECM + inflammatory induction | LIGHT drives fibrosis"),
 ("malignant_CD4", "IL13|S100A8.A9|MIF","myeloid",  "IL4R|TLR4.AGER|CD74.CXCR4",
  "MMP12+, CTSB/S/K+, CD163+ | elastolysis axis; Du 2022"),
 ("fibroblast|myeloid","A2M|SERPINE1|MMP2|TIMP1","malignant_CD4","LRP1",
  "clearance/signalling of the inhibitor surplus | proteases and inhibitors rise together"),
 ("malignant_CD4", "LAMA3",           "fibroblast|keratinocyte","ITGA6.ITGB4|ITGA3.ITGB1",
  "basement-membrane / epidermotropism | LAMA3 [NB28] clonal-up, unexplained"),
 # ---- nb28 panel extension (docs/NB28_panel_extension_spec.md), appended so indices 0-14 and
 # their PAIR_AXIS keys cannot move. Same edge construction, same paired test, same nulls.
 ("malignant_CD4", "CD40LG",  "fibroblast", "CD40",
  "juxtacrine | CD40 already up in NB28 fibroblasts"),
 ("malignant_CD4", "LTA",     "fibroblast", "LTBR",
  "juxtacrine | LTBR already up in NB28 fibroblasts"),
 ("malignant_CD4", "LTB",     "fibroblast", "LTBR",
  "juxtacrine | LTB is on the ambient list -- trust section A only"),
 ("malignant_CD4", "OSM",     "fibroblast", "OSMR",
  "gp130_stat3 | the TIMP1+/TIMP3- axis, ligand side measured directly"),
 ("malignant_CD4", "TGFB1",   "fibroblast", "TGFBR2",
  "tgfb_activation | the ligand itself, next to the ITGB8/LTBP1 activation route"),
 ("malignant_CD4", "PDGFB",   "fibroblast", "PDGFRB",
  "caf_expansion | PDGFRB already up in NB28 fibroblasts"),
 ("malignant_CD4", "ENPP2",   "fibroblast", "LPAR1",
  "lipid_purinergic | autotaxin-LPA, a fibroblast-activating route not yet in the panel"),
 ("malignant_CD4", "NT5E",    "fibroblast", "ADORA2B",
  "lipid_purinergic | CD73 adenosine on the A2B receptor"),
 ("malignant_CD4", "AREG",    "fibroblast", "EGFR",
  "egfr_ligands | amphiregulin is the classic T-cell EGFR ligand"),
 ("malignant_CD4", "HBEGF",   "fibroblast", "EGFR",
  "egfr_ligands | second EGFR ligand, read against AREG"),
 ("malignant_CD4", "IL11",    "fibroblast", "IL11RA",
  "gp130_stat3 | IL11RA already up in NB28 fibroblasts; IL11 is low-copy, expect abstention"),
 ("malignant_CD4", "SPP1",    "fibroblast|myeloid", "CD44",
  "ecm_signalling | osteopontin; the myeloid row says whether the clone is the dominant source"),
 ("malignant_CD4", "IL17A",   "fibroblast", "IL17RA",
  "il17_il22 | low-copy secreted cytokine, expect abstention"),
 ("malignant_CD4", "JAG1",    "fibroblast", "NOTCH3",
  "wnt_notch | NOTCH3 is the myofibroblast receptor"),
 ("malignant_CD4", "WNT5A",   "fibroblast", "ROR2",
  "wnt_notch | non-canonical Wnt on the ROR2 receptor"),
]
# Pairs 2 and 4 are a designed contrast: gp130 predicts TIMP1+ / TIMP3-, SMAD predicts TIMP3+.
# Running both against the same readout is the most informative single comparison here.

# =============================================================================
# PROTEASES AS EDGE MODIFIERS — not edges. Post-hoc layer on surviving pairs.
# =============================================================================

EDGE_MODIFIERS = {
 "DPP4":            ["CXCL12", "CCL22"],            # N-terminal truncation, changes the gradient
 "ADAM17":          ["IL6R", "TNFRSF1A", "TNF"],    # shedding -> trans-signalling
 "ADAMTS4|ADAMTS5": ["VCAN"],                       # versikine release
 "MMP2|MMP14":      ["TGFB1_latent", "CXCL12"],
 "FAP":             ["CXCL12", "FGF21"],
 "CTSS|CTSL":       ["CD74"],                       # MHC-II loading in apCAF
 "PLAU|PLAUR":      ["plasminogen->plasmin"],
}

# =============================================================================
# NEGATIVE CONTROLS — each supports a stated "this is NOT happening" claim
# =============================================================================

NEGATIVES = [
 # LMWHA axis: enzyme side and receptor side both failed
 "CD44", "HMMR", "HYAL1", "HYAL2", "HYAL3", "CEMIP", "TMEM2", "HAS1", "HAS2", "HAS3",
 # hydrodynamic migration: no malignant-specific program
 "WNK1", "OXSR1", "STK39", "SLC12A2",
 # PI-9: no transcript evidence of escape from granzyme B
 "SERPINB9",
 # receptor-side baseline: beta1 and alpha6 are LOWER in the clone
 "ITGB1", "ITGA6", "ITGAL", "ITGB2", "BSG",
 # cytotoxic-clonal discriminators for the donor split
 "GZMK", "GZMB", "GZMA", "PRF1", "GZMH", "CST7", "DPP4",
]

# =============================================================================
# EXCLUSIONS — pre-specify; these will score and mean nothing
# =============================================================================

EXCLUDE_AS_ARTEFACT = {
 # ambient leak: lineage-impossible genes assigned to the wrong compartment
 "in_fibroblast_or_myeloid": ["GZMA", "GZMB", "GZMH", "GZMK", "GZMM", "PRF1", "CTSW",
                              "ITGAL", "ITGB2", "WAS", "WIPF1", "PTK2B", "FERMT3"],
 "in_cd4_or_myeloid":        ["COL1A1", "COL1A2", "COL3A1", "COL6A1", "COL6A2", "COL6A3",
                              "DCN", "LUM", "SPARC", "POSTN", "FN1"],
 # mal_knn-specific: collapse under the sequence-defined split -> ambient, not malignant
 "malignant_under_mal_knn":  ["COL6A3", "COL27A1", "SDC4", "PLAT", "HTRA1", "ADAM9"],
 # apCAF until LYZ/AIF1/C1QA co-expression + doublet score are checked
 "in_apCAF_pending":         ["ADAMDEC1"],
 # depth-driven: <5% detection, direction library size favours. LAMA3/LTBP1 are in
 # PRIORITY_PAIRS because they are mechanistically interesting, NOT because they are
 # established — both need the depth-adjusted paired re-test first.
 "depth_suspect_clonal_up":  ["HYAL2", "HYAL3", "CEMIP", "LOXL3", "LAMA3", "LTBP1", "SERPINB8"],
}

# Abundance dominance: malignant CD4 is ~66% of CD4 and CD4 dominates MF skin, so mean-based
# scores rank malignant-sender edges highly regardless of specificity. Keep the LIANA+
# correction already in place and report edge specificity alongside magnitude.

# =============================================================================
# FLAT PANEL
# =============================================================================

def _flat(d):
    out = []
    for v in d.values():
        out += v if isinstance(v, list) else []
    return out

CCC_PANEL = {
 "cd4_sender":   sorted(set(_flat(CD4_SENDER))),
 "cd4_receiver": sorted(set(_flat(CD4_RECEIVER))),
 "fib_receiver": sorted(set(_flat(FIB_RECEIVER))),
 "fib_sender":   sorted(set(_flat(FIB_SENDER))),
 "mye_receiver": sorted(set(MYE_RECEIVER)),
 "mye_sender":   sorted(set(MYE_SENDER)),
 "negatives":    sorted(set(NEGATIVES)),
}

ALL_GENES = sorted(set(g for v in CCC_PANEL.values() for g in v))

if __name__ == "__main__":
    for k, v in CCC_PANEL.items():
        print(f"{k:14s} {len(v):4d}")
    print(f"{'UNIQUE TOTAL':14s} {len(ALL_GENES):4d}")
    print(f"{'priority pairs':14s} {len(PRIORITY_PAIRS):4d}")
