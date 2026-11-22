"""Config for the skin degradome analysis (notebook 28, atlas v2).

The question. `docs/cutaneous-matrix-summary.md` describes the cutaneous matrix as a set of
compartments -- fibrillar and basement-membrane collagens, elastic fibre, proteoglycan,
matricellular, the enzyme classes that dismantle each of them, hyaluronan turnover, the elastin
receptor complex, and the adhesion/motility machinery a lymphocyte uses to move through the
result. This module configures one exploratory pass over all of them: which skin cell type
carries each compartment, and how it moves healthy -> early -> advanced.

Scope. **Skin, MF + HC only** -- 1,002,481 cells, 112 donors, 123 samples, 12 studies.
Malignancy is annotated for MF and not for SS or CTCL_other, so including them would put cells
of unknown malignant status in a comparison whose axis is malignant status. Stage arms by donor:
HC 22, early 21, advanced 38, unknown 31 (unknown is plotted, never a contrast arm).

Design decisions, with the measurements that forced them.

  * Expression comes from `joint_annotated.h5ad` (42,347 symbols, raw counts), NOT
    `joint_mrvi_input_skin.h5ad`. The 10k-HVG object drops ADAM10 and ADAM17 -- both ubiquitous,
    neither ever highly variable -- and would drop much of the collagen and motility panel for
    the same reason. The object is rebuilt by one streaming pass.

  * **The size factor is summed over all 42,347 genes and only then are the columns subset to
    the panel.** Normalising after subsetting would rescale every cell by
    (panel counts / total counts), which varies by cell type -- precisely the axis this analysis
    measures. `degradome_helpers.assert_build_equivalence` gates it against a full-gene test
    set and runs before anything else in the notebook. This is what makes single-cell comparison
    across samples of very different library size legitimate.

  * The analysis unit is the **cell**, not the pseudobulk, but cells within a donor are not
    independent. So every test is stratified by study and every effect is reported next to the
    median per-donor difference, with donor medians overlaid on each violin. `MIN_CELLS` /
    `MIN_SAMPLES` still gate which (level, donor) pairs are eligible to contribute a median.

  * No study carries all three stage arms. Measured, MF+HC, donors per study: li2024 has
    early+advanced (17/19) and no HC; chennareddy2025 (4 HC / 4 advanced) and gaydosik2022
    (4/3) have HC+advanced and no early; gaydosik2019, gaydosik2023, il4ra2026 are HC-only;
    alkon2024 is early-only. The HC -> early -> advanced gradient is therefore drawn with study
    visible and never pooled silently. `STUDY_DOMINANCE` auto-derives the ban rather than
    asserting it: early is 17/21 donors li2024 = 0.81.

  * The panel is a frozen literal, not a runtime regex. A regex evaluated at run time would
    silently change the panel between runs. Every symbol below was checked against the v2 var.

The v1 predecessor (749,510 cells, 7 studies, a 156-column protease-only panel, its own frozen
gate constants) is retired: `notebooks/_archive/old/28_tme_degradome_v1.ipynb`, artifacts under
`data/degradome/_v1/`, `tables/_v1_deg42/`, `figures/degradome/_v1/`. None of its numbers are
comparable and none of them are carried here.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
NB_DIR = Path(__file__).resolve().parent.parent
ATLAS_JOINT = NB_DIR / "data" / "atlas_joint"

# 2,157,693 x 42,347; X byte-identical to layers['raw_counts'].
# NEVER sc.read_h5ad this file (112 GB) -- stream it (ccc_helpers.stream_lognorm_subset).
SOURCE_H5AD = ATLAS_JOINT / "joint_annotated.h5ad"
# 2,157,693 x 73. Carries cell_type_final / stage_group / skin_layer; does NOT carry any
# malignancy column -- those come from helpers/malignancy.py (see MALIGNANCY below).
OBS_PARQUET = ATLAS_JOINT / "atlas_obs_full_v2.parquet"
# nb22: cell_id, lineage, leiden_sub, subtype_fine, subtype_coarse; 328,461 rows.
SUBTYPE_CSV = ATLAS_JOINT / "skin_myeloid_fibro_subtypes.csv"
SUBTYPE_PROV = ATLAS_JOINT / "skin_myeloid_fibro_provenance.json"
# nb22b: cell_id, residual_source, leiden_resid, subtype_fine. Optional -- when absent the
# residual levels (Neutrophils beyond nb22's 306, NK, ILC) simply do not exist in the roster.
UNASSIGNED_CSV = ATLAS_JOINT / "skin_unassigned_subtypes.csv"

DEG_DIR = NB_DIR / "data" / "degradome"
DEG_ADATA = DEG_DIR / "skin_degradome.h5ad"
PSEUDOBULK_PQ = DEG_DIR / "skin_degradome_pseudobulk_full.parquet"
PSEUDOBULK_META = DEG_DIR / "skin_degradome_pseudobulk_meta.parquet"
# 3 donors at FULL gene width; exists only to gate the size factor (see
# degradome_helpers.assert_build_equivalence).
TESTSET_H5AD = DEG_DIR / "degradome_equivalence_testset.h5ad"
MANIFEST = DEG_DIR / "degradome_build_manifest.json"
# Every cached entry is stamped with the built object's mtime and size, so rebuilding
# skin_degradome.h5ad invalidates the lot automatically.
CACHE_DIR = DEG_DIR / "cache"

# ---------------------------------------------------------------- external healthy reference
# Reynolds 2021 (Science 371:eaba6500), CELLxGENE dataset f0f0d7c4-3bec-428e-9539-c99d36548d96 =
# E-MTAB-8142: 195,739 healthy adult skin cells, 5 donors, all female, all 10x 3' v2. This is the
# reference Li 2024 used to annotate the CTCL atlas, so its taxonomy lines up with cell_type_orig.
# Note what it does NOT give us: this deposit's `full_clustering` has 34 labels and collapses
# monocytes to `Mono_mac`, so it cannot anchor nb22's Mono_CD14/Mono_CD16 split -- that split
# rests on our own markers (cluster 22, FCGR3A 0.81), not on an external label. It is NOT in the
# joint atlas (docs/DATA_PROVENANCE.md:149, "ledger only, not cohorts"); our 83,643 HC skin cells
# come from il4ra2026 / gaydosik2023 / chennareddy2025 / gaydosik2019 / gaydosik2022 / herrera2021.
#
# One donor set, one lab, one chemistry, one sex. Nothing here is study-blockable against our
# atlas, so every external contrast is donor-level, unblocked, and carries a control-family
# calibration next to it (see helpers/external_ref.py).
EXT_DIR = NB_DIR / "data" / "external_ref"
EXT_H5AD = EXT_DIR / "reynolds2021_healthy_skin.h5ad"
EXT_PROV = EXT_DIR / "reynolds2021_provenance.json"
EXT_DEG_ADATA = DEG_DIR / "external_reynolds_degradome.h5ad"
EXT_NAME = "reynolds2021"
EXT_N_OBS = 195_739
EXT_DONOR_KEY = "donor_id"
EXT_LABEL_COL = "ext_label"          # the source's own fine clustering, carried verbatim
EXT_URL = ("https://datasets.cellxgene.cziscience.com/"
           "f8b5c4f4-7ece-4314-9f49-fc254bbd5c71.h5ad")
EXT_DATASET_ID = "f0f0d7c4-3bec-428e-9539-c99d36548d96"
EXT_COLLECTION_ID = "73f82ac8-15cc-4fcd-87f8-5683723fce7f"
EXT_DOI = "10.1126/science.aba6500"
EXT_ACCESSION = "E-MTAB-8142"

TAB_DIR = NB_DIR / "tables"
FIG_DIR = NB_DIR / "figures" / "degradome"
TAB_PREFIX = "deg28_"
FIG_PREFIX = "deg28_"


def tab(name: str) -> Path:
    """tables/deg28_<name>.csv"""
    return TAB_DIR / f"{TAB_PREFIX}{name}.csv"


def fig(name: str, ext: str = "png") -> Path:
    return FIG_DIR / f"{FIG_PREFIX}{name}.{ext}"


# ---------------------------------------------------------------- keys
COMPARTMENT = "Skin"
DISEASES = ["MF", "HC"]              # the scope; see the module docstring
CTCL_DISEASES = ["MF"]
HC_DISEASE = "HC"

SAMPLE_KEY = "sample_id"
DONOR_KEY = "donor"
STUDY_KEY = "study"
GROUPBY = "degradome_level"
LAYER = "lognorm"                    # log1p CP10K, size factor over all 42,347 genes
COUNTS_LAYER = "raw_counts"

CELLTYPE_SRC = "cell_type_final"     # nb21, 12 skin levels
SUBTYPE_SRC = "subtype_fine"         # nb22; subtype_coarse is the collapse
SUBTYPE_COARSE_SRC = "subtype_coarse"
RESIDUAL_SRC = "subtype_fine"        # nb22b, same column name in its own sidecar
SPLIT_LINEAGES = ["Myeloid", "Fibroblast"]

STAGE_KEY = "stage_group"            # HC | early | advanced | unknown
STAGE_ARMS = ["HC", "early", "advanced"]
LAYER_KEY = "skin_layer"

# ---------------------------------------------------------------- malignancy
# Exactly three calls, assembled by helpers/malignancy.py, never OR-ed together. See that
# module's docstring for why there is no combined flag.
MALIGNANCY = ["mal_tcr", "mal_cnv", "mal_knn", "mal_cluster"]
MALIG_HEADLINE = "mal_cluster"       # nb26b per-sample cluster call; headline since 2026-10-22
MALIG_SENSITIVITY = ["mal_tcr", "mal_cnv", "mal_knn"]
# mal_knn calls 23% of healthy CD4 malignant. The classifier never saw a healthy repertoire;
# this is its floor, and it is printed next to every figure that splits on mal_knn.
MALIG_HC_FP_KNN = 0.231
# mal_cluster has no HC floor because nb26b calls HC samples benign as a rule. That is an
# input to the call, not a property of it -- do not quote 0% as evidence of specificity.
MALIG_HC_FP_CLUSTER = 0.0
MALIG_HC_FP = {"mal_knn": MALIG_HC_FP_KNN, "mal_cluster": MALIG_HC_FP_CLUSTER}
MALIG_HC_FALSE_POSITIVE = MALIG_HC_FP_KNN    # back-compat alias; prefer MALIG_HC_FP[call]

CD4_MALIGNANT = "CD4_malignant"
CD4_REACTIVE = "CD4_reactive"
CD4_UNASSESSED = "CD4_unassessed"

# Carried onto the degradome object: the design axes, the gate inputs, and the depth columns.
OBS_COLS = [
    "cell_id", "donor", "real_donor", "sample_id", "study", "dataset", "compartment",
    "cell_type_final", "cell_type_broad", "cell_type", "lineage",
    "disease", "disease_stage", "entity", "stage_class", "stage_group",
    "skin_layer", "tissue", "lesion_type", "treatment_context", "sex", "tech", "cohort_region",
    "has_tcr", "n_genes", "total_counts", "pct_mito", "doublet_score",
]

# ---------------------------------------------------------------- roster
# nb22 subtype_fine levels. UNK is dropped from the roster (it has no identity to attribute to);
# the drop is recorded in the manifest and printed by the build.
MYELOID_LEVELS = ["LC", "cDC1", "cDC2", "DC_LAMP3", "pDC", "Mono_CD14", "Mono_CD16", "moDC",
                  "Mac_FOLR2", "Mac_SPP1_TREM2", "Mac_infl", "Neutrophils"]
FIBRO_LEVELS = ["F_papillary", "F_reticular", "F_mesenchymal", "F_myofibroblast",
                "F_inflammatory", "F_apCAF"]
TME_LEVELS = MYELOID_LEVELS + FIBRO_LEVELS

# nb22 subtype_fine counts over the full skin compartment (all diseases), asserted by
# degradome_helpers.assert_degradome_roster. If nb22 is re-run these move and the assert fires
# rather than the analysis silently changing substrate.
SIDECAR_COUNTS = {
    "F_inflammatory": 49966, "F_reticular": 49629, "F_apCAF": 27014, "Mac_FOLR2": 26475,
    "DC_LAMP3": 23723, "cDC2": 21689, "F_papillary": 15376, "Mac_infl": 14627,
    "moDC": 12447, "cDC1": 10684, "F_mesenchymal": 10696, "LC": 4051,
    "Mono_CD14": 7187, "Mono_CD16": 442,
    "Mac_SPP1_TREM2": 4034, "Myeloid_prolif": 1584, "pDC": 1361, "F_myofibroblast": 897,
    "Neutrophils": 306, "UNK": 46273,
}

# Levels statements are made about. Mono_CD16 is on the roster but not here: 442 cells over 73
# donors is ~6 cells each, so it can never clear MIN_CELLS in MIN_SAMPLES donors. nb22 §9 measures
# that; stating it up front is honest about what the split bought -- a population the atlas can
# show and cannot yet make a claim about.
NEVER_CLAIMABLE = ["Mono_CD16"]
CLAIM_LEVELS = ([lv for lv in TME_LEVELS if lv not in NEVER_CLAIMABLE]
                + [CD4_MALIGNANT, CD4_REACTIVE])

# Carried for the denominator and for context. No claim.
#
# NK and ILC exist only if nb22b has run, and they are here for a specific reason: nb21 folded
# ~55,877 NK-like cells into CD8, and build_degradome_level relabels CD8 cells from the nb22b
# sidecar. Without NK/ILC on this list those cells would be relabelled and then dropped for
# being outside the vocabulary -- so running nb22b would SHRINK the CD8 denominator and put
# the cells nowhere. They are context, not claim: NK granzyme content is real biology but it
# is not what this panel was assembled to ask about, and nb22b's own calls are new.
CONTEXT_LEVELS = [CD4_UNASSESSED, "CD4_Treg", "CD8", "B", "Plasma",
                  "Keratinocyte", "Vascular", "Mast", "Melanocyte",
                  "NK", "ILC"] + NEVER_CLAIMABLE

# Cells with no identity, dropped from the object entirely: they would sit in the denominator
# without being attributable to anything.
DROP_LEVELS = ["UNK", "Unk", "nan", "None", "NA", "Myeloid_prolif", "F_prolif", "Prolif",
               "Doublet", "LowQuality"]

CT_ORDER = CLAIM_LEVELS + CONTEXT_LEVELS
ALL_LEVELS = CT_ORDER

# ---------------------------------------------------------------- resolutions
# Low resolution = nb22 `subtype_coarse`, rebuilt from the fine levels on the object (the object
# carries only the fine label). Myeloid_prolif / UNK are not on the roster, so not here either.
MYELOID_COARSE = {
    "Mono_Mac": ["Mono_CD14", "Mono_CD16", "moDC", "Mac_FOLR2", "Mac_SPP1_TREM2", "Mac_infl"],
    "DC": ["cDC1", "cDC2", "DC_LAMP3", "pDC"],
    "LC": ["LC"],
    "Neutrophils": ["Neutrophils"],
}
FIBRO_COARSE = {
    "F_structural": ["F_papillary", "F_reticular", "F_mesenchymal", "F_myofibroblast"],
    "F_inflammatory": ["F_inflammatory"],
    "F_apCAF": ["F_apCAF"],
}

# Our level (either resolution) -> the external `ext_target` values pooled as its healthy
# comparator. Reynolds has no FOLR2/SPP1/monocyte split and no papillary/reticular/CAF split, so
# those levels are compared against the nearest parent (EXT_POOLED). pDC and Neutrophils have no
# counterpart in the deposit at all and are absent (-> no external arm).
_MONO_MAC_FINE = ["Mono_CD14", "Mono_CD16", "Mac_FOLR2", "Mac_SPP1_TREM2", "Mac_infl"]
EXT_ARM = {
    "CD4": ["CD4"],
    "LC": ["LC"], "cDC1": ["cDC1"], "cDC2": ["cDC2"], "DC_LAMP3": ["DC_LAMP3"],
    "moDC": ["moDC"],
    "DC": ["cDC1", "cDC2", "DC_LAMP3"],
    "Mono_Mac": ["Mono_Mac", "moDC"],
    **{lv: ["Mono_Mac"] for lv in _MONO_MAC_FINE},
    **{lv: ["Fibroblast"] for lv in list(FIBRO_COARSE) + FIBRO_LEVELS},
}
EXT_POOLED = set(_MONO_MAC_FINE) | set(FIBRO_COARSE) | set(FIBRO_LEVELS)

# Display order for the per-gene figures: proteases by catalytic class first, then inhibitors,
# hyaluronan, crosslinking, substrate, and the grip/motility machinery. Controls are left out.
FAMILY_ORDER = {
    "mmp": ["mmp"],
    "cathepsin": ["cathepsin"],
    "serine_other": ["serine_other"],
    "metallo_other": ["metallo_other"],
    "granzyme": ["granzyme"],
    "inhibitor": ["inhibitor"],
    "hyaluronan": ["ha_synthase", "ha_catabolic"],
    "crosslink": ["crosslink"],
    "glycosidase": ["glycosidase"],
    "collagen_fibrillar": ["collagen_fibrillar"],
    "collagen_bmz": ["collagen_bmz"],
    "collagen_minor": ["collagen_minor"],
    "proteoglycan": ["proteoglycan"],
    "elastic_fiber": ["elastic_fiber"],
    "matricellular": ["matricellular"],
    "adhesion_ligand": ["adhesion_ligand"],
    "matrix_receptor": ["matrix_receptor"],
    "motility": ["motility"],
}

# ---------------------------------------------------------------- gates
# A (level, donor) pair contributes a donor median only if it clears MIN_CELLS; a level is
# claimable on an axis only if both arms clear MIN_SAMPLES donors.
MIN_CELLS = 25
MIN_SAMPLES = 5
MIN_DONORS_PER_ARM = 4
# An arm whose minority side is more than this fraction one study may not be pooled.
STUDY_DOMINANCE = 0.8

# ---------------------------------------------------------------- the panel
# Grouped by catalytic class where that drives the biology, by matrix compartment where it does
# not. GENE_TO_FAMILY is flat, so a gene lives in exactly ONE family -- the four genes that used
# to sit in CONTEXT_GENES (COL1A1, COL1A2, VCAN, S100A8) moved into real families because they
# are now analysed rather than carried. CTSA stays in `cathepsin` despite its elastin-receptor
# role: re-keying it would silently redefine the cathepsin_cystatin denominator.
DEGRADOME_FAMILIES = {
    # ---- proteases -------------------------------------------------------------------
    "mmp": ["MMP1", "MMP2", "MMP3", "MMP7", "MMP8", "MMP9", "MMP10", "MMP11", "MMP12",
            "MMP13", "MMP14", "MMP15", "MMP16", "MMP17", "MMP19", "MMP23B", "MMP24",
            "MMP25", "MMP26", "MMP28"],
    "cathepsin": ["CTSA", "CTSB", "CTSC", "CTSD", "CTSE", "CTSF", "CTSG", "CTSH", "CTSK",
                  "CTSL", "CTSO", "CTSS", "CTSV", "CTSW", "CTSZ", "LGMN"],
    "granzyme": ["GZMA", "GZMB", "GZMH", "GZMK", "GZMM", "PRF1"],
    "serine_other": ["PLAU", "PLAUR", "PLAT", "PLG", "ELANE", "PRTN3",
                     "KLK5", "KLK6", "KLK7", "KLK8", "KLK9", "KLK10", "KLK11", "KLK12",
                     "KLK13", "KLK14",
                     "TMPRSS2", "TMPRSS4", "TMPRSS11D", "TMPRSS11E", "TMPRSS13", "ST14",
                     "PRSS8", "PRSS21", "PRSS22", "PRSS23",
                     "HTRA1", "HTRA3", "FURIN", "PCSK5", "PCSK6",
                     "FAP", "DPP4"],
    "metallo_other": ["ADAM8", "ADAM9", "ADAM10", "ADAM12", "ADAM15", "ADAM17", "ADAM19",
                      "ADAM28", "ADAMDEC1",
                      "ADAMTS1", "ADAMTS2", "ADAMTS3", "ADAMTS4", "ADAMTS5", "ADAMTS7",
                      "ADAMTS8", "ADAMTS9", "ADAMTS12", "ADAMTS13", "ADAMTS14", "ADAMTS15",
                      "ADAMTS20",
                      "ANPEP", "MME", "ECE1"],
    "inhibitor": ["TIMP1", "TIMP2", "TIMP3", "TIMP4",
                  "CST3", "CST6", "CST7", "CSTA", "CSTB",
                  "SERPINA1", "SERPINA3", "SERPINA5", "SERPINB1", "SERPINB2", "SERPINB6",
                  "SERPINB8", "SERPINB9", "SERPINE1", "SERPINE2", "SERPINF2", "SERPING1",
                  "SPINK5", "A2M", "SLPI", "PI3"],
    # ---- hyaluronan, crosslinking, glycosidases ---------------------------------------
    "ha_synthase": ["HAS1", "HAS2", "HAS3"],
    "ha_catabolic": ["HYAL1", "HYAL2", "HYAL3", "CEMIP", "TMEM2"],
    "crosslink": ["LOX", "LOXL1", "LOXL2", "LOXL3", "LOXL4"],
    "glycosidase": ["HPSE", "HPSE2", "NEU1"],
    # ---- the substrate the enzymes above act on ---------------------------------------
    "collagen_fibrillar": ["COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2", "COL5A3",
                           "COL11A1", "COL24A1", "COL27A1"],
    "collagen_bmz": ["COL4A1", "COL4A2", "COL4A3", "COL4A4", "COL4A5", "COL4A6",
                     "COL7A1", "COL17A1"],
    "collagen_minor": ["COL6A1", "COL6A2", "COL6A3", "COL12A1", "COL13A1", "COL14A1",
                       "COL15A1", "COL16A1", "COL18A1", "COL21A1"],
    "proteoglycan": ["DCN", "VCAN", "BGN", "ASPN", "LUM", "FMOD", "OGN", "PRELP", "DPT",
                     "HSPG2", "SDC1", "SDC4"],
    "elastic_fiber": ["ELN", "FBN1", "FBN2", "FBLN5", "MFAP2", "MFAP4", "MFAP5", "EMILIN1",
                      "LTBP1", "LTBP2"],
    "matricellular": ["POSTN", "TNC", "SPARC", "THBS1", "THBS2", "THBS4", "COMP", "HMCN1",
                      "FBLN1", "FBLN2", "EFEMP1", "EFEMP2", "NID1", "NID2"],
    "adhesion_ligand": ["FN1", "LAMA3", "LAMB3", "LAMC2", "LAMA5"],
    # ---- how a cell grips and moves through it ----------------------------------------
    "matrix_receptor": ["CD44", "BSG", "ITGAL", "ITGA2", "ITGAV", "ITGA6",
                        "ITGB1", "ITGB2", "ITGB3", "ITGB4"],
    "motility": ["WAS", "WIPF1", "ILK", "PTK2", "PTK2B", "TLN1", "FERMT3", "RAP1A", "RAP1B",
                 "WNK1", "OXSR1", "STK39", "SLC12A2"],
    # ---- context / controls ------------------------------------------------------------
    # Mast granule proteases: the attribution has a known right answer here (verification).
    "mast": ["CMA1", "CPA3", "TPSAB1", "TPSB2"],
    # Granulocyte identity only -- ELANE/PRTN3/CTSG/MMP8/MMP9 are already in the protease
    # families above and must not be duplicated.
    "granulocyte": ["MPO", "FCGR3B", "CSF3R", "CXCR1", "CXCR2", "S100A8", "S100A9",
                    "S100A12", "LTF", "CAMP", "LCN2", "SELL"],
    # The Th2 -> periostin -> TSLP loop: the ligand context for the matricellular figure.
    "th2_loop": ["IL4", "IL13", "TSLP", "IL4R", "IL13RA1", "CRLF2"],
    # Not a degradome enzyme; it is the reason IL1B leaves Mac_infl.
    "caspase": ["CASP1"],
}

# Deliberately excluded, so that "we did not look" is distinguishable from "it was not there":
#   COL2A1, COL9A1-3, COL10A1, COL11A2, COL19A1, COL20A1, COL22A1, COL23A1, COL25A1
#       -- cartilage/neural collagens. Present in var, structural zeros in skin; carrying them
#          would only inflate the multiple-testing denominator.
#   MMP18, MMP22  -- no human gene symbol (they are Xenopus / chicken).
#   MMP20, MMP21, MMP27, MMP23A -- enamel / undetected in skin; MMP23B is the expressed paralog.
EXCLUDED_WITH_REASON = {
    "cartilage_neural_collagens": ["COL2A1", "COL9A1", "COL9A2", "COL9A3", "COL10A1",
                                   "COL11A2", "COL19A1", "COL20A1", "COL22A1", "COL23A1",
                                   "COL25A1"],
    "no_human_symbol": ["MMP18", "MMP22"],
    "not_expressed_in_skin": ["MMP20", "MMP21", "MMP27", "MMP23A"],
}

# Families that carry claims about degradative machinery.
CLAIM_FAMILIES = ["mmp", "cathepsin", "granzyme", "serine_other", "metallo_other", "inhibitor",
                  "ha_synthase", "ha_catabolic", "crosslink", "glycosidase",
                  "motility", "matrix_receptor"]
# Described, never claimed as "degradome": these are what gets degraded.
SUBSTRATE_FAMILIES = ["collagen_fibrillar", "collagen_bmz", "collagen_minor", "proteoglycan",
                      "elastic_fiber", "matricellular", "adhesion_ligand"]
CONTROL_FAMILIES = ["mast", "granulocyte", "th2_loop", "caspase"]

# Display sets. Eight panels is what a figure can carry without becoming a heatmap.
HEADLINE_PROTEASE = ["mmp", "cathepsin", "serine_other", "metallo_other",
                     "ha_catabolic", "glycosidase", "inhibitor", "crosslink"]
HEADLINE_MATRIX = ["collagen_fibrillar", "collagen_bmz", "elastic_fiber", "proteoglycan",
                   "matricellular", "crosslink", "mmp", "inhibitor"]

# Families too small for a family mean to be a distribution -- plotted per gene instead.
MIN_FAMILY_SIZE_FOR_SCORE = 5

# Protease : inhibitor balance. Numerator is a family key or an explicit gene list; denominator
# is always an explicit list.
BALANCE_PAIRS = {
    "mmp_timp": ("mmp", ["TIMP1", "TIMP2", "TIMP3", "TIMP4"]),
    "cathepsin_cystatin": ("cathepsin", ["CST3", "CST6", "CST7", "CSTA", "CSTB"]),
    "serine_serpin": ("serine_other", ["SERPINA1", "SERPINA3", "SERPINB1", "SERPINB2",
                                       "SERPINB6", "SERPINB8", "SERPINB9", "SERPINE1",
                                       "SERPINE2", "SERPING1", "SLPI", "PI3", "SPINK5"]),
    "collagenase_timp": (["MMP1", "MMP8", "MMP13", "MMP14", "CTSK"],
                         ["TIMP1", "TIMP2", "TIMP3", "TIMP4"]),
    "ha_turnover": (["HYAL1", "HYAL2", "HYAL3", "CEMIP", "TMEM2"],
                    ["HAS1", "HAS2", "HAS3"]),
    # Degrade over deposit -- the one scalar that carries the whole question.
    "remodel_tone": (["MMP1", "MMP2", "MMP3", "MMP9", "MMP12", "MMP13", "MMP14",
                      "CTSK", "CTSS", "CTSL", "HPSE", "ELANE"],
                     ["COL1A1", "COL1A2", "COL3A1", "ELN", "FN1", "POSTN",
                      "LOX", "LOXL1", "TNC"]),
}

# Identity markers. Not analysed -- they ride along so the built object can prove it is what the
# build claims (a 280-gene object cannot be sanity-checked on a UMAP).
CONTEXT_GENES = [
    "PTPRC", "CD3E", "CD4", "CD8A", "FOXP3", "IL2RA", "MS4A1", "JCHAIN",
    "KRT14", "KRT10", "PDGFRB", "PECAM1", "LYVE1", "MLANA", "PMEL",
    "LYZ", "CD68", "C1QA", "FOLR2", "SPP1", "TREM2", "LAMP3", "CCR7", "CD207", "CD1A",
    "CLEC9A", "CD1C", "LILRA4", "IL3RA", "FCN1", "IL1B", "MKI67",
    "NKG7", "GNLY", "KLRD1",
]


# ---------------------------------------------------------------- fibrosis axes (nb28 section 9)
# How the malignant clone drives the stromal fibrotic program, as eight mechanistic axes.
#
# Deliberately NOT folded into DEGRADOME_FAMILIES: 44 of these 82 symbols already live in a
# degradome family, and GENE_TO_FAMILY is flat -- folding them in would silently re-key MMP12,
# TIMP1, CTSB and friends out of their catalytic class and change what `panel_coverage` reports
# for the existing figures. This is a second, overlapping view of the same object, scored next to
# the degradome panel and never substituted for it. FAMILY_ORDER is untouched, so sections 4-6
# reproduce unchanged.
FIBROSIS_AXES = {
    "th2_fibrotic": ["IL13", "IL4", "IL5", "IL31", "IL9", "IL22",
                     "POSTN", "TNC", "HAS2", "MMP12", "CD163",
                     "IL4R", "IL13RA1", "IL13RA2"],
    "gp130_stat3": ["OSM", "IL6", "IL11", "LIF",
                    "TIMP1", "TIMP3", "MMP1", "MMP3", "PLAUR",
                    "OSMR", "IL6ST", "IL6R", "IL11RA"],
    "tgfb_activation": ["ITGB8", "ITGB6",
                        "FAP", "PRSS23", "SERPINE1", "LOX", "LOXL1", "LOXL2", "LOXL3",
                        "ADAMTS2", "LTBP2", "FN1", "TGFBR1", "TGFBR2"],
    "juxtacrine": ["TNFSF14", "CD40LG", "TNFSF11",
                   "CTSK", "LTBR", "TNFRSF14", "CD40", "TNFRSF11A"],
    "infl_matrix_suppression": ["IL1A", "IL1B", "TNF",
                                "MMP9", "ELN", "FBLN1", "FBLN5", "MFAP4", "MFAP5",
                                "OGN", "ASPN", "IL1R1", "TNFRSF1A"],
    "caf_expansion": ["PDGFA", "PDGFB", "PDGFRA", "PDGFRB"],
    "bsg_resolution": ["PPIA", "PPIB", "BSG"],
    "myeloid_innate": ["S100A8", "S100A9", "MIF", "SPP1", "LGALS1", "LGALS3", "LGALS9",
                       "SERPINA1", "CTSB", "CTSC", "CTSH", "CTSS", "CTSZ"],
}

FIBROSIS_ORDER = list(FIBROSIS_AXES)

# `caf_expansion` (4 genes) and `bsg_resolution` (3) sit below MIN_FAMILY_SIZE_FOR_SCORE, so
# `family_score_cells` refuses them a mean and they are read per gene. Named here so the run log
# states it rather than leaving two axes quietly missing from the paired table.
FIBROSIS_NO_SCORE = [k for k, v in FIBROSIS_AXES.items() if len(v) < MIN_FAMILY_SIZE_FOR_SCORE]

# Which side of the axis a gene is read on: what a CD4 cell can emit, and what the stroma
# answers with. Drives the two-panel layout in nb28 section 9 -- a ligand is only evidence if it
# is in the T cell, a receptor only if it is in the fibroblast or myeloid cell.
FIBROSIS_SIDE = {
    "th2_fibrotic": {
        "ligand": ["IL13", "IL4", "IL5", "IL31", "IL9", "IL22"],
        "stroma": ["POSTN", "TNC", "HAS2", "MMP12", "CD163",
                   "IL4R", "IL13RA1", "IL13RA2"]},
    "gp130_stat3": {
        "ligand": ["OSM", "IL6", "IL11", "LIF"],
        "stroma": ["TIMP1", "TIMP3", "MMP1", "MMP3", "PLAUR",
                   "OSMR", "IL6ST", "IL6R", "IL11RA"]},
    "tgfb_activation": {
        # The T cell contributes the integrin that frees latent TGF-beta, not the ligand.
        "ligand": ["ITGB8", "ITGB6"],
        "stroma": ["FAP", "PRSS23", "SERPINE1", "LOX", "LOXL1", "LOXL2", "LOXL3",
                   "ADAMTS2", "LTBP2", "FN1", "TGFBR1", "TGFBR2"]},
    "juxtacrine": {
        "ligand": ["TNFSF14", "CD40LG", "TNFSF11"],
        "stroma": ["CTSK", "LTBR", "TNFRSF14", "CD40", "TNFRSF11A"]},
    "infl_matrix_suppression": {
        "ligand": ["IL1A", "IL1B", "TNF"],
        "stroma": ["MMP9", "ELN", "FBLN1", "FBLN5", "MFAP4", "MFAP5",
                   "OGN", "ASPN", "IL1R1", "TNFRSF1A"]},
    "caf_expansion": {
        "ligand": ["PDGFA", "PDGFB"],
        "stroma": ["PDGFRA", "PDGFRB"]},
    # BSG is homophilic: it is the ligand and the receptor, so it appears on both sides.
    "bsg_resolution": {
        "ligand": ["BSG", "PPIA", "PPIB"],
        "stroma": ["BSG"]},
    "myeloid_innate": {
        "ligand": ["S100A8", "S100A9", "MIF", "SPP1", "LGALS1", "LGALS3", "LGALS9"],
        "stroma": ["SERPINA1", "CTSB", "CTSC", "CTSH", "CTSS", "CTSZ"]},
}

# ---------------------------------------------------------------- CCC panel (nb28 section 10)
# The curated ligand-receptor panel, carried on the object so an edge the CCC section scores in
# the pseudobulk cube can be checked per cell as well. `ccc_panel` has no imports of its own and
# lives in helpers/ for the flat-import contract in MF/CLAUDE.md.
import ccc_panel as _ccc_panel  # noqa: E402  (a frozen literal, not a dependency)

CCC_AXIS_GENES = list(_ccc_panel.ALL_GENES)

# ------------------------------------------------- nb28 panel extension (docs/NB28_panel_extension_spec.md)
# The signals the clone emits and the receptors the stroma answers with, plus metabolic and
# transcription-factor controls. Deliberately NOT folded into DEGRADOME_FAMILIES: these are
# signalling genes, not degradome catalytic classes, and re-keying would change the family
# denominators `panel_coverage` and the balance ratios report. 30 of the 59 spec symbols are
# already on the panel through `ccc_panel`; `panel_genes` dedupes, so only the 29 new ones are
# appended. ITGAV and THBS1 are spec genes that already have a family (matrix_receptor,
# matricellular) and keep it -- the annotation update below never re-keys.
PANEL_EXTENSION = {
    "juxtacrine":        ["CD40LG", "LTA", "LTB"],
    "gp130_stat3_x":     ["OSM", "IL11", "IL11RA"],
    "tgfb_activation_x": ["TGFB1", "TGFB3", "LRRC32", "ITGB8", "TGM2"],
    "caf_expansion_x":   ["PDGFB"],
    "ecm_signalling":    ["SPP1"],
    "lipid_purinergic":  ["ENPP2", "ENTPD1", "NT5E", "LPAR1", "ADORA2B", "ADORA2A"],
    "egfr_ligands":      ["AREG", "HBEGF", "EGFR"],
    "angiogenic":        ["VEGFA"],
    "chemokine":         ["CCL2", "CXCL13", "CCR2"],
    "wnt_notch":         ["WNT5A", "JAG1", "ROR2", "NOTCH1", "NOTCH2", "NOTCH3"],
    "il17_il22":         ["IL17A", "IL17F", "IL22", "IL26", "IL17RA", "IL17RC", "IL22RA1"],
    "metabolic_control": ["SLC16A1", "SLC16A3", "LDHA", "ENO1", "PKM", "SLC2A1", "HIF1A", "MYC"],
    "regulators":        ["GATA3", "IRF4", "TOX", "BATF", "RUNX2", "STAT3", "RELB"],
    "mhc_ii":            ["HLA-DRA", "HLA-DRB1", "CIITA"],
}


def panel_genes(include_context: bool = True) -> list:
    """Every symbol the built object must carry, deduped, family order preserved.

    Four blocks, in this order: the degradome families, the eight fibrosis axes, the curated
    CCC panel, the nb28 panel extension. The later blocks overlap the families heavily -- the
    dedupe is what keeps the object one column per symbol while the views stay separate.
    """
    out, seen = [], set()

    def _add(genes):
        for g in genes:
            if g not in seen:
                out.append(g)
                seen.add(g)

    for genes in DEGRADOME_FAMILIES.values():
        _add(genes)
    for genes in FIBROSIS_AXES.values():
        _add(genes)
    _add(CCC_AXIS_GENES)
    for genes in PANEL_EXTENSION.values():
        _add(genes)
    if include_context:
        _add(CONTEXT_GENES)
    return out


GENE_TO_FAMILY = {g: fam for fam, genes in DEGRADOME_FAMILIES.items() for g in genes}
# Additive only: a gene the degradome families already own keeps its family, so the catalytic
# classes -- and every denominator built on them -- are unchanged.
GENE_TO_FAMILY.update({g: fam for fam, genes in PANEL_EXTENSION.items()
                       for g in genes if g not in GENE_TO_FAMILY})


def family_genes(key):
    """Resolve a BALANCE_PAIRS numerator: a family key, or an explicit gene list."""
    if isinstance(key, (list, tuple)):
        return list(key)
    return list(DEGRADOME_FAMILIES[key])


# ---------------------------------------------------------------- attribution controls
MIN_DETECTION = 0.01
OWNER_DONOR_FRAC = 2 / 3
MIN_DONORS_FOR_OWNERSHIP = 5
N_SHUFFLE = 20
SEED = 0
CONTROL_MIN_CELLS = 5

# Attribution has a known right answer for these. Values are the LINEAGE-level owner the
# arithmetic must recover; a miss means the method is broken and nothing downstream is readable.
POSITIVE_CONTROLS_RAW = {          # abundance-weighted: who supplies the tissue
    "KLK5": {"Keratinocyte"}, "KLK7": {"Keratinocyte"}, "SPINK5": {"Keratinocyte"},
    "LAMB3": {"Keratinocyte"}, "LAMC2": {"Keratinocyte"},
    "COL1A1": {"Fibroblast"}, "COL1A2": {"Fibroblast"}, "ELN": {"Fibroblast"},
    "C1QA": {"Myeloid"}, "CTSS": {"Myeloid"},
    "CTSK": {"Myeloid", "Fibroblast"},
}
POSITIVE_CONTROLS_CPM = {          # per-transcriptome intensity: who is built to make it
    "GZMA": {"CD8"}, "GZMK": {"CD8"}, "GZMH": {"CD8"},
    "CD207": {"Myeloid"}, "MLANA": {"Melanocyte"},
}
# Plurality only -- real multi-source biology or coverage too thin for the ownership bar.
POSITIVE_CONTROLS_PLURALITY_RAW = {
    "TPSAB1": {"Mast"}, "TPSB2": {"Mast"}, "CPA3": {"Mast"},
}
POSITIVE_CONTROLS_PLURALITY_CPM = {
    "CMA1": {"Mast"}, "TPSAB1": {"Mast"},
}
# Added only if nb22b clears the coverage gate for Neutrophils; the notebook wires these in
# conditionally rather than failing when the level does not exist.
NEUTROPHIL_CONTROLS_CPM = {"ELANE": {"Neutrophils"}, "PRTN3": {"Neutrophils"},
                           "FCGR3B": {"Neutrophils"}}

# GZMB is deliberately NOT a control: plasma cells and pDC express it and per unit of
# transcriptome they beat CD8 in skin. A control whose right answer has to be looked up after
# seeing the result is not a control.

# ---------------------------------------------------------------- forbidden
FORBIDDEN_CONTRASTS = {
    "three_way_pooled": "HC vs early vs advanced pooled across studies is forbidden: NO study "
                        "carries all three arms. li2024 has early+advanced and no HC; "
                        "chennareddy2025 and gaydosik2022 have HC+advanced and no early. Draw "
                        "the gradient with study visible, and test only the two within-study "
                        "contrasts.",
    "stage_pooled": "early-vs-advanced pooled is forbidden -- early is 17 of 21 donors li2024 "
                    "(0.81 > STUDY_DOMINANCE), so a pooled fit compares li2024 to everyone "
                    "else. rindler2021_mc (5 advanced / 2 early) and alkon2024 (0 / 2) do not "
                    "clear MIN_DONORS_PER_ARM.",
    "hc_pooled_headline": "CTCL-vs-HC pooled confounds disease with study: HC spans 6 studies "
                          "and MF spans 10. FOUR studies carry both arms at >= 4 donors each "
                          "-- il4ra2026 (11 MF / 4 HC), chennareddy2025 (7/4), gaydosik2019 "
                          "(5/4), gaydosik2022 (5/4); herrera2021 has 5 MF and 1 HC and does "
                          "not qualify. So a study-BLOCKED HC contrast is legitimate here and "
                          "is the only HC axis in this notebook that is not confounded by "
                          "cohort. Pooled without blocking is still forbidden. "
                          "(Counted 2026-09-18 from atlas_obs_full_v2.parquet; the earlier "
                          "'only chennareddy2025 and gaydosik2022' was a v1 count.)",
    "cell_level_pvalues": "a p-value computed over cells, unstratified, is not a test of "
                          "anything -- cells within a donor are not independent and one "
                          "deep-sequenced patient can carry it. Stratify by study and report "
                          "the median per-donor difference next to every p-value.",
    "combined_malignancy": "the three malignancy calls are never OR-ed. mal_cnv derives from "
                           "smoothed regional expression, which is partly the signal this "
                           "analysis then attributes; a disjunction hides which measurement "
                           "fired. Pick one, say which, show the others survive.",
    "knn_in_hc": "mal_knn calls 23% of healthy CD4 malignant -- it never saw a healthy "
                 "repertoire. CD4_malignant is not a meaningful level inside the HC arm; in HC, "
                 "CD4 is reactive by construction.",
    "cluster_in_hc": "under mal_cluster the HC arm is 0% malignant because nb26b assigns HC "
                     "samples benign as a rule. The CD4 split stays MF-only for the same "
                     "reason it did under mal_knn: in HC there is nothing to split.",
    "context_level_claims": "Keratinocyte, CD8, Mast, Vascular, Plasma, Melanocyte, CD4_Treg "
                            "and CD4_unassessed are denominator only. They make the shares "
                            "interpretable; they are not the subject.",
}
