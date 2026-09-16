"""Config for the skin TME degradome analysis (notebook 42).

The question. nb31 scores a single `protease` program on malignant subclones
(`subclone_helpers.PROGRAM_SETS["protease"]`, 11 genes, `sc.tl.score_genes` per cell) and
renders it as one column of a centred program heatmap. It is never tested, the per-gene dot
plot is dead code, and the one time the panel was tested gene-by-gene
(`old/23_subclonal_evolution.ipynb` cell 17) the sole up-call was TIMP1 -- the inhibitor. The
read "ECM proteolysis in MF skin is not tumour-intrinsic" is supported, but the next question
was never asked: then which cell makes it? nb10c produced 13 myeloid / fibroblast subtypes that
did not exist when the panel was written, and nb40 uses them only for ligand-receptor work.
This module configures the expression-side answer.

Design decisions, with the numbers that forced them.

  * The CLAIM roster is the 13 nb10c `subtype_ccc` levels plus `CD4_malignant` as the reference
    comparator (it is what nb31 measured). Every other skin level is carried in the object and
    enters the attribution denominator, flagged claim=False. "Fraction of skin output
    attributable to level X" is undefined without a complete denominator, so Keratinocyte,
    CD8, Mast, Vascular and the rest have to be present -- but nothing is claimed about them.
    Mast (CMA1/CPA3/TPSAB1/TPSB2) will own several genes; that is the positive control, not a
    result.

  * Expression comes from `joint_annotated.h5ad` (40,821 symbols, raw counts), NOT
    `joint_mrvi_input_skin.h5ad`. The 10k-HVG object carries 168 of these genes but drops
    ADAM10 and ADAM17 -- both in the nb31 panel, both too ubiquitous to be HVG. Dropping the
    two genes the tumour-intrinsic panel is being compared against is not acceptable, so the
    object is rebuilt by one streaming pass.

  * The CTCL-vs-HC contrast is defined for 6 of the 13 levels, not 13. Skin has 9 HC donors in
    total, from 3 studies. At MIN_CELLS=25, HC donors per level: cDC 8, F_reticular 8,
    F_papillary 8, F_mesenchymal 7, Mac_FOLR2 5, F_apCAF 4, and ZERO for DC_LAMP3, Mac_infl,
    Mono_moDC, Mac_SPP1_TREM2, LC, pDC, F_inflammatory. Those seven are not null -- they are
    absent from healthy skin at the coverage gate, which is an abundance observation reported
    separately (nb42 section 5) and never as a fold change.

  * The contrast is run WITHIN study. chennareddy2025 (18 CTCL / 4 HC) and gaydosik2019
    (5 CTCL / 4 HC) each carry both arms at every testable level; the other five skin studies
    are CTCL-only, so a pooled test would confound disease with study. Pooled is a labelled
    sensitivity arm, never a headline number.

  * The panel is a frozen literal list, not a runtime regex. The regex was run once against
    `joint_annotated.h5ad` var to propose candidates; the accepted symbols are pasted below and
    `assert_panel()` checks them against the built object. A regex evaluated at run time would
    silently change the panel between runs.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
NB_DIR = Path(__file__).resolve().parent.parent
ATLAS_JOINT = NB_DIR / "data" / "atlas_joint"

# 1,173,694 x 40,821; X byte-identical to layers['raw_counts'] (float64 CSR).
# NEVER sc.read_h5ad this file -- stream it (ccc_helpers.stream_lognorm_subset).
SOURCE_H5AD = ATLAS_JOINT / "joint_annotated.h5ad"
# 1,173,694 x 76; the only place cell_type_final / mal_tcr_alice / stage_group live.
OBS_PARQUET = ATLAS_JOINT / "atlas_obs_full.parquet"
# nb10c: cell_id, lineage, leiden_sub, subtype_fine, subtype_ccc; 160,195 rows.
SUBTYPE_CSV = ATLAS_JOINT / "skin_myeloid_fibro_subtypes.csv"
SUBTYPE_PROV = ATLAS_JOINT / "skin_myeloid_fibro_provenance.json"

DEG_DIR = NB_DIR / "data" / "degradome"
DEG_ADATA = DEG_DIR / "skin_degradome.h5ad"
PSEUDOBULK_PQ = DEG_DIR / "skin_degradome_pseudobulk_full.parquet"
PSEUDOBULK_META = DEG_DIR / "skin_degradome_pseudobulk_meta.parquet"
# 3 donors at FULL gene width; exists only to gate the size factor (see
# degradome_helpers.assert_build_equivalence).
TESTSET_H5AD = DEG_DIR / "degradome_equivalence_testset.h5ad"
MANIFEST = DEG_DIR / "degradome_build_manifest.json"
# Results too slow to recompute on every pass (the DESeq2 fits are ~30 of the notebook's 37
# minutes). Every entry is stamped with the built object's mtime and size, so rebuilding
# skin_degradome.h5ad invalidates the lot automatically -- a silently stale cache is worse
# than a slow notebook.
CACHE_DIR = DEG_DIR / "cache"

TAB_DIR = NB_DIR / "tables"
FIG_DIR = NB_DIR / "figures" / "degradome"
TAB_PREFIX = "deg42_"
FIG_PREFIX = "deg42_"


def tab(name: str) -> Path:
    """tables/deg42_<name>.csv"""
    return TAB_DIR / f"{TAB_PREFIX}{name}.csv"


def fig(name: str, ext: str = "png") -> Path:
    return FIG_DIR / f"{FIG_PREFIX}{name}.{ext}"


# ---------------------------------------------------------------- keys
COMPARTMENT = "Skin"
SAMPLE_KEY = "sample_id"
DONOR_KEY = "donor"
STUDY_KEY = "study"
GROUPBY = "degradome_level"
LAYER = "lognorm"
COUNTS_LAYER = "raw_counts"

CELLTYPE_SRC = "cell_type_final"     # nb10b, 12 skin levels
SUBTYPE_SRC = "subtype_ccc"          # nb10c coverage-gated collapse; subtype_fine is report-only
SPLIT_LINEAGES = ["Myeloid", "Fibroblast"]

# Same malignancy source as nb40: ALICE-TCR alone. mal_combined is `alice | inferCNV`, and
# inferCNV derives its call from smoothed regional expression -- partly the same measurement
# this notebook then attributes. ALICE is a TCR-sequence fact, so call and expression are
# independent. TCR_ASSESSED_SRC demotes CD4 with no recovered CDR3 to CD4_unassessed: ALICE
# never tested them, and absence of data is not evidence of benignity.
MALIG_SRC = "mal_tcr_alice"
TCR_ASSESSED_SRC = "assessed_tcr_nb30"

# Carried onto the degradome object. The design axes, the gate inputs, and the depth columns
# the spillover control needs -- nothing that would make a 150-gene object look reusable for DE.
OBS_COLS = [
    "cell_id", "donor", "real_donor", "sample_id", "study", "dataset", "compartment",
    "cell_type_final", "cell_type_broad", "cell_type", "lineage",
    "mal_tcr_alice", "mal_combined", "assessed_tcr_nb30", "malignant_evidence",
    "disease", "disease_stage", "entity", "stage_class", "stage_group",
    "skin_layer", "tissue", "lesion_type", "treatment_context", "sex", "tech", "cohort_region",
    "n_genes", "total_counts", "pct_mito", "doublet_score",
]

# ---------------------------------------------------------------- roster
# The 13 nb10c levels. Cell counts are the sidecar's `subtype_ccc` value counts and are
# asserted by degradome_helpers.assert_degradome_roster -- if nb10c is re-run these move and
# the assert fires rather than the analysis silently changing substrate.
MYELOID_LEVELS = ["LC", "cDC", "DC_LAMP3", "pDC", "Mono_moDC",
                  "Mac_FOLR2", "Mac_SPP1_TREM2", "Mac_infl"]
FIBRO_LEVELS = ["F_papillary", "F_reticular", "F_mesenchymal", "F_inflammatory", "F_apCAF"]
TME_LEVELS = MYELOID_LEVELS + FIBRO_LEVELS

SIDECAR_COUNTS = {
    "cDC": 30650, "F_reticular": 23128, "DC_LAMP3": 19068, "F_inflammatory": 14144,
    "Mac_FOLR2": 13943, "Mac_infl": 12731, "F_apCAF": 10236, "F_papillary": 8262,
    "Mono_moDC": 7047, "F_mesenchymal": 6146, "Mac_SPP1_TREM2": 3835, "LC": 2861, "pDC": 1113,
}

CD4_MALIGNANT = "CD4_malignant"
CD4_REACTIVE = "CD4_reactive"
CD4_UNASSESSED = "CD4_unassessed"

# Levels statements are made about. CD4_malignant is here because it is the comparator the
# whole question is framed against; the other CD4 levels are context.
CLAIM_LEVELS = TME_LEVELS + [CD4_MALIGNANT]

# Carried for the denominator only. No claim, no test, no gate.
CONTEXT_LEVELS = [CD4_REACTIVE, CD4_UNASSESSED, "CD8", "Tregs", "B", "Plasma",
                  "Keratinocyte", "Vascular", "Mast", "Melanocyte"]

# Cells with no identity. Dropped from the object entirely: they would sit in the denominator
# without being attributable to anything, which biases every share downward by an unknown
# amount. UNK is 22,354 nb10b skin leftovers (3.0%); the 7,031 myeloid/fibroblast cells whose
# nb10c subtype_ccc is null (UNK + proliferating + pericyte contamination) go the same way.
# Both drops are recorded in the manifest and printed by the build.
DROP_LEVELS = ["UNK", "nan", "None", "NA"]

# Display / row order for every table and figure: claim levels first, context after.
CT_ORDER = CLAIM_LEVELS + CONTEXT_LEVELS
ALL_LEVELS = CT_ORDER

# ---------------------------------------------------------------- claim gate
# Same gate as nb40 / nb10c section 8: a level must clear MIN_CELLS in MIN_SAMPLES donors
# inside the CTCL window to be claimable.
MIN_CELLS = 25
MIN_SAMPLES = 5
CTCL_DISEASES = ["MF", "SS", "CTCL_other"]
HC_DISEASE = "HC"
STUDY_DOMINANCE = 0.8    # top_study_frac above this -> flagged, carried into every table

# ---------------------------------------------------------------- disease contrast
DISEASE_KEY = "disease_grp"          # built by degradome_helpers: "CTCL" | "HC" | "other"
# The only two skin studies carrying BOTH arms. Everything else is CTCL-only.
PAIRED_STUDIES = ["chennareddy2025", "gaydosik2019"]
DESEQ_DESIGN = "~ study + disease_grp"
DESEQ_CONTRAST = ("disease_grp", "CTCL", "HC")
MIN_DONORS_PER_ARM = 4

# Measured from atlas_obs_full.parquet |><| skin_myeloid_fibro_subtypes.csv at MIN_CELLS=25.
# Asserted in nb42 section 0: if these move, the notebook's power statement is stale.
DISEASE_TESTABLE = ["cDC", "F_reticular", "F_papillary", "F_mesenchymal", "Mac_FOLR2", "F_apCAF"]

# ---------------------------------------------------------------- epidermis vs dermis
# Paired WITHIN donor (`~ donor + skin_layer`): the two layers come off the same biopsy, so a
# donor-blocked design removes exactly the between-patient variance that a pooled fit would
# call signal. Only buus2025 and li2024 separate layers at all -- 78,349 of the CTCL TME cells
# are skin_layer == "whole" and contribute nothing here, which is a coverage fact worth
# printing rather than hiding.
#
# THE BINDING CONSTRAINT IS PAIRING, NOT CELL COUNT, and the first version of this block got
# that wrong. It claimed "donors with BOTH layers >= MIN_CELLS: cDC 10, DC_LAMP3 9, Mac_infl 6",
# which was counted per (donor, layer) unit rather than per donor carrying both layers; at
# LAYER_MIN_PAIRED_DONORS = 4 no level cleared it and the contrast silently returned an empty
# frame for every level. Measured properly:
#
#   * buus2025 has 3 donors (PT09, PT22, PT23) with genuine paired epidermis and dermis preps
#     off the same biopsy -- thousands of cells on both sides.
#   * li2024's `skin_layer` is a property of the SAMPLE, and each li2024 donor contributed one
#     layer: CTCL1 is 21,055 epidermal against 53 dermal cells, CTCL2/4/8 the mirror image
#     (114 / 128 / 260 epidermal). Their "pairs" are prep bleed-through, not a second arm, and
#     none of them clears a per-level gate on the minor layer. No study restriction is applied
#     -- the pairing gate excludes them on its own, which is stronger than naming them.
#
# So the ceiling on this contrast is 3 paired donors, and LAYER_MIN_CELLS is relaxed to 10
# because at n=3 the pairing is what limits it, not the depth of any one pseudobulk. Both
# numbers are reported next to every layer result; the per-level pairing is derived at run time
# by `degradome_helpers.layer_pairing` rather than frozen as a literal, precisely because the
# frozen literal was the thing that was wrong. CD4_malignant pairs in only 2 donors (PT22 has
# no malignant compartment at the gate) and is an abstention here, not a null.
LAYER_KEY = "skin_layer"
LAYER_CONTRAST = ("skin_layer", "epidermis", "dermis")
LAYER_DESIGN = "~ donor + skin_layer"
LAYER_MIN_PAIRED_DONORS = 3
LAYER_MIN_CELLS = 10

# ---------------------------------------------------------------- advanced vs early
# `early` exists in ONE study. Donors by stage: li2024 19 advanced / 17 early; brunner2024,
# buus2025, gaydosik2019, herrera2021, rindler2021_fi are advanced-only; chennareddy2025 is
# 7 advanced / 11 unknown. So the contrast is run inside li2024 and nowhere else, which makes
# it free of study confounding by construction rather than by adjustment.
STAGE_KEY = "stage_class"
STAGE_CONTRAST = ("stage_class", "advanced", "early")
STAGE_DESIGN = "~ stage_class"
STAGE_STUDY = "li2024"
STAGE_MIN_DONORS_PER_ARM = 4
DISEASE_UNDEFINED = ["DC_LAMP3", "Mac_infl", "Mono_moDC", "Mac_SPP1_TREM2",
                     "LC", "pDC", "F_inflammatory"]
HC_DONORS_AT_GATE = {"cDC": 8, "F_reticular": 8, "F_papillary": 8, "F_mesenchymal": 7,
                     "Mac_FOLR2": 5, "F_apCAF": 4, "DC_LAMP3": 0, "Mac_infl": 0,
                     "Mono_moDC": 0, "Mac_SPP1_TREM2": 0, "LC": 0, "pDC": 0,
                     "F_inflammatory": 0}

# ---------------------------------------------------------------- the clonal axis
# nb31 §A produces `nested_subclone` = transcriptomic MAJOR (Leiden on the MRVI mu embedding)
# subdivided by CNV MINOR, over the `tcr_malignant_alice` compartment. 92 subclones across 25
# donors; 75,972 of this object's 76,349 CD4_malignant cells carry a label and the 377 that do
# not are cells nb31's arm-CNV matrix could not cover -- reported, never imputed.
#
# This is the axis nb31 scored its `protease` program on, so it is the axis on which the
# tumour-intrinsic claim has to be re-tested. Two analyses are run here and both mirror what
# the reference notebooks did, on the full 112-gene panel rather than on 11 genes:
#
#   * the per-subclone dot plot nb31 computed and never consumed (`panels` at nb31 cell 11);
#   * the within-donor Wilcoxon over the panel that `old/23_subclonal_evolution.ipynb` cell 17
#     ran once, whose sole up-call was TIMP1 -- the inhibitor.
#
# The test is WITHIN donor (each subclone against the rest of that patient's malignant
# compartment) because there are no biological replicates inside one patient; pseudobulk DESeq2
# across subclones of one donor would be pseudo-replication. That is the same call nb31 and
# old/23 made, and `subclone_helpers.subclone_markers_wilcoxon` is reused verbatim so the
# numbers are comparable to theirs.
SUBCLONE_PARQUET = ATLAS_JOINT / "subclones_v3.parquet"
SUBCLONE_COL = "nested_subclone"
CLONE_MIN_CELLS = 25             # subclone must clear this to enter the panel (89 of 92 do)
CLONE_MIN_SUBCLONES = 2          # donors with fewer are not testable on this axis
CLONE_LFC = 1.0                  # |log2FC| for an up/down call, matching the contrasts above

# ---------------------------------------------------------------- the panel
# 120 degradome symbols, every one verified present in joint_annotated.h5ad var (40,821
# symbols, unique -- no duplicate collisions on this panel). Grouped by catalytic class where
# that is what drives the biology, by family where it is not. With CONTEXT_GENES the built
# object carries 156 columns.
DEGRADOME_FAMILIES = {
    # Matrix metalloproteinases. The ECM-degradation core of the nb31 panel.
    "mmp": ["MMP1", "MMP2", "MMP3", "MMP7", "MMP8", "MMP9", "MMP10", "MMP11", "MMP12",
            "MMP13", "MMP14", "MMP15", "MMP16", "MMP17", "MMP19", "MMP23B", "MMP24",
            "MMP25", "MMP28"],
    # Cysteine cathepsins + legumain. nb31 carried exactly one of these (CTSC).
    "cathepsin": ["CTSA", "CTSB", "CTSC", "CTSD", "CTSE", "CTSF", "CTSG", "CTSH", "CTSK",
                  "CTSL", "CTSO", "CTSS", "CTSV", "CTSW", "CTSZ", "LGMN"],
    # Lymphocyte granule serine proteases. nb31 carried GZMB alone, which is also the gene
    # most exposed to ambient bleed in this atlas (see docs/TCR_costim_gene_lists...).
    "granzyme": ["GZMA", "GZMB", "GZMH", "GZMK", "GZMM", "PRF1"],
    # Other serine proteases: plasminogen activation, neutrophil granule, epidermal kallikrein
    # cascade, type-II transmembrane, HtrA, proprotein convertases, and the two fibroblast
    # surface peptidases (FAP, DPP4).
    "serine_other": ["PLAU", "PLAUR", "PLAT", "ELANE", "PRTN3",
                     "KLK5", "KLK6", "KLK7", "KLK8", "KLK9", "KLK10", "KLK11", "KLK12",
                     "KLK13", "KLK14",
                     "TMPRSS2", "TMPRSS4", "TMPRSS11D", "TMPRSS11E", "TMPRSS13", "ST14",
                     "PRSS8", "PRSS21", "PRSS22", "PRSS23",
                     "HTRA1", "HTRA3", "FURIN", "PCSK5", "PCSK6",
                     "FAP", "DPP4"],
    # ADAM / ADAMTS sheddases and procollagen peptidases + three ectopeptidases.
    "metallo_other": ["ADAM8", "ADAM9", "ADAM10", "ADAM12", "ADAM15", "ADAM17", "ADAM19",
                      "ADAM28", "ADAMDEC1",
                      "ADAMTS1", "ADAMTS2", "ADAMTS4", "ADAMTS5", "ADAMTS9", "ADAMTS12",
                      "ADAMTS14",
                      "ANPEP", "MME", "ECE1"],
    # Mast-cell granule proteases. CONTEXT ONLY -- these exist in the panel so that the
    # attribution has a known right answer to recover (verification check 4).
    "mast": ["CMA1", "CPA3", "TPSAB1", "TPSB2"],
    # Endogenous inhibitors. The denominator of the balance readout; TIMP1 is the one gene
    # old/23 ever called up in malignant subclones.
    "inhibitor": ["TIMP1", "TIMP2", "TIMP3", "TIMP4",
                  "CST3", "CST6", "CST7", "CSTA", "CSTB",
                  "SERPINA1", "SERPINA3", "SERPINB1", "SERPINB2", "SERPINB6", "SERPINB8",
                  "SERPINB9", "SERPINE1", "SERPINE2", "SERPING1",
                  "SPINK5", "A2M", "SLPI", "PI3"],
    # Inflammatory caspase; kept apart from the cathepsins because it is not a degradome
    # enzyme, it is the reason IL1B leaves Mac_infl.
    "caspase": ["CASP1"],
}

# Families that carry claims. `mast` is a control, `caspase` is context for Mac_infl.
CLAIM_FAMILIES = ["mmp", "cathepsin", "granzyme", "serine_other", "metallo_other", "inhibitor"]

# Protease : inhibitor balance pairs (numerator family, denominator gene list).
BALANCE_PAIRS = {
    "mmp_timp": ("mmp", ["TIMP1", "TIMP2", "TIMP3", "TIMP4"]),
    "cathepsin_cystatin": ("cathepsin", ["CST3", "CST6", "CST7", "CSTA", "CSTB"]),
    "serine_serpin": ("serine_other", ["SERPINA1", "SERPINA3", "SERPINB1", "SERPINB2",
                                       "SERPINB6", "SERPINB8", "SERPINB9", "SERPINE1",
                                       "SERPINE2", "SERPING1", "SLPI", "PI3", "SPINK5"]),
}

# Identity markers. Not analysed -- they ride along so the built object can prove it is what
# the build claims (a 114-gene object cannot be sanity-checked on a UMAP).
CONTEXT_GENES = [
    "PTPRC", "CD3E", "CD4", "CD8A", "FOXP3", "IL2RA", "MS4A1", "JCHAIN",
    "KRT14", "KRT10", "COL1A1", "COL1A2", "PDGFRB", "PECAM1", "LYVE1", "MLANA", "PMEL",
    "LYZ", "CD68", "C1QA", "FOLR2", "SPP1", "TREM2", "LAMP3", "CCR7", "CD207", "CD1A",
    "CLEC9A", "CD1C", "LILRA4", "IL3RA", "FCN1", "S100A8", "VCAN", "IL1B", "MKI67",
]

# The 12 genes nb31 actually scored, verbatim from subclone_helpers.PAPER_PANELS["protease"].
# Duplicated here rather than imported so this module has no dependency on subclone_helpers;
# degradome_helpers.assert_nb31_panel() checks the two are still identical.
NB31_PANEL = ["MMP14", "MMP25", "MMP9", "MMP2", "PLAU", "PLAUR", "GZMB",
              "ADAM17", "ADAM10", "ADAMTS4", "CTSC", "TIMP1"]


def panel_genes(include_context: bool = True) -> list:
    """Every symbol the built object must carry, deduped, family order preserved."""
    out, seen = [], set()
    for genes in DEGRADOME_FAMILIES.values():
        for g in genes:
            if g not in seen:
                out.append(g)
                seen.add(g)
    if include_context:
        for g in CONTEXT_GENES:
            if g not in seen:
                out.append(g)
                seen.add(g)
    return out


GENE_TO_FAMILY = {g: fam for fam, genes in DEGRADOME_FAMILIES.items() for g in genes}

# ---------------------------------------------------------------- attribution
# A level owns a gene when it is the top producer in at least this fraction of the donors that
# carry both the level and the gene. 2/3 is the same bar nb40 used for donor reproducibility.
OWNER_DONOR_FRAC = 2 / 3
MIN_DONORS_FOR_OWNERSHIP = 5
# Genes detected below this fraction in every level are dropped from the attribution and the
# drop is logged -- a share computed from three counts is noise, not attribution.
MIN_DETECTION = 0.01
N_SHUFFLE = 20          # label-shuffle rounds for the ownership null ceiling
SEED = 0

# Attribution has a known right answer for these genes. If it does not recover them the method
# is broken and nothing downstream is worth reading, so nb42 section 2 hard-fails on a miss.
#
# THE CONTROLS RUN ON THEIR OWN CUBE, and it is not the analysis cube. Two reasons, both of
# which are properties of the roster rather than of the arithmetic:
#
#   * granularity. The analysis splits the TME 13 ways while Keratinocyte (88,692), Vascular
#     (44,700) and CD4_unassessed (190,142) stay pooled. `share_raw` is abundance-weighted by
#     design, so no single fibroblast subtype can out-produce an unsplit keratinocyte level
#     even when fibroblasts collectively dominate the gene. Controls therefore run at LINEAGE
#     granularity (the 13 subtypes collapsed back to Myeloid / Fibroblast), where "COL1A1 comes
#     from fibroblasts" is a statement the roster can actually express.
#   * coverage. Mast is 1,149 skin cells: median 6 per donor, eligible in 13 of 62 donors.
#     At MIN_CELLS=25 the tryptases are unattributable in this atlas -- which is a true fact
#     about the data, not a failure of the method. Controls use CONTROL_MIN_CELLS so the
#     arithmetic is tested in the regime where the answer is knowable.
#
# Values are the lineage-level owner the attribution must recover.
CONTROL_MIN_CELLS = 5

# Tier 1 -- OWNERSHIP. The producer must clear the full OWNER_DONOR_FRAC bar. These are the
# genes with one dominant source and enough coverage to prove it, so anything less than
# ownership means the arithmetic is wrong.
POSITIVE_CONTROLS_RAW = {          # abundance-weighted output: who supplies the tissue
    "KLK5": {"Keratinocyte"}, "KLK7": {"Keratinocyte"}, "SPINK5": {"Keratinocyte"},
    "COL1A1": {"Fibroblast"}, "COL1A2": {"Fibroblast"},
    "C1QA": {"Myeloid"}, "CTSS": {"Myeloid"},
    "CTSK": {"Myeloid", "Fibroblast"},
}

POSITIVE_CONTROLS_CPM = {          # per-transcriptome intensity: who is built to make it
    "GZMA": {"CD8"}, "GZMK": {"CD8"}, "GZMH": {"CD8"},
    "CD207": {"Myeloid"}, "MLANA": {"Melanocyte"},
}

# Tier 2 -- PLURALITY. The producer must be the TOP level, but is not required to own the gene.
# Two distinct reasons a known-source gene cannot reach the ownership bar, neither of which is
# a defect in the method:
#
#   * coverage. Mast is 1,149 skin cells; even at CONTROL_MIN_CELLS it is eligible in ~36 of 82
#     donors, and donors where mast is top-but-ineligible are abstentions that count against
#     its win fraction. It reaches 0.36-0.64, never 0.67. The tryptases are genuinely
#     unattributable at donor resolution in this atlas.
#   * real multi-source biology. Mast granule proteases are made by mast cells and essentially
#     nothing else, so the plurality is still a meaningful check on them.
#
# GZMB IS DELIBERATELY NOT A CONTROL. It was tried as "intensity -> CD8" and the measured top
# level is Plasma (0.52 of donors), which on inspection is correct -- plasma cells and pDC
# express GZMB, and per unit of transcriptome they beat CD8 in skin. A control whose right
# answer has to be looked up after seeing the result is not a control, so it was removed rather
# than widened to fit. This matters downstream: GZMB is in NB31_PANEL, and its apparent
# malignant-CD4 share sits alongside Plasma, CD8, pDC and mast contributions. Treat any GZMB
# number in this notebook as multi-source, and read it next to the spillover flag.
POSITIVE_CONTROLS_PLURALITY_RAW = {
    "TPSAB1": {"Mast"}, "TPSB2": {"Mast"}, "CPA3": {"Mast"},
}
POSITIVE_CONTROLS_PLURALITY_CPM = {
    "CMA1": {"Mast"}, "TPSAB1": {"Mast"},
}

# ---------------------------------------------------------------- forbidden
# Same discipline as ccc_data.FORBIDDEN_CONTRASTS. These are not oversights.
FORBIDDEN_CONTRASTS = {
    "stage_pooled": "early-vs-advanced pooled across studies is forbidden -- `early` exists "
                    "only inside li2024, so a cross-study fit compares li2024 to everyone "
                    "else. The within-li2024 contrast (STAGE_STUDY) is legitimate and is run.",
    "layer_pooled": "epidermis-vs-dermis unpaired is forbidden: only buus2025 and li2024 "
                    "separate layers, and the two layers of one biopsy are the same patient. "
                    "Run donor-blocked (LAYER_DESIGN) or not at all.",
    "hc_pooled_headline": "5 of 7 skin studies are CTCL-only; pooled CTCL-vs-HC confounds "
                          "disease with study. Run within PAIRED_STUDIES; pooled is a "
                          "labelled sensitivity arm.",
    "seven_undefined_levels": "DC_LAMP3, Mac_infl, Mono_moDC, Mac_SPP1_TREM2, LC, pDC and "
                              "F_inflammatory have zero HC donors at MIN_CELLS. Absence of a "
                              "comparator is not a null result and must not be reported as a "
                              "fold change.",
    "context_level_claims": "Keratinocyte, CD8, Mast, Vascular, Plasma, Melanocyte, Tregs and "
                            "the non-malignant CD4 levels are in the denominator only. They "
                            "make the shares interpretable; they are not the subject.",
}
