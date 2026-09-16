"""Shared constants for the MF / CTCL skin cell-cell-communication analysis.

Every CCC notebook (34-36) imports its cell-type vocabulary, focal axes, context
windows and run parameters from here, so the grouping stays identical across
notebooks and no notebook carries a magic number.

Design decisions, with the numbers that forced them (all from
data/atlas_joint/atlas_obs_full.parquet, compartment == "Skin", 749,510 cells):

  * Expression comes from joint_annotated.h5ad (40,821 gene symbols, raw counts),
    NOT joint_mrvi_input_skin.h5ad -- the 10k-HVG object drops 9 LR genes
    (MIF, PVR, ICOSLG, CD44, LTBR, ITGB1, NOTCH1, IL4R, IL7).
  * Malignancy is ALICE-TCR alone (mal_tcr_alice), NOT mal_combined. mal_combined
    is `alice | inferCNV`, and half its malignant CD4 were cnv_only -- a call
    derived from smoothed regional expression, i.e. partly the same measurement
    liana then scores. ALICE is a TCR-sequence fact, so the call and the scored
    expression are independent. mal_combined/mal_cnv stay as sensitivity arms.
  * Only CD4 is split, because ALICE is a CD4-only caller:
    CD4 -> 76,349 malignant / 72,207 reactive / 190,142 unassessed.
  * "Unassessed" means ALICE could never test the cell: 130,484 outside the nb30
    scored set (100% skin_layer=="whole") plus 59,658 inside it with no TRB CDR3
    recovered. ALICE-negative by absence of data is not evidence of benignity, so
    these are their own level and never pooled into the reactive comparator.
  * CD8 and Tregs stay single levels and carry no malignant call at all under
    ALICE. (mal_combined called 21,200 CD8 and 792 Tregs malignant, every one
    cnv_only with zero TCR support -- a CNV false positive or ambient bleed.)
  * Stage / disease contrasts are NOT in this phase: early-vs-advanced exists only
    inside li2024 (7 vs 7 donors), SS is buus2025 alone, and HC skin has 1,524 CD4
    across 9 donors. See FORBIDDEN_CONTRASTS.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
NB_DIR = Path(__file__).resolve().parent.parent
ATLAS_JOINT = NB_DIR / "data" / "atlas_joint"

# 1,173,694 x 40,821 gene symbols; X is byte-identical to layers['raw_counts']
# (float64 CSR, ~2,125 nnz/cell). NEVER sc.read_h5ad this file -- stream it.
SOURCE_H5AD = ATLAS_JOINT / "joint_annotated.h5ad"
# 1,173,694 x 76; index == cell_id == joint_annotated obs index. The h5ad obs does
# NOT carry cell_type_final / mal_combined / stage_group -- only this parquet does.
OBS_PARQUET = ATLAS_JOINT / "atlas_obs_full.parquet"

CCC_DIR = NB_DIR / "data" / "ccc"
CCC_ADATA = CCC_DIR / "ccc_skin.h5ad"
PSEUDOBULK_PQ = CCC_DIR / "ccc_skin_pseudobulk_full.parquet"
PSEUDOBULK_META = CCC_DIR / "ccc_skin_pseudobulk_meta.parquet"
TESTSET_H5AD = CCC_DIR / "ccc_equivalence_testset.h5ad"
MANIFEST = CCC_DIR / "ccc_build_manifest.json"

TAB_DIR = NB_DIR / "tables"
FIG_DIR = NB_DIR / "figures" / "ccc"
FINAL_FIG_DIR = NB_DIR / "figures" / "final"

# ---------------------------------------------------------------- keys
COMPARTMENT = "Skin"
SAMPLE_KEY = "sample_id"      # 99 skin samples
DONOR_KEY = "donor"           # 82 skin donors == 82 real_donor; the replicate unit
STUDY_KEY = "study"           # 7 skin studies; the batch axis
GROUPBY = "ccc_celltype"
LAYER = "lognorm"
COUNTS_LAYER = "raw_counts"

CELLTYPE_SRC = "cell_type_final"   # nb10b, 12 skin levels
# The primary call is ALICE-TCR alone (nb30 cell 19: per-donor founder TRB clone
# plus its ALICE-significant <=1-aa CDR3 family, OLGA Pgen model, CD4 only).
# NOT mal_combined -- that is `alice | cnv_malig_cluster`, and inferCNV derives its
# call from smoothed regional expression, i.e. partly the same measurement liana
# then scores as ligand/receptor abundance. mal_combined stays a sensitivity arm.
MALIG_SRC = "mal_tcr_alice"        # nb30 v3
MALIG_ALT = ["mal_combined", "mal_cnv", "mal_tcr_dom"]
EVIDENCE_SRC = "malignant_evidence"
FINE_SRC = "cell_type"             # li2024 50-level; "Unknown" for non-li2024 skin

# Was ALICE able to test this cell at all? A CD4 with no TRB CDR3 is ALICE-negative
# by absence of data, not by evidence, so it is demoted to CD4_unassessed rather
# than pooled into the reactive comparator.
# NOTE the nb30 denominator, not `assessed_tcr`: the latter is the nb10 10x-VDJ
# clone table, where li2024 -- the largest cohort -- contributes 0 cells. nb30
# folded li2024's CDR3s in separately (114,807 cells).
TCR_ASSESSED_SRC = "assessed_tcr_nb30"

# Carried onto the CCC object. Everything the windows, audits and (deferred)
# designs need, and nothing that would make the object look reusable for DE.
OBS_COLS = [
    "cell_id", "donor", "real_donor", "sample_id", "study", "dataset", "compartment",
    "cell_type_final", "cell_type_broad", "cell_type", "cell_type_T2", "lineage",
    "mal_combined", "mal_cnv", "mal_tcr_alice", "mal_tcr_dom", "malignant_evidence",
    "assessed_cnv", "assessed_tcr", "assessed_tcr_nb30",
    "cnv_score", "clone_id", "clone_size", "is_expanded",
    "disease", "disease_stage", "entity", "stage_class", "stage_clean", "stage_group",
    "skin_layer", "tissue", "lesion_type", "treatment_context", "blood_involvement",
    "sex", "tech", "cohort_region",
    "n_genes", "total_counts", "pct_mito", "doublet_score",
]

# ---------------------------------------------------------------- vocabulary
CD4_MALIGNANT = "CD4_malignant"     #  76,349 cells, 29 donors >= MIN_CELLS
CD4_REACTIVE = "CD4_reactive"       #  72,207 cells, 34 donors -- ALICE-negative, CDR3+
CD4_UNASSESSED = "CD4_unassessed"   # 190,142 cells, never testable; own level

# Levels dropped from the CCC object entirely. UNK is 22,354 nb10b leftovers with
# no resolved identity; a CCC edge on an unidentified sender means nothing.
DROP_LEVELS = ["UNK", "nan", "None", "NA"]

TME_CORE = ["CD8", "Myeloid", "Fibroblast"]
TME_EXTENDED = ["Keratinocyte", "Vascular", "B", "Plasma", "Tregs"]
# Reported, never claimed: Mast clears MIN_CELLS alongside malignant CD4 in 4
# donors, Melanocyte in 13 but with no CCC hypothesis attached.
TME_THIN = ["Mast", "Melanocyte"]

CT_ORDER = [
    CD4_MALIGNANT, CD4_REACTIVE, CD4_UNASSESSED, "CD8", "Tregs",
    "B", "Plasma", "Myeloid", "Mast",
    "Keratinocyte", "Fibroblast", "Vascular", "Melanocyte",
]

# Donors (of 82) clearing MIN_CELLS on BOTH sides, against CD4_malignant.
# This is what sets the axis roster and the MIN_SAMPLES gate below.
AXIS_DONORS = {
    CD4_REACTIVE: 29, "Myeloid": 28, "CD8": 27, "Vascular": 26, "Fibroblast": 24,
    "Keratinocyte": 22, "B": 20, "Plasma": 17, "Melanocyte": 13, "Tregs": 10, "Mast": 4,
}

CELL_DEFINITIONS = {
    CD4_MALIGNANT: (
        "CD4 T cells carrying the ALICE tumour clone (nb30 mal_tcr_alice): the "
        "per-donor founder TRB clonotype plus the ALICE-significant <=1-aa CDR3 "
        "family around it. 76,349 cells, 29 donors. The call is a TCR-sequence "
        "fact, independent of the expression liana scores -- which is exactly why "
        "it is the primary and inferCNV is not. 62,096 of these also carry a CNV "
        "call ('both'); the remaining 14,253 are tcr_only."
    ),
    CD4_REACTIVE: (
        "CD4 T cells with a recovered TRB CDR3 that ALICE tested and did NOT "
        "assign to the tumour clone family. 72,207 cells, 34 donors. The "
        "comparator that makes 'malignant-specific' mean something -- but it is "
        "lesional reactive CD4, not normal-skin CD4 (the atlas has no usable "
        "normal-skin T cells). 40,447 of them do carry a cnv_only call; under an "
        "ALICE-primary design that disagreement is deliberate, and §13 sweeps it."
    ),
    CD4_UNASSESSED: (
        "CD4 T cells ALICE could never test. Two mechanisms pooled: 130,484 "
        "outside the nb30 scored set entirely (all skin_layer == 'whole'; li2024 "
        "66,234, chennareddy2025 51,856, gaydosik2019 11,900, brunner2024 494) "
        "and 59,658 inside it with no TRB CDR3 recovered. Both track study, "
        "chemistry and depth, so the level is a protocol artifact. Audited, "
        "never used as a contrast."
    ),
    "CD8": (
        "CD8 T cells, pooled. ALICE is a CD4-only caller, so no CD8 cell carries "
        "a malignant call under the primary definition at all. (mal_combined "
        "called 21,200 of 51,831 malignant, every one of them cnv_only with zero "
        "TCR support -- a CNV/ambient artifact; see drop_cd8_malcall.)"
    ),
    "Tregs": (
        "FOXP3+ regulatory T cells. 1,936 cells, 20 donors -- report only. Like "
        "CD8, they carry no ALICE call (792 mal_combined positives, all cnv_only)."
    ),
    "Myeloid": (
        "Macrophages, monocytes, cDC/moDC, Langerhans cells and pDC pooled. The "
        "li2024 fine taxonomy resolves these but is 'Unknown' for the 329,931 "
        "non-li2024 skin cells, so the pooled level is what all 7 studies share."
    ),
    "Fibroblast": "Dermal fibroblasts (F1/F2/F3 pooled; the split is li2024-only).",
    "Keratinocyte": "Epidermal keratinocytes, basal through differentiated.",
    "Vascular": "Vascular and lymphatic endothelium plus pericytes.",
    "B": "B cells -- the CXCL13/TLS axis partner.",
    "Plasma": "Antibody-secreting plasma cells.",
    "Mast": "Mast cells. 1,149 cells, 4 donors on the malignant axis -- not claimable.",
    "Melanocyte": "Melanocytes. No CCC hypothesis; reported for completeness.",
}

# ---------------------------------------------------------------- focal axes
# axis name -> (senders, receivers). Both directions are built by
# ccc_helpers.build_groupby_pairs and passed to liana as `groupby_pairs`, so the
# N^2 grid is never computed and the multiple-testing surface stays bounded.
FOCAL_AXES = {
    "malig_vs_core": ([CD4_MALIGNANT], TME_CORE),
    "malig_vs_extended": ([CD4_MALIGNANT], TME_EXTENDED),
    "malig_vs_thin": ([CD4_MALIGNANT], TME_THIN),
    "malig_vs_reactive": ([CD4_MALIGNANT], [CD4_REACTIVE]),
    "reactive_vs_core": ([CD4_REACTIVE], TME_CORE),
    "reactive_vs_extended": ([CD4_REACTIVE], TME_EXTENDED),
    # Negative control: the only axis computable in healthy skin.
    "structural_hc": (["Fibroblast"], ["Keratinocyte", "Vascular", "Myeloid"]),
}

# ---------------------------------------------------------------- the analyses
# Each entry is one rank_aggregate run. Phase 1 is deliberately pooled: no stage,
# disease or layer stratification (see FORBIDDEN_CONTRASTS for why).
CTCL_DISEASES = ["MF", "SS", "CTCL_other"]

ANALYSES = {
    "ctcl_all_core": {
        "axis": "malig_vs_core",
        "disease": CTCL_DISEASES,
        "note": (
            "The primary descriptive run: malignant CD4 <-> CD8 / Myeloid / "
            "Fibroblast over all CTCL skin, all 7 studies. Pooling studies is "
            "defensible here because the axis is malignant-CD4 vs TME *within* "
            "the same tissue, not a between-condition comparison. 24-32 donors "
            "clear MIN_CELLS on both sides."
        ),
    },
    "ctcl_all_extended": {
        "axis": "malig_vs_extended",
        "disease": CTCL_DISEASES,
        "note": (
            "Keratinocyte / Vascular / B / Plasma / Tregs. B (21 donors) carries "
            "the CXCL13-CXCR5 TLS hypothesis; Tregs (11 donors) is report-only."
        ),
    },
    "ctcl_all_thin": {
        "axis": "malig_vs_thin",
        "disease": CTCL_DISEASES,
        "note": (
            "Mast (5 donors) and Melanocyte (29). Run so that absence is on the "
            "record; below MIN_SAMPLES, so nothing here enters a headline figure."
        ),
    },
    "ctcl_all_reactive": {
        "axis": "malig_vs_reactive",
        "disease": CTCL_DISEASES,
        "note": (
            "Malignant CD4 <-> reactive CD4, 29 donors. Sender and receiver share "
            "a lineage, so shared-gene and autocrine artifacts are maximal: read "
            "only pairs whose ligand is asymmetric between the two levels."
        ),
    },
    "ctcl_all_comparator": {
        "axis": "reactive_vs_core",
        "disease": CTCL_DISEASES,
        "note": (
            "THE comparator. No claim of the form 'malignant CD4 signals X to Y' "
            "is reportable unless the same pair is absent or weaker with reactive "
            "CD4 as the sender in the same tissue. Paired with the rank-delta "
            "table, not read on its own."
        ),
    },
    "ctcl_all_comparator_extended": {
        "axis": "reactive_vs_extended",
        "disease": CTCL_DISEASES,
        "note": "Comparator for the extended partners; same reading rule.",
    },
    "hc_structural": {
        "axis": "structural_hc",
        "disease": ["HC"],
        "note": (
            "Negative control. HC skin has 27,534 cells but only 1,524 CD4 and "
            "300 CD8 across 9 donors, so no T-cell axis is computable -- only the "
            "structural fibroblast -> keratinocyte/vascular/myeloid grid. "
            "CD4_malignant must be 0 cells here; that is the expected result."
        ),
    },
}

# ---------------------------------------------------------------- run parameters
RESOURCE_NAME = "consensus"   # human; NO ortholog mapping (unlike the Myeloma port)

EXPR_PROP = 0.1
EXPR_PROP_SWEEP = [0.05, 0.1, 0.2]
MIN_CELLS = 25       # liana's default of 5 is far too permissive per donor
MIN_SAMPLES = 5      # donors clearing MIN_CELLS before a level may be *claimed*
N_PERMS = 1000
SEED = 1337
N_JOBS = 8

# Balanced subsampling, for two reasons rather than one:
#   (a) 1000 permutations over ~400k cells is slow interactively;
#   (b) liana's permutation p-value shrinks with group size, so CD4_malignant
#       (76,349 cells) would score p=0 on nearly every pair purely on n.
# The un-subsampled run is the sensitivity check, not the default.
SUBSAMPLE_MAX_PER_LEVEL = 15_000
SUBSAMPLE_MAX_PER_DONOR_PER_LEVEL = 1_500   # keeps li2024 (56% of skin) from dominating
SUBSAMPLE_SEED = SEED
DOWNSAMPLE_SIZES = [5_000, 15_000, 40_000]
N_SHUFFLES = 3
TOP_N = 20

# Pseudobulk filters for the deferred differential phase. The cube is written by
# the build job as a by-product of the same streaming pass, so phase 3 needs no
# further access to the 48 GB source.
PSEUDOBULK_MIN_CELLS = 25
PSEUDOBULK_MIN_COUNTS = 1000
PSEUDOBULK_MIN_SAMPLES = 3

# ---------------------------------------------------------------- contrasts
# Declared here so the deferred phase inherits the reasoning, and so the traps
# are named rather than rediscovered.
CONTRASTS = {
    "malignant_vs_reactive": {
        "design": "~ donor + ccc_celltype",
        "contrast": ["ccc_celltype", CD4_MALIGNANT, CD4_REACTIVE],
        "unit": DONOR_KEY,
        "n_units": 26,
        "note": (
            "The headline contrast when phase 3 runs. Paired within donor, so "
            "donor, study, chemistry and stage all cancel -- the only CCC "
            "contrast in this atlas free of study confounding."
        ),
    },
    "advanced_vs_early_li2024": {
        "design": "~ skin_layer + stage_group",
        "contrast": ["stage_group", "advanced", "early"],
        "unit": DONOR_KEY,
        "studies": ["li2024"],
        "n_units": 14,
        "note": (
            "7 vs 7 donors, li2024 only. Cohort-wide this contrast is impossible: "
            "every non-li2024 donor with malignant CD4 is stage_group == 'other'."
        ),
    },
    "epidermis_vs_dermis": {
        "design": "~ donor + skin_layer",
        "contrast": ["skin_layer", "epidermis", "dermis"],
        "unit": DONOR_KEY,
        "note": "Paired where a donor contributed both layers; epidermotropism read-out.",
    },
}

FORBIDDEN_CONTRASTS = {
    "mf_vs_ss": (
        "SS skin is buus2025 alone (55,483 of 55,483 cells) -- a study contrast "
        "wearing a disease label."
    ),
    "stage_cohort_wide": (
        "stage_group early/advanced is 7 vs 7 donors INSIDE li2024; every other "
        "donor with malignant CD4 is 'other'. Cohort-wide it is a study contrast."
    ),
    "ctcl_vs_hc_on_T_axes": (
        "HC skin has 1,524 CD4 and 300 CD8 across 9 donors. There is no "
        "normal-skin T-cell baseline in this atlas."
    ),
    "assessed_vs_unassessed": (
        "CD4_unassessed pools two protocol artifacts, neither of them biology. "
        "130,484 cells sit outside the nb30 scored set -- 100% skin_layer == "
        "'whole', li2024 66,234 + chennareddy2025 51,856 + gaydosik2019 11,900 + "
        "brunner2024 494. The other 59,658 were scored but yielded no TRB CDR3, "
        "and CDR3 recovery tracks chemistry and sequencing depth. Either way the "
        "contrast reads out the assay, not the tumour."
    ),
}

# Arms recurrently altered in CTCL. A headline ligand or receptor sitting on one
# of these, in a cnv_only-heavy level, may be reporting gene dosage rather than
# signalling -- flag it, do not silently drop it.
CNV_RECURRENT_ARMS = ["17q", "10q", "6q", "8q", "1p", "20q", "13q", "9p"]

# ---------------------------------------------------------------- curated LR panel
# Force-plotted regardless of whether a pair clears expr_prop, so a negative is
# reported rather than silently dropped. Receptor tuples denote heteromeric
# complexes and are joined with "_" to match liana's complex naming.
# Sources: docs/CTCL_atlas_notebook_replication_spec.md ("Cell 9", Fig 5f/6f/8a)
# and docs/TCR_costim_gene_lists_and_interpretation.md.
LR_PANEL = {
    "b_cell_tls_axis": [
        ("CXCL13", "CXCR5"),
        ("CD40LG", "CD40"),
        ("CD28", "CD86"),
        ("CD28", "CD80"),
        ("TNF", "TNFRSF1B"),
        ("TNFSF13B", "TNFRSF13B"),
        ("TNFSF13B", "TNFRSF13C"),
        ("TNFSF13", "TNFRSF13B"),
        ("LTB", "LTBR"),
        ("LTA", "LTBR"),
        ("CXCL12", "CXCR4"),
    ],
    "costim_tme_to_malignant": [
        ("CD80", "CD28"),
        ("CD86", "CD28"),
        ("ICOSLG", "ICOS"),
        ("TNFSF4", "TNFRSF4"),
        ("TNFSF9", "TNFRSF9"),
        ("TNFSF18", "TNFRSF18"),
        ("CD70", "CD27"),
        ("CD58", "CD2"),
        # PVR-CD226 is not in the consensus resource; PVR-CD96 is (stored flipped
        # as CD96 -> PVR). Same nectin-family axis, and it is actually scorable.
        ("PVR", "CD96"),
        ("NECTIN2", "CD226"),
        ("TNFSF14", "TNFRSF14"),
        ("TNFSF8", "TNFRSF8"),      # CD30L -> CD30; transformation / brentuximab target
    ],
    "checkpoint_restraint": [
        ("CD274", "PDCD1"),
        ("PDCD1LG2", "PDCD1"),
        ("PVR", "TIGIT"),
        ("NECTIN2", "TIGIT"),
        ("LGALS9", "HAVCR2"),
        ("CD80", "CTLA4"),
        ("CD86", "CTLA4"),
        ("HLA-E", "KLRC1"),
    ],
    "th2_stroma_remodelling": [
        ("IL13", ("IL13RA1", "IL4R")),
        ("IL4", ("IL4R", "IL2RG")),
        ("IL31", ("IL31RA", "OSMR")),
        ("AREG", "EGFR"),
        ("TGFB1", ("TGFBR1", "TGFBR2")),
        ("IL22", ("IL22RA1", "IL10RB")),
    ],
    "chemokine_recruitment": [
        ("CCL17", "CCR4"),          # the mogamulizumab axis
        ("CCL22", "CCR4"),
        ("CCL19", "CCR7"),
        ("CCL21", "CCR7"),
        ("CXCL9", "CXCR3"),
        ("CXCL10", "CXCR3"),
        ("CCL5", "CCR5"),
        ("CXCL16", "CXCR6"),
        ("CCL27", "CCR10"),
    ],
    "survival_signal3": [
        ("IL15", ("IL15RA", "IL2RB")),
        ("IL7", "IL7R"),
        ("IL2", ("IL2RA", "IL2RB")),
    ],
    "malignant_to_myeloid": [
        ("CSF1", "CSF1R"),
        ("CSF2", ("CSF2RA", "CSF2RB")),
        ("IFNG", ("IFNGR1", "IFNGR2")),
        ("CD40LG", "CD40"),
    ],
    "adhesion_migration": [
        ("ICAM1", ("ITGAL", "ITGB2")),
        ("SELE", "CD44"),        # SELE-SELPLG is absent from the consensus resource
        ("SELP", "SELPLG"),      # stored flipped as SELPLG -> SELP
    ],
}

# Interactions whose ranking must be read sceptically.
LR_CAVEATS = {
    "MIF": (
        "MIF is ubiquitous and very highly expressed; a top-ranked MIF->CD74 or "
        "MIF->CXCR4 edge is a null-model failure until shown otherwise. Treat it "
        "as the positive control for the FAILURE mode, not as biology."
    ),
    "B2M": "Housekeeping-level expression; any B2M edge is an artifact of the null.",
    "HLA-DRA": (
        "MHC-II 'interactions' in the consensus resource largely track myeloid "
        "identity rather than a signalling event."
    ),
    "CD74": "See MIF and HLA-DRA -- CD74 is the receiver side of both failure modes.",
    "IL13": (
        "Th2 cytokine mRNA is poorly captured by 3' 10x, so a negative IL13->IL4R "
        "edge is uninformative. That is exactly why it is in the forced panel."
    ),
    "IL4": "See IL13.",
    "TNFSF8": (
        "CD30L->CD30 marks large-cell transformation; expect it only in advanced "
        "or LCT donors, and read its absence as coverage, not biology."
    ),
    "PVR": (
        "The consensus resource stores the nectin axis with PVR on the RECEPTOR "
        "side (TIGIT -> PVR, CD96 -> PVR), which is backwards biologically: TIGIT "
        "and CD96 are the receptors. ccc_helpers.resolve_pairs flips the direction "
        "to match the resource, so read these edges as 'the PVR-expressing cell is "
        "liana's target' and interpret the biology in the opposite direction."
    ),
    "TIGIT": "See PVR -- direction is stored inverted in the resource.",
}

# The consensus resource does not spell interactions the way people write them:
# complex subunits are sorted alphabetically, some ligands are complexes
# (CSF1 -> CSF1R is stored CSF1_IL34 -> CSF1R), some receptors carry obligate
# extra chains (IL7R -> IL2RG_IL7R, KLRC1 -> KLRC1_KLRD1, TGFBR1_TGFBR2 also
# appears as ACVR1_TGFBR1_TGFBR2), and a few pairs are stored with the receptor
# in the ligand column. ccc_helpers.resolve_pairs / resolve_panel / resolve_controls
# translate the human-readable spellings above onto the resource's, so a real
# interaction is never reported as absent because of a naming convention.
# 47 of the 49 curated pairs resolve; the two that do not are noted inline.

# Pairs that MUST appear, with direction and partner asserted. Drawn from the
# replication spec (Fig 5f/6f/8a) and CTCL literature.
POSITIVE_CONTROLS = [
    (CD4_MALIGNANT, "B", "CXCL13", "CXCR5"),
    (CD4_MALIGNANT, "B", "CD40LG", "CD40"),
    ("B", CD4_MALIGNANT, "CD86", "CD28"),
    ("B", CD4_MALIGNANT, "TNF", "TNFRSF1B"),
    ("Myeloid", CD4_MALIGNANT, "CD86", "CD28"),
    ("Myeloid", CD4_MALIGNANT, "CCL17", "CCR4"),
    ("Myeloid", CD4_MALIGNANT, "CCL22", "CCR4"),
    ("Myeloid", CD4_MALIGNANT, "CD58", "CD2"),
    ("Fibroblast", CD4_MALIGNANT, "CXCL12", "CXCR4"),
    (CD4_MALIGNANT, "Myeloid", "CSF1", "CSF1R"),
    (CD4_MALIGNANT, "Fibroblast", "TGFB1", "TGFBR1_TGFBR2"),
]

# The three the replication spec names explicitly; held to a stricter bar.
SPEC_MUST_HAVES = [
    (CD4_MALIGNANT, "B", "CXCL13", "CXCR5"),
    (CD4_MALIGNANT, "B", "CD40LG", "CD40"),
    ("B", CD4_MALIGNANT, "CD86", "CD28"),
]

# Lineage-impossible edges: a T cell does not send KITLG or COL1A1.
NEGATIVE_CONTROLS = [
    (CD4_MALIGNANT, "Fibroblast", "KITLG", "KIT"),
    (CD4_MALIGNANT, "Keratinocyte", "COL1A1", "ITGA2_ITGB1"),
    (CD4_MALIGNANT, "Keratinocyte", "KRT1", "*"),
]

# Genes whose presence in the resource is checked after selection, so a missing
# control is diagnosed as a resource gap rather than a biological negative.
RESOURCE_COVERAGE_CHECK = [
    "CXCL13", "CXCR5", "CD40LG", "CD40", "CD28", "CD86", "CD80",
    "CCL17", "CCL22", "CCR4", "CXCL12", "CXCR4", "TGFB1",
    "CSF1", "CSF1R", "CD58", "CD2", "TNF", "TNFRSF1B",
    "TNFSF8", "TNFRSF8", "PDCD1", "CD274", "LGALS9", "HAVCR2",
    "MIF", "CD74", "PVR", "ICOSLG", "IL4R", "IL7R", "NOTCH1", "CD44", "LTBR", "ITGB1",
]

# ---------------------------------------------------------------- figures
HEATMAP_CMAP = "viridis"
HEATMAP_BAD = "0.92"          # grey: structurally absent, NOT low signal
PVAL_ALPHA = 0.05
HEATMAP_TOP_N = 15
DOTPLOT_TOP_N = 20
DOTPLOT_SIZE = (7, 7)

CAVEAT_BLOCK = (
    "LIANA permutation p-values are computed with the CELL as the unit and are "
    "pseudoreplicated across donors (76,349 malignant CD4 cells come from 29 "
    "donors). They size the dots and mark the heatmap cells; they are not "
    "evidence. Donor-level inference is the deferred differential phase. "
    "Grey heatmap cells mean the pair failed min_cells in that stratum -- read "
    "them against the coverage table printed beneath each figure, not as biology."
)
