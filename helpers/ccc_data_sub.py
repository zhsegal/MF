"""Config for the sub-type CCC re-run (notebooks 38-39).

`ccc_data.py` is the v1 config: 13 pooled levels, of which `Myeloid` (93,435 cells) and
`Fibroblast` (66,760) are single levels. That is the weakest part of the v1 ligand-receptor
map -- "malignant CD4 -> Myeloid" averages Langerhans cells, cDC1/cDC2, LAMP3+ migratory DC,
pDC, monocytes and three functionally opposite macrophage programmes into one sender, and
"-> Fibroblast" averages papillary, reticular, inflammatory and myofibroblastic dermal
fibroblasts.

This module re-uses everything in ccc_data and overrides only what changes:

  * the grouping column becomes `ccc_celltype_sub`, built by
    `ccc_helpers.build_ccc_celltype_sub` from `cell_type_final` + `mal_tcr_alice` + the nb10c
    sidecar `skin_myeloid_fibro_subtypes.csv`;
  * the roster is narrowed to CD4_malignant, CD4_reactive, CD8, B, Keratinocyte and the new
    myeloid / fibroblast sub-levels.

Two things this narrowing is NOT.

  It is not a claim that Vascular, Tregs, Plasma, Mast and Melanocyte are uninteresting: they
  are computed in nb35/36 and those tables stand. Dropping them here bounds the multiple-testing
  surface so the sub-level axes -- which are 8-13 levels where v1 had 2 -- do not arrive with a
  larger grid than v1 had in total.

  It is not a re-analysis of the malignancy call. `CD4_unassessed` is out of the roster because
  nothing in this phase reads it (it was audited, never claimed, in v1); the malignant/reactive
  split is byte-identical to v1's, which is what makes the B-cell axis a usable regression test
  -- B is untouched, so CXCL13-CXCR5 / CD40LG-CD40 / CD86-CD28 must reproduce.

The CCC object is NOT rebuilt. `data/ccc/ccc_skin.h5ad` (727,156 x 1,832) already contains
every myeloid and fibroblast cell; only the grouping column and the roster change, so nb38
attaches the new label by a cell_id join and subsets in memory. The one stale by-product is
`ccc_skin_pseudobulk_full.parquet`, which is keyed by (ccc_celltype x sample) and feeds only the
deferred donor-level differential phase -- re-run jobs/run_ccc_build.py if and when phase 3 starts.
"""

from ccc_data import *  # noqa: F401,F403 -- paths, LR_PANEL, LR_CAVEATS, run params, contrasts
import ccc_data as _v1

# ---------------------------------------------------------------- inputs
# nb10c: cell_id, lineage, leiden_sub, subtype_fine, subtype_ccc -- one row per myeloid or
# fibroblast skin cell (93,435 + 66,760 = 160,195).
SUBTYPE_CSV = ATLAS_JOINT / "skin_myeloid_fibro_subtypes.csv"  # noqa: F405
SUBTYPE_PROV = ATLAS_JOINT / "skin_myeloid_fibro_provenance.json"  # noqa: F405

# ---------------------------------------------------------------- keys
GROUPBY = "ccc_celltype_sub"
SUBTYPE_SRC = "subtype_ccc"     # the coverage-driven collapse; subtype_fine is report-only
SPLIT_LINEAGES = ["Myeloid", "Fibroblast"]

# ---------------------------------------------------------------- outputs
TAB_DIR = _v1.TAB_DIR                       # shared with v1; every file carries the prefix
TAB_PREFIX = "ccc_sub_"
FIG_DIR = _v1.NB_DIR / "figures" / "ccc_sub"
FINAL_FIG_DIR = _v1.FINAL_FIG_DIR           # shared; SVGs are named ccc_sub_*
FIG_PREFIX = "ccc_sub_"


def tab(name: str):
    """tables/ccc_sub_<name>.csv -- keeps the v1 tables addressable alongside these."""
    return TAB_DIR / f"{TAB_PREFIX}{name}.csv"


def fig(name: str, ext: str = "png"):
    return FIG_DIR / f"{FIG_PREFIX}{name}.{ext}"


# ---------------------------------------------------------------- vocabulary
# THESE TWO LISTS ARE AUTHORED BY nb10c. Its section 7 prints them; paste them here. nb38
# section 0 asserts the sidecar's actual levels equal these, so a mismatch is caught before a
# single liana run rather than showing up as an empty dot plot.
#
# The values below are the expected landing zone of the nb10c collapse (fine states that clear
# MIN_CELLS in fewer than MIN_SAMPLES donors get merged into their parent there: cDC1 + cDC2 ->
# cDC, Mono + moDC -> Mono_moDC, Mac_ISG -> Mac_infl, F_myofibroblast -> F_mesenchymal).
MYELOID_LEVELS = [
    "LC",                # Langerhans: CD207/CD1A/EPCAM
    "cDC",               # cDC1 + cDC2
    "DC_LAMP3",          # LAMP3/CCR7/FSCN1 migratory (mreg) DC
    "pDC",               # LILRA4/IL3RA/TCF4
    "Mono_moDC",         # S100A8/9/FCN1/VCAN + CD14 moDC
    "Mac_FOLR2",         # resident macrophage, the M2-like pole
    "Mac_SPP1_TREM2",    # TAM / lipid-associated macrophage
    "Mac_infl",          # IL1B/CXCL9-11, the M1-like pole (+ Mac_ISG)
]
FIBRO_LEVELS = [
    "F_papillary",       # secretory-papillary: APCDD1/WIF1/COL6A5
    "F_reticular",       # secretory-reticular: WISP2/MFAP5/PI16
    "F_mesenchymal",     # mesenchymal / myoCAF: POSTN/COMP/CTHRC1 (+ F_myofibroblast)
    "F_inflammatory",    # pro-inflammatory / iCAF: CCL19/CXCL12/IL6/MMP1
    "F_apCAF",           # MHC-II-high fibroblast
]
LYMPHOID_LEVELS = ["CD8", "B"]
STRUCTURAL_LEVELS = ["Keratinocyte"]

KEEP_LEVELS = [
    CD4_MALIGNANT,  # noqa: F405
    CD4_REACTIVE,  # noqa: F405
    *LYMPHOID_LEVELS,
    *STRUCTURAL_LEVELS,
    *MYELOID_LEVELS,
    *FIBRO_LEVELS,
]

# Everything the roster excludes. UNK/nan carry no identity; the rest are a deliberate scope
# narrowing (see the module docstring), not a judgement on the v1 results for them.
DROP_LEVELS = [
    *_v1.DROP_LEVELS,
    CD4_UNASSESSED,  # noqa: F405
    "Tregs", "Plasma", "Vascular", "Mast", "Melanocyte",
]

CT_ORDER = list(KEEP_LEVELS)

CELL_DEFINITIONS = {
    CD4_MALIGNANT: _v1.CELL_DEFINITIONS[CD4_MALIGNANT],  # noqa: F405
    CD4_REACTIVE: _v1.CELL_DEFINITIONS[CD4_REACTIVE],  # noqa: F405
    "CD8": _v1.CELL_DEFINITIONS["CD8"],
    "B": _v1.CELL_DEFINITIONS["B"],
    "Keratinocyte": _v1.CELL_DEFINITIONS["Keratinocyte"],
    "LC": (
        "Langerhans cells (CD207/CD1A/EPCAM/CLDN1). Epidermal, and the antigen-presenting "
        "partner an epidermotropic malignant CD4 actually meets -- which is why they are their "
        "own level rather than part of a pooled DC bucket."
    ),
    "cDC": (
        "cDC1 (CLEC9A/XCR1/BATF3) and cDC2 (CD1C/CLEC10A/FCER1A) pooled. Kept together because "
        "cDC1 is ~1-2% of skin myeloid cells and does not clear MIN_CELLS in enough donors "
        "alone; the split is reported in subtype_fine."
    ),
    "DC_LAMP3": (
        "LAMP3+/CCR7+/FSCN1+ migratory or mreg DC. Carries CCL17/CCL22, i.e. the CCR4 axis that "
        "mogamulizumab targets -- in v1 that edge was attributed to 'Myeloid' as a whole."
    ),
    "pDC": "Plasmacytoid DC (LILRA4/IL3RA/TCF4). Rare; expect report-only.",
    "Mono_moDC": (
        "Classical monocytes (S100A8/9, FCN1, VCAN) and monocyte-derived DC (CD14 + CD1C/CD209) "
        "pooled -- the boundary between them is a gradient in tissue and does not survive the "
        "donor gate as two levels."
    ),
    "Mac_FOLR2": (
        "Resident macrophages: FOLR2, LYVE1, SELENOP, C1QA-C, MRC1, CD163, F13A1. The M2-like "
        "pole of the polarization axis, but named for its programme -- M1/M2 is an in-vitro "
        "axis that does not partition tissue macrophages (nb10c section 5 scores it)."
    ),
    "Mac_SPP1_TREM2": (
        "SPP1+/TREM2+ lipid-associated / tumour-associated macrophages (APOC1, GPNMB, ACP5). "
        "The state most consistently tumour-restricted across cancers, and the expected CSF1R "
        "receiver on the malignant-CD4 -> myeloid axis."
    ),
    "Mac_infl": (
        "Inflammatory macrophages: IL1B, CXCL1/2, CXCL9/10/11, TREM1, CD300E (+ the ISG-high "
        "cluster merged in). The M1-like pole."
    ),
    "F_papillary": (
        "Secretory-papillary fibroblasts: APCDD1, WIF1, ID1, COL18A1, PTGDS, COL6A5, COL23A1. "
        "Upper dermis -- the compartment an epidermotropic infiltrate passes through."
    ),
    "F_reticular": (
        "Secretory-reticular fibroblasts: WISP2, SLPI, MFAP5, TSPAN8, MGP, PI16, CD36."
    ),
    "F_mesenchymal": (
        "Mesenchymal fibroblasts carrying the myoCAF programme: ASPN, POSTN, COMP, COL11A1, "
        "CTHRC1, LRRC15, FAP (+ the contractile ACTA2/TAGLN cluster merged in)."
    ),
    "F_inflammatory": (
        "Pro-inflammatory fibroblasts / iCAF: CCL19, APOE, CXCL2/3, CXCL12, IL6, MMP1, MMP3, "
        "CXCL8, IL24. The expected CXCL12 source -- the CXCL12/CXCR4 axis reported to drive MF "
        "cell migration and doxorubicin resistance."
    ),
    "F_apCAF": (
        "MHC-II-high fibroblasts (HLA-DRA/HLA-DQA1/CD74). li2024 reports MHC-II+ fibroblasts "
        "maintaining the TH2-like tumour cells, so the level exists to test that directly."
    ),
}

# ---------------------------------------------------------------- focal axes
# Grouped so each liana run stays a bounded grid: 1 sender x 5-8 receivers, both directions.
FOCAL_AXES = {
    "malig_vs_myeloid": ([CD4_MALIGNANT], MYELOID_LEVELS),  # noqa: F405
    "malig_vs_fibro": ([CD4_MALIGNANT], FIBRO_LEVELS),  # noqa: F405
    "malig_vs_lymphoid": ([CD4_MALIGNANT], LYMPHOID_LEVELS),  # noqa: F405
    "malig_vs_structural": ([CD4_MALIGNANT], STRUCTURAL_LEVELS),  # noqa: F405
    "malig_vs_reactive": ([CD4_MALIGNANT], [CD4_REACTIVE]),  # noqa: F405
    "reactive_vs_myeloid": ([CD4_REACTIVE], MYELOID_LEVELS),  # noqa: F405
    "reactive_vs_fibro": ([CD4_REACTIVE], FIBRO_LEVELS),  # noqa: F405
    "reactive_vs_lymphoid": ([CD4_REACTIVE], LYMPHOID_LEVELS),  # noqa: F405
    "reactive_vs_structural": ([CD4_REACTIVE], STRUCTURAL_LEVELS),  # noqa: F405
    # Negative control: the only axis computable in healthy skin, now at sub-level resolution.
    "structural_hc": (FIBRO_LEVELS, STRUCTURAL_LEVELS + MYELOID_LEVELS),
}

ANALYSES = {
    "sub_myeloid": {
        "axis": "malig_vs_myeloid",
        "disease": CTCL_DISEASES,  # noqa: F405
        "note": (
            "The primary run. Malignant CD4 <-> each myeloid state. The question is whether the "
            "v1 'malignant CD4 <-> Myeloid' edges resolve onto one state or are shared: "
            "CCL17/CCL22 -> CCR4 is expected on cDC / DC_LAMP3, CSF1 -> CSF1R on "
            "Mac_SPP1_TREM2. An edge that stays uniform across all 8 levels is evidence the "
            "pooled level was not hiding anything -- report it as such."
        ),
    },
    "sub_fibro": {
        "axis": "malig_vs_fibro",
        "disease": CTCL_DISEASES,  # noqa: F405
        "note": (
            "Malignant CD4 <-> each fibroblast state. CXCL12 -> CXCR4 is the named hypothesis "
            "and is expected on F_inflammatory; TGFB1 -> TGFBR1_TGFBR2 on F_mesenchymal. "
            "F_apCAF exists to test li2024's MHC-II+ fibroblast claim directly."
        ),
    },
    "sub_lymphoid": {
        "axis": "malig_vs_lymphoid",
        "disease": CTCL_DISEASES,  # noqa: F405
        "note": (
            "CD8 and B, both unchanged from v1. This is the REGRESSION TEST: CXCL13 -> CXCR5, "
            "CD40LG -> CD40 and CD86 -> CD28 must reproduce their v1 ranks, because neither the "
            "sender nor these receivers changed. If they move, the change came from the "
            "subsampling or the roster, not from biology."
        ),
    },
    "sub_structural": {
        "axis": "malig_vs_structural",
        "disease": CTCL_DISEASES,  # noqa: F405
        "note": "Keratinocytes, pooled as in v1. Included for the epidermotropism axis.",
    },
    "sub_reactive": {
        "axis": "malig_vs_reactive",
        "disease": CTCL_DISEASES,  # noqa: F405
        "note": (
            "Malignant CD4 <-> reactive CD4. Sender and receiver share a lineage, so shared-gene "
            "and autocrine artifacts are maximal: read only pairs whose ligand is asymmetric "
            "between the two levels."
        ),
    },
    "sub_comparator_myeloid": {
        "axis": "reactive_vs_myeloid",
        "disease": CTCL_DISEASES,  # noqa: F405
        "note": (
            "THE comparator for the myeloid axis. No claim of the form 'malignant CD4 signals X "
            "to myeloid state Y' is reportable unless the same pair is absent or weaker with "
            "reactive CD4 as the sender, in the same tissue and against the same state. Read "
            "with the rank-delta table, never on its own."
        ),
    },
    "sub_comparator_fibro": {
        "axis": "reactive_vs_fibro",
        "disease": CTCL_DISEASES,  # noqa: F405
        "note": "Comparator for the fibroblast axis; same reading rule.",
    },
    "sub_comparator_lymphoid": {
        "axis": "reactive_vs_lymphoid",
        "disease": CTCL_DISEASES,  # noqa: F405
        "note": "Comparator for CD8 / B; also the reactive half of the v1 regression test.",
    },
    "sub_comparator_structural": {
        "axis": "reactive_vs_structural",
        "disease": CTCL_DISEASES,  # noqa: F405
        "note": "Comparator for keratinocytes.",
    },
    "hc_structural": {
        "axis": "structural_hc",
        "disease": ["HC"],
        "note": (
            "Negative control. HC skin has 27,534 cells but only 1,524 CD4 and 300 CD8 across 9 "
            "donors, so no T-cell axis is computable -- only fibroblast sub-level -> "
            "keratinocyte / myeloid. CD4_malignant must be 0 cells here; that is the expected "
            "result. Most sub-levels will fail MIN_CELLS in HC, and that failure is the point: "
            "it says the sub-level resolution is a CTCL-window statement."
        ),
    },
}

# ---------------------------------------------------------------- run parameters
# Levels are smaller and more numerous than v1's, so the v1 cap of 15,000 would leave
# CD4_malignant an order of magnitude larger than pDC or F_apCAF -- and liana's permutation
# p-value shrinks with group size. 8,000 keeps the groups closer without starving the big ones.
SUBSAMPLE_MAX_PER_LEVEL = 8_000
SUBSAMPLE_MAX_PER_DONOR_PER_LEVEL = 1_000

# ---------------------------------------------------------------- controls
# The v1 controls re-pointed at the state each edge is expected to live on. A control landing on
# a DIFFERENT sub-level is not a failure -- it is the result this whole re-run exists to produce
# -- so nb38/39 report where each one actually landed rather than only whether it was found.
POSITIVE_CONTROLS = [
    (CD4_MALIGNANT, "B", "CXCL13", "CXCR5"),  # noqa: F405
    (CD4_MALIGNANT, "B", "CD40LG", "CD40"),  # noqa: F405
    ("B", CD4_MALIGNANT, "CD86", "CD28"),  # noqa: F405
    ("B", CD4_MALIGNANT, "TNF", "TNFRSF1B"),  # noqa: F405
    ("cDC", CD4_MALIGNANT, "CD86", "CD28"),  # noqa: F405
    ("cDC", CD4_MALIGNANT, "CCL17", "CCR4"),  # noqa: F405
    ("DC_LAMP3", CD4_MALIGNANT, "CCL22", "CCR4"),  # noqa: F405
    ("Mac_FOLR2", CD4_MALIGNANT, "CD58", "CD2"),  # noqa: F405
    ("F_inflammatory", CD4_MALIGNANT, "CXCL12", "CXCR4"),  # noqa: F405
    (CD4_MALIGNANT, "Mac_SPP1_TREM2", "CSF1", "CSF1R"),  # noqa: F405
    (CD4_MALIGNANT, "F_mesenchymal", "TGFB1", "TGFBR1_TGFBR2"),  # noqa: F405
]

# Held to a stricter bar: these three are the replication spec's, they involve only levels that
# did NOT change between v1 and this run, and they must reproduce.
SPEC_MUST_HAVES = list(_v1.SPEC_MUST_HAVES)

NEGATIVE_CONTROLS = [
    (CD4_MALIGNANT, "F_reticular", "KITLG", "KIT"),  # noqa: F405
    (CD4_MALIGNANT, "Keratinocyte", "COL1A1", "ITGA2_ITGB1"),  # noqa: F405
    (CD4_MALIGNANT, "Keratinocyte", "KRT1", "*"),  # noqa: F405
]

# Edges whose SUB-LEVEL destination is the point of the re-run. nb38 section 11 reports, for each,
# which level actually carries it and how far ahead of the runner-up -- the concentration is the
# result, in both directions.
RESOLUTION_TESTS = {
    "ccr4_axis": {
        "pairs": [("CCL17", "CCR4"), ("CCL22", "CCR4")],
        "receiver": CD4_MALIGNANT,  # noqa: F405
        "candidates": MYELOID_LEVELS,
        "expected": ["cDC", "DC_LAMP3"],
        "why": "the mogamulizumab axis; in v1 it was attributed to 'Myeloid' as a whole",
    },
    "csf1_axis": {
        "pairs": [("CSF1", "CSF1R")],
        "sender": CD4_MALIGNANT,  # noqa: F405
        "candidates": MYELOID_LEVELS,
        "expected": ["Mac_SPP1_TREM2", "Mac_FOLR2"],
        "why": "malignant CD4 -> macrophage maintenance; the TAM state is the expected receiver",
    },
    "cxcl12_axis": {
        "pairs": [("CXCL12", "CXCR4")],
        "receiver": CD4_MALIGNANT,  # noqa: F405
        "candidates": FIBRO_LEVELS,
        "expected": ["F_inflammatory"],
        "why": "the CAF axis reported to drive MF migration and doxorubicin resistance",
    },
    "mhcii_axis": {
        "pairs": [("HLA-DRA", "CD4")],
        "receiver": CD4_MALIGNANT,  # noqa: F405
        "candidates": FIBRO_LEVELS + MYELOID_LEVELS,
        "expected": ["F_apCAF"],
        "why": (
            "li2024's MHC-II+ fibroblast claim. Read against LR_CAVEATS['HLA-DRA'] -- MHC-II "
            "edges in the consensus resource largely track myeloid identity, so a myeloid win "
            "here is uninformative and only F_apCAF clearing the myeloid levels would be a result"
        ),
    },
    "tgfb_axis": {
        "pairs": [("TGFB1", "TGFBR1_TGFBR2")],
        "sender": CD4_MALIGNANT,  # noqa: F405
        "candidates": FIBRO_LEVELS,
        "expected": ["F_mesenchymal"],
        "why": "fibroblast-to-myoCAF conversion is the canonical TGFB1 read-out",
    },
}

CAVEAT_BLOCK = _v1.CAVEAT_BLOCK + (
    " One caveat is specific to this run: splitting a level into k sub-levels multiplies the "
    "tested grid by k while each sub-level carries fewer cells and fewer donors, so a pair that "
    "was solid on pooled Myeloid can look weaker on every sub-level without any biology having "
    "changed. Read the sub-level result against the v1 pooled result (tables/ccc_liana_*.csv), "
    "not on its own."
)
