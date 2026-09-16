"""Helpers to join the Li-2024 CTCL skin atlas + standalone GEO cohorts into one
multi-tissue atlas with a harmonized obs schema and a cross-platform TCR clone
table.

Used by ``10_build_joint_atlas.ipynb``. Heavy steps (loading the 419k-cell atlas,
the full concat, scVI/scANVI) run on the GPU kernel — this module only defines
the loaders / harmonizers / clone logic.

Conventions mirror ``07_d1_semantic_geom_cd4.ipynb``: gene-symbol ``var_names``,
raw counts kept in ``layers['raw_counts']``, sample metadata in ``meta/*.tsv``.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Canonical obs schema
# ----------------------------------------------------------------------------
# Every cell in the joint atlas carries exactly these columns. cell_type is the
# atlas 49-type taxonomy (transferred to cohorts by scANVI in step 4).
CANON_OBS = [
    "cell_id",          # <dataset>|<sample_id>|<barcode>   (globally unique)
    "dataset",          # Li2024_atlas | D1 | D3 | H | D5 | D6 | B1 | B2 | B4
    "study",            # citation key
    "donor",            # dataset-namespaced patient id  (Li__CTCL1, B2__CTCL2 ...)
    "real_donor",       # cross-dataset patient identity (for dedup; see DONOR_ALIASES)
    "sample_id",        # dataset-namespaced sample/library id
    "sex",              # M | F | unknown
    "disease",          # MF | SS | CTCL_other | HC | AD | Pso | unknown
    "disease_stage",    # IA..IVB | NA
    "compartment",      # Skin | Blood | LN
    "tissue",           # canonical: Epidermis | Dermis | Skin | PBMC | LN  (whole_skin->Skin, Blood->PBMC)
    "tech",             # 10x_5p | 10x_3p | 10x_flex | ECCITE | BD_Rhapsody
    "cell_type_orig",   # source label as-is (or "")
    "cell_type",        # harmonized / scANVI-transferred (atlas vocab); "Unknown" until transfer
]

# TCR columns are joined from the clone table (NaN/False where no TCR):
TCR_OBS = [
    "has_tcr", "clone_id", "tra_cdr3", "trb_cdr3",
    "clone_size", "is_expanded", "is_dominant_clone", "is_malignant",
]

DISEASE_VOCAB = {"MF", "SS", "CTCL_other", "HC", "AD", "Pso", "unknown"}
COMPARTMENT_VOCAB = {"Skin", "Blood", "LN"}

# Cross-deposit duplicate handling -- SUPERSEDED 2026-09-05 by the barcode gate.
#
# This list was a bare-string membership test on `real_donor`, applied in concat_joint
# without scoping to a dataset. That is unsafe: GSE173205 and GSE247047 both publish
# donors literally named P112/P115/P116/P121, so ingesting either would have silently
# deleted their cells. It was also incomplete -- the gate finds 45 identity groups where
# this names 6 patients.
#
# The authority is now tables/atlas_dedup_v2_decisions.csv, keyed on the full namespaced
# sample_id, built by check_overlap.py + make_dedup_decisions.py. See
# docs/ATLAS_V2_DEDUP_GATE.md. Kept only so old callers/notebooks still import.
SHARED_BRUNNER = ["P112", "P115", "P116", "P121", "P303", "PGS"]

DEDUP_LEDGER = "tables/atlas_dedup_v2_decisions.csv"
PATIENT_KEY_TABLE = "tables/atlas_patient_key.csv"

# Friendly dataset names (author+year) for the `dataset` obs column. Internal
# registry keys / namespaces stay as the short codes for cache + join stability.
DATASET_NAME = {
    "Li2024_atlas": "li24",
    "D1": "chennareddy25",
    "D3": "brunner24",
    "H": "herrera21",
    "D5": "rindler21",
    "D6": "gaydosik19",
    "B1": "borcherding19",
    "B2": "borcherding23",
    "B3": "borcherding24",
    "B4": "geskin26",
    "B5": "buus25",
    # --- v2 additions (2026-09-05). Dedup verdicts: docs/ATLAS_V2_DEDUP_GATE.md ---
    "B6": "ren23",
    "B7": "harro23",
    "B8": "dorando26",
    "D7": "rindler21mc",
    "D8": "alkon24",
    "D9": "lyp26",
    "D10": "jonak21",
    "D11": "brentuximab26",
    "D12": "pacritinib26",
    "D13": "gaydosik22",
    "D14": "gaydosik23",
    "D15": "il4ra26",
    "D16": "song24",      # TCR only -- its expression is already in the Li object
}

# Per-paper QC thresholds (keyed by friendly name). Sourced from each paper's
# Methods where available; same-lab proxy / moderate default where the supplement
# is gated (see `source`). %mito from MT- gene symbols. Researched 2026-06.
QC_CONFIG = {
    "li24":          dict(prefiltered=True, source="Li 2024 NatImmunol (atlas already QC'd: >=400 genes, >=1000 counts, <20% mito, Scrublet)"),
    "chennareddy25": dict(min_genes=500, max_pct_mito=10, source="proxy brunner24 (Vienna; Appendix S1 gated)"),
    "brunner24":     dict(min_genes=500, max_pct_mito=10, source="Brunner 2024 JACI (>500 genes, <10% mito, scran doublets)"),
    "herrera21":     dict(min_genes=200, max_pct_mito=20, source="default (Herrera 2021 reports only Scrublet0.2 + HTODemux)"),
    "rindler21":     dict(min_genes=200, max_genes=4000, max_pct_mito=12, source="Rindler 2021 FrontImmunol (200-4000 genes, <12% mito)"),
    "gaydosik19":    dict(min_genes=200, max_pct_mito=15, source="default skin (Gaydosik 2019 CCR supplement gated)"),
    "borcherding19": dict(min_genes=200, max_genes=3500, max_pct_mito=9, source="Borcherding 2019 CCR (200-3500 genes, <9% mito)"),
    "borcherding23": dict(min_genes=200, max_genes=3500, max_pct_mito=9, source="proxy borcherding19 (same lab; 2023 Methods sparse)"),
    "geskin26":      dict(min_genes=200, max_pct_mito=20, source="Geskin 2026 CIR (>=200 genes; mito was regressed -> cap 20%)"),
    "buus25":        dict(min_genes=200, max_pct_mito=20, source="proxy herrera21 (multi-tissue CITE-seq; Buus 2025 CancDisc Methods TBD -> verify)"),
    # --- v2 additions. No entry here means a SILENT fallback to min_genes=200/mito=20
    # (see qc_filter), so every new cohort gets an explicit row even when it is a proxy.
    "ren23":         dict(min_genes=200, max_pct_mito=20, source="default blood 5' (GSE197619 Methods not verified -> verify)"),
    "harro23":       dict(min_genes=200, max_pct_mito=20, source="default (GSE207679 is sorted CD4+CD26- blood; Methods not verified -> verify)"),
    "dorando26":     dict(min_genes=200, max_genes=3500, max_pct_mito=9, source="proxy borcherding19 (WashU, same clinic)"),
    "rindler21mc":   dict(min_genes=200, max_genes=4000, max_pct_mito=12, source="proxy rindler21 (same group + year: Rindler 2021 FrontImmunol 200-4000 genes, <12% mito)"),
    "alkon24":       dict(min_genes=500, max_pct_mito=10, source="proxy brunner24 (Vienna, as chennareddy25)"),
    "lyp26":         dict(min_genes=500, max_pct_mito=10, source="proxy brunner24 (Vienna)"),
    "jonak21":       dict(min_genes=500, max_pct_mito=10, source="proxy brunner24 (Vienna)"),
    "brentuximab26": dict(min_genes=200, max_pct_mito=15, source="default skin (GSE303446 Methods not verified -> verify)"),
    "pacritinib26":  dict(min_genes=200, max_pct_mito=15, source="default skin (GSE309807 / Wilcox in-house)"),
    "gaydosik22":    dict(min_genes=200, max_pct_mito=15, source="proxy gaydosik19 (Pittsburgh, same lab)"),
    "gaydosik23":    dict(min_genes=200, max_pct_mito=15, source="proxy gaydosik19 (Pittsburgh, same lab)"),
    "il4ra26":       dict(min_genes=200, max_pct_mito=20, source="proxy geskin26 (Pittsburgh, same lab + year)"),
}
QC_OBS = ["n_genes", "total_counts", "pct_mito", "doublet_score", "predicted_doublet"]


# ----------------------------------------------------------------------------
# Dataset registry
# ----------------------------------------------------------------------------
def dataset_registry(nb_dir: Path) -> dict:
    """Per-dataset build config. ``nb_dir`` = notebooks/MF."""
    data = nb_dir / "data"
    R = {
        "Li2024_atlas": dict(
            study="li2024", source="atlas_h5ad",
            path=data / "CTCL_all_final_portal_tags.h5ad",
            tech_default="10x", compartment_default="Skin",
            has_tcr=False, primary=True, in_expression=True,
        ),
        "D1": dict(
            study="chennareddy2025", source="concat",
            path=data / "D1_chennareddy2025/processed/concat.h5ad",
            meta=data / "D1_chennareddy2025/meta/patients.tsv",
            raw=data / "D1_chennareddy2025/raw",
            tech_default="10x_5p", compartment_default="Skin",
            has_tcr=True, tcr_fmt="10x", in_expression=True,
        ),
        "D3": dict(
            study="brunner2024", source="concat",
            path=data / "D3_brunner2024/processed/concat.h5ad",
            meta=data / "D3_brunner2024/meta/samples.tsv",
            raw=data / "D3_brunner2024/raw",
            tech_default="10x_5p", compartment_default="Skin",
            has_tcr=True, tcr_fmt="10x", in_expression=True,
        ),
        "H": dict(
            study="herrera2021", source="eccite",
            raw=data / "H_herrera2021/raw",
            meta=data / "H_herrera2021/meta/patients.tsv",
            tech_default="ECCITE", compartment_default=None,  # per-sample (Blood/Skin)
            has_tcr=True, tcr_fmt="eccite", in_expression=True,
        ),
        "D5": dict(
            study="rindler2021_fi", source="10x_mtx",
            raw=data / "D5_rindler2021frontimm_multitissue/raw",
            meta=data / "D5_rindler2021frontimm_multitissue/meta/samples.tsv",
            tech_default="10x_5p", compartment_default=None,
            features_name="features", has_tcr=True, tcr_fmt="10x", in_expression=True,
        ),
        "D6": dict(
            study="gaydosik2019", source="csv_genesxcells",
            raw=data / "D6_gaydosik2019_skin/raw",
            meta=data / "D6_gaydosik2019_skin/meta/samples.tsv",
            tech_default="10x_3p", compartment_default="Skin",
            has_tcr=False, in_expression=True,
        ),
        "B1": dict(
            study="borcherding2019", source="10x_mtx",
            raw=data / "B1_borcherding2019_blood/raw",
            meta=data / "B1_borcherding2019_blood/meta/samples.tsv",
            tech_default="10x", compartment_default="Blood",
            features_name="genes", has_tcr=False, in_expression=True,
        ),
        "B2": dict(
            study="borcherding2023", source="10x_mtx",
            raw=data / "B2_borcherding2023_blood/raw",
            meta=data / "B2_borcherding2023_blood/meta/samples.tsv",
            tech_default="10x_5p", compartment_default="Blood",
            features_name="features", has_tcr=True, tcr_fmt="10x", in_expression=True,
        ),
        "B3": dict(
            study="borcherding2024", source="bd_rhapsody",
            raw=data / "B3_borcherding2024_blood/raw",
            meta=data / "B3_borcherding2024_blood/meta/samples.tsv",
            tech_default="BD_Rhapsody", compartment_default="Blood",
            # EXCLUDED entirely: 453-gene targeted panel (not in expression atlas)
            # AND deposit's *_tcr.csv has only a paired-chain boolean, NO CDR3
            # sequences -> contributes no clones. Kept on disk for record only.
            has_tcr=False, in_expression=False, excluded=True,
        ),
        "B4": dict(
            study="geskin2026", source="cellranger_h5_hto",  # hashtag-multiplexed: HTODemux per lane
            raw=data / "B4_geskin2026_dupilumab_blood/raw",
            meta=data / "B4_geskin2026_dupilumab_blood/meta/samples.tsv",
            tech_default="10x_5p", compartment_default="Blood",
            has_tcr=True, tcr_fmt="10x", in_expression=True,  # only SZ* libraries have contigs
        ),
        "B5": dict(
            study="buus2025", source="cite_h5",
            raw=data / "B5_buus2025_blood_skin/raw",
            meta=data / "B5_buus2025_blood_skin/meta/samples.tsv",
            tech_default="10x_5p", compartment_default=None,  # per-sample Blood/Skin
            # CITE-seq: filtered_feature_bc_matrix.h5 carries GEX + Antibody Capture;
            # GEX -> expression atlas, ADT kept cohort-local on obsm['X_adt'].
            has_tcr=True, tcr_fmt="10x", in_expression=True, has_adt=True,
        ),

        # ------------------------------------------------------------------ v2 additions
        # Added 2026-09-05 after the barcode dedup gate (docs/ATLAS_V2_DEDUP_GATE.md).
        # Duplicate samples are NOT excluded here -- concat_joint drops them from
        # tables/atlas_dedup_v2_decisions.csv, so each cohort's standardized cache stays a
        # faithful copy of its deposit and the dedup stays auditable in one place.
        "B6": dict(
            study="ren2023", source="10x_mtx",
            raw=data / "B6_ren2023_blood/raw",
            meta=data / "B6_ren2023_blood/meta/samples.tsv",
            tech_default="10x_5p", compartment_default="Blood",
            # GSE197619 renames the triplet: _genes.tsv.gz + _counts.mtx.gz
            features_name="genes", matrix_name="counts",
            has_tcr=True, tcr_fmt="10x", in_expression=True,
        ),
        "B7": dict(
            study="harro2023", source="10x_mtx",
            raw=data / "B7_harro2023_blood/raw",
            meta=data / "B7_harro2023_blood/meta/samples.tsv",
            tech_default="10x_5p", compartment_default="Blood",
            features_name="features", has_tcr=True, tcr_fmt="10x", in_expression=True,
        ),
        "B8": dict(
            study="dorando2026", source="10x_mtx",
            raw=data / "B8_dorando2026_blood/raw",
            meta=data / "B8_dorando2026_blood/meta/samples.tsv",
            tech_default="10x_5p", compartment_default="Blood",
            features_name="features", has_tcr=True, tcr_fmt="10x_tar", in_expression=True,
        ),
        "D7": dict(
            study="rindler2021_mc", source="10x_mtx",
            raw=data / "D7_rindler2021mc_skin/raw",
            meta=data / "D7_rindler2021mc_skin/meta/samples.tsv",
            tech_default="10x_5p", compartment_default="Skin",
            features_name="features", has_tcr=False, in_expression=True,
        ),
        "D8": dict(
            study="alkon2024", source="10x_mtx",
            raw=data / "D8_alkon2024_parapsoriasis/raw",
            meta=data / "D8_alkon2024_parapsoriasis/meta/samples.tsv",
            tech_default="10x_5p", compartment_default="Skin",
            features_name="features", has_tcr=False, in_expression=True,
        ),
        "D9": dict(
            study="lyp2026", source="10x_mtx",
            raw=data / "D9_lyp_vs_ctcl_skin/raw",
            meta=data / "D9_lyp_vs_ctcl_skin/meta/samples.tsv",
            tech_default="10x_5p", compartment_default="Skin",
            features_name="features", has_tcr=False, in_expression=True,
        ),
        "D10": dict(
            study="jonak2021", source="10x_mtx",
            raw=data / "D10_jonak2021_discordant/raw",
            meta=data / "D10_jonak2021_discordant/meta/samples.tsv",
            tech_default="10x_5p", compartment_default="Skin",
            features_name="features", has_tcr=False, in_expression=True,
        ),
        "D11": dict(
            study="brentuximab2026", source="10x_mtx",
            raw=data / "D11_brentuximab_skin/raw",
            meta=data / "D11_brentuximab_skin/meta/samples.tsv",
            tech_default="10x_5p", compartment_default="Skin",
            features_name="features", has_tcr=False, in_expression=True,
        ),
        "D12": dict(
            study="pacritinib2026", source="10x_mtx",
            raw=data / "D12_pacritinib_skin/raw",
            meta=data / "D12_pacritinib_skin/meta/samples.tsv",
            tech_default="10x_5p", compartment_default="Skin",
            # EXCLUDED: GSE309807's only two samples are the DMSO and Pacritinib arms of
            # one ex-vivo-cultured biopsy -- there is no primary-tissue sample. Same rule
            # that dropped B5's 6 culture-only patients (see DATA_PROVENANCE sec 2).
            features_name="features", has_tcr=False, in_expression=False, excluded=True,
        ),
        "D13": dict(
            study="gaydosik2022", source="cellranger_h5_raw",   # deposit is UNFILTERED
            raw=data / "D13_gaydosik2022_skin_blood/raw",
            meta=data / "D13_gaydosik2022_skin_blood/meta/samples.tsv",
            tech_default="10x_5p", compartment_default=None,    # per-sample Skin/Blood
            has_tcr=True, tcr_fmt="10x", in_expression=True,
        ),
        "D14": dict(
            study="gaydosik2023", source="cellranger_h5_raw",   # deposit is UNFILTERED
            raw=data / "D14_gaydosik2023_skin/raw",
            meta=data / "D14_gaydosik2023_skin/meta/samples.tsv",
            tech_default="10x_5p", compartment_default="Skin",
            has_tcr=False, in_expression=True,
        ),
        "D15": dict(
            study="il4ra2026", source="cellranger_h5",          # filtered
            raw=data / "D15_il4ra_blockade_skin/raw",
            meta=data / "D15_il4ra_blockade_skin/meta/samples.tsv",
            tech_default="10x_5p", compartment_default="Skin",
            has_tcr=True, tcr_fmt="10x", in_expression=True,
        ),
        "D16": dict(
            study="song2024", source="zip_10x",
            raw=data / "D16_song2024_transformed/raw",
            meta=data / "D16_song2024_transformed/meta/samples.tsv",
            tech_default="10x_5p", compartment_default="Skin",
            # EXPRESSION EXCLUDED: the barcode gate shows all 8 PT donors are already in the
            # Li object, which merged both the Tumor and PP libraries. Ingesting this would
            # double-count them. Kept for its V(D)J only, which supersedes the TRUST4 route
            # and is the ONLY TCR for PT47/PT50. See docs/ATLAS_V2_DEDUP_GATE.md sec 2.
            has_tcr=True, tcr_fmt="10x_zip", in_expression=False, excluded=True,
        ),
    }
    return R


# ----------------------------------------------------------------------------
# Expression loaders -> AnnData (cells x genes, gene-symbol var_names, raw int X)
# ----------------------------------------------------------------------------
def _read_mtx_triplet(prefix: Path, features_name: str = "features",
                      matrix_name: str = "matrix"):
    """10x mtx triplet -> (csr cells x genes, gene_symbols list, barcodes list).

    `features_name` / `matrix_name` cover deposits that rename the triplet: GSE197619
    ships `_genes.tsv.gz` + `_counts.mtx.gz` rather than `_features` + `_matrix`.
    """
    import scipy.io as sio
    import scipy.sparse as sp

    mtx = Path(f"{prefix}_{matrix_name}.mtx.gz")
    bc = Path(f"{prefix}_barcodes.tsv.gz")
    ft = Path(f"{prefix}_{features_name}.tsv.gz")
    with gzip.open(mtx, "rb") as fh:
        X = sio.mmread(fh).tocsr().T.tocsr()  # genes x cells -> cells x genes
    barcodes = pd.read_csv(bc, header=None, sep="\t")[0].astype(str).tolist()
    feats = pd.read_csv(ft, header=None, sep="\t")
    symbols = feats[1].astype(str).tolist() if feats.shape[1] > 1 else feats[0].astype(str).tolist()
    return sp.csr_matrix(X, dtype=np.float32), symbols, barcodes


def _read_csv_genesxcells(path: Path):
    """Gaydosik per-sample CSV (genes x cells) -> (csr cells x genes, symbols, barcodes)."""
    import scipy.sparse as sp

    df = pd.read_csv(path, index_col=0)            # rows=genes, cols=cells
    symbols = df.index.astype(str).tolist()
    barcodes = df.columns.astype(str).tolist()
    X = sp.csr_matrix(df.to_numpy(dtype=np.float32).T)  # cells x genes
    return X, symbols, barcodes


def _read_eccite_gex(path: Path):
    """Herrera ECCITE GEX tsv (genes x cells, dense) -> (csr cells x genes, ...)."""
    return _read_csv_genesxcells_tsv(path)


def _read_csv_genesxcells_tsv(path: Path):
    import scipy.sparse as sp

    df = pd.read_csv(path, sep="\t", index_col=0)
    symbols = df.index.astype(str).tolist()
    barcodes = df.columns.astype(str).tolist()
    X = sp.csr_matrix(df.to_numpy(dtype=np.float32).T)
    return X, symbols, barcodes


def make_anndata(X, symbols, barcodes, sample_id: str):
    """Assemble a per-sample AnnData with raw_counts layer + namespaced obs_names."""
    import anndata as ad

    a = ad.AnnData(X=X)
    a.var_names = [str(s) for s in symbols]
    a.var_names_make_unique()
    a.obs_names = [f"{sample_id}|{b}" for b in barcodes]
    a.layers["raw_counts"] = a.X.copy()
    a.obs["_barcode"] = [_clean_bc(b) for b in barcodes]
    a.obs["sample_id"] = sample_id
    return a


def _clean_bc(bc: str) -> str:
    """Strip the trailing 10x lane suffix so GEX and VDJ barcodes match."""
    return re.sub(r"-\d+$", "", str(bc))


# ----------------------------------------------------------------------------
# obs harmonization
# ----------------------------------------------------------------------------
def standardize_obs(adata, dataset: str, study: str, row: pd.Series | dict,
                    cell_type_orig_col: str | None = None):
    """Attach the CANON_OBS columns to one already-loaded per-sample AnnData.

    ``row`` is the matching sample-metadata record (a meta/*.tsv row) providing
    donor / disease / stage / tissue / sex / compartment. Missing values are
    filled with vocab-safe defaults.
    """
    g = lambda k, d="": str(row[k]) if (k in row and pd.notna(row[k])) else d  # noqa: E731

    donor = namespace(dataset, g("donor") or g("participant_id") or g("patient_id") or g("sample_id"))
    obs = adata.obs
    obs["dataset"] = DATASET_NAME.get(dataset, dataset)
    obs["study"] = study
    obs["donor"] = donor
    obs["real_donor"] = g("real_donor") or g("participant_id") or g("patient_id") or donor
    obs["sample_id"] = namespace(dataset, obs["sample_id"].iloc[0] if "sample_id" in obs else g("sample_id"))
    obs["sex"] = g("sex", "unknown")
    obs["disease"] = _norm_disease(g("disease"))
    obs["disease_stage"] = g("disease_stage") or g("stage") or g("stage_class") or "NA"
    obs["compartment"] = _norm_compartment(g("compartment") or g("tissue"))
    obs["tissue"] = _norm_tissue(g("tissue") or obs["compartment"].iloc[0])
    obs["tech"] = g("tech") or ""
    obs["cell_type_orig"] = obs[cell_type_orig_col].astype(str) if cell_type_orig_col and cell_type_orig_col in obs else ""
    obs["cell_type"] = "Unknown"
    obs["cell_id"] = obs.index.astype(str)
    return adata


def namespace(dataset: str, value: str) -> str:
    value = str(value)
    return value if value.startswith(f"{dataset}__") else f"{dataset}__{value}"


def _norm_disease(v: str) -> str:
    v = (v or "").strip()
    if v in DISEASE_VOCAB:
        # already canonical -- identity. Without this, "CTCL_other" falls through the map
        # to "unknown" (there is a "CTCL" key but no "CTCL_other" key) and the vocab assert
        # in run_build_joint.py cannot catch it, because "unknown" is itself legal.
        return v
    m = {
        "HC": "HC", "Healthy control": "HC", "NOR": "HC", "Control": "HC",
        "MF": "MF", "CD4_MF": "MF", "leukemic MF": "MF",
        "SS": "SS", "Sezary": "SS",
        "BL": "CTCL_other", "Berti": "CTCL_other", "gd_MF": "CTCL_other",
        "eCTCL": "CTCL_other", "CTCL": "CTCL_other",
        "AD": "AD", "Pso": "Pso", "Psoriasis": "Pso",
    }
    return m.get(v, "unknown" if v == "" else m.get(v.split()[0], "unknown"))


def _norm_compartment(v: str) -> str:
    v = (v or "").strip().lower()
    if any(k in v for k in ["pbmc", "blood", "peripheral"]):
        return "Blood"
    if v in ("ln", "lymph node", "lymph_node"):
        return "LN"
    return "Skin"  # epidermis/dermis/whole_skin/skin dissociated -> Skin


def _norm_tissue(v: str) -> str:
    """Canonical tissue vocab (consistent casing + granularity):
    {Epidermis, Dermis, Skin, PBMC, LN}. Atlas `whole_skin` and cohort `skin`
    both -> Skin; blood -> PBMC (compartment already carries Blood)."""
    v = str(v).strip().lower()
    if "epiderm" in v:
        return "Epidermis"
    if "derm" in v:                       # dermis (epidermis caught above)
        return "Dermis"
    if any(k in v for k in ["pbmc", "blood", "peripheral"]):
        return "PBMC"
    if "lymph" in v or v in ("ln", "lymph_node"):
        return "LN"
    return "Skin"                         # skin / whole_skin / skin dissociated / biopsy


# ----------------------------------------------------------------------------
# TCR ingestion -> AIRR-ish per-cell table -> CDR3 clone id
# ----------------------------------------------------------------------------
# We build a uniform per-cell frame with columns
#   sample_id, _barcode, tra_cdr3, trb_cdr3
# then derive a platform-agnostic clone_id from the CDR3-aa pair. scirpy can be
# used for fuzzy CDR3 clustering (see notebook), but the exact-CDR3 key below is
# version-proof and what the cohort papers use for clonality.

def airr_from_10x_contig(contig_path: Path) -> pd.DataFrame:
    """10x filtered_contig_annotations.csv(.gz) -> per-cell TRA/TRB CDR3-aa."""
    df = pd.read_csv(contig_path)
    cols = {c.lower(): c for c in df.columns}
    bc = cols.get("barcode", "barcode")
    chain = cols.get("chain", "chain")
    cdr3 = cols.get("cdr3", cols.get("cdr3_aa"))
    if cdr3 is None:
        # A deposit can ship a contig table with the CDR3 columns stripped: GSE182861's
        # MF24 (GSM5538784) has `...,productive,reads,umis,clonotype_id,...` and no
        # `cdr3`/`cdr3_nt` at all, replaced by the submitter's own clonotype labels. Our
        # clone_id is a CDR3 key, so there is nothing to build -- same situation that
        # excluded B3/GSE192836 entirely. Skip the sample loudly instead of raising; it
        # keeps its CNV-based malignant call.
        _nm = getattr(contig_path, "name", None) or str(contig_path)
        print(f"    [TCR] {Path(_nm).name if isinstance(_nm, str) else _nm}: NO cdr3 column "
              f"(has {sorted(cols)[:8]}...) -> sample contributes no clones")
        return None
    prod = cols.get("productive")
    d = df[[bc, chain, cdr3] + ([prod] if prod else [])].copy()
    d.columns = ["barcode", "chain", "cdr3"] + (["productive"] if prod else [])
    if prod:
        d = d[d["productive"].astype(str).str.lower().isin(["true", "1", "yes"])]
    d = d[d["cdr3"].notna() & (d["cdr3"].astype(str) != "None")]
    return _collapse_chains(d)


def airr_from_bd_tcr(tcr_csv: Path) -> pd.DataFrame:
    """BD Rhapsody *_tcr.csv(.gz) -> per-cell TRA/TRB CDR3-aa.

    BD VDJ exports vary; we look for cell-id + chain (TCR_Alpha/Beta) + CDR3 aa
    columns case-insensitively.
    """
    df = pd.read_csv(tcr_csv)
    low = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in low:
                return low[n]
        return None

    bc = pick("cell_index", "cell", "barcode", "cell_id")
    cdr3a = pick("tcr_alpha_cdr3_translation_dominant", "cdr3a", "alpha_cdr3", "tra_cdr3")
    cdr3b = pick("tcr_beta_cdr3_translation_dominant", "cdr3b", "beta_cdr3", "trb_cdr3")
    if bc and (cdr3a or cdr3b):  # wide format: one row per cell
        out = pd.DataFrame({"barcode": df[bc].astype(str)})
        out["tra_cdr3"] = df[cdr3a].astype(str) if cdr3a else ""
        out["trb_cdr3"] = df[cdr3b].astype(str) if cdr3b else ""
        return _clean_cdr3_frame(out)
    # long format fallback
    chain = pick("chain", "tcr_chain", "locus")
    cdr3 = pick("cdr3", "cdr3_aa", "cdr3_translation")
    d = df[[bc, chain, cdr3]].copy()
    d.columns = ["barcode", "chain", "cdr3"]
    return _collapse_chains(d)


def airr_from_eccite(clonotype_ab_path: Path) -> pd.DataFrame:
    """Herrera ECCITE ``*_clonotypeAB.tsv.gz`` -> per-cell TRA/TRB CDR3-aa.

    The file is a (clonotype x cell) membership matrix whose row names encode the
    CDR3s, e.g. ``clonotypeAB-TRA:CAGS...|TRB:CASS...``. For each cell we take the
    clonotype row with the largest (usually only) nonzero entry.
    """
    m = pd.read_csv(clonotype_ab_path, sep="\t", index_col=0)
    arr = m.to_numpy()
    rownames = m.index.astype(str).to_numpy()
    cells = m.columns.astype(str)
    nz = (arr != 0).any(axis=0)
    top = arr.argmax(axis=0)
    rows = []
    for j, cell in enumerate(cells):
        if not nz[j]:
            continue
        tra, trb = _parse_clonotype_ab(rownames[top[j]])
        rows.append({"barcode": cell, "tra_cdr3": tra, "trb_cdr3": trb})
    return _clean_cdr3_frame(pd.DataFrame(rows, columns=["barcode", "tra_cdr3", "trb_cdr3"]))


def _parse_clonotype_ab(name: str):
    """``clonotypeAB-TRA:X|TRB:Y[|...]`` -> (first TRA cdr3, first TRB cdr3)."""
    s = str(name).split("clonotypeAB-", 1)[-1]
    tra = trb = ""
    for part in s.split("|"):
        if part.startswith("TRA:") and not tra:
            tra = part[4:]
        elif part.startswith("TRB:") and not trb:
            trb = part[4:]
    return tra, trb


def airr_from_trust4(airr_tsv: Path) -> pd.DataFrame:
    """TRUST4 ``*_barcode_airr.tsv`` -> per-cell TRA/TRB CDR3-aa.

    TRUST4's per-barcode AIRR output (5' GEX read-mined) is long-format: one row
    per contig with ``cell_id`` (the bare 16bp barcode), ``locus`` (TRA/TRB) and
    ``junction_aa`` (CDR3). Keep productive rows, then collapse to one TRA + one
    TRB per cell via the shared :func:`_collapse_chains`. TRB-primary by design —
    5' mining recovers TRB far more reliably than TRA.
    """
    df = pd.read_csv(airr_tsv, sep="\t", dtype=str)
    low = {c.lower(): c for c in df.columns}
    bc = low.get("cell_id", low.get("barcode", "cell_id"))
    locus = low.get("locus", low.get("chain", "locus"))
    cdr3 = low.get("junction_aa", low.get("cdr3_aa", low.get("cdr3", "junction_aa")))
    prod = low.get("productive")
    keep = [bc, locus, cdr3] + ([prod] if prod else [])
    d = df[keep].copy()
    d.columns = ["barcode", "chain", "cdr3"] + (["productive"] if prod else [])
    if prod:
        d = d[d["productive"].astype(str).str.lower().isin(["true", "t", "1", "yes"])]
    d = d[d["cdr3"].notna() & (d["cdr3"].astype(str) != "None") & (d["cdr3"].astype(str) != "")]
    return _collapse_chains(d)


def _collapse_chains(d: pd.DataFrame) -> pd.DataFrame:
    """Long (barcode, chain, cdr3) -> wide (barcode, tra_cdr3, trb_cdr3).

    Keeps the first productive chain per locus (TRA/TRB) per cell.
    """
    d = d.copy()
    d["chain"] = d["chain"].astype(str).str.upper()
    tra = (d[d["chain"].isin(["TRA", "ALPHA"])]
           .drop_duplicates("barcode").set_index("barcode")["cdr3"])
    trb = (d[d["chain"].isin(["TRB", "BETA"])]
           .drop_duplicates("barcode").set_index("barcode")["cdr3"])
    out = pd.DataFrame(index=sorted(set(d["barcode"])))
    out["tra_cdr3"] = tra.reindex(out.index).fillna("")
    out["trb_cdr3"] = trb.reindex(out.index).fillna("")
    out = out.reset_index().rename(columns={"index": "barcode"})
    return _clean_cdr3_frame(out)


def _clean_cdr3_frame(out: pd.DataFrame) -> pd.DataFrame:
    out["barcode"] = out["barcode"].map(_clean_bc)
    for c in ("tra_cdr3", "trb_cdr3"):
        out[c] = out[c].astype(str).replace({"nan": "", "None": ""})
    return out.drop_duplicates("barcode").reset_index(drop=True)


def clone_id_from_cdr3(tra: str, trb: str) -> str:
    """Platform-agnostic exact clone key, **TRB-primary**.

    TRB-CDR3 is the clonality gold standard and TRA frequently drops out in 10x
    VDJ, so keying on TRB alone (when present) avoids splitting one clone across
    cells that did/did not recover a TRA. Falls back to TRA when no TRB.
    NOTE: this is the version-proof exact key; the notebook additionally runs
    scirpy clonotype-cluster (hamming cutoff 1) to merge 1-mismatch variants.
    """
    trb = (trb or "").strip()
    tra = (tra or "").strip()
    if trb:
        return f"TRB:{trb}"
    if tra:
        return f"TRA:{tra}"
    return ""


def build_clone_table(per_cell: pd.DataFrame, dataset: str,
                      sample_to_donor: dict,
                      frac_thresh: float = 0.05,
                      ratio_thresh: float = 2.0,
                      expanded_min: int = 2,
                      cell_to_donor: dict | None = None) -> pd.DataFrame:
    """Given per-cell CDR3 (cols: sample_id, barcode, tra_cdr3, trb_cdr3) for one
    dataset, compute clone_id, clone_size (per sample), and per-sample
    dominant/malignant flags (07_d1 thresholds, generalized).

    Returns a frame keyed by cell_id with the TCR_OBS columns.
    """
    df = per_cell.copy()
    df["clone_id"] = [clone_id_from_cdr3(a, b) for a, b in zip(df["tra_cdr3"], df["trb_cdr3"])]
    df = df[df["clone_id"] != ""]
    df["dataset"] = dataset

    # Dominance is a PER-PATIENT statistic. Grouping by sample_id is wrong on a
    # hashtag-multiplexed lane: several donors share one denominator and one `top` clone,
    # so at most one of them can ever be called malignant and the rest are silently False.
    # `cell_to_donor` (cell_id -> donor, read from the cohort's standardized obs) restores
    # per-donor granularity where a lane was demultiplexed. CHANGES v1 BEHAVIOUR for the
    # hashed cohorts (B4's 6 TCR lanes) -- see docs/ATLAS_V1_PARAMETERS.md sec 4.
    df["_cid"] = [f"{namespace(dataset, s_)}|{b}" for s_, b in zip(df["sample_id"], df["barcode"])]
    df["_ns_sid"] = [namespace(dataset, s_) for s_ in df["sample_id"]]
    df["_donor"] = [(cell_to_donor or {}).get(c, "") for c in df["_cid"]]
    # Subdivide a sample ONLY when it actually holds more than one donor (a hashed lane).
    # Grouping on donor alone would instead MERGE a donor's separate biopsies -- D1 keys
    # P318_thin and P318_thick as one patient -- which silently changes the per-sample
    # dominance v1 measured its 0.981/0.834 benchmark on. Falling back to the bare
    # `sample_id` would also strip the dataset namespace from clone_id.
    multi = {s_ for s_, g in df.groupby("_ns_sid")["_donor"]
             if len({d for d in g if d}) > 1}
    df["_grp"] = [d if (ns in multi and d) else ns
                  for ns, d in zip(df["_ns_sid"], df["_donor"])]

    rows = []
    for _grp, sub in df.groupby("_grp"):
        sizes = sub["clone_id"].value_counts()
        n = len(sub)
        top = sizes.index[0]
        top_n = int(sizes.iloc[0])
        second_n = int(sizes.iloc[1]) if len(sizes) > 1 else 0
        dom_frac = top_n / max(1, n)
        ratio = top_n / max(1, second_n) if second_n else np.inf
        is_dom_sample = (dom_frac >= frac_thresh) and (ratio >= ratio_thresh)
        for _, r in sub.iterrows():
            cid = r["clone_id"]
            sz = int(sizes[cid])
            sid = r["sample_id"]
            rows.append({
                "cell_id": r["_cid"],
                "dataset": dataset,
                "sample_id": namespace(dataset, sid),
                "donor": r["_donor"] or sample_to_donor.get(sid, namespace(dataset, sid)),
                "has_tcr": True,
                # clones are patient-private; on a demultiplexed lane that means per DONOR,
                # not per lane, or two patients' identical CDR3s would merge into one clone
                "clone_id": f"{_grp}::{cid}",
                "tra_cdr3": r["tra_cdr3"],
                "trb_cdr3": r["trb_cdr3"],
                "clone_size": sz,
                "is_expanded": sz >= expanded_min,
                "is_dominant_clone": is_dom_sample and (cid == top),
                "is_malignant": is_dom_sample and (cid == top),
            })
    cols = ["cell_id", "dataset", "sample_id", "donor"] + TCR_OBS
    return pd.DataFrame(rows, columns=cols)


# ----------------------------------------------------------------------------
# Step-1 standardizers: each returns one AnnData with EXACTLY CANON_OBS columns,
# X = raw counts, raw_counts layer, gene-symbol var_names, consistent cell_id.
# ----------------------------------------------------------------------------
_TECH_MAP = {"10x": "10x_5p", "flex": "10x_flex"}

# column maps for the already-built concat.h5ad cohorts
_CONCAT_MAP = {
    "D1": dict(donor="participant_id", sample="sample_id_in_raw", disease="disease",
               stage="stage_class", celltype="subset", compartment="Skin",
               tissue="skin", tech="10x_5p"),
    "D3": dict(donor="patient_id", sample="sample_id_in_raw", disease="group",
               stage=None, celltype=None, compartment="Skin",
               tissue="Skin", tech="10x_5p"),
}


def _keep_canon(adata):
    """Drop all obs columns except CANON_OBS (TCR joined later); fix dup genes."""
    for c in CANON_OBS:
        if c not in adata.obs:
            adata.obs[c] = ""
    adata.obs = adata.obs[CANON_OBS].copy()
    adata = adata[:, ~adata.var_names.duplicated()].copy()
    if "raw_counts" not in adata.layers:
        adata.layers["raw_counts"] = adata.X.copy()
    return adata


def make_cell_ids(dataset, sample_ids, barcodes_clean):
    return [f"{namespace(dataset, s)}|{b}" for s, b in zip(sample_ids, barcodes_clean)]


def standardize_atlas(reg_entry):
    import scanpy as sc

    a = sc.read_h5ad(reg_entry["path"])
    a.X = a.layers["raw_counts"].copy()
    raw_donor = a.obs["donor"].astype(str)
    a.obs["real_donor"] = raw_donor.values
    a.obs["donor"] = [namespace("Li2024_atlas", x) for x in raw_donor]
    a.obs["sample_id"] = a.obs["donor"].values
    a.obs["study"] = "li2024"
    a.obs["dataset"] = DATASET_NAME["Li2024_atlas"]
    a.obs["disease"] = "MF"               # CTCL_all = MF/CTCL skin atlas (see README caveat re PT donors)
    a.obs["disease_stage"] = a.obs["stage"].astype(str)
    a.obs["compartment"] = "Skin"
    a.obs["tissue"] = a.obs["tissue"].astype(str).map(_norm_tissue)  # Epidermis/Dermis/Skin (whole_skin->Skin)
    a.obs["tech"] = a.obs["tech"].astype(str).map(lambda t: _TECH_MAP.get(t, t))
    a.obs["sex"] = a.obs["sex"].astype(str)
    a.obs["cell_type_orig"] = a.obs["cell_type"].astype(str)   # keep
    a.obs["cell_type"] = a.obs["cell_type"].astype(str)        # KNOWN labels -> scANVI reference
    a.obs["cell_id"] = "Li2024_atlas|" + a.obs_names.astype(str)
    return _keep_canon(a)


def standardize_concat(label, reg_entry):
    import scanpy as sc

    m = _CONCAT_MAP[label]
    a = sc.read_h5ad(reg_entry["path"])
    if "raw_counts" not in a.layers:
        a.layers["raw_counts"] = a.X.copy()
    a.X = a.layers["raw_counts"].copy()

    sid = a.obs[m["sample"]].astype(str)
    donor = a.obs[m["donor"]].astype(str)
    # reconstruct clean barcode from obs_name = "<sid>_<barcode>"
    names = a.obs_names.astype(str)
    bc = [n[len(s) + 1:] if n.startswith(s + "_") else n.rsplit("_", 1)[-1]
          for n, s in zip(names, sid)]
    cb = [_clean_bc(b) for b in bc]

    a.obs["real_donor"] = donor.values
    a.obs["donor"] = [namespace(label, d) for d in donor]
    a.obs["sample_id"] = [namespace(label, s) for s in sid]
    a.obs["study"] = reg_entry["study"]
    a.obs["dataset"] = DATASET_NAME.get(label, label)
    a.obs["disease"] = a.obs[m["disease"]].astype(str).map(_norm_disease)
    a.obs["disease_stage"] = a.obs[m["stage"]].astype(str) if m["stage"] else "NA"
    a.obs["compartment"] = m["compartment"]
    a.obs["tissue"] = _norm_tissue(m["tissue"])
    a.obs["tech"] = m["tech"]
    a.obs["sex"] = "unknown"
    a.obs["cell_type_orig"] = a.obs[m["celltype"]].astype(str) if m["celltype"] else ""
    a.obs["cell_type"] = "Unknown"
    a.obs["cell_id"] = make_cell_ids(label, sid, cb)
    return _keep_canon(a)


def _load_one_sample(label, reg_entry, row):
    """Dispatch a single sample to the right expression loader -> AnnData."""
    raw = reg_entry["raw"]
    # per-row override: a cohort can mix loaders. GSE182861 deposits plain lanes next to
    # hashtag-multiplexed ones (`SZ29_MF25_GEX_HTO`, `HC1and2_GEX`), which is invisible in
    # the file names and only stated in the series-matrix titles.
    _ld = row.get("loader")
    src = (str(_ld).strip() if (_ld is not None and pd.notna(_ld) and str(_ld).strip())
           else reg_entry["source"])
    gex = row["gex_file"]
    sid = str(row["sample_id"])
    if src == "10x_mtx":
        mname = reg_entry.get("matrix_name", "matrix")
        prefix = str(Path(raw) / gex).replace(f"_{mname}.mtx.gz", "")
        X, sym, bc = _read_mtx_triplet(Path(prefix),
                                       reg_entry.get("features_name", "features"), mname)
    elif src == "csv_genesxcells":
        X, sym, bc = _read_csv_genesxcells(Path(raw) / gex)
    elif src in ("cellranger_h5_hto", "cellranger_h5_hto_raw"):
        a = _load_hto_lane(Path(raw) / gex, sid,
                           call_cells=src.endswith("_raw"))
        return a, list(a.obs["_barcode"]), sid
    elif src in ("cellranger_h5", "cellranger_h5_raw"):
        import scanpy as sc
        a = sc.read_10x_h5(str(Path(raw) / gex))
        a.var_names_make_unique()
        if src == "cellranger_h5_raw":
            a = _call_cells_raw_h5(a, sid)
        a.layers["raw_counts"] = a.X.copy()
        a.obs["sample_id"] = sid
        a.obs["_barcode"] = [_clean_bc(b) for b in a.obs_names]
        return a, list(a.obs["_barcode"]), sid
    else:
        raise ValueError(f"bad source {src}")
    a = make_anndata(X, sym, bc, sid)
    return a, list(a.obs["_barcode"]), sid


def _load_hto_lane(h5_path: Path, sid: str, call_cells: bool = False):
    """One hashtag-multiplexed CellRanger lane -> GEX AnnData + per-cell `_hto` label.

    Same approach as load_geskin26_hto (B4) but per sample, so a cohort can mix plain and
    hashed lanes. HTO features are `HTO1..HTO10` inside the Antibody Capture block; the
    remaining Antibody Capture features are a 20-marker surface panel (kept out of the
    expression matrix; not currently carried to obsm -- see docs/ATLAS_V2_DEDUP_GATE.md).
    """
    import numpy as np
    import scanpy as sc
    import scipy.sparse as sp

    full = sc.read_10x_h5(h5_path, gex_only=False)
    full.var_names_make_unique()
    ft = full.var["feature_types"].astype(str).to_numpy()
    is_hto = np.asarray(full.var_names.str.upper().str.startswith("HTO"))
    gex = full[:, ft == "Gene Expression"].copy()
    if call_cells:
        keep0 = np.asarray((gex.X > 0).sum(1)).ravel() >= RAW_H5_PRECALL_MIN_GENES
        n0 = gex.n_obs
        gex = gex[keep0].copy()
        full = full[keep0].copy()
        print(f"  [hto raw] {sid}: {n0:,} droplets -> {gex.n_obs:,}")

    hto_class = np.array(["NA"] * gex.n_obs, dtype=object)
    method = "none"
    if is_hto.sum() >= 2:
        H = full[:, is_hto]
        Hm = H.X.toarray() if sp.issparse(H.X) else np.asarray(H.X)
        cols = list(H.var_names.astype(str))
        # Keep only the hashtags actually used in THIS lane. The 10-plex panel is shared
        # across the deposit, so every lane carries 7-8 unused tags at background level;
        # feeding those to hashsolo makes it call nearly everything Doublet/Negative
        # (GSM5538785 kept 2 cells of 5,364). Real vs background is 3+ orders of magnitude
        # apart (1e6-1e7 vs 1e1-1e4), so 1% of the top tag is a wide margin. The surviving
        # count reproduces each lane's title exactly: HC1and2 -> 2, HB1_2_3 -> 3.
        tot = np.asarray(Hm.sum(0)).ravel()
        used_i = [i for i, t in enumerate(tot) if t >= 0.01 * tot.max()] if tot.max() else []
        used = [cols[i] for i in used_i]
        print(f"  [hto] {sid}: hashtags in use {used} "
              f"(dropped {len(cols) - len(used)} background tags)")

        cls = None
        if len(used) >= 2:
            try:
                tmp = gex.copy()
                for i in used_i:
                    tmp.obs[cols[i]] = Hm[:, i]
                sc.external.pp.hashsolo(tmp, used)
                cand = tmp.obs["Classification"].astype(str).to_numpy()
                frac = float(np.mean(~np.isin(cand, ["Doublet", "Negative"])))
                if frac >= HTO_MIN_SINGLET_FRAC:
                    cls, method = cand, "hashsolo"
                else:
                    print(f"  [hto] {sid}: hashsolo kept only {frac:.1%} singlets "
                          f"-> falling back to CLR argmax")
            except Exception as e:  # noqa: BLE001
                print(f"  [hto] {sid}: hashsolo failed ({e!r}) -> CLR argmax")

        if cls is None and len(used) >= 2:
            # Seurat-style HTODemux fallback: centred-log-ratio across the real tags, then
            # argmax with a margin. These lanes carry heavy ambient (every cell has counts
            # of every used tag), which hashsolo's model handles badly, but the tags still
            # separate donors cleanly -- log2(top1/top2) spans -6..+5 and the CLR argmax
            # split reproduces the expected donors per lane.
            M = np.log1p(Hm[:, used_i].astype(float))
            clr = M - M.mean(axis=1, keepdims=True)
            order = np.sort(clr, axis=1)
            margin = order[:, -1] - order[:, -2]
            am = clr.argmax(axis=1)
            cls = np.where(margin >= HTO_CLR_MARGIN,
                           np.array(used, dtype=object)[am], "Negative")
            method = f"clr_argmax(margin>={HTO_CLR_MARGIN})"

        if cls is not None:
            keep = ~np.isin(cls, ["Doublet", "Negative"])
            n0 = gex.n_obs
            if keep.sum() < HTO_MIN_KEEP_FRAC * n0:
                # refuse to silently throw a lane away; carry it undemuxed and flag it
                print(f"  [hto] {sid}: only {keep.sum()}/{n0} cells assigned by {method} "
                      f"-> KEEPING LANE UNDEMUXED as a single pooled donor")
                hto_class = np.array(["POOLED"] * n0, dtype=object)
                method += "+pooled_fallback"
            else:
                gex = gex[keep].copy()
                hto_class = np.asarray(cls, dtype=object)[keep]
                print(f"  [hto] {sid}: {n0:,} -> {gex.n_obs:,} singlets by {method} over "
                      f"{len(set(hto_class))} hashtags "
                      f"{dict(zip(*np.unique(hto_class, return_counts=True)))}")
    else:
        print(f"  [hto] {sid}: WARNING no HTO features found -- lane kept undemuxed")

    gex.obs = gex.obs[[]].copy()
    gex.layers["raw_counts"] = gex.X.copy()
    gex.obs["sample_id"] = sid
    gex.obs["_barcode"] = [_clean_bc(b) for b in gex.obs_names]
    if method != "none":
        # only claim a per-cell hashtag when demux actually ran; otherwise leave `_hto`
        # unset so load_persample_dataset keeps the donor from samples.tsv rather than
        # silently rewriting it to "<lane>_NA".
        gex.obs["_hto"] = hto_class
    gex.uns["hto_demux"] = {"sample_id": sid, "method": method,
                            "n_cells": int(gex.n_obs)}
    return gex


HTO_MIN_SINGLET_FRAC = 0.30   # below this, hashsolo is judged to have failed on the lane
HTO_CLR_MARGIN = 0.5          # CLR top1-top2 gap required to call a singlet
HTO_MIN_KEEP_FRAC = 0.10      # below this, keep the lane pooled rather than discard it

RAW_H5_PRECALL_MIN_GENES = 100   # deliberately below every QC_CONFIG min_genes


def _call_cells_raw_h5(a, sid, min_genes=RAW_H5_PRECALL_MIN_GENES):
    """Drop empty droplets from an unfiltered `raw_feature_bc_matrix.h5`.

    GSE182861 / GSE206123 deposit RAW matrices -- the full ~737k 10x whitelist, mostly
    empty droplets. Every other cohort in the registry supplies CellRanger-*filtered*
    matrices, so without this those two would enter QC with 100x the cells and a
    completely different ambient profile.

    This is a floor, not the QC call: the threshold is below every QC_CONFIG min_genes,
    so `qc_filter` remains the authority on which cells are kept.
    """
    import numpy as np

    n0 = a.n_obs
    ngene = np.asarray((a.X > 0).sum(1)).ravel()
    a = a[ngene >= min_genes].copy()
    print(f"  [raw h5] {sid}: {n0:,} droplets -> {a.n_obs:,} at >={min_genes} genes")
    return a


def _load_hto_map(reg_entry):
    """meta/hto_map.tsv -> {(sample_id, HTOn): {donor, compartment, tissue, disease, keep}}.

    A hashed lane's single samples.tsv row cannot describe its hashtags: GSE182861 puts a
    blood and a skin-tumour sample in one lane, and GSE293752 puts a vehicle-control and a
    drug-treated arm in one lane. Built by data/finalize_samples.py from the deposit's own
    free-text descriptions.
    """
    meta = reg_entry.get("meta")
    if not meta:
        return {}
    f = Path(meta).parent / "hto_map.tsv"
    if not f.exists():
        return {}
    t = pd.read_csv(f, sep="\t", dtype=str).fillna("")
    return {(r["sample_id"], r["hto"].upper()): r.to_dict() for _, r in t.iterrows()}


def load_persample_dataset(label, reg_entry, meta_df):
    """Build + standardize one cohort from per-sample raw files."""
    import anndata as ad

    hto_map = _load_hto_map(reg_entry)
    parts = []
    for _, row in meta_df.iterrows():
        if not (Path(reg_entry["raw"]) / row["gex_file"]).exists():
            print(f"  [{label}] MISSING {row['gex_file']} — skip"); continue
        a, cb, sid = _load_one_sample(label, reg_entry, row)
        if "raw_counts" not in a.layers:
            a.layers["raw_counts"] = a.X.copy()
        hto = a.obs["_hto"].to_numpy() if "_hto" in a.obs else None
        if hto is not None and hto_map:
            # drop hashtags the map marks keep=no (ex-vivo drug arms), before anything
            # downstream sees them
            keepm = np.array([hto_map.get((sid, str(h).upper()), {}).get("keep", "yes")
                              != "no" for h in hto])
            if not keepm.all():
                n0 = a.n_obs
                a = a[keepm].copy()
                cb = [c for c, k in zip(cb, keepm) if k]
                hto = hto[keepm]
                print(f"  [{label}] {sid}: dropped {n0 - a.n_obs:,} cells of excluded "
                      f"hashtag arms -> {a.n_obs:,}")
        a.obs["_cb"] = cb
        standardize_obs(a, label, reg_entry["study"], row,
                        cell_type_orig_col=None)
        if hto is not None:
            # hashed lane: one library holds several donors (or arms) -> donor is per-cell.
            # sample_id stays lane-based so the per-lane contig file still joins.
            recs = [hto_map.get((sid, str(h).upper()), {}) for h in hto]
            names = [r.get("donor") or f"{sid}_{h}" for r, h in zip(recs, hto)]
            a.obs["real_donor"] = names
            a.obs["donor"] = [namespace(label, n) for n in names]
            # per-hashtag clinical fields where the map supplies them (a lane can mix
            # compartments: GSE182861's SZ29 lane is one blood + one skin-tumour sample)
            for col, norm in (("compartment", _norm_compartment),
                              ("tissue", _norm_tissue), ("disease", _norm_disease)):
                vals = [r.get(col) for r in recs]
                if any(vals):
                    a.obs[col] = [norm(v) if v else a.obs[col].iloc[i]
                                  for i, v in enumerate(vals)]
        a.obs["cell_id"] = make_cell_ids(label, [sid] * a.n_obs, cb)
        a.obs["sample_id"] = namespace(label, sid)
        a.obs_names = a.obs["cell_id"].to_numpy()   # unique index (CellRanger h5 barcodes repeat across samples)
        parts.append(_keep_canon(a))
        print(f"  [{label}] {sid}: {a.shape}")
    out = ad.concat(parts, axis=0, join="outer", fill_value=0, index_unique=None)
    return out


def load_eccite_dataset(reg_entry, meta_df):
    """Herrera ECCITE: per-GSM GEX tsv (genes x cells). Donor/compartment from title."""
    import anndata as ad

    raw = Path(reg_entry["raw"])
    parts = []
    for _, row in meta_df.iterrows():
        gsm = str(row["sample_id"])
        hits = sorted(raw.glob(f"{gsm}_*_GEX.tsv.gz"))
        if not hits:
            print(f"  [H] MISSING GEX for {gsm} — skip"); continue
        token = hits[0].name.replace(f"{gsm}_", "").replace("_GEX.tsv.gz", "")  # e.g. SS1_Blood
        donor, comp = token.split("_", 1)
        X, sym, bc = _read_csv_genesxcells_tsv(hits[0])
        sid = f"{donor}_{comp}"
        a = make_anndata(X, sym, bc, sid)
        cb = list(a.obs["_barcode"])
        # disease_state is the authoritative clinical field (SS / leukemic MF / Healthy control);
        # patients.tsv's `disease` column mislabels the SS donors as MF, so prefer disease_state.
        _dstate = str(row.get("disease_state", "") or "").strip()
        _disease = _dstate if _dstate and _dstate.lower() != "nan" else str(row.get("disease", "") or "")
        srow = {
            "donor": donor, "sample_id": sid,
            "disease": _disease,
            "disease_stage": str(row.get("disease_stage", "")) or "NA",
            "sex": str(row.get("sex", "unknown")),
            "tissue": comp, "compartment": comp, "tech": "ECCITE",
        }
        standardize_obs(a, "H", "herrera2021", srow)
        a.obs["cell_id"] = make_cell_ids("H", [sid] * a.n_obs, cb)
        a.obs["sample_id"] = namespace("H", sid)
        a.obs_names = a.obs["cell_id"].to_numpy()
        parts.append(_keep_canon(a))
        print(f"  [H] {sid}: {a.shape}")
    return ad.concat(parts, axis=0, join="outer", fill_value=0, index_unique=None)


def qc_filter(adata, dataset_friendly, run_scrublet=True, verbose=True):
    """Apply per-paper QC (QC_CONFIG) to one standardized AnnData (X = raw counts).

    Adds QC_OBS metrics, applies min/max-genes + %mito cutoffs + min_cells_per_gene=3,
    then per-sample Scrublet (scanpy built-in). li24 is prefiltered -> metrics only.
    Records provenance in ``adata.uns['qc']``.
    """
    import numpy as np
    import scanpy as sc

    cfg = QC_CONFIG.get(dataset_friendly, dict(min_genes=200, max_pct_mito=20, source="default"))
    n_in = adata.n_obs
    X = adata.layers["raw_counts"]
    mt = np.asarray(adata.var_names.str.upper().str.startswith("MT-"))
    tot = np.asarray(X.sum(1)).ravel()
    adata.obs["n_genes"] = np.asarray((X > 0).sum(1)).ravel()
    adata.obs["total_counts"] = tot
    mt_counts = np.asarray(X[:, mt].sum(1)).ravel() if mt.any() else np.zeros(n_in)
    adata.obs["pct_mito"] = np.where(tot > 0, 100.0 * mt_counts / tot, 0.0)
    adata.obs["doublet_score"] = np.nan
    adata.obs["predicted_doublet"] = False

    if cfg.get("prefiltered"):
        adata.uns["qc"] = {"dataset": dataset_friendly, "prefiltered": True,
                           "source": cfg.get("source", ""), "n_in": n_in, "n_kept": n_in}
        if verbose:
            print(f"[QC] {dataset_friendly}: prefiltered -> kept all {n_in}")
        return adata

    keep = adata.obs["n_genes"] >= cfg.get("min_genes", 0)
    if cfg.get("max_genes"):
        keep &= adata.obs["n_genes"] <= cfg["max_genes"]
    if cfg.get("max_pct_mito") is not None:
        keep &= adata.obs["pct_mito"] <= cfg["max_pct_mito"]
    adata = adata[keep.values].copy()
    sc.pp.filter_genes(adata, min_cells=3)

    n_doublet = 0
    if run_scrublet:
        # auto-thresholding needs scikit-image; fall back to a fixed cutoff (Herrera
        # 2021 used 0.2) when skimage is absent so QC still runs without a new dep.
        try:
            import skimage  # noqa: F401
            scrub_kw = {}
        except ImportError:
            scrub_kw = {"threshold": 0.25}
            if verbose:
                print("[QC]   scikit-image absent -> Scrublet fixed threshold 0.25 "
                      "(install scikit-image for auto-thresholding)")
        sid_col = adata.obs["sample_id"].to_numpy()
        dscore = np.full(adata.n_obs, np.nan)
        dflag = np.zeros(adata.n_obs, dtype=bool)
        for sid in pd.unique(sid_col):
            mask = sid_col == sid          # positional boolean — robust to non-unique index
            if mask.sum() < 30:
                continue
            sub = adata[mask].copy()
            try:
                sc.pp.scrublet(sub, random_state=0, **scrub_kw)
                dscore[mask] = sub.obs["doublet_score"].to_numpy()
                dflag[mask] = sub.obs["predicted_doublet"].to_numpy()
            except Exception as e:  # noqa: BLE001 - scrublet can fail on degenerate samples
                if verbose:
                    print(f"[QC]   scrublet skipped for {sid}: {e}")
        adata.obs["doublet_score"] = dscore
        adata.obs["predicted_doublet"] = dflag
        n_doublet = int(dflag.sum())
        adata = adata[~dflag].copy()

    adata.uns["qc"] = {
        "dataset": dataset_friendly, "source": cfg.get("source", ""),
        "thresholds": {k: cfg.get(k) for k in ("min_genes", "max_genes", "max_pct_mito")},
        "scrublet": run_scrublet, "n_in": n_in, "n_kept": adata.n_obs, "n_doublet": n_doublet,
    }
    if verbose:
        print(f"[QC] {dataset_friendly}: {n_in} -> {adata.n_obs} "
              f"({100 * adata.n_obs / max(1, n_in):.1f}%)  doublets={n_doublet}  | {cfg.get('source', '')}")
    return adata


def load_geskin26_hto(reg_entry, meta_df, run_hto=True):
    """B4 (Geskin 2026): hashtag-multiplexed CITE-seq. Per lane: split GEX vs HTO,
    HashSolo-demultiplex, keep singlets (drop Doublet/Negative), set
    ``donor = lane+HTO`` (real patient granularity). ``cell_id``/``sample_id`` stay
    lane-based so the per-lane TCR contigs still join. ADT/HTO features are dropped
    from the expression matrix.
    """
    import anndata as ad
    import numpy as np
    import scanpy as sc
    import scipy.sparse as sp

    raw = Path(reg_entry["raw"])
    parts = []
    for _, row in meta_df.iterrows():
        h5 = raw / row["gex_file"]
        if not h5.exists():
            print(f"  [B4] MISSING {row['gex_file']} — skip"); continue
        lane = str(row["sample_id"])
        full = sc.read_10x_h5(h5, gex_only=False)
        full.var_names_make_unique()
        ft = full.var["feature_types"].astype(str)
        is_hto = np.asarray(full.var_names.str.startswith("HTO"))
        gex = full[:, (ft == "Gene Expression").to_numpy()].copy()
        cb = [_clean_bc(b) for b in gex.obs_names]
        hto_class = np.array(["NA"] * gex.n_obs, dtype=object)

        if run_hto and is_hto.sum() >= 2:
            hto = full[:, is_hto]
            Hm = hto.X.toarray() if sp.issparse(hto.X) else np.asarray(hto.X)
            cols = list(hto.var_names.astype(str))
            for i, c in enumerate(cols):
                gex.obs[c] = Hm[:, i]
            sc.external.pp.hashsolo(gex, cols)
            hto_class = gex.obs["Classification"].astype(str).to_numpy()
            keep = ~np.isin(hto_class, ["Doublet", "Negative"])
            n0 = gex.n_obs
            gex = gex[keep].copy()
            cb = [c for c, k in zip(cb, keep) if k]
            hto_class = hto_class[keep]
            print(f"  [B4] {lane}: {n0} -> {gex.n_obs} singlets "
                  f"({len(set(hto_class))} hashtags)")

        gex.layers["raw_counts"] = gex.X.copy()
        gex.obs = gex.obs[[]].copy()  # drop hashsolo / HTO columns
        gex.obs["dataset"] = DATASET_NAME["B4"]
        gex.obs["study"] = reg_entry["study"]
        gex.obs["donor"] = [namespace("B4", f"{lane}_{h}") for h in hto_class]
        gex.obs["real_donor"] = [f"{lane}_{h}" for h in hto_class]
        gex.obs["sample_id"] = namespace("B4", lane)
        gex.obs["sex"] = str(row.get("sex", "unknown"))
        gex.obs["disease"] = _norm_disease(str(row.get("disease", "")))
        gex.obs["disease_stage"] = str(row.get("disease_stage", "NA")) or "NA"
        gex.obs["compartment"] = _norm_compartment(str(row.get("compartment", "Blood")))
        gex.obs["tissue"] = _norm_tissue(str(row.get("tissue", "PBMC")))
        gex.obs["tech"] = str(row.get("tech", "10x_5p"))
        gex.obs["cell_type_orig"] = ""
        gex.obs["cell_type"] = "Unknown"
        gex.obs["cell_id"] = make_cell_ids("B4", [lane] * gex.n_obs, cb)
        gex.obs_names = gex.obs["cell_id"].to_numpy()
        parts.append(_keep_canon(gex))
    return ad.concat(parts, axis=0, join="outer", fill_value=0, index_unique=None)


def load_buus2025_cite(reg_entry, meta_df):
    """B5 (Buus 2025): CITE-seq + scTCR. Each 10x ``filtered_feature_bc_matrix.h5``
    carries Gene Expression + Antibody Capture. GEX -> expression atlas (X /
    ``raw_counts``); ADT kept COHORT-LOCAL on ``obsm['X_adt']`` (cells x panel) +
    ``uns['adt_panel']`` -- NOT broadcast into the shared ``var``. Only primary ex-vivo
    samples (perturbation timecourse filtered out upstream in meta/samples.tsv).
    """
    import anndata as ad
    import numpy as np
    import scanpy as sc

    raw = Path(reg_entry["raw"])
    parts = []
    for _, row in meta_df.iterrows():
        h5 = raw / row["gex_file"]
        if not h5.exists():
            print(f"  [B5] MISSING {row['gex_file']} — skip"); continue
        full = sc.read_10x_h5(str(h5), gex_only=False)
        full.var_names_make_unique()
        ft = full.var["feature_types"].astype(str) if "feature_types" in full.var else None
        if ft is not None and (ft == "Antibody Capture").any():
            is_adt = (ft == "Antibody Capture").to_numpy()
            gex = full[:, ~is_adt].copy()
            adt = full[:, is_adt].copy()
        else:
            gex, adt = full.copy(), None  # GEX-only deposit -> no ADT (see runtime checkpoint)
        sid = str(row["sample_id"])
        cb = [_clean_bc(b) for b in gex.obs_names]
        gex.layers["raw_counts"] = gex.X.copy()
        gex.obs = gex.obs[[]].copy()
        gex.obs["sample_id"] = sid
        standardize_obs(gex, "B5", reg_entry["study"], row, cell_type_orig_col=None)
        gex.obs["cell_id"] = make_cell_ids("B5", [sid] * gex.n_obs, cb)
        gex.obs["sample_id"] = namespace("B5", sid)
        gex.obs_names = gex.obs["cell_id"].to_numpy()
        a = _keep_canon(gex)
        if adt is not None and adt.n_vars:
            Am = adt.X.toarray() if hasattr(adt.X, "toarray") else np.asarray(adt.X)
            # stash matrix + its OWN protein names; the Buus CITE panel size varies across
            # samples (97-127), so we align by name (not position) after concat.
            a.uns["_adt_names"] = list(adt.var_names.astype(str))
            a.obsm["X_adt"] = Am.astype(np.float32)
        parts.append(a)
        print(f"  [B5] {sid}: GEX {a.shape}" + (f"  ADT {adt.n_vars}" if adt is not None else "  (no ADT)"))
    out = ad.concat(parts, axis=0, join="outer", fill_value=0, index_unique=None)
    # ad.concat drops obsm of inconsistent width across parts -> rebuild X_adt over the UNION
    # protein panel, placed BY NAME (cells of a sample lacking a protein stay NaN there).
    adt_parts = [(p.obs_names.to_numpy(), p.uns["_adt_names"], p.obsm["X_adt"])
                 for p in parts if "X_adt" in p.obsm]
    if adt_parts:
        names, seen = [], set()
        for _, nm, _ in adt_parts:
            for n in nm:
                if n not in seen:
                    seen.add(n); names.append(n)
        col = {n: i for i, n in enumerate(names)}
        pos = {cid: i for i, cid in enumerate(out.obs_names)}
        A = np.full((out.n_obs, len(names)), np.nan, dtype=np.float32)
        for cids, nm, mat in adt_parts:
            A[np.ix_([pos[c] for c in cids], [col[n] for n in nm])] = mat
        out.obsm["X_adt"] = A
        out.uns["adt_panel"] = names
        print(f"  [B5] ADT union panel: {len(names)} proteins across {len(adt_parts)} samples")
    return out


def build_standardized(label, nb_dir, force=False, run_scrublet=True):
    """Top-level: build/cache data/<LABEL>/processed/standardized.h5ad (post-QC)."""
    import scanpy as sc

    reg = dataset_registry(nb_dir)[label]
    # output dir: atlas has no per-label dir -> put under atlas_joint
    if label == "Li2024_atlas":
        out = nb_dir / "data" / "atlas_joint" / "atlas_standardized.h5ad"
    else:
        folder = reg["path"].parents[1] if reg.get("source") == "concat" else reg["raw"].parent
        out = folder / "processed" / "standardized.h5ad"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        # reuse only if the cache is QC-current (has uns['qc']); else it's a stale
        # pre-QC build -> rebuild. Lets a rerun resume without redoing QC'd datasets.
        import h5py
        try:
            with h5py.File(out, "r") as f:
                fresh = "uns" in f and "qc" in f["uns"]
        except Exception:  # noqa: BLE001
            fresh = False
        if fresh:
            print(f"[{label}] cached (QC) -> {out}")
            return out
        print(f"[{label}] cache present but pre-QC -> rebuilding")

    if reg["source"] == "atlas_h5ad":
        a = standardize_atlas(reg)
    elif reg["source"] == "concat":
        a = standardize_concat(label, reg)
    elif reg["source"] == "eccite":
        a = load_eccite_dataset(reg, pd.read_csv(reg["meta"], sep="\t"))
    elif reg["source"] == "cellranger_h5_hto":
        a = load_geskin26_hto(reg, pd.read_csv(reg["meta"], sep="\t"))
    elif reg["source"] == "cite_h5":
        a = load_buus2025_cite(reg, pd.read_csv(reg["meta"], sep="\t"))
    else:
        a = load_persample_dataset(label, reg, pd.read_csv(reg["meta"], sep="\t"))

    a = qc_filter(a, DATASET_NAME.get(label, label), run_scrublet=run_scrublet)

    assert set(CANON_OBS).issubset(a.obs.columns), set(CANON_OBS) - set(a.obs.columns)
    assert a.obs["cell_id"].is_unique, "duplicate cell_id within dataset"
    a.write_h5ad(out)
    print(f"[{label}] wrote {out}  shape={a.shape}  donors={a.obs['donor'].nunique()}")
    return out


# ----------------------------------------------------------------------------
# Step 2: unified TCR clone table (10x VDJ + ECCITE matrix; CDR3 exact key)
# ----------------------------------------------------------------------------
def build_tcr_table(nb_dir, force=False):
    """Build/cache data/atlas_joint/tcr_clones.parquet across all TCR cohorts.

    10x cohorts (D1, D3, D5, B2, B4) via filtered_contig_annotations; H (ECCITE)
    via the clonotypeAB membership matrix. B1/D6 have no TCR; B3 has no CDR3.
    Optional scirpy fuzzy merging is applied in the notebook on top of this.
    """
    out = Path(nb_dir) / "data" / "atlas_joint" / "tcr_clones.parquet"
    if out.exists() and not force:
        print(f"[TCR] cached -> {out}")
        return out
    REG = dataset_registry(Path(nb_dir))
    # derive from the registry instead of a hardcoded list, so adding a TCR cohort is one
    # `has_tcr=True` entry. D16 is has_tcr + in_expression=False: its V(D)J is the ONLY TCR
    # for Li's PT47/PT50 and supersedes the TRUST4 route, while its expression is already
    # in the Li object (see docs/ATLAS_V2_DEDUP_GATE.md).
    labels = [k for k, v in REG.items() if v.get("has_tcr")]
    frames = []
    _c2d_cache: dict[str, dict] = {}

    def _cell_to_donor(lbl):
        """cell_id -> donor from the cohort's standardized obs.

        Needed because a demultiplexed lane's donor is per CELL, and the contig file only
        knows the lane. Read straight from the built cache; if it is absent the caller
        falls back to lane-level grouping (v1 behaviour).
        """
        if lbl not in _c2d_cache:
            m = {}
            try:
                import h5py
                pth = build_standardized(lbl, nb_dir, force=False)
                with h5py.File(pth, "r") as f:
                    cid = [x.decode() for x in f["obs"]["cell_id"][:]] \
                        if not isinstance(f["obs"]["cell_id"], h5py.Group) else None
                    if cid is None:
                        g = f["obs"]["cell_id"]
                        cats = np.array([x.decode() for x in g["categories"][:]])
                        cid = cats[g["codes"][:]]
                    gd = f["obs"]["donor"]
                    if isinstance(gd, h5py.Group):
                        dc = np.array([x.decode() for x in gd["categories"][:]])
                        dv = dc[gd["codes"][:]]
                    else:
                        dv = np.array([x.decode() for x in gd[:]])
                m = dict(zip(cid, dv))
            except Exception as e:                                # noqa: BLE001
                print(f"[{lbl}] cell->donor map unavailable ({e!r}); "
                      f"clone dominance falls back to per-lane")
            _c2d_cache[lbl] = m
        return _c2d_cache[lbl]

    def _li_barcode_index():
        """(donor, clean_barcode) -> Li cell_id, for grafting an external V(D)J deposit
        onto the Li object. Li's cell_id is `Li2024_atlas|<barcode>-0_<donor>_...`, which
        is not the `<sample>|<barcode>` form every other cohort uses."""
        import h5py
        pth = build_standardized("Li2024_atlas", nb_dir, force=False)
        with h5py.File(pth, "r") as f:
            g = f["obs"]["cell_id"]
            if isinstance(g, h5py.Group):
                cats = np.array([x.decode() for x in g["categories"][:]])
                cid = cats[g["codes"][:]]
            else:
                cid = np.array([x.decode() for x in g[:]])
            gd = f["obs"]["real_donor"]
            if isinstance(gd, h5py.Group):
                dc = np.array([x.decode() for x in gd["categories"][:]])
                dv = dc[gd["codes"][:]]
            else:
                dv = np.array([x.decode() for x in gd[:]])
        out = {}
        for c, d in zip(cid, dv):
            bc = _clean_bc(c.split("|", 1)[-1].split("-")[0])
            out[(str(d), bc)] = c
        return out
    for label in labels:
        reg = REG[label]
        raw = Path(reg["raw"])
        meta = pd.read_csv(reg["meta"], sep="\t")
        per_cell, s2d, skipped = [], {}, []
        for _, r in meta.iterrows():
            sid, airr = _tcr_for_sample(label, reg, raw, r)
            if airr is None or airr.empty:
                skipped.append(sid)
                continue
            airr = airr.copy()
            airr["sample_id"] = sid
            per_cell.append(airr)
            _d = r.get("participant_id", r.get("patient_id", r.get("donor", sid)))
            s2d[sid] = namespace(label, str(_d))
        if skipped:
            print(f"[{label}] {len(skipped)} samples contributed no clones: {sorted(skipped)}")
        if not per_cell:
            print(f"[{label}] no TCR found"); continue
        pc = pd.concat(per_cell, ignore_index=True)
        ct = build_clone_table(pc, label, s2d, cell_to_donor=_cell_to_donor(label))

        if not REG[label].get("in_expression"):
            # This cohort supplies V(D)J for cells that live under ANOTHER cohort's
            # namespace. D16/GSE272005 re-deposits the 8 MDA `PT` donors that Li already
            # holds, and is the only source of TCR for PT47/PT50. Without this remap its
            # cell_ids match nothing and the join in concat_joint drops all of it silently.
            idx = _li_barcode_index()
            donors = ct["sample_id"].astype(str).str.split("__").str[-1].str.split("_").str[0]
            bcs = ct["cell_id"].astype(str).str.split("|").str[-1]
            new_ids = [idx.get((d, b)) for d, b in zip(donors, bcs)]
            hit = [i for i, x in enumerate(new_ids) if x]
            print(f"[{label}] grafted onto Li cells: {len(hit):,}/{len(ct):,} contigs "
                  f"matched a Li barcode ({len(set(donors[hit]))} donors)")
            ct = ct.iloc[hit].copy()
            ct["cell_id"] = [new_ids[i] for i in hit]
            ct["dataset"] = DATASET_NAME["Li2024_atlas"]
        frames.append(ct)
        print(f"[{label}] TCR cells={len(ct)}  samples={pc['sample_id'].nunique()}  "
              f"malignant={int(ct['is_malignant'].sum())}")
    allct = pd.concat(frames, ignore_index=True)
    allct = allct.drop_duplicates("cell_id")
    allct.to_parquet(out, index=False)
    print(f"[TCR] wrote {out}  cells={len(allct)}  clones={allct['clone_id'].nunique()}")
    return out


def _airr_from_archive(path: Path):
    """filtered_contig_annotations.csv inside a .tar.gz (B8) or .zip (D16) VDJ bundle."""
    import io
    import tarfile
    import zipfile

    target = "filtered_contig_annotations.csv"
    if path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz":
        with tarfile.open(path, "r:gz") as tf:
            m = next((x for x in tf.getnames() if x.endswith(target)), None)
            if m is None:
                return None
            return airr_from_10x_contig(io.BytesIO(tf.extractfile(m).read()))
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            m = next((x for x in zf.namelist() if x.endswith(target)), None)
            if m is None:
                return None
            return airr_from_10x_contig(io.BytesIO(zf.read(m)))
    return airr_from_10x_contig(path)


def _tcr_for_sample(label, reg, raw, r):
    """Locate + parse the TCR file for one sample -> (sample_id, airr_df)."""
    if label == "D1":
        sid = str(r["sample_id_in_raw"])
        hits = sorted(raw.glob(f"{r['gsm_vdj']}_*_filtered_contig_annotations.csv.gz"))
        return sid, (airr_from_10x_contig(hits[0]) if hits else None)
    if label == "D3":
        sid = f"{r['patient_id']}_{r['tissue']}"
        hits = sorted(raw.glob(f"{r['gsm_vdj']}_*_filtered_contig_annotations.csv.gz"))
        return sid, (airr_from_10x_contig(hits[0]) if hits else None)
    if label in ("D5", "B2", "B4", "B5"):
        sid = str(r["sample_id"])
        cf = str(r.get("contig_file", "") or "").strip()
        path = raw / cf
        return sid, (airr_from_10x_contig(path) if cf and path.exists() else None)
    # v2 cohorts: one `contig_file` column, which may be a plain CSV, a .tar.gz (B8) or a
    # .zip (D16). Generic -- a new cohort needs no branch here, only a samples.tsv column.
    if label in ("B6", "B7", "B8", "D13", "D15", "D16"):
        sid = str(r["sample_id"])
        cf = str(r.get("contig_file", "") or "").strip()
        path = raw / cf
        return sid, (_airr_from_archive(path) if cf and path.exists() else None)
    if label == "H":
        gsm = str(r["sample_id"])
        hits = sorted(raw.glob(f"{gsm}_*_clonotypeAB.tsv.gz"))
        if not hits:
            return gsm, None
        token = hits[0].name.replace(f"{gsm}_", "").replace("_clonotypeAB.tsv.gz", "")
        donor, comp = token.split("_", 1)
        return f"{donor}_{comp}", airr_from_eccite(hits[0])
    return str(r.get("sample_id", "")), None


# ----------------------------------------------------------------------------
# Step 3: dedup + concatenate + join TCR
# ----------------------------------------------------------------------------
def detect_duplicate_cells(standardized_paths: dict, nb_dir=None, shared=None):
    """Report the duplicate samples the barcode gate found, per dataset.

    The previous implementation's docstring promised a cleaned-barcode comparison but the
    body only counted cells by bare `real_donor` for D1/D3 against a 6-name hardcoded list,
    so it could confirm a known overlap but never discover one. The real comparison lives in
    `check_overlap.py`; this now just reports its verdicts against the built caches.
    """
    if nb_dir is None:                       # infer from any standardized path
        anyp = Path(next(iter(standardized_paths.values())))
        nb_dir = next(q for q in anyp.parents if (q / "atlas_join_helpers.py").exists())
    drop_ids, _ = load_dedup_ledger(nb_dir)
    report = {}
    for label in standardized_paths:
        hits = sorted(s_ for s_ in drop_ids if s_.startswith(f"{label}__"))
        if hits:
            report[label] = hits
    return report


def load_dedup_ledger(nb_dir):
    """Read the barcode-gate verdicts -> (drop_sample_ids, sample_id -> patient_key).

    Replaces the bare-string `real_donor.isin(SHARED_BRUNNER)` test. Keys are full
    namespaced sample_ids (`D3__P112_Skin`), so a new cohort that happens to name a donor
    `P112` is unaffected.
    """
    led = Path(nb_dir) / DEDUP_LEDGER
    if not led.exists():
        raise FileNotFoundError(
            f"{led} missing -- run check_overlap.py then make_dedup_decisions.py "
            "before building the atlas (see docs/ATLAS_V2_DEDUP_GATE.md)")
    d = pd.read_csv(led, dtype=str)
    drop = set(d.loc[d["status"] == "DROP", "sample_id"])

    pk_path = Path(nb_dir) / PATIENT_KEY_TABLE
    pk = {}
    if pk_path.exists():
        t = pd.read_csv(pk_path, dtype=str)
        pk = dict(zip(t["sample_id"], t["patient_key"]))
    return drop, pk


def concat_joint(nb_dir, standardized_paths: dict, tcr_parquet, force=False,
                 brunner_keep="D1"):
    """Dedup shared Brunner patients, concat all standardized h5ads (outer join
    on gene symbols), left-join the TCR table. Writes joint_raw.h5ad.
    """
    import anndata as ad
    import scanpy as sc

    out = Path(nb_dir) / "data" / "atlas_joint" / "joint_raw.h5ad"
    if out.exists() and not force:
        print(f"[concat] cached -> {out}")
        return out

    drop_ids, patient_key = load_dedup_ledger(nb_dir)
    parts, dropped, dropped_ids = [], 0, set()
    adt_store, adt_panel = [], None  # cohort-local CITE-seq ADT (e.g. B5)
    for label, p in standardized_paths.items():
        a = sc.read_h5ad(p)
        sid = a.obs["sample_id"].astype(str)
        mask = sid.isin(drop_ids)
        if mask.any():
            n = int(mask.sum())
            hit = sorted(sid[mask].unique())
            dropped += n
            dropped_ids.update(hit)
            a = a[~mask.values].copy()
            print(f"[concat] {label}: dropped {n} cells / {len(hit)} duplicate samples: "
                  f"{', '.join(hit)}")
        if "X_adt" in a.obsm:
            adt_store.append((a.obs["cell_id"].astype(str).to_numpy(), a.obsm["X_adt"]))
            adt_panel = adt_panel or list(a.uns.get("adt_panel", []))
        parts.append(a)
    # The real invariant: nothing that should have been dropped survived. (Checked below,
    # after concat.) A ledger DROP can legitimately match nothing -- it may target a library
    # the cohort never ingested at all: D3's samples.tsv has no P303_Blood row, and D8's five
    # atopic-dermatitis samples were excluded as out of scope. Those are reported, not fatal.
    # What WOULD be fatal is a token drift (D3's raw `P112_HC1` vs its obs `P112_Skin`),
    # which shows up as an unmatched drop for a sample that IS present -- impossible by
    # construction here, since the match is what removes it.
    unmatched = {s_ for s_ in drop_ids
                 if s_.split("__")[0] in standardized_paths} - dropped_ids
    if unmatched:
        print(f"[concat] {len(unmatched)} ledger DROPs matched no cell (library never "
              f"ingested by that cohort): {sorted(unmatched)}")

    joint = ad.concat(parts, axis=0, join="outer", fill_value=0, index_unique=None)
    joint.obs_names = joint.obs["cell_id"].astype(str)
    assert joint.obs_names.is_unique, "duplicate cell_id after concat"
    survived = set(joint.obs["sample_id"].astype(str).unique()) & drop_ids
    assert not survived, f"duplicate samples survived the dedup: {sorted(survived)[:10]}"
    # re-attach cohort-local ADT: ad.concat drops an obsm not on every part, so place
    # the CITE-seq cohort's X_adt into a full (n_obs x panel) array (NaN elsewhere).
    if adt_store and adt_panel:
        A = np.full((joint.n_obs, len(adt_panel)), np.nan, dtype=np.float32)
        pos = {c: i for i, c in enumerate(joint.obs_names)}
        for cids, mat in adt_store:
            A[[pos[c] for c in cids]] = mat
        joint.obsm["X_adt"] = A
        joint.uns["adt_panel"] = adt_panel
        print(f"[concat] attached X_adt: {int((~np.isnan(A[:,0])).sum())} cells x {len(adt_panel)} proteins")
    # friendly dataset names (also fixes any standardized cache built with short codes)
    joint.obs["dataset"] = joint.obs["dataset"].map(lambda d: DATASET_NAME.get(d, d)).astype("category")
    # canonical tissue vocab (also fixes caches built before _norm_tissue)
    joint.obs["tissue"] = joint.obs["tissue"].astype(str).map(_norm_tissue).astype("category")

    # cross-deposit patient identity. `donor` is namespaced per deposit, so one patient
    # published by two deposits (Vienna's MF309 == D1's P76) lands as two donors -- the same
    # defect as v1's D5__MFIVB vs D1__P303. patient_key collapses them; donor is untouched.
    _sid = joint.obs["sample_id"].astype(str)
    joint.obs["patient_key"] = (_sid.map(patient_key)
                                .fillna(joint.obs["donor"].astype(str)).astype("category"))
    _np_ = joint.obs["patient_key"].nunique()
    print(f"[concat] patient_key: {_np_} patients across "
          f"{joint.obs['donor'].nunique()} donor-units")

    # join TCR
    tcr = pd.read_parquet(tcr_parquet).set_index("cell_id")
    for c in TCR_OBS:
        if c in ("clone_size",):
            joint.obs[c] = tcr[c].reindex(joint.obs_names).fillna(0).astype(int).values
        elif c in ("has_tcr", "is_expanded", "is_dominant_clone", "is_malignant"):
            joint.obs[c] = tcr[c].reindex(joint.obs_names).fillna(False).astype(bool).values
        else:
            joint.obs[c] = tcr[c].reindex(joint.obs_names).fillna("").astype(str).values

    if "raw_counts" not in joint.layers:
        joint.layers["raw_counts"] = joint.X.copy()
    joint.write_h5ad(out)
    print(f"[concat] wrote {out}  shape={joint.shape}  "
          f"duplicate cells dropped={dropped} across {len(dropped_ids)} samples")
    print(joint.obs.groupby('dataset', observed=True).size())
    return out


# ----------------------------------------------------------------------------
# Step 3b: merge external per-sample metadata + resolve provisional lineage
# ----------------------------------------------------------------------------
_META_NUMERIC = {"malignant_frac", "tcr_recovery"}


def annotate_from_csv(adata, csv_path):
    """Broadcast per-sample columns from a metadata CSV onto adata.obs by sample_id.

    Adds only columns not already present (skips the per-sample count aggregates).
    Numeric columns (malignant_frac, tcr_recovery) cast to float; rest categorical.
    """
    meta = pd.read_csv(csv_path, dtype=str).drop_duplicates("sample_id").set_index("sample_id")
    skip = set(adata.obs.columns) | {"n_donors", "n_cells", "n_tcr", "n_clones", "n_malignant"}
    add = [c for c in meta.columns if c not in skip]
    sid = adata.obs["sample_id"].astype(str)
    missing = sorted(set(sid.unique()) - set(meta.index))
    if missing:
        print(f"[annotate] WARNING: {len(missing)} obs samples absent from CSV: {missing[:10]}")
    for c in add:
        mapped = sid.map(meta[c])
        if c in _META_NUMERIC:
            adata.obs[c] = pd.to_numeric(mapped, errors="coerce").astype(float)
        else:
            adata.obs[c] = pd.Categorical(mapped.fillna("NA"))
    print(f"[annotate] added {len(add)} columns from {Path(csv_path).name}: {add}")
    return adata, add


# malignant-clone proxy + lineage markers (CD8B absent in atlas -> CD8A carries it)
MALIGNANT_PROXY_LABELS = {"tumor_cell"}
LINEAGE_MARKERS = {
    "CD4": ["CD4"], "CD8": ["CD8A", "CD8B"],
    "gd": ["TRGC1", "TRGC2", "TRDC"], "ab": ["TRAC", "TRBC1", "TRBC2"],
}

# D1 (Chennareddy GSE266862) clinical diagnosis is authoritative -> lineage + entity.
# The expression resolver mis-calls the γδ-MF cases (γδ malignant cells under-express
# TRGC/TRDC, and tiny malignant fractions give unreliable gd ratios), so for D1 we
# take the clinical `subset` from data/D1_chennareddy2025/meta/patients.tsv instead.
D1_SUBSET_LINEAGE = {"Berti": "CD8", "gd_MF": "gamma_delta", "CD4_MF": "CD4", "HC": "not_applicable"}
D1_SUBSET_ENTITY = {"Berti": "CD8_aggressive_epidermotropic_CTCL_Berti",
                    "gd_MF": "MF_gamma_delta", "CD4_MF": "MF_classic"}
# fallback entity from a (re)resolved lineage, for TCR-less expression/inferCNV calls (li24)
LINEAGE_ENTITY = {"CD8": "MF_CD8", "gamma_delta": "MF_gamma_delta", "CD4": "MF_classic"}


def _lineage_scores(sub):
    """Mean log1p library-normalized expression of each marker group over cells."""
    import numpy as np

    X = sub.layers["raw_counts"] if "raw_counts" in sub.layers else sub.X
    tot = np.asarray(X.sum(1)).ravel().astype(float)
    tot[tot == 0] = 1.0
    out = {}
    for k, genes in LINEAGE_MARKERS.items():
        gi = [sub.var_names.get_loc(g) for g in genes if g in sub.var_names]
        if not gi:
            out[k] = 0.0
            continue
        v = np.asarray(X[:, gi].sum(1)).ravel() / tot * 1e4
        out[k] = float(np.mean(np.log1p(v)))
    return out


def resolve_lineage(adata, min_cells=20, gd_frac_thresh=0.5, clinical_d1_tsv=None):
    """Confirm/overwrite provisional lineage (rows where ``lineage_resolve`` != 'no').

    Resolution source per flagged sample, in priority order:

    1. **D1 clinical** (``clinical_d1_tsv`` = patients.tsv): the Chennareddy `subset`
       diagnosis is authoritative (Berti→CD8, gd_MF→γδ, CD4_MF→CD4). Expression scores
       are still recorded for audit and a ``lineage_disagree`` flag is set where the
       expression call would have differed.
    2. **Already literature-resolved** (orig ``lineage_resolve`` startswith "resolved_",
       e.g. Liu2022): keep the curated lineage, sync entity, no re-call.
    3. **Expression / inferCNV** (the uncertain "yes_*" li24 PT* samples, TCR-less):
       γδ if gd/(gd+ab) ≥ thresh, else CD8 if CD8>CD4 else CD4. Marked
       ``needs_infercnv_verification`` since the malignant label is not TCR-backed.
    4. **Unresolvable** (no malignant cells, not covered by 1): left unchanged,
       ``lineage_resolve`` = "unresolved".

    Malignant clone = ``is_malignant`` cells, or ``cell_type_orig`` in
    MALIGNANT_PROXY_LABELS (atlas `tumor_cell`). The 'no' rows keep their
    high-confidence lineage. ``entity`` is synced to the resolved lineage whenever the
    lineage changed or the current entity is a "*_RESOLVE" placeholder (curated
    non-placeholder entities — Sezary, MF/SS_leukemic — are preserved). Writes
    lineage_orig, lineage, entity, lineage_resolve, lineage_method, lineage_disagree,
    lineage_score_*; returns a per-sample report.
    """
    import numpy as np

    if "lineage" not in adata.obs:
        raise KeyError("run annotate_from_csv first (no 'lineage' column)")
    obs = adata.obs
    obs["lineage_orig"] = obs["lineage"].astype(str)
    lineage = obs["lineage"].astype(str).to_numpy().copy()
    entity = obs["entity"].astype(str).to_numpy().copy()
    resolve = obs["lineage_resolve"].astype(str).to_numpy().copy()
    method = np.array(["high_confidence"] * adata.n_obs, dtype=object)
    disagree = np.zeros(adata.n_obs, dtype=bool)
    for k in ("cd4", "cd8", "gd", "ab"):
        obs[f"lineage_score_{k}"] = np.nan

    d1_subset = {}
    if clinical_d1_tsv is not None:
        t = pd.read_csv(clinical_d1_tsv, sep="\t", dtype=str)
        d1_subset = dict(zip(t["participant_id"], t["subset"]))

    is_mal = obs["is_malignant"].astype(bool).to_numpy()
    is_proxy = obs["cell_type_orig"].astype(str).isin(MALIGNANT_PROXY_LABELS).to_numpy()
    mal_mask = is_mal | is_proxy
    flag = obs["lineage_resolve"].astype(str).to_numpy()
    sid = obs["sample_id"].astype(str).to_numpy()

    def _sync_entity(cur, final_lin, orig_lin):
        if "RESOLVE" in cur or final_lin != orig_lin:
            return LINEAGE_ENTITY.get(final_lin, cur)
        return cur

    rows = []
    for s in pd.unique(sid[flag != "no"]):
        smask = sid == s
        orig_flag = resolve[smask][0]
        orig_lin = lineage[smask][0]
        mmask = smask & mal_mask
        n = int(mmask.sum())

        # expression call (recorded for every flagged sample with enough malignant cells)
        expr_call, scd, gd_frac = None, None, float("nan")
        if n >= min_cells:
            scd = _lineage_scores(adata[mmask])
            gd_frac = scd["gd"] / (scd["gd"] + scd["ab"] + 1e-9)
            expr_call = "gamma_delta" if gd_frac >= gd_frac_thresh else ("CD8" if scd["CD8"] > scd["CD4"] else "CD4")
            obs.loc[smask, "lineage_score_cd4"] = scd["CD4"]
            obs.loc[smask, "lineage_score_cd8"] = scd["CD8"]
            obs.loc[smask, "lineage_score_gd"] = scd["gd"]
            obs.loc[smask, "lineage_score_ab"] = scd["ab"]

        is_d1 = s.startswith("D1__")
        participant = s.split("__", 1)[1] if "__" in s else s
        dis = False
        if is_d1 and participant in d1_subset:                       # (1) clinical
            subset = d1_subset[participant]
            final = D1_SUBSET_LINEAGE.get(subset, orig_lin)
            ent = D1_SUBSET_ENTITY.get(subset, _sync_entity(entity[smask][0], final, orig_lin))
            meth = f"resolved_clinical(patients.tsv:{subset})"
            res = "resolved"
            dis = expr_call is not None and expr_call != final
        elif orig_flag.startswith("resolved_"):                       # (2) literature
            final = orig_lin
            ent = _sync_entity(entity[smask][0], final, orig_lin)
            meth = orig_flag
            res = "resolved"
        elif expr_call is not None:                                   # (3) expression/inferCNV
            final = expr_call
            ent = _sync_entity(entity[smask][0], final, orig_lin)
            meth = f"resolved_from_malignant_clone(n={n});needs_infercnv_verification"
            res = "resolved"
        else:                                                         # (4) unresolvable
            final = orig_lin
            ent = entity[smask][0]
            meth = "unresolved(no_malignant_cells)"
            res = "unresolved"

        lineage[smask] = final
        entity[smask] = ent
        method[smask] = meth
        resolve[smask] = res
        disagree[smask] = dis
        rows.append({"sample_id": s, "n_mal": n, "expr_call": expr_call or "NA",
                     "clinical": d1_subset.get(participant, "") if is_d1 else "",
                     "final": final, "entity": ent, "lineage_orig": orig_lin,
                     "disagree": dis, "resolve": res,
                     "gd_frac": round(gd_frac, 3) if gd_frac == gd_frac else None,
                     **({k: round(v, 3) for k, v in scd.items()} if scd else {})})
    obs["lineage"] = pd.Categorical(lineage)
    obs["entity"] = pd.Categorical(entity)
    obs["lineage_resolve"] = pd.Categorical(resolve)
    obs["lineage_method"] = pd.Categorical(method)
    obs["lineage_disagree"] = disagree
    return pd.DataFrame(rows)
