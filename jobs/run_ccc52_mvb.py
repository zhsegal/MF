#!/usr/bin/env python
"""nb52 - differential cell-cell communication of malignant vs benign CD4 T cells (MF/SS atlas).

Estimand. Within a patient, malignant (M) and benign (B) CD4 cells face the same partner cells,
so for any multiplicative ligand-receptor score the partner term cancels in the M-vs-B contrast:
the differential interaction IS a paired differential expression of ligand and receptor genes on
the CD4 side. Partner cells act only as a gate -- is the cognate gene expressed in that cell
type. The CD4-side result is the inference; the partner assignment is descriptive. No ambient
correction, no LIANA scoring, no MrVI/latent DE, no permutation tests.

Skin and blood are analysed completely separately.

Stages (one per bsub, chained with AFTER=<jobid> ./run_ccc52_mvb.sh <stage>):

  lr    mrvi_env.  Rebuild the filtered LR resource: LIANA `consensus` intersected with
        support >= 6 of the 15 human resources -> 1,422 interactions / 1,043 genes. The repo has
        no such file on disk; `helpers/ccc_panel.py:12` is the only trace of the spec.
  prep  Cohort, arms, the blood benign-arm cleanup, partner types, and the row-label vectors
        that the streaming pass consumes.
  pass  ONE streamed pass over data/atlas_joint/joint_annotated.h5ad (112 GB, layers/raw_counts)
        that accumulates every count sum and detection count this analysis needs.
  fit   pydeseq2 paired DE, partner gating, tables, figures, report.

Why joint_annotated and not the compartment objects: blood_T_annotated.h5ad is a 10,000-HVG
subset (590 of 1,043 LR genes; CXADR and IL22RA1 absent), and neither compartment object holds
non-T partner cells. joint_annotated is the only full-gene source for both.

Nothing under data/ is written. All output goes to reports/ccc52_mvb/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

NB_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NB_DIR / "helpers"))

ATLAS = NB_DIR / "data" / "atlas_joint"
OUT = NB_DIR / "reports" / "ccc52_mvb"

CONFIG = dict(
    # ---- sources -----------------------------------------------------------------
    JOINT_H5AD=ATLAS / "joint_annotated.h5ad",          # 2,157,693 x 42,347, layers/raw_counts
    JOINT_LAYER="raw_counts",
    OBS_PARQUET=ATLAS / "atlas_obs_full_v2.parquet",
    MAL_PARQUET={"skin": ATLAS / "skin_T_malignancy_cluster_v1.parquet",
                 "blood": ATLAS / "blood_T_malignancy_cluster_v1.parquet"},
    BLOOD_T_H5AD=ATLAS / "blood_T_annotated.h5ad",      # 10k HVG; used only for marker scoring
    BLOOD_CLUSTERS=ATLAS / "blood_T_clusters_v1.parquet",   # nb32b per-donor Leiden, res 1.0
    BLOOD_SUBTYPES=ATLAS / "blood_subtypes.csv",            # nb31b non-T fine labels
    BLOOD_CT_FIX=ATLAS / "blood_T_celltype_corrections_v1.csv",   # nb32b CD8 -> NK, 12 clusters
    # ---- arms --------------------------------------------------------------------
    MAL_CALL="mal_cluster",          # the repo HEADLINE since 2026-10-22 (helpers/malignancy.py)
    CD4_LABEL="CD4",                 # cell_type_final; CD4_Treg is a separate label in skin
    TREG_LABEL="CD4_Treg",
    # ---- cohort ------------------------------------------------------------------
    DROP_DATASETS={"blood": ["geskin26"]},   # 100% ex vivo 2-day culture -> not in-vivo biology
    SKIN_NON_CD4_PHENOTYPES=("CD8", "gd", "DN"),
    SKIN_NON_CD4_DX=("CD8_aggr_CTCL", "pcAECTCL"),
    SKIN_OFF_SPECTRUM_DX=("LyP", "SPP", "LPP"),
    # ---- blood benign-arm cleanup (no Treg / naive label exists in blood) --------
    BLOOD_TREG_MARKERS=["FOXP3"],
    BLOOD_NAIVE_UP=["CCR7", "SELL", "TCF7", "LEF1"],
    BLOOD_NAIVE_DOWN=["S100A4"],
    BLOOD_TREG_DETECT=0.15,      # cluster FOXP3 detection fraction above this -> Treg cluster
    # A naive cluster must be HIGH on all four naive markers in absolute terms and below the
    # across-cluster mean for S100A4. A purely relative (z) rule was tried first and rejected:
    # S100A4 detection saturates (median 0.89), so a low-depth cluster where every marker is low
    # scored as naive -- it flagged 7 clusters whose own naive markers were under 0.30.
    BLOOD_NAIVE_MIN={"CCR7": 0.50, "SELL": 0.55, "TCF7": 0.50, "LEF1": 0.45},
    BLOOD_NAIVE_S100A4_Z=0.0,    # S100A4 z must be below this
    BLOOD_MIN_CLUSTER=20,        # clusters with fewer benign CD4 cells are left alone
    # ---- eligibility -------------------------------------------------------------
    MIN_CELLS_PER_ARM=50,
    MIN_PARTNER_CELLS=200,
    PARTNER_DROP={"skin": ["Unk"], "blood": ["Unk"]},
    BLOOD_SUBTYPE_DROP=("Ambient_soup", "LowQC", "Doublet"),
    # ---- statistics ---------------------------------------------------------------
    MIN_COUNTS=10,               # a gene needs >= MIN_COUNTS in >= MIN_FRAC_PB of pseudobulks
    MIN_FRAC_PB=0.25,
    PADJ=0.05,
    LFC=1.0,
    MIN_DETECT=0.10,             # detected in >= 10% of cells of the higher arm
    PARTNER_DETECT=0.10,         # cognate gene in >= 10% of cells of the partner type
    LR_SUPPORT=6,                # >= 6 of the 15 human LIANA resources
    SEED=0,
    CHUNK=20000,                 # rows per streamed chunk
)

PRIOR_AXES = [
    # (CD4-side gene, [cognate genes named in the brief], note)
    ("CSF2",    [],                     "GM-CSF, myeloid recruitment"),
    ("CCR4",    ["CCL17", "CCL22"],     "mogamulizumab target; skin-homing"),
    ("CD70",    ["CD27"],               "the project's strongest existing LIANA hit"),
    ("CD27",    ["CD70"],               "receptor side of the same axis"),
    ("JAML",    ["CXADR"],              "epidermal gd/T-keratinocyte co-stimulation"),
    ("CXCR4",   ["CXCL12"],             "retention / egress"),
    ("SPN",     ["SIGLEC1"],            "CD43-sialoadhesin, myeloid contact"),
    ("CD28",    ["CD80", "CD86"],       "canonical co-stimulation"),
    ("CTLA4",   ["CD80", "CD86"],       "co-inhibition"),
    ("IL22",    ["IL22RA1"],            "epithelial-acting cytokine"),
    ("KIR3DL2", ["HLA-A"],              "lacutamab target; Sezary marker"),
    ("TOX",     [],                     "POSITIVE CONTROL for the malignant contrast, not an LR gene"),
]

TEAL = "#0F766E"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _subunits(x):
    """LIANA stores complexes underscore-joined ('A_B'); same convention as ccc_utils:195."""
    return [s for s in str(x).split("_") if s]


# ======================================================================================
# stage: lr
# ======================================================================================
def stage_lr(args):
    """Rebuild the filtered resource. Needs liana -> mrvi_env."""
    import liana as li
    from collections import Counter

    OUT.mkdir(parents=True, exist_ok=True)
    human = [r for r in li.rs.show_resources() if r not in ("consensus", "mouseconsensus")]
    log(f"{len(human)} human resources: {human}")
    assert len(human) == 15, f"expected 15 human resources, got {len(human)}"

    cnt = Counter()
    for r in human:
        d = li.rs.select_resource(r)[["ligand", "receptor"]].drop_duplicates()
        for lig, rec in d.itertuples(index=False):
            cnt[(lig, rec)] += 1

    con = li.rs.select_resource("consensus")[["ligand", "receptor"]].drop_duplicates()
    con = con.reset_index(drop=True)
    con["support"] = [cnt.get((l, r), 0) for l, r in con.itertuples(index=False)]
    con.to_csv(OUT / "lr_resource_consensus_full.csv", index=False)
    log(f"unfiltered consensus: {len(con):,} interactions")

    keep = con[con["support"] >= CONFIG["LR_SUPPORT"]].reset_index(drop=True)
    genes = sorted({g for col in ("ligand", "receptor") for v in keep[col] for g in _subunits(v)})
    log(f"filtered (support >= {CONFIG['LR_SUPPORT']}): {len(keep):,} interactions, "
        f"{len(genes):,} unique genes")
    assert len(keep) == 1422, f"resource size changed: {len(keep)} != 1422 (liana version drift?)"
    assert len(genes) == 1043, f"resource gene count changed: {len(genes)} != 1043"

    keep.to_csv(OUT / "lr_resource_1422.csv", index=False)
    (OUT / "lr_genes.json").write_text(json.dumps(genes))
    log(f"wrote {OUT/'lr_resource_1422.csv'}")


# ======================================================================================
# stage: prep
# ======================================================================================
def _read_h5ad_str(h5, key):
    import h5py
    d = h5[key]
    if isinstance(d, h5py.Group):                     # categorical
        cats = np.array([x.decode() if isinstance(x, bytes) else x for x in d["categories"][:]])
        codes = d["codes"][:]
        out = np.where(codes < 0, "", cats[np.clip(codes, 0, None)])
        return out
    v = d[:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in v])


def _blood_marker_detection(cell_ids_wanted):
    """Detection (count > 0) of the Treg/naive markers per blood T cell, from the 10k-HVG object.

    Returns a frame indexed by cell_id with one boolean column per marker. The blood object is
    only 16.5 GB and we touch six gene columns, so this is a cheap second pass -- it cannot be
    folded into the main pass over joint_annotated, because the labels that pass consumes are
    what this computes.
    """
    import h5py
    from scipy import sparse

    markers = (CONFIG["BLOOD_TREG_MARKERS"] + CONFIG["BLOOD_NAIVE_UP"]
               + CONFIG["BLOOD_NAIVE_DOWN"])
    with h5py.File(CONFIG["BLOOD_T_H5AD"], "r") as f:
        var = _read_h5ad_str(f["var"], f["var"].attrs.get("_index", "_index"))
        cid = _read_h5ad_str(f["obs"], "cell_id")
        gidx = {g: int(np.flatnonzero(var == g)[0]) for g in markers if (var == g).any()}
        missing = [g for g in markers if g not in gidx]
        if missing:
            raise RuntimeError(f"markers absent from blood object: {missing}")
        cols = np.array([gidx[g] for g in markers])
        want = pd.Index(cid).isin(pd.Index(cell_ids_wanted))
        L = f["layers"]["counts"]
        n = len(cid)
        det = np.zeros((n, len(markers)), dtype=bool)
        pos = {c: i for i, c in enumerate(cols)}
        indptr_all = L["indptr"][:]
        for a in range(0, n, CONFIG["CHUNK"]):
            b = min(a + CONFIG["CHUNK"], n)
            if not want[a:b].any():
                continue
            lo, hi = int(indptr_all[a]), int(indptr_all[b])
            ind = L["indices"][lo:hi]
            dat = L["data"][lo:hi]
            rows = np.repeat(np.arange(a, b), np.diff(indptr_all[a:b + 1]))
            sel = np.isin(ind, cols) & (dat > 0)
            if sel.any():
                rr = rows[sel]
                cc = np.array([pos[int(x)] for x in ind[sel]])
                det[rr, cc] = True
    return pd.DataFrame(det, index=pd.Index(cid, name="cell_id"), columns=markers)


def _clean_blood_benign(obs_cd4):
    """Drop Treg-like and naive-like clusters from the blood benign arm.

    Blood carries no Treg and no naive-CD4 label: `cell_type_T` is {CD4, CD8, NK} and
    `cell_type_final` has one flat `CD4` over 554,369 cells. The brief's fallback therefore
    fires. Rather than run a fresh Leiden on the 10-dimensional MrVI u latent, this reuses the
    per-donor clusters nb32b already computed (`blood_T_clusters_v1.parquet`, res 1.0, keyed
    `<donor>__<k>`), which have the advantage of being within-donor and so free of batch
    structure. Clusters are scored on marker DETECTION fractions among their benign CD4 cells.
    """
    blood_benign = obs_cd4[(obs_cd4["tissue2"] == "blood") & (obs_cd4["arm"] == "B")]
    log(f"blood benign CD4 before cleanup: {len(blood_benign):,}")

    clus = pd.read_parquet(CONFIG["BLOOD_CLUSTERS"], columns=["cell_id", "cluster"])
    clus = clus.set_index("cell_id")["cluster"]
    cl = clus.reindex(blood_benign.index)

    det = _blood_marker_detection(blood_benign.index)
    det = det.reindex(blood_benign.index)

    tab = det.groupby(cl.values).mean()
    tab["n_cells"] = cl.value_counts().reindex(tab.index).astype(int)
    big = tab["n_cells"] >= CONFIG["BLOOD_MIN_CLUSTER"]

    def _z(col):
        v = tab[col].where(big)
        return (tab[col] - v.mean()) / (v.std(ddof=0) + 1e-9)

    tab["naive_z"] = (np.mean([_z(g) for g in CONFIG["BLOOD_NAIVE_UP"]], axis=0)
                      - np.mean([_z(g) for g in CONFIG["BLOOD_NAIVE_DOWN"]], axis=0))
    tab["s100a4_z"] = _z("S100A4")
    tab["is_treg"] = big & (tab[CONFIG["BLOOD_TREG_MARKERS"][0]] >= CONFIG["BLOOD_TREG_DETECT"])
    naive = big & (tab["s100a4_z"] < CONFIG["BLOOD_NAIVE_S100A4_Z"])
    for g, thr in CONFIG["BLOOD_NAIVE_MIN"].items():
        naive &= tab[g] >= thr
    tab["is_naive"] = naive & ~tab["is_treg"]
    tab["dropped"] = tab["is_treg"] | tab["is_naive"]
    tab.index.name = "cluster"
    tab.reset_index().to_csv(OUT / "blood_benign_subclusters.csv", index=False)

    sets = {}
    for kind, col in (("CD4_Treg_like", "is_treg"), ("CD4_naive_like", "is_naive")):
        cls = set(tab.index[tab[col]])
        sets[kind] = set(blood_benign.index[cl.isin(cls).fillna(False).values])
    log(f"blood benign clusters: {len(tab)} total, "
        f"{int(tab['is_treg'].sum())} Treg-like ({len(sets['CD4_Treg_like']):,} cells), "
        f"{int(tab['is_naive'].sum())} naive-like ({len(sets['CD4_naive_like']):,} cells)")
    return sets


def stage_prep(args):
    import polars as pl

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(CONFIG["SEED"])

    cols = ["cell_id", "dataset", "study", "donor", "sample_id", "patient_key",
            "clin_person_key", "compartment", "tech", "disease", "clin_dx", "clin_phenotype",
            "cell_type_final"]
    obs = (pl.read_parquet(CONFIG["OBS_PARQUET"], columns=cols)
             .with_columns([pl.col(c).cast(pl.Utf8) for c in cols])
             .to_pandas().set_index("cell_id"))
    log(f"obs: {obs.shape}")
    obs["tissue2"] = obs["compartment"].map({"Skin": "skin", "Blood": "blood"})

    # ---- malignancy call -----------------------------------------------------------
    mal = pd.concat([pd.read_parquet(p, columns=["cell_id", CONFIG["MAL_CALL"]])
                     for p in CONFIG["MAL_PARQUET"].values()]).set_index("cell_id")
    obs["mal"] = mal[CONFIG["MAL_CALL"]].reindex(obs.index)

    # ---- unit key: the repo's curated person key, falling back to patient_key -------
    obs["unit"] = (obs["clin_person_key"].replace({"": np.nan, "nan": np.nan, "None": np.nan})
                   .fillna(obs["patient_key"]).fillna(obs["donor"]))

    # ---- arms ----------------------------------------------------------------------
    is_cd4 = obs["cell_type_final"].eq(CONFIG["CD4_LABEL"]) & obs["tissue2"].notna()
    arm = pd.Series(pd.NA, index=obs.index, dtype="object")
    arm[is_cd4 & obs["mal"].eq(True)] = "M"
    arm[is_cd4 & obs["mal"].eq(False)] = "B"
    obs["arm"] = arm

    # ---- cohort rules (helpers/target_cohort.py), minus the "same person" rule, which
    #      patient-level aggregation already handles ---------------------------------
    drop = pd.Series(pd.NA, index=obs.index, dtype="object")
    skin = obs["tissue2"].eq("skin")
    drop[skin & (obs["clin_phenotype"].isin(CONFIG["SKIN_NON_CD4_PHENOTYPES"])
                 | obs["clin_dx"].isin(CONFIG["SKIN_NON_CD4_DX"]))] = "malignant clone is not CD4"
    drop[skin & obs["clin_dx"].isin(CONFIG["SKIN_OFF_SPECTRUM_DX"])] = "diagnosis outside MF/SS"
    for t, ds in CONFIG["DROP_DATASETS"].items():
        drop[obs["tissue2"].eq(t) & obs["dataset"].isin(ds)] = "ex vivo culture cohort (geskin2026)"
    obs["drop_reason"] = drop
    obs["cohort_keep"] = drop.isna()

    excl = (obs[~obs["cohort_keep"]].groupby(["tissue2", "unit", "drop_reason"], dropna=False)
            .size().rename("n_cells").reset_index())
    excl.to_csv(OUT / "cohort_excluded.csv", index=False)
    log(f"cohort-excluded units: {excl.groupby('tissue2')['unit'].nunique().to_dict()}")

    # ---- blood benign-arm cleanup ---------------------------------------------------
    cd4 = obs[obs["arm"].notna() & obs["cohort_keep"]]
    blood_drop = _clean_blood_benign(cd4)
    obs["blood_cd4_sub"] = pd.NA
    for kind, cells in blood_drop.items():
        m = obs.index.isin(cells)
        obs.loc[m, "arm"] = pd.NA
        obs.loc[m, "drop_reason"] = f"blood benign {kind} cluster"
        obs.loc[m, "blood_cd4_sub"] = kind      # kept as a partner type, not discarded entirely

    # skin: CD4_Treg is a separate cell_type_final label, so it can never reach either arm
    assert not (obs["cell_type_final"].eq(CONFIG["TREG_LABEL"]) & obs["arm"].notna()).any(), \
        "a CD4_Treg-labelled cell reached an arm"

    # ---- cell counts per unit x arm, eligibility ------------------------------------
    cd4 = obs[obs["arm"].notna() & obs["cohort_keep"]]
    cnt = (cd4.groupby(["tissue2", "unit", "arm"]).size().unstack("arm")
           .reindex(columns=["M", "B"]).fillna(0).astype(int))
    meta = cd4.groupby(["tissue2", "unit"]).agg(
        study=("study", lambda s: "|".join(sorted(set(s)))),
        dataset=("dataset", lambda s: "|".join(sorted(set(s)))),
        tech=("tech", lambda s: "|".join(sorted(set(s)))),
        dx=("clin_dx", "first"),
        n_samples=("sample_id", "nunique"),
        n_donor_keys=("donor", "nunique"))
    cnt = cnt.join(meta)
    cnt["eligible"] = (cnt["M"] >= CONFIG["MIN_CELLS_PER_ARM"]) & (cnt["B"] >= CONFIG["MIN_CELLS_PER_ARM"])
    cnt = cnt.reset_index().rename(columns={"tissue2": "tissue", "M": "n_malignant", "B": "n_benign"})
    cnt.to_csv(OUT / "cell_counts_donor_arm.csv", index=False)
    elig = cnt[cnt["eligible"]]
    log("eligible units: " + str(elig.groupby("tissue").size().to_dict()))
    assert (elig["n_malignant"] >= CONFIG["MIN_CELLS_PER_ARM"]).all()
    assert (elig["n_benign"] >= CONFIG["MIN_CELLS_PER_ARM"]).all()

    keep_units = {t: set(g["unit"]) for t, g in elig.groupby("tissue")}

    # ---- partner types ---------------------------------------------------------------
    part = obs["cell_type_final"].copy()
    fix = pd.read_csv(CONFIG["BLOOD_CT_FIX"])            # nb32b: 12 clusters CD8 -> NK
    part.loc[part.index.isin(fix["cell_id"])] = fix.set_index("cell_id")["cell_type_T"].reindex(
        part.index[part.index.isin(fix["cell_id"])]).values
    sub = pd.read_csv(CONFIG["BLOOD_SUBTYPES"], usecols=["cell_id", "subtype_fine"])
    bad = set(sub.loc[sub["subtype_fine"].isin(CONFIG["BLOOD_SUBTYPE_DROP"]), "cell_id"])
    part.loc[part.index.isin(bad)] = pd.NA
    part.loc[part.eq(CONFIG["CD4_LABEL"])] = pd.NA                 # replaced by the arms below
    part.loc[obs["arm"].eq("M")] = "CD4_malignant"
    part.loc[obs["arm"].eq("B")] = "CD4_benign"
    # Blood has no Treg label, so the clusters the benign-arm cleanup removed are the only Tregs
    # the compartment can offer. They leave the ARM but stay available as partner types -- the
    # brief asks explicitly for Tregs among the partners.
    sub_lab = obs["blood_cd4_sub"]
    part.loc[sub_lab.notna()] = sub_lab[sub_lab.notna()]
    for t, bad_t in CONFIG["PARTNER_DROP"].items():
        part.loc[obs["tissue2"].eq(t) & part.isin(bad_t)] = pd.NA
    in_cohort = pd.Series(
        [isinstance(t, str) and u in keep_units.get(t, set())
         for t, u in zip(obs["tissue2"], obs["unit"])], index=obs.index)
    part[~in_cohort] = pd.NA
    obs["partner"] = part

    ptab = (obs[obs["partner"].notna()].groupby(["tissue2", "partner"]).size()
            .rename("n_cells").reset_index().rename(columns={"tissue2": "tissue"}))
    ptab["homotypic"] = ptab["partner"].eq("CD4_malignant")
    ptab["keep"] = ptab["n_cells"] >= CONFIG["MIN_PARTNER_CELLS"]
    ptab.to_csv(OUT / "partner_types.csv", index=False)
    log("partner types kept: " + str(ptab[ptab["keep"]].groupby("tissue").size().to_dict()))
    keep_part = set(zip(ptab.loc[ptab["keep"], "tissue"], ptab.loc[ptab["keep"], "partner"]))

    # ---- row-label vectors, aligned to the joint object's own cell order --------------
    import h5py
    with h5py.File(CONFIG["JOINT_H5AD"], "r") as f:
        joint_ids = _read_h5ad_str(f["obs"], "cell_id")
    log(f"joint cells: {len(joint_ids):,}")

    units_long = pd.DataFrame(
        [{"tissue": r["tissue"], "unit": r["unit"], "arm": a}
         for _, r in elig.iterrows() for a in ("M", "B")])
    units_long["unit_idx"] = np.arange(len(units_long))
    units_long = units_long.merge(
        cnt[["tissue", "unit", "study", "dataset", "tech", "dx", "n_samples", "n_donor_keys"]],
        on=["tissue", "unit"], how="left")
    units_long.to_csv(OUT / "units.csv", index=False)

    partners = ptab[ptab["keep"]].reset_index(drop=True)[["tissue", "partner", "n_cells", "homotypic"]]
    partners["partner_idx"] = np.arange(len(partners))
    partners.to_csv(OUT / "partners.csv", index=False)

    ukey = pd.Series(units_long["unit_idx"].values,
                     index=pd.MultiIndex.from_frame(units_long[["tissue", "unit", "arm"]]))
    pkey = pd.Series(partners["partner_idx"].values,
                     index=pd.MultiIndex.from_frame(partners[["tissue", "partner"]]))

    o = obs.reindex(pd.Index(joint_ids, name="cell_id"))
    keep_cell = o["cohort_keep"].fillna(False).to_numpy().astype(bool)

    def _lookup(keys, table):
        """Map a list of tuples onto an index -> integer table; -1 where absent."""
        d = {k: int(v) for k, v in table.items()}
        return np.fromiter((d.get(k, -1) for k in keys), dtype=np.int32, count=len(o))

    row_unit = _lookup(list(zip(o["tissue2"], o["unit"], o["arm"])), ukey)
    row_unit[~keep_cell] = -1
    row_partner = _lookup(list(zip(o["tissue2"], o["partner"])), pkey)
    row_partner[~keep_cell] = -1

    techs = sorted({f"{t}|{x}" for t, x, r in zip(o["tissue2"], o["tech"], row_unit) if r >= 0})
    tmap = {k: i for i, k in enumerate(techs)}
    row_tech = np.array([tmap.get(f"{t}|{x}", -1) if r >= 0 else -1
                         for t, x, r in zip(o["tissue2"], o["tech"], row_unit)], dtype=np.int32)

    log(f"labelled rows: unit {int((row_unit >= 0).sum()):,}  "
        f"partner {int((row_partner >= 0).sum()):,}  techs {techs}")
    np.savez_compressed(OUT / "row_labels.npz", row_unit=row_unit, row_partner=row_partner,
                        row_tech=row_tech, techs=np.array(techs), n_unit=len(units_long),
                        n_partner=len(partners))

    # the count table the pass asserts against
    chk = (pd.Series(row_unit[row_unit >= 0]).value_counts().sort_index()
           .rename("n_cells").rename_axis("unit_idx"))
    chk.to_csv(OUT / "_expected_unit_cells.csv")
    log("prep done")


# ======================================================================================
# stage: pass
# ======================================================================================
def stage_pass(args):
    """One streamed pass over joint_annotated.h5ad.

    Everything downstream needs exactly four things per group: the summed raw counts, the number
    of cells in which each gene is detected, the cell count, and (for the chemistry split) the
    same detection counts per tissue x chemistry. All four are linear in the cells, so one pass
    over the CSR with a 0/1 group matrix per chunk produces all of them without materialising
    anything. The accumulators are written once, at the end -- the `short` queue is preemptable.
    """
    import h5py
    from scipy import sparse

    lab = np.load(OUT / "row_labels.npz", allow_pickle=False)
    row_unit, row_partner, row_tech = lab["row_unit"], lab["row_partner"], lab["row_tech"]
    n_unit, n_partner = int(lab["n_unit"]), int(lab["n_partner"])
    n_tech = int(row_tech.max()) + 1

    with h5py.File(CONFIG["JOINT_H5AD"], "r") as f:
        L = f["layers"][CONFIG["JOINT_LAYER"]]
        n, G = (int(x) for x in f["X"].attrs["shape"])
        var = _read_h5ad_str(f["var"], f["var"].attrs.get("_index", "_index"))
        assert len(row_unit) == n, f"row labels ({len(row_unit)}) != joint cells ({n})"
        log(f"streaming {n:,} x {G:,} from {CONFIG['JOINT_H5AD'].name}, layer "
            f"{CONFIG['JOINT_LAYER']}")

        PB = np.zeros((n_unit, G), dtype=np.float64)
        DET = np.zeros((n_unit, G), dtype=np.float64)
        PSUM = np.zeros((n_partner, G), dtype=np.float64)
        PDET = np.zeros((n_partner, G), dtype=np.float64)
        DTECH = np.zeros((n_tech, G), dtype=np.float64)
        NCELL = np.zeros(n_unit, dtype=np.int64)
        NPART = np.zeros(n_partner, dtype=np.int64)
        NTECH = np.zeros(n_tech, dtype=np.int64)

        indptr = L["indptr"][:]
        chunk = CONFIG["CHUNK"]
        t0 = time.time()
        checked = False
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            ru, rp = row_unit[a:b], row_partner[a:b]
            if not ((ru >= 0).any() or (rp >= 0).any()):
                continue
            lo, hi = int(indptr[a]), int(indptr[b])
            M = sparse.csr_matrix(
                (L["data"][lo:hi], L["indices"][lo:hi], indptr[a:b + 1] - lo),
                shape=(b - a, G))
            if not checked:
                d = M.data
                assert np.all(d == np.round(d)), "counts layer is not integer-valued"
                assert np.all(d >= 0), "negative values in the counts layer"
                log(f"counts verified integer (first chunk, {M.nnz:,} nnz)")
                checked = True
            B = M.copy()
            B.data = (B.data > 0).astype(np.float64)

            def _G(lbl, nrow):
                s = np.flatnonzero(lbl >= 0)
                return sparse.csr_matrix(
                    (np.ones(len(s)), (lbl[s], s)), shape=(nrow, b - a))

            if (ru >= 0).any():
                g = _G(ru, n_unit)
                PB += (g @ M).toarray()
                DET += (g @ B).toarray()
                NCELL += np.bincount(ru[ru >= 0], minlength=n_unit)
                rt = row_tech[a:b]
                if (rt >= 0).any():
                    DTECH += (_G(rt, n_tech) @ B).toarray()
                    NTECH += np.bincount(rt[rt >= 0], minlength=n_tech)
            if (rp >= 0).any():
                g = _G(rp, n_partner)
                PSUM += (g @ M).toarray()
                PDET += (g @ B).toarray()
                NPART += np.bincount(rp[rp >= 0], minlength=n_partner)
            if (a // chunk) % 20 == 0:
                log(f"  {b:,}/{n:,}  {time.time() - t0:.0f}s")

    exp = pd.read_csv(OUT / "_expected_unit_cells.csv").set_index("unit_idx")["n_cells"]
    got = pd.Series(NCELL, index=np.arange(n_unit))
    assert (got.reindex(exp.index).values == exp.values).all(), \
        "cell counts from the pass disagree with the prep table"
    log(f"pass done in {time.time() - t0:.0f}s; unit cells {NCELL.sum():,}, "
        f"partner cells {NPART.sum():,}")

    np.savez_compressed(OUT / "pass_accumulators.npz", PB=PB, DET=DET, PSUM=PSUM, PDET=PDET,
                        DTECH=DTECH, NCELL=NCELL, NPART=NPART, NTECH=NTECH, var=var)
    log(f"wrote {OUT/'pass_accumulators.npz'}")


# ======================================================================================
# stage: fit -- DE machinery
# ======================================================================================
def _bh(p):
    p = np.asarray(p, dtype=float)
    ok = ~np.isnan(p)
    out = np.full(p.shape, np.nan)
    q = p[ok]
    n = len(q)
    if n == 0:
        return out
    order = np.argsort(q)
    ranked = q[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


def _deseq(counts, meta, n_cpus):
    """Paired pseudobulk DESeq2 with LFC shrinkage.

    The explicit DefaultInference is not cosmetic: helpers/pseq30_helpers.py:147 documents that
    DeseqStats without one spawns a worker per host core and stalls on these 256-core nodes.
    """
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    from pydeseq2.default_inference import DefaultInference

    inf = DefaultInference(n_cpus=n_cpus)
    dds = DeseqDataSet(counts=counts.astype(int), metadata=meta, design="~donor + arm",
                       refit_cooks=True, inference=inf, quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=["arm", "M", "B"], inference=inf, quiet=True)
    st.summary()
    cols = list(dds.obsm["design_matrix"].columns)
    coeff = [c for c in cols if c.startswith("arm")]
    assert len(coeff) == 1, f"cannot resolve the arm coefficient among {cols[:5]}... -> {coeff}"
    raw = st.results_df[["baseMean", "log2FoldChange", "pvalue"]].rename(
        columns={"log2FoldChange": "log2FC_raw"})
    st.lfc_shrink(coeff=coeff[0])
    out = raw.join(st.results_df[["log2FoldChange", "lfcSE"]].rename(
        columns={"log2FoldChange": "log2FC"}))
    return out


def _tissue_matrices(tissue, units, acc, var):
    """Pseudobulk counts, detection fractions and per-unit CPM log2FC for one tissue."""
    u = units[units["tissue"] == tissue].copy()
    idx = u["unit_idx"].to_numpy()
    PB = acc["PB"][idx]
    DET = acc["DET"][idx]
    NC = acc["NCELL"][idx]
    name = (u["unit"] + "::" + u["arm"]).to_numpy()
    counts = pd.DataFrame(np.rint(PB).astype(np.int64), index=name, columns=var)
    meta = pd.DataFrame({"donor": u["unit"].to_numpy(), "arm": u["arm"].to_numpy()}, index=name)
    detf = pd.DataFrame(DET / np.maximum(NC, 1)[:, None], index=name, columns=var)
    return u, counts, meta, detf, NC


def _pooled_detection(u, acc, var):
    """Detection fraction pooled over all eligible units of a tissue, per arm."""
    out = {}
    for arm in ("M", "B"):
        s = u[u["arm"] == arm]
        idx = s["unit_idx"].to_numpy()
        out[arm] = pd.Series(acc["DET"][idx].sum(0) / max(acc["NCELL"][idx].sum(), 1), index=var)
    return out


def _per_donor_lfc(u, acc, var):
    """Per-donor log2FC from CPM with pseudocount 1 (the concordance statistic)."""
    PB = acc["PB"]
    piv = {}
    for arm in ("M", "B"):
        s = u[u["arm"] == arm].set_index("unit")
        m = PB[s["unit_idx"].to_numpy()]
        cpm = m / np.maximum(m.sum(1, keepdims=True), 1) * 1e6
        piv[arm] = pd.DataFrame(cpm, index=s.index, columns=var)
    donors = piv["M"].index.intersection(piv["B"].index)
    return np.log2((piv["M"].loc[donors] + 1) / (piv["B"].loc[donors] + 1))


def _measured_by_tech(acc, techs, tissue):
    """Genes with at least one detected cell in a given tissue x chemistry.

    A 10x Flex probe panel leaves its off-panel genes at exactly zero in every cell, which is how
    the FFPE chemistry is separated from the whole-transcriptome ones. A gene that is genuinely
    never expressed is indistinguishable from an off-panel one; that makes the split conservative
    (such a gene fails the count filter anyway).
    """
    out = {}
    for i, t in enumerate(techs):
        tt, chem = str(t).split("|", 1)
        if tt != tissue:
            continue
        out[chem] = acc["DTECH"][i] > 0
    return out


def _fit_tissue(tissue, units, acc, var, lr, techs, n_cpus):
    u, counts, meta, detf, NC = _tissue_matrices(tissue, units, acc, var)
    log(f"[{tissue}] {len(u) // 2} donors, {len(counts)} pseudobulks, {counts.shape[1]:,} genes")

    expressed = ((counts >= CONFIG["MIN_COUNTS"]).mean(0) >= CONFIG["MIN_FRAC_PB"]).to_numpy()
    log(f"[{tissue}] genes passing >= {CONFIG['MIN_COUNTS']} counts in "
        f">= {CONFIG['MIN_FRAC_PB']:.0%} of pseudobulks: {expressed.sum():,}")

    measured = _measured_by_tech(acc, techs, tissue)
    chems = sorted(measured)
    log(f"[{tissue}] chemistries: {chems}")
    common = np.ones(len(var), bool)
    for m in measured.values():
        common &= m

    is_flex = u.set_index("unit")["tech"].astype(str).str.contains("flex", case=False)
    is_flex = is_flex[~is_flex.index.duplicated()]

    fits = []
    if len(chems) > 1:
        g1 = expressed & common
        fits.append(("all-chemistry", meta.index, g1))
        # Every expressed gene lands in exactly one fit: those measurable in every chemistry go
        # to fit (i) with all donors, the rest to fit (ii) with the non-Flex donors only.
        # Intersecting fit (ii) with the non-Flex measured sets as well would strand the genes a
        # small chemistry happens to miss, so it is deliberately not done.
        g2 = expressed & ~common
        has_flex = bool(is_flex.any())
        keep_u = ([k for k in meta.index if not bool(is_flex.get(k.split("::")[0], False))]
                  if has_flex else list(meta.index))
        fname2 = "non-Flex" if has_flex else "chemistry-restricted"
        if g2.sum() and len(keep_u) >= 6:
            fits.append((fname2, pd.Index(keep_u), g2))
        elif g2.sum():
            log(f"[{tissue}] {int(g2.sum()):,} chemistry-restricted genes dropped: "
                f"only {len(keep_u) // 2} donors outside the probe-panel chemistry")
    else:
        fits.append(("single-chemistry", meta.index, expressed))

    lr_genes = set()
    for col in ("ligand", "receptor"):
        for v in lr[col]:
            lr_genes.update(_subunits(v))

    pooled = _pooled_detection(u, acc, var)
    pdl = _per_donor_lfc(u, acc, var)

    out = []
    for fname, rows, genes in fits:
        gs = np.asarray(var)[genes]
        log(f"[{tissue}] fit '{fname}': {len(rows)} pseudobulks x {len(gs):,} genes")
        res = _deseq(counts.loc[rows, gs], meta.loc[rows], n_cpus)
        res = res.reset_index().rename(columns={"index": "gene"})
        res["fit"] = fname
        res["tissue"] = tissue
        res["n_donors"] = len(rows) // 2
        out.append(res)
    de = pd.concat(out, ignore_index=True)

    de["pct_M"] = pooled["M"].reindex(de["gene"]).to_numpy()
    de["pct_B"] = pooled["B"].reindex(de["gene"]).to_numpy()
    de["in_lr_resource"] = de["gene"].isin(lr_genes)

    # BH over the LR-resource genes only, once per tissue per fit
    de["padj"] = np.nan
    for fname, g in de.groupby("fit"):
        m = g.index[g["in_lr_resource"]]
        de.loc[m, "padj"] = _bh(de.loc[m, "pvalue"].to_numpy())

    # donor concordance
    sub = pdl.reindex(columns=de["gene"].unique())
    sign_model = np.sign(de["log2FC"].to_numpy())
    conc, nelig = [], []
    for gene, s in zip(de["gene"], sign_model):
        v = sub[gene].to_numpy() if gene in sub.columns else np.array([])
        v = v[np.isfinite(v)]
        nelig.append(len(v))
        conc.append(int(np.sum(np.sign(v) == s)) if len(v) and s != 0 else 0)
    de["donors_concordant"], de["donors_eligible"] = conc, nelig

    lig = {g for v in lr["ligand"] for g in _subunits(v)}
    rec = {g for v in lr["receptor"] for g in _subunits(v)}
    de["role"] = np.where(de["gene"].isin(lig) & de["gene"].isin(rec), "both",
                 np.where(de["gene"].isin(lig), "ligand",
                 np.where(de["gene"].isin(rec), "receptor", "other")))

    higher = np.where(de["log2FC"] > 0, de["pct_M"], de["pct_B"])
    de["pct_higher_arm"] = higher
    de["significant"] = (de["padj"] < CONFIG["PADJ"]) & (de["log2FC"].abs() >= CONFIG["LFC"]) \
        & (higher >= CONFIG["MIN_DETECT"]) & de["in_lr_resource"]
    de["direction"] = np.where(de["log2FC"] > 0, "up in M", "down in M")
    return de, u, pdl


def _partner_profile(tissue, partners, acc, var):
    """Detection fraction and CP10K mean per partner type, pooled over eligible donors."""
    p = partners[partners["tissue"] == tissue]
    idx = p["partner_idx"].to_numpy()
    det = pd.DataFrame(acc["PDET"][idx] / np.maximum(acc["NPART"][idx], 1)[:, None],
                       index=p["partner"].to_numpy(), columns=var)
    s = acc["PSUM"][idx]
    cp10k = pd.DataFrame(s / np.maximum(s.sum(1, keepdims=True), 1) * 1e4,
                         index=p["partner"].to_numpy(), columns=var)
    return p, det, cp10k


def _complex_ok(complex_str, det_row, thr):
    subs = _subunits(complex_str)
    vals = [det_row.get(s, np.nan) for s in subs]
    if any(pd.isna(v) for v in vals):
        return False, np.nan
    return bool(min(vals) >= thr), float(min(vals))


def _interactions(tissue, de, lr, partners, acc, var, pooled_det, genes=None):
    """One row per CD4-side gene x cognate gene x partner type passing the gate."""
    p, pdet, pcp = _partner_profile(tissue, partners, acc, var)
    hom = dict(zip(p["partner"], p["homotypic"]))
    if genes is None:
        genes = de.loc[de["significant"], "gene"].tolist()
    gset = set(genes)
    info = de.set_index("gene")

    rows = []
    for _, r in lr.iterrows():
        lsubs, rsubs = _subunits(r["ligand"]), _subunits(r["receptor"])
        for side, mine, cognate in (("outgoing", lsubs, r["receptor"]),
                                    ("incoming", rsubs, r["ligand"])):
            for g in mine:
                if g not in gset or g not in info.index:
                    continue
                rec = info.loc[g]
                if isinstance(rec, pd.DataFrame):
                    rec = rec.iloc[0]
                arm_det = pooled_det["M"] if rec["log2FC"] > 0 else pooled_det["B"]
                others = [s for s in mine if s != g]
                cx_ok = all(arm_det.get(s, 0) >= CONFIG["MIN_DETECT"] for s in others)
                for partner in pdet.index:
                    ok, pct = _complex_ok(cognate, pdet.loc[partner], CONFIG["PARTNER_DETECT"])
                    if not ok:
                        continue
                    rows.append(dict(
                        tissue=tissue, cd4_gene=g, cd4_complex=(r["ligand"] if side == "outgoing"
                                                                else r["receptor"]),
                        cognate_gene=cognate, direction=side,
                        sign="gained in M" if rec["log2FC"] > 0 else "lost in M",
                        log2FC=rec["log2FC"], padj=rec["padj"],
                        pct_M=rec["pct_M"], pct_B=rec["pct_B"],
                        partner_type=partner, homotypic=bool(hom.get(partner, False)),
                        partner_pct=pct,
                        partner_mean_cp10k=float(np.mean(
                            [pcp.loc[partner].get(s, np.nan) for s in _subunits(cognate)])),
                        cd4_complex_ok=bool(cx_ok), support=int(r["support"])))
    out = pd.DataFrame(rows)
    if len(out):
        out = out.drop_duplicates(
            subset=["cd4_gene", "cognate_gene", "partner_type", "direction"])
    return out, pdet, pcp


# ======================================================================================
# stage: fit -- prior axes, figures, report
# ======================================================================================
def _prior_axes(de_by_tissue, lr_full, partners, acc, var, pooled_by_tissue):
    """The pre-specified axes, reported regardless of significance.

    Five of them (JAML-CXADR, SPN-SIGLEC1, IL22-IL22RA1, CSF2's receptor, and TOX, which is not
    an LR gene at all) fall outside the support >= 6 resource, so their cognate pairs are
    resolved against the UNFILTERED consensus and flagged.
    """
    rows = []
    for tissue, de in de_by_tissue.items():
        info = de.set_index("gene")
        p, pdet, pcp = _partner_profile(tissue, partners, acc, var)
        for gene, cognates, note in PRIOR_AXES:
            rec = info.loc[gene] if gene in info.index else None
            if isinstance(rec, pd.DataFrame):
                rec = rec.iloc[0]
            cogs = list(cognates)
            if not cogs:
                hits = lr_full[(lr_full["ligand"].str.split("_").apply(lambda x: gene in x))
                               | (lr_full["receptor"].str.split("_").apply(lambda x: gene in x))]
                cogs = sorted({(r["receptor"] if gene in _subunits(r["ligand"]) else r["ligand"])
                               for _, r in hits.iterrows()})
            gated = []
            for c in cogs:
                for partner in pdet.index:
                    ok, pct = _complex_ok(c, pdet.loc[partner], CONFIG["PARTNER_DETECT"])
                    if ok:
                        gated.append(f"{c}:{partner} ({pct:.0%})")
            rows.append(dict(
                tissue=tissue, gene=gene, note=note,
                tested=rec is not None,
                log2FC=float(rec["log2FC"]) if rec is not None else np.nan,
                pvalue=float(rec["pvalue"]) if rec is not None else np.nan,
                padj=float(rec["padj"]) if rec is not None else np.nan,
                padj_family=("LR genes" if (rec is not None and bool(rec["in_lr_resource"]))
                             else "none (outside the filtered resource)"),
                in_filtered_resource=bool(rec["in_lr_resource"]) if rec is not None else False,
                pct_M=float(rec["pct_M"]) if rec is not None else np.nan,
                pct_B=float(rec["pct_B"]) if rec is not None else np.nan,
                donors_concordant=int(rec["donors_concordant"]) if rec is not None else 0,
                donors_eligible=int(rec["donors_eligible"]) if rec is not None else 0,
                fit=rec["fit"] if rec is not None else "",
                cognate_genes="; ".join(map(str, cogs)),
                gated_partners="; ".join(gated[:12])))
    return pd.DataFrame(rows)


def _style():
    import logging
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Carlito and Calibri are not installed on these nodes; they stay first in the stack so the
    # figures pick them up wherever they are, and DejaVu Sans is the fallback here. Silence the
    # per-glyph findfont warning rather than let it flood the job log.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": ["Carlito", "Calibri", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 400, "figure.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })
    return plt


def _save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=400, facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    log(f"wrote {stem}.png / .pdf")


def _fig_heatmap(tissue, de, pdl, units, top_n=40):
    """Per-donor log2FC for the top significant LR genes; donors grouped by study."""
    plt = _style()
    from matplotlib.colors import LinearSegmentedColormap

    sig = de[de["significant"]].reindex(
        de[de["significant"]]["log2FC"].abs().sort_values(ascending=False).index).head(top_n)
    if sig.empty:
        log(f"[{tissue}] no significant genes -- heatmap skipped")
        return None
    genes = sig["gene"].tolist()
    M = pdl.reindex(columns=genes)
    meta = (units[units["tissue"] == tissue].drop_duplicates("unit")
            .set_index("unit")[["study", "dataset"]])
    meta = meta.reindex(M.index)
    order = meta.sort_values(["study"]).index
    M = M.loc[order]

    cmap = LinearSegmentedColormap.from_list("teal", ["#8C5A12", "#FFFFFF", TEAL])
    v = float(np.nanpercentile(np.abs(M.to_numpy()), 98)) or 1.0
    fig, ax = plt.subplots(figsize=(max(5.0, 0.085 * len(order) + 2.4),
                                    max(3.0, 0.14 * len(genes) + 1.3)),
                           layout="constrained")
    im = ax.imshow(M.T.to_numpy(), aspect="auto", cmap=cmap, vmin=-v, vmax=v,
                   interpolation="nearest")
    ax.set_yticks(range(len(genes)))
    roles = sig.set_index("gene")["role"].reindex(genes)
    ax.set_yticklabels([f"{g}  ({'L' if r == 'ligand' else 'R' if r == 'receptor' else 'LR'})"
                        for g, r in zip(genes, roles)])
    studies = meta.loc[order, "study"].astype(str).to_numpy()
    bnd = [0] + [i for i in range(1, len(studies)) if studies[i] != studies[i - 1]] + [len(studies)]
    ax.set_xticks([(bnd[i] + bnd[i + 1]) / 2 - 0.5 for i in range(len(bnd) - 1)])
    ax.set_xticklabels([studies[bnd[i]].split("|")[0] for i in range(len(bnd) - 1)],
                       rotation=90, fontsize=6)
    for b in bnd[1:-1]:
        ax.axvline(b - 0.5, color="white", lw=0.8)
    ax.set_xlabel(f"donors ({len(order)}), grouped by study")
    ax.set_title(f"{tissue}: per-donor log2FC (malignant vs benign CD4), "
                 f"top {len(genes)} significant LR genes")
    fig.colorbar(im, ax=ax, shrink=0.55, label="log2FC (CPM+1), per donor")
    _save(fig, f"fig_{tissue}_donor_lfc_heatmap")
    return genes


def _fig_dotplot(tissue, genes, inter, pdet, pcp):
    """Partner type x cognate gene: size = % expressing, colour = mean CP10K."""
    plt = _style()
    sub = inter[inter["cd4_gene"].isin(genes)] if len(inter) else inter
    if sub.empty:
        log(f"[{tissue}] no gated interactions -- dot plot skipped")
        return
    cog = (sub.groupby("cognate_gene")["cd4_gene"].nunique()
           .sort_values(ascending=False).head(40).index.tolist())
    parts = sorted(sub["partner_type"].unique())
    D = np.full((len(parts), len(cog)), np.nan)
    C = np.full((len(parts), len(cog)), np.nan)
    for j, c in enumerate(cog):
        subs = _subunits(c)
        for i, p in enumerate(parts):
            if p not in pdet.index:
                continue
            vals = [pdet.loc[p].get(s, np.nan) for s in subs]
            cps = [pcp.loc[p].get(s, np.nan) for s in subs]
            if any(pd.isna(v) for v in vals):
                continue
            D[i, j] = min(vals)
            C[i, j] = float(np.mean(cps))
    fig, ax = plt.subplots(figsize=(max(5.0, 0.21 * len(cog) + 2.0),
                                    max(3.0, 0.26 * len(parts) + 1.4)),
                           layout="constrained")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("teal1", ["#FFFFFF", TEAL])
    X, Y = np.meshgrid(np.arange(len(cog)), np.arange(len(parts)))
    gate = np.isfinite(D) & (D >= CONFIG["PARTNER_DETECT"])
    below = np.isfinite(D) & ~gate
    vmax = float(np.nanmax(np.log1p(C))) if np.isfinite(C).any() else 1.0
    # Below-gate cells are drawn, faintly, so the gate itself is readable off the figure.
    ax.scatter(X[below], Y[below], s=np.clip(D[below], 0, 1) * 110 + 4,
               c=np.log1p(C[below]), cmap=cmap, vmin=0, vmax=vmax,
               edgecolors="#BDBDBD", linewidths=0.3, alpha=0.45)
    sc = ax.scatter(X[gate], Y[gate], s=np.clip(D[gate], 0, 1) * 110 + 4,
                    c=np.log1p(C[gate]), cmap=cmap, vmin=0, vmax=vmax,
                    edgecolors=TEAL, linewidths=0.5)
    ax.set_xticks(range(len(cog)))
    ax.set_xticklabels(cog, rotation=90, fontsize=6)
    ax.set_yticks(range(len(parts)))
    ax.set_yticklabels(parts)
    ax.set_xlim(-0.6, len(cog) - 0.4)
    ax.set_ylim(-0.6, len(parts) - 0.4)
    ax.set_title(f"{tissue}: cognate-gene expression in partner cell types "
                 f"(outlined = passes the {CONFIG['PARTNER_DETECT']:.0%} gate)")
    fig.colorbar(sc, ax=ax, shrink=0.55, label="log1p mean CP10K")
    for sz, lab in ((0.25, "25%"), (0.5, "50%"), (1.0, "100%")):
        ax.scatter([], [], s=sz * 110 + 4, c="white", edgecolors=TEAL, linewidths=0.5, label=lab)
    ax.scatter([], [], s=30, c="white", edgecolors="#BDBDBD", linewidths=0.3, alpha=0.45,
               label=f"below the {CONFIG['PARTNER_DETECT']:.0%} gate")
    ax.legend(title="% cells expressing", loc="upper left", bbox_to_anchor=(1.12, 1.0),
              frameon=False, labelspacing=1.0)
    _save(fig, f"fig_{tissue}_partner_dotplot")


def _md(df, cols=None, floatfmt=3, max_rows=None):
    d = df if cols is None else df[cols]
    if max_rows is not None:
        d = d.head(max_rows)
    d = d.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else
                            (f"{v:.2e}" if (v != 0 and abs(v) < 1e-3) else f"{v:.{floatfmt}f}"))
        else:
            d[c] = d[c].astype(str)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    rule = "|" + "|".join(["---"] * len(d.columns)) + "|"
    body = "\n".join("| " + " | ".join(r) + " |" for r in d.astype(str).to_numpy())
    return "\n".join([head, rule, body])


def _versions():
    import importlib.metadata
    out = {}
    for m in ("numpy", "pandas", "scipy", "pydeseq2", "anndata", "h5py", "matplotlib", "polars"):
        try:
            out[m] = importlib.metadata.version(m)
        except Exception:
            out[m] = "?"
    out["python"] = sys.version.split()[0]
    return out


def _tissue_results_section(tissue, de, inter, top_n=30):
    s = de[de["significant"]]
    up, dn = s[s["log2FC"] > 0], s[s["log2FC"] < 0]
    L = []
    P = L.append
    P(f"Genes tested: **{de['gene'].nunique():,}** over "
      f"{de['fit'].nunique()} fit(s) ({', '.join(sorted(de['fit'].unique()))}); of those "
      f"**{int(de['in_lr_resource'].sum()):,}** are in the filtered LR resource and carry a BH "
      f"adjusted p-value.")
    P("")
    P(f"Significant at padj < {CONFIG['PADJ']}, |shrunken log2FC| >= {CONFIG['LFC']}, and "
      f">= {CONFIG['MIN_DETECT']:.0%} detection in the higher arm: **{len(s)}** genes — "
      f"**{len(up)} up in malignant**, **{len(dn)} down**. "
      f"By role: ligand {int((s['role'] == 'ligand').sum())}, "
      f"receptor {int((s['role'] == 'receptor').sum())}, "
      f"both {int((s['role'] == 'both').sum())} "
      f"(up: L {int((up['role'] == 'ligand').sum())} / R {int((up['role'] == 'receptor').sum())}; "
      f"down: L {int((dn['role'] == 'ligand').sum())} / R {int((dn['role'] == 'receptor').sum())}).")
    P("")
    cols = ["gene", "role", "baseMean", "log2FC", "padj", "pct_M", "pct_B",
            "donors_concordant", "donors_eligible", "fit"]
    P(f"**Top {min(top_n, len(up))} up in malignant** (by shrunken log2FC):")
    P("")
    P(_md(up.sort_values("log2FC", ascending=False), cols, max_rows=top_n) if len(up)
      else "_none_")
    P("")
    P(f"**Top {min(top_n, len(dn))} down in malignant**:")
    P("")
    P(_md(dn.sort_values("log2FC"), cols, max_rows=top_n) if len(dn) else "_none_")
    P("")
    if len(inter):
        top_genes = set(up.sort_values("log2FC", ascending=False).head(top_n)["gene"]) | \
                    set(dn.sort_values("log2FC").head(top_n)["gene"])
        it = inter[inter["cd4_gene"].isin(top_genes)]
        P(f"**Gated interactions for those genes** — {len(it):,} CD4-gene x cognate x partner "
          f"rows pass the {CONFIG['PARTNER_DETECT']:.0%} partner gate "
          f"({it['cd4_gene'].nunique()} CD4-side genes, {it['partner_type'].nunique()} partner "
          f"types). The full set for every significant gene is in "
          f"`interactions_{tissue}.csv` ({len(inter):,} rows).")
        P("")
        P(_md(it.sort_values(["cd4_gene", "partner_pct"], ascending=[True, False]),
              ["cd4_gene", "sign", "direction", "cognate_gene", "partner_type",
               "partner_pct", "partner_mean_cp10k"], max_rows=80))
    else:
        P("_No interaction passed the partner gate._")
    return "\n".join(L)


def _report(de_by_tissue, inter_by_tissue, cnt, excl, partners, prior, units, pdl_by_tissue,
            lr, versions, blood_sub):
    T = []
    P = T.append
    tissues = [t for t in ("skin", "blood") if t in de_by_tissue]

    P("# Differential cell–cell communication of malignant vs benign CD4 T cells (MF/SS atlas)")
    P("")
    P(f"_Generated {time.strftime('%Y-%m-%d %H:%M')} by `jobs/run_ccc52_mvb.py`. "
      f"Skin and blood are analysed completely separately._")
    P("")

    # ---- 1. summary
    P("## 1. Summary")
    P("")
    for t in tissues:
        de = de_by_tissue[t]
        s = de[de["significant"]]
        up, dn = s[s["log2FC"] > 0], s[s["log2FC"] < 0]
        u = units[units["tissue"] == t]
        nd = u["unit"].nunique()
        it = inter_by_tissue[t]
        def _names(g, asc):
            n = g.sort_values("log2FC", ascending=asc)["gene"].tolist()
            return (", ".join(n[:8]) + (", …" if len(n) > 8 else "")) if n else "none"
        top, topd = _names(up, False), _names(dn, True)
        P(f"**{t.capitalize()}.** Across **{nd} eligible donors** "
          f"({int(cnt[(cnt['tissue'] == t) & cnt['eligible']]['n_malignant'].sum()):,} malignant "
          f"and {int(cnt[(cnt['tissue'] == t) & cnt['eligible']]['n_benign'].sum()):,} benign CD4 "
          f"cells), the paired pseudobulk model tested {de['gene'].nunique():,} genes and "
          f"{int(de['in_lr_resource'].sum()):,} of the 1,043 ligand–receptor genes. "
          f"**{len(s)} LR genes** meet all three call criteria — **{len(up)} up in the malignant "
          f"arm** ({top}) and **{len(dn)} down** ({topd}). "
          f"{len(it):,} CD4-gene × cognate-gene × partner-type rows pass the "
          f"{CONFIG['PARTNER_DETECT']:.0%} partner gate across "
          f"{it['partner_type'].nunique() if len(it) else 0} partner types.")
        P("")

    # ---- 2. rationale
    P("## 2. Rationale and estimand")
    P("")
    P("Within a donor, malignant (M) and benign (B) CD4 cells face the same partner cells. For "
      "any multiplicative ligand–receptor score the partner term cancels in the M-vs-B contrast, "
      "so the differential interaction is exactly a **paired differential expression of ligand "
      "and receptor genes on the CD4 side**. Partner cells only act as a gate: is the cognate "
      "gene expressed in that cell type. The CD4-side result is the inference; the partner "
      "assignment is descriptive. Nothing here is an ambient correction, a LIANA specificity "
      "score, a latent-variable model, or a permutation test.")
    P("")

    # ---- 3. data
    P("## 3. Data")
    P("")
    P("### Paths used")
    P("")
    P(_md(pd.DataFrame([
        {"role": "counts (raw integers, full gene space)",
         "path": str(CONFIG["JOINT_H5AD"].relative_to(NB_DIR)) + f"  [layers/{CONFIG['JOINT_LAYER']}]"},
        {"role": "cell metadata", "path": str(CONFIG["OBS_PARQUET"].relative_to(NB_DIR))},
        {"role": "malignancy call, skin", "path": str(CONFIG["MAL_PARQUET"]["skin"].relative_to(NB_DIR))},
        {"role": "malignancy call, blood", "path": str(CONFIG["MAL_PARQUET"]["blood"].relative_to(NB_DIR))},
        {"role": "blood per-donor clusters (benign-arm cleanup)",
         "path": str(CONFIG["BLOOD_CLUSTERS"].relative_to(NB_DIR))},
        {"role": "blood marker expression (10k HVG)", "path": str(CONFIG["BLOOD_T_H5AD"].relative_to(NB_DIR))},
        {"role": "blood non-T fine labels", "path": str(CONFIG["BLOOD_SUBTYPES"].relative_to(NB_DIR))},
        {"role": "blood CD8→NK correction", "path": str(CONFIG["BLOOD_CT_FIX"].relative_to(NB_DIR))},
        {"role": "LR resource (rebuilt here)", "path": "reports/ccc52_mvb/lr_resource_1422.csv"},
    ])))
    P("")
    P("### obs columns used")
    P("")
    P(_md(pd.DataFrame([
        {"quantity": "dataset / study", "column": "`dataset`, `study`"},
        {"quantity": "donor key", "column": "`donor` (dataset-prefixed, e.g. `D8__P78`; no cross-study collisions)"},
        {"quantity": "patient (unit of analysis)", "column": "`clin_person_key`, falling back to `patient_key`, then `donor`"},
        {"quantity": "sample", "column": "`sample_id`"},
        {"quantity": "tissue", "column": "`compartment` (Skin / Blood / LN)"},
        {"quantity": "malignancy", "column": f"`{CONFIG['MAL_CALL']}` from the cluster-malignancy parquets"},
        {"quantity": "cell type", "column": "`cell_type_final` (19 levels; myeloid and fibroblast sub-clusters already collapsed)"},
        {"quantity": "chemistry", "column": "`tech` (10x_5p, 10x_flex, 10x_3p, ECCITE, 10x)"},
        {"quantity": "clinical", "column": "`clin_dx`, `clin_phenotype`, `clin_ex_vivo`"},
    ])))
    P("")
    P("### Cells per donor × tissue × arm")
    P("")
    P(f"All {len(cnt):,} donor × tissue units that have at least one cell in either arm after the "
      f"cohort rules. `eligible` marks the ones with >= {CONFIG['MIN_CELLS_PER_ARM']} cells in "
      f"**each** arm.")
    P("")
    P(_md(cnt.sort_values(["tissue", "n_malignant"], ascending=[True, False]),
          ["tissue", "unit", "n_malignant", "n_benign", "eligible", "study", "tech",
           "dx", "n_samples", "n_donor_keys"]))
    P("")
    P("### Eligible donors per dataset and tissue")
    P("")
    el = cnt[cnt["eligible"]]
    P(_md(el.groupby(["tissue", "study"]).size().rename("n_eligible_donors").reset_index()
          .sort_values(["tissue", "n_eligible_donors"], ascending=[True, False])))
    P("")
    P("### Donors excluded, and why")
    P("")
    ex = (excl.groupby(["tissue2", "drop_reason"])
          .agg(n_units=("unit", "nunique"), n_cells=("n_cells", "sum")).reset_index()
          .rename(columns={"tissue2": "tissue"}))
    P(_md(ex))
    P("")
    P("Units that survived the cohort rules but failed the "
      f"{CONFIG['MIN_CELLS_PER_ARM']}-cells-per-arm floor: " +
      ", ".join(f"{t} {int(((cnt['tissue'] == t) & ~cnt['eligible']).sum())}"
                for t in tissues) + ".")
    P("")
    P("### Blood benign-arm cleanup")
    P("")
    P("Blood carries no Treg and no naive-CD4 label (`cell_type_T` is {CD4, CD8, NK}; "
      "`cell_type_final` has one flat `CD4` over 554,369 cells), so the brief's fallback applies. "
      "The benign blood CD4 cells were scored on the per-donor clusters nb32b had already "
      "computed (`blood_T_clusters_v1.parquet`, Leiden res 1.0, keyed `<donor>__<k>`), using "
      "marker **detection fractions**: a cluster is Treg-like if FOXP3 is detected in "
      f">= {CONFIG['BLOOD_TREG_DETECT']:.0%} of its benign CD4 cells, and naive-like if it is "
      "detected above " + ", ".join(f"{v:.0%} for {k}" for k, v in CONFIG["BLOOD_NAIVE_MIN"].items())
      + " and its S100A4 detection sits below the across-cluster mean. "
      "A purely relative composite-z rule was tried first and rejected: S100A4 detection "
      "saturates (median 0.89 across clusters), so low-depth clusters in which every marker is "
      "low scored as naive — it flagged seven clusters whose own naive markers were all under "
      "0.30. The absolute rule flags none of those. "
      f"Of {len(blood_sub):,} clusters, {int(blood_sub['is_treg'].sum())} were called Treg-like "
      f"and {int(blood_sub['is_naive'].sum())} naive-like; their cells left the benign arm. "
      "Per-cluster numbers are in `blood_benign_subclusters.csv`. In skin no such step is needed: "
      "`CD4_Treg` is its own `cell_type_final` level, and the script asserts that no "
      "Treg-labelled cell reaches either arm.")
    P("")
    P("### Partner cell types")
    P("")
    P(f"A coarse annotation per tissue, taken from `cell_type_final` (which already collapses the "
      f"nb22 myeloid and fibroblast sub-clusters to major types), with the malignant and benign "
      f"CD4 arms added as types of their own. Pooled over eligible donors of that tissue; types "
      f"below {CONFIG['MIN_PARTNER_CELLS']} cells are dropped.")
    P("")
    P(_md(partners.sort_values(["tissue", "n_cells"], ascending=[True, False]),
          ["tissue", "partner", "n_cells", "homotypic"]))
    P("")
    T.append(_report_tail(de_by_tissue, inter_by_tissue, prior, units, lr, versions, tissues))
    return "\n".join(T)


def _report_tail(de_by_tissue, inter_by_tissue, prior, units, lr, versions, tissues):
    T = []
    P = T.append

    # ---- 4. methods
    P("## 4. Methods")
    P("")
    P("**Counts.** Raw integer counts were read from "
      f"`{CONFIG['JOINT_H5AD'].relative_to(NB_DIR)}`, layer `{CONFIG['JOINT_LAYER']}` "
      "(2,157,693 × 42,347, CSR). Integer-ness and non-negativity are asserted on the first "
      "streamed chunk. The compartment objects were deliberately not used for counts: "
      "`blood_T_annotated.h5ad` is a 10,000-HVG subset that holds only 590 of the 1,043 LR "
      "genes, and neither compartment object contains non-T partner cells.")
    P("")
    P(f"**Arms.** Malignancy is `{CONFIG['MAL_CALL']}`, the repo headline call since 2026-10-22 "
      "(`helpers/malignancy.py`): per-sample Leiden over all T cells with one call per cluster, "
      "anchored per sample on the ALICE TCR clone. M = CD4 T cells called malignant; B = CD4 T "
      "cells called non-malignant. `cell_type_final == \"CD4\"` already excludes `CD4_Treg`, "
      "which is a separate level. Cells with a null (unassessed) call enter neither arm.")
    P("")
    P("**Unit of analysis.** The donor, defined as `clin_person_key` (the repo's curated person "
      "key, which merges the same patient across centres) falling back to `patient_key` and then "
      "`donor`. All samples and all timepoints of a patient within a tissue are summed into one "
      "pseudobulk per arm.")
    P("")
    P("**Cohort rules.** Applied before summation, following `helpers/target_cohort.py`: skin "
      "donors whose malignant clone is not CD4 (`clin_phenotype` in CD8/gd/DN, or `clin_dx` in "
      "CD8_aggr_CTCL/pcAECTCL) are dropped, because their CD4 cells sit in the benign arm beside "
      "a tumour that is a different lineage; skin donors with a diagnosis outside the MF/SS "
      "spectrum (LyP, SPP, LPP) are dropped; the whole geskin2026 blood cohort is dropped. The "
      "'same person under two donor keys' rule is not applied as a drop — patient-level "
      "aggregation already handles it by summing.")
    P("")
    P(f"**Eligibility.** A donor × tissue needs >= {CONFIG['MIN_CELLS_PER_ARM']} cells in each arm.")
    P("")
    P(f"**Pseudobulk.** Raw counts summed over all cells of a donor × arm within a tissue, in one "
      f"streamed pass over the atlas in chunks of {CONFIG['CHUNK']:,} cells. The same pass "
      "accumulates per-group detection counts (cells with count > 0) and per-chemistry detection "
      "counts.")
    P("")
    P(f"**LR resource.** LIANA `consensus` intersected with interactions supported by "
      f">= {CONFIG['LR_SUPPORT']} of the 15 human LIANA resources = **1,422 interactions over "
      f"1,043 unique genes** (asserted). Complexes are underscore-joined and each subunit is "
      "treated as a gene. No blacklist was applied — see §10.")
    P("")
    P(f"**Gene filter.** A gene enters a fit if it has >= {CONFIG['MIN_COUNTS']} counts in "
      f">= {CONFIG['MIN_FRAC_PB']:.0%} of that fit's pseudobulks.")
    P("")
    P("**Model.** pydeseq2, paired pseudobulk, exact formula **`~ donor + arm`**, reference level "
      "`B`, contrast `arm M vs B`, fitted on the full transcriptome (not only LR genes). "
      "`refit_cooks=True`. Shrunken log2 fold changes come from `DeseqStats.lfc_shrink` on the "
      "`arm` coefficient; the unshrunken estimate is retained as `log2FC_raw`.")
    P("")
    P("**Mixed chemistries.** A 10x Flex probe panel leaves off-panel genes at exactly zero in "
      "every cell of that chemistry. Where a tissue mixes chemistries, fit (i) uses **all** "
      "eligible donors over the genes detected in every chemistry present, and fit (ii) uses "
      "**non-Flex donors only** for every remaining expressed gene. The two gene sets are "
      "disjoint and together cover every gene that passes the count filter, so no gene is "
      "stranded by a small chemistry that happens to miss it. The `fit` column records which fit "
      "each gene came from. Flex (the only probe-panel chemistry here) is li2024 and skin-only, "
      "so in blood fit (ii) keeps every donor and is labelled `chemistry-restricted` rather than "
      "`non-Flex`. The chemistries actually present per tissue are: " +
      "; ".join(f"**{t}** " + ", ".join(sorted(
          set(units.loc[units['tissue'] == t, 'tech'].astype(str))))
          for t in tissues) + ".")
    P("")
    P("**Multiple testing.** Benjamini–Hochberg applied to the DESeq2 Wald p-values **only over "
      "the genes present in the filtered LR resource**, once per tissue per fit. Genes outside "
      "the resource carry a p-value and no padj. DESeq2's own independent-filtering padj is not "
      "used, because its family is the whole transcriptome.")
    P("")
    P(f"**Call.** padj < {CONFIG['PADJ']}, |shrunken log2FC| >= {CONFIG['LFC']}, and the gene "
      f"detected in >= {CONFIG['MIN_DETECT']:.0%} of the cells of the higher arm (pooled over "
      f"eligible donors of that tissue).")
    P("")
    P("**Donor concordance.** Per-donor log2FC from CPM with pseudocount 1, "
      "`log2((CPM_M + 1)/(CPM_B + 1))`; the reported count is the number of donors whose sign "
      "matches the model estimate.")
    P("")
    P("**Complexes.** Every subunit is evaluated as a gene. A complex is called on the CD4 side "
      f"through its significant subunit provided every other subunit is detected in "
      f">= {CONFIG['MIN_DETECT']:.0%} of the cells of that arm (`cd4_complex_ok`).")
    P("")
    P(f"**Partner gate.** The cognate gene — all subunits, for a complex — must be detected in "
      f">= {CONFIG['PARTNER_DETECT']:.0%} of the cells of the partner type, pooled over eligible "
      "donors of that tissue. No statistical test is applied on the partner side. `outgoing` = "
      "CD4 ligand to partner receptor; `incoming` = partner ligand to CD4 receptor.")
    P("")
    P(f"**Reproducibility.** Seed {CONFIG['SEED']}. Versions: " +
      ", ".join(f"{k} {v}" for k, v in versions.items()) + ".")
    P("")

    # ---- 5/6. results
    for i, t in enumerate(tissues):
        P(f"## {5 + i}. Results, {t}")
        P("")
        P(_tissue_results_section(t, de_by_tissue[t], inter_by_tissue[t]))
        P("")
        for stem, cap in ((f"fig_{t}_donor_lfc_heatmap", "per-donor log2FC"),
                          (f"fig_{t}_partner_dotplot", "partner dot plot")):
            if (OUT / f"{stem}.png").exists():
                P(f"![{t} {cap}]({stem}.png)")
                P("")

    # ---- 7. skin vs blood
    P("## 7. Skin vs blood")
    P("")
    if len(tissues) == 2:
        a, b = de_by_tissue["skin"], de_by_tissue["blood"]
        sa = set(a.loc[a["significant"], "gene"])
        sb = set(b.loc[b["significant"], "gene"])
        both = sorted(sa & sb)
        ja = a[a["in_lr_resource"]].drop_duplicates("gene").set_index("gene")["log2FC"]
        jb = b[b["in_lr_resource"]].drop_duplicates("gene").set_index("gene")["log2FC"]
        comm = ja.index.intersection(jb.index)
        r = float(np.corrcoef(ja[comm], jb[comm])[0, 1]) if len(comm) > 2 else np.nan
        rs = float(pd.Series(ja[comm]).corr(pd.Series(jb[comm]), method="spearman")) \
            if len(comm) > 2 else np.nan
        agree = int((np.sign(ja[comm]) == np.sign(jb[comm])).sum())
        P(f"Significant in skin only: **{len(sa - sb)}**; blood only: **{len(sb - sa)}**; "
          f"both: **{len(both)}**.")
        P("")
        P(f"- shared: {', '.join(both) if both else '_none_'}")
        P(f"- skin only: {', '.join(sorted(sa - sb)[:60])}"
          f"{' …' if len(sa - sb) > 60 else ''}")
        P(f"- blood only: {', '.join(sorted(sb - sa)[:60])}"
          f"{' …' if len(sb - sa) > 60 else ''}")
        P("")
        P(f"Across the **{len(comm):,} LR genes tested in both tissues**, the shrunken log2FC "
          f"correlate at Pearson r = **{r:.3f}** (Spearman {rs:.3f}), and {agree:,} of "
          f"{len(comm):,} ({agree / max(len(comm), 1):.0%}) agree in sign.")
        P("")
    P("**Significance in one tissue only is not evidence of tissue specificity.** The two "
      "compartments differ in eligible donor count, in cell numbers per arm, and in the "
      "composition of the benign arm, so they differ in power; a gene can miss the threshold in "
      "one tissue while its estimate is the same size and sign in both. Only the effect-size "
      "comparison above speaks to tissue specificity, and it does so weakly.")
    P("")

    # ---- 8. prior axes
    P("## 8. Prior axes")
    P("")
    P("Reported regardless of significance, both tissues. `padj_family` says whether the gene is "
      "inside the BH family (the filtered LR resource); five of the named axes are not, and "
      "their cognate pairs are resolved against the unfiltered consensus instead. **TOX is the "
      "positive control for the malignant contrast**, not an LR gene.")
    P("")
    P(_md(prior.sort_values(["gene", "tissue"]),
          ["gene", "tissue", "log2FC", "padj", "padj_family", "pct_M", "pct_B",
           "donors_concordant", "donors_eligible", "fit", "cognate_genes", "gated_partners"]))
    P("")

    # ---- 9. limitations
    P("## 9. Limitations")
    P("")
    for line in [
        "**No proximity information.** Dissociated scRNA-seq says nothing about whether two cells "
        "were ever in contact. Every interaction reported here is a statement about co-expression "
        "within a tissue, not about an engagement.",
        "**Partner assignments are not corrected for ambient RNA and are suggestive only.** No "
        "ambient correction was performed anywhere in this analysis, by design. A partner type "
        "that sits next to a highly expressing neighbour will carry that neighbour's transcripts, "
        "and the gate is a detection fraction, which ambient contamination inflates directly.",
        "**Benign arm composition.** The benign arm is whatever the malignancy call left behind. "
        "In skin that is conventional CD4 minus the `CD4_Treg` level; in blood it is conventional "
        "CD4 after the Treg-like and naive-like clusters were removed by a marker rule, which is "
        "a cruder instrument than a curated label. Residual activation-state differences between "
        "the arms are not separable from malignancy here.",
        "**Contact-dependent interactions in blood are weakly interpretable.** Circulating cells "
        "are not in stable contact. A gained CD28–CD86 or CTLA4–CD80 edge in blood is a statement "
        "about the transcript, not about a synapse.",
        "**Number of eligible blood donors.** Blood has fewer eligible donors than skin, and the "
        "geskin2026 exclusion removed the largest blood cohort outright. Power is correspondingly "
        "lower, which is the main reason a skin-only hit is not a tissue-specificity claim.",
        "**The malignancy call is a cluster call.** `mal_cluster` assigns a whole per-sample "
        "cluster, so a benign cell inside a malignant cluster is labelled malignant and vice "
        "versa. `helpers/malignancy.py` also notes that healthy-control samples are called benign "
        "by construction, so a 0% HC rate is an input, not a result.",
    ]:
        P(f"- {line}")
    P("")
    T.append(_report_end(de_by_tissue, tissues))
    return "\n".join(T)


def _report_end(de_by_tissue, tissues):
    T = []
    P = T.append
    P("## 10. Decisions and deviations; upstream errors noticed")
    P("")
    P("### Decisions and deviations from the brief")
    P("")
    for line in [
        "**The filtered LR resource was rebuilt, not loaded — it does not exist on disk.** The "
        "brief names it as an artefact of a rebuilt LIANA+ pipeline. A scoped search of `data/`, "
        "`tables/`, `docs/` and `helpers/` found no such file; `helpers/ccc_panel.py:12` carries "
        "the specification as a run note and nothing implements it. It was reproduced exactly: "
        "LIANA `consensus` intersected with support >= 6 of the 15 human resources gives 1,422 "
        "interactions over 1,043 genes, and the script asserts both numbers.",
        "**No blacklist was applied.** No blacklist file exists in this repo, and "
        "`docs/CCC_PIPELINE_PORTABLE_SPEC.md:83` states that `LR_CAVEATS` is *commentary only, "
        "never a blacklist*, because the v1 version silently deleted the IL4/IL13 pairs the "
        "analysis existed to report (`helpers/ccc_utils.py:1157`). Applying one would have "
        "contradicted the repo's own standing instruction, so the caveat list is not used as a "
        "filter.",
        "**Arms come from `mal_cluster`, not from the inferCNV/TCR `combined_malignant` flag.** "
        "The brief asks for 'the final malignancy calls (inferCNV/TCR pipeline)'. That combined "
        "flag was retired: `helpers/malignancy.py` states the four calls are never OR-ed, because "
        "the CNV call derives from smoothed regional expression — partly the same signal a "
        "downstream expression analysis then attributes. `mal_cluster` has been the headline "
        "since 2026-10-22 and is anchored per sample on a TCR sequence fact.",
        "**The unit is the curated person key.** `clin_person_key` falling back to `patient_key` "
        "and then `donor`, so longitudinal and multi-site samples of one patient are summed and "
        "the two patients recorded under two donor keys are merged.",
        "**geskin2026 was dropped from blood in full**, on the user's instruction. All 260,455 of "
        "its cells are ex vivo 2-day cultures (213,181 drug-treated, 47,274 vehicle), which are "
        "not in-vivo biology.",
        "**The blood benign arm was cleaned by a cluster marker rule, not a fresh Leiden.** Blood "
        "has no Treg and no naive-CD4 label, so the brief's fallback applies. Rather than "
        "re-cluster on the 10-dimensional MrVI u latent, the per-donor clusters nb32b had already "
        "computed were scored on marker detection fractions — within-donor, so free of batch "
        "structure. Thresholds and per-cluster numbers are in §3 and "
        "`blood_benign_subclusters.csv`.",
        "**Counts were taken from `joint_annotated.h5ad`, not the compartment objects**, so that "
        "blood is analysed over the full 42,347-gene space rather than a 10,000-gene HVG subset.",
        "**Five prior axes fall outside the filtered resource** (JAML–CXADR, SPN–SIGLEC1, "
        "IL22–IL22RA1, CSF2's receptor, and TOX, which is not an LR gene). They are reported from "
        "the unfiltered consensus and flagged, since the brief asks for them regardless of "
        "significance.",
        "**DESeq2's own independent-filtering padj is discarded.** Its family is the whole "
        "transcriptome; the brief asks for BH over the LR resource only, so BH is recomputed from "
        "the Wald p-values over that family, per tissue per fit.",
        "**The 'same person under two donor keys' cohort rule is applied as a merge, not a "
        "drop**, because patient-level summation is what the brief asks for.",
        "**Carlito and Calibri are not installed on these compute nodes.** They remain first in "
        "the font stack, so the figures render in them anywhere the fonts exist; here they fall "
        "back to DejaVu Sans. Everything else in the figure specification — PNG and PDF at 400 "
        "dpi, `pdf.fonttype=42`, white background, the single teal accent #0F766E, no decorative "
        "elements — is as requested.",
    ]:
        P(f"- {line}")
    P("")
    P("### Upstream data errors and inconsistencies noticed")
    P("")
    for line in [
        "**The brief's description of geskin2026 does not match the atlas.** It says 5 patients "
        "and 28 units. On disk geskin2026 is blood-only, 260,455 cells, 11 `patient_key` values, "
        "65 `donor` keys that are lane × HTO wells, and 11 libraries. "
        "`docs/DATA_PROVENANCE.md:61` records that the HTO-to-patient map was never deposited, so "
        "the 65 donor keys are pseudo-donors, not people.",
        "**`blood_T_annotated.h5ad` is a 10,000-gene HVG subset while `skin_T_annotated.h5ad` "
        "carries all 42,347.** This asymmetry is not flagged in either object. Using the "
        "compartment objects naively would have tested 590 of 1,043 LR genes in blood against "
        "1,028 in skin, and would have silently dropped CXADR and IL22RA1 from the prior-axes "
        "table.",
        "**The blood NK/CD8 labels are mixtures.** The blood object holds 606 NK against 112,609 "
        "CD8. `README.md` states that nb31 called CD8 on `CD8A`, which NK cells express as a "
        "homodimer, and that `blood_T_celltype_corrections_v1.csv` corrects only the 12 clusters "
        "the nb32b review examined while roughly 100 further CD8-labelled clusters carry the same "
        "signature. The blood CD8 and NK partner rows should be read as mixtures.",
        "**`atlas_obs_full_v2.parquet` carries stale columns.** Its `has_tcr` is wrong for li2024 "
        "(`helpers/malignancy.py` documents 0.072 there against 0.476 in the V(D)J fold-in "
        "cache), and its `is_malignant` is an older atlas-level call that disagrees with every "
        "current one. Neither is used here.",
        "**`_v2`/`_v3` filename suffixes are not recency markers** in this tree — "
        "`CLAUDE.md` records that `skin_T_annotated_v2.h5ad` was older than the unsuffixed file. "
        "Every input above was pinned by reading the module that owns it, not by its suffix.",
    ]:
        P(f"- {line}")
    P("")
    P("## 11. File manifest")
    P("")
    files = [
        ("ccc_malignant_vs_benign_cd4_report.md", "this report"),
        ("de_lr_genes_skin.csv", "every tested LR gene in skin, with the top-hit columns"),
        ("de_lr_genes_blood.csv", "every tested LR gene in blood"),
        ("de_all_genes_skin.csv", "the full-transcriptome fit for skin (no BH outside the LR family)"),
        ("de_all_genes_blood.csv", "the full-transcriptome fit for blood"),
        ("interactions_skin.csv", "CD4 gene × cognate gene × partner type passing the gate, skin"),
        ("interactions_blood.csv", "the same for blood"),
        ("cell_counts_donor_arm.csv", "cells per donor × tissue × arm, with eligibility"),
        ("prior_axes.csv", "the pre-specified axes plus the TOX control, both tissues"),
        ("partner_types.csv", "every candidate partner type with its pooled cell count and the keep flag"),
        ("partners.csv", "the partner types that passed the cell-count floor, with their indices"),
        ("cohort_excluded.csv", "every unit removed by a cohort rule, with the reason"),
        ("blood_benign_subclusters.csv", "per-cluster marker detection for the blood benign cleanup"),
        ("units.csv", "the eligible donor × arm pseudobulk index"),
        ("lr_resource_1422.csv", "the rebuilt filtered resource (1,422 interactions)"),
        ("lr_resource_consensus_full.csv", "unfiltered LIANA consensus with per-pair support"),
        ("lr_genes.json", "the 1,043 unique genes of the filtered resource"),
        ("_expected_unit_cells.csv", "cells per pseudobulk, written by `prep` and asserted by `pass`"),
        ("pseudobulk_skin.parquet", "donor × arm summed raw counts, skin (full gene space)"),
        ("pseudobulk_blood.parquet", "donor × arm summed raw counts, blood (full gene space)"),
        ("pass_accumulators.npz", "the streamed count sums and detection counts"),
        ("row_labels.npz", "cell → (donor × arm) and cell → partner-type label vectors"),
    ]
    for t in ("skin", "blood"):
        for stem, cap in ((f"fig_{t}_donor_lfc_heatmap",
                           f"per-donor log2FC, top 40 significant LR genes, {t}"),
                          (f"fig_{t}_partner_dotplot", f"partner type × cognate gene, {t}")):
            if (OUT / f"{stem}.png").exists():
                files.append((f"{stem}.png / .pdf", cap))
    files = [f for f in files if (OUT / f[0].split(" / ")[0]).exists()]
    P(_md(pd.DataFrame(files, columns=["file", "contents"])))
    P("")
    P("_Figures are opt-in: the `fit` stage writes them only when run with `--figures`._")
    P("")
    P("`for_analysis/` holds the text-only subset of the above (no figures, no `.npz`, no "
      "parquet) with its own `INDEX.md`, for handing the result to another model or reader. "
      "It is rewritten by every `fit` run, and by `run_ccc52_mvb.sh bundle` on its own.")
    P("")
    P("Produced by `jobs/run_ccc52_mvb.py` (stages `lr`, `prep`, `pass`, `fit`), submitted with "
      "`jobs/run_ccc52_mvb.sh <stage>`. Nothing under `data/` was written.")
    return "\n".join(T)


def stage_fit(args):
    import os

    npz = np.load(OUT / "pass_accumulators.npz", allow_pickle=False)
    acc = {k: npz[k] for k in npz.files}
    var = acc.pop("var").astype(str)
    techs = np.load(OUT / "row_labels.npz", allow_pickle=False)["techs"].astype(str)

    units = pd.read_csv(OUT / "units.csv")
    partners = pd.read_csv(OUT / "partners.csv")
    cnt = pd.read_csv(OUT / "cell_counts_donor_arm.csv")
    excl = pd.read_csv(OUT / "cohort_excluded.csv")
    blood_sub = pd.read_csv(OUT / "blood_benign_subclusters.csv")
    lr = pd.read_csv(OUT / "lr_resource_1422.csv")
    lr_full = pd.read_csv(OUT / "lr_resource_consensus_full.csv")
    n_cpus = int(os.environ.get("LSB_DJOB_NUMPROC", "4"))
    log(f"fitting with {n_cpus} cpus; {len(units) // 2} donor-tissue units total")

    de_by, inter_by, pdl_by, pooled_by = {}, {}, {}, {}
    for tissue in ("skin", "blood"):
        if not (units["tissue"] == tissue).any():
            log(f"[{tissue}] no eligible donors -- skipped")
            continue
        de, u, pdl = _fit_tissue(tissue, units, acc, var, lr, techs, n_cpus)
        pooled = _pooled_detection(u, acc, var)
        inter, pdet, pcp = _interactions(tissue, de, lr, partners, acc, var, pooled)
        de_by[tissue], inter_by[tissue], pdl_by[tissue], pooled_by[tissue] = de, inter, pdl, pooled

        cols = ["gene", "role", "baseMean", "log2FC", "log2FC_raw", "lfcSE", "pvalue", "padj",
                "pct_M", "pct_B", "donors_concordant", "donors_eligible", "fit", "significant",
                "direction", "tissue", "n_donors"]
        de.loc[de["in_lr_resource"], cols].sort_values("padj").to_csv(
            OUT / f"de_lr_genes_{tissue}.csv", index=False)
        de[cols].to_csv(OUT / f"de_all_genes_{tissue}.csv", index=False)
        inter.to_csv(OUT / f"interactions_{tissue}.csv", index=False)

        idx = u["unit_idx"].to_numpy()
        pb = pd.DataFrame(np.rint(acc["PB"][idx]).astype(np.int64),
                          index=(u["unit"] + "::" + u["arm"]).to_numpy(), columns=var)
        pb.to_parquet(OUT / f"pseudobulk_{tissue}.parquet")
        log(f"[{tissue}] {int(de['significant'].sum())} significant LR genes, "
            f"{len(inter):,} gated interaction rows")

        if args.figures:
            genes = _fig_heatmap(tissue, de, pdl, units)
            if genes:
                _fig_dotplot(tissue, genes, inter, pdet, pcp)

    prior = _prior_axes(de_by, lr_full, partners, acc, var, pooled_by)
    prior.to_csv(OUT / "prior_axes.csv", index=False)

    txt = _report(de_by, inter_by, cnt, excl, partners, prior, units, pdl_by, lr,
                  _versions(), blood_sub)
    path = OUT / "ccc_malignant_vs_benign_cd4_report.md"
    path.write_text(txt)

    stage_bundle(args)

    print()
    print(path)
    for t in ("skin", "blood"):
        if t not in de_by:
            continue
        de, it = de_by[t], inter_by[t]
        s = de[de["significant"]]
        nd = units[units["tissue"] == t]["unit"].nunique()
        print(f"{t}: {nd} donors | {de['gene'].nunique():,} genes tested, "
              f"{int(de['in_lr_resource'].sum()):,} LR | {len(s)} significant "
              f"({int((s['log2FC'] > 0).sum())} up, {int((s['log2FC'] < 0).sum())} down) | "
              f"{len(it):,} gated interactions")
    tox = prior[prior["gene"] == "TOX"]
    print("TOX control: " + "; ".join(
        f"{r.tissue} log2FC {r.log2FC:+.2f} (p {r.pvalue:.1e})" for r in tox.itertuples()))
    print(f"resource: {len(lr):,} interactions, "
          f"{len({g for c in ('ligand', 'receptor') for v in lr[c] for g in _subunits(v)}):,} genes")


# ======================================================================================
# stage: bundle -- the text-only set to hand to another model
# ======================================================================================
#: (filename, one-line description, include even though it is large)
BUNDLE = [
    ("ccc_malignant_vs_benign_cd4_report.md", "the full write-up: rationale, data, methods, "
     "results per tissue, prior axes, limitations, deviations", False),
    ("de_lr_genes_skin.csv", "every tested ligand-receptor gene in SKIN", False),
    ("de_lr_genes_blood.csv", "every tested ligand-receptor gene in BLOOD", False),
    ("interactions_skin.csv", "SKIN: one row per CD4 gene x cognate gene x partner type that "
     "passes the partner gate", False),
    ("interactions_blood.csv", "BLOOD: the same", False),
    ("prior_axes.csv", "the 12 pre-specified axes plus the TOX positive control, both tissues, "
     "reported regardless of significance", False),
    ("cell_counts_donor_arm.csv", "cells per donor x tissue x arm, with the eligibility flag", False),
    ("cohort_excluded.csv", "every donor removed by a cohort rule, with the reason", False),
    ("partner_types.csv", "partner cell types and their pooled cell counts per tissue", False),
    ("blood_benign_subclusters.csv", "per-cluster marker detection behind the blood benign-arm "
     "Treg/naive cleanup", False),
    ("units.csv", "the eligible donor x arm pseudobulk index", False),
    ("lr_resource_1422.csv", "the filtered LR resource: 1,422 interactions with their support", False),
    ("de_all_genes_skin.csv", "SKIN full-transcriptome fit (no padj outside the LR family) -- "
     "LARGE, look genes up rather than reading it whole", True),
    ("de_all_genes_blood.csv", "BLOOD full-transcriptome fit -- LARGE, same caveat", True),
]


def stage_bundle(args):
    """Copy the text-only outputs into one self-describing sub-directory.

    Everything a reader needs to interpret the result and nothing they cannot open: no figures,
    no .npz, no parquet. Rewritten from scratch each time so it can never drift from the tables
    beside it.
    """
    import shutil

    dest = OUT / "for_analysis"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    rows, missing = [], []
    for name, desc, large in BUNDLE:
        src = OUT / name
        if not src.exists():
            missing.append(name)
            continue
        shutil.copy2(src, dest / name)
        n = sum(1 for _ in src.open()) - 1 if name.endswith(".csv") else None
        rows.append((name, desc, large, src.stat().st_size, n))
    if missing:
        log(f"WARNING: not in {OUT}, skipped: {missing}")

    def _sz(b):
        return f"{b / 1e6:.1f} MB" if b >= 1e6 else f"{b / 1e3:.0f} KB"

    tab = pd.DataFrame([{"file": n, "size": _sz(b), "rows": ("" if r is None else f"{r:,}"),
                         "contents": d} for n, d, _, b, r in rows])
    big = [n for n, _, l, _, _ in rows if l]

    txt = f"""# Malignant vs benign CD4 differential cell-cell communication — analysis bundle

Everything here is plain text. Start with `ccc_malignant_vs_benign_cd4_report.md`; the CSVs are
the same numbers in machine-readable form. Generated {time.strftime('%Y-%m-%d %H:%M')} by
`jobs/run_ccc52_mvb.py`.

## What the analysis is

Paired pseudobulk differential expression of ligand and receptor genes between a patient's
**malignant (M)** and **benign (B)** CD4 T cells, run separately in skin and blood. The logic: in
one donor both arms face the same partner cells, so in any multiplicative LR score the partner
term cancels and the differential interaction reduces to a paired DE on the CD4 side. **The
CD4-side statistics are the inference. The partner assignment is descriptive only** — partner
cells act purely as a gate (is the cognate gene expressed in that cell type), with no test and no
ambient correction.

## Files

{_md(tab)}

## How to read the gene tables

- `log2FC` is the **shrunken** estimate from the `~ donor + arm` model, M relative to B; a
  positive value means higher in the malignant arm. `log2FC_raw` is the unshrunken Wald estimate.
- `padj` is BH **over the genes in the filtered LR resource only**, computed per tissue per fit.
  A gene outside that family has a `pvalue` and a blank `padj` — it was still fitted, it is just
  not in the multiple-testing family. Do not compare a blank padj to a significant one.
- `significant` already encodes all three call criteria together: padj < {CONFIG['PADJ']},
  |log2FC| >= {CONFIG['LFC']}, and detection in >= {CONFIG['MIN_DETECT']:.0%} of the cells of the
  higher arm. Use that column rather than re-thresholding.
- `pct_M` / `pct_B` are detection fractions (cells with a non-zero count) pooled over eligible
  donors — not expression levels.
- `donors_concordant` / `donors_eligible` is the per-donor sign check: how many donors' own
  CPM-based log2FC has the same sign as the model estimate. It is the main guard against a result
  driven by one or two patients, and it is worth quoting next to any claim.
- `fit` says which model the gene came from. Where a tissue mixes chemistries, genes detected in
  every chemistry are fitted on all donors and the rest on the non-probe-panel donors only.

## Things that will mislead you if you do not know them

- **Significance in one tissue only is not evidence of tissue specificity.** The compartments
  differ in donor count, cells per arm and benign-arm composition, so they differ in power.
  Section 7 of the report gives the effect-size correlation, which is the comparison that counts.
- **The benign arms are not the same thing in the two tissues.** Skin excludes Tregs via an
  existing label; blood has no Treg or naive-CD4 label at all, so those cells were removed by a
  marker rule on per-donor clusters (`blood_benign_subclusters.csv`).
- **Blood CD8 and NK partner rows are mixtures.** The upstream annotation called CD8 on `CD8A`,
  which NK cells also express; only 12 clusters were corrected.
- **Blood is far more imbalanced than skin** — the benign arm is small in Sezary patients. Check
  `cell_counts_donor_arm.csv` before reading much into a blood effect size.
- No ambient-RNA correction was applied anywhere, by design. Partner detection fractions are
  inflated by contamination from neighbouring cell types.
"""
    if big:
        txt += ("\n- " + ", ".join(f"`{b}`" for b in big) +
                " are several MB; look genes up in them rather than reading them whole.\n")
    (dest / "INDEX.md").write_text(txt)

    total = sum(b for _, _, _, b, _ in rows) + len(txt)
    log(f"bundle: {len(rows)} files, {total / 1e6:.1f} MB -> {dest}")
    print(dest)
    return dest


STAGES = {"lr": stage_lr, "prep": stage_prep, "pass": stage_pass,
          "fit": stage_fit, "bundle": stage_bundle}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--figures", action="store_true",
                    help="fit stage: also draw the two figures per tissue (off by default)")
    args = ap.parse_args()
    log(f"stage {args.stage}; output dir {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    STAGES[args.stage](args)
    log(f"stage {args.stage} complete")


if __name__ == "__main__":
    main()
