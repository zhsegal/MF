#!/usr/bin/env python
"""Headless build of the joint atlas through joint_annotated.h5ad (nb10 Steps 1-3b).

The interactive Jupyter kernel is capped at 63 GB; the concat of ~1.5M cells peaks well
above that and OOM-kills the kernel. This runs the same helper calls + the two nb10
annotate/refine cells in a big-memory batch job so the kernel never touches the concat.

Outputs: data/atlas_joint/{joint_raw.h5ad, joint_annotated.h5ad}, updated
data/sample_metadata_final.csv. Afterwards run MrVI (jobs/run_mrvi_joint.sh) + the
annotation/downstream notebooks.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

NB_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NB_DIR))
import atlas_join_helpers as H  # noqa: E402

OUT = NB_DIR / "data" / "atlas_joint"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-std", action="store_true", help="rebuild ALL standardized (else reuse cache; B5 always builds if missing)")
    ap.add_argument("--force-tcr", action="store_true", help="rebuild tcr_clones.parquet")
    ap.add_argument("--no-scrublet", action="store_true")
    args = ap.parse_args()

    REG = H.dataset_registry(NB_DIR)
    EXPR = [k for k, v in REG.items() if v.get("in_expression")]
    print("EXPR:", EXPR)

    # --- Step 1: per-dataset standardized (cached reused; B5 rebuilt) --------------------
    std_paths, failed = {}, {}
    for label in EXPR:
        try:
            std_paths[label] = H.build_standardized(label, NB_DIR, force=args.force_std,
                                                     run_scrublet=not args.no_scrublet)
        except Exception as e:  # noqa: BLE001
            failed[label] = repr(e)
            print(f"  [{label}] FAILED: {e!r}")
    if failed:
        raise SystemExit(f"standardized build failed: {failed}")
    print("built:", list(std_paths))

    # --- Step 2: TCR clone table --------------------------------------------------------
    tcr_parquet = H.build_tcr_table(NB_DIR, force=args.force_tcr)

    # --- Step 3: dedup + concat (the memory-heavy step) ---------------------------------
    print("dedup ledger hits:", H.detect_duplicate_cells(std_paths, NB_DIR))
    joint_path = H.concat_joint(NB_DIR, std_paths, tcr_parquet, force=True)
    adata_raw = sc.read_h5ad(joint_path)
    adata_raw.obs["dataset"] = adata_raw.obs["dataset"].map(lambda d: H.DATASET_NAME.get(d, d)).astype("category")
    print("joint_raw:", adata_raw.shape, "| ADT:", "X_adt" in adata_raw.obsm)

    # --- B5 metadata rows into sample_metadata_final.csv (nb10 cell b5meta01) -----------
    _append_b5_metadata(adata_raw)

    # --- Step 3b: annotate + resolve lineage (nb10 cell a7241a16) -----------------------
    ANNOT = OUT / "joint_annotated.h5ad"
    FINAL = NB_DIR / "data" / "sample_metadata_final.csv"
    D1_CLIN = NB_DIR / "data" / "D1_chennareddy2025" / "meta" / "patients.tsv"
    adata_raw = sc.read_h5ad(OUT / "joint_raw.h5ad")
    adata_raw, added = H.annotate_from_csv(adata_raw, FINAL)
    H.resolve_lineage(adata_raw, min_cells=20, gd_frac_thresh=0.5, clinical_d1_tsv=D1_CLIN)
    meta_ids = set(pd.read_csv(FINAL, dtype=str)["sample_id"])
    missing = sorted(set(adata_raw.obs["sample_id"].astype(str).unique()) - meta_ids)
    assert not missing, f"{len(missing)} obs samples absent from CSV: {missing[:10]}"
    assert not (set(adata_raw.obs["disease"].unique()) - H.DISEASE_VOCAB), "bad disease vocab"
    assert not (set(adata_raw.obs["compartment"].unique()) - H.COMPARTMENT_VOCAB), "bad compartment vocab"
    assert adata_raw.obs["lineage"].isna().sum() == 0, "NaN lineage after resolve"
    adata_raw.write_h5ad(ANNOT)
    print(f"[merge] added {len(added)} cols; wrote {ANNOT} shape={adata_raw.shape}")

    # --- D1 clinical override + PT* provisional (nb10 cell 2ee13474) --------------------
    _refine_lineage(adata_raw)
    adata_raw.write_h5ad(ANNOT)
    print(f"DONE -> {ANNOT}  ({adata_raw.n_obs:,} cells, {adata_raw.obs['donor'].nunique()} donors, "
          f"{adata_raw.obs['dataset'].nunique()} datasets)")


def _append_b5_metadata(adata_raw):
    FINAL = NB_DIR / "data" / "sample_metadata_final.csv"
    b = adata_raw.obs[adata_raw.obs["dataset"].astype(str) == "buus25"]
    if not len(b):
        print("[B5] no buus25 cells — skipping metadata append"); return
    cur = pd.read_csv(FINAL, dtype=str)
    CONST = dict(dataset="buus25", study="buus2025", n_donors="1", disease="SS",
                 disease_stage="", tech="10x_5p", sex="unknown", repo="GEO",
                 accession="GSE284075", malignant_call_method="TCR+ALICE",
                 tcr_available="yes", malignant_labeled="no", malignant_frac="",
                 tcr_recovery="", stage_class="advanced", stage_clean="", lineage="CD4",
                 entity="Sezary", fmf_status="not_applicable", lineage_resolve="no",
                 treatment_context="ex_vivo_baseline", blood_involvement="yes",
                 lesion_type="unknown", cohort_country="Denmark",
                 cohort_region="European", origin_resolve="no", east_asian="False")
    rows = []
    for sid, sub in b.groupby(b["sample_id"].astype(str)):
        comp, tis = str(sub["compartment"].iloc[0]), str(sub["tissue"].iloc[0])
        r = dict.fromkeys(cur.columns, ""); r.update(CONST)
        r.update(sample_id=sid, donor=str(sub["donor"].iloc[0]), compartment=comp,
                 tissue=tis, organ=("blood" if comp == "Blood" else "skin"),
                 tissue_detail=tis, n_cells=str(len(sub)),
                 n_tcr=str(int(sub["has_tcr"].sum())),
                 n_clones=str(sub.loc[sub["clone_id"].astype(str) != "", "clone_id"].nunique()),
                 n_malignant=str(int(sub["is_malignant"].sum())))
        rows.append(r)
    new = pd.DataFrame(rows, columns=cur.columns)
    kept = cur[~cur["sample_id"].isin(new["sample_id"])]
    pd.concat([kept, new], ignore_index=True).to_csv(FINAL, index=False)
    print(f"[B5] wrote {len(new)} buus25 metadata rows ({new.donor.nunique()} donors)")


def _refine_lineage(adata_raw):
    obs = adata_raw.obs
    for c in ["lineage", "entity", "lineage_method", "lineage_resolve"]:
        obs[c] = obs[c].astype(str)
    pat = obs["sample_id"].astype(str).str.split("__").str[-1]
    clin = pd.read_csv(NB_DIR / "data" / "D1_chennareddy2025" / "meta" / "patients.tsv", sep="\t")
    sub2 = {"gd_MF": ("gamma_delta", "MF_gamma_delta"),
            "Berti": ("CD8", "CD8_aggressive_epidermotropic_CTCL")}
    clin_map = {p: sub2[s] for p, s in zip(clin["sample_id_in_raw"], clin["subset"]) if s in sub2}
    expr = obs["lineage"].copy()
    is_d1 = obs["dataset"].eq("chennareddy25") & pat.isin(set(clin_map))
    for p, (lin, ent) in clin_map.items():
        m = is_d1 & pat.eq(p)
        obs.loc[m, "lineage"] = lin
        obs.loc[m, "entity"] = ent
    obs["lineage_disagreement"] = is_d1 & obs["lineage"].ne(expr) & ~expr.str.contains("RESOLVE")
    obs.loc[is_d1, "lineage_method"] = "clinical(patients.tsv)"
    obs.loc[is_d1, "lineage_resolve"] = "resolved_clinical"
    pt = obs["dataset"].eq("li24") & pat.isin(["PT11", "PT53", "PT55"])
    obs.loc[pt, "lineage"] = "unresolved_provisional"
    obs.loc[pt, "entity"] = "MF_unresolved"
    obs.loc[pt, "lineage_method"] = "inferCNV_state_no_TCR(unverified)"
    obs.loc[pt, "lineage_resolve"] = "provisional_inferCNV_unverified"
    for c in ["is_malignant", "is_dominant_clone"]:
        if c in obs and obs[c].astype(str).eq("2 values").any():
            obs.drop(columns=c, inplace=True)
    for c in ["lineage", "entity", "lineage_method", "lineage_resolve"]:
        obs[c] = obs[c].astype("category")


if __name__ == "__main__":
    main()
