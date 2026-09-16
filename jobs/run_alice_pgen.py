"""Headless OLGA Pgen / ALICE sweep for nb21 (heavy step, CPU).

Reproduces the load + cohort + ALICE pipeline of ``21_tcr_alice_neighborhood.ipynb`` and
writes the two cached results the notebook reads:
  data/atlas_joint/alice_cd4_per_donor.parquet   (Use 1)
  data/atlas_joint/alice_cd8.parquet             (Use 3, per-donor + per-stage)

Re-run is cheap once cached. Use ``--force`` to recompute.
"""
import sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1]   # .../notebooks/MF
sys.path.insert(0, str(NB_DIR))

import numpy as np
import pandas as pd
import atlas_join_helpers as H
import skin_T_cnv_helpers as C
import alice_helpers as A

FORCE = "--force" in sys.argv
FRAC_THRESH, RATIO_THRESH, EXPANDED_MIN = 0.05, 1.33, 2
CD4_T_TYPES = ["CD4_Cm", "CD4_Th2", "CD4_cytotoxic", "CD4_mem"]
CD8_T_TYPES = ["CD8_effector", "CD8_em"]
DROP_ENTITIES = {"MF_gamma_delta", "CD8_aggressive_epidermotropic_CTCL"}
ALPHA, Q = 0.05, None

OUT = NB_DIR / "data" / "atlas_joint"
OBJ = OUT / "skin_T_annotated.h5ad"
TCR_OBJ = OUT / "skin_T_tcr_annotated.h5ad"
LI_TCR = NB_DIR / "data" / "Li2024_atlas" / "li2024_tcr_malignancy.parquet"
ALICE_CD4 = OUT / "alice_cd4_per_donor.parquet"
ALICE_CD8 = OUT / "alice_cd8.parquet"

adata = C.build_or_load_tcr_object(OBJ, TCR_OBJ, LI_TCR, H)
is_li = adata.obs["study"].astype(str).eq("li2024").to_numpy()
C.recompute_dominant_clone(adata, H, is_li, FRAC_THRESH, RATIO_THRESH, EXPANDED_MIN)
clone_summary = C.clone_summary_table(adata, FRAC_THRESH, RATIO_THRESH)
clone_summary = clone_summary.merge(
    adata.obs[["donor", "entity", "study"]].astype(str).drop_duplicates("donor"),
    on="donor", how="left")

drop = set()
if {"D5__MFIVB", "D1__P303"} <= set(clone_summary["donor"]):
    drop.add("D1__P303")
drop |= set(clone_summary.loc[clone_summary["entity"].isin(DROP_ENTITIES), "donor"])
small = (clone_summary["n_tcr_cells"] < 300) & (clone_summary["study"] != "herrera2021")
drop |= set(clone_summary.loc[small, "donor"])
keep = [d for d in clone_summary["donor"] if d not in drop]
adata = adata[adata.obs["donor"].isin(keep)].copy()
hc = adata.obs["disease"].astype(str).eq("HC").to_numpy()
adata.obs.loc[hc, ["tcr_is_malignant", "tcr_is_dominant_clone"]] = False
print(f"[alice] kept donors={len(keep)} cells={adata.n_obs}", flush=True)

obs = adata.obs
clono_cd4 = A.clonotype_table(obs[obs["cell_type_T"].isin(CD4_T_TYPES)], group="donor")
clono_cd8 = A.clonotype_table(obs[obs["cell_type_T"].isin(CD8_T_TYPES)], group="donor")
stage = obs[["donor", "disease_stage"]].astype(str).drop_duplicates().set_index("donor")["disease_stage"]
clono_cd8 = clono_cd8.assign(disease_stage=clono_cd8["donor"].map(stage).astype(str))

pgen = A.make_pgen(A.load_olga_trb())

if FORCE or not ALICE_CD4.exists():
    res = A.run_alice_by_group(clono_cd4, pgen, group="donor", Q=Q, alpha=ALPHA)
    res.to_parquet(ALICE_CD4, index=False)
    print(f"[alice] wrote {ALICE_CD4} significant={int(res['significant'].sum())}", flush=True)

if FORCE or not ALICE_CD8.exists():
    pd_ = A.run_alice_by_group(clono_cd8, pgen, group="donor", Q=Q, alpha=ALPHA); pd_["scope"] = "donor"
    ps = A.run_alice_by_group(clono_cd8, pgen, group="disease_stage", Q=Q, alpha=ALPHA); ps["scope"] = "stage"
    res8 = pd.concat([pd_, ps], ignore_index=True)
    res8.to_parquet(ALICE_CD8, index=False)
    print(f"[alice] wrote {ALICE_CD8} significant={int(res8['significant'].sum())}", flush=True)
print("[alice] done", flush=True)
