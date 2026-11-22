# 20_skin/24_subclone_tcr_signaling.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/20_skin/24_subclone_tcr_signaling.ipynb` and is never run._

# 24 — Subclonal evolution + TCR/co-stimulation signaling (on the re-annotation)

Consolidates the old `old/23_subclonal_evolution` (subclonal evolution) + `old/26_tcr_signaling_subclones` (TCR-signaling subclones) onto the **current**
re-annotation object `skin_T_tcr_annotated_v4.h5ad` (`21_reannotation` labels + the li2024 V(D)J fold-in from `23_malignancy_tcr_cnv`)
and the v5 malignancy outputs (`skin_T_malignancy_v5`, `skin_T_arm_cnv_v5`). Helpers reused verbatim
(`subclone_helpers`, `alice_helpers`, `tcr_signaling_helpers`).

**Malignant compartment = `mal_cluster`** (26b per-sample cluster call), assembled through
`helpers/malignancy.py` — the project headline since 2026-10-22. The ALICE call (`mal_tcr`) is still
loaded: it is what the per-donor founder logic in §A needs, and it is the sensitivity arm.
Donor eligibility requires **both** ≥ `MIN_MAL` mal_cluster-malignant cells **and** arm-CNV coverage —
`mal_cluster` admits donors with no V(D)J, but subclone detection still runs off the inferCNV arm matrix.

> Rebuilt 2026-10-22 from the `_v3` run. Outputs carry `_v4`; the `_v3` artifacts (ALICE-only arms) are left in place.

- **§A** — subclonal (divergent-evolution) structure: per-donor CNV subclones, transcriptomic
  MAJOR → CNV MINOR nested subclones, trunk/branch arm events, functional programs, tropism.
- **§B** — TCR / co-stimulation signaling: where in the TCR cascade each malignant clone/subclone
  concentrates dysregulation, vs lineage-matched reactive-CD4, cross-checked against arm-CNV.

> **HEAVY** — loads/normalizes the ~18 GB object; run on the GPU/compute kernel, not the login node.
> The subclone detection itself is CPU-light off the cached arm-CNV matrix.

# §A — Subclonal (divergent-evolution) analysis

```python  # cell 2
# ============================================================
# §A.0  Parameters
# ============================================================
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import sys
from pathlib import Path


def _resolve_nb_dir() -> Path:
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data from {start}")


NB_DIR = _resolve_nb_dir(); print("NB_DIR =", NB_DIR)
OUT_DIR = NB_DIR / "data" / "atlas_joint"
FIG_DIR = NB_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)

# ---- inputs (nb23 v4 object + v5 malignancy outputs; malignancy itself via helpers/malignancy.py) ----
OBJ           = OUT_DIR / "skin_T_tcr_annotated_v4.h5ad"        # nb23 cache: nb21 labels + li2024 V(D)J folded in
MALIG_PARQUET = OUT_DIR / "skin_T_malignancy_v5.parquet"        # nb23 Step 8: cnv_arm_malignant + tcr_* columns
ARM           = OUT_DIR / "skin_T_arm_cnv_v5.parquet"           # nb23: per-cell 41-arm inferCNV matrix + held-out diploid null
MALIG_CALL    = "mal_cluster"                                   # 26b; headline. 'mal_tcr' reproduces the _v3 run
GMT           = NB_DIR.parent / "lib1_immune.gmt"

# ---- knobs (as nb23) ----
SEED = 0
MIN_MAL = 200
K_MAX = 4
MIN_SUB = 50
MIN_LAYER_CELLS = 30
# subclone gates: calibrated against the held-out diploid null (nb30 v4 arm cache carries it),
# not hard-coded. SIL_MIN / MIN_ARM_DELTA are the fallback for a null-less matrix, never a floor.
SIL_MIN = 0.15
MIN_ARM_DELTA = 0.03
NULL_Z = 3.0
NULL_DRAWS = 8
Z_VLIM = 3.0        # arm-profile colour limit in diploid SDs (fixed, not self-normalizing)
Z_THR, EPS = 2.5, 0.01
BRANCH_DELTA = 0.03
AUC_THR, SIL_THR = 0.65, 0.10
LATENT = "X_mrvi_u"
LEIDEN_RES = 0.2
N_NEIGHBORS = 15

# ---- outputs (v4: mal_cluster arms; the v3 set was ALICE-only and is left in place) ----
SUBCLONE_PARQUET = OUT_DIR / "subclones_v4.parquet"
SUMMARY_CSV      = OUT_DIR / "subclone_summary_v4.csv"
TRUNKBRANCH_CSV  = OUT_DIR / "subclone_trunk_branch_v4.csv"
ELIG_CSV         = OUT_DIR / "subclone_donor_eligibility_v4.csv"
SIGN_JSON        = OUT_DIR / "subclone_signatures_v4.json"
DE_DIR           = OUT_DIR / "subclone_de"; DE_DIR.mkdir(exist_ok=True)
```

```
NB_DIR = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
```

```python  # cell 3
import importlib, gc
import numpy as np, pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

sys.path.insert(0, str(NB_DIR / "helpers"))
import subclone_helpers as S
import alice_helpers as A
import skin_T_cnv_helpers as H
import malignancy as MAL
for _m in (S, A, H, MAL):
    importlib.reload(_m)
np.random.seed(SEED)
sc.settings.verbosity = 1
plt.rcParams["figure.dpi"] = 120
```

```
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
```

```python  # cell 4
# ============================================================
# §A.0  Load: malignant compartment + cached arm-CNV matrix
# ============================================================
if not os.environ.get("LSB_JOBID"):
    print("⚠️  No LSB_JOBID detected — you may be on the login node; loading skin_T_annotated.h5ad (~18 GB) can crash it.")
adata = sc.read_h5ad(OBJ)
print("loaded", OBJ.name, adata.shape)
# X and 'counts' are unused (rebuilt from raw_counts in §A.9 / §B); free them to keep peak RAM low.
adata.X = None
if "counts" in adata.layers:
    del adata.layers["counts"]
# new re-annotation carries cell_type_T (CD4/CD8/Tregs/UNK); alias to cell_type_T2 for the shared helpers.
adata.obs["cell_type_T2"] = adata.obs["cell_type_T"].astype(str)

# Malignancy: all four calls assembled by helpers/malignancy.py, then MALIG_CALL picked as the arm.
# `has_tcr` comes from THIS object (the nb23 V(D)J fold-in), not from atlas_obs_full_v2 whose li2024
# has_tcr is stale -- see the malignancy module docstring.
_mal_df = MAL.load_malignancy(adata.obs_names, has_tcr=adata.obs["has_tcr"].to_numpy())
for _c in MAL.CALLS:
    adata.obs[_c] = _mal_df[_c].to_numpy()
    adata.obs[f"{_c}_defined"] = _mal_df[f"{_c}_defined"].to_numpy()
# ALICE stays available: the per-donor founder logic in §A needs it, and it is the sensitivity arm.
adata.obs["tcr_malignant_alice"] = adata.obs["mal_tcr"].to_numpy()
print(MAL.coverage_report(_mal_df).to_string(index=False))
del _mal_df
assert MALIG_PARQUET.exists(), f"{MALIG_PARQUET} missing — run 23_malignancy_tcr_cnv.ipynb first"
_mp = pd.read_parquet(MALIG_PARQUET)
_bool = {"cnv_arm_malignant", "tcr_is_dominant_clone", "tcr_is_expanded", "tcr_is_malignant"}
for c in ["cnv_arm_malignant", "tcr_clone_id", "tcr_is_dominant_clone", "tcr_is_expanded",
          "tcr_clone_size", "tcr_is_malignant"]:
    if c in _mp.columns:
        s = _mp[c].reindex(adata.obs_names)
        adata.obs[c] = (s.fillna(False).astype(bool).to_numpy() if c in _bool
                        else s.fillna("" if c == "tcr_clone_id" else 0).to_numpy())
del _mp
# The arm: MALIG_CALL where it is defined. Undefined is NOT benign -- those cells are carried as
# `malig_undefined` and excluded from the eligibility count rather than counted as reactive.
mal = (adata.obs[MALIG_CALL].to_numpy() & adata.obs[f"{MALIG_CALL}_defined"].to_numpy())
adata.obs["is_malig"] = mal
adata.obs["malig_undefined"] = ~adata.obs[f"{MALIG_CALL}_defined"].to_numpy()
print(f"malignant cells ({MALIG_CALL}):", int(mal.sum()),
      "| undefined:", int(adata.obs["malig_undefined"].sum()),
      "| mal_tcr (ALICE, sensitivity):", int(adata.obs["mal_tcr"].sum()),
      "| cnv_arm_malignant (nb23):", int(adata.obs["cnv_arm_malignant"].sum()))

# ---- cached arm-CNV matrix (nb30 v4) + its held-out healthy diploid null ----
# The v4 inferCNV pass relabels half the shared reference as `ref_null` and renames those rows
# `<run>|NULL|<cell>`, so they traverse the identical estimator as the tumour cells and ARE the
# empirical diploid null. nb31 used to discard them at the reindex below; they now calibrate the
# subclone gates (§A.1) and anchor the cohort arm-profile plot (§A figures).
_arm_raw = pd.read_parquet(ARM).set_index("obs_name")
ARM_COLS = S.arm_order([c for c in _arm_raw.columns if c != "donor"])
_arm_query, _null_arm = S.split_null_rows(_arm_raw)
NULL_ARMS = _null_arm[[*ARM_COLS, "ref_source", "donor"]]
NULL_X = NULL_ARMS[ARM_COLS].fillna(0.0).to_numpy(dtype=float)   # built once, reused per donor
del _arm_raw, _null_arm
arm_df = _arm_query.reindex(adata.obs_names)
print("arm matrix:", _arm_query.shape, "| healthy diploid null:", NULL_X.shape)
print(NULL_ARMS["ref_source"].value_counts().to_string())

# per-donor arm-CNV coverage (cache may omit donors that lacked a diploid reference for inferCNV).
cov_by_donor = (arm_df[ARM_COLS].notna().any(axis=1)
                .groupby(adata.obs["donor"].astype(str)).mean())

# ---- sample selection: eligibility for subclone detection (>= MIN_MAL malignant cells + arm CNV) ----
elig_tbl = (adata.obs.assign(_m=mal)
            .groupby("donor", observed=True)
            .agg(study=("study", "first"), disease=("disease", "first"),
                 n_cells=("_m", "size"), n_malignant=("_m", "sum"))
            .reset_index())
elig_tbl["n_malignant"] = elig_tbl["n_malignant"].astype(int)
elig_tbl["arm_cnv_cov"] = elig_tbl["donor"].map(lambda d: float(cov_by_donor.get(d, 0.0)))


def _exclude_reason(row):
    if str(row["disease"]) == "HC":
        return "healthy control — no malignant compartment"
    if row["n_malignant"] == 0:
        return f"no {MALIG_CALL}-malignant cells"
    if row["n_malignant"] < MIN_MAL:
        return f"<{MIN_MAL} malignant cells — too few for subclone KMeans"
    if row["arm_cnv_cov"] == 0.0:
        # mal_cluster admits donors with no V(D)J, but subclone detection runs off the arm matrix.
        return "no arm-CNV coverage — donor absent from the inferCNV arm cache (nb23)"
    return ""


elig_tbl["reason_excluded"] = elig_tbl.apply(_exclude_reason, axis=1)
elig_tbl["eligible"] = elig_tbl["reason_excluded"].eq("")
elig_tbl = elig_tbl.sort_values(["eligible", "n_malignant"], ascending=[False, False]).reset_index(drop=True)
elig_tbl.to_csv(ELIG_CSV, index=False)

ELIG = sorted(elig_tbl.loc[elig_tbl["eligible"], "donor"])
n_arm_excl = int(elig_tbl["reason_excluded"].str.startswith("no arm-CNV").sum())
print(f"cohort donors: {len(elig_tbl)} | eligible (>= {MIN_MAL} malignant + arm CNV): {len(ELIG)}"
      f" | excluded: {int((~elig_tbl['eligible']).sum())} (of which {n_arm_excl} for 0 arm-CNV coverage)")
print("arms:", len(ARM_COLS), "| wrote", ELIG_CSV)
```

```
loaded skin_T_annotated.h5ad (395233, 40821)

/tmp/ipykernel_2298638/1309773230.py:18: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  .fillna(False).astype(bool).to_numpy())
/tmp/ipykernel_2298638/1309773230.py:26: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  adata.obs[c] = (s.fillna(False).astype(bool).to_numpy() if c in _bool
/tmp/ipykernel_2298638/1309773230.py:26: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  adata.obs[c] = (s.fillna(False).astype(bool).to_numpy() if c in _bool
/tmp/ipykernel_2298638/1309773230.py:26: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  adata.obs[c] = (s.fillna(False).astype(bool).to_numpy() if c in _bool
/tmp/ipykernel_2298638/1309773230.py:26: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  adata.obs[c] = (s.fillna(False).astype(bool).to_numpy() if c in _bool

malignant cells (TCR-ALICE): 76349 | cnv_arm_malignant (nb30): 65247
arm matrix: (236233, 42) | healthy diploid null: (209900, 41)
ref_source
healthy_pbmc    209621
hc_atlas           279
cohort donors: 82 | eligible (>= 200 malignant + arm CNV): 25 | excluded: 57 (of which 0 for 0 arm-CNV coverage)
arms: 41 | wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/subclone_donor_eligibility_v3.csv
```

## §A.1 — Calculation (compute once, no plots)

```python  # cell 6
# Per-donor TCR founder + ALICE <=1-aa variant family (mirror nb30 PT35 exception).
PT35 = "Li2024_atlas__PT35"
mal_obs = adata.obs[mal].copy()
clono = A.clonotype_table(mal_obs, group="donor")
founder_sets, family_sets = {}, {}
for d, sub in clono.groupby("donor", observed=True):
    if d == PT35:
        seeds = set(sub.sort_values("n_cells", ascending=False)["cdr3"].iloc[:2])
    elif sub["is_founder"].astype(bool).any():
        seeds = set(sub.loc[sub["is_founder"].astype(bool), "cdr3"])
    elif len(sub):
        seeds = {sub.sort_values("n_cells", ascending=False)["cdr3"].iloc[0]}
    else:
        seeds = set()
    founder_sets[d] = seeds
    family_sets[d] = A.founder_family(sub, seeds=seeds) if seeds else set()
n_var = {d: len(family_sets[d] - founder_sets[d]) for d in founder_sets}
print("donors with >=1 ALICE <=1-aa TCR-beta variant:",
      sum(v > 0 for v in n_var.values()), "/", len(n_var))
```

```
donors with >=1 ALICE <=1-aa TCR-beta variant: 0 / 17
```

```python  # cell 7
# per-donor CNV subclones (KMeans on the arm matrix, gates null-anchored) + summary + dominant subclone.
sub_label = pd.Series("", index=adata.obs_names, dtype=object)
summary_rows, centroids = [], {}
for d in ELIG:
    cells = adata.obs_names[mal & (adata.obs["donor"].astype(str).values == d)]
    Ad = arm_df.reindex(cells)[ARM_COLS]
    lab, meta = S.detect_subclones(Ad, k_max=K_MAX, min_sub=MIN_SUB, seed=SEED,
                                   sil_min=SIL_MIN, min_arm_delta=MIN_ARM_DELTA,
                                   null_arms=NULL_X, null_z=NULL_Z, null_draws=NULL_DRAWS)
    sub_label.loc[cells] = [f"{d}_s{c}" for c in lab]
    cen = meta["centroids"]; cen.index = [f"{d}_s{i}" for i in range(len(cen))]
    centroids[d] = cen
    summary_rows.append({"donor": d, "n_malig": len(cells), "k": meta["k"],
                         "silhouette": meta["silhouette"], "max_arm_delta": meta["max_arm_delta"],
                         "centroid_spread": meta["centroid_spread"],
                         "frac_cnv_confirmed": round(float(adata.obs.loc[cells, "cnv_arm_malignant"].mean()), 3),
                         "sizes": ",".join(map(str, meta["sizes"])), "n_tcr_variants": n_var.get(d, 0),
                         "sil_thr": meta["sil_thr"], "arm_delta_thr": meta["arm_delta_thr"],
                         "sil_margin": meta["sil_margin"], "null_calibrated": meta["null_calibrated"]})

adata.obs["cnv_subclone"] = sub_label.values
vc = adata.obs.loc[sub_label.values != "", "cnv_subclone"].value_counts()
dom_sub = {d: vc[[s for s in vc.index if s.rsplit("_s", 1)[0] == d]].idxmax() for d in ELIG}
adata.obs["is_dominant_subclone"] = adata.obs["cnv_subclone"].isin(set(dom_sub.values())).values

summary = pd.DataFrame(summary_rows)
meta_cols = ["study", "disease", "disease_stage", "entity"]
summary = summary.merge(adata.obs[["donor", *meta_cols]].astype(str).drop_duplicates("donor"),
                        on="donor", how="left")
n_multi = int((summary["k"] > 1).sum())
print(f">= 2 CNV subclones: {n_multi}/{len(summary)} = {n_multi/max(1,len(summary)):.0%} donors (paper: 84%)")
print(f"TCR/CNV concordance (mean frac_cnv_confirmed): {summary['frac_cnv_confirmed'].mean():.2f}")
print(f"gates: null-anchored (mean + {NULL_Z}sd of a size/k-matched diploid split) for "
      f"{int(summary['null_calibrated'].sum())}/{len(summary)} donors")
summary.sort_values("k", ascending=False)
```

```
>= 2 CNV subclones: 2/25 = 8% donors (paper: 84%)
TCR/CNV concordance (mean frac_cnv_confirmed): 0.59
gates: null-anchored (mean + 3.0sd of a size/k-matched diploid split) for 25/25 donors

                  donor  n_malig  k  silhouette  max_arm_delta  \
2              D1__P171     5235  2      0.1829         0.0959   
18  Li2024_atlas__CTCL8     5314  2      0.0931         0.0807   
0              B5__PT09     5253  1         NaN         0.0000   
3              D1__P204     2343  1         NaN         0.0000   
1              B5__PT23     3702  1         NaN         0.0000   
5              D1__P312      724  1         NaN         0.0000   
6               D1__P76     4599  1         NaN         0.0000   
7               D1__PGS     2259  1         NaN         0.0000   
4              D1__P311     3815  1         NaN         0.0000   
8              D3__P192     3028  1         NaN         0.0000   
9              D3__P196      434  1         NaN         0.0000   
11  Li2024_atlas__CTCL1     7090  1         NaN         0.0000   
10            D5__MFIVB     1036  1         NaN         0.0000   
13  Li2024_atlas__CTCL3     8162  1         NaN         0.0000   
14  Li2024_atlas__CTCL4      743  1         NaN         0.0000   
15  Li2024_atlas__CTCL5     6604  1         NaN         0.0000   
12  Li2024_atlas__CTCL2     1840  1         NaN         0.0000   
16  Li2024_atlas__CTCL6     3119  1         NaN         0.0000   
17  Li2024_atlas__CTCL7     3122  1         NaN         0.0000   
19   Li2024_atlas__PT11     1619  1         NaN         0.0000   
20   Li2024_atlas__PT35     2865  1         NaN         0.0000   
21   Li2024_atlas__PT47      257  1         NaN         0.0000   
22   Li2024_atlas__PT50      413  1         NaN         0.0000   
23   Li2024_atlas__PT53     1240  1         NaN         0.0000   
24   Li2024_atlas__PT56     1156  1         NaN         0.0000   

    centroid_spread  frac_cnv_confirmed      sizes  n_tcr_variants  sil_thr  \
2            0.1486               0.990  4181,1054               0   0.0763   
18           0.1086               0.854  2678,2636               0   0.0763   
0            0.0000               0.011       5253               0   0.0763   
3            0.0000               0.066       2343               0   0.0735   
1            0.0000               0.708       3702               0   0.0707   
5            0.0000               0.419        724               0   0.0816   
6            0.0000               0.604       4599               0   0.0707   
7            0.0000               0.946       2259               0   0.0735   
4            0.0000               0.851       3815               0   0.0707   
8            0.0000               0.280       3028               0   0.0726   
9            0.0000               0.283        434               0   0.0783   
11           0.0000               0.958       7090               0   0.0697   
10           0.0000               0.911       1036               0   0.0752   
13           0.0000               0.668       8162               0   0.0697   
14           0.0000               0.048        743               0   0.0816   
15           0.0000               0.911       6604               0   0.0763   
12           0.0000               0.359       1840               0   0.0735   
16           0.0000               0.074       3119               0   0.0726   
17           0.0000               0.132       3122               0   0.0726   
19           0.0000               0.991       1619               0   0.0742   
20           0.0000               0.984       2865               0   0.0726   
21           0.0000               0.984        257               0   0.0836   
22           0.0000               0.591        413               0   0.0781   
23           0.0000               0.943       1240               0   0.0742   
24           0.0000               0.139       1156               0   0.0752   

    arm_delta_thr  sil_margin  null_ca
... [3236 more chars truncated]
```

```python  # cell 8
# trunk (pan-clonal, ancestral) vs branch (subclone-divergent) arm events per donor.
# Restored from the nb23 framework (Herrmann/Iyer 2025 §3); feeds §B.6 arm x activity crosses.
benign_mask = ((~adata.obs["tcr_is_dominant_clone"].to_numpy())
               & (adata.obs["cell_type"].astype(str).values != "tumor_cell"))
tb_rows = []
for d in ELIG:
    cen = centroids[d]
    dmask = adata.obs["donor"].astype(str).values == d
    A_all = arm_df.loc[adata.obs_names[mal & dmask], ARM_COLS].dropna(how="all")
    A_ben = arm_df.loc[adata.obs_names[benign_mask & dmask], ARM_COLS].dropna(how="all")
    if len(A_all) < 20 or len(A_ben) < 20:
        continue
    clone_events = S.call_arm_events(A_all, A_ben, z_thr=Z_THR, eps=EPS)
    tb = S.trunk_branch(clone_events, cen, branch_delta=BRANCH_DELTA)
    trunk_arms = ";".join(f"{a}{'+' if tb['trunk_signed'][a] > 0 else '-'}" for a in tb["trunk"])
    branch_arms = ";".join(
        f"{a}{'+' if cen[a].loc[cen[a].abs().idxmax()] > 0 else '-'}" for a in tb["branch"])
    row = summary.loc[summary["donor"] == d].iloc[0]
    tb_rows.append({"donor": d, "k": int(row["k"]), "trunk_arms": trunk_arms,
                    "branch_arms": branch_arms, "n_trunk": len(tb["trunk"]),
                    "n_branch": len(tb["branch"]), "study": row.get("study", ""),
                    "disease": row.get("disease", ""), "disease_stage": row.get("disease_stage", "")})
trunk_branch_df = pd.DataFrame(tb_rows)
trunk_branch_df.to_csv(TRUNKBRANCH_CSV, index=False)
print("wrote", TRUNKBRANCH_CSV, trunk_branch_df.shape)
trunk_branch_df.head()
```

```
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/subclone_trunk_branch_v3.csv (25, 9)

      donor  k trunk_arms                                        branch_arms  \
0  B5__PT09  1                                                                 
1  B5__PT23  1                                                                 
2  D1__P171  2             chr5p-;chr7p+;chr10p+;chr13q-;chr18p+;chr20p+;...   
3  D1__P204  1                                                                 
4  D1__P311  1                                                                 

   n_trunk  n_branch            study disease disease_stage  
0        0         0         buus2025      SS            NA  
1        0         0         buus2025      SS            NA  
2        0         7  chennareddy2025      MF      advanced  
3        0         0  chennareddy2025      MF      advanced  
4        0         0  chennareddy2025      MF      advanced
```

```python  # cell 9
# transcriptomic MAJOR (Leiden on MRVI mu embedding) + CNV MINOR tracks -> nested subclones.
_lat = LATENT if LATENT in adata.obsm else ("X_scVI" if "X_scVI" in adata.obsm else "X_mrvi_u")
tracks, umap_by_donor, mm = S.major_minor_tracks(
    adata, _lat, mal, ELIG, cnv_col="cnv_subclone",
    leiden_res=LEIDEN_RES, n_neighbors=N_NEIGHBORS, seed=SEED)
for col in ["major_subclone", "minor_subclone", "subclone_label"]:
    adata.obs[col] = tracks[col].reindex(adata.obs_names).fillna("").astype(str).values
adata.obs["sub_umap1"] = tracks["sub_umap1"].reindex(adata.obs_names).values
adata.obs["sub_umap2"] = tracks["sub_umap2"].reindex(adata.obs_names).values
summary = summary.merge(mm[["donor", "n_major", "n_minor", "has_major", "has_minor",
                            "nmi_major_vs_minor"]], on="donor", how="left")
n_both = int((mm["has_major"] & mm["has_minor"]).sum())
print(f"MAJOR = Leiden (res={LEIDEN_RES}) on {_lat}; MINOR = CNV subclones "
      f"| clones with both tracks split: {n_both}/{len(mm)}")
```

```
/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/subclone_helpers.py:286: FutureWarning: In the future, the default backend for leiden will be igraph instead of leidenalg.

 To achieve the future defaults please pass: flavor="igraph" and n_iterations=2.  directed must also be False to work with igraph's implementation.
  sc.tl.leiden(sub, resolution=leiden_res, random_state=seed, key_added="_leiden")

MAJOR = Leiden (res=0.2) on X_mrvi_u; MINOR = CNV subclones | clones with both tracks split: 2/25
```

```python  # cell 10
# cohort-wide NESTED subclones (transcriptomic MAJOR -> CNV MINOR within each major).
nested, split_summary = S.nested_subclone_labels(
    umap_by_donor, arm_df, ARM_COLS, ELIG,
    k_max=K_MAX, min_sub=MIN_SUB, seed=SEED, sil_min=SIL_MIN, min_arm_delta=MIN_ARM_DELTA,
    null_arms=NULL_X, null_z=NULL_Z, null_draws=NULL_DRAWS)
adata.obs["nested_subclone"] = nested.reindex(adata.obs_names).fillna("").astype(str).values
n_split = int((split_summary["k"] > 1).sum())
print(f"nested subclones: {nested.nunique()} across {len(ELIG)} eligible donors "
      f"| majors that split further on CNV: {n_split}/{len(split_summary)}")
```

```
nested subclones: 92 across 25 eligible donors | majors that split further on CNV: 2/90
```

```python  # cell 11
# malignant cells, log-normalized from raw counts (scoring + DE) + panels + dominant nested.
mal_ad = adata[adata.obs["cnv_subclone"].astype(str).values != ""].copy()
mal_ad.X = mal_ad.layers["raw_counts"].copy()
sc.pp.normalize_total(mal_ad, target_sum=1e4); sc.pp.log1p(mal_ad)
mal_ad.obs["nested_subclone"] = adata.obs.loc[mal_ad.obs_names, "nested_subclone"].astype(str).values

vc = mal_ad.obs["nested_subclone"].value_counts(); vc = vc[vc.index != ""]
by_donor = {}
for s, n in vc.items():
    by_donor.setdefault(s.rsplit("_", 1)[0], []).append((s, n))
dom_nested = {d: max(v, key=lambda t: t[1])[0] for d, v in by_donor.items()}
multi_nested = sorted(d for d, v in by_donor.items() if len(v) >= 2)
mal_ad.obs["is_dominant_nested"] = mal_ad.obs["nested_subclone"].isin(set(dom_nested.values())).values

panels = S.present_panels(mal_ad, S.PAPER_PANELS)
print("malignant cells for DE/scoring:", mal_ad.shape, "| donors:", mal_ad.obs["donor"].nunique())
print(f"nested subclones: {vc.shape[0]} | donors with >=2 nested subclones: {len(multi_nested)}")
```

```
WARNING: adata.X seems to be already log-transformed.
malignant cells for DE/scoring: (75972, 40821) | donors: 25
nested subclones: 92 | donors with >=2 nested subclones: 25
```

```python  # cell 12
# functional-program scores + skin-layer tropism (nested axis) + persist.
added = S.score_programs(mal_ad, prefix="prog_", cell_cycle=True, seed=SEED)
prog_cols = [c for c in added if c.startswith("prog_")]

layer = mal_ad.obs["tissue"].astype(str).str.lower()
is_epi = layer.str.contains("epiderm"); is_derm = layer.str.contains("derm") & ~is_epi
mal_ad.obs["skin_layer"] = np.where(is_epi, "epidermis", np.where(is_derm, "dermis", "other"))
trop_tbl, cell_trop = S.classify_subclone_tropism(mal_ad.obs, "nested_subclone", min_cells=MIN_LAYER_CELLS)
mal_ad.obs["tropism"] = cell_trop.values

MENU = S.layer_split_donors(mal_ad.obs, "nested_subclone", min_cells=MIN_LAYER_CELLS)

adata.obs[["donor", "cnv_subclone", "nested_subclone", "is_dominant_subclone"]].to_parquet(SUBCLONE_PARQUET)
summary.to_csv(SUMMARY_CSV, index=False)
print(f"tropism classes: {trop_tbl['tropism'].value_counts().to_dict()}")
print("wrote", SUBCLONE_PARQUET.name, "+", SUMMARY_CSV.name)
```

```
tropism classes: {'unclassified': 52, 'dermal': 16, 'epidermal': 15, 'mixed': 9}
wrote subclones_v3.parquet + subclone_summary_v3.csv
```

```python  # cell 13
# Export gene signatures for downstream SPATIAL scoring (nb24/25 Visium).
import json
TOP_SUB, TOP_MAL = 30, 50
signatures = {}
for d in multi_nested:
    sub = mal_ad[mal_ad.obs["donor"].astype(str).values == d].copy()
    if sub.obs["nested_subclone"].nunique() < 2:
        continue
    de = S.subclone_markers_wilcoxon(sub, "nested_subclone", n_genes=sub.n_vars)
    de = de[(de["pval_adj"] < 0.05) & (de["log2fc"] > 0)]
    for s, g in de.sort_values("score", ascending=False).groupby("subclone", observed=True):
        genes = g["gene"].head(TOP_SUB).tolist()
        if len(genes) >= 5:
            signatures[f"subclone__{s}"] = genes

rng = np.random.default_rng(SEED)
m = adata.obs["is_malig"].to_numpy()
idx_m, idx_b = np.where(m)[0], np.where(~m)[0]
CAP = 20000
take = np.concatenate([rng.choice(idx_m, min(CAP, idx_m.size), replace=False),
                       rng.choice(idx_b, min(CAP, idx_b.size), replace=False)])
mvb = adata[take].copy()
mvb.X = mvb.layers["raw_counts"].copy()
sc.pp.normalize_total(mvb, target_sum=1e4); sc.pp.log1p(mvb)
mvb.obs["malig_grp"] = np.where(mvb.obs["is_malig"].to_numpy(), "malignant", "benign")
de_mal = S.subclone_markers_wilcoxon(mvb, "malig_grp", n_genes=mvb.n_vars)
de_mal = de_mal[(de_mal["subclone"] == "malignant") & (de_mal["pval_adj"] < 0.05) & (de_mal["log2fc"] > 0)]
signatures["malignant_overall"] = de_mal.sort_values("score", ascending=False)["gene"].head(TOP_MAL).tolist()
del mvb; gc.collect()

for cat, genes in S.PAPER_PANELS.items():
    signatures[f"panel__{cat}"] = list(dict.fromkeys(genes))
for prog, genes in S.PROGRAM_SETS.items():
    signatures[f"program__{prog}"] = list(dict.fromkeys(genes))

SIGN_JSON.write_text(json.dumps(signatures, indent=1))
print(f"wrote {SIGN_JSON.name}: {len(signatures)} signatures")
```

```
WARNING: adata.X seems to be already log-transformed.
wrote subclone_signatures_v3.json: 109 signatures
```

## §A — Cohort overview & per-sample figures

Cohort CNV-subclone arm profiles, prevalence of ≥2 subclones, program-score heatmap. For a per-sample
UMAP + arm-CNV view, set `SAMPLE` to any donor in `MENU` and call `S.plot_nested_sample`.

```python  # cell 15
print(f"{len(MENU)} samples with epidermis/dermis separation:")
print(MENU.to_string(index=False))
S.plot_cohort_arm_profiles(centroids, ELIG, ARM_COLS, null_arms=NULL_ARMS, vlim=Z_VLIM,
                           save=FIG_DIR / "subclone_v4_arm_profiles_cohort.png")
S.plot_subclone_prevalence(summary, k_max=K_MAX, n_multi=n_multi, save=FIG_DIR / "subclone_v4_prevalence.png")
S.plot_program_heatmap(mal_ad.obs, "nested_subclone", prog_cols,
                       donor_order=multi_nested, save=FIG_DIR / "subclone_v4_programs.png")
```

```
10 samples with epidermis/dermis separation:
              donor  n_subclones  n_epi  n_derm   subclones
Li2024_atlas__CTCL5            6   4603    2001 A,B,C,D,E,F
           B5__PT09            5   3078    2175   A,B,C,D,E
Li2024_atlas__CTCL2            5   1128     712   A,B,C,D,E
Li2024_atlas__CTCL8            5    552    4762 A1,A2,B,C,D
Li2024_atlas__CTCL3            4   2846    5316     A,B,C,D
Li2024_atlas__CTCL1            4   2227    4863     A,B,C,D
Li2024_atlas__CTCL6            3   2851     268       A,B,C
Li2024_atlas__CTCL4            3    697      46       A,B,C
           B5__PT23            3    130    3572       A,B,C
Li2024_atlas__CTCL7            2   2833     289         A,B

<Figure size 1320x487.2 with 2 Axes>
<Figure size 600x360 with 1 Axes>
<Figure size 600x1766.4 with 2 Axes>
```

```python  # cell 16
# Per-sample subclone structure — pick ONE donor from MENU above.
SAMPLE = MENU["donor"].iloc[0] if len(MENU) else (ELIG[0] if ELIG else None)
if SAMPLE is not None:
    print("selected:", SAMPLE)
    S.plot_nested_sample(adata, SAMPLE, umap_by_donor, arm_df, ARM_COLS, mal, fig_dir=FIG_DIR)
```

```
selected: Li2024_atlas__CTCL5

<Figure size 2688x768 with 8 Axes>
```

# §B — TCR / co-stimulation signaling across malignant clones & subclones

Where in the TCR/costim cascade each malignant population concentrates dysregulation (module scores +
individual genes), vs lineage-matched reactive-CD4, cross-checked against arm-level inferCNV.
Reuses the in-memory object from §A; rebuilds `X` from `raw_counts` and normalizes the whole object.

```python  # cell 18
import warnings
import tcr_signaling_helpers as th
importlib.reload(th)

DATA = OUT_DIR
DE_DIR_S = DATA / "tcr_signaling_de"; DE_DIR_S.mkdir(exist_ok=True)
SUBCLONES_V3 = SUBCLONE_PARQUET
ARM_CNV_V3   = ARM
TRUNK_BRANCH_V3 = TRUNKBRANCH_CSV
MALIG_COL = "is_malig"          # = MALIG_CALL (mal_cluster) where defined; see §A.0
SUBCLONE_COL = "nested_subclone"
GROUPS = ("malignant", "reactive_CD4")
```

```python  # cell 19
# rebuild X from raw_counts on the full object, normalize+log1p (mirror nb26 §0)
try:
    del mal_ad
except NameError:
    pass
gc.collect()
adata.X = adata.layers["raw_counts"].copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print("normalized full object for signaling scoring:", adata.shape)
```

```
WARNING: adata.X seems to be already log-transformed.
normalized full object for signaling scoring: (395233, 40821)
```

```python  # cell 20
# join subclone assignments + per-cell arm-CNV
sub = pd.read_parquet(SUBCLONES_V3)
if "cell_id" in sub.columns:
    sub = sub.set_index("cell_id")
for c in ["nested_subclone", "cnv_subclone", "is_dominant_subclone"]:
    if c in sub.columns:
        adata.obs[c] = sub[c].reindex(adata.obs_names)

arm = pd.read_parquet(ARM_CNV_V3)
if "obs_name" in arm.columns:
    arm = arm.set_index("obs_name")
ARM_COLS = [c for c in arm.columns if c.startswith("chr")]
arm = arm.reindex(adata.obs_names)
print("arm cnv coverage:", arm[ARM_COLS].notna().any(axis=1).mean())
```

```
arm cnv coverage: 0.5977056571693153
```

```python  # cell 21
# groups: malignant vs lineage-matched reactive CD4
malig = adata.obs[MALIG_COL].astype("boolean").fillna(False).to_numpy()
ct = adata.obs["cell_type_T2"].astype(str)
grp = np.where(malig, "malignant",
               np.where((~malig) & (ct == "CD4"), "reactive_CD4", "other"))
adata.obs["malig_group"] = pd.Categorical(grp, categories=["malignant", "reactive_CD4", "other"])

tl = adata.obs["tissue"].astype(str).str.lower()
adata.obs["skin_layer"] = np.select(
    [tl.str.contains("epi"), tl.str.contains("derm")], ["epidermis", "dermis"], default="other")

print(adata.obs["malig_group"].value_counts())
if "stage_class" in adata.obs:
    print(adata.obs["stage_class"].value_counts())
print(adata.obs["skin_layer"].value_counts())
```

```
malig_group
reactive_CD4    262349
malignant        76349
other            56535
Name: count, dtype: int64
stage_class
advanced    200444
early       138308
unknown      54589
HC            1892
Name: count, dtype: int64
skin_layer
other        215358
dermis       134044
epidermis     45831
Name: count, dtype: int64
```

## §B.1 — Score modules (+ `_nolabile` sensitivity re-score)

```python  # cell 23
cov = th.module_coverage(adata)
try:
    display(cov)
except NameError:
    print(cov)

score_cols = th.score_tcr_modules(adata, seed=SEED, prefix="sig_")
score_cols_nl = th.score_tcr_modules(adata, seed=SEED, prefix="sig_nolabile_", drop=th.DISSOC_LABILE)
print(f"{len(score_cols)} module scores added")

mod_scores = adata.obs[score_cols].copy()
genes_all = sorted({g for gs in th.TCR_MODULES.values() for g in gs} & set(adata.var_names))
gene_expr = th._dense_gene_frame(adata, genes_all)
ACT_COLS = [f"sig_{m}" for m in th.ACTIVITY_MODULES if f"sig_{m}" in score_cols]
print(len(genes_all), "module genes present;", len(ACT_COLS), "activity modules")
```

```
module  n_present  n_total missing  context_only
0    COSTIM_coinhibitory_receptors         13       13                 False
1   COSTIM_ligands_on_TME_partners         13       13                 False
2         COSTIM_pi3k_akt_activity          7        7                 False
3       COSTIM_pi3k_akt_components         13       13                  True
4      COSTIM_receptors_activating         13       13                 False
5                    L0_coreceptor          3        3                  True
6           L0_pan_t_identity_loss          5        5                 False
7            L0_tcr_cd3_components          6        6                  True
8              L1_proximal_kinases         11       11                  True
9         L2_gef_gtpase_components          9        9                  True
10   L2_lat_signalosome_components          8        8                  True
11        L2_plcg1_node_components          5        5                  True
12         L3a_cbm_nfkb_components         17       17                  True
13               L3a_nfkb_activity         19       19                 False
14     L3b_calcium_nfat_components         16       16                  True
15               L3b_nfat_activity         12       12                 False
16           L3c_ap1_mapk_activity         13       13                 False
17         L3c_ras_mapk_components         14       14                  True
18                    L4_ieg_acute         19       19                 False
19        L5_chronicity_exhaustion         12       12                 False
20           L5_sustained_tf_layer         16       16                 False
21                 NEGREG_rheostat         21       21                 False
22                SIGNAL3_jak_stat         16       16                 False
23 module scores added
225 module genes present; 4 activity modules
```

## §B.2 — Within-sample: malignant vs reactive-CD4 (per donor)

```python  # cell 25
grp_s = adata.obs["malig_group"]
donor_s = adata.obs["donor"].astype(str)
keep = grp_s.isin(GROUPS)

mod_stats = th.wilcoxon_two_group(mod_scores[keep.values], score_cols, grp_s[keep], GROUPS,
                                  level="module", donor_series=donor_s[keep])
gene_stats = th.wilcoxon_two_group(gene_expr[keep.values], genes_all, grp_s[keep], GROUPS,
                                   level="gene", donor_series=donor_s[keep])
mod_stats["module"] = mod_stats["feature"].str.replace("sig_", "", regex=False)
mod_stats["context_only"] = mod_stats["module"].isin(th.CONTEXT_ONLY)
mod_stats.to_csv(DE_DIR_S / "within_sample_module_v4.csv", index=False)
gene_stats.to_csv(DE_DIR_S / "within_sample_gene_v4.csv", index=False)
mod_stats.head()
```

```
feature   level    n1    n2          U  \
0          sig_L0_tcr_cd3_components  module  7090  1361  5701072.0   
1         sig_L0_pan_t_identity_loss  module  7090  1361  4258008.0   
2                  sig_L0_coreceptor  module  7090  1361  2852785.0   
3            sig_L1_proximal_kinases  module  7090  1361  5248449.0   
4  sig_L2_lat_signalosome_components  module  7090  1361  5830156.0   

               p  cliffs_delta    mean_a    mean_b  mean_diff    log2fc  \
0   2.166020e-26      0.181632  0.278096  0.147625   0.130471  0.913645   
1   6.221618e-12     -0.117465 -0.271366 -0.133653  -0.137713       NaN   
2  1.908266e-126     -0.408718 -0.313469 -0.183795  -0.129674       NaN   
3   2.754726e-07      0.087819  0.282701  0.246599   0.036102  0.197108   
4   3.281981e-34      0.208386  0.357261  0.244717   0.112544  0.545865   

                 donor    group_a       group_b            fdr  \
0  Li2024_atlas__CTCL1  malignant  reactive_CD4   4.981846e-26   
1  Li2024_atlas__CTCL1  malignant  reactive_CD4   7.949845e-12   
2  Li2024_atlas__CTCL1  malignant  reactive_CD4  2.194506e-125   
3  Li2024_atlas__CTCL1  malignant  reactive_CD4   3.017081e-07   
4  Li2024_atlas__CTCL1  malignant  reactive_CD4   1.078365e-33   

            direction                         module  context_only  
0     up_in_malignant          L0_tcr_cd3_components          True  
1  up_in_reactive_CD4         L0_pan_t_identity_loss         False  
2  up_in_reactive_CD4                  L0_coreceptor          True  
3     up_in_malignant            L1_proximal_kinases          True  
4     up_in_malignant  L2_lat_signalosome_components          True
```

```python  # cell 26
# module x donor dotplot + clustered heatmap
plot_df = mod_stats.copy(); plot_df["feature"] = plot_df["module"]
fig, ax = th.module_score_dotplot(plot_df, group_col="donor", feature_col="feature",
                                  title="Malignant vs reactive-CD4 module scores (per donor)")
fig.savefig(FIG_DIR / "tcr_signaling_within_sample_module_dotplot.png", dpi=150, bbox_inches="tight")

def _first(s): return s.astype(str).iloc[0]
def _dom_layer(s):
    s = s[s.isin(["epidermis", "dermis"])]
    return s.value_counts().idxmax() if len(s) else np.nan

_ob = adata.obs.groupby("donor", observed=True)
donor_annot = pd.DataFrame({"study": _ob["study"].agg(_first),
                            "stage": _ob["stage_class"].agg(_first) if "stage_class" in adata.obs else "",
                            "layer": _ob["skin_layer"].agg(_dom_layer)})
STRIP_PALS = {"stage": {"early": "#4daf4a", "advanced": "#e41a1c"},
              "layer": {"epidermis": "#377eb8", "dermis": "#ff7f00"}}
cg = th.module_donor_clustermap(mod_stats, group_col="donor", feature_col="module",
                                value_col="cliffs_delta", col_annot=donor_annot,
                                annot_palettes=STRIP_PALS,
                                title="Malignant vs reactive-CD4 module Δ (donors clustered)")
cg.savefig(FIG_DIR / "tcr_signaling_within_sample_module_clustermap.png", dpi=150, bbox_inches="tight")
```

```
<Figure size 2220x948 with 2 Axes>
<Figure size 1080x840 with 5 Axes>
```

```python  # cell 27
# gene-level consistency across donors + per-family lollipops + activity-module violins
g = (gene_stats.assign(sig=gene_stats["fdr"] < 0.05)
     .groupby("feature")
     .agg(cliffs_delta=("cliffs_delta", "mean"), n_sig=("sig", "sum"), n_donor=("donor", "nunique"))
     .reset_index())
g["module"] = g["feature"].map(lambda x: th.GENE_TO_MODULE.get(x, [""])[0])
top = g.reindex(g["cliffs_delta"].abs().sort_values(ascending=False).index).head(40)
ax = th.lollipop_effects(top, effect="cliffs_delta", label="feature", fdr_col=None,
                         title="Top genes: malignant vs reactive-CD4 (mean Cliff's δ)", figsize=(5, 8))
ax.figure.savefig(FIG_DIR / "tcr_signaling_within_sample_gene_lollipop.png", dpi=150, bbox_inches="tight")

fam_figs = th.lollipop_per_family(g, effect="cliffs_delta", feature_col="feature", fdr_col=None,
                                  title_prefix="mal vs reactive-CD4: ")
for _m, _fig in fam_figs.items():
    _fig.savefig(FIG_DIR / f"tcr_signaling_within_sample_lollipop_{_m}.png", dpi=150, bbox_inches="tight")

obs_v = adata.obs.loc[keep.values, ["malig_group", *ACT_COLS]].copy()
for col in ACT_COLS:
    fig, _ = th.module_violin(obs_v, col, split_col="malig_group", order=list(GROUPS), title=col)
    fig.savefig(FIG_DIR / f"tcr_signaling_within_sample_violin_{col}.png", dpi=150, bbox_inches="tight")
print(len(fam_figs), "family lollipops;", len(ACT_COLS), "activity-module violins")
```

```
/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/tcr_signaling_helpers.py:503: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
  fig, ax = plt.subplots(figsize=(width, h))

23 family lollipops; 4 activity-module violins

<Figure size 600x960 with 1 Axes>
<Figure size 540x309.6 with 1 Axes>
<Figure size 540x276 with 1 Axes>
<Figure size 540x216 with 1 Axes>
<Figure size 540x477.6 with 1 Axes>
<Figure size 540x376.8 with 1 Axes>
<Figure size 540x410.4 with 1 Axes>
<Figure size 540x276 with 1 Axes>
<Figure size 540x679.2 with 1 Axes>
<Figure size 540x746.4 with 1 Axes>
<Figure size 540x645.6 with 1 Axes>
<Figure size 540x511.2 with 1 Axes>
<Figure size 540x578.4 with 1 Axes>
<Figure size 540x544.8 with 1 Axes>
<Figure size 540x746.4 with 1 Axes>
<Figure size 540x645.6 with 1 Axes>
<Figure size 540x511.2 with 1 Axes>
<Figure size 540x544.8 with 1 Axes>
<Figure size 540x544.8 with 1 Axes>
<Figure size 540x544.8 with 1 Axes>
<Figure size 540x544.8 with 1 Axes>
<Figure size 540x343.2 with 1 Axes>
<Figure size 540x645.6 with 1 Axes>
<Figure size 540x813.6 with 1 Axes>
<Figure size 720x420 with 1 Axes>
<Figure size 720x420 with 1 Axes>
<Figure size 720x420 with 1 Axes>
<Figure size 720x420 with 1 Axes>
```

## §B.3 — Within malignant: each nested subclone vs its donor's reactive-CD4

```python  # cell 29
sub_series = adata.obs[SUBCLONE_COL].astype(str)
grp_all = adata.obs["malig_group"].astype(str)
don_all = adata.obs["donor"].astype(str)

subclone_mod = th.subclone_vs_control_stats(mod_scores, score_cols, sub_series, grp_all, don_all,
                                            level="module", control_group="reactive_CD4")
subclone_gene = th.subclone_vs_control_stats(gene_expr, genes_all, sub_series, grp_all, don_all,
                                             level="gene", control_group="reactive_CD4")
subclone_mod["module"] = subclone_mod["feature"].str.replace("sig_", "", regex=False)
subclone_mod["context_only"] = subclone_mod["module"].isin(th.CONTEXT_ONLY)
subclone_gene["module"] = subclone_gene["feature"].map(lambda x: th.GENE_TO_MODULE.get(x, [""])[0])
subclone_mod.to_csv(DE_DIR_S / "subclone_vs_control_module_v4.csv", index=False)
subclone_gene.to_csv(DE_DIR_S / "subclone_vs_control_gene_v4.csv", index=False)
print(subclone_mod["subclone"].nunique(), "subclones (module);",
      subclone_gene["subclone"].nunique(), "(gene) vs same-donor reactive-CD4")
```

```
91 subclones (module); 91 (gene) vs same-donor reactive-CD4
```

```python  # cell 30
# cohort module x subclone dotplot + clustered heatmap + per-subclone gene volcanoes
if not subclone_mod.empty:
    n_sub = subclone_mod["subclone"].nunique()
    fig, ax = th.module_score_dotplot(subclone_mod, group_col="subclone", feature_col="module",
                                      title="Subclone vs reactive-CD4 module scores",
                                      figsize=(max(6, 0.28 * n_sub + 3), 8))
    fig.savefig(FIG_DIR / "tcr_signaling_subclone_module_dotplot.png", dpi=150, bbox_inches="tight")

    _gs = adata.obs[adata.obs[SUBCLONE_COL].astype(str) != ""].groupby(SUBCLONE_COL, observed=True)
    sub_meta = pd.DataFrame({"sample": _gs["donor"].agg(_first),
                             "study": _gs["study"].agg(_first),
                             "stage": _gs["stage_class"].agg(_first) if "stage_class" in adata.obs else "",
                             "layer": _gs["skin_layer"].agg(_dom_layer)})
    cg = th.module_donor_clustermap(subclone_mod, group_col="subclone", feature_col="module",
                                    value_col="cliffs_delta", col_annot=sub_meta,
                                    annot_palettes=STRIP_PALS, show_col_labels=False,
                                    title="Subclone vs reactive-CD4 module Δ (subclones clustered)",
                                    figsize=(max(9, 0.18 * n_sub + 4), 7))
    cg.savefig(FIG_DIR / "tcr_signaling_subclone_module_clustermap.png", dpi=150, bbox_inches="tight")

VOLC_DIR = FIG_DIR / "tcr_signaling_subclone_volcano"; VOLC_DIR.mkdir(exist_ok=True)
subs_all = sorted(subclone_gene["subclone"].unique()) if not subclone_gene.empty else []
for s in subs_all:
    d = subclone_gene[subclone_gene["subclone"] == s]
    ax = th.gene_volcano(d, effect="cliffs_delta", p_col="fdr", label="feature",
                         module_col="module", title=f"{s} vs reactive-CD4")
    ax.figure.savefig(VOLC_DIR / f"{s.replace('/', '_')}.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
print(len(subs_all), "subclone volcanoes ->", VOLC_DIR)
```

```
91 subclone volcanoes -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/tcr_signaling_subclone_volcano

<Figure size 3417.6x960 with 2 Axes>
<Figure size 2445.6x840 with 5 Axes>
```

## §B.4 — Global + compartment: all malignant vs all reactive-CD4

```python  # cell 32
mod_pooled = th.wilcoxon_two_group(mod_scores[keep.values], score_cols, grp_s[keep], GROUPS,
                                   level="module", donor_series=None)
mod_pooled["module"] = mod_pooled["feature"].str.replace("sig_", "", regex=False)
mod_pooled["context_only"] = mod_pooled["module"].isin(th.CONTEXT_ONLY)
mod_pooled.to_csv(DE_DIR_S / "global_module_v4.csv", index=False)
ax = th.lollipop_effects(mod_pooled, label="module",
                         title="All malignant vs all reactive-CD4 (module Cliff's δ)", figsize=(5, 7))
ax.figure.savefig(FIG_DIR / "tcr_signaling_global_module_lollipop.png", dpi=150, bbox_inches="tight")

gene_pooled = th.wilcoxon_two_group(gene_expr[keep.values], genes_all, grp_s[keep], GROUPS,
                                    level="gene", donor_series=None)
gene_pooled["module"] = gene_pooled["feature"].map(lambda x: th.GENE_TO_MODULE.get(x, [""])[0])
gene_pooled.to_csv(DE_DIR_S / "global_gene_v4.csv", index=False)
act_genes = gene_pooled[gene_pooled["module"].isin(th.ACTIVITY_MODULES)]
ax = th.lollipop_effects(act_genes, label="feature",
                         title="Activity-module genes: malignant vs reactive-CD4", figsize=(5, 9))
ax.figure.savefig(FIG_DIR / "tcr_signaling_global_activity_gene_lollipop.png", dpi=150, bbox_inches="tight")
```

```
<Figure size 600x840 with 1 Axes>
<Figure size 600x1080 with 1 Axes>
```

```python  # cell 33
# stratified by disease stage / skin layer
def stratified_module(strat_col, strata):
    out = []
    if strat_col not in adata.obs:
        return pd.DataFrame()
    for s in strata:
        m = keep.values & (adata.obs[strat_col].astype(str) == s).values
        if m.sum() < 50:
            continue
        r = th.wilcoxon_two_group(mod_scores[m], score_cols, grp_s[m], GROUPS,
                                  level="module", donor_series=None)
        r["stratum"] = s
        out.append(r)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

stage_stats = stratified_module("stage_class", ["early", "advanced"])
layer_stats = stratified_module("skin_layer", ["epidermis", "dermis"])
for df, name in [(stage_stats, "stage"), (layer_stats, "layer")]:
    if df.empty:
        continue
    df["feature"] = df["feature"].str.replace("sig_", "", regex=False)
    fig, ax = th.module_score_dotplot(df, group_col="stratum", feature_col="feature",
                                      title=f"Malignant vs reactive-CD4 by {name}")
    fig.savefig(FIG_DIR / f"tcr_signaling_by_{name}_module_dotplot.png", dpi=150, bbox_inches="tight")
```

```
<Figure size 480x948 with 2 Axes>
<Figure size 480x948 with 2 Axes>
```

## §B.5 — Signaling-mode localization per subclone

```python  # cell 35
mal = adata[malig].copy()
mal_donor = mal.obs["donor"].astype(str)
contrast_df = th.subclone_contrast_vector(mal.obs, SUBCLONE_COL, score_prefix="sig_")
contrast_df = contrast_df[contrast_df.index.astype(str).str.len() > 0]
g2 = th.signaling_mode_heatmap(contrast_df, title="Subclone signaling-mode contrasts")
g2.savefig(FIG_DIR / "tcr_signaling_mode_heatmap.png", dpi=150, bbox_inches="tight")

from scipy.cluster.hierarchy import fcluster, linkage
contrast_names = [c for c in th.CONTRASTS if c in contrast_df.columns]
cvec = contrast_df[contrast_names].fillna(0.0)
if len(cvec) >= 4:
    Z = linkage(cvec, method="ward")
    contrast_df["mode_cluster"] = fcluster(Z, t=4, criterion="maxclust")
    names = {}
    for cl, subc in contrast_df.groupby("mode_cluster"):
        mc = subc[contrast_names].mean()
        names[cl] = f"{mc.abs().idxmax()}{'+' if mc[mc.abs().idxmax()] > 0 else '-'}"
    contrast_df["mode_label"] = contrast_df["mode_cluster"].map(names)
    print(contrast_df["mode_label"].value_counts())
```

```
mode_label
ap1_vs_nfat-    63
ap1_vs_nfat+    29
Name: count, dtype: int64

<Figure size 720x960 with 4 Axes>
```

## §B.6 — arm-CNV × downstream activity cross

```python  # cell 37
tb = pd.read_csv(TRUNK_BRANCH_V3)
arm_events = th.parse_arm_events(tb)
sub_means = mal.obs.groupby(SUBCLONE_COL, observed=True)[ACT_COLS].mean()
sub_means["donor"] = mal.obs.groupby(SUBCLONE_COL, observed=True)["donor"].agg(
    lambda x: x.astype(str).iloc[0])
sub_means = sub_means[sub_means.index.astype(str).str.len() > 0].reset_index()

for armn, direction, module, note in th.ARM_ACTIVITY_MAP:
    col = f"sig_{module}"
    if col not in sub_means.columns:
        continue
    fig, ax = th.arm_activity_box(sub_means, arm_events, armn, direction, module,
                                  title=f"{armn}{direction} -> {module}\n{note}")
    fig.savefig(FIG_DIR / f"tcr_signaling_cnv_{armn}{direction}_{module}.png", dpi=150, bbox_inches="tight")

# support: per-donor Spearman(continuous arm score, matched activity module) over malignant cells
from scipy.stats import spearmanr
rows = []
arm_mal = arm.reindex(mal.obs_names)
for armn, direction, module, _ in th.ARM_ACTIVITY_MAP:
    acol = f"sig_{module}"
    if armn not in arm_mal.columns or acol not in mal.obs:
        continue
    for d in mal_donor.unique():
        m = (mal_donor == d).values
        x = arm_mal[armn].to_numpy()[m]; y = mal.obs[acol].to_numpy()[m]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 30:
            continue
        rho, p = spearmanr(x[ok], y[ok])
        rows.append({"donor": d, "arm": armn, "direction": direction, "module": module,
                     "rho": rho, "p": p, "n": int(ok.sum())})
arm_corr = pd.DataFrame(rows)
arm_corr.to_csv(DE_DIR_S / "arm_activity_correlation_v4.csv", index=False)
print("arm x activity correlations:", arm_corr.shape)
```

```
arm x activity correlations: (196, 7)

<Figure size 480x420 with 1 Axes>
<Figure size 480x420 with 1 Axes>
<Figure size 480x420 with 1 Axes>
```

## §B.7 — Sensitivity (IEG dissociation) & save

```python  # cell 39
labile_mods = [m for m in ["L4_ieg_acute", "L3c_ap1_mapk_activity"] if f"sig_{m}" in adata.obs]
comp = []
for m in labile_mods:
    full = th.wilcoxon_two_group(mod_scores[keep.values], [f"sig_{m}"], grp_s[keep], GROUPS,
                                 level="module", donor_series=None)
    nl = th.wilcoxon_two_group(adata.obs.loc[keep, [f"sig_nolabile_{m}"]], [f"sig_nolabile_{m}"],
                               grp_s[keep], GROUPS, level="module", donor_series=None)
    comp.append({"module": m, "cliffs_full": full["cliffs_delta"].iloc[0],
                 "cliffs_nolabile": nl["cliffs_delta"].iloc[0]})
print(pd.DataFrame(comp))

out = contrast_df.reset_index().rename(columns={"index": SUBCLONE_COL})
out.to_csv(DATA / "tcr_signaling_subclone_modes_v4.csv", index=False)
print("wrote", DATA / "tcr_signaling_subclone_modes_v4.csv")
print("DE tables in", DE_DIR_S, "| figures in", FIG_DIR)
```

```
module  cliffs_full  cliffs_nolabile
0           L4_ieg_acute    -0.223134        -0.101150
1  L3c_ap1_mapk_activity    -0.228703        -0.129259
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/tcr_signaling_subclone_modes_v3.csv
DE tables in /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/tcr_signaling_de | figures in /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures
```

# §C — Machine-readable numeric summary

One flat `KEY: value` dump of every number this notebook produces: object/compartment counts,
donor eligibility, per-donor CNV subclones (k, silhouette, TCR/CNV concordance), MAJOR/MINOR and
nested subclones, trunk vs branch arm events, layer tropism, exported signatures, and for §B the
group sizes, module coverage, per-donor and pooled malignant-vs-reactive-CD4 effects, subclone
contrasts, signaling-mode clusters, arm-CNV × activity correlations and the IEG sensitivity check —
plus an index of the figures/tables on disk.

Run **last** (it reuses the in-memory objects from §A and §B). Blocks whose inputs are missing are
skipped. Also written to `tables/subclone_tcr_signaling_summary.txt`.

```python  # cell 41
# --- §C · Machine-readable summary: every number this notebook reports, in one dump ---
# Flat `KEY: value` lines covering §A (subclonal structure) and §B (TCR/costim signaling).
# Written to tables/subclone_tcr_signaling_summary.txt so an agent can read the result
# without re-running the heavy object.
_LOG = []


def P(s=""):
    _LOG.append(str(s)); print(s)


def SEC(t):
    P(); P("=" * 78); P(f"## {t}"); P("=" * 78)


def kv(k, v):
    P(f"{k}: {v}")


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "n/a"


def has(name):
    return name in globals() and globals()[name] is not None


def top_effects(df, label_col, n=10, effect="cliffs_delta", fdr_col="fdr"):
    """Rank features by |effect|; return printable lines (sign = direction vs control)."""
    if df is None or len(df) == 0 or effect not in df:
        return ["   (empty)"]
    agg = {effect: (effect, "mean")}
    if fdr_col in df:
        agg["frac_fdr05"] = (fdr_col, lambda s: round(float((s < 0.05).mean()), 2))
    if "donor" in df:
        agg["n_donor"] = ("donor", "nunique")
    if "subclone" in df:
        agg["n_subclone"] = ("subclone", "nunique")
    g = df.groupby(label_col, observed=True).agg(**agg).reset_index()
    g = g.reindex(g[effect].abs().sort_values(ascending=False).index).head(n)
    return [f"   {str(r[label_col])[:38]:<38} " +
            " ".join(f"{c}={r[c]:.3g}" for c in g.columns if c != label_col)
            for _, r in g.iterrows()]


TAB_DIR = NB_DIR / "tables"; TAB_DIR.mkdir(exist_ok=True)
obs = adata.obs
N = adata.n_obs

SEC("0 · PROVENANCE & PARAMETERS")
kv("notebook", "31_subclone_tcr_signaling.ipynb — subclonal evolution + TCR/costim signaling")
kv("object", str(OBJ))
kv("malignancy_definition",
   f"{MALIG_CALL} (26b per-sample cluster call, assembled by helpers/malignancy.py); "
   "undefined clusters are excluded, not counted reactive")
kv("inputs", f"{MAL.CLUSTER_PQ.name}, {MAL.ALICE_PQ.name}, {MALIG_PARQUET.name}, {ARM.name}")
kv("outputs", f"{SUBCLONE_PARQUET.name}, {SUMMARY_CSV.name}, {TRUNKBRANCH_CSV.name}, "
              f"{ELIG_CSV.name}, {SIGN_JSON.name}")
kv("params", f"SEED={SEED} MIN_MAL={MIN_MAL} K_MAX={K_MAX} MIN_SUB={MIN_SUB} "
             f"SIL_MIN={SIL_MIN} MIN_ARM_DELTA={MIN_ARM_DELTA} NULL_Z={NULL_Z} NULL_DRAWS={NULL_DRAWS} "
             f"Z_THR={Z_THR} EPS={EPS} BRANCH_DELTA={BRANCH_DELTA} MIN_LAYER_CELLS={MIN_LAYER_CELLS}")
kv("major_track", f"Leiden res={LEIDEN_RES}, n_neighbors={N_NEIGHBORS} on {LATENT}")
kv("minor_track", "KMeans on the 41-arm inferCNV matrix (nb30 skin_T_arm_cnv_v4)")
kv("subclone_gates", f"null-anchored: mean + {NULL_Z} x sd of a size/k-matched KMeans split of "
                     f"{NULL_X.shape[0]:,} held-out healthy diploid cells ({NULL_DRAWS} draws); "
                     f"SIL_MIN/MIN_ARM_DELTA are the null-less fallback.")

SEC("1 · OBJECT & COMPARTMENT")
kv("n_cells", f"{N:,}")
kv("n_genes", f"{adata.n_vars:,}")
kv("n_donors", obs["donor"].nunique())
kv("n_studies", obs["study"].nunique())
P("\n-- T subtype (cell_type_T2) --")
for k, v in obs["cell_type_T2"].astype(str).value_counts().items():
    P(f"   {k:<14} n={v:>8,} ({100 * v / N:5.1f}%)")
_mal = obs["is_malig"].to_numpy().astype(bool)
_cnv = obs["cnv_arm_malignant"].to_numpy().astype(bool)
kv(f"malignant({MALIG_CALL})", f"{int(_mal.sum()):,} ({pct(int(_mal.sum()), N)})")
_alc = obs["mal_tcr"].to_numpy().astype(bool)
kv("malignant_ALICE(mal_tcr, sensitivity)", f"{int(_alc.sum()):,} ({pct(int(_alc.sum()), N)})")
kv("cnv_arm_malignant(nb30)", f"{int(_cnv.sum()):,} ({pct(int(_cnv.sum()), N)})")
kv("TCR_and_CNV_overlap", f"{int((_mal & _cnv).sum()):,} | jaccard="
                          f"{(_mal & _cnv).sum() / max(1, (_mal | _cnv).sum()):.3f}")
kv("cells_with_arm_CNV_coverage", pct(int(arm[ARM_COLS].notna().any(axis=1).sum()), N)
   if has("arm") else "n/a")
kv("donors_with_ALICE_1aa_variant_family",
   f'{sum(v > 0 for v in n_var.values())}/{len(n_var)}' if has("n_var") else "n/a")

SEC("2 · DONOR ELIGIBILITY FOR SUBCLONE DETECTION")
kv("cohort_donors", len(elig_tbl))
kv("eligible(>=MIN_MAL malignant)", f'{int(elig_tbl["eligible"].sum())} '
   f'({pct(int(elig_tbl["eligible"].sum()), len(elig_tbl))})')
P("\n-- exclusion reasons --")
for k, v in elig_tbl.loc[~elig_tbl["eligible"], "reason_excluded"].value_counts().items():
    P(f"   {k:<52} n_donors={v}")
P("\n-- eligible donors (malignant cells) --")
P(elig_tbl.loc[elig_tbl["eligible"], ["donor", "study", "disease", "n_cells", "n_malignant"]]
  .to_string(index=False))

SEC("3 · CNV SUBCLONES PER DONOR (§A.1)")
kv("donors_analyzed", len(summary))
kv("donors_with>=2_subclones", f'{int((summary["k"] > 1).sum())}/{len(summary)} = '
   f'{(summary["k"] > 1).mean():.0%}  (paper reference: 84%)')
P("\n-- k distribution --")
for k, v in summary["k"].value_counts().sort_index().items():
    P(f"   k={k}: {v} donors")
kv("silhouette_median", f'{summary["silhouette"].median():.3f}')
kv("max_arm_delta_median", f'{summary["max_arm_delta"].median():.3f}')
kv("frac_cnv_confirmed_mean(TCR/CNV concordance)", f'{summary["frac_cnv_confirmed"].mean():.3f}')
kv("n_tcr_variants_total", int(summary["n_tcr_variants"].sum()))
P("\n-- per-donor summary --")
P(summary.sort_values("k", ascending=False).to_string(index=False))

if has("mm"):
    SEC("4 · MAJOR (transcriptomic) / MINOR (CNV) / NESTED SUBCLONES")
    kv("clones_split_on_both_tracks", f'{int((mm["has_major"] & mm["has_minor"]).sum())}/{len(mm)}')
    kv("n_major_median", f'{mm["n_major"].median():.0f}')
    kv("n_minor_median", f'{mm["n_minor"].median():.0f}')
    kv("nmi_major_vs_minor_median", f'{mm["nmi_major_vs_minor"].median():.3f}'
       if "nmi_major_vs_minor" in mm else "n/a")
    kv("n_nested_subclones", int(nested.nunique()))
    kv("majors_that_split_further_on_CNV",
       f'{int((split_summary["k"] > 1).sum())}/{len(split_summary)}')
    if has("multi_nested"):
        kv("donors_with>=2_nested_subclones", f"{len(multi_nested)}: {', '.join(multi_nested)}")
    if has("vc"):
        P("\n-- nested subclone sizes (cells) --")
        P(vc.to_string())

if has("trunk_branch_df") and len(trunk_branch_df):
    SEC("5 · TRUNK vs BRANCH ARM EVENTS (§A.1)")
    kv("donors_with_trunk_branch_call", len(trunk_branch_df))
    kv("n_trunk_median/max", f'{trunk_branch_df["n_trunk"].median():.0f} / '
                             f'{trunk_branch_df["n_trunk"].max():.0f}')
    kv("n_branch_median/max", f'{trunk_branch_df["n_branch"].median():.0f} / '
                              f'{trunk_branch_df["n_branch"].max():.0f}')
    kv("donors_with_any_branch_event", int((trunk_branch_df["n_branch"] > 0).sum()))
    for col, lab in [("trunk_arms", "trunk (ancestral, pan-clonal)"),
                     ("branch_arms", "branch (subclone-divergent)")]:
        ev_counts = (trunk_branch_df[col].fillna("").str.split(";").explode()
                     .replace("", np.nan).dropna().value_counts())
        P(f"\n-- most recurrent {lab} arm events (n_donors) --")
        for k, v in ev_counts.head(12).items():
            P(f"   {k:<10} {v}")
    P("\n-- per donor --")
    P(trunk_branch_df.to_string(index=False))

if has("trop_tbl"):
    SEC("6 · SKIN-LAYER TROPISM OF NESTED SUBCLONES")
    kv("min_cells_per_layer", MIN_LAYER_CELLS)
    for k, v in trop_tbl["tropism"].value_counts().items():
        P(f"   {k:<20} n_subclones={v}")
    if has("MENU"):
        kv("donors_with_epidermis/dermis_split", len(MENU))
        P(MENU.to_string(index=False))

if has("signatures"):
    SEC("7 · EXPORTED GENE SIGNATURES (for spatial scoring, nb24/25)")
    kv("file", str(SIGN_JSON))
    kv("n_signatures", len(signatures))
    for pref in ["subclone__", "panel__", "program__", "malignant_overall"]:
        keys = [k for k in signatures if k.startswith(pref)]
        kv(f"n_{pref.rstrip('_')}", f"{len(keys)} "
           f"(median size {int(np.median([len(signatures[k]) for k in keys]))})" if keys else "0")
    kv("malignant_overall_top20", ", ".join(signatures.get("malignant_overall", [])[:20]))

SEC("8 · SIGNALING GROUPS (§B)")
P("-- malig_group (foreground vs lineage-matched control) --")
for k, v in obs["malig_group"].value_counts().items():
    P(f"   {k:<14} n={v:>8,} ({100 * v / N:5.1f}%)")
P("-- skin_layer --")
for k, v in obs["skin_layer"].astype(str).value_counts().items():
    P(f"   {k:<14} n={v:>8,}")
if "stage_class" in obs:
    P("-- stage_class --")
    for k, v in obs["stage_class"].astype(str).value_counts().items():
        P(f"   {k:<14} n={v:>8,}")
kv("n_module_scores", len(score_cols))
kv("activity_modules", ", ".join(c.replace("sig_", "") for c in ACT_COLS))
kv("module_genes_present", len(genes_all))
if has("cov"):
    P("\n-- module gene coverage (present/total) --")
    P(cov[["module", "n_present", "n_total", "context_only"]].to_string(index=False))

SEC("9 · WITHIN-SAMPLE: malignant vs reactive-CD4, per donor (§B.2)")
kv("donors_tested", mod_stats["donor"].nunique() if len(mod_stats) else 0)
kv("effect", "Cliff's delta, positive = higher in malignant; frac_fdr05 = fraction of donors FDR<0.05")
P("\n-- top modules (mean over donors) --")
for l in top_effects(mod_stats, "module", n=14):
    P(l)
P("\n-- top genes (mean over donors) --")
for l in top_effects(gene_stats, "feature", n=20):
    P(l)

if has("mod_pooled"):
    SEC("10 · GLOBAL POOLED: all malignant vs all reactive-CD4 (§B.4)")
    P("-- modules --")
    P(mod_pooled[["module", "cliffs_delta", "p", "fdr", "direction"]]
      .reindex(mod_pooled["cliffs_delta"].abs().sort_values(ascending=False).index)
      .to_string(index=False))
    if has("gene_pooled"):
        P("\n-- top 25 genes --")
        gp = gene_pooled.reindex(
            gene_pooled["cliffs_delta"].abs().sort_values(ascending=False).index).head(25)
        P(gp[["feature", "module", "cliffs_delta", "fdr", "direction"]].to_string(index=False))

if has("subclone_mod") and len(subclone_mod):
    SEC("11 · SUBCLONE vs SAME-DONOR reactive-CD4 (§B.3)")
    kv("n_subclones_module_level", subclone_mod["subclone"].nunique())
    kv("n_subclones_gene_level",
       subclone_gene["subclone"].nunique() if has("subclone_gene") else 0)
    P("\n-- modules ranked by mean effect across subclones --")
    for l in top_effects(subclone_mod, "module", n=14):
        P(l)
    P("\n-- per-subclone strongest module --")
    best = (subclone_mod.reindex(subclone_mod["cliffs_delta"].abs()
                                 .sort_values(ascending=False).index)
            .drop_duplicates("subclone")
            .sort_values("subclone"))
    P(best[["subclone", "donor", "module", "cliffs_delta", "fdr"]].to_string(index=False)
      if "donor" in best else best[["subclone", "module", "cliffs_delta", "fdr"]].to_string(index=False))

for _df, _name in [(stage_stats if has("stage_stats") else None, "stage_class"),
                   (layer_stats if has("layer_stats") else None, "skin_layer")]:
    if _df is None or _df.empty:
        continue
    SEC(f"12 · STRATIFIED by {_name} (§B.4)")
    for s, d in _df.groupby("stratum", observed=True):
        d2 = d.reindex(d["cliffs_delta"].abs().sort_values(ascending=False).index).head(6)
        P(f"-- {s} (n_modules={len(d)}) --")
        P(d2[["feature", "cliffs_delta", "fdr"]].to_string(index=False))

if has("contrast_df") and len(contrast_df):
    SEC("13 · SIGNALING-MODE LOCALIZATION PER SUBCLONE (§B.5)")
    kv("n_subclones", len(contrast_df))
    kv("contrasts", ", ".join(f"{k}={v[0]}-{v[1]}" for k, v in th.CONTRASTS.items()))
    if "mode_label" in contrast_df:
        P("\n-- dominant signaling mode (ward, 4 clusters) --")
        for k, v in contrast_df["mode_label"].value_counts().items():
            P(f"   {k:<32} n_subclones={v}")
    _cn = [c for c in th.CONTRASTS if c in contrast_df.columns]
    P("\n-- contrast values per subclone --")
    P(contrast_df[_cn + [c for c in ["mode_label"] if c in contrast_df]]
      .round(2).to_string())

if has("arm_corr") and len(arm_corr):
    SEC("14 · arm-CNV x DOWNSTREAM ACTIVITY (§B.6)")
    kv("mapping", "; ".join(f"{a}{d}->{m}" for a, d, m, _ in th.ARM_ACTIVITY_MAP))
    kv("n_donor_x_arm_tests", len(arm_corr))
    ac = (arm_corr.groupby(["arm", "direction", "module"], observed=True)
          .agg(n_donors=("donor", "nunique"), median_rho=("rho", "median"),
               frac_p05=("p", lambda s: round(float((s < 0.05).mean()), 2)),
               median_n_cells=("n", "median"))
          .reset_index().sort_values("median_rho", key=abs, ascending=False))
    P(ac.to_string(index=False))

if has("comp") and len(comp):
    SEC("15 · SENSITIVITY: IEG/dissociation-labile genes dropped (§B.7)")
    kv("interpretation", "cliffs_nolabile close to cliffs_full => effect not driven by "
                         "dissociation-induced IEGs")
    P(pd.DataFrame(comp).to_string(index=False))

SEC("16 · ARTIFACT INDEX")
P(f"figures ({FIG_DIR}):")
for f in sorted(FIG_DIR.glob("subclone_v4*.png")) + sorted(FIG_DIR.glob("tcr_signaling*.png")):
    P(f"   {f.name:<58} {f.stat().st_size / 1e3:7.0f} kB")
if has("VOLC_DIR") and VOLC_DIR.exists():
    kv("subclone_volcanoes", f"{len(list(VOLC_DIR.glob('*.png')))} png in {VOLC_DIR}")
P(f"tables ({DE_DIR_S}):")
for f in sorted(DE_DIR_S.glob("*.csv")):
    P(f"   {f.name:<58} {f.stat().st_size / 1e3:7.0f} kB")
P("data outputs:")
for f in [SUBCLONE_PARQUET, SUMMARY_CSV, TRUNKBRANCH_CSV, ELIG_CSV, SIGN_JSON,
          DATA / "tcr_signaling_subclone_modes_v4.csv"]:
    P(f"   {f.name:<58} exists={f.exists()}")

_TXT = TAB_DIR / "subclone_tcr_signaling_summary.txt"
_TXT.write_text("\n".join(_LOG))
print(f"\n[summary written -> {_TXT}]")
```

```
==============================================================================
## 0 · PROVENANCE & PARAMETERS
==============================================================================
notebook: 31_subclone_tcr_signaling.ipynb — subclonal evolution + TCR/costim signaling
object: /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/skin_T_annotated.h5ad
malignancy_definition: tcr_malignant_alice (nb30 ALICE dominant founder + <=1-aa TRB-CDR3 variant family)
inputs: alice_malignancy_v3.parquet, skin_T_malignancy_v4.parquet, skin_T_arm_cnv_v4.parquet
outputs: subclones_v3.parquet, subclone_summary_v3.csv, subclone_trunk_branch_v3.csv, subclone_donor_eligibility_v3.csv, subclone_signatures_v3.json
params: SEED=0 MIN_MAL=200 K_MAX=4 MIN_SUB=50 SIL_MIN=0.15 MIN_ARM_DELTA=0.03 NULL_Z=3.0 NULL_DRAWS=8 Z_THR=2.5 EPS=0.01 BRANCH_DELTA=0.03 MIN_LAYER_CELLS=30
major_track: Leiden res=0.2, n_neighbors=15 on X_mrvi_u
minor_track: KMeans on the 41-arm inferCNV matrix (nb30 skin_T_arm_cnv_v4)
subclone_gates: null-anchored: mean + 3.0 x sd of a size/k-matched KMeans split of 209,900 held-out healthy diploid cells (8 draws); SIL_MIN/MIN_ARM_DELTA are the null-less fallback.

==============================================================================
## 1 · OBJECT & COMPARTMENT
==============================================================================
n_cells: 395,233
n_genes: 40,821
n_donors: 82
n_studies: 7

-- T subtype (cell_type_T2) --
   CD4            n= 338,698 ( 85.7%)
   CD8            n=  51,831 ( 13.1%)
   UNK            n=   2,768 (  0.7%)
   Tregs          n=   1,936 (  0.5%)
malignant_ALICE(tcr_malignant_alice): 76,349 (19.32%)
cnv_arm_malignant(nb30): 65,247 (16.51%)
TCR_and_CNV_overlap: 49,090 | jaccard=0.531
cells_with_arm_CNV_coverage: 59.77%
donors_with_ALICE_1aa_variant_family: 0/17

==============================================================================
## 2 · DONOR ELIGIBILITY FOR SUBCLONE DETECTION
==============================================================================
cohort_donors: 82
eligible(>=MIN_MAL malignant): 25 (30.49%)

-- exclusion reasons --
   no ALICE-malignant cells (no dominant TCR clone)     n_donors=42
   healthy control — no malignant compartment           n_donors=9
   <200 malignant cells — too few for subclone KMeans   n_donors=6

-- eligible donors (malignant cells) --
              donor           study    disease  n_cells  n_malignant
Li2024_atlas__CTCL3          li2024         MF    13398         8162
Li2024_atlas__CTCL1          li2024         MF     9294         7090
Li2024_atlas__CTCL5          li2024         MF    15239         6604
Li2024_atlas__CTCL8          li2024         MF    12237         5314
           B5__PT09        buus2025         SS    18610         5253
           D1__P171 chennareddy2025         MF     6512         5235
            D1__P76 chennareddy2025         MF     7210         4599
           D1__P311 chennareddy2025         MF     5495         3815
           B5__PT23        buus2025         SS     8316         3702
Li2024_atlas__CTCL7          li2024         MF    56512         3122
Li2024_atlas__CTCL6          li2024         MF    17988         3119
           D3__P192     brunner2024 CTCL_other     4543         3028
 Li2024_atlas__PT35          li2024         MF     5093         2865
           D1__P204 chennareddy2025         MF    11922         2343
            D1__PGS chennareddy2025         MF     3257         2259
Li2024_atlas__CTCL2          li2024         MF     5362         1840
 Li2024_atlas__PT11          li2024         MF     2444         1619
 Li2024_atlas__PT53          li2024         MF     2592         1240
 Li2024_atlas__PT56          li2024         MF     5610         1156
          D5__MFIVB  rindler2021_fi         MF     1567         1036
Li2024_atlas__CTCL4          li2024         MF    12999          743
           D1__P312 chennareddy2025         MF     2031          724
           D3__P196    
... [64481 more chars truncated]
```
