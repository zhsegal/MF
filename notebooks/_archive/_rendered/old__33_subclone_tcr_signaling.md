# old/33_subclone_tcr_signaling.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/33_subclone_tcr_signaling.ipynb` and is never run._

# 33 — Blood subclonal evolution + TCR/co-stimulation signaling

Blood analogue of `24_subclone_tcr_signaling`. Runs on the blood re-annotation `blood_T_annotated.h5ad` (`31_reannotation`) and the
`32_malignancy_tcr_cnv` blood malignancy outputs (`alice_malignancy_blood_v1`, `blood_T_malignancy_v3`,
`blood_T_arm_cnv_v3`). Malignant compartment = `tcr_malignant_alice` (ALICE dominant founder + ≤1-aa
β-variant family). Helpers reused verbatim (`subclone_helpers`, `alice_helpers`,
`tcr_signaling_helpers`).

**Blood-specific vs `24_subclone_tcr_signaling`:**
- No **skin-layer tropism** axis (peripheral blood has no epidermis/dermis) — `classify_subclone_tropism`
  / `layer_split_donors` / `skin_layer` are dropped.
- §B.4 stratification is by **`stage_class` (early/advanced)** and **`disease` (SS vs MF)** instead of
  skin layer.
- No `PT35` special case — founders are derived generically.
- The blood arm-CNV cache is keyed by `sample_id`, so the eligible-donor coverage check is done on
  per-cell coverage grouped by patient `donor` (not on the arm cache's `donor` label).

- **§A** — subclonal (divergent-evolution) structure.
- **§B** — TCR / co-stimulation signaling.

> **HEAVY** — loads/normalizes the ~8.7 GB object; run on the GPU/compute kernel, not the login node.

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

# ---- inputs (blood re-annotation + nb34 blood malignancy outputs) ----
OBJ           = OUT_DIR / "blood_T_annotated.h5ad"                 # nb33 (carries X_mrvi_u, raw_counts)
ALICE_MAL     = OUT_DIR / "alice_malignancy_blood_v1.parquet"      # nb34: tcr_malignant_alice per cell
MALIG_PARQUET = OUT_DIR / "blood_T_malignancy_v3.parquet"          # nb34: cnv_arm_malignant + tcr_* columns
ARM           = OUT_DIR / "blood_T_arm_cnv_v3.parquet"             # nb34: per-cell 41-arm inferCNV matrix
GMT           = NB_DIR.parent / "lib1_immune.gmt"

# ---- knobs (as nb31) ----
SEED = 0
MIN_MAL = 200
K_MAX = 4
MIN_SUB = 50
# ---- subclone gates: calibrated against the held-out diploid null, not hard-coded ----
# nb31's absolute gates were tuned on the SKIN v3 arm matrix (window=250 over ~7.5k HVGs). The blood
# v3 matrix is window=100 over ~20k genes with no dynamic-threshold zeroing, so its arm amplitudes
# are about half (donor-mean |arm| q95 0.014 vs 0.026) and the absolute cuts land in the wrong
# place: KMeans k=2 forced on KNOWN-DIPLOID null cells scores arm_delta 0.047 +/- 0.001 and
# silhouette 0.065 +/- 0.002, so MIN_ARM_DELTA=0.03 sits BELOW the noise floor (inert) while
# SIL_MIN=0.15 sits ~2x above it (k=1 for 46/53 donors). Both gates are now mean + NULL_Z * sd of
# the size- and k-matched null; SIL_MIN / MIN_ARM_DELTA are the fallback for an arm matrix that
# carries no null (i.e. the skin path), never a floor on top of the null.
SIL_MIN = 0.15
MIN_ARM_DELTA = 0.03
NULL_Z = 3.0
NULL_DRAWS = 8
Z_VLIM = 3.0        # arm-profile colour limit in diploid SDs -- FIXED, not self-normalizing, so a
                    # change of inferCNV reference is visible instead of being rescaled away
Z_THR, EPS = 2.5, 0.01
BRANCH_DELTA = 0.03
AUC_THR, SIL_THR = 0.65, 0.10
LATENT = "X_mrvi_u"
LEIDEN_RES = 0.2
N_NEIGHBORS = 15

# ---- outputs (_blood_v1) ----
SUBCLONE_PARQUET = OUT_DIR / "subclones_blood_v1.parquet"
SUMMARY_CSV      = OUT_DIR / "subclone_summary_blood_v1.csv"
TRUNKBRANCH_CSV  = OUT_DIR / "subclone_trunk_branch_blood_v1.csv"
ELIG_CSV         = OUT_DIR / "subclone_donor_eligibility_blood_v1.csv"
SIGN_JSON        = OUT_DIR / "subclone_signatures_blood_v1.json"
DE_DIR           = OUT_DIR / "subclone_de_blood"; DE_DIR.mkdir(exist_ok=True)
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
for _m in (S, A, H):
    importlib.reload(_m)
np.random.seed(SEED)
sc.settings.verbosity = 1
plt.rcParams["figure.dpi"] = 120
```

```python  # cell 4
# ============================================================
# §A.0  Load: malignant compartment + cached arm-CNV matrix
# ============================================================
if not os.environ.get("LSB_JOBID"):
    print("⚠️  No LSB_JOBID detected — you may be on the login node; loading blood_T_annotated.h5ad (~8.7 GB) can crash it.")
adata = sc.read_h5ad(OBJ)
print("loaded", OBJ.name, adata.shape)
# X and 'counts' are unused (rebuilt from raw_counts in §A / §B); free them to keep peak RAM low.
adata.X = None
if "counts" in adata.layers:
    del adata.layers["counts"]
# blood re-annotation carries cell_type_T (CD4/CD8/NK); alias to cell_type_T2 for the shared helpers.
adata.obs["cell_type_T2"] = adata.obs["cell_type_T"].astype(str)

# Malignancy = TCR-ALICE only (nb34). Join tcr_malignant_alice + the TCR/CNV columns from the parquets.
_al = pd.read_parquet(ALICE_MAL).set_index("cell_id")
adata.obs["tcr_malignant_alice"] = (_al["tcr_malignant_alice"].reindex(adata.obs_names)
                                    .fillna(False).astype(bool).to_numpy())
assert MALIG_PARQUET.exists(), f"{MALIG_PARQUET} missing — run 34_blood_malignant_annotation_tcr_cnv.ipynb first"
_mp = pd.read_parquet(MALIG_PARQUET)
_bool = {"cnv_arm_malignant", "tcr_is_dominant_clone", "tcr_is_expanded", "tcr_is_malignant"}
_str = {"tcr_clone_id", "sample_id"}
# sample_id = the inferCNV run (nb34 ran per sample_id, so the SZ5/SZ30 HTO lanes share one
# run); needed to pull each sample's OWN held-out diploid null for the per-sample profile.
for c in ["cnv_arm_malignant", "tcr_clone_id", "tcr_is_dominant_clone", "tcr_is_expanded",
          "tcr_clone_size", "tcr_is_malignant", "sample_id"]:
    if c in _mp.columns:
        s = _mp[c].reindex(adata.obs_names)
        if c in _bool:
            adata.obs[c] = s.fillna(False).astype(bool).to_numpy()
        elif c in _str:
            # sample_id/tcr_clone_id come back Categorical; fillna("") on one raises unless
            # "" is already a category, so drop to object first.
            adata.obs[c] = s.astype(object).fillna("").astype(str).to_numpy()
        else:
            adata.obs[c] = s.fillna(0).to_numpy()
del _al, _mp
mal = adata.obs["tcr_malignant_alice"].to_numpy()
adata.obs["is_malig"] = mal
print("malignant cells (TCR-ALICE):", int(mal.sum()),
      "| cnv_arm_malignant (nb34):", int(adata.obs["cnv_arm_malignant"].sum()))

# ---- cached arm-CNV matrix (nb34, keyed by sample_id) + its healthy-CD4 diploid null ----
# The v3 pass relabels half the shared reference as `ref_null` and renames those rows
# `<run>|NULL|<cell>`, so they traversed the identical estimator and ARE the empirical diploid
# null: healthy CD4, either `hc_atlas` (same-atlas healthy blood CD4) or `healthy_pbmc`
# (external OneK1K). nb35 used to discard them silently at the reindex below; they now
# calibrate the subclone gates (§A.1) and anchor the arm-profile plots (§A figures).
_arm_raw = pd.read_parquet(ARM).set_index("obs_name")
ARM_COLS = S.arm_order([c for c in _arm_raw.columns if c != "donor"])
_arm_query, _null_arm = S.split_null_rows(_arm_raw)
NULL_ARMS = _null_arm[[*ARM_COLS, "ref_source", "donor"]]
NULL_X = NULL_ARMS[ARM_COLS].fillna(0.0).to_numpy(dtype=float)   # built once, reused per donor
del _arm_raw, _null_arm
arm_df = _arm_query.reindex(adata.obs_names)
print("arm matrix:", _arm_query.shape, "| healthy-CD4 diploid null:", NULL_X.shape)
print(NULL_ARMS["ref_source"].value_counts().to_string())

# donor -> inferCNV run (sample_id), so a per-sample plot can use that sample's own null
RUN_OF = (adata.obs.loc[mal, ["donor", "sample_id"]].astype(str)
          .groupby("donor", observed=True)["sample_id"].agg(lambda s: s.iloc[0]).to_dict())
# nb34 only computed arm CNV for samples that had a same-sample CD8 diploid reference (>=20 CD8).
# CD8-sparse Sezary samples (e.g. B5__PT15: 619 CD4 / 10 CD8) are absent from the cache and cannot
# be recomputed with the CD8-ref path — so 0 arm coverage is an eligibility exclusion, not an error.
# Coverage is per-cell (cache keyed by sample_id) grouped by patient donor.
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
        return "no ALICE-malignant cells (no dominant TCR clone)"
    if row["n_malignant"] < MIN_MAL:
        return f"<{MIN_MAL} malignant cells — too few for subclone KMeans"
    if row["arm_cnv_cov"] == 0.0:
        return "no arm-CNV coverage — sample lacked >=20 CD8 diploid reference for inferCNV (nb34)"
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
loaded blood_T_annotated.h5ad (302362, 10000)
malignant cells (TCR-ALICE): 122510 | cnv_arm_malignant (nb34): 83043
arm matrix: (184329, 42) | healthy-CD4 diploid null: (247663, 41)
ref_source
healthy_pbmc    227686
hc_atlas         19977
cohort donors: 98 | eligible (>= 200 malignant + arm CNV): 53 | excluded: 45 (of which 1 for 0 arm-CNV coverage)
arms: 41 | wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/subclone_donor_eligibility_blood_v1.csv
```

## §A.0b — Blood cohort at a glance

Descriptive summary of the dataset the rest of `33_subclone_tcr_signaling` runs on: studies, patients/samples, disease mix,
CD4/CD8/NK composition, TCR coverage and the ALICE-malignant compartment. Pure description — nothing
is fit here. Donor-level table written to `blood_cohort_summary_v1.csv`; figures to
`figures/blood_cohort_overview_v1.png` and `figures/blood_cohort_malignant_burden_v1.png`.

```python  # cell 6
# ============================================================
# §A.0b  Blood cohort at a glance — summary tables
# ============================================================
BLOOD_COHORT_CSV = OUT_DIR / "blood_cohort_summary_v1.csv"

obs = adata.obs
_mal_i = mal.astype(int)

# paired-TCR presence: prefer the explicit flag, fall back to a non-empty TRB CDR3.
if "has_tcr" in obs:
    _s = obs["has_tcr"]
    has_tcr = _s.to_numpy(dtype=bool) if _s.dtype == bool else _s.astype(str).isin(["True", "true", "1"]).to_numpy()
elif "trb_cdr3" in obs:
    has_tcr = obs["trb_cdr3"].astype(str).str.len().gt(0).to_numpy()
else:
    has_tcr = np.zeros(adata.n_obs, dtype=bool)

_w = obs.assign(_m=_mal_i, _t=has_tcr.astype(int), _n=1)

# ---- donor level ----
don = (_w.groupby("donor", observed=True)
         .agg(study=("study", "first"), disease=("disease", "first"), sex=("sex", "first"),
              stage=("stage_class", "first"), tech=("tech", "first"),
              n_samples=("sample_id", "nunique"), n_cells=("_n", "sum"),
              n_tcr=("_t", "sum"), n_malig=("_m", "sum")))
for c in ["study", "disease", "sex", "stage", "tech"]:
    don[c] = don[c].astype(str)
don["pct_malig"] = (100 * don["n_malig"] / don["n_cells"]).round(2)
don["pct_tcr"] = (100 * don["n_tcr"] / don["n_cells"]).round(1)
don["eligible"] = don.index.isin(ELIG)
don.to_csv(BLOOD_COHORT_CSV)

DIS_ORDER = [d for d in ["MF", "SS", "HC"] if (obs["disease"].astype(str) == d).any()]
CT_ORDER = [c for c in ["CD4", "CD8", "NK"] if (obs["cell_type_T2"].astype(str) == c).any()]
ct_counts = obs["cell_type_T2"].astype(str).value_counts()
N, NM = adata.n_obs, int(_mal_i.sum())
arm_cov = float(arm_df[ARM_COLS].notna().any(axis=1).mean())


def _pc(n, d=N):
    return f"{int(n):,} ({100 * n / d:.1f}%)"


rows = [
    ("Cells (T / NK, post-QC)", f"{N:,}"),
    ("Genes", f"{adata.n_vars:,}"),
    ("Tissue", " / ".join(sorted(obs["tissue"].astype(str).unique()))),
    ("Studies", f"{obs['study'].nunique()}  ({', '.join(sorted(obs['study'].astype(str).unique()))})"),
    ("Patients (donors)", f"{len(don):,}"),
    ("Samples (libraries)", f"{obs['sample_id'].nunique():,}"),
    ("Cells per donor — median [IQR]",
     f"{don['n_cells'].median():,.0f} [{don['n_cells'].quantile(.25):,.0f}–{don['n_cells'].quantile(.75):,.0f}]"),
    ("", ""),
    *[(f"Donors — {d}", f"{int((don['disease'] == d).sum())}  ({int(don.loc[don['disease'] == d, 'n_cells'].sum()):,} cells)")
      for d in DIS_ORDER],
    ("", " "),
    *[(f"Cells — {c}", _pc(ct_counts.get(c, 0))) for c in CT_ORDER],
    ("", "  "),
    ("Cells with paired TCR", _pc(has_tcr.sum())),
    ("Cells in donor-dominant clone", _pc(obs["tcr_is_dominant_clone"].sum()) if "tcr_is_dominant_clone" in obs else "n/a"),
    ("Malignant cells (ALICE, tcr_malignant_alice)", _pc(NM)),
    ("CNV-arm-malignant cells (nb34)", _pc(obs["cnv_arm_malignant"].sum()) if "cnv_arm_malignant" in obs else "n/a"),
    ("Donors with ≥1 malignant cell", f"{int((don['n_malig'] > 0).sum())} / {len(don)}"),
    ("Donors with ≥1 ALICE ≤1-aa TCR variant", "computed in §A.1"),
    ("", "   "),
    ("Arm-CNV coverage (cells)", f"{100 * arm_cov:.1f}%  over {len(ARM_COLS)} arms"),
    ("Healthy-CD4 diploid null cells", f"{NULL_X.shape[0]:,}"),
    (f"Donors eligible for subclone analysis (≥{MIN_MAL} malig + arm CNV)", f"{len(ELIG)} / {len(don)}"),
]
overview = pd.DataFrame(rows, columns=["metric", "value"]).set_index("metric")

# ---- per study ----
by_study = (_w.groupby("study", observed=True)
              .agg(donors=("donor", "nunique"), samples=("sample_id", "nunique"),
                   cells=("_n", "sum"), tcr=("_t", "sum"), malignant=("_m", "sum")))
_dis_don = don.groupby(["study", "disease"], observed=True).size().unstack(fill_value=0)
for d in DIS_ORDER:
    by_study[f"{d}_donors"] = _dis_don.get(d, 0)
_ct = pd.crosstab(obs["study"].astype(str), obs["cell_type_T2"].astype(str), normalize="index") * 100
for c in CT_ORDER:
    by_study[f"%{c}"] = _ct.get(c, 0.0).round(1)
by_study["%TCR"] = (100 * by_study["tcr"] / by_study["cells"]).round(1)
by_study["%malig"] = (100 * by_study["malignant"] / by_study["cells"]).round(1)
by_study = by_study.drop(columns=["tcr"]).sort_values("cells", ascending=False)

# ---- per disease ----
by_disease = (_w.groupby("disease", observed=True)
                .agg(donors=("donor", "nunique"), samples=("sample_id", "nunique"),
                     cells=("_n", "sum"), malignant=("_m", "sum"))
                .reindex(DIS_ORDER))
_ctd = pd.crosstab(obs["disease"].astype(str), obs["cell_type_T2"].astype(str), normalize="index") * 100
for c in CT_ORDER:
    by_disease[f"%{c}"] = _ctd.get(c, 0.0).round(1).reindex(DIS_ORDER)
by_disease["%malig"] = (100 * by_disease["malignant"] / by_disease["cells"]).round(1)
by_disease["median_cells/donor"] = don.groupby("disease")["n_cells"].median().reindex(DIS_ORDER).astype(int)


def _show(df, title):
    print(f"\n── {title} " + "─" * max(2, 72 - len(title)))
    try:
        display(df)
    except NameError:
        print(df.to_string())


_show(overview, "Blood cohort — key numbers")
_show(by_study, "Per study")
_show(by_disease, "Per disease")
print("\nwrote", BLOOD_COHORT_CSV.name, "(donor-level table)")
```

```
── Blood cohort — key numbers ──────────────────────────────────────────────

                                                                                                value
metric                                                                                               
Cells (T / NK, post-QC)                                                                       302,362
Genes                                                                                          10,000
Tissue                                                                                           PBMC
Studies                                             6  (borcherding2019, borcherding2023, buus2025...
Patients (donors)                                                                                  98
Samples (libraries)                                                                                40
Cells per donor — median [IQR]                                                    2,501 [1,660–3,740]
                                                                                                     
Donors — MF                                                                         8  (19,542 cells)
Donors — SS                                                                       89  (279,852 cells)
Donors — HC                                                                          1  (2,968 cells)
                                                                                                     
Cells — CD4                                                                           240,993 (79.7%)
Cells — CD8                                                                            60,638 (20.1%)
Cells — NK                                                                                 731 (0.2%)
                                                                                                     
Cells with paired TCR                                                                 167,489 (55.4%)
Cells in donor-dominant clone                                                         122,510 (40.5%)
Malignant cells (ALICE, tcr_malignant_alice)                                          122,510 (40.5%)
CNV-arm-malignant cells (nb34)                                                         83,043 (27.5%)
Donors with ≥1 malignant cell                                                                 62 / 98
Donors with ≥1 ALICE ≤1-aa TCR variant                                               computed in §A.1
                                                                                                     
Arm-CNV coverage (cells)                                                          61.0%  over 41 arms
Healthy-CD4 diploid null cells                                                                247,663
Donors eligible for subclone analysis (≥200 mal...                                            53 / 98

── Per study ───────────────────────────────────────────────────────────────

                 donors  samples   cells  malignant  MF_donors  SS_donors  \
study                                                                       
geskin2026           65        7  167150      50175          0         65   
buus2025             17       20   61591      35976          0         17   
borcherding2023       5        5   36935      29603          0          5   
herrera2021           8        8   21109       6320          7          0   
borcherding2019       2        1   14176          0          0          2   
rindler2021_fi        1        1    1401        436          1          0   

                 HC_donors  %CD4  %CD8  %NK  %TCR  %malig  
study                                                      
geskin2026               0  78.4  21.5  0.1  40.1    30.0  
buus2025                 0  81.7  18.1  0.2  77.3    58.4  
borcherding2023          0  88.3  10.9  0.9  94.8    80.1  
herrera2021              1  62.1  37.6  0.4  78.6    29.9  
borcherdin
... [754 more chars truncated]
```

```python  # cell 7
# ============================================================
# §A.0b  Blood cohort at a glance — figures
# ============================================================
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # fixed categorical order
DIS_C = {"MF": "#2a78d6", "SS": "#eb6834", "HC": "#4a3aa7"}                # validated CVD-safe trio
CT_C = {"CD4": "#2a78d6", "CD8": "#1baf7a", "NK": "#eda100"}
INK, MUTED = "#0b0b0b", "#52514e"


def _clean(ax, axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=axis, color="#e5e5e2", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=MUTED)


fig, axes = plt.subplots(2, 3, figsize=(15.5, 8))

# (a) cells per study
ax = axes[0, 0]
s = by_study.sort_values("cells")
ax.barh(s.index, s["cells"] / 1e3, color=SER[0], height=0.62)
for y, (c, d) in enumerate(zip(s["cells"], s["donors"])):
    ax.text(c / 1e3 + s["cells"].max() / 1e3 * 0.02, y, f"{d}d", va="center", fontsize=7, color=MUTED)
ax.set_xlabel("cells (thousands)", fontsize=8, color=MUTED)
ax.set_xlim(0, s["cells"].max() / 1e3 * 1.15)
ax.set_title(f"{s.index[-1]} contributes {100 * s['cells'].iloc[-1] / s['cells'].sum():.0f}% of all cells",
             fontsize=9, color=INK)
_clean(ax, "x")

# (b) donors & samples per disease
ax = axes[0, 1]
x = np.arange(len(DIS_ORDER))
ax.bar(x - 0.17, by_disease["donors"], width=0.32, color=[DIS_C[d] for d in DIS_ORDER])
ax.bar(x + 0.17, by_disease["samples"], width=0.32, color=[DIS_C[d] for d in DIS_ORDER], alpha=0.45)
for xi, d in zip(x, DIS_ORDER):
    ax.text(xi - 0.17, by_disease.loc[d, "donors"], f"{int(by_disease.loc[d, 'donors'])}",
            ha="center", va="bottom", fontsize=8, color=INK)
    ax.text(xi + 0.17, by_disease.loc[d, "samples"], f"{int(by_disease.loc[d, 'samples'])}",
            ha="center", va="bottom", fontsize=8, color=MUTED)
ax.set_xticks(x); ax.set_xticklabels(DIS_ORDER)
ax.set_ylabel("count", fontsize=8, color=MUTED)
ax.legend(handles=[Patch(fc="#9a9a95", label="donors"), Patch(fc="#9a9a95", alpha=0.45, label="samples")],
          fontsize=7, frameon=False, loc="upper right")
_n_mf, _n_ss = int(by_disease.loc["MF", "donors"]), int(by_disease.loc["SS", "donors"])
ax.set_title(f"{_n_mf} MF vs {_n_ss} SS patients (+ {int(by_disease.loc['HC', 'donors'])} healthy)"
             if "HC" in by_disease.index else f"{_n_mf} MF vs {_n_ss} SS patients", fontsize=9, color=INK)
_clean(ax)

# (c) T-cell composition per disease
ax = axes[0, 2]
comp = pd.crosstab(obs["disease"].astype(str), obs["cell_type_T2"].astype(str), normalize="index") * 100
comp = comp.reindex(DIS_ORDER)
x = np.arange(len(DIS_ORDER)); w = 0.8 / len(CT_ORDER)
for i, c in enumerate(CT_ORDER):
    ax.bar(x + (i - (len(CT_ORDER) - 1) / 2) * w, comp[c], width=w - 0.03, color=CT_C[c], label=c)
ax.set_xticks(x); ax.set_xticklabels(DIS_ORDER)
ax.set_ylabel("% of cells", fontsize=8, color=MUTED)
ax.legend(fontsize=7, frameon=False, ncol=len(CT_ORDER))
_cd4 = comp["CD4"] if "CD4" in comp else pd.Series(dtype=float)
ax.set_title(f"CD4 dominance: {_cd4.get('SS', float('nan')):.0f}% in SS vs {_cd4.get('HC', float('nan')):.0f}% in HC",
             fontsize=9, color=INK)
_clean(ax)

# (d) cells per donor
ax = axes[1, 0]
_v = don["n_cells"].to_numpy()
ax.hist(_v, bins=np.logspace(np.log10(max(_v.min(), 1)), np.log10(_v.max()), 24), color=SER[0])
ax.axvline(np.median(_v), color="#e34948", ls="--", lw=1.2)
ax.set_xscale("log")
ax.set_xlabel("cells per donor", fontsize=8, color=MUTED)
ax.set_ylabel("donors", fontsize=8, color=MUTED)
ax.set_title(f"donor depth spans {_v.min():,}–{_v.max():,} cells (median {np.median(_v):,.0f})",
             fontsize=9, color=INK)
_clean(ax)

# (e) per-donor malignant burden by disease
ax = axes[1, 1]
rng = np.random.default_rng(SEED)
for i, d in enumerate(DIS_ORDER):
    y = don.loc[don["disease"] == d, "pct_malig"].to_numpy()
    ax.scatter(i + rng.uniform(-0.16, 0.16, y.size), y, s=16, color=DIS_C[d],
               edgecolor="white", linewidth=0.5, zorder=3)
    ax.hlines(np.median(y), i - 0.28, i + 0.28, color=INK, lw=1.6, zorder=4)
ax.set_xticks(range(len(DIS_ORDER))); ax.set_xticklabels(DIS_ORDER)
ax.set_ylabel("% malignant cells (ALICE)", fontsize=8, color=MUTED)
_med = don.groupby("disease")["pct_malig"].median()
ax.set_title(f"malignant burden: SS median {_med.get('SS', float('nan')):.0f}% vs "
             f"MF {_med.get('MF', float('nan')):.0f}%", fontsize=9, color=INK)
_clean(ax)

# (f) eligibility funnel
ax = axes[1, 2]
steps = ["all donors", "≥1 malignant cell", f"≥{MIN_MAL} malignant cells", "+ arm CNV → eligible"]
vals = [len(elig_tbl), int((elig_tbl["n_malignant"] > 0).sum()),
        int((elig_tbl["n_malignant"] >= MIN_MAL).sum()), len(ELIG)]
cols = [SER[0]] * 3 + [SER[2]]
ax.barh(steps[::-1], vals[::-1], color=cols[::-1], height=0.6)
for y, v in enumerate(vals[::-1]):
    ax.text(v + max(vals) * 0.02, y, f"{v}", va="center", fontsize=8, color=INK)
ax.set_xlim(0, max(vals) * 1.14)
ax.set_xlabel("donors", fontsize=8, color=MUTED)
ax.set_title(f"{len(ELIG)}/{len(elig_tbl)} donors carry the subclone analysis", fontsize=9, color=INK)
_clean(ax, "x")

fig.suptitle(f"Blood cohort — {N:,} T/NK cells · {len(don)} donors · {obs['sample_id'].nunique()} samples · "
             f"{obs['study'].nunique()} studies · {100 * NM / N:.1f}% ALICE-malignant",
             fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(FIG_DIR / "blood_cohort_overview_v1.png", dpi=200, bbox_inches="tight")
plt.show()

# ---- per-donor malignant compartment (the donors the rest of nb35 runs on) ----
t = don[don["n_malig"] > 0].sort_values("n_malig", ascending=False)
fig2, ax = plt.subplots(figsize=(max(7, 0.20 * len(t)), 3.4))
ax.bar(np.arange(len(t)), t["n_malig"], color=[DIS_C.get(d, SER[0]) for d in t["disease"]],
       edgecolor=np.where(t["eligible"], "#0b0b0b", "none"), linewidth=0.7)
ax.axhline(MIN_MAL, color="#e34948", ls="--", lw=1.1)
ax.set_yscale("log")
ax.set_xticks(np.arange(len(t)))
ax.set_xticklabels(t.index, rotation=90, fontsize=5)
ax.set_ylabel("malignant cells (log)", fontsize=8, color=MUTED)
ax.set_xlim(-0.8, len(t) - 0.2)
ax.legend(handles=[Patch(fc=DIS_C[d], label=d) for d in DIS_ORDER if d in set(t["disease"])] +
                  [Patch(fc="white", ec="#0b0b0b", label="eligible"),
                   Line2D([], [], color="#e34948", ls="--", label=f"MIN_MAL = {MIN_MAL}")],
          fontsize=7, frameon=False, ncol=5, loc="upper right")
ax.set_title(f"{len(ELIG)} of {len(t)} donors with a malignant clone clear the {MIN_MAL}-cell subclone gate",
             fontsize=9, color=INK)
_clean(ax)
fig2.tight_layout()
fig2.savefig(FIG_DIR / "blood_cohort_malignant_burden_v1.png", dpi=200, bbox_inches="tight")
plt.show()
```

```
<Figure size 1860x960 with 6 Axes>
<Figure size 1488x408 with 1 Axes>
```

## §A.1 — Calculation (compute once, no plots)

```python  # cell 9
# Per-donor TCR founder + ALICE <=1-aa variant family (generic; no skin PT35 special case).
mal_obs = adata.obs[mal].copy()
clono = A.clonotype_table(mal_obs, group="donor")
founder_sets, family_sets = {}, {}
for d, sub in clono.groupby("donor", observed=True):
    if sub["is_founder"].astype(bool).any():
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
donors with >=1 ALICE <=1-aa TCR-beta variant: 0 / 62
```

```python  # cell 10
# per-donor CNV subclones (KMeans on the arm matrix) + summary + dominant subclone.
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
                         "sil_margin": meta["sil_margin"],
                         "null_calibrated": meta["null_calibrated"]})

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
      f"{int(summary['null_calibrated'].sum())}/{len(summary)} donors "
      f"| median sil>={summary['sil_thr'].median():.4f}, arm_delta>={summary['arm_delta_thr'].median():.4f}")
summary.sort_values("k", ascending=False)
```

```
>= 2 CNV subclones: 36/53 = 68% donors (paper: 84%)
TCR/CNV concordance (mean frac_cnv_confirmed): 0.63
gates: null-anchored (mean + 3.0sd of a size/k-matched diploid split) for 53/53 donors | median sil>=0.0573, arm_delta>=0.0551

             donor  n_malig  k  silhouette  max_arm_delta  centroid_spread  \
4    B4__SZ30_HTO6      411  4      0.0543         0.0896           0.0863   
15   B4__SZ38_HTO3     4287  4      0.0621         0.0728           0.1001   
17   B4__SZ38_HTO5     3333  4      0.0552         0.0667           0.0995   
38        B5__PT09     1358  4      0.0691         0.0977           0.1315   
42        B5__PT16     1685  4      0.0538         0.0837           0.0775   
16   B4__SZ38_HTO4     3127  3      0.0710         0.0613           0.1025   
12   B4__SZ33_HTO9      616  3      0.0664         0.0719           0.0834   
6    B4__SZ30_HTO8      326  3      0.0657         0.0751           0.0872   
7    B4__SZ30_HTO9      330  3      0.0736         0.0787           0.0873   
11   B4__SZ33_HTO8      868  3      0.0696         0.0692           0.0982   
10   B4__SZ33_HTO7      889  3      0.0713         0.0711           0.0883   
8   B4__SZ33_HTO10      605  3      0.0731         0.0721           0.0975   
9    B4__SZ33_HTO6      499  3      0.0776         0.0783           0.0908   
18   B4__SZ38_HTO6     3554  3      0.0699         0.0636           0.1052   
13   B4__SZ38_HTO1     3983  3      0.0875         0.0677           0.1138   
14   B4__SZ38_HTO2     3724  3      0.0576         0.0689           0.0768   
33        B5__PT04      385  3      0.1031         0.0772           0.1184   
32        B5__PT02     5142  3      0.0833         0.0631           0.0923   
28    B4__SZ5_HTO4     3240  3      0.0716         0.0670           0.1201   
3   B4__SZ30_HTO10      252  3      0.0692         0.0726           0.0879   
1        B2__CTCL5      489  3      0.0830         0.0703           0.0807   
39        B5__PT10     1664  3      0.0558         0.0702           0.0691   
41        B5__PT14     5305  3      0.0670         0.0880           0.1352   
31        B5__PT01     2971  3      0.0671         0.0634           0.0819   
48          H__SS2      537  3      0.0725         0.0641           0.0842   
34        B5__PT05     2957  3      0.0682         0.0656           0.0858   
29    B4__SZ5_HTO5     3549  3      0.0753         0.0722           0.1302   
35        B5__PT06     4782  3      0.0586         0.0641           0.0683   
5    B4__SZ30_HTO7      372  2      0.0983         0.0650           0.0793   
36        B5__PT07     2178  2      0.0909         0.0713           0.0840   
37        B5__PT08     4163  2      0.0877         0.0725           0.0813   
51          H__SS5      732  2      0.0757         0.0559           0.0620   
27    B4__SZ5_HTO3     3288  2      0.4300         0.0634           0.1607   
26    B4__SZ5_HTO2     4051  2      0.4299         0.0708           0.1682   
25    B4__SZ5_HTO1     2574  2      0.4440         0.0715           0.1668   
30    B4__SZ5_HTO6     2910  2      0.4342         0.0633           0.1543   
2        B2__CTCL6      518  1         NaN         0.0000           0.0000   
20   B4__SZ39_HTO2      479  1         NaN         0.0000           0.0000   
19   B4__SZ39_HTO1      407  1         NaN         0.0000           0.0000   
0        B2__CTCL2    28596  1         NaN         0.0000           0.0000   
21   B4__SZ39_HTO3      376  1         NaN         0.0000           0.0000   
24   B4__SZ39_HTO6      560  1         NaN         0.0000           0.0000   
23   B4__SZ39_HTO5      364  1         NaN         0.0000           0.0000   
22   B4__SZ39_HTO4      407  1         NaN         0.0000           0.0000   
40        B5__PT13      507  1         NaN         0.0000           0.0000   
44        B5__PT21      646  1         NaN         0.0000           0.0000   
43        B5__PT18     1555  1         NaN         0.0000           0.0000   
46        H__MFIV1     1
... [10218 more chars truncated]
```

```python  # cell 11
# trunk (pan-clonal, ancestral) vs branch (subclone-divergent) arm events per donor.
# benign baseline = non-dominant-clone cells (blood has no 'tumor_cell' label).
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
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/subclone_trunk_branch_blood_v1.csv (53, 9)

            donor  k                       trunk_arms  \
0       B2__CTCL2  1                                    
1       B2__CTCL5  3                           chr7q+   
2       B2__CTCL6  1  chr10p+;chr13q-;chr17p-;chr17q+   
3  B4__SZ30_HTO10  3                                    
4   B4__SZ30_HTO6  4                                    

                              branch_arms  n_trunk  n_branch            study  \
0                                                0         0  borcherding2023   
1                 chr11p-;chr11q-;chr18p+        1         3  borcherding2023   
2                                                4         0  borcherding2023   
3                  chr7q+;chr18p+;chr21q+        0         3       geskin2026   
4  chr7q+;chr15q+;chr18p+;chr18q+;chr21q+        0         5       geskin2026   

  disease disease_stage  
0      SS            NA  
1      SS            NA  
2      SS            NA  
3      SS           nan  
4      SS           nan
```

```python  # cell 12
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
MAJOR = Leiden (res=0.2) on X_mrvi_u; MINOR = CNV subclones | clones with both tracks split: 34/53
```

```python  # cell 13
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
nested subclones: 256 across 53 eligible donors | majors that split further on CNV: 52/167
```

```python  # cell 14
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
malignant cells for DE/scoring: (121038, 10000) | donors: 53
nested subclones: 256 | donors with >=2 nested subclones: 50
```

```python  # cell 15
# functional-program scores + persist (no skin-layer tropism axis for blood).
added = S.score_programs(mal_ad, prefix="prog_", cell_cycle=True, seed=SEED)
prog_cols = [c for c in added if c.startswith("prog_")]

adata.obs[["donor", "cnv_subclone", "nested_subclone", "is_dominant_subclone"]].to_parquet(SUBCLONE_PARQUET)
summary.to_csv(SUMMARY_CSV, index=False)
print("program scores:", prog_cols)
print("wrote", SUBCLONE_PARQUET.name, "+", SUMMARY_CSV.name)
```

```
program scores: ['prog_prolif', 'prog_oxphos', 'prog_glycolysis', 'prog_cytokine', 'prog_th1', 'prog_th2', 'prog_th17', 'prog_skin_homing', 'prog_recirc_homing', 'prog_protease']
wrote subclones_blood_v1.parquet + subclone_summary_blood_v1.csv
```

```python  # cell 16
# Export gene signatures for downstream SPATIAL/expression scoring.
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
wrote subclone_signatures_blood_v1.json: 254 signatures
```

## §A — Cohort overview & per-sample figures

Cohort CNV-subclone arm profiles, prevalence of ≥2 subclones, program-score heatmap. For a per-sample
UMAP + arm-CNV view, set `SAMPLE` to any eligible donor and call `S.plot_nested_sample`.

```python  # cell 18
# Cohort arm profiles z-scored against the healthy-CD4 diploid null, on a FIXED colour scale, with
# the reference rows appended at the bottom (they must read flat -- that is the visual zero).
_sub_mask = adata.obs["cnv_subclone"].astype(str) != ""
_sub_sizes = adata.obs.loc[_sub_mask, "cnv_subclone"].value_counts().to_dict()


def _f1(s):
    return str(s.astype(str).iloc[0])


_gs = adata.obs[_sub_mask].groupby("cnv_subclone", observed=True)
_row_annot = pd.DataFrame({"sample": _gs["donor"].agg(_f1), "disease": _gs["disease"].agg(_f1)})
if "stage_class" in adata.obs:
    _row_annot["stage"] = _gs["stage_class"].agg(_f1)

S.plot_cohort_arm_profiles(centroids, ELIG, ARM_COLS, null_arms=NULL_ARMS,
                           row_annot=_row_annot, sizes=_sub_sizes, vlim=Z_VLIM,
                           save=FIG_DIR / "subclone_blood_v1_arm_profiles_cohort.png")

# Same figure one row PER SAMPLE (subclones collapsed): the mean arm profile over each eligible
# sample's malignant cells. Reads as the cohort karyotype map -- what recurs across samples --
# where the per-clone version above answers what diverges within a sample.
_samp_prof, _samp_sizes = {}, {}
for d in ELIG:
    _cells = adata.obs_names[mal & (adata.obs["donor"].astype(str).values == d)]
    _samp_prof[d] = arm_df.reindex(_cells)[ARM_COLS].mean(axis=0).to_frame(d).T
    _samp_sizes[d] = len(_cells)
_gd = adata.obs[mal].groupby("donor", observed=True)
_samp_annot = pd.DataFrame({"study": _gd["study"].agg(_f1), "disease": _gd["disease"].agg(_f1)})
if "stage_class" in adata.obs:
    _samp_annot["stage"] = _gd["stage_class"].agg(_f1)
_samp_annot = _samp_annot.reindex(ELIG)
S.plot_cohort_arm_profiles(_samp_prof, ELIG, ARM_COLS, null_arms=NULL_ARMS,
                           row_annot=_samp_annot, sizes=_samp_sizes, vlim=Z_VLIM,
                           title="Per-sample malignant arm profiles vs healthy-CD4 diploid null",
                           save=FIG_DIR / "subclone_blood_v1_arm_profiles_per_sample.png")

S.plot_subclone_prevalence(summary, k_max=K_MAX, n_multi=n_multi, save=FIG_DIR / "subclone_blood_v1_prevalence.png")
S.plot_program_heatmap(mal_ad.obs, "nested_subclone", prog_cols,
                       donor_order=multi_nested, save=FIG_DIR / "subclone_blood_v1_programs.png")
```

```
<Figure size 1392x2083.2 with 3 Axes>
<Figure size 1392x924 with 3 Axes>
<Figure size 600x360 with 1 Axes>
<Figure size 600x4857.6 with 2 Axes>
```

```python  # cell 19
# ---- per-sample arm profiles: every eligible sample against its OWN healthy-CD4 diploid null ----
# The null is redrawn per inferCNV run, so `run=RUN_OF[d]` keeps the reference batch-matched to the
# sample (falls back to the pooled null when that run held out too few cells).
SAMPLE_FIG_DIR = FIG_DIR / "subclone_blood_v1_sample_arm_profiles"; SAMPLE_FIG_DIR.mkdir(exist_ok=True)
for d in ELIG:
    S.plot_sample_arm_profile(centroids[d], d, ARM_COLS, null_arms=NULL_ARMS, sizes=_sub_sizes,
                              run=RUN_OF.get(d), vlim=Z_VLIM, show=False,
                              save=SAMPLE_FIG_DIR / f"{d}.png")
print(len(ELIG), "per-sample arm profiles ->", SAMPLE_FIG_DIR)

# Per-sample subclone structure — pick ONE eligible donor (shown inline).
SAMPLE = ELIG[0] if ELIG else None
if SAMPLE is not None:
    print("selected:", SAMPLE, "| inferCNV run:", RUN_OF.get(SAMPLE))
    S.plot_sample_arm_profile(centroids[SAMPLE], SAMPLE, ARM_COLS, null_arms=NULL_ARMS,
                              sizes=_sub_sizes, run=RUN_OF.get(SAMPLE), vlim=Z_VLIM)
    S.plot_nested_sample(adata, SAMPLE, umap_by_donor, arm_df, ARM_COLS, mal, fig_dir=FIG_DIR)
```

```
53 per-sample arm profiles -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/subclone_blood_v1_sample_arm_profiles
selected: B2__CTCL2 | inferCNV run: B2__CTCL2

<Figure size 1080x583.2 with 3 Axes>
<Figure size 1920x768 with 6 Axes>
```

# §B — TCR / co-stimulation signaling across malignant clones & subclones

Where in the TCR/costim cascade each malignant population concentrates dysregulation (module scores +
individual genes), vs lineage-matched reactive-CD4, cross-checked against arm-level inferCNV.
Reuses the in-memory object from §A; rebuilds `X` from `raw_counts` and normalizes the whole object.

```python  # cell 21
import warnings
import tcr_signaling_helpers as th
importlib.reload(th)

DATA = OUT_DIR
DE_DIR_S = DATA / "tcr_signaling_de_blood"; DE_DIR_S.mkdir(exist_ok=True)
SUBCLONES_V3 = SUBCLONE_PARQUET
ARM_CNV_V3   = ARM
TRUNK_BRANCH_V3 = TRUNKBRANCH_CSV
MALIG_COL = "tcr_malignant_alice"
SUBCLONE_COL = "nested_subclone"
GROUPS = ("malignant", "reactive_CD4")
```

```python  # cell 22
# rebuild X from raw_counts on the full object, normalize+log1p
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
normalized full object for signaling scoring: (302362, 10000)
```

```python  # cell 23
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
arm cnv coverage: 0.6096301783954333
```

```python  # cell 24
# groups: malignant vs lineage-matched reactive CD4
malig = adata.obs[MALIG_COL].astype("boolean").fillna(False).to_numpy()
ct = adata.obs["cell_type_T2"].astype(str)
grp = np.where(malig, "malignant",
               np.where((~malig) & (ct == "CD4"), "reactive_CD4", "other"))
adata.obs["malig_group"] = pd.Categorical(grp, categories=["malignant", "reactive_CD4", "other"])

# blood stratification axes: disease stage + disease subtype (no skin layer).
print(adata.obs["malig_group"].value_counts())
if "stage_class" in adata.obs:
    print(adata.obs["stage_class"].astype(str).value_counts())
print(adata.obs["disease"].astype(str).value_counts())
```

```
malig_group
malignant       122510
reactive_CD4    118483
other            61369
Name: count, dtype: int64
stage_class
advanced    299394
HC            2968
Name: count, dtype: int64
disease
SS    279852
MF     19542
HC      2968
Name: count, dtype: int64
```

## §B.1 — Score modules (+ `_nolabile` sensitivity re-score)

```python  # cell 26
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
module  n_present  n_total  \
0    COSTIM_coinhibitory_receptors         12       13   
1   COSTIM_ligands_on_TME_partners         11       13   
2         COSTIM_pi3k_akt_activity          4        7   
3       COSTIM_pi3k_akt_components          2       13   
4      COSTIM_receptors_activating         12       13   
5                    L0_coreceptor          2        3   
6           L0_pan_t_identity_loss          5        5   
7            L0_tcr_cd3_components          6        6   
8              L1_proximal_kinases          9       11   
9         L2_gef_gtpase_components          5        9   
10   L2_lat_signalosome_components          3        8   
11        L2_plcg1_node_components          3        5   
12         L3a_cbm_nfkb_components          4       17   
13               L3a_nfkb_activity         14       19   
14     L3b_calcium_nfat_components          7       16   
15               L3b_nfat_activity         12       12   
16           L3c_ap1_mapk_activity         13       13   
17         L3c_ras_mapk_components          3       14   
18                    L4_ieg_acute         19       19   
19        L5_chronicity_exhaustion         12       12   
20           L5_sustained_tf_layer         14       16   
21                 NEGREG_rheostat         16       21   
22                SIGNAL3_jak_stat         13       16   

                                              missing  context_only  
0                                                BTLA         False  
1                                          ICOSLG,PVR         False  
2                                FOXO1,RPS6KB1,SLC2A1         False  
3   PIK3R1,PIK3CA,PTEN,AKT1,AKT2,PDPK1,MTOR,RICTOR...          True  
4                                            TNFRSF14         False  
5                                                 CD4          True  
6                                                             False  
7                                                              True  
8                                         CSK,UBASH3A          True  
9                                VAV1,VAV2,SOS1,CDC42          True  
10                           LCP2,GRB2,NCK1,WAS,WIPF1          True  
11                                         PLCG1,DGKZ          True  
12  CARD11,BCL10,TRAF6,TRAF2,MAP3K7,TAB1,TAB2,CHUK...          True  
13                      NFKBIE,BIRC2,CFLAR,NFKB2,RELB         False  
14  PLCG1,STIM1,STIM2,ORAI1,ORAI3,PPP3CB,PPP3R1,CA...          True  
15                                                            False  
16                                                            False  
17  SOS1,KRAS,NRAS,RAF1,MAP2K1,MAP2K2,MAPK1,MAPK3,...          True  
18                                                            False  
19                                                            False  
20                                         RUNX3,ZEB1         False  
21                        DGKZ,CBL,ITCH,RC3H1,UBASH3A         False  
22                               STAT5A,STAT5B,IL15RA         False  
23 module scores added
151 module genes present; 4 activity modules
```

## §B.2 — Within-sample: malignant vs reactive-CD4 (per donor)

```python  # cell 28
grp_s = adata.obs["malig_group"]
donor_s = adata.obs["donor"].astype(str)
keep = grp_s.isin(GROUPS)

mod_stats = th.wilcoxon_two_group(mod_scores[keep.values], score_cols, grp_s[keep], GROUPS,
                                  level="module", donor_series=donor_s[keep])
gene_stats = th.wilcoxon_two_group(gene_expr[keep.values], genes_all, grp_s[keep], GROUPS,
                                   level="gene", donor_series=donor_s[keep])
mod_stats["module"] = mod_stats["feature"].str.replace("sig_", "", regex=False)
mod_stats["context_only"] = mod_stats["module"].isin(th.CONTEXT_ONLY)
mod_stats.to_csv(DE_DIR_S / "within_sample_module_blood_v1.csv", index=False)
gene_stats.to_csv(DE_DIR_S / "within_sample_gene_blood_v1.csv", index=False)
mod_stats.head()
```

```
feature   level    n1    n2          U  \
0          sig_L0_tcr_cd3_components  module  1586  1093  1238852.0   
1         sig_L0_pan_t_identity_loss  module  1586  1093   350670.0   
2                  sig_L0_coreceptor  module  1586  1093   734077.0   
3            sig_L1_proximal_kinases  module  1586  1093   762162.0   
4  sig_L2_lat_signalosome_components  module  1586  1093   840782.0   

               p  cliffs_delta    mean_a    mean_b  mean_diff    log2fc  \
0   9.184974e-80      0.429309  1.097377  0.778731   0.318646  0.494862   
1  1.249850e-151     -0.595419  0.214838  0.703675  -0.488837 -1.711658   
2   1.553810e-11     -0.153069 -0.368912 -0.083124  -0.285788       NaN   
3   1.064145e-07     -0.120666  0.075963  0.141837  -0.065873 -0.900858   
4   1.869355e-01     -0.029959 -0.311382 -0.263726  -0.047656       NaN   

    donor    group_a       group_b            fdr           direction  \
0  H__SS1  malignant  reactive_CD4   4.225088e-79     up_in_malignant   
1  H__SS1  malignant  reactive_CD4  1.437328e-150  up_in_reactive_CD4   
2  H__SS1  malignant  reactive_CD4   2.552688e-11  up_in_reactive_CD4   
3  H__SS1  malignant  reactive_CD4   1.359741e-07  up_in_reactive_CD4   
4  H__SS1  malignant  reactive_CD4   2.047388e-01  up_in_reactive_CD4   

                          module  context_only  
0          L0_tcr_cd3_components          True  
1         L0_pan_t_identity_loss         False  
2                  L0_coreceptor          True  
3            L1_proximal_kinases          True  
4  L2_lat_signalosome_components          True
```

```python  # cell 29
# module x donor clustered heatmap (module x donor dotplot dropped -- uninformative)
def _first(s): return s.astype(str).iloc[0]

_ob = adata.obs.groupby("donor", observed=True)
donor_annot = pd.DataFrame({"study": _ob["study"].agg(_first),
                            "stage": _ob["stage_class"].agg(_first) if "stage_class" in adata.obs else "",
                            "disease": _ob["disease"].agg(_first)})
STRIP_PALS = {"stage": {"early": "#4daf4a", "advanced": "#e41a1c"},
              "disease": {"SS": "#984ea3", "MF": "#ff7f00", "HC": "#999999"}}
cg = th.module_donor_clustermap(mod_stats, group_col="donor", feature_col="module",
                                value_col="cliffs_delta", col_annot=donor_annot,
                                annot_palettes=STRIP_PALS,
                                title="Malignant vs reactive-CD4 module Δ (donors clustered, blood)")
cg.savefig(FIG_DIR / "tcr_signaling_blood_within_sample_module_clustermap.png", dpi=150, bbox_inches="tight")
```

```
<Figure size 1080x840 with 5 Axes>
```

```python  # cell 30
# gene-level consistency across donors + per-family lollipops + activity-module violins
# g = (gene_stats.assign(sig=gene_stats["fdr"] < 0.05)
#      .groupby("feature")
#      .agg(cliffs_delta=("cliffs_delta", "mean"), n_sig=("sig", "sum"), n_donor=("donor", "nunique"))
#      .reset_index())
# g["module"] = g["feature"].map(lambda x: th.GENE_TO_MODULE.get(x, [""])[0])
# top = g.reindex(g["cliffs_delta"].abs().sort_values(ascending=False).index).head(40)
# ax = th.lollipop_effects(top, effect="cliffs_delta", label="feature", fdr_col=None,
#                          title="Top genes: malignant vs reactive-CD4 (mean Cliff's δ, blood)", figsize=(5, 8))
# ax.figure.savefig(FIG_DIR / "tcr_signaling_blood_within_sample_gene_lollipop.png", dpi=150, bbox_inches="tight")

# fam_figs = th.lollipop_per_family(g, effect="cliffs_delta", feature_col="feature", fdr_col=None,
#                                   title_prefix="mal vs reactive-CD4: ")
# for _m, _fig in fam_figs.items():
#     _fig.savefig(FIG_DIR / f"tcr_signaling_blood_within_sample_lollipop_{_m}.png", dpi=150, bbox_inches="tight")

# obs_v = adata.obs.loc[keep.values, ["malig_group", *ACT_COLS]].copy()
# for col in ACT_COLS:
#     fig, _ = th.module_violin(obs_v, col, split_col="malig_group", order=list(GROUPS), title=col)
#     fig.savefig(FIG_DIR / f"tcr_signaling_blood_within_sample_violin_{col}.png", dpi=150, bbox_inches="tight")
# print(len(fam_figs), "family lollipops;", len(ACT_COLS), "activity-module violins")
```

## §B.3 — Within malignant: each nested subclone vs its donor's reactive-CD4

```python  # cell 32
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
subclone_mod.to_csv(DE_DIR_S / "subclone_vs_control_module_blood_v1.csv", index=False)
subclone_gene.to_csv(DE_DIR_S / "subclone_vs_control_gene_blood_v1.csv", index=False)
print(subclone_mod["subclone"].nunique(), "subclones (module);",
      subclone_gene["subclone"].nunique(), "(gene) vs same-donor reactive-CD4")
```

```
240 subclones (module); 240 (gene) vs same-donor reactive-CD4
```

```python  # cell 33
# cohort clustered heatmap + per-subclone gene volcanoes (module x subclone dotplot dropped -- uninformative)
if not subclone_mod.empty:
    n_sub = subclone_mod["subclone"].nunique()
    _gs = adata.obs[adata.obs[SUBCLONE_COL].astype(str) != ""].groupby(SUBCLONE_COL, observed=True)
    sub_meta = pd.DataFrame({"sample": _gs["donor"].agg(_first),
                             "study": _gs["study"].agg(_first),
                             "stage": _gs["stage_class"].agg(_first) if "stage_class" in adata.obs else "",
                             "disease": _gs["disease"].agg(_first)})
    cg = th.module_donor_clustermap(subclone_mod, group_col="subclone", feature_col="module",
                                    value_col="cliffs_delta", col_annot=sub_meta,
                                    annot_palettes=STRIP_PALS, show_col_labels=False,
                                    title="Subclone vs reactive-CD4 module Δ (subclones clustered, blood)",
                                    figsize=(max(9, 0.18 * n_sub + 4), 7))
    cg.savefig(FIG_DIR / "tcr_signaling_blood_subclone_module_clustermap.png", dpi=150, bbox_inches="tight")

VOLC_DIR = FIG_DIR / "tcr_signaling_blood_subclone_volcano"; VOLC_DIR.mkdir(exist_ok=True)
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
240 subclone volcanoes -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/tcr_signaling_blood_subclone_volcano

<Figure size 5664x840 with 5 Axes>
```

## §B.4 — Global + compartment: all malignant vs all reactive-CD4

```python  # cell 35
# mod_pooled = th.wilcoxon_two_group(mod_scores[keep.values], score_cols, grp_s[keep], GROUPS,
#                                    level="module", donor_series=None)
# mod_pooled["module"] = mod_pooled["feature"].str.replace("sig_", "", regex=False)
# mod_pooled["context_only"] = mod_pooled["module"].isin(th.CONTEXT_ONLY)
# mod_pooled.to_csv(DE_DIR_S / "global_module_blood_v1.csv", index=False)
# ax = th.lollipop_effects(mod_pooled, label="module",
#                          title="All malignant vs all reactive-CD4 (module Cliff's δ, blood)", figsize=(5, 7))
# ax.figure.savefig(FIG_DIR / "tcr_signaling_blood_global_module_lollipop.png", dpi=150, bbox_inches="tight")

# gene_pooled = th.wilcoxon_two_group(gene_expr[keep.values], genes_all, grp_s[keep], GROUPS,
#                                     level="gene", donor_series=None)
# gene_pooled["module"] = gene_pooled["feature"].map(lambda x: th.GENE_TO_MODULE.get(x, [""])[0])
# gene_pooled.to_csv(DE_DIR_S / "global_gene_blood_v1.csv", index=False)
# act_genes = gene_pooled[gene_pooled["module"].isin(th.ACTIVITY_MODULES)]
# ax = th.lollipop_effects(act_genes, label="feature",
#                          title="Activity-module genes: malignant vs reactive-CD4 (blood)", figsize=(5, 9))
# ax.figure.savefig(FIG_DIR / "tcr_signaling_blood_global_activity_gene_lollipop.png", dpi=150, bbox_inches="tight")
```

## §B.5 — Signaling-mode localization per subclone

**Question:** not *how much* TCR signaling a subclone has, but *which node of the cascade* it sits on
relative to the other subclones in the cohort.

**How the numbers are built** (`th.subclone_contrast_vector`, helper lines ~319–338):

1. **Cells** — malignant only (`MALIG_COL = tcr_malignant_alice`), grouped by `nested_subclone`
   (transcriptomic MAJOR × CNV MINOR from §A). Empty labels dropped. No reactive-CD4 control here
   (that contrast is §B.2/§B.3).
2. **Module scores** — the `sig_<module>` columns from §B.1 (`sc.tl.score_genes` on log1p-normalized
   `X`: mean expression of the module's genes minus the mean of an expression-bin-matched control
   set, so the score is *signed* and centered near 0).
3. **Per-subclone mean** — average each `sig_<module>` over the cells of the subclone
   → `subclone × module` matrix of raw means.
4. **z down each column** — each module is z-scored *across subclones*
   (`(x - mean) / sd`, `ddof=0`; a constant column → 0). This puts modules with different dynamic
   ranges on one scale. Consequence: **all values are relative within this cohort** — z = +1.5 means
   "high for this module compared with the other blood subclones", not "high in absolute terms".
5. **Contrast = difference of two z columns** — `value = z(numerator) - z(denominator)`.
   The source doc defines these as ratios, but `score_genes` output is signed, so ratios are
   ill-defined; the z-difference is the robust analogue. The five §2 contrasts (`th.CONTRASTS`):

   | contrast | numerator − denominator | reads as |
   |---|---|---|
   | `calcium_vs_cbm` | `L3b_nfat_activity` − `L3a_nfkb_activity` | Ca²⁺/NFAT arm vs CBM→NF-κB arm |
   | `autonomous_nfkb` | `L3a_nfkb_activity` − `L4_ieg_acute` | NF-κB *without* acute IEG ⇒ ligand-independent |
   | `costim_amp` | `COSTIM_pi3k_akt_activity` − `L4_ieg_acute` | PI3K/AKT amplification uncoupled from acute trigger |
   | `chronic_vs_acute` | `L5_chronicity_exhaustion` − `L4_ieg_acute` | chronic/exhausted state vs fresh stimulation |
   | `ap1_vs_nfat` | `L3c_ap1_mapk_activity` − `L3b_nfat_activity` | MAPK/AP-1 branch vs NFAT branch |

6. **Mode clusters** (only if ≥ 4 subclones) — Ward linkage on the 5-d contrast vectors (NaN→0),
   `fcluster(t=4, criterion="maxclust")` → `mode_cluster`; each cluster is named by the contrast with
   the largest |mean| in that cluster plus its sign (e.g. `autonomous_nfkb+`).

**Caveats:** column-wise z means a cohort-wide shift is invisible by construction; small subclones
get noisier means (no size weighting); `k=4` is fixed, not selected.

The figure is the clustered `subclone × contrast` heatmap; the cell below also prints every
intermediate (raw means → z → contrasts → dominant mode → cluster profiles).

```python  # cell 37
mal = adata[malig].copy()
mal_donor = mal.obs["donor"].astype(str)
contrast_df = th.subclone_contrast_vector(mal.obs, SUBCLONE_COL, score_prefix="sig_")
contrast_df = contrast_df[contrast_df.index.astype(str).str.len() > 0]
g2 = th.signaling_mode_heatmap(contrast_df, title="Subclone signaling-mode contrasts (blood)")
g2.savefig(FIG_DIR / "tcr_signaling_blood_mode_heatmap.png", dpi=150, bbox_inches="tight")

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

# ---------- numerical results of §B.5 (every step of the calculation) ----------
mod_cols = [c for c in contrast_df.columns if c.startswith("sig_")]
raw_means = contrast_df[mod_cols]


def _z(x):
    s = x.std(ddof=0)
    return (x - x.mean()) / s if s > 0 else x * 0.0


z_means = raw_means.apply(_z, axis=0)

with pd.option_context("display.width", 220, "display.max_rows", 400, "display.max_columns", 50):
    print(f"n_subclones = {len(contrast_df)} | n_contrasts = {len(contrast_names)} | "
          f"malignant cells = {int(malig.sum())}")
    print("\ncontrast definitions  [value = z(num) - z(den), z taken across subclones]")
    for k, (num, den) in th.CONTRASTS.items():
        print(f"  {k:<17} = z(sig_{num}) - z(sig_{den})")

    print("\n-- (1) raw per-subclone module-score means (score_genes, averaged over cells) --")
    print(raw_means.round(4).to_string())

    print("\n-- (2) same means, z-scored down each module column (across subclones) --")
    print(z_means.round(2).to_string())

    print("\n-- (3) contrast matrix (what the heatmap shows) --")
    print(contrast_df[contrast_names].round(2).to_string())

    print("\n-- (4) per-contrast distribution across subclones --")
    print(contrast_df[contrast_names].describe().T[["mean", "std", "min", "50%", "max"]]
          .round(2).to_string())

    dom = contrast_df[contrast_names].abs().idxmax(axis=1)
    dom_val = pd.Series([contrast_df.at[i, c] for i, c in dom.items()], index=dom.index)
    tab = pd.DataFrame({"dominant_contrast": dom, "value": dom_val.round(2)})
    for c in ("mode_cluster", "mode_label"):
        if c in contrast_df:
            tab[c] = contrast_df[c]
    print("\n-- (5) dominant (largest |value|) mode per subclone --")
    print(tab.to_string())
    print("\ndominant-contrast counts:")
    print(dom.value_counts().to_string())

    if "mode_cluster" in contrast_df:
        print("\n-- (6) mode-cluster mean profiles (ward linkage, maxclust=4) --")
        prof = (contrast_df.groupby("mode_cluster")[contrast_names].mean().round(2)
                .join(contrast_df.groupby("mode_cluster").size().rename("n_subclones"))
                .join(contrast_df.groupby("mode_cluster")["mode_label"].first()))
        print(prof.to_string())
        print("\nmode_label counts:")
        print(contrast_df["mode_label"].value_counts().to_string())
```

```
n_subclones = 256 | n_contrasts = 5 | malignant cells = 122510

contrast definitions  [value = z(num) - z(den), z taken across subclones]
  calcium_vs_cbm    = z(sig_L3b_nfat_activity) - z(sig_L3a_nfkb_activity)
  autonomous_nfkb   = z(sig_L3a_nfkb_activity) - z(sig_L4_ieg_acute)
  costim_amp        = z(sig_COSTIM_pi3k_akt_activity) - z(sig_L4_ieg_acute)
  chronic_vs_acute  = z(sig_L5_chronicity_exhaustion) - z(sig_L4_ieg_acute)
  ap1_vs_nfat       = z(sig_L3c_ap1_mapk_activity) - z(sig_L3b_nfat_activity)

-- (1) raw per-subclone module-score means (score_genes, averaged over cells) --
                   sig_COSTIM_pi3k_akt_activity  sig_L3a_nfkb_activity  sig_L3b_nfat_activity  sig_L3c_ap1_mapk_activity  sig_L4_ieg_acute  sig_L5_chronicity_exhaustion
nested_subclone                                                                                                                                                         
B2__CTCL2_A                             -0.2536                 0.1677                 0.0125                     0.3460            0.3287                        0.0846
B2__CTCL2_B                             -0.2493                 0.1606                -0.0012                     0.3199            0.3020                        0.0519
B2__CTCL2_C                             -0.0824                 0.1231                -0.0018                     0.2841            0.2585                        0.0551
B2__CTCL2_D                             -0.3679                -0.0144                -0.0414                    -0.0091            0.0058                        0.0371
B2__CTCL5_A1                            -0.3374                 0.2949                -0.1113                     0.7199            0.6609                       -0.0125
B2__CTCL5_A2                            -0.3234                 0.3043                -0.0971                     0.7494            0.6849                        0.0355
B2__CTCL5_A3                            -0.2316                 0.3616                -0.0626                     0.8389            0.8594                        0.0987
B2__CTCL6_A                             -0.4165                 0.1151                -0.0441                     0.0377           -0.0052                        0.1170
B2__CTCL6_B                             -0.3173                 0.1267                -0.0758                     0.1413            0.1178                        0.0415
B2__CTCL6_C                             -0.1417                 0.2124                -0.0398                     0.2165            0.1959                        0.0437
B4__SZ30_HTO10_A1                       -0.1312                 0.2965                 0.1237                    -0.0315            0.0334                        0.2679
B4__SZ30_HTO10_A2                       -0.0107                 0.2973                 0.1547                    -0.0055            0.0721                        0.3253
B4__SZ30_HTO10_A3                       -0.1447                 0.2101                 0.0604                    -0.0044            0.0042                        0.2615
B4__SZ30_HTO10_B                        -0.0673                 0.2798                 0.1052                    -0.0370            0.0216                        0.2814
B4__SZ30_HTO6_A1                        -0.1011                 0.2535                 0.1327                    -0.0322            0.0519                        0.2478
B4__SZ30_HTO6_A2                        -0.1947                 0.1720                 0.0422                     0.0034            0.0417                        0.2044
B4__SZ30_HTO6_A3                        -0.1735                 0.1606                 0.0377                    -0.0426            0.0517                        0.1763
B4__SZ30_HTO6_B                          0.1224                 0.3469                 0.2360                     0.0364            0.0957                        0.4746
B4__SZ30_HTO6_C            
... [127919 more chars truncated]
```

## §B.6 — arm-CNV × downstream activity cross



```python  # cell 40
# tb = pd.read_csv(TRUNK_BRANCH_V3)
# arm_events = th.parse_arm_events(tb)
# sub_means = mal.obs.groupby(SUBCLONE_COL, observed=True)[ACT_COLS].mean()
# sub_means["donor"] = mal.obs.groupby(SUBCLONE_COL, observed=True)["donor"].agg(
#     lambda x: x.astype(str).iloc[0])
# sub_means = sub_means[sub_means.index.astype(str).str.len() > 0].reset_index()

# for armn, direction, module, note in th.ARM_ACTIVITY_MAP:
#     col = f"sig_{module}"
#     if col not in sub_means.columns:
#         continue
#     fig, ax = th.arm_activity_box(sub_means, arm_events, armn, direction, module,
#                                   title=f"{armn}{direction} -> {module}\n{note}")
#     fig.savefig(FIG_DIR / f"tcr_signaling_blood_cnv_{armn}{direction}_{module}.png", dpi=150, bbox_inches="tight")

# # support: per-donor Spearman(continuous arm score, matched activity module) over malignant cells
# from scipy.stats import spearmanr
# rows = []
# arm_mal = arm.reindex(mal.obs_names)
# for armn, direction, module, _ in th.ARM_ACTIVITY_MAP:
#     acol = f"sig_{module}"
#     if armn not in arm_mal.columns or acol not in mal.obs:
#         continue
#     for d in mal_donor.unique():
#         m = (mal_donor == d).values
#         x = arm_mal[armn].to_numpy()[m]; y = mal.obs[acol].to_numpy()[m]
#         ok = np.isfinite(x) & np.isfinite(y)
#         if ok.sum() < 30:
#             continue
#         rho, p = spearmanr(x[ok], y[ok])
#         rows.append({"donor": d, "arm": armn, "direction": direction, "module": module,
#                      "rho": rho, "p": p, "n": int(ok.sum())})
# arm_corr = pd.DataFrame(rows)
# arm_corr.to_csv(DE_DIR_S / "arm_activity_correlation_blood_v1.csv", index=False)
# print("arm x activity correlations:", arm_corr.shape)
```

## §B.7 — Sensitivity (IEG dissociation) & save

```python  # cell 42
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
out.to_csv(DATA / "tcr_signaling_subclone_modes_blood_v1.csv", index=False)
print("wrote", DATA / "tcr_signaling_subclone_modes_blood_v1.csv")
print("DE tables in", DE_DIR_S, "| figures in", FIG_DIR)
```

```
module  cliffs_full  cliffs_nolabile
0           L4_ieg_acute     0.090205        -0.046185
1  L3c_ap1_mapk_activity     0.205084         0.040829
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/tcr_signaling_subclone_modes_blood_v1.csv
DE tables in /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/tcr_signaling_de_blood | figures in /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures
```

# §C — Machine-readable numeric summary

One flat `KEY: value` dump of every number this notebook produces (blood cohort). Run **last** (reuses
the in-memory objects from §A/§B). Blocks whose inputs are missing are skipped. Also written to
`tables/subclone_tcr_signaling_blood_summary.txt`.

```python  # cell 44
# --- §C · Machine-readable summary: every number this notebook reports, in one dump ---
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
kv("notebook", "35_blood_subclone_tcr_signaling.ipynb — blood subclonal evolution + TCR/costim signaling")
kv("object", str(OBJ))
kv("malignancy_definition",
   "tcr_malignant_alice (nb34 ALICE dominant founder + <=1-aa TRB-CDR3 variant family)")
kv("inputs", f"{ALICE_MAL.name}, {MALIG_PARQUET.name}, {ARM.name}")
kv("outputs", f"{SUBCLONE_PARQUET.name}, {SUMMARY_CSV.name}, {TRUNKBRANCH_CSV.name}, "
              f"{ELIG_CSV.name}, {SIGN_JSON.name}")
kv("params", f"SEED={SEED} MIN_MAL={MIN_MAL} K_MAX={K_MAX} MIN_SUB={MIN_SUB} "
             f"NULL_Z={NULL_Z} NULL_DRAWS={NULL_DRAWS} "
             f"SIL_MIN={SIL_MIN} MIN_ARM_DELTA={MIN_ARM_DELTA} Z_THR={Z_THR} EPS={EPS} "
             f"BRANCH_DELTA={BRANCH_DELTA}")
kv("major_track", f"Leiden res={LEIDEN_RES}, n_neighbors={N_NEIGHBORS} on {LATENT}")
kv("minor_track", "KMeans on the 41-arm inferCNV matrix (nb34 blood_T_arm_cnv_v3)")
kv("subclone_gates", f"null-anchored: mean + {NULL_Z} x sd of a size/k-matched KMeans split of "
                     f"{NULL_X.shape[0]:,} known-diploid healthy-CD4 cells ({NULL_DRAWS} draws). "
                     f"Absolute SIL_MIN={SIL_MIN}/MIN_ARM_DELTA={MIN_ARM_DELTA} are fallback only "
                     f"-- on this matrix a diploid split scores arm_delta ~0.047, silhouette ~0.065")
kv("diploid_reference", "held-out half of the shared reference (`<run>|NULL|<cell>` rows): "
                        "hc_atlas = same-atlas healthy blood CD4, healthy_pbmc = external OneK1K CD4")

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
kv("malignant_ALICE(tcr_malignant_alice)", f"{int(_mal.sum()):,} ({pct(int(_mal.sum()), N)})")
kv("cnv_arm_malignant(nb34)", f"{int(_cnv.sum()):,} ({pct(int(_cnv.sum()), N)})")
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
if "sil_thr" in summary:
    kv("null_calibrated_donors", f'{int(summary["null_calibrated"].sum())}/{len(summary)}')
    kv("gate_median(sil_thr/arm_delta_thr)",
       f'{summary["sil_thr"].median():.4f} / {summary["arm_delta_thr"].median():.4f}')
    _marg = summary[(summary["k"] > 1) & (summary["sil_margin"] < 1.25)]
    kv("sil_margin_median(silhouette / its own null gate; 1.0 = at the noise floor)",
       f'{summary["sil_margin"].median():.2f}')
    kv("splits_within_25pct_of_the_null_gate(marginal)",
       f'{len(_marg)}/{int((summary["k"] > 1).sum())}'
       + (f': {", ".join(_marg["donor"].astype(str))}' if len(_marg) else ""))
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

if has("signatures"):
    SEC("7 · EXPORTED GENE SIGNATURES")
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
P("-- disease --")
for k, v in obs["disease"].astype(str).value_counts().items():
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
                   (disease_stats if has("disease_stats") else None, "disease")]:
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
for f in sorted(FIG_DIR.glob("subclone_blood_v1*.png")) + sorted(FIG_DIR.glob("tcr_signaling_blood*.png")):
    P(f"   {f.name:<58} {f.stat().st_size / 1e3:7.0f} kB")
if has("SAMPLE_FIG_DIR") and SAMPLE_FIG_DIR.exists():
    kv("per_sample_arm_profiles", f"{len(list(SAMPLE_FIG_DIR.glob('*.png')))} png in {SAMPLE_FIG_DIR}")
if has("VOLC_DIR") and VOLC_DIR.exists():
    kv("subclone_volcanoes", f"{len(list(VOLC_DIR.glob('*.png')))} png in {VOLC_DIR}")
P(f"tables ({DE_DIR_S}):")
for f in sorted(DE_DIR_S.glob("*.csv")):
    P(f"   {f.name:<58} {f.stat().st_size / 1e3:7.0f} kB")
P("data outputs:")
for f in [SUBCLONE_PARQUET, SUMMARY_CSV, TRUNKBRANCH_CSV, ELIG_CSV, SIGN_JSON,
          DATA / "tcr_signaling_subclone_modes_blood_v1.csv"]:
    P(f"   {f.name:<58} exists={f.exists()}")

_TXT = TAB_DIR / "subclone_tcr_signaling_blood_summary.txt"
_TXT.write_text("\n".join(_LOG))
print(f"\n[summary written -> {_TXT}]")
```

```
==============================================================================
## 0 · PROVENANCE & PARAMETERS
==============================================================================
notebook: 35_blood_subclone_tcr_signaling.ipynb — blood subclonal evolution + TCR/costim signaling
object: /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/blood_T_annotated.h5ad
malignancy_definition: tcr_malignant_alice (nb34 ALICE dominant founder + <=1-aa TRB-CDR3 variant family)
inputs: alice_malignancy_blood_v1.parquet, blood_T_malignancy_v3.parquet, blood_T_arm_cnv_v3.parquet
outputs: subclones_blood_v1.parquet, subclone_summary_blood_v1.csv, subclone_trunk_branch_blood_v1.csv, subclone_donor_eligibility_blood_v1.csv, subclone_signatures_blood_v1.json
params: SEED=0 MIN_MAL=200 K_MAX=4 MIN_SUB=50 NULL_Z=3.0 NULL_DRAWS=8 SIL_MIN=0.15 MIN_ARM_DELTA=0.03 Z_THR=2.5 EPS=0.01 BRANCH_DELTA=0.03
major_track: Leiden res=0.2, n_neighbors=15 on X_mrvi_u
minor_track: KMeans on the 41-arm inferCNV matrix (nb34 blood_T_arm_cnv_v3)
subclone_gates: null-anchored: mean + 3.0 x sd of a size/k-matched KMeans split of 247,663 known-diploid healthy-CD4 cells (8 draws). Absolute SIL_MIN=0.15/MIN_ARM_DELTA=0.03 are fallback only -- on this matrix a diploid split scores arm_delta ~0.047, silhouette ~0.065
diploid_reference: held-out half of the shared reference (`<run>|NULL|<cell>` rows): hc_atlas = same-atlas healthy blood CD4, healthy_pbmc = external OneK1K CD4

==============================================================================
## 1 · OBJECT & COMPARTMENT
==============================================================================
n_cells: 302,362
n_genes: 10,000
n_donors: 98
n_studies: 6

-- T subtype (cell_type_T2) --
   CD4            n= 240,993 ( 79.7%)
   CD8            n=  60,638 ( 20.1%)
   NK             n=     731 (  0.2%)
malignant_ALICE(tcr_malignant_alice): 122,510 (40.52%)
cnv_arm_malignant(nb34): 83,043 (27.46%)
TCR_and_CNV_overlap: 63,840 | jaccard=0.450
cells_with_arm_CNV_coverage: 60.96%
donors_with_ALICE_1aa_variant_family: 0/62

==============================================================================
## 2 · DONOR ELIGIBILITY FOR SUBCLONE DETECTION
==============================================================================
cohort_donors: 98
eligible(>=MIN_MAL malignant): 53 (54.08%)

-- exclusion reasons --
   no ALICE-malignant cells (no dominant TCR clone)     n_donors=35
   <200 malignant cells — too few for subclone KMeans   n_donors=8
   no arm-CNV coverage — sample lacked >=20 CD8 diploid reference for inferCNV (nb34) n_donors=1
   healthy control — no malignant compartment           n_donors=1

-- eligible donors (malignant cells) --
         donor           study disease  n_cells  n_malignant
     B2__CTCL2 borcherding2023      SS    31565        28596
      B5__PT14        buus2025      SS     7422         5305
      B5__PT02        buus2025      SS     5835         5142
      B5__PT06        buus2025      SS     5858         4782
 B4__SZ38_HTO3      geskin2026      SS     4707         4287
      B5__PT08        buus2025      SS     4703         4163
  B4__SZ5_HTO2      geskin2026      SS     5004         4051
 B4__SZ38_HTO1      geskin2026      SS     4408         3983
 B4__SZ38_HTO2      geskin2026      SS     4145         3724
 B4__SZ38_HTO6      geskin2026      SS     3937         3554
  B4__SZ5_HTO5      geskin2026      SS     4107         3549
 B4__SZ38_HTO5      geskin2026      SS     3675         3333
  B4__SZ5_HTO3      geskin2026      SS     3917         3288
  B4__SZ5_HTO4      geskin2026      SS     3750         3240
 B4__SZ38_HTO4      geskin2026      SS     3378         3127
      B5__PT01        buus2025      SS     3396         2971
      B5__PT05        buus2025      SS     5020         2957
  B4__SZ5_HTO6      geskin2026      SS     3405         2910
  B4__SZ5_HTO1      geskin2026      SS     3001         2574
      B5__PT07        buus2025      SS     2628  
... [110985 more chars truncated]
```
