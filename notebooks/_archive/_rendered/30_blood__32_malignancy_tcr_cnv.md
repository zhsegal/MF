# 30_blood/32_malignancy_tcr_cnv.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/30_blood/32_malignancy_tcr_cnv.ipynb` and is never run._

# 32 — Blood malignant annotation: ALICE TCR + inferCNV (v4, on the **v2** annotation)

Blood analogue of `23_malignancy_tcr_cnv`. Runs the per-cell malignancy call on the blood T-cell re-annotation
`data/atlas_joint/blood_T_annotated.h5ad` (`31_reannotation`, rebuilt 2026-09-07: **667,584 T/NK cells,
75 patients**). Reuses the ALICE (`alice_helpers`) and inferCNV (`skin_T_cnv_helpers`) engines.

> **v1 → v2 cohort bump, 2026-10-22.** The `_v1`/`_v3` artifacts on disk were written in August
> against the **v1** blood annotation (~302k T cells, 98 patients) and their `cell_id`s do not
> match this object. Every output below is therefore bumped — ALICE to `_blood_v2`, CNV to `_v4` —
> and the old files are left in place. This is the same failure `11_atlas_v2_qc` exists to catch:
> the filenames alone would have let a v1 cache load silently under a v2 input.
>
> Downstream: `32b_cluster_malignancy` consumes `alice_malignancy_blood_v2.parquet` and the
> per-sample clonality table. `33_subclone_tcr_signaling` was retired on 2026-10-22.

**Blood-specific vs `23_malignancy_tcr_cnv`:**
- The blood T object already carries TCR obs (`tra_cdr3`/`trb_cdr3`/`has_tcr`/`clone_id`), MrVI
  latents (`X_mrvi_u`) and `raw_counts` — so it is loaded directly (no Li2024 TCR fold-in).
- inferCNV uses a **pooled** diploid reference (same-sample CD8 + healthy blood CD4 + external
  healthy PBMC CD4), because blood is ~92 % malignant CD4 (Sézary) so benign CD4 is scarce and
  there is no healthy-skin atlas to lean on.
- No skin-specific pieces: no Li2024 parquet, no `PT35` special case, no external healthy atlas.

**v3 (this revision)** rebuilds the CNV half after v2 called **61 %** of the held-out healthy
control malignant and sat at TPR 0.66 / FPR 0.66 — chance — on cells carrying both signals:

| | v2 | v3 |
|---|---|---|
| gene space | 10k HVG → 7,463 positioned, `window=250` | full 40,821 → ~20k positioned, `window=100` |
| estimator | infercnvpy's bounded multi-category dead-band; `dynamic_threshold` per cell chunk | one pooled reference vector; no dynamic threshold; shuffled chunks |
| `nonclonal` | baseline (tumour-contaminated) | query |
| threshold | GMM crossover + *median* of the donor's own diploid cells, ×0.8 | 0.99 quantile of a **held-out diploid null**, depth-conditioned |
| statistic | `mean|X_cnv|` (unsigned magnitude) | signed projection onto a **de novo** arm-level clone signature |
| no-signal donors | silently called negative | `indeterminate`; label falls back to TCR |

The CNV signature is derived **without any TCR input**, so the CNV call remains independent
evidence rather than a clonality amplifier. Outputs are suffixed `_v3` and feed `33_subclone_tcr_signaling`; v2 stays on
disk for comparison.

> **HEAVY** cells (full-gene object build, OLGA Pgen sweep, two per-sample inferCNV passes) run on
> the GPU/compute kernel — not the login node. Submit with `jobs/run_blood_cnv.sh`.

```python  # cell 1
# ============================================================
# Parameters — inputs, cohort filter, ALICE + inferCNV knobs.
# ============================================================
import hashlib, json
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
TAB_DIR = NB_DIR / "tables"; TAB_DIR.mkdir(exist_ok=True)

# ---- inputs (the blood re-annotation from nb33) ----
ANNOT_OBJ = OUT_DIR / "blood_T_annotated.h5ad"           # nb33 re-annotation (10k HVG)
JOINT_H5  = OUT_DIR / "joint_annotated.h5ad"             # full-atlas rows, 40,821 genes
# v3: inferCNV runs on the FULL gene space. The HVG object leaves only ~7.5k positioned genes,
# so a 250-gene window spans ~3% of the genome, and HVGs are selected for exactly the
# state-driven variance inferCNV assumes away. Built once by C.build_fullgene_blood_cnv_object.
# Versioned with the cohort: the unsuffixed file on disk holds the 302,362-cell v1 blood
# set, and the cache is keyed by filename, so reusing the name would have made this whole v4
# run a v1 run under v4 output names.
OBJ = OUT_DIR / "blood_T_fullgene_cnv_input_v2.h5ad"
GENE_MIN_CELLS = 20                                      # expression filter, not a variance filter
GTF = NB_DIR / "data" / "cache" / "Homo_sapiens.GRCh38.110.chr.gtf.gz"
# external healthy-PBMC CD4 diploid reference (OneK1K); fetch with jobs/download_onek1k.sh
ONEK1K_H5        = NB_DIR / "data" / "external_ref" / "onek1k.h5ad"
HEALTHY_PBMC_REF = OUT_DIR / "healthy_pbmc_cd4_ref_v1.h5ad"   # cached ~12k-cell subset

# ---- dominant-clone rule + cohort filter ----
FRAC_THRESH, RATIO_THRESH, EXPANDED_MIN = 0.05, 1.33, 2
CD4_T_TYPES = ["CD4"]          # CD4 = malignant (Sezary) arm
CD8_T_TYPES = ["CD8"]          # CD8 = reactive arm + inferCNV diploid reference
# Judge dominance on the CD4 arm only. Pooling CD4+CD8 lets a co-expanded reactive CD8
# clone sit just under the tumour clone and drag top/second below RATIO_THRESH -- that
# silently benign-ed 4 of the 5 SZ30 dupilumab lanes (and B5__PT23) despite a ~60% CD4
# tumour fraction. Set to None for the legacy pooled CD4+CD8 rule.
DOM_LINEAGE = CD4_T_TYPES
DROP_ENTITIES = {"MF_gamma_delta", "CD8_aggressive_epidermotropic_CTCL"}
KEPT_LINEAGES = {"CD4", "not_applicable"}   # blood: disease samples are CD4-lineage; HC = not_applicable

# ---- ALICE params ----
ALPHA = 0.05
Q = None              # None -> calibrate per repertoire from the null bulk
SEED = 0

# ---- outputs (_blood_v2 = built on the v2 blood annotation; the _v1 set stays on disk) ----
ALICE_CD4_PARQUET = OUT_DIR / "alice_cd4_per_donor_blood_v2.parquet"
ALICE_CD8_PARQUET = OUT_DIR / "alice_cd8_blood_v2.parquet"
ALICE_MAL_PARQUET = OUT_DIR / "alice_malignancy_blood_v2.parquet"   # <- 32b reads this
USE1_CSV = OUT_DIR / "alice_use1_subclone_families_blood_v2.csv"
USE2_CSV = OUT_DIR / "alice_use2_crosspatient_blood_v2.csv"
USE3_CSV = OUT_DIR / "alice_use3_reactive_cd8_blood_v2.csv"
# Per-sample clonality, the blood analogue of nb23 Step 4. 32b needs it to decide which samples
# ALICE covers; without it every sample would be treated as "apply" and the cluster model would
# train on nothing.
CLONALITY_CSV = TAB_DIR / "blood_tcr_clonality_per_sample.csv"
# v3 = full gene space + mean-subtraction estimator + null-anchored, de novo arm-consensus caller.
# (v1 = same-sample-CD8 reference; v2 = pooled reference with the self-referential GMM caller,
#  which called 61% of the held-out healthy control malignant. nb35 reads v3.)
OUT_PARQUET = OUT_DIR / "blood_T_malignancy_v4.parquet"          # per-cell malignancy
ARM_CACHE   = OUT_DIR / "blood_T_arm_cnv_v4.parquet"             # per-arm CNV
QC_CSV      = TAB_DIR / "blood_cnv_v4_qc.csv"                    # per-donor CNV QC (Step 15)
```

```
NB_DIR = /home/projects/nyosef/zvise/MF
```

```python  # cell 2
import sys, gc, importlib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

sys.path.insert(0, str(NB_DIR / "helpers"))
import atlas_join_helpers as H
import skin_T_cnv_helpers as C
import alice_helpers as A
for _m in (H, C, A):
    importlib.reload(_m)
np.random.seed(SEED)
sc.settings.verbosity = 1
```

```
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
```

## Step 1 — load blood T object, recompute dominant clone, cohort filter · HEAVY

The blood T object already carries TCR obs and `raw_counts`, so it is read directly. `cell_type_T`
(CD4/CD8/NK) is aliased to `cell_type_T2` for the shared helpers. `C.recompute_dominant_clone`
rebuilds the unified TRB clone key + per-donor dominant-clone call from `tra_cdr3`/`trb_cdr3`
(there is **no** Li2024 fold-in for blood, so `is_li` is all-False and `_li_clone_key` is empty).

```python  # cell 4
# Full-gene CNV input: same cells and same obs as nb33's blood_T_annotated.h5ad, re-expressed
# over the 40,821-gene space (filtered to genes seen in >= GENE_MIN_CELLS cells). Built once by
# h5py row-slice from the 48 GB joint object; cached, so later runs just read it back.
adata = C.build_fullgene_blood_cnv_object(JOINT_H5, ANNOT_OBJ, OBJ, min_cells=GENE_MIN_CELLS)
assert "raw_counts" in adata.layers, "expected raw_counts layer"
print("CNV gene space:", adata.n_vars, "genes")
adata.obs["cell_type_T2"] = adata.obs["cell_type_T"].astype(str)
adata.obs["has_tcr"] = adata.obs["has_tcr"].astype(bool)
adata.obs["cached_malignant"] = (adata.obs["sezary_like"].astype(bool)
                                 if "sezary_like" in adata.obs else False)
print("cell_type_T:\n", adata.obs["cell_type_T"].value_counts())
print("\nstudy:\n", adata.obs["study"].value_counts())

# blood has no Li2024 TCR fold-in — build the clone key purely from CDR3
adata.obs["_li_clone_key"] = ""
is_li = np.zeros(adata.n_obs, dtype=bool)
dom_mask = (adata.obs["cell_type_T2"].isin(DOM_LINEAGE).to_numpy()
            if DOM_LINEAGE else None)   # dominance judged on the CD4 arm only
dom_tbl = C.recompute_dominant_clone(adata, H, is_li, FRAC_THRESH, RATIO_THRESH,
                                     EXPANDED_MIN, dom_mask=dom_mask)
clone_summary = C.clone_summary_table(adata, FRAC_THRESH, RATIO_THRESH, dom_mask=dom_mask)

META_COLS = ["study", "dataset", "disease", "disease_stage", "tissue", "entity", "sex", "tech"]
meta_cols = [c for c in META_COLS if c in adata.obs.columns]
donor_meta = (adata.obs[["donor", *meta_cols]].astype(str)
              .groupby("donor", observed=True)
              .agg(lambda s: ", ".join(sorted(s[s != "nan"].unique()))))
clone_summary = clone_summary.merge(donor_meta, on="donor", how="left")

# sample-selection: drop non-ab-CD4 entities + size gate (>=300 TCR cells, herrera exempt).
drop = {}
for d in clone_summary.loc[clone_summary["entity"].isin(DROP_ENTITIES), "donor"]:
    drop.setdefault(d, "non-ab-CD4 entity")
small = (clone_summary["n_tcr_cells"] < 300) & (clone_summary["study"] != "herrera2021")
for d in clone_summary.loc[small, "donor"]:
    drop.setdefault(d, "n_tcr_cells < 300")
keep_donors = clone_summary.loc[~clone_summary["donor"].isin(drop), "donor"].tolist()
clone_summary = clone_summary[~clone_summary["donor"].isin(drop)].reset_index(drop=True)
adata = adata[adata.obs["donor"].isin(keep_donors)].copy()

# HC samples are benign by definition.
hc = adata.obs["disease"].astype(str).eq("HC").to_numpy()
adata.obs.loc[hc, ["tcr_is_malignant", "tcr_is_dominant_clone"]] = False
clone_summary.loc[clone_summary["disease"].eq("HC"), "malignant"] = False
print("dropped donors:", drop)
print("kept donors:", len(keep_donors), "| cells:", adata.n_obs)
```

```
loaded cached full-gene CNV object (blood_T_fullgene_cnv_input_v2.h5ad): (667584, 29636)
CNV gene space: 29636 genes
cell_type_T:
 cell_type_T
CD4    554369
CD8    112609
NK        606
Name: count, dtype: int64


study:
 study
geskin2026         167599
ren2023            135396
harro2023          129793
dorando2026         77703
buus2025            61215
borcherding2023     36630
gaydosik2022        22799
herrera2021         20886
borcherding2019     14170
rindler2021_fi       1393
Name: count, dtype: int64

dropped donors: {'B4__SZ33_HTO5': 'n_tcr_cells < 300', 'B2__CTCL4': 'n_tcr_cells < 300'}
kept donors: 93 | cells: 557772
```

```python  # cell 5
# Which donors does the CD4 restriction move? Re-derive the legacy pooled rule from obs
# (clone ids are already assigned, so this is a cheap groupby -- no adata copy).
if DOM_LINEAGE:
    _o = adata.obs
    _tcr = _o["has_tcr"].to_numpy() & (_o["tcr_clone_id"].astype(str).to_numpy() != "")
    _df = pd.DataFrame({"donor": _o["donor"].astype(str).values,
                        "clone": _o["tcr_clone_id"].astype(str).values})[_tcr]
    _rows = []
    for _d, _s in _df.groupby("donor", sort=False):
        _sz = _s["clone"].value_counts()
        _top, _sec = int(_sz.iloc[0]), int(_sz.iloc[1]) if len(_sz) > 1 else 0
        _frac = _top / len(_s); _ratio = _top / _sec if _sec else np.inf
        _rows.append({"donor": _d, "top_clone_pooled": _sz.index[0], "top_n_pooled": _top,
                      "second_n_pooled": _sec, "dom_frac_pooled": round(_frac, 3),
                      "ratio_pooled": round(_ratio, 2) if np.isfinite(_ratio) else np.inf,
                      "is_dominant_pooled": (_frac >= FRAC_THRESH) and (_ratio >= RATIO_THRESH)})
    pooled = pd.DataFrame(_rows)

    cmp = (dom_tbl.set_index("donor")[["n_tcr", "n_dom_pool", "top_clone", "top_n",
                                       "dom_frac", "ratio", "is_dominant"]]
           .join(pooled.set_index("donor"), how="inner"))   # inner: cohort filter already ran
    changed = cmp[cmp["is_dominant"] != cmp["is_dominant_pooled"]]
    print(f"dominant donors: CD4-only {int(cmp['is_dominant'].sum())} "
          f"| pooled {int(cmp['is_dominant_pooled'].sum())} | changed {len(changed)}")
    print(changed.to_string())
```

```
dominant donors: CD4-only 84 | pooled 80 | changed 6
                n_tcr  n_dom_pool              top_clone  top_n  dom_frac  ratio  is_dominant       top_clone_pooled  top_n_pooled  second_n_pooled  dom_frac_pooled  ratio_pooled  is_dominant_pooled
donor                                                                                                                                                                                                 
B2__CTCL3        1801         800      TRB:CASSPQDTDTQYF      9     0.011   1.80        False    TRB:CASSLREATSYEQYF           369              123            0.205          3.00                True
B4__SZ30_HTO10    813         538  TRB:CASSLGLGTSYNSPLHF    250     0.465   3.25         True  TRB:CASSLGLGTSYNSPLHF           251              198            0.309          1.27               False
B4__SZ30_HTO6    1278         810  TRB:CASSLGLGTSYNSPLHF    407     0.502   3.51         True  TRB:CASSLGLGTSYNSPLHF           408              365            0.319          1.12               False
B4__SZ30_HTO7    1137         718  TRB:CASSLGLGTSYNSPLHF    370     0.515   4.46         True  TRB:CASSLGLGTSYNSPLHF           371              307            0.326          1.21               False
B4__SZ30_HTO9    1050         676  TRB:CASSLGLGTSYNSPLHF    331     0.490   2.98         True  TRB:CASSLGLGTSYNSPLHF           331              273            0.315          1.21               False
B5__PT23         2146         727    TRB:CASSPLAGSSYEQYF    112     0.154   6.59         True    TRB:CSASGLPSRTDTQYF           228              177            0.106          1.29               False
```

```python  # cell 6
# Per-clonotype tables: CD4 malignant cohort (Use 1/2) and CD8 infiltrate (Use 3).
obs = adata.obs
cd4 = obs[obs["cell_type_T2"].isin(CD4_T_TYPES)]
cd8 = obs[obs["cell_type_T2"].isin(CD8_T_TYPES)]
clono_cd4 = A.clonotype_table(cd4, group="donor")
clono_cd8 = A.clonotype_table(cd8, group="donor")
print("CD4 clonotypes:", len(clono_cd4), "| donors:", clono_cd4["donor"].nunique(),
      "| founders:", int(clono_cd4["is_founder"].sum()))
print("CD8 clonotypes:", len(clono_cd8), "| donors:", clono_cd8["donor"].nunique())
clono_cd4.head()
```

```
CD4 clonotypes: 111134 | donors: 93 | founders: 84
CD8 clonotypes: 26067 | donors: 93

       donor                cdr3  n_cells  is_founder
0  B2__CTCL2    CASSHAGAGGPYEQYF    29065        True
1  B2__CTCL2       CASSITAYNEQFF        6       False
2  B2__CTCL2     CASIPATGKNSPLHF        5       False
3  B2__CTCL2  CASSLRGRLSSGANVLTF        4       False
4  B2__CTCL2     CASSFWRGPSDEQFF        4       False
```

## Step 2 — OLGA generative null + sanity check

`A.load_olga_trb()` loads the default human-TRB IGoR model. Expected ≤1-aa neighbors of a clonotype =
`N_unique × Q × Σ Pgen(single-mismatch variants)`; observed vs expected is a Poisson survival test,
BH-corrected per repertoire. **Requires OLGA** — run `%pip install olga` once in this kernel.

```python  # cell 8
# %pip install olga    # uncomment + run once if olga is not importable
pgen_model = A.load_olga_trb()
pgen = A.make_pgen(pgen_model)

# sanity: a textbook single-aa-variant ALICE edge
a, b = "CASSQDRALENTIYF", "CASSQDRTLENTIYF"
print("pgen(a) =", pgen(a), " pgen(b) =", pgen(b))
print("b is a single-mismatch variant of a:", b in set(A.one_mismatch_variants(a)))
print("graph edge a-b:", A.neighbor_graph([a, b]).has_edge(a, b))
```

```
pgen(a) = 7.856194078998701e-11  pgen(b) = 4.702072793866696e-11
b is a single-mismatch variant of a: True
graph edge a-b: True
```

## Step 3 — Use 1: per-patient malignant subclone-family recovery · HEAVY (OLGA sweep)

For each CD4 donor: ALICE on the full clonotype table, then take the ≤1-aa connected component
containing the dominant founder = the malignant β-variant family. Cached (donor-set guard).

```python  # cell 10
alice_cd4 = pd.read_parquet(ALICE_CD4_PARQUET) if ALICE_CD4_PARQUET.exists() else None
if alice_cd4 is None or set(alice_cd4["donor"]) != set(clono_cd4["donor"]):
    if alice_cd4 is not None:
        print("CD4 cache stale (donor set changed) -> recompute")
    alice_cd4 = A.run_alice_by_group(clono_cd4, pgen, group="donor", Q=Q, alpha=ALPHA)
    alice_cd4.to_parquet(ALICE_CD4_PARQUET, index=False)
    print("wrote", ALICE_CD4_PARQUET)
else:
    print("loaded cached", ALICE_CD4_PARQUET)
print("CD4 donors:", alice_cd4["donor"].nunique(),
      "| significant clonotypes:", int(alice_cd4["significant"].sum()))
```

```
wrote /home/projects/nyosef/zvise/MF/data/atlas_joint/alice_cd4_per_donor_blood_v2.parquet
CD4 donors: 93 | significant clonotypes: 31
```

```python  # cell 11
# Founder subclone-family per donor + the tumor fraction / dominance it recovers.
# Malignant donors: founder = dominant clone. Non-malignant donors: founder = LARGEST clonotype.
def _dominance(sub, members):
    total = sub["n_cells"].sum()
    n = sub.loc[sub["cdr3"].isin(members), "n_cells"].sum()
    rest = sub.loc[~sub["cdr3"].isin(members), "n_cells"]
    second = int(rest.max()) if len(rest) else 0
    dom_frac = n / total if total else 0.0
    fold = (n / second) if second else np.inf
    return int(n), total, dom_frac, fold


rows = []
for donor, sub in clono_cd4.groupby("donor", observed=True):
    is_mal = bool(sub["is_founder"].any())
    if is_mal:
        founders = set(sub.loc[sub["is_founder"], "cdr3"]); src = "dominant"
    else:
        founders = {sub.sort_values("n_cells", ascending=False)["cdr3"].iloc[0]}; src = "largest_clonotype"
    sig = set(alice_cd4.loc[(alice_cd4["donor"] == donor) & alice_cd4["significant"], "cdr3"])
    fam = A.founder_family(sub, seeds=founders)
    fam_sig = set(founders) | (fam & sig)

    ex_n, total, ex_frac, ex_fold = _dominance(sub, founders)
    fam_n, _, fam_frac, fam_fold = _dominance(sub, fam_sig)
    exact_call = (ex_frac >= FRAC_THRESH) and (ex_fold >= RATIO_THRESH)
    family_call = (fam_frac >= FRAC_THRESH) and (fam_fold >= RATIO_THRESH)
    rows.append({
        "donor": donor, "founder_source": src, "founder": "; ".join(sorted(founders)),
        "n_variant_clones": max(len(fam) - len(founders), 0),
        "n_family_significant": len(fam & sig),
        "exact_frac": ex_frac, "family_frac": fam_frac, "delta_frac": (fam_n - ex_n) / total,
        "exact_fold": ex_fold, "family_fold": fam_fold,
        "exact_call": exact_call, "family_call": family_call,
        "flipped_by_family": (not is_mal) and family_call})

use1 = (pd.DataFrame(rows)
        .merge(clone_summary[["donor", "study", "disease", "disease_stage", "entity", "malignant"]],
               on="donor", how="left")
        .sort_values(["flipped_by_family", "delta_frac"], ascending=False))
use1.to_csv(USE1_CSV, index=False)
print("wrote", USE1_CSV, "| donors:", len(use1),
      "| non-malignant flipped by family:", int(use1["flipped_by_family"].sum()))
use1
```

```
wrote /home/projects/nyosef/zvise/MF/data/atlas_joint/alice_use1_subclone_families_blood_v2.csv | donors: 93 | non-malignant flipped by family: 0

            donor founder_source            founder  n_variant_clones  \
50       B5__PT16       dominant      CASSLGTGAEVFF                 1   
40       B5__PT04       dominant    CAIGPDREKRNTIYF                 1   
59        B6__P10       dominant     CSASPGLGANVLTF                 6   
52       B5__PT21       dominant   CSASGHGGMGNQPQHF                 1   
6   B4__SZ30_HTO7       dominant  CASSLGLGTSYNSPLHF                 1   
..            ...            ...                ...               ...   
87         H__SS1       dominant    CASSRQGGSGNTIYF                 0   
88         H__SS2       dominant   CASKLFMDREKGEAFF                 0   
89         H__SS3       dominant      CASSTTGNTEAFF                 0   
90         H__SS4       dominant   CASSRLGGAYNSPLHF                 0   
92         H__SS6       dominant     CASRSQGSYNEQFF                 0   

    n_family_significant  exact_frac  family_frac  delta_frac  exact_fold  \
50                     2    0.619989     0.927512    0.307523    2.016071   
40                     2    0.717224     0.989717    0.272494    2.632075   
59                     5    0.709699     0.712842    0.003142  185.535714   
52                     2    0.310428     0.312350    0.001922  161.500000   
6                      2    0.593900     0.595506    0.001605   13.703704   
..                   ...         ...          ...         ...         ...   
87                     0    0.602530     0.602530    0.000000  524.000000   
88                     0    0.645123     0.645123    0.000000  109.800000   
89                     0    0.826987     0.826987    0.000000  999.000000   
90                     0    0.348422     0.348422    0.000000   28.222222   
92                     0    0.576655     0.576655    0.000000  248.250000   

    family_fold  exact_call  family_call  flipped_by_family        study  \
50   140.750000        True         True              False     buus2025   
40   385.000000        True         True              False     buus2025   
59   186.357143        True         True              False      ren2023   
52   162.500000        True         True              False     buus2025   
6     13.740741        True         True              False   geskin2026   
..          ...         ...          ...                ...          ...   
87   524.000000        True         True              False  herrera2021   
88   109.800000        True         True              False  herrera2021   
89   999.000000        True         True              False  herrera2021   
90    28.222222        True         True              False  herrera2021   
92   248.250000        True         True              False  herrera2021   

       disease disease_stage          entity  malignant  
50          SS            NA          Sezary       True  
40          SS            NA          Sezary       True  
59  CTCL_other           IIB      CTCL_other       True  
52          SS            NA          Sezary       True  
6           SS                        Sezary       True  
..         ...           ...             ...        ...  
87          MF                MF/SS_leukemic       True  
88          MF                MF/SS_leukemic       True  
89          MF                MF/SS_leukemic       True  
90          MF                MF/SS_leukemic       True  
92          MF                MF/SS_leukemic       True  

[93 rows x 18 columns]
```

## Step 4 — Use 2: cross-patient convergence (expected negative)

Pool one founder per malignant donor and test for ≤1-aa neighborhood enrichment **across patients**
(shared antigen / superantigen). Underpowered by design — a companion to GLIPH2, not a substitute.

```python  # cell 13
founders = (clono_cd4[clono_cd4["is_founder"]]
            .drop_duplicates(["donor", "cdr3"])[["donor", "cdr3"]].copy())
pool = founders[["cdr3"]].copy(); pool["is_founder"] = False
use2 = A.alice_test(pool, pgen, Q=Q, alpha=ALPHA)
use2 = use2.merge(founders[["donor", "cdr3"]], on="cdr3", how="left")
use2.to_csv(USE2_CSV, index=False)
print("pooled founders:", len(founders),
      "| cross-patient significant:", int(use2["significant"].sum()))
use2.head(10)
```

```
pooled founders: 84 | cross-patient significant: 0

               cdr3  is_founder  obs_deg    Q  exp_deg  pois_p  q_bh  \
0  CASSHAGAGGPYEQYF       False        0  1.0      0.0     1.0   1.0   
1      CASSSRVYGYTF       False        0  1.0      0.0     1.0   1.0   
2     CASSNQGNQPQHF       False        0  1.0      0.0     1.0   1.0   
3    CASSPGTAYYGYTF       False        0  1.0      0.0     1.0   1.0   
4    CASSPGTAYYGYTF       False        0  1.0      0.0     1.0   1.0   
5    CASSPGTAYYGYTF       False        0  1.0      0.0     1.0   1.0   
6    CASSPGTAYYGYTF       False        0  1.0      0.0     1.0   1.0   
7    CASSPGTAYYGYTF       False        0  1.0      0.0     1.0   1.0   
8    CASSPGTAYYGYTF       False        0  1.0      0.0     1.0   1.0   
9    CASSPGTAYYGYTF       False        0  1.0      0.0     1.0   1.0   

   significant         donor  
0        False     B2__CTCL2  
1        False     B2__CTCL5  
2        False     B2__CTCL6  
3        False  B4__SZ5_HTO1  
4        False  B4__SZ5_HTO2  
5        False  B4__SZ5_HTO3  
6        False  B4__SZ5_HTO4  
7        False  B4__SZ5_HTO5  
8        False  B4__SZ5_HTO6  
9        False  B4__SZ5_HTO1
```

## Step 5 — Use 3: reactive antigen-driven clusters in the CD8 infiltrate · HEAVY

ALICE on the CD8 T cells of the same donors — per donor and pooled per disease stage — to surface
convergent, antigen-driven reactive TIL clusters. Cached (donor-set guard).

```python  # cell 15
stage = obs[["donor", "disease_stage"]].astype(str).drop_duplicates().set_index("donor")["disease_stage"]
clono_cd8 = clono_cd8.assign(disease_stage=clono_cd8["donor"].map(stage).astype(str))
alice_cd8 = pd.read_parquet(ALICE_CD8_PARQUET) if ALICE_CD8_PARQUET.exists() else None
cached_donors = set(alice_cd8.loc[alice_cd8["scope"] == "donor", "donor"]) if alice_cd8 is not None else None
if alice_cd8 is None or cached_donors != set(clono_cd8["donor"]):
    if alice_cd8 is not None:
        print("CD8 cache stale (donor set changed) -> recompute")
    per_donor = A.run_alice_by_group(clono_cd8, pgen, group="donor", Q=Q, alpha=ALPHA)
    per_donor["scope"] = "donor"
    by_stage = A.run_alice_by_group(clono_cd8, pgen, group="disease_stage", Q=Q, alpha=ALPHA)
    by_stage["scope"] = "stage"
    alice_cd8 = pd.concat([per_donor, by_stage], ignore_index=True)
    alice_cd8.to_parquet(ALICE_CD8_PARQUET, index=False)
    print("wrote", ALICE_CD8_PARQUET)
else:
    print("loaded cached", ALICE_CD8_PARQUET)
use3 = alice_cd8[alice_cd8["significant"]].sort_values("obs_deg", ascending=False)
use3.to_csv(USE3_CSV, index=False)
print("CD8 donors:", alice_cd8.loc[alice_cd8['scope'] == 'donor', 'donor'].nunique(),
      "| reactive CD8 significant clonotypes:", len(use3))
use3
```

```
wrote /home/projects/nyosef/zvise/MF/data/atlas_joint/alice_cd8_blood_v2.parquet

CD8 donors: 93 | reactive CD8 significant clonotypes: 44

               donor               cdr3  n_cells  is_founder disease_stage  \
45761         H__HC1  CASSSLSGGATNEKLFF        1       False           nan   
24109         H__HC1  CASSSLSGGATNEKLFF        1       False           nan   
24110         H__HC1  CASSSLSLGATNEKLFF        3       False           nan   
24111         H__HC1  CASSSLTGGATNEKLFF        1       False           nan   
45763         H__HC1  CASSSLTGGATNEKLFF        1       False           nan   
45762         H__HC1  CASSSLSLGATNEKLFF        3       False           nan   
24112         H__HC1  CASSSLSSGATNEKLFF        2       False           nan   
24113         H__HC1  CASSSLSEGATNEKLFF        1       False           nan   
45764         H__HC1  CASSSLSSGATNEKLFF        2       False           nan   
45765         H__HC1  CASSSLSEGATNEKLFF        1       False           nan   
28407       B5__PT14    CSASRDSFTQETQYF        3       False            NA   
28410        B7__MF2    CSASTDSFTQETQYF        1       False            NA   
28408       B5__PT23   CASDREGDSSGNTIYF      129       False            NA   
24114         H__HC1  CASSSLTTGATNEKLFF        1       False           nan   
45766         H__HC1  CASSSLTTGATNEKLFF        1       False           nan   
28409        B7__MF2    CSASVDSFTQETQYF        2       False            NA   
5531        B5__PT16      CATSTTGGETQYF        1       False            NA   
5530        B5__PT16        CAWSSNTEAFF       18       False            NA   
5529        B5__PT16        CAWSPNTEAFF       22       False            NA   
5528        B5__PT16      CATSRTGGETQYF       35       False            NA   
450        B2__CTCL6      CASSLTVTGTQYF        1       False            NA   
449        B2__CTCL6      CASSLTSTGTQYF        1       False            NA   
1          B2__CTCL2   CASSSGQGFYNSPLHF        1       False            NA   
0          B2__CTCL2   CASSNGQGFYNSPLHF        1       False            NA   
2254   B4__SZ32_HTO6     CATSRGSDQETQYF        1       False           nan   
2253   B4__SZ32_HTO6     CATSRGSFQETQYF        1       False           nan   
1361   B4__SZ32_HTO2     CATSRVSEQETQYF        1       False           nan   
1360   B4__SZ32_HTO2     CATSRASEQETQYF        2       False           nan   
23896      D13__SZ29   CASSPTSGTYTDTQYF        1       False            NA   
23895      D13__SZ29   CASSQTSGTYTDTQYF       11       False            NA   
11038         B6__P1   CATSDPGTGTTGELFF       60       False           IVA   
11039         B6__P1   CATSDPGQGTTGELFF        7       False           IVA   
17236        B7__MF2    CSASVDSFTQETQYF        2       False            NA   
17237        B7__MF2    CSASTDSFTQETQYF        1       False            NA   
20455       B8__1135     CASSRGWSSYEQYF        5       False            NA   
20456       B8__1135     CASSPGWSSYEQYF        1       False            NA   
5695        B5__PT21    CASSPGTAYQETQYF        1       False            NA   
5694        B5__PT21    CASSGGTAYQETQYF        1       False            NA   
25205         H__SS2     CASSYYRGGEKLFF        1       False           nan   
25206         H__SS2     CASSLYRGGEKLFF        1       False           nan   
24669       H__MFIV1   CASSQGTSGVIYEQYF       57       False           nan   
24668       H__MFIV1   CASSPGTSGVIYEQYF       68       False           nan   
25203         H__SS2      CASSLFGADTQYF        5       False           nan   
25204         H__SS2      CASSLMGADTQYF        1       False           nan   

       obs_deg            Q   exp_deg        pois_p          q_bh  \
45761        5    27.100028  0.013044  3.112742e-12  9.918754e-09   
24109        5  1234.015462  0.056085  4.413505e-09  1.233575e-06   
24110        4  1234.015462  0.011219  6.541439e-10  3.656664e-07   
24111        4  1234.015462  0.167392  2.862407e-05  3.200171e-03   
4576
... [4014 more chars truncated]
```

```python  # cell 16
# CD8 coverage: every donor, with clonotype count + #significant hits.
cd8_cov = (alice_cd8[alice_cd8["scope"] == "donor"]
           .groupby("donor", observed=True)
           .agg(n_clonotypes=("cdr3", "size"),
                n_with_neighbor=("obs_deg", lambda s: int((s >= 1).sum())),
                n_significant=("significant", "sum"))
           .reset_index()
           .merge(clone_summary[["donor", "study", "disease", "disease_stage", "malignant"]],
                  on="donor", how="left")
           .sort_values("n_significant", ascending=False))
print("CD8 donors:", len(cd8_cov), "| with >=1 reactive hit:", int((cd8_cov["n_significant"] > 0).sum()))
cd8_cov
```

```
CD8 donors: 93 | with >=1 reactive hit: 13

        donor  n_clonotypes  n_with_neighbor  n_significant            study  \
85     H__HC1           559               28              6      herrera2021   
50   B5__PT16           103                4              4         buus2025   
88     H__SS2           194                4              4      herrera2021   
3   B2__CTCL6           243                4              2  borcherding2023   
0   B2__CTCL2           173                4              2  borcherding2023   
..        ...           ...              ...            ...              ...   
87     H__SS1           253                0              0      herrera2021   
89     H__SS3           116                0              0      herrera2021   
90     H__SS4           223                5              0      herrera2021   
91     H__SS5            54                0              0      herrera2021   
92     H__SS6           277                0              0      herrera2021   

   disease disease_stage  malignant  
85      HC                    False  
50      SS            NA       True  
88      MF                     True  
3       SS            NA       True  
0       SS            NA       True  
..     ...           ...        ...  
87      MF                     True  
89      MF                     True  
90      MF                     True  
91      MF                     True  
92      MF                     True  

[93 rows x 8 columns]
```

## Step 6 — ALICE figures

```python  # cell 18
plt.rcParams["figure.dpi"] = 120

d = use1.sort_values("family_frac", ascending=False)
fig, ax = plt.subplots(figsize=(5, 3))
ax.scatter(d["exact_frac"], d["family_frac"], s=18)
lim = [0, max(d["family_frac"].max(), d["exact_frac"].max()) * 1.05]
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xlabel("exact-match founder frac"); ax.set_ylabel("ALICE family frac")
ax.set_title("ALICE recovers beta-variant tumor cells (blood)")
fig.tight_layout(); fig.savefig(FIG_DIR / "alice_blood_v2_use1_family_vs_exact.png", bbox_inches="tight")

m = alice_cd4  # already carries is_founder (passed through alice_test)
fig, ax = plt.subplots(figsize=(5, 3))
ax.scatter(m["exp_deg"] + 1e-9, m["obs_deg"], s=6, alpha=.3, label="all")
f = m[m["is_founder"].fillna(False)]
ax.scatter(f["exp_deg"] + 1e-9, f["obs_deg"], s=20, color="tab:red", label="founder")
ax.set_xscale("log"); ax.set_xlabel("expected degree (null)"); ax.set_ylabel("observed degree")
ax.set_title("CD4 clonotype neighborhood enrichment (blood)"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG_DIR / "alice_blood_v2_use1_obs_vs_exp.png", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(use2["obs_deg"], bins=range(0, int(use2["obs_deg"].max()) + 2))
ax.set_xlabel("cross-patient neighbor degree"); ax.set_ylabel("founders")
ax.set_title(f"Cross-patient convergence: {int(use2['significant'].sum())} hits")
fig.tight_layout(); fig.savefig(FIG_DIR / "alice_blood_v2_use2_crosspatient_hist.png", bbox_inches="tight")
plt.show()
```

```
<Figure size 600x360 with 1 Axes>
<Figure size 600x360 with 1 Axes>
<Figure size 600x360 with 1 Axes>
```

## Step 7 — final ALICE malignant call → `tcr_malignant_alice`

Per donor: dominant CD4 clone + its ALICE-significant ≤1-aa neighbors = malignant. Donors with no
dominant clone stay benign. (No skin `PT35` special case — founders are derived generically.)

```python  # cell 20
mal_sets = {}
for donor, sub in clono_cd4.groupby("donor", observed=True):
    sig = set(alice_cd4.loc[(alice_cd4["donor"] == donor) & alice_cd4["significant"], "cdr3"])
    if sub["is_founder"].any():
        seeds = set(sub.loc[sub["is_founder"], "cdr3"])
    else:
        continue
    fam = A.founder_family(sub, seeds=seeds)
    mal_sets[donor] = seeds | (fam & sig)

# Map malignant CDR3 sets back to CD4 cells (TRB from the unified clone key, as in clonotype_table).
trb = (adata.obs["tcr_clone_id"].astype(str)
       .str.extract(r"^TRB:([A-Z]+)$", expand=False).fillna(""))
trb = trb.mask(trb == "", adata.obs["trb_cdr3"].astype(str))
donor = adata.obs["donor"].astype(str)
is_cd4 = adata.obs["cell_type_T2"].isin(CD4_T_TYPES).to_numpy()

mal = np.zeros(adata.n_obs, dtype=bool)
for dn, mset in mal_sets.items():
    mal |= is_cd4 & (donor == dn).to_numpy() & trb.isin(mset).to_numpy()
adata.obs["tcr_malignant_alice"] = mal

pd.DataFrame({"cell_id": adata.obs_names, "tcr_malignant_alice": mal}).to_parquet(ALICE_MAL_PARQUET, index=False)
print("malignant donors:", len(mal_sets), "| malignant cells:", int(mal.sum()))
print("wrote", ALICE_MAL_PARQUET)
```

```
malignant donors: 84 | malignant cells: 292553
wrote /home/projects/nyosef/zvise/MF/data/atlas_joint/alice_malignancy_blood_v2.parquet
```

### Step 3b — the per-sample clonality table (blood analogue of nb23 Step 4)

Skin has `tables/skin_tcr_clonality_per_sample.csv`; blood never had one, and
`32b_cluster_malignancy` cannot run without it. It is what tells the cluster model which samples
ALICE actually covers: `tcr_cohort` (did ALICE get a usable repertoire) and `is_clonal` (did it
find a founder). Samples outside the cohort become `apply` rather than training rows.

**The unit is the donor, not `sample_id`.** Blood clustering groups by donor because the geskin
`sample_id` is an HTO lane pooling 5-6 donors, so a cluster spanning one lane would span several
patients and its ALICE family share would mean nothing. The column is still named `sample_id`
so the shared model code reads it unchanged.


```python  # cell 22
# ============================================================
# Step 3b — per-sample (= per-donor) clonality, for 32b
# ============================================================
_cd4_tcr = (adata.obs["cell_type_T2"].isin(CD4_T_TYPES).to_numpy()
            & adata.obs["has_tcr"].astype(bool).to_numpy())
_don = adata.obs["donor"].astype(str)

clon = (pd.DataFrame({"donor": _don.to_numpy(),
                      "is_cd4": adata.obs["cell_type_T2"].isin(CD4_T_TYPES).to_numpy(),
                      "cd4_tcr": _cd4_tcr,
                      "malig": adata.obs["tcr_malignant_alice"].to_numpy()})
        .groupby("donor")
        .agg(n_cd4=("is_cd4", "sum"), n_cd4_tcr=("cd4_tcr", "sum"), n_clonal=("malig", "sum"))
        .reset_index())

# tcr_cohort: ALICE needs a repertoire it can calibrate an OLGA null on. Same gate as the cohort
# filter above (>= 300 TCR+ cells), so a donor ALICE never saw is never used as a training row.
clon["tcr_cohort"] = clon["n_cd4_tcr"] >= 300
clon["is_clonal"] = clon["donor"].isin(set(mal_sets)) & clon["tcr_cohort"]
clon["frac_clonal_of_cd4"] = clon["n_clonal"] / clon["n_cd4"].where(clon["n_cd4"] > 0)
clon["family_frac"] = clon["n_clonal"] / clon["n_cd4_tcr"].where(clon["n_cd4_tcr"] > 0)
clon["donor_is_clonal"] = clon["is_clonal"]

_meta = (adata.obs[["donor", *[c for c in ("study", "dataset", "disease", "disease_stage",
                                           "entity", "tissue", "sex", "tech")
                               if c in adata.obs.columns]]]
         .astype(str).groupby("donor", observed=True).first().reset_index())
clon = clon.merge(_meta, on="donor", how="left")

# The grouping unit is the donor; the column keeps the name the shared model code expects.
clon.insert(0, "sample_id", clon["donor"])
clon.to_csv(CLONALITY_CSV, index=False)

print(f"{len(clon)} donors | ALICE cohort {int(clon['tcr_cohort'].sum())} "
      f"| clonal {int(clon['is_clonal'].sum())} | apply-only "
      f"{int((~clon['tcr_cohort']).sum())}")
print("by disease:")
print(clon.groupby("disease")[["tcr_cohort", "is_clonal"]].sum().to_string())
print("\nwrote", CLONALITY_CSV)
clon.head()
```

```
93 donors | ALICE cohort 93 | clonal 84 | apply-only 0
by disease:
            tcr_cohort  is_clonal
disease                          
CTCL_other          16         15
HC                   4          0
MF                  11          8
SS                  62         61

wrote /home/projects/nyosef/zvise/MF/tables/blood_tcr_clonality_per_sample.csv

        sample_id           donor  n_cd4  n_cd4_tcr  n_clonal  tcr_cohort  \
0       B2__CTCL2       B2__CTCL2  30525      29967     29065        True   
1       B2__CTCL3       B2__CTCL3    815        800         0        True   
2       B2__CTCL5       B2__CTCL5    709        644       490        True   
3       B2__CTCL6       B2__CTCL6   1125       1080       518        True   
4  B4__SZ30_HTO10  B4__SZ30_HTO10    894        538       250        True   

   is_clonal  frac_clonal_of_cd4  family_frac  donor_is_clonal  \
0       True            0.952170     0.969900             True   
1      False            0.000000     0.000000            False   
2       True            0.691114     0.760870             True   
3       True            0.460444     0.479630             True   
4       True            0.279642     0.464684             True   

             study        dataset disease disease_stage  entity tissue  \
0  borcherding2023  borcherding23      SS            NA  Sezary   PBMC   
1  borcherding2023  borcherding23      SS            NA  Sezary   PBMC   
2  borcherding2023  borcherding23      SS            NA  Sezary   PBMC   
3  borcherding2023  borcherding23      SS            NA  Sezary   PBMC   
4       geskin2026       geskin26      SS           nan  Sezary   PBMC   

       sex    tech  
0  unknown  10x_5p  
1  unknown  10x_5p  
2  unknown  10x_5p  
3  unknown  10x_5p  
4  unknown  10x_5p
```

```python  # cell 23
pd.crosstab(adata.obs["tcr_malignant_alice"], adata.obs["cell_type_T2"], margins=True)
```

```
cell_type_T2            CD4    CD8   NK     All
tcr_malignant_alice                            
False                192238  72855  126  265219
True                 292553      0    0  292553
All                  484791  72855  126  557772
```

## Step 8 — carry ALICE as the TCR malignant truth

The ALICE tumor call becomes `tcr_is_malignant` **and** `tcr_is_dominant_clone` (so it also drives the
benign/diploid inferCNV reference). Herrera SS donors, if present, are relabeled `disease='SS'`.

```python  # cell 25
alice = adata.obs["tcr_malignant_alice"].astype(bool).to_numpy()
adata.obs["tcr_is_malignant"]      = alice
adata.obs["tcr_is_dominant_clone"] = alice

_ss = adata.obs["donor"].astype(str).isin([f"H__SS{i}" for i in range(1, 7)])
if _ss.any():
    adata.obs["disease"] = adata.obs["disease"].astype(str)
    adata.obs.loc[_ss, "disease"] = "SS"
    adata.obs["disease"] = adata.obs["disease"].astype("category")
    print("relabeled Herrera SS donors -> disease='SS':", int(_ss.sum()), "cells")
print("alice malignant:", int(alice.sum()), "/", adata.n_obs)
```

```
relabeled Herrera SS donors -> disease='SS': 13204 cells
alice malignant: 292553 / 557772
```

## Step 9 — inferCNV input prep: pooled diploid reference · HEAVY (GPU kernel)

**v3 diagnosis (why this was rebuilt).** Under v2 the held-out healthy control `H__HC1_Blood`
came back **61 %** malignant — worse than v1's 48 % — and on cells carrying both signals the CNV
call sat at TPR 0.66 / FPR 0.66, i.e. exactly chance. Three separate causes, measured on
`blood_T_malignancy_v2.parquet`:

1. **The score.** inferCNV ran on the 10k-HVG object: 7,463 positioned genes, `window=250`, so one
   window covered ~3 % of the genome, and HVGs are chosen for the cross-cell variance inferCNV
   assumes away. HC1's own run put 1,055 of *its own sample's* CD4 into the reference, and its
   remaining cells still scored `cnv_cell_score ≈ 0.0057` — mid-range among Sézary patients
   (0.0041–0.019). A cell scored against its own sample looked like a tumour, because
   `mean|X_cnv|` was measuring per-cell shot noise. v3 runs on the **full gene space**.
2. **The estimator.** With ≥2 `reference_cat` entries infercnvpy switches from mean subtraction to
   a *bounded* difference (anything inside `[min, max]` of the per-category means becomes logFC 0).
   v2's four heterogeneous categories made that dead-band wide — suppressing signal, not noise —
   and `dynamic_threshold=1.5·std(chunk)` made the noise floor depend on how each cell chunk
   happened to be composed. v3 passes one pooled reference vector and disables both.
3. **The rule.** See Step 13.

The four baseline categories, and what v3 does with each:

| `cnv_ref` | cells | bias | v3 |
|---|---|---|---|
| `nonclonal` | within-sample non-dominant CD4 | right batch + lineage, **tumour-contaminated** | moved to the query — in a high-burden sample it made the reference absorb the tumour's own CNV |
| `cd8_ref` | same-sample CD8 | right batch, wrong lineage | baseline; **half held out as the batch-matched null** |
| `hc_atlas` | healthy blood CD4 (`H__HC1_Blood`) | right tissue + lineage, **one donor** | baseline; half held out |
| `healthy_pbmc` | external healthy PBMC CD4 (OneK1K) | right lineage, ~300 donors, wrong batch (10x 3′) | baseline; half held out |

`nonclonal` / `cd8_ref` stay inside `acnv` per sample; `hc_atlas` / `healthy_pbmc` come back in
`SHARED_REF`, which Step 11 concatenates onto every sample block.

The atlas has exactly **one** healthy blood sample, so it cannot be both the reference and the
control: 2/3 of `H__HC1_Blood`'s CD4 goes to `hc_atlas` and the remaining third stays in the query
as a **held-out negative control**. Its called-malignant fraction is this notebook's acceptance
test — see Step 13.

> **Pooled-lane note:** `sample_id` is the 10x/HTO lane; in geskin 11 of 49 lanes pool >1 patient.
> inferCNV runs **per `sample_id`** (so `cd8_ref`/`nonclonal` stay batch-matched); malignancy is
> called **per `real_donor`** in Step 13.

```python  # cell 27
# One-off: subset OneK1K (1.25M cells, 981 healthy donors) to a ~12k-cell CD4 reference spread
# over ~300 donors. h5py row-slice read — never sc.read_h5ad the 4.4 GB source on the login node.
# The cache keeps OneK1K's full gene space; the intersection with the query happens below, so it
# does not need rebuilding for v3.
healthy_pbmc = C.build_healthy_pbmc_ref(ONEK1K_H5, HEALTHY_PBMC_REF,
                                        n_donors=300, n_per_donor=40, seed=SEED)

print("lineage:\n", adata.obs["lineage"].value_counts())
# v3 runs on the full gene space, so the positioned-gene count should land ~20-25k (v2's HVG
# object gave 7,463). Keep a floor well above that to catch a symbol/GTF mismatch.
acnv, SHARED_REF, CNV_DONORS, HC_CONTROL = C.prepare_blood_pooled_cnv_inputs(
    adata, GTF, healthy_pbmc, SEED, kept_lineages=KEPT_LINEAGES,
    hc_ref_frac=2 / 3, min_cd8_ref=20, min_positioned=12000)
del healthy_pbmc
gc.collect()
print("CNV samples:", len(CNV_DONORS), "| acnv:", acnv.shape,
      "| shared ref:", SHARED_REF.shape, "| HC held-out control cells:", len(HC_CONTROL))
print("detection depth (n_genes_cnv) — query vs shared reference:")
print(pd.concat([acnv.obs.groupby("cnv_ref", observed=True)["n_genes_cnv"].describe(),
                 SHARED_REF.obs.groupby("cnv_ref", observed=True)["n_genes_cnv"].describe()])
      [["count", "25%", "50%", "75%"]].round(0).to_string())
```

```
loaded cached healthy-PBMC ref (healthy_pbmc_cd4_ref_v1.h5ad): 12000 cells, 300 donors
lineage:
 lineage
unresolved_provisional    353904
CD4                       200974
not_applicable              2894
Name: count, dtype: int64

healthy blood CD4: 1024 -> hc_atlas reference | 512 -> held-out control (stays in query)
external healthy PBMC reference: 12000 cells
usable samples (CD8 >= 20): 39 | dropped: []

INFO:root:Extracted GTF attributes: ['gene_id', 'gene_version', 'gene_name', 'gene_source', 'gene_biotype', 'transcript_id', 'transcript_version', 'transcript_name', 'transcript_source', 'transcript_biotype', 'tag', 'transcript_support_level', 'exon_number', 'exon_id', 'exon_version', 'protein_id', 'protein_version', 'ccds_id']

WARNING: GTF file misses annotation for 3 genes in adata.

WARNING: Skipped 45 genes because of duplicate identifiers in GTF file.

cnv_ref: {'query': np.int64(146165), 'cd8_ref': np.int64(33443), 'nonclonal': np.int64(23117)} | shared ref: 13024 {'healthy_pbmc': np.int64(12000), 'hc_atlas': np.int64(1024)} | genes on chr1-22/X/Y: 18757

CNV samples: 39 | acnv: (202725, 18757) | shared ref: (13024, 18757) | HC held-out control cells: 512
detection depth (n_genes_cnv) — query vs shared reference:
                 count     25%     50%     75%
cnv_ref                                       
cd8_ref        33443.0  1082.0  1435.0  1931.0
nonclonal      23117.0  1205.0  1564.0  1972.0
query         146165.0   964.0  1565.0  2451.0
hc_atlas        1024.0  1146.0  1287.0  1468.0
healthy_pbmc   12000.0   640.0   829.0  1068.0
```

## Step 10 — inferCNV knobs (parameters only, no computation)

`WINDOW_SIZE`, `REF_MODE`, `DYN_THRESH`, `SHUFFLE`, `NULL_FRAC` and `NULL_CATS` are consumed by
Step 11 **and** Step 12 — they must match, or the two passes hold out different null cells.
`CNV_LEIDEN_RES` / `TOPK_FRAC` are Step 11 only. `REF_CATS` orders the baseline categories; a
category is used for a sample only if that sample's block carries ≥20 of its cells after the null
holdout.

v2's `CLUSTER_THR_SCALE = 0.8` is gone — it multiplied an already self-referential threshold by
0.8 to call *more* cells.

```python  # cell 29
# >>> KNOBS <<<
WINDOW_SIZE    = 100     # infercnvpy default; with ~20k positioned genes this is FINER than
                         # v2's 250 over 7.5k genes (which spanned ~3% of the genome per window)
TOPK_FRAC      = 0.10    # focal fraction of windows per cell
CNV_LEIDEN_RES = 0.5     # CNV-cluster Leiden resolution (cache tagged by this)

# ---- v3 estimator: see the four knobs documented on C.run_per_donor_infercnv ----
REF_MODE   = "mean"      # one pooled reference vector, NOT infercnvpy's multi-category dead-band
DYN_THRESH = None        # no 1.5*std(chunk) zeroing — its threshold depended on chunk composition
SHUFFLE    = True        # permute before chunking so every chunk has the same query/ref mix
NULL_FRAC  = 0.5         # half the baseline is held out as the diploid null (see Step 13)
# nonclonal is NOT a baseline in v3: it is within-sample non-dominant CD4, so in a high-burden
# sample it carries tumour and the reference absorbs the tumour's own CNV (SZ5: 97% TCR-malignant,
# 9-15% CNV-called under v2). Those cells stay in the query and get called like any other.
REF_CATS  = ("hc_atlas", "cd8_ref", "healthy_pbmc")
NULL_CATS = ("cd8_ref",)   # batch-matched half of the null; the shared ref is the other half

# ---- v3.1 calling knobs (Step 13 only — no inferCNV recompute) ----
# v3.0 passed its specificity gate (HC1 1.5%, precision 0.93) but recalled only 45% of the TCR
# call, median per-donor AUROC 0.656. Two causes, both in the anchor step, both fixed here.
MIN_CLUSTER_FRAC = 0.05  # an anchor cluster must hold >=5% of the donor's CD4, not just 30 cells.
                         # Every v3.0 anchor below 2% (PT07 n=44, PT09 n=36, PT16 n=54) ranked at
                         # AUROC <=0.66: a signature fitted to 40 cells is fitted to their noise.
MAX_EVENT_ARMS   = 15    # now truncates the signature to the 15 strongest arms by |z| instead of
GLOBAL_ARM_FRAC  = 0.7   # rejecting the cluster. Sezary karyotypes really do carry 13-15 arms, and
                         # v3.0 threw those clones out (PT02/SS3 sat exactly at the cap) leaving a
                         # tiny residual cluster as the anchor. Only a cluster shifted on >70% of
                         # all arms is a genuine global offset.
CNV_NULL_Q = 0.98        # quantile of the diploid null that a cell must clear. Swept on the cached
                         # v3 arm matrix: 0.99 -> recall 0.455 / precision 0.937 / null FPR 0.5%,
                         # 0.98 -> 0.523 / 0.933 / 1.0%, 0.95 -> 0.658 / 0.926 / 2.4% but the HC1
                         # control goes to 6.3% and breaks the <=5% gate. 0.98 is the last step
                         # that buys recall for free and it puts the null FPR on its 1% target.

FINAL_CNV_COL = "cnv_arm_malignant"   # the CNV call Step 14 combines with the TCR call.
                                      # "cnv_malignant_union" folds in the focal-cluster channel;
                                      # Step 15 prints the gate for all three either way.
CNV_N_JOBS = 8
CNV_CHUNK  = 1000        # smaller than v2's 2500: 20k genes densify to ~160 MB per chunk
FORCE_CNV  = False       # set True to recompute
CNV_CACHE  = OUT_DIR / f"blood_T_pooledref_cnv_per_cell_v4_res{CNV_LEIDEN_RES:g}.parquet"
```

## Step 11 — per-sample inferCNV vs the pooled reference · HEAVY (long-running)

`C.run_per_donor_infercnv` concatenates `SHARED_REF` onto each sample block, runs
`cnv.tl.infercnv`, then PCA / neighbors / Leiden on **query + reference together**, and fills
`cnv_score` (per `cnv_leiden` cluster), `cnv_cell_score`, `cnv_focal_score` (mean of the top
`TOPK_FRAC` windows), `cnv_leiden`. Cached to `CNV_CACHE`.

**What v3 changes here** (all four are knobs in Step 10, defaults unchanged for `old/22_final_figures`/`23_malignancy_tcr_cnv`):

- `ref_mode="mean"` — one pooled reference vector, so the estimator is plain mean subtraction and
  does not silently become a dead-band as reference categories are added.
- `dynamic_threshold=None` + `shuffle=True` — v2's noise threshold was `1.5 × std(cell chunk)`
  computed per chunk **in obs order**, so a donor with few query cells had them pooled into a
  reference-dominated chunk (low std, more noise survives) while a large donor got tumour-only
  chunks. Scores were not comparable across donors; the dry run confirms they are now identical
  under different `chunksize`.
- `null_frac=0.5` — half the baseline is relabelled and scored as query. Those cells are known
  diploid and traverse the identical pipeline, so their score distribution is the empirical null
  Step 13 thresholds against. Half comes from the shared reference (lineage-matched, wrong batch)
  and half from same-sample CD8 (batch-matched, wrong lineage), so the noisier of the two sets the
  tail. v2 discarded every reference score, which is why no absolute cut was possible.

Returns the full per-cell frame including the `ref_null` rows — Step 13 needs them.

> ~4× the v2 work (4× the genes). Submit with `jobs/run_blood_cnv.sh`, not interactively.

```python  # cell 31
PER_CELL = C.run_per_donor_infercnv(
    acnv, SHARED_REF, CNV_DONORS, CNV_CACHE, window=WINDOW_SIZE, topk_frac=TOPK_FRAC,
    leiden_res=CNV_LEIDEN_RES, ref_cats=REF_CATS, ref_mode=REF_MODE,
    dynamic_threshold=DYN_THRESH, shuffle=SHUFFLE, null_frac=NULL_FRAC, null_cats=NULL_CATS,
    n_jobs=CNV_N_JOBS, chunk=CNV_CHUNK, seed=SEED, force=FORCE_CNV)

print("\nper-cell rows:", PER_CELL.shape)
print(PER_CELL["cnv_ref"].value_counts().to_string())
print("\nnull cells per sample (need >= 50 for a usable threshold):")
print(PER_CELL[PER_CELL.cnv_ref.eq("ref_null")].groupby("donor", observed=True)
      .size().describe()[["min", "25%", "50%", "max"]].round(0).to_string())
```

```
[B2__CTCL2] query= 29623  nonclonal=  902  null= 6904  ref=['hc_atlas', 'cd8_ref', 'healthy_pbmc']

  0%|          | 0/45 [00:00<?, ?it/s]
  2%|▏         | 1/45 [00:00<00:36,  1.20it/s]
 13%|█▎        | 6/45 [00:01<00:05,  7.55it/s]
 20%|██        | 9/45 [00:01<00:05,  6.49it/s]
 27%|██▋       | 12/45 [00:01<00:03,  8.91it/s]
 33%|███▎      | 15/45 [00:01<00:02, 11.32it/s]
 38%|███▊      | 17/45 [00:02<00:03,  7.95it/s]
 42%|████▏     | 19/45 [00:02<00:02,  9.44it/s]
 49%|████▉     | 22/45 [00:02<00:01, 12.06it/s]
 56%|█████▌    | 25/45 [00:03<00:02,  8.87it/s]
 60%|██████    | 27/45 [00:03<00:01,  9.81it/s]
 69%|██████▉   | 31/45 [00:03<00:01, 13.69it/s]
 73%|███████▎  | 33/45 [00:03<00:01,  9.16it/s]
 78%|███████▊  | 35/45 [00:03<00:00, 10.46it/s]
 84%|████████▍ | 38/45 [00:03<00:00, 13.00it/s]
 91%|█████████ | 41/45 [00:04<00:00,  9.66it/s]
 98%|█████████▊| 44/45 [00:04<00:00, 11.87it/s]
100%|██████████| 45/45 [00:04<00:00,  9.79it/s]


/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/infercnvpy/tl/__init__.py:24: FutureWarning: In the future, the default backend for leiden will be igraph instead of leidenalg.

 To achieve the future defaults please pass: flavor="igraph" and n_iterations=2.  directed must also be False to work with igraph's implementation.
  return sc.tl.leiden(

[B2__CTCL3] query=    15  nonclonal=  800  null= 7017  ref=['hc_atlas', 'cd8_ref', 'healthy_pbmc']

  0%|          | 0/16 [00:00<?, ?it/s]
  6%|▋         | 1/16 [00:00<00:11,  1.32it/s]
 25%|██▌       | 4/16 [00:00<00:02,  5.74it/s]
 50%|█████     | 8/16 [00:00<00:00, 11.85it/s]
 69%|██████▉   | 11/16 [00:01<00:00,  7.94it/s]
100%|██████████| 16/16 [00:01<00:00,  9.86it/s]


[B2__CTCL5] query=   555  nonclonal=  154  null= 6640  ref=['hc_atlas', 'cd8_ref', 'healthy_pbmc']

  0%|          | 0/15 [00:00<?, ?it/s]
  7%|▋         | 1/15 [00:00<00:10,  1.28it/s]
 27%|██▋       | 4/15 [00:00<00:02,  5.50it/s]
 47%|████▋     | 7/15 [00:01<00:00,  9.02it/s]
 60%|██████    | 9/15 [00:01<00:00,  6.75it/s]
 87%|████████▋ | 13/15 [00:01<00:00, 10.81it/s]
100%|██████████| 15/15 [00:01<00:00,  8.84it/s]


[B2__CTCL6] query=   563  nonclonal=  562  null= 6859  ref=['hc_atlas', 'cd8_ref', 'healthy_pbmc']

  0%|          | 0/15 [00:00<?, ?it/s]
  7%|▋         | 1/15 [00:00<00:10,  1.36it/s]
 27%|██▋       | 4/15 [00:00<00:01,  5.85it/s]
 47%|████▋     | 7/15 [00:00<00:00, 10.29it/s]
 67%|██████▋   | 10/15 [00:01<00:00,  7.67it/s]
 87%|████████▋ | 13/15 [00:01<00:00, 10.54it/s]
100%|██████████| 15/15 [00:01<00:00,  9.26it/s]


[B4__SZ30] query=  4231  nonclonal= 1632  null= 8486  ref=['hc_atlas', 'cd8_ref', 'healthy_pbmc']

  0%|          | 0/23 [00:00<?, ?it/s]
  4%|▍         | 1/23 [00:00<00:17,  1.29it/s]
 17%|█▋        | 4/23 [00:00<00:03,  5.66it/s]
 26%|██▌       | 6/23 [00:00<00:02,  8.13it/s]
 39%|███▉      | 9/23 [00:01<00:02,  6.55it/s]
 52%|█████▏    | 12/23 [00:01<00:01,  9.16it/s]
 65%|██████▌   | 15/23 [00:01<00:00, 11.59it/s]
 74%|███████▍  | 17/23 [00:02<00:00,  8.01it/s]
 87%|████████▋ | 20/23 [00:02<00:00, 10.01it/s]
100%|██████████| 23/23 [00:02<00:00,  9.11it/s]


[B4__SZ32] query=  2280  nonclonal= 4112  null= 8510  ref=['hc_atlas', 'cd8_ref', 'healthy_pbmc']

  0%|          | 0/24 [00:00<?, ?it/s]
  4%|▍         | 1/24 [00:00<00:17,  1.29it/s]
 17%|█▋        | 4/24 [00:00<00:03,  5.70it/s]
 33%|███▎      | 8/24 [00:01<00:01, 11.28it/s]
 46%|████▌     | 11/24 [00:01<00:01,  8.05it/s]
 54%|█████▍    | 13/24 [00:01<00:01,  9.08it/s]
 71%|███████   | 17/24 [00:02<00:00,  8.14it/s]
 88%|████████▊ | 21/24 [00:02<00:00, 10.94it/s]
100%|██████████| 24/24 [00:02<00:00,  9.81it/s]


[B4__SZ33] query=  6001  nonclonal=  226  null= 6637  ref=['hc_atlas', 'cd8_ref', 'healthy_pbmc']

  0%|          | 0/20 [00:00<?, ?it/s]
  5%|▌         | 1/20 [00:00<00:13,  1.37it/s]
 20%|██        | 4/20 [00:00<00:02,  5.98it/s]
 35%|███▌      | 7/20 [00:00<00:01, 10
... [17239 more chars truncated]
```

## Step 12 — per-cell arm-level CNV · HEAVY (GPU kernel)

Second inferCNV pass with `calculate_gene_values=True`, averaging per-gene CNV within each
chromosome arm into a compact `cells × ~41 arm` matrix. In v2 this ran last and fed only `33_subclone_tcr_signaling`;
in v3 it runs **before** the call, because the caller is built on it.

Same estimator and the same `seed` as Step 11, so `_split_null` holds out exactly the same cells —
that is what lets the arm null describe the same diploid cells as the score null. `keep_cats` now
retains `nonclonal` (a query category in v3) and `ref_null` (the null) alongside `query`.

Checkpoints per sample in `<cache>_parts/`, so a resubmitted job resumes.

```python  # cell 33
import os

FORCE_ARM  = False
ARM_CHUNK  = 100
N_ARM_JOBS = len(os.sched_getaffinity(0))   # respects the LSF cpuset
print("arm CNV n_jobs =", N_ARM_JOBS)

# Same estimator and the same seed as Step 11, so `_split_null` holds out exactly the same cells
# in both passes — that is what lets the arm null and the score null describe the same cells.
arm = C.compute_arm_cnv_per_cell(
    acnv, CNV_DONORS, ARM_CACHE, shared_ref=SHARED_REF, reference_cat=REF_CATS,
    window=WINDOW_SIZE, keep_cats=("query", "nonclonal", "ref_null"), ref_mode=REF_MODE,
    dynamic_threshold=DYN_THRESH, shuffle=SHUFFLE, null_frac=NULL_FRAC, null_cats=NULL_CATS,
    n_jobs=N_ARM_JOBS, chunk=ARM_CHUNK, seed=SEED, force=FORCE_ARM)
ARM_COLS = [c for c in arm.columns if c != "donor"]
print("wrote", ARM_CACHE, "| arm matrix:", arm[ARM_COLS].shape, "| arms:", ARM_COLS)

_null_arm = set(arm.index[arm.index.astype(str).str.contains(r"\|NULL\|")])
_null_pc = set(PER_CELL.index[PER_CELL.cnv_ref.eq("ref_null")])
print(f"shared-reference null cells — arm pass {len(_null_arm)}, score pass "
      f"{len(_null_pc & set(arm.index))}: {_null_arm <= _null_pc}")
assert _null_arm <= _null_pc, "the two inferCNV passes held out different reference cells"
```

```
arm CNV n_jobs = 16

  0%|          | 0/445 [00:00<?, ?it/s]
  0%|          | 1/445 [00:14<1:45:27, 14.25s/it]
  1%|          | 5/445 [00:14<15:44,  2.15s/it]  
  2%|▏         | 9/445 [00:14<07:13,  1.00it/s]
  2%|▏         | 9/445 [00:26<07:13,  1.00it/s]
  4%|▍         | 17/445 [00:28<10:13,  1.43s/it]
  4%|▍         | 18/445 [00:28<09:16,  1.30s/it]
  4%|▍         | 20/445 [00:28<07:18,  1.03s/it]
  5%|▍         | 22/445 [00:28<05:38,  1.25it/s]
  5%|▌         | 24/445 [00:28<04:18,  1.63it/s]
  6%|▌         | 27/445 [00:28<02:54,  2.40it/s]
  7%|▋         | 33/445 [00:42<09:08,  1.33s/it]
  8%|▊         | 34/445 [00:42<08:17,  1.21s/it]
  8%|▊         | 36/445 [00:43<06:45,  1.01it/s]
 10%|▉         | 43/445 [00:43<03:07,  2.14it/s]
 11%|█         | 48/445 [00:43<02:06,  3.14it/s]
 11%|█         | 48/445 [00:56<02:06,  3.14it/s]
 11%|█         | 49/445 [00:56<09:56,  1.51s/it]
 12%|█▏        | 52/445 [00:56<07:12,  1.10s/it]
 12%|█▏        | 55/445 [00:57<05:15,  1.23it/s]
 13%|█▎        | 57/445 [00:57<04:13,  1.53it/s]
 14%|█▎        | 61/445 [00:57<02:40,  2.39it/s]
 14%|█▍        | 64/445 [00:57<02:01,  3.13it/s]
 15%|█▍        | 66/445 [01:11<10:43,  1.70s/it]
 15%|█▌        | 68/445 [01:11<08:14,  1.31s/it]
 16%|█▌        | 70/445 [01:11<06:25,  1.03s/it]
 16%|█▋        | 73/445 [01:11<04:17,  1.45it/s]
 17%|█▋        | 75/445 [01:11<03:17,  1.88it/s]
 18%|█▊        | 78/445 [01:12<02:12,  2.76it/s]
 18%|█▊        | 81/445 [01:25<10:14,  1.69s/it]
 18%|█▊        | 82/445 [01:25<08:58,  1.48s/it]
 19%|█▉        | 85/445 [01:25<05:52,  1.02it/s]
 20%|█▉        | 87/445 [01:26<05:12,  1.15it/s]
 21%|██        | 94/445 [01:27<02:25,  2.42it/s]
 22%|██▏       | 97/445 [01:39<07:45,  1.34s/it]
 22%|██▏       | 98/445 [01:39<07:00,  1.21s/it]
 22%|██▏       | 99/445 [01:39<06:08,  1.06s/it]
 23%|██▎       | 101/445 [01:39<04:36,  1.24it/s]
 23%|██▎       | 102/445 [01:40<04:02,  1.42it/s]
 23%|██▎       | 104/445 [01:40<02:53,  1.97it/s]
 24%|██▎       | 105/445 [01:40<02:27,  2.30it/s]
 24%|██▍       | 107/445 [01:40<01:46,  3.17it/s]
 25%|██▍       | 110/445 [01:40<01:10,  4.78it/s]
 25%|██▌       | 112/445 [01:41<01:19,  4.17it/s]
 25%|██▌       | 113/445 [01:53<12:18,  2.22s/it]
 26%|██▌       | 114/445 [01:53<10:15,  1.86s/it]
 26%|██▋       | 117/445 [01:53<05:42,  1.04s/it]
 27%|██▋       | 118/445 [01:54<04:54,  1.11it/s]
 27%|██▋       | 119/445 [01:54<04:02,  1.35it/s]
 27%|██▋       | 121/445 [01:54<02:41,  2.01it/s]
 27%|██▋       | 122/445 [01:54<02:18,  2.34it/s]
 28%|██▊       | 123/445 [01:54<01:54,  2.82it/s]
 28%|██▊       | 125/445 [01:54<01:15,  4.25it/s]
 29%|██▊       | 127/445 [01:54<00:56,  5.61it/s]
 29%|██▊       | 127/445 [02:06<00:56,  5.61it/s]
 29%|██▉       | 129/445 [02:07<11:49,  2.24s/it]
 29%|██▉       | 130/445 [02:07<09:54,  1.89s/it]
 29%|██▉       | 131/445 [02:07<07:50,  1.50s/it]
 30%|██▉       | 133/445 [02:07<04:54,  1.06it/s]
 30%|███       | 135/445 [02:08<03:31,  1.47it/s]
 31%|███       | 137/445 [02:08<02:27,  2.09it/s]
 31%|███       | 139/445 [02:08<01:57,  2.61it/s]
 32%|███▏      | 141/445 [02:08<01:24,  3.58it/s]
 32%|███▏      | 143/445 [02:10<02:19,  2.16it/s]
 32%|███▏      | 144/445 [02:10<02:08,  2.34it/s]
 33%|███▎      | 145/445 [02:21<12:27,  2.49s/it]
 33%|███▎      | 146/445 [02:21<10:07,  2.03s/it]
 33%|███▎      | 147/445 [02:22<07:47,  1.57s/it]
 34%|███▎      | 150/445 [02:22<04:04,  1.21it/s]
 34%|███▍      | 152/445 [02:22<02:55,  1.67it/s]
 35%|███▍      | 154/445 [02:22<02:08,  2.26it/s]
 35%|███▍      | 155/445 [02:23<01:54,  2.53it/s]
 35%|███▌      | 157/445 [02:23<01:21,  3.51it/s]
 36%|███▌      | 158/445 [02:23<01:15,  3.79it/s]
 36%|███▌      | 159/445 [02:24<02:31,  1.89it/s]
 36%|███▌      | 160/445 [02:25<02:10,  2.19it/s]
 36%|███▌      | 161/445 [02:35<14:18,  3.02s/it]
 36%|███▋      | 162/445 [02:36<11:06,  2.36s/it]
 37%|███▋      | 163/445 [02:36<08:13,  1.75s/
... [180334 more chars truncated]
```

## Step 13 — malignancy call: null-anchored, de novo signed arm consensus

**What was wrong with v2's caller.** `call_per_donor` set
`thr = 0.8 × max(GMM crossover on the donor's own scores, median of the donor's own diploid cells)`.
Both terms are internal to the donor, and the "floor" was the *median* of the diploid
distribution, so by construction ~half the diploid cells exceed it; `CLUSTER_THR_SCALE = 0.8`
lowered it another 20 %. A tumour-free donor is therefore cut at roughly its own median and comes
back ~50–60 % malignant no matter what the reference does — HC1's 18 CNV clusters spanned
0.0007–0.0070 as a smooth continuum with no gap, and the threshold landed at 0.0050. The 2-component
GMM never had anything to reject: its only guard, `exp(μ_hi−μ_lo) > 1.1`, is passed trivially by
unimodal log-normal data. And it thresholded `cnv_score`, the per-cluster mean — the weakest of the
three available scores (per-donor AUROC 0.689, vs 0.720 focal and 0.728 per-cell).

**13a — score-level null (diagnostic).** `C.call_per_donor_null` thresholds `cnv_focal_score` at
the 0.99 quantile of the donor's own held-out diploid cells, conditioned on detection depth
(query cells are binned on their own depth quantiles and judged against null cells in the same
bin; bins the null does not cover are reported, not extrapolated). This is *not* the call —
held-out reference cells helped build the baseline they are scored against, so this null is
anti-conservative wherever query and reference differ by batch. It is here to expose that gap.

**13b — the call.** `C.arm_consensus_score`. `mean|X_cnv|` is a magnitude, so scattered dropout
noise scores like a clean chr7q gain — in v2 the healthy control's false positives carried arm
amplitudes (mean |arm| 0.0029) indistinguishable from real Sézary cells (0.0035). A real event is
**signed**, **contiguous** and **shared across a clone**, so each donor's CNV clusters are tested
arm-by-arm against the held-out null: a cluster carries an event where its mean is ≥5 standard
errors from the null *and* shifted by ≥0.01 absolute; a cluster with >15 event arms is a global
offset, not a karyotype, and is rejected. The largest surviving cluster defines the donor's signed
signature, and every cell scores as its projection onto it, cut at the null projection's 0.99
quantile.

No TCR anywhere in that path, so the CNV call remains **independent evidence** and can still be
used to corroborate the TCR call. A donor with no cluster clearing the null gets no signature and
is marked **`indeterminate`** rather than silently negative — Step 14 then falls back to TCR alone
for it.

**Acceptance test.** The printout ends with the held-out `H__HC1_Blood` control: **≤5 %** is the
gate (v2: 61 %, v1: 48 %). Expect `indeterminate` — a tumour-free donor should produce no
signature at all.

```python  # cell 35
# ---- 13a. score-level diagnostic: how far the query sits above its own diploid null ----
# Not the call. Reference cells built the baseline they are scored against, so this null is
# anti-conservative wherever query and reference differ by batch/chemistry; it is here to show
# that gap, and to flag donors whose cells fall outside the null's depth range.
CD4 = acnv.obs["cnv_ref"].isin(["query", "nonclonal"]).to_numpy()
obs_pt = acnv.obs[CD4].copy()
obs_pt["cnv_run"] = obs_pt["donor"].astype(str)        # sample_id = the inferCNV unit
obs_pt["donor"] = obs_pt["real_donor"].astype(str)     # call per PATIENT
PT_DONORS = sorted(obs_pt["donor"].unique())

# PER_CELL['donor'] is the sample_id for query and null rows alike. Query rows lift to their own
# patient; a null row cannot, because a hashtag-pooled sample (B4__SZ5 -> SZ5_HTO1..6) carries one
# null shared by six patients. v3.0 mapped sample -> patient with .first(), which handed the whole
# null to whichever patient came first and left the other five at "null too small" — 27 of the 32
# indeterminate samples. Null rows now stay keyed to the run and `run_col` matches on that.
pc_pt = PER_CELL.copy()
pc_pt["cnv_run"] = pc_pt["donor"].astype(str)
_pt_of_cell = acnv.obs["real_donor"].astype(str).reindex(pc_pt.index)
_pc_is_null = pc_pt["cnv_ref"].astype(str).eq("ref_null").to_numpy()
pc_pt["donor"] = np.where(_pc_is_null | _pt_of_cell.isna().to_numpy(),
                          pc_pt["cnv_run"], _pt_of_cell.to_numpy())

score_call, score_thr, score_diag = C.call_per_donor_null(
    pc_pt, PT_DONORS, obs_pt, score_col="cnv_focal_score", q=CNV_NULL_Q,
    depth_col="n_genes_cnv", cluster_col="cnv_leiden", run_col="cnv_run")

# ---- 13b. the call: de novo signed arm consensus ----
arm_proj, arm_call, arm_state, SIG = C.arm_consensus_score(
    arm, pc_pt, PT_DONORS, arm_cols=ARM_COLS, cluster_col="cnv_leiden",
    depth_col="n_genes_cnv", q=CNV_NULL_Q, run_col="cnv_run",
    min_cluster_frac=MIN_CLUSTER_FRAC, max_event_arms=MAX_EVENT_ARMS,
    global_arm_frac=GLOBAL_ARM_FRAC)

_arm = pd.Series(arm_proj, index=arm.index).reindex(acnv.obs_names)
acnv.obs["cnv_arm_proj"] = _arm.to_numpy(dtype=float)
acnv.obs["cnv_arm_malignant"] = (pd.Series(arm_call, index=arm.index)
                                 .reindex(acnv.obs_names).eq(True).to_numpy() & CD4)
acnv.obs["cnv_state"] = (pd.Series(arm_state, index=arm.index).reindex(acnv.obs_names)
                         .fillna("indeterminate").astype(str))
acnv.obs["cnv_score_above_null"] = (pd.Series(score_call, index=obs_pt.index)
                                    .reindex(acnv.obs_names).eq(True).to_numpy())

# ---- 13c. second channel: CNV clusters that clear their own diploid null on focal score ----
# `cnv_focal_score` out-ranks the arm projection wherever the arm signature comes out weak (v3.0:
# CTCL6 arm 0.52 / focal 0.99, PT21 0.62 / 0.97, SS4 0.78 / 0.98, all six SZ32 ~0.72 / ~0.82), so
# the focal null test is kept as a *cluster-level* call — a whole CNV cluster sitting above its
# depth-matched null is evidence that one cell's tail is not, and per-cell `cnv_score_above_null`
# on its own sprays false positives. Keyed by (patient, cluster): a pooled sample's Leiden labels
# are shared across its patients, so the cluster name alone would leak the call between them.
_enr = set()
for r in score_diag.itertuples():
    s = getattr(r, "enriched_clusters", "")
    if isinstance(s, str) and s:
        _enr |= {(str(r.donor), c) for c in s.split(",")}
_in_enr = np.fromiter(
    ((p, c) in _enr for p, c in zip(acnv.obs["real_donor"].astype(str),
                                    acnv.obs["cnv_leiden"].astype(str))),
    dtype=bool, count=acnv.n_obs)
acnv.obs["cnv_focal_cluster_malignant"] = (
    _in_enr & acnv.obs["cnv_score_above_null"].to_numpy() & CD4)
acnv.obs["cnv_malignant_union"] = (acnv.obs["cnv_arm_malignant"].to_numpy()
                                   | acnv.obs["cnv_focal_cluster_malignant"].to_numpy())
print(f"\nfocal-enriched clusters: {len(_enr)} across "
      f"{int(score_diag['n_clusters_enriched'].gt(0).sum())} patients")

# ---- 13d. callability, per channel ----
# Two different absences of evidence: no diploid null at all (nothing can be said), vs a null but
# no cluster clearing it (the arm signature abstains, the focal channel may still call).
ARM_CALLABLE = set(SIG.loc[SIG["callable"], "donor"].astype(str))
NULL_CALLABLE = set(score_diag.loc[score_diag["thr_median"].notna(), "donor"].astype(str))
print(f"\ncallable patients: arm signature {len(ARM_CALLABLE)}/{len(PT_DONORS)}"
      f" | usable diploid null {len(NULL_CALLABLE)}/{len(PT_DONORS)}   [v3.0: 31/63 both]"
      f"\n  null but no arm signature: {sorted(NULL_CALLABLE - ARM_CALLABLE)}"
      f"\n  no usable null:            {sorted(set(PT_DONORS) - NULL_CALLABLE)}")

METHODS = {"arm_consensus": "cnv_arm_malignant",
           "focal_cluster": "cnv_focal_cluster_malignant",
           "union": "cnv_malignant_union"}
CALLABLE = ARM_CALLABLE if FINAL_CNV_COL == "cnv_arm_malignant" else NULL_CALLABLE
for col in ["cnv_score", "cnv_cell_score", "cnv_focal_score", "cnv_arm_proj"]:
    adata.obs[col] = acnv.obs[col].where(CD4).reindex(adata.obs_names)
adata.obs["cnv_leiden"] = acnv.obs["cnv_leiden"].reindex(adata.obs_names).fillna("")
adata.obs["cnv_state"] = (acnv.obs["cnv_state"].reindex(adata.obs_names)
                          .fillna("indeterminate").astype(str))
for col in ["cnv_arm_malignant", "cnv_score_above_null",
            "cnv_focal_cluster_malignant", "cnv_malignant_union"]:
    adata.obs[col] = acnv.obs[col].reindex(adata.obs_names).eq(True).to_numpy()
_pt = adata.obs["real_donor"].astype(str)
adata.obs["cnv_arm_callable"] = _pt.isin(ARM_CALLABLE).to_numpy()
adata.obs["cnv_null_callable"] = _pt.isin(NULL_CALLABLE).to_numpy()
adata.obs["cnv_callable"] = _pt.isin(CALLABLE).to_numpy()   # tracks FINAL_CNV_COL
print()
for nm, col in METHODS.items():
    print(f"{nm:>14}: {int(adata.obs[col].sum()):>6}/{adata.n_obs} malignant")

# ---- acceptance test: the held-out healthy-control CD4 ----
hc = adata.obs_names.isin(HC_CONTROL)
print(f"\nHC held-out control ({int(hc.sum())} CD4 cells of H__HC1_Blood)"
      f"   [gate <= 5%; v2 was 61%, v1 48%, v3.0 1.5%]")
for nm, col in METHODS.items():
    print(f"  {nm:<14} {int(adata.obs.loc[hc, col].sum()):>4} malignant "
          f"({adata.obs.loc[hc, col].mean():.1%})")
print("  donor state    ", adata.obs.loc[hc, "cnv_state"].value_counts().to_dict())
SIG
```

```
[B2__CTCL2] null n= 6904 thr=0.0944  called=8889/30525 (29.1%)  enriched clusters=0/4
[B2__CTCL3] null n= 7017 thr=0.1066  called=44/815 (5.4%)  enriched clusters=0/5
[B2__CTCL5] null n= 6640 thr=0.0903  called=257/709 (36.2%)  enriched clusters=0/4
[B2__CTCL6] null n= 6859 thr=0.1100  called=207/1125 (18.4%)  enriched clusters=0/4
[B5__PT01] null n= 6473 thr=0.0948  called=2190/3299 (66.4%)  enriched clusters=3/5  [40% of cells outside the null's depth range]
[B5__PT02] null n= 6561 thr=0.0960  called=3538/5652 (62.6%)  enriched clusters=2/5
[B5__PT04] null n= 6519 thr=0.1135  called=371/437 (84.9%)  enriched clusters=1/5  [60% of cells outside the null's depth range]
[B5__PT05] null n= 6509 thr=0.0854  called=4249/4933 (86.1%)  enriched clusters=3/6
[B5__PT06] null n= 6534 thr=0.1047  called=2352/5681 (41.4%)  enriched clusters=1/6
[B5__PT07] null n= 6527 thr=0.0870  called=2406/2603 (92.4%)  enriched clusters=2/4
[B5__PT08] null n=13120 thr=0.0966  called=3747/4547 (82.4%)  enriched clusters=3/9  [20% of cells outside the null's depth range]
[B5__PT09] null n= 6660 thr=0.0864  called=3153/3423 (92.1%)  enriched clusters=2/6
[B5__PT10] null n= 6558 thr=0.0887  called=1431/1869 (76.6%)  enriched clusters=2/4
[B5__PT13] null n= 6469 thr=0.0862  called=493/571 (86.3%)  enriched clusters=1/3

[B5__PT14] null n=13443 thr=0.1104  called=1945/6589 (29.5%)  enriched clusters=2/12
[B5__PT15] null n= 6500 thr=0.0955  called=552/586 (94.2%)  enriched clusters=1/3  [60% of cells outside the null's depth range]
[B5__PT16] null n= 7237 thr=0.1095  called=525/3072 (17.1%)  enriched clusters=2/6  [60% of cells outside the null's depth range]
[B5__PT18] null n= 6749 thr=0.1047  called=1030/3521 (29.3%)  enriched clusters=1/6
[B5__PT21] null n=13457 thr=0.1121  called=799/2628 (30.4%)  enriched clusters=2/10
[B5__PT22] null n= 6706 thr=0.0989  called=277/1209 (22.9%)  enriched clusters=1/5
[B5__PT23] null n= 8660 thr=0.1116  called=273/1028 (26.6%)  enriched clusters=0/4
[D5__MFIVB] null n= 6676 thr=0.0978  called=309/1066 (29.0%)  enriched clusters=3/5
[H__HC1] null n= 7089 thr=0.1006  called=12/512 (2.3%)  enriched clusters=0/1
[H__MFIV1] null n= 7494 thr=0.0970  called=142/2733 (5.2%)  enriched clusters=1/4
[H__SS1] null n= 6840 thr=0.0988  called=482/2748 (17.5%)  enriched clusters=0/5
[H__SS2] null n= 7107 thr=0.1025  called=357/963 (37.1%)  enriched clusters=1/5
[H__SS3] null n= 6616 thr=0.0943  called=339/1352 (25.1%)  enriched clusters=0/4
[H__SS4] null n= 6900 thr=0.1001  called=172/861 (20.0%)  enriched clusters=1/5
[H__SS5] null n= 6776 thr=0.1085  called=390/1492 (26.1%)  enriched clusters=1/5
[H__SS6] null n= 6868 thr=0.0972  called=795/1956 (40.6%)  enriched clusters=2/4

[SZ30_HTO10] null n= 8486 thr=0.1058  called=197/894 (22.0%)  enriched clusters=2/5
[SZ30_HTO6] null n= 8486 thr=0.1061  called=376/1512 (24.9%)  enriched clusters=2/5
[SZ30_HTO7] null n= 8486 thr=0.1059  called=340/1271 (26.8%)  enriched clusters=2/5
[SZ30_HTO8] null n= 8486 thr=0.1059  called=231/984 (23.5%)  enriched clusters=2/6
[SZ30_HTO9] null n= 8486 thr=0.1061  called=262/1202 (21.8%)  enriched clusters=2/5
[SZ32_HTO1] null n= 8510 thr=0.1033  called=489/1644 (29.7%)  enriched clusters=1/5
[SZ32_HTO2] null n= 8510 thr=0.1036  called=233/985 (23.7%)  enriched clusters=1/5
[SZ32_HTO3] null n= 8510 thr=0.1033  called=206/855 (24.1%)  enriched clusters=1/5
[SZ32_HTO4] null n= 8510 thr=0.1033  called=249/1137 (21.9%)  enriched clusters=1/5
[SZ32_HTO5] null n= 8510 thr=0.1033  called=81/474 (17.1%)  enriched clusters=1/5
[SZ32_HTO6] null n= 8510 thr=0.1033  called=303/1297 (23.4%)  enriched clusters=1/5
[SZ33_HTO10] null n= 6637 thr=0.0883  called=989/1083 (91.3%)  enriched clusters=3/6
[SZ33_HTO6] null n= 6637 thr=0.0891  called=796/856 (93.0%)  enriched clusters=2/6
[SZ33_HTO7] null n= 6637 thr=0.0880  called=1568/1672 (93.8%)  enriched clusters=3/7
[SZ33_HTO8] null n= 6637 thr=0.0877  called=1483/1571 (94.4%)  enriched clusters=3/7
[SZ33_HTO9] null n= 6637
... [17712 more chars truncated]
```

```python  # cell 36
# CNV call vs the orthogonal TCR call — malignant fraction by study & donor.
avail = (adata.obs["has_tcr"].to_numpy() & adata.obs["cnv_arm_proj"].notna().to_numpy()
         & adata.obs["cnv_callable"].to_numpy())
agree = C.strategy_agreement(adata, METHODS, "tcr_is_malignant", avail)
print(f"agreement vs TCR on {int(avail.sum())} TCR+ query cells in callable donors:")
print(agree.to_string())

rename = {"tcr_is_malignant": "TCR", "cnv_arm_malignant": "CNV arm consensus"}
frac_cols = ["tcr_is_malignant", *METHODS.values()]
q = adata.obs[adata.obs["cnv_arm_proj"].notna()]
for by, fs in [("study", (6, 3)), ("real_donor", (14, 3))]:
    tbl = q.groupby(by, observed=True)[frac_cols].mean().rename(columns=rename)
    ax = tbl.plot.bar(figsize=fs)
    ax.set_ylabel("malignant fraction"); ax.set_xlabel("")
    ax.set_title(f"CNV vs TCR — malignant fraction by {by} (blood)")
    # mark donors where the CNV caller abstained: a zero bar there means "no evidence"
    if by == "real_donor":
        for i, d in enumerate(tbl.index.astype(str)):
            if d not in CALLABLE:
                ax.get_xticklabels()[i].set_color("tab:red")
        ax.text(0.99, 0.95, "red = CNV indeterminate", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="tab:red")
    plt.tight_layout(); plt.savefig(FIG_DIR / f"blood_T_v4_cnv_methods_frac_by_{by}.png", dpi=150)
    plt.show()
```

```
agreement vs TCR on 147255 TCR+ query cells in callable donors:
               n_called  precision  recall     f1  jaccard
strategy                                                  
arm_consensus    110675      0.957   0.856  0.904    0.825
focal_cluster     35769      0.939   0.272  0.421    0.267
union            111815      0.952   0.861  0.904    0.826

<Figure size 720x360 with 1 Axes>
<Figure size 1680x360 with 1 Axes>
```

## Step 13c — per-sample inferCNV heatmaps · HEAVY (GPU kernel)

Chromosome heatmaps for a curated panel, to eyeball whether the calls make sense. `acnv.obs["donor"]`
is `sample_id`, so the five SZ30 HTO lanes are one inferCNV sample (`B4__SZ30`).

| sample | why |
|---|---|
| `B4__SZ30` | the dominance-rule case: HTO7 (was benign) and HTO8 (malignant) carry the *same* founder CDR3 |
| `H__HC1_Blood` | healthy control — should read flat; under both v1 and v2 it did not |
| `B5__PT23_Blood` | 0 TCR-malignant under the old dominance rule, 155 CNV-malignant |
| `B2__CTCL2` | clean positive control (TCR/CNV F1 ≈ 0.99) |

Re-runs `cnv.tl.infercnv` at `window=150` for these samples only (Step 11 keeps no `X_cnv`).
Payloads cached as `.npz` under `cnv_heatmap_cache_v3`, so re-plots are free. Worth diffing against
the v2 heatmaps: on the full gene space the banding driven by HVG selection should be gone, and
HC1 should read flat.

```python  # cell 38
HEATMAP_SAMPLES = ["B4__SZ30", "H__HC1_Blood", "B5__PT23_Blood", "B2__CTCL2"]
HM_WINDOW, HM_VLIM_SCALE, HM_CMAP = 150, 3.0, "RdBu_r"
CNV_HM_CACHE = OUT_DIR / "cnv_heatmap_cache_v4"; CNV_HM_CACHE.mkdir(exist_ok=True)
FORCE_CNV_HEATMAP = False

hm_sel = acnv.obs["donor"].astype(str).isin(HEATMAP_SAMPLES).to_numpy()
missing = sorted(set(HEATMAP_SAMPLES) - set(acnv.obs.loc[hm_sel, "donor"].astype(str)))
if missing:
    print("WARNING: not in acnv (skipped):", missing)
sub = acnv[hm_sel].copy()
print("heatmap cells:", sub.n_obs, "| samples:", sorted(sub.obs["donor"].astype(str).unique()))

C.run_cnv_heatmaps(sub, SHARED_REF, call_col=FINAL_CNV_COL,
                   reference_cat=REF_CATS, fig_prefix="blood_T_v3_cnv_heatmap",
                   fig_dir=FIG_DIR, n_per_study=len(HEATMAP_SAMPLES),
                   window=HM_WINDOW, vlim_scale=HM_VLIM_SCALE, cmap=HM_CMAP,
                   order_by="cnv_leiden", n_jobs=CNV_N_JOBS, chunk=CNV_CHUNK,
                   cnv_cache_dir=CNV_HM_CACHE, force_cnv=FORCE_CNV_HEATMAP)
del sub
gc.collect()
```

```
heatmap cells: 48460 | samples: ['B2__CTCL2', 'B4__SZ30', 'B5__PT23_Blood', 'H__HC1_Blood']
heatmap donors (4/study, by TCR burden): ['B2__CTCL2', 'B4__SZ30', 'B5__PT23_Blood', 'H__HC1_Blood'] | call_col = cnv_arm_malignant | ORDER_BY = cnv_leiden

  0%|          | 0/45 [00:00<?, ?it/s]
  2%|▏         | 1/45 [00:01<00:48,  1.10s/it]
 11%|█         | 5/45 [00:01<00:07,  5.27it/s]
 18%|█▊        | 8/45 [00:01<00:04,  8.42it/s]
 24%|██▍       | 11/45 [00:02<00:06,  5.23it/s]
 31%|███       | 14/45 [00:02<00:04,  7.41it/s]
 38%|███▊      | 17/45 [00:03<00:05,  5.22it/s]
 42%|████▏     | 19/45 [00:03<00:04,  6.35it/s]
 49%|████▉     | 22/45 [00:03<00:02,  8.54it/s]
 56%|█████▌    | 25/45 [00:04<00:03,  5.79it/s]
 60%|██████    | 27/45 [00:04<00:02,  6.90it/s]
 67%|██████▋   | 30/45 [00:04<00:01,  9.12it/s]
 73%|███████▎  | 33/45 [00:05<00:02,  5.93it/s]
 78%|███████▊  | 35/45 [00:05<00:01,  7.08it/s]
 84%|████████▍ | 38/45 [00:05<00:00,  9.38it/s]
 91%|█████████ | 41/45 [00:06<00:00,  6.15it/s]
 98%|█████████▊| 44/45 [00:06<00:00,  7.94it/s]
100%|██████████| 45/45 [00:06<00:00,  6.78it/s]


[B2__CTCL2] n=44464  ref=['hc_atlas', 'cd8_ref', 'healthy_pbmc']

[B2__CTCL2] cached heatmap inputs -> blood_T_v3_cnv_heatmap_B2__CTCL2.npz

[borcherding2023/B2__CTCL2] color vlim=±0.2500

<Figure size 1200x600 with 6 Axes>
  0%|          | 0/23 [00:00<?, ?it/s]
  4%|▍         | 1/23 [00:01<00:24,  1.11s/it]
 17%|█▋        | 4/23 [00:01<00:04,  4.18it/s]
 39%|███▉      | 9/23 [00:02<00:02,  4.67it/s]
 52%|█████▏    | 12/23 [00:02<00:01,  6.69it/s]
 70%|██████▉   | 16/23 [00:02<00:00,  9.79it/s]
 83%|████████▎ | 19/23 [00:03<00:00,  6.05it/s]
100%|██████████| 23/23 [00:03<00:00,  6.62it/s]


[B4__SZ30] n=22849  ref=['hc_atlas', 'cd8_ref', 'healthy_pbmc']

[B4__SZ30] cached heatmap inputs -> blood_T_v3_cnv_heatmap_B4__SZ30.npz
[geskin2026/B4__SZ30] color vlim=±0.2931

<Figure size 1200x600 with 6 Axes>
  0%|          | 0/19 [00:00<?, ?it/s]
  5%|▌         | 1/19 [00:01<00:21,  1.19s/it]
 32%|███▏      | 6/19 [00:01<00:02,  5.99it/s]
 47%|████▋     | 9/19 [00:02<00:02,  4.26it/s]
 63%|██████▎   | 12/19 [00:02<00:01,  6.31it/s]
 84%|████████▍ | 16/19 [00:02<00:00,  9.63it/s]
100%|██████████| 19/19 [00:03<00:00,  6.11it/s]
100%|██████████| 19/19 [00:03<00:00,  5.60it/s]


[B5__PT23_Blood] n=18349  ref=['hc_atlas', 'cd8_ref', 'healthy_pbmc']
[B5__PT23_Blood] cached heatmap inputs -> blood_T_v3_cnv_heatmap_B5__PT23_Blood.npz

[buus2025/B5__PT23_Blood] color vlim=±0.4272

<Figure size 1200x600 with 6 Axes>
  0%|          | 0/15 [00:00<?, ?it/s]
  7%|▋         | 1/15 [00:01<00:16,  1.15s/it]
 40%|████      | 6/15 [00:01<00:01,  6.19it/s]
 60%|██████    | 9/15 [00:02<00:01,  4.48it/s]
 73%|███████▎  | 11/15 [00:02<00:00,  5.68it/s]
100%|██████████| 15/15 [00:02<00:00,  6.29it/s]


[H__HC1_Blood] n=14894  ref=['hc_atlas', 'cd8_ref', 'healthy_pbmc']

[H__HC1_Blood] cached heatmap inputs -> blood_T_v3_cnv_heatmap_H__HC1_Blood.npz
[herrera2021/H__HC1_Blood] color vlim=±0.1956

<Figure size 1200x600 with 6 Axes>
0
```

## Step 14 — combine TCR + CNV & persist

`combined_malignant = TCR ∨ (CNV ∧ donor is callable)`. In an `indeterminate` donor the CNV caller
found no clone-level event clearing the diploid null, so a negative there is absence of evidence,
not evidence of absence — those cells fall back to the TCR label and are tagged
`*_cnv_indeterminate` in `malignant_evidence`.

Persisted per-cell to `blood_T_malignancy_v3.parquet` (feeds `33_subclone_tcr_signaling`), with the per-donor signature
and null diagnostics alongside in `tables/`.

```python  # cell 40
FINAL_CNV = FINAL_CNV_COL
tcr_m = adata.obs["tcr_is_malignant"].to_numpy()
cnv_m = adata.obs[FINAL_CNV].to_numpy().astype(bool)
# A CNV-only call counts only where the CNV caller is actually callable. In an indeterminate
# donor no cluster cleared the diploid null, so "not malignant by CNV" is an absence of evidence,
# not evidence of absence — the label falls back to TCR alone. `cnv_callable` tracks whichever
# channel FINAL_CNV_COL names (arm signature, or any usable null for the union).
cnv_ok = adata.obs["cnv_callable"].to_numpy()
adata.obs["combined_malignant"] = tcr_m | (cnv_m & cnv_ok)
adata.obs["malignant_evidence"] = np.select(
    [tcr_m & cnv_m & cnv_ok, tcr_m & ~cnv_ok, tcr_m, ~tcr_m & cnv_m & cnv_ok, ~cnv_ok],
    ["both", "tcr_only_cnv_indeterminate", "tcr_only", "cnv_only", "none_cnv_indeterminate"],
    default="none")
print("combined malignant:", int(adata.obs["combined_malignant"].sum()), "/", adata.n_obs)
print("\nevidence:\n", adata.obs["malignant_evidence"].value_counts())

# scored, not signature-carrying: a donor with a usable null but no arm signature still has a
# focal score, so availability keys on the score rather than on cnv_arm_proj (NaN when abstaining)
both_avail = (adata.obs["has_tcr"].to_numpy() & cnv_ok
              & adata.obs["cnv_focal_score"].notna().to_numpy())
print(f"\nTCR x CNV crosstab ({int(both_avail.sum())} cells with both signals, callable donors):")
ct = pd.crosstab(adata.obs.loc[both_avail, "tcr_is_malignant"],
                 adata.obs.loc[both_avail, FINAL_CNV], rownames=["tcr"], colnames=["cnv"])
print(ct)
if ct.shape == (2, 2):
    tpr = ct.loc[True, True] / ct.loc[True].sum()
    fpr = ct.loc[False, True] / ct.loc[False].sum()
    print(f"\nvs the TCR call:  TPR={tpr:.2f}  FPR={fpr:.2f}"
          f"   [v2 was 0.66 / 0.66 — i.e. chance; v3.0 0.45 / 0.17]")
```

```
combined malignant: 314320 / 557772

evidence:
 malignant_evidence
none_cnv_indeterminate        184992
tcr_only_cnv_indeterminate    168912
both                          105898
none                           58460
cnv_only                       21767
tcr_only                       17743
Name: count, dtype: int64

TCR x CNV crosstab (147255 cells with both signals, callable donors):
cnv    False   True 
tcr                 
False  18837    4777
True   17743  105898

vs the TCR call:  TPR=0.86  FPR=0.20   [v2 was 0.66 / 0.66 — i.e. chance; v3.0 0.45 / 0.17]
```

```python  # cell 41
keep = ["donor", "real_donor", "sample_id", "study", "disease", "cell_type", "cell_type_T2",
        "cached_malignant", "has_tcr", "tcr_clone_id", "tcr_clone_size", "tcr_is_expanded",
        "tcr_is_dominant_clone", "tcr_is_malignant", "tcr_malignant_alice",
        "cnv_score", "cnv_cell_score", "cnv_focal_score", "cnv_arm_proj", "cnv_leiden",
        "cnv_score_above_null", "cnv_arm_malignant", "cnv_state", "cnv_callable",
        "combined_malignant", "malignant_evidence"]
out = adata.obs[[c for c in keep if c in adata.obs.columns]].copy()
out.index.name = "obs_name"
out.to_parquet(OUT_PARQUET)
print("wrote", OUT_PARQUET, out.shape)

SIG.to_csv(TAB_DIR / "blood_cnv_v4_signatures.csv", index=False)
score_diag.to_csv(TAB_DIR / "blood_cnv_v4_score_null_diag.csv", index=False)
print("wrote per-donor signature + null diagnostics to", TAB_DIR)
out.head()
```

```
wrote /home/projects/nyosef/zvise/MF/data/atlas_joint/blood_T_malignancy_v4.parquet (557772, 26)
wrote per-donor signature + null diagnostics to /home/projects/nyosef/zvise/MF/tables

                                    donor real_donor     sample_id  \
obs_name                                                             
H__HC1_Blood|H19_AAACCTGTCCACGTTC  H__HC1     H__HC1  H__HC1_Blood   
H__HC1_Blood|H19_AAACCTGTCTGCGGCA  H__HC1     H__HC1  H__HC1_Blood   
H__HC1_Blood|H19_AAACGGGTCGGATGTT  H__HC1     H__HC1  H__HC1_Blood   
H__HC1_Blood|H19_AAAGATGAGGCGTACA  H__HC1     H__HC1  H__HC1_Blood   
H__HC1_Blood|H19_AAAGATGCAAGCGAGT  H__HC1     H__HC1  H__HC1_Blood   

                                         study disease cell_type cell_type_T2  \
obs_name                                                                        
H__HC1_Blood|H19_AAACCTGTCCACGTTC  herrera2021      HC   Unknown          CD4   
H__HC1_Blood|H19_AAACCTGTCTGCGGCA  herrera2021      HC   Unknown          CD4   
H__HC1_Blood|H19_AAACGGGTCGGATGTT  herrera2021      HC   Unknown          CD4   
H__HC1_Blood|H19_AAAGATGAGGCGTACA  herrera2021      HC   Unknown          CD4   
H__HC1_Blood|H19_AAAGATGCAAGCGAGT  herrera2021      HC   Unknown          CD4   

                                   cached_malignant  has_tcr  \
obs_name                                                       
H__HC1_Blood|H19_AAACCTGTCCACGTTC             False     True   
H__HC1_Blood|H19_AAACCTGTCTGCGGCA             False     True   
H__HC1_Blood|H19_AAACGGGTCGGATGTT             False     True   
H__HC1_Blood|H19_AAAGATGAGGCGTACA             False     True   
H__HC1_Blood|H19_AAAGATGCAAGCGAGT             False     True   

                                              tcr_clone_id  ...  \
obs_name                                                    ...   
H__HC1_Blood|H19_AAACCTGTCCACGTTC      TRB:CASSRPQGSDTEAFF  ...   
H__HC1_Blood|H19_AAACCTGTCTGCGGCA        TRB:CASSYSTSGEQFF  ...   
H__HC1_Blood|H19_AAACGGGTCGGATGTT     TRB:CASSYFPLAGGNEQYF  ...   
H__HC1_Blood|H19_AAAGATGAGGCGTACA  TRB:CARSDNGTSGRTPTGELFF  ...   
H__HC1_Blood|H19_AAAGATGCAAGCGAGT      TRB:CASSDPGWGNNEQFF  ...   

                                   cnv_cell_score  cnv_focal_score  \
obs_name                                                             
H__HC1_Blood|H19_AAACCTGTCCACGTTC             NaN              NaN   
H__HC1_Blood|H19_AAACCTGTCTGCGGCA             NaN              NaN   
H__HC1_Blood|H19_AAACGGGTCGGATGTT             NaN              NaN   
H__HC1_Blood|H19_AAAGATGAGGCGTACA             NaN              NaN   
H__HC1_Blood|H19_AAAGATGCAAGCGAGT             NaN              NaN   

                                   cnv_arm_proj  cnv_leiden  \
obs_name                                                      
H__HC1_Blood|H19_AAACCTGTCCACGTTC           NaN               
H__HC1_Blood|H19_AAACCTGTCTGCGGCA           NaN               
H__HC1_Blood|H19_AAACGGGTCGGATGTT           NaN               
H__HC1_Blood|H19_AAAGATGAGGCGTACA           NaN               
H__HC1_Blood|H19_AAAGATGCAAGCGAGT           NaN               

                                   cnv_score_above_null  cnv_arm_malignant  \
obs_name                                                                     
H__HC1_Blood|H19_AAACCTGTCCACGTTC                 False              False   
H__HC1_Blood|H19_AAACCTGTCTGCGGCA                 False              False   
H__HC1_Blood|H19_AAACGGGTCGGATGTT                 False              False   
H__HC1_Blood|H19_AAAGATGAGGCGTACA                 False              False   
H__HC1_Blood|H19_AAAGATGCAAGCGAGT                 False              False   

                                       cnv_state  cnv_callable  \
obs_name                                                         
H__HC1_Blood|H19_AAACCTGTCCACGTTC  indeterminate          True   
H__HC1_Blood|H19_AAACCTGTCTGCGGCA  indeterminate          True   
H__HC1_Blood|H19_AAACGGGTCGGATGTT  indeterminate          True   
H__HC1_Blood|H19_AAAGATGAGGC
... [652 more chars truncated]
```

```python  # cell 42
# TCR<->CNV quality (TCR+ cells in callable donors): sensitivity / specificity vs the TCR call.
avail = (adata.obs["has_tcr"].to_numpy() & adata.obs["cnv_focal_score"].notna().to_numpy()
         & adata.obs["cnv_callable"].to_numpy())
quality = C.tcr_cnv_quality(adata, METHODS, "tcr_is_malignant", avail)
print(f"TCR<->CNV quality on {int(avail.sum())} TCR+ cells in callable donors:")
print(quality.to_string())
print("\nagreement with the TCR call (v3.0 arm_consensus: precision 0.931, recall 0.454):")
print(C.strategy_agreement(adata, METHODS, "tcr_is_malignant", avail).to_string())
```

```
TCR<->CNV quality on 147255 TCR+ cells in callable donors:
               n_tcr_malig  n_tcr_benign      tp     tn  sensitivity  specificity
strategy                                                                         
arm_consensus       123641         23614  105898  18837        0.856        0.798
focal_cluster       123641         23614   33587  21432        0.272        0.908
union               123641         23614  106475  18274        0.861        0.774

agreement with the TCR call (v3.0 arm_consensus: precision 0.931, recall 0.454):
               n_called  precision  recall     f1  jaccard
strategy                                                  
arm_consensus    110675      0.957   0.856  0.904    0.825
focal_cluster     35769      0.939   0.272  0.421    0.267
union            111815      0.952   0.861  0.904    0.826
```

## Step 15 — CNV QC · the table this rebuild is judged on

Writes `tables/blood_cnv_v3_qc.csv` and prints the seven checks, each against its v2 number:

1. **HC1 held-out control FPR** — the gate. ≤5 %; v2 was 61 %.
2. **Second negative controls** — `B2__CTCL3` (0 % TCR-malignant, 72 % CNV in v2) and
   `B5__PT22` (8 % / 33 %) should collapse or go `indeterminate`.
3. **Discrimination** — per-donor AUROC vs `tcr_is_malignant`, restricted to donors with ≥50
   TCR-positive **and** ≥50 TCR-negative cells. AUROC at 99 % prevalence is meaningless, which is
   why v2's F1 diagnostics read 0.99 on donors where the call was worthless.
4. **Operating point** — TPR/FPR. v2: 0.66 / 0.66.
5. **Coverage** — callable vs indeterminate donors, and surviving `cnv_only` calls.
6. **Null FPR** — the held-out diploid cells' own call rate, which should sit at ~1 % by
   construction. If it does not, the null is not doing its job.
7. **Known biology** — recurrent arm events across callable donors should recover the canonical
   Sézary karyotype (chr7 gain, chr10q loss, chr17p loss, chr8q gain). `H__SS5` is the positive
   control: it already showed chr7q at +0.087 in v2.

```python  # cell 44
# ============================================================
# Step 15 — CNV QC. Everything the v3 rebuild has to be judged on, in one table.
# ============================================================
from sklearn.metrics import roc_auc_score

# every scored CD4 cell, including donors that have a null but no arm signature (cnv_arm_proj NaN)
qdf = adata.obs[adata.obs["cnv_focal_score"].notna()].copy()
qdf["pt"] = qdf["real_donor"].astype(str)
rows = []
for pt, x in qdf.groupby("pt", observed=True):
    t = x[x["has_tcr"].to_numpy()]
    yt = t["tcr_is_malignant"].to_numpy().astype(bool)
    balanced = yt.sum() >= 50 and (~yt).sum() >= 50      # AUROC is meaningless at 99% prevalence
    r = {"donor": pt, "n_cd4": len(x), "n_tcr": len(t),
         "tcr_malig_frac": round(float(yt.mean()), 3) if len(t) else np.nan,
         "cnv_state": x["cnv_state"].mode().iat[0],
         "arm_callable": bool(x["cnv_arm_callable"].iat[0]),
         "null_callable": bool(x["cnv_null_callable"].iat[0]),
         "balanced": balanced}
    for nm, col in METHODS.items():
        r[f"frac_{nm}"] = round(float(x[col].mean()), 3)
    for col, nm in [("cnv_arm_proj", "auc_arm"), ("cnv_focal_score", "auc_focal")]:
        s = t[col].to_numpy(dtype=float)
        ok = np.isfinite(s)
        r[nm] = (round(float(roc_auc_score(yt[ok], s[ok])), 3)
                 if balanced and ok.sum() > 100 and yt[ok].any() and (~yt[ok]).any() else np.nan)
    if balanced:
        for nm, col in METHODS.items():
            r[f"tpr_{nm}"] = round(float(t.loc[yt, col].mean()), 3)
            r[f"fpr_{nm}"] = round(float(t.loc[~yt, col].mean()), 3)
        prj = t["cnv_arm_proj"].to_numpy(dtype=float)
        if np.isfinite(prj[~yt]).any() and np.isfinite(prj[yt]).any():
            r["tpr_at_5pct_fpr"] = round(
                float((prj[yt] > np.nanpercentile(prj[~yt], 95)).mean()), 3)
    rows.append(r)
qc = (pd.DataFrame(rows)
      .merge(SIG[["donor", "anchor_frac", "n_event_arms", "signature", "null_fpr",
                  "frac_uncovered", "reason"]], on="donor", how="left")
      .merge(score_diag[["donor", "n_null", "thr_median", "n_clusters_enriched"]],
             on="donor", how="left")
      .sort_values("tcr_malig_frac", ascending=False))
qc.to_csv(QC_CSV, index=False)
print("wrote", QC_CSV)

bal = qc[qc["balanced"]]
hc = adata.obs_names.isin(HC_CONTROL)
print("\n1. per method: HC1 held-out control, then the median per-donor operating point")
print(f"   {'method':<14} {'HC1':>7}  {'TPR':>5} {'FPR':>5}   [gate HC1 <= 5%; v2 61% at .66/.66]")
for nm, col in METHODS.items():
    print(f"   {nm:<14} {adata.obs.loc[hc, col].mean():>7.1%}  "
          f"{bal[f'tpr_{nm}'].median():>5.2f} {bal[f'fpr_{nm}'].median():>5.2f}")
for d in ["B2__CTCL3", "B5__PT22"]:
    s = qc[qc.donor.eq(d)]
    if len(s):
        print(f"2. {d:<14} ..... arm {float(s.frac_arm_consensus.iat[0]):.1%} / "
              f"union {float(s.frac_union.iat[0]):.1%} malignant, state={s.cnv_state.iat[0]}"
              f"   [v2 72% / 33%; v3.0 5.9% / 16.5%]")
print(f"3. median per-donor AUROC ......... arm {bal.auc_arm.median():.3f} | "
      f"focal {bal.auc_focal.median():.3f}   [v3.0 0.656 / 0.708]  (n={len(bal)} balanced donors)")
print(f"4. anchor size .................... median {qc.anchor_frac.median():.1%} of the donor's "
      f"CD4   [v3.0 anchored 3 donors at 1-2%]")
print(f"5. coverage ....................... arm signature {int(qc.arm_callable.sum())}/{len(qc)}"
      f" | usable null {int(qc.null_callable.sum())}/{len(qc)}"
      f" | cnv_only cells {int(adata.obs.malignant_evidence.eq('cnv_only').sum())}"
      f"   [v3.0 31/59 arm]")
print(f"6. null FPR (held-out diploid) .... median {qc.null_fpr.median():.1%}  [target ~1%]")
print("\n7. recurrent arm events across callable donors:")
_ev = (SIG.loc[SIG["callable"], "signature"].str.split(", ").explode().value_counts())
print(_ev.head(12).to_string())
qc
```

```
wrote /home/projects/nyosef/zvise/MF/tables/blood_cnv_v4_qc.csv

1. per method: HC1 held-out control, then the median per-donor operating point
   method             HC1    TPR   FPR   [gate HC1 <= 5%; v2 61% at .66/.66]
   arm_consensus     1.2%   0.87  0.24
   focal_cluster     0.0%   0.05  0.10
   union             1.2%   0.87  0.29
2. B2__CTCL3      ..... arm 9.2% / union 9.2% malignant, state=benign   [v2 72% / 33%; v3.0 5.9% / 16.5%]
2. B5__PT22       ..... arm 31.0% / union 31.2% malignant, state=benign   [v2 72% / 33%; v3.0 5.9% / 16.5%]
3. median per-donor AUROC ......... arm 0.801 | focal 0.667   [v3.0 0.656 / 0.708]  (n=55 balanced donors)
4. anchor size .................... median 57.5% of the donor's CD4   [v3.0 anchored 3 donors at 1-2%]
5. coverage ....................... arm signature 64/64 | usable null 64/64 | cnv_only cells 21767   [v3.0 31/59 arm]
6. null FPR (held-out diploid) .... median 1.0%  [target ~1%]

7. recurrent arm events across callable donors:
signature
chr19q-    59
chr17p-    48
chr10p+    40
chr18p+    39
chr17q+    36
chr7p+     32
chr8q+     29
chr10q-    28
chr21q+    27
chr5p-     24
chr6p+     24
chr7q+     24

        donor  n_cd4  n_tcr  tcr_malig_frac  cnv_state  arm_callable  \
49  SZ38_HTO4   3386   3229           0.988  malignant          True   
46  SZ38_HTO1   4843   4202           0.985  malignant          True   
47  SZ38_HTO2   4177   3900           0.985  malignant          True   
48  SZ38_HTO3   4725   4442           0.985  malignant          True   
50  SZ38_HTO5   3701   3466           0.983  malignant          True   
..        ...    ...    ...             ...        ...           ...   
20   B5__PT23   1028    727           0.154     benign          True   
39  SZ32_HTO5    474    391           0.130     benign          True   
19   B5__PT22   1209    972           0.080     benign          True   
22     H__HC1    512    497           0.000     benign          True   
1   B2__CTCL3    815    800           0.000     benign          True   

    null_callable  balanced  frac_arm_consensus  frac_focal_cluster  ...  \
49           True     False               0.995               0.663  ...   
46           True      True               0.975               0.794  ...   
47           True      True               0.991               0.369  ...   
48           True      True               0.990               0.690  ...   
50           True      True               0.991               0.352  ...   
..            ...       ...                 ...                 ...  ...   
20           True      True               0.351               0.000  ...   
39           True      True               0.120               0.135  ...   
19           True      True               0.310               0.071  ...   
22           True     False               0.012               0.000  ...   
1            True     False               0.092               0.000  ...   

    tpr_at_5pct_fpr  anchor_frac  n_event_arms  \
49              NaN     0.317779            15   
46            0.454     0.379310            15   
47            0.117     0.410821            15   
48            0.270     0.357460            15   
50            0.329     0.369900            15   
..              ...          ...           ...   
20            0.339     0.545720            15   
39            0.235     0.582278             4   
19            0.269     0.744417             5   
22              NaN     1.000000             3   
1               NaN     0.985276             8   

                                            signature  null_fpr  \
49  chr17p-, chr17q+, chr10p+, chr8p-, chr8q+, chr...  0.011575   
46  chr17q+, chr17p-, chr8p-, chr5p-, chr19q-, chr...  0.011575   
47  chr17p-, chr17q+, chr10p+, chr8p-, chr8q+, chr...  0.011270   
48  chr17p-, chr17q+, chr10p+, chr8p-, chr8q+, chr...  0.010813   
50  chr17p-, chr17q+, chr10p+, chr5p-, chr18p+, ch...  0.011422   
..                                               
... [1217 more chars truncated]
```
