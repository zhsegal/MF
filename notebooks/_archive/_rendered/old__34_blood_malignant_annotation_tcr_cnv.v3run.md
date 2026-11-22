# old/34_blood_malignant_annotation_tcr_cnv.v3run.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/34_blood_malignant_annotation_tcr_cnv.v3run.ipynb` and is never run._

# 34 — Blood malignant annotation: ALICE TCR + inferCNV (v3)

Blood analogue of nb30. Runs the per-cell malignancy call on the blood T-cell re-annotation
`data/atlas_joint/blood_T_annotated.h5ad` (nb33; ~302k T cells: 240,993 CD4 / 60,638 CD8 / 731 NK,
98 patients). Reuses the ALICE (`alice_helpers`) and inferCNV (`skin_T_cnv_helpers`) engines.

**Blood-specific vs nb30:**
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
evidence rather than a clonality amplifier. Outputs are suffixed `_v3` and feed nb35; v2 stays on
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
        for sub in [Path("."), Path("notebooks/MF"), Path("scvi-tools-neural-nmf/notebooks/MF")]:
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
OBJ = OUT_DIR / "blood_T_fullgene_cnv_input.h5ad"
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

# ---- outputs (_blood_v1) ----
ALICE_CD4_PARQUET = OUT_DIR / "alice_cd4_per_donor_blood_v1.parquet"
ALICE_CD8_PARQUET = OUT_DIR / "alice_cd8_blood_v1.parquet"
ALICE_MAL_PARQUET = OUT_DIR / "alice_malignancy_blood_v1.parquet"
USE1_CSV = OUT_DIR / "alice_use1_subclone_families_blood_v1.csv"
USE2_CSV = OUT_DIR / "alice_use2_crosspatient_blood_v1.csv"
USE3_CSV = OUT_DIR / "alice_use3_reactive_cd8_blood_v1.csv"
# v3 = full gene space + mean-subtraction estimator + null-anchored, de novo arm-consensus caller.
# (v1 = same-sample-CD8 reference; v2 = pooled reference with the self-referential GMM caller,
#  which called 61% of the held-out healthy control malignant. nb35 reads v3.)
OUT_PARQUET = OUT_DIR / "blood_T_malignancy_v3.parquet"          # per-cell malignancy (feeds nb35)
ARM_CACHE   = OUT_DIR / "blood_T_arm_cnv_v3.parquet"             # per-arm CNV (feeds nb35)
QC_CSV      = TAB_DIR / "blood_cnv_v3_qc.csv"                    # per-donor CNV QC (Step 15)
```

```
NB_DIR = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
```

```python  # cell 2
import sys, gc, importlib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

sys.path.insert(0, str(NB_DIR))
import atlas_join_helpers as H
import skin_T_cnv_helpers as C
import alice_helpers as A
for _m in (H, C, A):
    importlib.reload(_m)
np.random.seed(SEED)
sc.settings.verbosity = 1
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
loaded cached full-gene CNV object (blood_T_fullgene_cnv_input.h5ad): (302362, 28168)
CNV gene space: 28168 genes
cell_type_T:
 cell_type_T
CD4    240993
CD8     60638
NK        731
Name: count, dtype: int64

study:
 study
geskin2026         167150
buus2025            61591
borcherding2023     36935
herrera2021         21109
borcherding2019     14176
rindler2021_fi       1401
Name: count, dtype: int64
dropped donors: {'B4__SZ33_HTO5': 'n_tcr_cells < 300', 'B2__CTCL4': 'n_tcr_cells < 300'}
kept donors: 64 | cells: 204178
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
dominant donors: CD4-only 62 | pooled 58 | changed 6
                n_tcr  n_dom_pool              top_clone  top_n  dom_frac  ratio  is_dominant       top_clone_pooled  top_n_pooled  second_n_pooled  dom_frac_pooled  ratio_pooled  is_dominant_pooled
donor                                                                                                                                                                                                 
B2__CTCL3        1802         840      TRB:CASSPQDTDTQYF     12     0.014   2.40        False    TRB:CASSLREATSYEQYF           369              123            0.205          3.00                True
B4__SZ30_HTO10    816         513  TRB:CASSLGLGTSYNSPLHF    252     0.491   3.19         True  TRB:CASSLGLGTSYNSPLHF           253              198            0.310          1.28               False
B4__SZ30_HTO6    1290         776  TRB:CASSLGLGTSYNSPLHF    411     0.530   3.57         True  TRB:CASSLGLGTSYNSPLHF           411              372            0.319          1.10               False
B4__SZ30_HTO7    1147         716  TRB:CASSLGLGTSYNSPLHF    371     0.518   4.42         True  TRB:CASSLGLGTSYNSPLHF           373              310            0.325          1.20               False
B4__SZ30_HTO9    1057         663  TRB:CASSLGLGTSYNSPLHF    330     0.498   2.89         True  TRB:CASSLGLGTSYNSPLHF           331              276            0.313          1.20               False
B5__PT23         2172         676    TRB:CASSPLAGSSYEQYF    121     0.179   5.26         True    TRB:CSASGLPSRTDTQYF           228              178            0.105          1.28               False
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
CD4 clonotypes: 19966 | donors: 64 | founders: 62
CD8 clonotypes: 8427 | donors: 64

       donor              cdr3  n_cells  is_founder
0  B2__CTCL2  CASSHAGAGGPYEQYF    28596        True
1  B2__CTCL2   CASSFWRGPSDEQFF        4       False
2  B2__CTCL2    CSVVKQGAPGGYTF        2       False
3  B2__CTCL2    CASSSQVGIGELFF        2       False
4  B2__CTCL2      CASRGNQETQYF        2       False
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
loaded cached /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/alice_cd4_per_donor_blood_v1.parquet
CD4 donors: 64 | significant clonotypes: 27
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
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/alice_use1_subclone_families_blood_v1.csv | donors: 64 | non-malignant flipped by family: 0

        donor founder_source           founder  n_variant_clones  \
50   B5__PT16       dominant     CASSLGTGAEVFF                 1   
40   B5__PT04       dominant   CAIGPDREKRNTIYF                 1   
55  D5__MFIVB       dominant   CASSQDRALENTIYF                 1   
51   B5__PT18       dominant   CASSLGRQGNYGYTF                 1   
52   B5__PT21       dominant  CSASGHGGMGNQPQHF                 1   
..        ...            ...               ...               ...   
58     H__SS1       dominant   CASSRQGGSGNTIYF                 0   
59     H__SS2       dominant  CASKLFMDREKGEAFF                 0   
60     H__SS3       dominant     CASSTTGNTEAFF                 0   
61     H__SS4       dominant  CASSRLGGAYNSPLHF                 0   
63     H__SS6       dominant    CASRSQGSYNEQFF                 0   

    n_family_significant  exact_frac  family_frac  delta_frac  exact_fold  \
50                     2    0.628348     0.940290    0.311942    2.014311   
40                     2    0.709924     0.979644    0.269720    2.632075   
55                     2    0.648696     0.758261    0.109565    5.920635   
51                     2    0.664237     0.666810    0.002573    2.291420   
52                     2    0.278283     0.280017    0.001734  160.500000   
..                   ...         ...          ...         ...         ...   
58                     0    0.623673     0.623673    0.000000  793.000000   
59                     0    0.653285     0.653285    0.000000  134.250000   
60                     0    0.783307     0.783307    0.000000  976.000000   
61                     0    0.391975     0.391975    0.000000   84.666667   
63                     0    0.610366     0.610366    0.000000  333.666667   

    family_fold  exact_call  family_call  flipped_by_family           study  \
50   153.181818        True         True              False        buus2025   
40   385.000000        True         True              False        buus2025   
55   109.000000        True         True              False  rindler2021_fi   
51     2.300296        True         True              False        buus2025   
52   161.500000        True         True              False        buus2025   
..          ...         ...          ...                ...             ...   
58   793.000000        True         True              False     herrera2021   
59   134.250000        True         True              False     herrera2021   
60   976.000000        True         True              False     herrera2021   
61    84.666667        True         True              False     herrera2021   
63   333.666667        True         True              False     herrera2021   

   disease disease_stage          entity  malignant  
50      SS            NA          Sezary       True  
40      SS            NA          Sezary       True  
55      MF           IVB      MF_classic       True  
51      SS            NA          Sezary       True  
52      SS            NA          Sezary       True  
..     ...           ...             ...        ...  
58      MF                MF/SS_leukemic       True  
59      MF                MF/SS_leukemic       True  
60      MF                MF/SS_leukemic       True  
61      MF                MF/SS_leukemic       True  
63      MF                MF/SS_leukemic       True  

[64 rows x 18 columns]
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
pooled founders: 62 | cross-patient significant: 0

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
loaded cached /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/alice_cd8_blood_v1.parquet
CD8 donors: 64 | reactive CD8 significant clonotypes: 35

               donor               cdr3  n_cells  is_founder disease_stage  \
6105          H__HC1  CASSSLSGGATNEKLFF        1       False           nan   
12022         H__HC1  CASSSLSGGATNEKLFF        1       False           nan   
6107          H__HC1  CASSSLTGGATNEKLFF        1       False           nan   
12024         H__HC1  CASSSLTGGATNEKLFF        1       False           nan   
12023         H__HC1  CASSSLSLGATNEKLFF        3       False           nan   
12025         H__HC1    CASSDSSGSTDTQYF        1       False           nan   
6106          H__HC1  CASSSLSLGATNEKLFF        3       False           nan   
12027         H__HC1  CASSSLSEGATNEKLFF        1       False           nan   
12026         H__HC1  CASSSLSSGATNEKLFF        3       False           nan   
6109          H__HC1  CASSSLSEGATNEKLFF        1       False           nan   
6108          H__HC1  CASSSLSSGATNEKLFF        3       False           nan   
5411        B5__PT23   CASDREGDSSGNTIYF      130       False            NA   
8906        B5__PT23   CASDREGDSSGNTIYF      130       False            NA   
12028         H__HC1  CASSSLTTGATNEKLFF        1       False           nan   
6110          H__HC1  CASSSLTTGATNEKLFF        1       False           nan   
1564   B4__SZ32_HTO2     CATSRVSEQETQYF        1       False           nan   
1563   B4__SZ32_HTO2     CATSRASEQETQYF        2       False           nan   
1358   B4__SZ32_HTO1     CASSLVGSEETQYF        1       False           nan   
2079   B4__SZ32_HTO6     CATSRGSFQETQYF        1       False           nan   
5413        B5__PT23   CASDWEGDSSGNTIYF        1       False            NA   
5412        B5__PT23   CASDREGDSTGNTIYF        1       False            NA   
5414        B5__PT23     CASKDRSTYDEQFF        1       False            NA   
2487   B4__SZ39_HTO1    CASSLRQGAYNEQFF        1       False           nan   
2488   B4__SZ39_HTO1    CASSLRQGEYNEQFF        1       False           nan   
2489   B4__SZ39_HTO1     CASSQDSSTDTQYF        1       False           nan   
3435        B5__PT02       CASSWVTGELFF        1       False            NA   
3436        B5__PT02       CASSGVTGELFF        1       False            NA   
2080   B4__SZ32_HTO6     CATSRGSDQETQYF        1       False           nan   
2486   B4__SZ39_HTO1     CASSRDSSTDTQYF        1       False           nan   
7255          H__SS2     CASSLYRGGEKLFF        1       False           nan   
7254          H__SS2     CASSYYRGGEKLFF        1       False           nan   
6643        H__MFIV1   CASSQGTSGVIYEQYF       57       False           nan   
6642        H__MFIV1   CASSPGTSGVIYEQYF       68       False           nan   
6645        H__MFIV1     CATSDTGTGETQYF        1       False           nan   
6644        H__MFIV1     CATSDTGQGETQYF        1       False           nan   

       obs_deg           Q   exp_deg        pois_p          q_bh  significant  \
6105         5  758.007771  0.033095  3.218644e-10  8.642058e-08         True   
12022        5   35.518770  0.012597  2.615566e-12  6.319207e-09         True   
6107         4  758.007771  0.098776  3.665441e-06  3.936683e-04         True   
12024        4   35.518770  0.037596  8.078226e-08  7.806798e-05         True   
12023        4   35.518770  0.002520  1.676253e-12  6.319207e-09         True   
12025        4   35.518770  0.202164  5.923863e-05  4.089158e-02         True   
6106         4  758.007771  0.006620  7.960360e-11  4.274713e-08         True   
12027        3   35.518770  0.002712  3.315927e-09  4.045806e-06         True   
12026        3   35.518770  0.002721  3.349177e-09  4.045806e-06         True   
6109         3  758.007771  0.007124  5.993510e-08  8.126881e-06         True   
6108         3  758.007771  0.007148  6.053542e-08  8.126881e-06         True   
5411         2  690.202606  0.000931  4.
... [2442 more chars truncated]
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
CD8 donors: 64 | with >=1 reactive hit: 9

            donor  n_clonotypes  n_with_neighbor  n_significant        study  \
56         H__HC1           537               31              6  herrera2021   
26  B4__SZ39_HTO1           113                4              4   geskin2026   
54       B5__PT23           215               11              4     buus2025   
57       H__MFIV1           299                4              4  herrera2021   
39       B5__PT02            63                2              2     buus2025   
..            ...           ...              ...            ...          ...   
58         H__SS1           313                2              0  herrera2021   
60         H__SS3            53                0              0  herrera2021   
61         H__SS4           268                7              0  herrera2021   
62         H__SS5           277                0              0  herrera2021   
63         H__SS6           363                2              0  herrera2021   

   disease disease_stage  malignant  
56      HC                    False  
26      SS                     True  
54      SS            NA       True  
57      MF                     True  
39      SS            NA       True  
..     ...           ...        ...  
58      MF                     True  
60      MF                     True  
61      MF                     True  
62      MF                     True  
63      MF                     True  

[64 rows x 8 columns]
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
fig.tight_layout(); fig.savefig(FIG_DIR / "alice_blood_v1_use1_family_vs_exact.png", bbox_inches="tight")

m = alice_cd4  # already carries is_founder (passed through alice_test)
fig, ax = plt.subplots(figsize=(5, 3))
ax.scatter(m["exp_deg"] + 1e-9, m["obs_deg"], s=6, alpha=.3, label="all")
f = m[m["is_founder"].fillna(False)]
ax.scatter(f["exp_deg"] + 1e-9, f["obs_deg"], s=20, color="tab:red", label="founder")
ax.set_xscale("log"); ax.set_xlabel("expected degree (null)"); ax.set_ylabel("observed degree")
ax.set_title("CD4 clonotype neighborhood enrichment (blood)"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG_DIR / "alice_blood_v1_use1_obs_vs_exp.png", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(use2["obs_deg"], bins=range(0, int(use2["obs_deg"].max()) + 2))
ax.set_xlabel("cross-patient neighbor degree"); ax.set_ylabel("founders")
ax.set_title(f"Cross-patient convergence: {int(use2['significant'].sum())} hits")
fig.tight_layout(); fig.savefig(FIG_DIR / "alice_blood_v1_use2_crosspatient_hist.png", bbox_inches="tight")
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
malignant donors: 62 | malignant cells: 122510
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/alice_malignancy_blood_v1.parquet
```

```python  # cell 21
pd.crosstab(adata.obs["tcr_malignant_alice"], adata.obs["cell_type_T2"], margins=True)
```

```
cell_type_T2            CD4    CD8   NK     All
tcr_malignant_alice                            
False                 46030  35101  537   81668
True                 122510      0    0  122510
All                  168540  35101  537  204178
```

## Step 8 — carry ALICE as the TCR malignant truth

The ALICE tumor call becomes `tcr_is_malignant` **and** `tcr_is_dominant_clone` (so it also drives the
benign/diploid inferCNV reference). Herrera SS donors, if present, are relabeled `disease='SS'`.

```python  # cell 23
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
relabeled Herrera SS donors -> disease='SS': 13347 cells
alice malignant: 122510 / 204178
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

```python  # cell 25
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
CD4               201210
not_applicable      2968
Name: count, dtype: int64
healthy blood CD4: 1055 -> hc_atlas reference | 527 -> held-out control (stays in query)
external healthy PBMC reference: 12000 cells
usable samples (CD8 >= 20): 38 | dropped: ['B5__PT15_Blood']

INFO:root:Extracted GTF attributes: ['gene_id', 'gene_version', 'gene_name', 'gene_source', 'gene_biotype', 'transcript_id', 'transcript_version', 'transcript_name', 'transcript_source', 'transcript_biotype', 'tag', 'transcript_support_level', 'exon_number', 'exon_id', 'exon_version', 'protein_id', 'protein_version', 'ccds_id']

WARNING: GTF file misses annotation for 3 genes in adata.
WARNING: Skipped 45 genes because of duplicate identifiers in GTF file.
cnv_ref: {'query': np.int64(144011), 'cd8_ref': np.int64(35091), 'nonclonal': np.int64(22855)} | shared ref: 13055 {'healthy_pbmc': np.int64(12000), 'hc_atlas': np.int64(1055)} | genes on chr1-22/X/Y: 18373
CNV samples: 38 | acnv: (201957, 18373) | shared ref: (13055, 18373) | HC held-out control cells: 527
detection depth (n_genes_cnv) — query vs shared reference:
                 count     25%     50%     75%
cnv_ref                                       
cd8_ref        35091.0  1076.0  1439.0  1907.0
nonclonal      22855.0  1219.0  1587.0  2046.0
query         144011.0   970.0  1566.0  2446.0
hc_atlas        1055.0  1148.0  1285.0  1481.0
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

```python  # cell 27
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

FINAL_CNV_COL = "cnv_arm_malignant"   # the CNV call Step 14 combines with the TCR call
CNV_N_JOBS = 8
CNV_CHUNK  = 1000        # smaller than v2's 2500: 20k genes densify to ~160 MB per chunk
FORCE_CNV  = False       # set True to recompute
CNV_CACHE  = OUT_DIR / f"blood_T_pooledref_cnv_per_cell_v3_res{CNV_LEIDEN_RES:g}.parquet"
```

## Step 11 — per-sample inferCNV vs the pooled reference · HEAVY (long-running)

`C.run_per_donor_infercnv` concatenates `SHARED_REF` onto each sample block, runs
`cnv.tl.infercnv`, then PCA / neighbors / Leiden on **query + reference together**, and fills
`cnv_score` (per `cnv_leiden` cluster), `cnv_cell_score`, `cnv_focal_score` (mean of the top
`TOPK_FRAC` windows), `cnv_leiden`. Cached to `CNV_CACHE`.

**What v3 changes here** (all four are knobs in Step 10, defaults unchanged for nb22/nb30):

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

```python  # cell 29
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
loaded cached inferCNV (blood_T_pooledref_cnv_per_cell_v3_res0.5.parquet) for 38 donors

per-cell rows: (449620, 7)
cnv_ref
ref_null     265126
query        144011
nonclonal     22855
cd8_ref       17628

null cells per sample (need >= 50 for a usable threshold):
min    6483.0
25%    6621.0
50%    6850.0
max    8731.0
```

## Step 12 — per-cell arm-level CNV · HEAVY (GPU kernel)

Second inferCNV pass with `calculate_gene_values=True`, averaging per-gene CNV within each
chromosome arm into a compact `cells × ~41 arm` matrix. In v2 this ran last and fed only nb35;
in v3 it runs **before** the call, because the caller is built on it.

Same estimator and the same `seed` as Step 11, so `_split_null` holds out exactly the same cells —
that is what lets the arm null describe the same diploid cells as the score null. `keep_cats` now
retains `nonclonal` (a query category in v3) and `ref_null` (the null) alongside `query`.

Checkpoints per sample in `<cache>_parts/`, so a resubmitted job resumes.

```python  # cell 31
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
arm CNV n_jobs = 1
loaded cached arm-CNV (blood_T_arm_cnv_v3.parquet) for 38 samples (431992, 43)
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/blood_T_arm_cnv_v3.parquet | arm matrix: (431992, 41) | arms: ['chr10p', 'chr10q', 'chr11p', 'chr11q', 'chr12p', 'chr12q', 'chr13q', 'chr14q', 'chr15q', 'chr16p', 'chr16q', 'chr17p', 'chr17q', 'chr18p', 'chr18q', 'chr19p', 'chr19q', 'chr1p', 'chr1q', 'chr20p', 'chr20q', 'chr21q', 'chr22q', 'chr2p', 'chr2q', 'chr3p', 'chr3q', 'chr4p', 'chr4q', 'chr5p', 'chr5q', 'chr6p', 'chr6q', 'chr7p', 'chr7q', 'chr8p', 'chr8q', 'chr9p', 'chr9q', 'chrXp', 'chrXq']
shared-reference null cells — arm pass 247663, score pass 265126: True
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

```python  # cell 33
# ---- 13a. score-level diagnostic: how far the query sits above its own diploid null ----
# Not the call. Reference cells built the baseline they are scored against, so this null is
# anti-conservative wherever query and reference differ by batch/chemistry; it is here to show
# that gap, and to flag donors whose cells fall outside the null's depth range.
CD4 = acnv.obs["cnv_ref"].isin(["query", "nonclonal"]).to_numpy()
obs_pt = acnv.obs[CD4].copy()
obs_pt["donor"] = obs_pt["real_donor"].astype(str)     # call per PATIENT; inferCNV ran per sample
PT_DONORS = sorted(obs_pt["donor"].unique())

# PER_CELL['donor'] is sample_id for query and null rows alike -> lift both to real_donor
_samp2pt = acnv.obs.groupby("donor", observed=True)["real_donor"].first().astype(str)
pc_pt = PER_CELL.copy()
pc_pt["donor"] = (pc_pt["donor"].astype(str).map(_samp2pt)
                  .fillna(pc_pt["donor"].astype(str)))

score_call, score_thr, score_diag = C.call_per_donor_null(
    pc_pt, PT_DONORS, obs_pt, score_col="cnv_focal_score", q=0.99,
    depth_col="n_genes_cnv", cluster_col="cnv_leiden")

# ---- 13b. the call: de novo signed arm consensus ----
arm_proj, arm_call, arm_state, SIG = C.arm_consensus_score(
    arm, pc_pt, PT_DONORS, arm_cols=ARM_COLS, cluster_col="cnv_leiden",
    depth_col="n_genes_cnv", q=0.99)

for col, val in [("cnv_arm_proj", arm_proj), ("cnv_arm_malignant", arm_call),
                 ("cnv_state", arm_state)]:
    acnv.obs[col] = pd.Series(val, index=arm.index).reindex(acnv.obs_names).to_numpy()
acnv.obs["cnv_arm_malignant"] = acnv.obs["cnv_arm_malignant"].fillna(False).astype(bool)
acnv.obs["cnv_state"] = acnv.obs["cnv_state"].fillna("indeterminate").astype(str)
acnv.obs.loc[~CD4, "cnv_arm_malignant"] = False        # cd8_ref is baseline, never a candidate
acnv.obs["cnv_score_above_null"] = (pd.Series(score_call, index=obs_pt.index)
                                    .reindex(acnv.obs_names).fillna(False).astype(bool))

CALLABLE = set(SIG.loc[SIG["callable"], "donor"])
print(f"\ncallable patients: {len(CALLABLE)}/{len(PT_DONORS)}"
      f"  | indeterminate: {sorted(set(PT_DONORS) - CALLABLE)}")

METHODS = {"arm_consensus": "cnv_arm_malignant"}
for col in ["cnv_score", "cnv_cell_score", "cnv_focal_score", "cnv_arm_proj"]:
    adata.obs[col] = acnv.obs[col].where(CD4).reindex(adata.obs_names)
adata.obs["cnv_leiden"] = acnv.obs["cnv_leiden"].reindex(adata.obs_names).fillna("")
adata.obs["cnv_state"] = (acnv.obs["cnv_state"].reindex(adata.obs_names)
                          .fillna("indeterminate").astype(str))
for col in ["cnv_arm_malignant", "cnv_score_above_null"]:
    adata.obs[col] = acnv.obs[col].reindex(adata.obs_names).fillna(False).astype(bool)
# donor-level callability: a False CNV call means something different in an indeterminate donor
adata.obs["cnv_callable"] = adata.obs["real_donor"].astype(str).isin(CALLABLE).to_numpy()
print(f"\narm-consensus malignant: {int(adata.obs[FINAL_CNV_COL].sum())}/{adata.n_obs}")

# ---- acceptance test: the held-out healthy-control CD4 ----
hc = adata.obs_names.isin(HC_CONTROL)
print(f"\nHC held-out control ({int(hc.sum())} CD4 cells of H__HC1_Blood):"
      f"\n  arm-consensus call   {int(adata.obs.loc[hc, FINAL_CNV_COL].sum())} malignant "
      f"({adata.obs.loc[hc, FINAL_CNV_COL].mean():.1%})   [target <= 5%; v2 was 61%, v1 48%]"
      f"\n  donor state          {adata.obs.loc[hc, 'cnv_state'].value_counts().to_dict()}"
      f"\n  score-above-null     {adata.obs.loc[hc, 'cnv_score_above_null'].mean():.1%} "
      f"(diagnostic only)")
SIG
```

```
[B2__CTCL2] null n= 7356 thr=0.0985  called=2271/29623 (7.7%)  enriched clusters=0/4
[B2__CTCL3] null n= 7019 thr=0.1105  called=17/861 (2.0%)  enriched clusters=0/5
[B2__CTCL5] null n= 6671 thr=0.0942  called=184/697 (26.4%)  enriched clusters=0/4
[B2__CTCL6] null n= 6835 thr=0.1122  called=168/1224 (13.7%)  enriched clusters=0/5
[B5__PT01] null n= 6510 thr=0.1040  called=1348/3277 (41.1%)  enriched clusters=1/4  [40% of cells outside the null's depth range]
[B5__PT02] null n= 6602 thr=0.1047  called=2202/5615 (39.2%)  enriched clusters=2/6
[B5__PT04] null n= 6529 thr=0.1176  called=331/443 (74.7%)  enriched clusters=1/5  [60% of cells outside the null's depth range]
[B5__PT05] null n= 6563 thr=0.0943  called=2931/4859 (60.3%)  enriched clusters=2/6
[B5__PT06] null n= 6585 thr=0.1129  called=1598/5582 (28.6%)  enriched clusters=1/5
[B5__PT07] null n= 6542 thr=0.0899  called=2281/2586 (88.2%)  enriched clusters=1/4
[B5__PT08] null n=13202 thr=0.1076  called=1957/4421 (44.3%)  enriched clusters=0/9
[B5__PT09] null n= 6746 thr=0.0968  called=2756/3338 (82.6%)  enriched clusters=2/5
[B5__PT10] null n= 6620 thr=0.0947  called=1039/1804 (57.6%)  enriched clusters=1/4
[B5__PT13] null n= 6483 thr=0.0912  called=452/557 (81.1%)  enriched clusters=1/4
[B5__PT14] null n=13643 thr=0.1100  called=1696/6271 (27.0%)  enriched clusters=2/12
[B5__PT16] null n= 7315 thr=0.1107  called=423/3015 (14.0%)  enriched clusters=1/5  [60% of cells outside the null's depth range]
[B5__PT18] null n= 7077 thr=0.1117  called=534/2911 (18.3%)  enriched clusters=1/6
[B5__PT21] null n=13372 thr=0.1176  called=665/2869 (23.2%)  enriched clusters=1/10
[B5__PT22] null n= 6826 thr=0.1103  called=163/1190 (13.7%)  enriched clusters=1/6
[B5__PT23] null n= 8731 thr=0.1170  called=238/964 (24.7%)  enriched clusters=1/5
[D5__MFIVB] null n= 6905 thr=0.1018  called=142/641 (22.2%)  enriched clusters=1/5  [20% of cells outside the null's depth range]
[H__HC1] null n= 7106 thr=0.1067  called=6/527 (1.1%)  enriched clusters=0/1
[H__MFIV1] null n= 7508 thr=0.1016  called=64/2714 (2.4%)  enriched clusters=1/4
[H__SS1] null n= 6904 thr=0.1034  called=344/2679 (12.8%)  enriched clusters=0/5
[H__SS2] null n= 7143 thr=0.1079  called=327/934 (35.0%)  enriched clusters=1/5
[H__SS3] null n= 6593 thr=0.1015  called=226/1403 (16.1%)  enriched clusters=0/4
[H__SS4] null n= 6969 thr=0.1037  called=139/771 (18.0%)  enriched clusters=0/5
[H__SS5] null n= 6990 thr=0.1106  called=353/1152 (30.6%)  enriched clusters=1/5
[H__SS6] null n= 6933 thr=0.1007  called=754/1865 (40.4%)  enriched clusters=1/4
[SZ30_HTO10] null too small (0 < 50) — no calls
[SZ30_HTO6] null too small (0 < 50) — no calls
[SZ30_HTO7] null too small (0 < 50) — no calls
[SZ30_HTO8] null too small (0 < 50) — no calls
[SZ30_HTO9] null n= 8511 thr=0.1105  called=216/1204 (17.9%)  enriched clusters=1/5
[SZ32_HTO1] null too small (0 < 50) — no calls
[SZ32_HTO2] null too small (0 < 50) — no calls
[SZ32_HTO3] null too small (0 < 50) — no calls
[SZ32_HTO4] null n= 8069 thr=0.1080  called=206/1329 (15.5%)  enriched clusters=1/5
[SZ32_HTO5] null too small (0 < 50) — no calls
[SZ32_HTO6] null too small (0 < 50) — no calls
[SZ33_HTO10] null n= 6736 thr=0.1022  called=700/1050 (66.7%)  enriched clusters=2/6
[SZ33_HTO6] null too small (0 < 50) — no calls
[SZ33_HTO7] null too small (0 < 50) — no calls
[SZ33_HTO8] null too small (0 < 50) — no calls
[SZ33_HTO9] null too small (0 < 50) — no calls
[SZ38_HTO1] null n= 6866 thr=0.1235  called=1032/4292 (24.0%)  enriched clusters=1/6  [20% of cells outside the null's depth range]
[SZ38_HTO2] null too small (0 < 50) — no calls
[SZ38_HTO3] null too small (0 < 50) — no calls
[SZ38_HTO4] null too small (0 < 50) — no calls
[SZ38_HTO5] null too small (0 < 50) — no calls
[SZ38_HTO6] null too small (0 < 50) — no calls
[SZ39_HTO1] null too small (0 < 50) — no calls
[SZ39_HTO2] null n= 7562 thr=0.1089  called=199/1387 (14.3%)  enriched clusters=1/4
[SZ39_HTO3] null too small (0 < 50) — no calls
[SZ39_HT
... [10617 more chars truncated]
```

```python  # cell 34
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
    plt.tight_layout(); plt.savefig(FIG_DIR / f"blood_T_v3_cnv_methods_frac_by_{by}.png", dpi=150)
    plt.show()
```

```
agreement vs TCR on 87015 TCR+ query cells in callable donors:
               n_called  precision  recall    f1  jaccard
strategy                                                 
arm_consensus     35376      0.931   0.454  0.61    0.439

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

```python  # cell 36
HEATMAP_SAMPLES = ["B4__SZ30", "H__HC1_Blood", "B5__PT23_Blood", "B2__CTCL2"]
HM_WINDOW, HM_VLIM_SCALE, HM_CMAP = 150, 3.0, "RdBu_r"
CNV_HM_CACHE = OUT_DIR / "cnv_heatmap_cache_v3"; CNV_HM_CACHE.mkdir(exist_ok=True)
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
heatmap cells: 48641 | samples: ['B2__CTCL2', 'B4__SZ30', 'B5__PT23_Blood', 'H__HC1_Blood']
heatmap donors (4/study, by TCR burden): ['B2__CTCL2', 'B4__SZ30', 'B5__PT23_Blood', 'H__HC1_Blood'] | call_col = cnv_arm_malignant | ORDER_BY = cnv_leiden
[B2__CTCL2] loaded cached heatmap inputs (blood_T_v3_cnv_heatmap_B2__CTCL2.npz)  X_cnv=(29087, 1451)
[borcherding2023/B2__CTCL2] color vlim=±0.2503

<Figure size 1200x600 with 6 Axes>
[B4__SZ30] loaded cached heatmap inputs (blood_T_v3_cnv_heatmap_B4__SZ30.npz)  X_cnv=(4297, 1451)
[geskin2026/B4__SZ30] color vlim=±0.2945

<Figure size 1200x600 with 6 Axes>
[B5__PT23_Blood] loaded cached heatmap inputs (blood_T_v3_cnv_heatmap_B5__PT23_Blood.npz)  X_cnv=(409, 1451)
[buus2025/B5__PT23_Blood] color vlim=±0.4316

<Figure size 1200x600 with 6 Axes>
[H__HC1_Blood] loaded cached heatmap inputs (blood_T_v3_cnv_heatmap_H__HC1_Blood.npz)  X_cnv=(527, 1451)
[herrera2021/H__HC1_Blood] color vlim=±0.1961

<Figure size 1200x600 with 6 Axes>
0
```

## Step 14 — combine TCR + CNV & persist

`combined_malignant = TCR ∨ (CNV ∧ donor is callable)`. In an `indeterminate` donor the CNV caller
found no clone-level event clearing the diploid null, so a negative there is absence of evidence,
not evidence of absence — those cells fall back to the TCR label and are tagged
`*_cnv_indeterminate` in `malignant_evidence`.

Persisted per-cell to `blood_T_malignancy_v3.parquet` (feeds nb35), with the per-donor signature
and null diagnostics alongside in `tables/`.

```python  # cell 38
FINAL_CNV = FINAL_CNV_COL
tcr_m = adata.obs["tcr_is_malignant"].to_numpy()
cnv_m = adata.obs[FINAL_CNV].to_numpy().astype(bool)
# A CNV-only call counts only where the CNV caller is actually callable. In an indeterminate
# donor no cluster cleared the diploid null, so "not malignant by CNV" is an absence of evidence,
# not evidence of absence — the label falls back to TCR alone.
cnv_ok = adata.obs["cnv_callable"].to_numpy()
adata.obs["combined_malignant"] = tcr_m | (cnv_m & cnv_ok)
adata.obs["malignant_evidence"] = np.select(
    [tcr_m & cnv_m & cnv_ok, tcr_m & ~cnv_ok, tcr_m, ~tcr_m & cnv_m & cnv_ok, ~cnv_ok],
    ["both", "tcr_only_cnv_indeterminate", "tcr_only", "cnv_only", "none_cnv_indeterminate"],
    default="none")
print("combined malignant:", int(adata.obs["combined_malignant"].sum()), "/", adata.n_obs)
print("\nevidence:\n", adata.obs["malignant_evidence"].value_counts())

both_avail = (adata.obs["has_tcr"].to_numpy() & cnv_ok
              & adata.obs["cnv_arm_proj"].notna().to_numpy())
print(f"\nTCR x CNV crosstab ({int(both_avail.sum())} cells with both signals, callable donors):")
ct = pd.crosstab(adata.obs.loc[both_avail, "tcr_is_malignant"],
                 adata.obs.loc[both_avail, FINAL_CNV], rownames=["tcr"], colnames=["cnv"])
print(ct)
if ct.shape == (2, 2):
    tpr = ct.loc[True, True] / ct.loc[True].sum()
    fpr = ct.loc[False, True] / ct.loc[False].sum()
    print(f"\nvs the TCR call:  TPR={tpr:.2f}  FPR={fpr:.2f}"
          f"   [v2 was 0.66 / 0.66 — i.e. chance]")
```

```
combined malignant: 132416 / 204178

evidence:
 malignant_evidence
tcr_only_cnv_indeterminate    49896
none                          43284
tcr_only                      39669
both                          32945
none_cnv_indeterminate        28478
cnv_only                       9906
Name: count, dtype: int64

TCR x CNV crosstab (87015 cells with both signals, callable donors):
cnv    False  True 
tcr                
False  11970   2431
True   39669  32945

vs the TCR call:  TPR=0.45  FPR=0.17   [v2 was 0.66 / 0.66 — i.e. chance]
```

```python  # cell 39
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

SIG.to_csv(TAB_DIR / "blood_cnv_v3_signatures.csv", index=False)
score_diag.to_csv(TAB_DIR / "blood_cnv_v3_score_null_diag.csv", index=False)
print("wrote per-donor signature + null diagnostics to", TAB_DIR)
out.head()
```

```
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/blood_T_malignancy_v3.parquet (204178, 26)
wrote per-donor signature + null diagnostics to /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/tables

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
H__HC1_Blood|H19_AAACCTGTCCACGTTC              True     True   
H__HC1_Blood|H19_AAACCTGTCTGCGGCA              True     True   
H__HC1_Blood|H19_AAACGGGTCGGATGTT              True     True   
H__HC1_Blood|H19_AAAGATGAGGCGTACA              True     True   
H__HC1_Blood|H19_AAAGATGCAAGCGAGT              True     True   

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
H__HC1_Blood|H19_AAACGGGTCGGATGTT        0.027328         0.073367   
H__HC1_Blood|H19_AAAGATGAGGCGTACA             NaN              NaN   
H__HC1_Blood|H19_AAAGATGCAAGCGAGT             NaN              NaN   

                                   cnv_arm_proj      cnv_leiden  \
obs_name                                                          
H__HC1_Blood|H19_AAACCTGTCCACGTTC           NaN                   
H__HC1_Blood|H19_AAACCTGTCTGCGGCA           NaN                   
H__HC1_Blood|H19_AAACGGGTCGGATGTT      0.030869  H__HC1_Blood_2   
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
H_
... [744 more chars truncated]
```

```python  # cell 40
# TCR<->CNV quality (TCR+ cells in callable donors): sensitivity / specificity vs the TCR call.
avail = (adata.obs["has_tcr"].to_numpy() & adata.obs["cnv_arm_proj"].notna().to_numpy()
         & adata.obs["cnv_callable"].to_numpy())
quality = C.tcr_cnv_quality(adata, METHODS, "tcr_is_malignant", avail)
print(f"TCR<->CNV quality on {int(avail.sum())} TCR+ cells in callable donors:")
print(quality.to_string())
```

```
TCR<->CNV quality on 87015 TCR+ cells in callable donors:
               n_tcr_malig  n_tcr_benign     tp     tn  sensitivity  specificity
strategy                                                                        
arm_consensus        72614         14401  32945  11970        0.454        0.831
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

```python  # cell 42
# ============================================================
# Step 15 — CNV QC. Everything the v3 rebuild has to be judged on, in one table.
# ============================================================
from sklearn.metrics import roc_auc_score

qdf = adata.obs[adata.obs["cnv_arm_proj"].notna()].copy()
qdf["pt"] = qdf["real_donor"].astype(str)
rows = []
for pt, x in qdf.groupby("pt", observed=True):
    y = x["tcr_is_malignant"].to_numpy().astype(bool)
    t = x[x["has_tcr"].to_numpy()]
    yt = t["tcr_is_malignant"].to_numpy().astype(bool)
    balanced = yt.sum() >= 50 and (~yt).sum() >= 50      # AUROC is meaningless at 99% prevalence
    r = {"donor": pt, "n_cd4": len(x), "n_tcr": len(t),
         "tcr_malig_frac": round(float(yt.mean()), 3) if len(t) else np.nan,
         "cnv_state": x["cnv_state"].mode().iat[0],
         "cnv_callable": bool(x["cnv_callable"].iat[0]),
         "cnv_malig_frac": round(float(x[FINAL_CNV_COL].mean()), 3),
         "balanced": balanced}
    for col, nm in [("cnv_arm_proj", "auc_arm"), ("cnv_focal_score", "auc_focal")]:
        s = t[col].to_numpy(dtype=float)
        ok = np.isfinite(s)
        r[nm] = (round(float(roc_auc_score(yt[ok], s[ok])), 3)
                 if balanced and ok.sum() > 100 else np.nan)
    if balanced:
        neg = t.loc[~yt, "cnv_arm_proj"].to_numpy(dtype=float)
        pos = t.loc[yt, "cnv_arm_proj"].to_numpy(dtype=float)
        r["tpr_at_5pct_fpr"] = round(float((pos > np.nanpercentile(neg, 95)).mean()), 3)
        r["tpr"] = round(float(t.loc[yt, FINAL_CNV_COL].mean()), 3)
        r["fpr"] = round(float(t.loc[~yt, FINAL_CNV_COL].mean()), 3)
    rows.append(r)
qc = (pd.DataFrame(rows)
      .merge(SIG[["donor", "n_event_arms", "signature", "null_fpr", "frac_uncovered", "reason"]],
             on="donor", how="left")
      .merge(score_diag[["donor", "n_null", "thr_median"]], on="donor", how="left")
      .sort_values("tcr_malig_frac", ascending=False))
qc.to_csv(QC_CSV, index=False)
print("wrote", QC_CSV)

bal = qc[qc["balanced"]]
print(f"\n1. HC1 held-out control FPR ....... "
      f"{adata.obs.loc[adata.obs_names.isin(HC_CONTROL), FINAL_CNV_COL].mean():.1%}  "
      f"[gate <= 5%; v2 61%]")
for d in ["B2__CTCL3", "B5__PT22"]:
    s = qc[qc.donor.eq(d)]
    if len(s):
        print(f"2. {d:<14} ............. {float(s.cnv_malig_frac.iat[0]):.1%} malignant, "
              f"state={s.cnv_state.iat[0]}  [v2: 72% / 33%]")
print(f"3. median per-donor AUROC ......... arm {bal.auc_arm.median():.3f} | "
      f"focal {bal.auc_focal.median():.3f}   [v2 focal 0.66]  (n={len(bal)} balanced donors)")
print(f"4. operating point ................ TPR {bal.tpr.median():.2f} / FPR {bal.fpr.median():.2f}"
      f"   [v2 0.66 / 0.66]")
print(f"5. coverage ....................... callable {int(qc.cnv_callable.sum())}/{len(qc)} donors"
      f" | cnv_only cells "
      f"{int(adata.obs.malignant_evidence.eq('cnv_only').sum())}")
print(f"6. null FPR (held-out diploid) .... median {qc.null_fpr.median():.1%}  [target ~1%]")
print("\n7. recurrent arm events across callable donors:")
_ev = (SIG.loc[SIG["callable"], "signature"].str.split(", ").explode().value_counts())
print(_ev.head(12).to_string())
qc
```

```
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/tables/blood_cnv_v3_qc.csv

1. HC1 held-out control FPR ....... 1.5%  [gate <= 5%; v2 61%]
2. B2__CTCL3      ............. 5.9% malignant, state=benign  [v2: 72% / 33%]
2. B5__PT22       ............. 16.5% malignant, state=benign  [v2: 72% / 33%]
3. median per-donor AUROC ......... arm 0.656 | focal 0.708   [v2 focal 0.66]  (n=48 balanced donors)
4. operating point ................ TPR 0.21 / FPR 0.10   [v2 0.66 / 0.66]
5. coverage ....................... callable 31/59 donors | cnv_only cells 9906
6. null FPR (held-out diploid) .... median 0.5%  [target ~1%]

7. recurrent arm events across callable donors:
signature
chr19q-    23
chr18p+    14
chr17p-    14
chr10p+    14
chr22q-    12
chr18q-    10
chr7p+      9
chr21q+     9
chr17q+     9
chr6q+      9
chr9q-      6
chr12p+     6

         donor  n_cd4  n_tcr  tcr_malig_frac  cnv_state  cnv_callable  \
41   SZ38_HTO1   4292   4027           0.989     benign          True   
44   SZ38_HTO4   3288   3161           0.989     benign         False   
42   SZ38_HTO2   4006   3774           0.987     benign         False   
43   SZ38_HTO3   4590   4346           0.986     benign         False   
45   SZ38_HTO5   3589   3380           0.986     benign         False   
46   SZ38_HTO6   3830   3608           0.985     benign         False   
6     B5__PT06   5582   4867           0.983  malignant          True   
0    B2__CTCL2  29623  29132           0.982     benign          True   
37   SZ33_HTO6    832    509           0.980     benign         False   
40   SZ33_HTO9   1022    630           0.978     benign         False   
9     B5__PT10   1804   1707           0.975  malignant          True   
58    SZ5_HTO6   3208   2984           0.975     benign         False   
4     B5__PT02   5615   5279           0.974  malignant          True   
56    SZ5_HTO4   3554   3335           0.972     benign          True   
55    SZ5_HTO3   3705   3386           0.971     benign         False   
54    SZ5_HTO2   4688   4176           0.970     benign         False   
53    SZ5_HTO1   2855   2654           0.970     benign         False   
57    SZ5_HTO5   3880   3663           0.969     benign         False   
36  SZ33_HTO10   1050    625           0.968     benign          True   
38   SZ33_HTO7   1613    919           0.967     benign         False   
39   SZ33_HTO8   1538    900           0.964     benign         False   
7     B5__PT07   2586   2267           0.961     benign          True   
10    B5__PT14   6271   5576           0.951  malignant          True   
8     B5__PT09   3338   1432           0.948  malignant          True   
11    B5__PT16   3015   1955           0.862  malignant          True   
23      H__SS5   1152    879           0.833  malignant          True   
5     B5__PT05   4859   3598           0.822  malignant          True   
2    B2__CTCL5    697    628           0.779     benign          True   
21      H__SS3   1403   1276           0.765  malignant          True   
16   D5__MFIVB    641    584           0.747     benign          True   
12    B5__PT18   2911   2369           0.656  malignant          True   
20      H__SS2    934    833           0.645     benign          True   
19      H__SS1   2679   2600           0.610  malignant          True   
24      H__SS6   1865   1653           0.606     benign          True   
28   SZ30_HTO8   1006    583           0.559     benign         False   
26   SZ30_HTO6   1491    776           0.530     benign         False   
18    H__MFIV1   2714   2368           0.521     benign          True   
27   SZ30_HTO7   1281    716           0.520     benign         False   
29   SZ30_HTO9   1204    663           0.498     benign          True   
25  SZ30_HTO10    875    513           0.491     benign         False   
50   SZ39_HTO4   1048    911           0.447     benign         False   
3    B2__CTCL6   1224   1170           0.443     benign          True 
... [14504 more chars truncated]
```
