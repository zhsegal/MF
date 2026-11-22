# old/23_subclonal_evolution.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/23_subclonal_evolution.ipynb` and is never run._

# 23 — Subclonal (divergent-evolution) analysis of MF malignant clones

Scales nb20's Q1b prototype (CNV subclones for the top-3 donors) to the **whole eligible cohort** and
adds the divergent-evolution characterization of **Herrmann/Iyer et al., *Cancer Discovery* 2025**
("Divergent Evolution of Malignant Subclones … in T-cell Cancer", L-CTCL/Sézary; PMC12498100).

**Malignant compartment** per patient = the `tcr_malignant_alice` cells (nb21: dominant TCR founder +
ALICE ≤1-aa β-variant family) — cells sharing one clonal TCR, as in the paper.

**Notebook layout** (reorganized — load, then compute once, then plot):
- **§0 Setup & Load** — parameters, imports, malignant compartment + cached arm-CNV matrix, eligibility.
- **§1 Calculation** — everything computed once, *no plots*: TCR founder/family sets; per-donor CNV
  subclones (`cnv_subclone`); transcriptomic MAJOR + CNV MINOR tracks; **nested subclones**
  (`nested_subclone = "{donor}_{Letter}{n}"`, MAJOR → CNV MINOR); program scores; skin-layer tropism;
  the epidermis/dermis sample menu.
- **Section 1 — Per-sample subclone structure** — pick one sample from the epidermis/dermis menu →
  per-MAJOR UMAP panels + tissue UMAP + arm-CNV heatmap (`S.plot_nested_sample`).
- **Section 2 — Subclonal functional divergence** — pick several samples → paper 5-category marker
  dot plots + per-sample statistics (within-donor Wilcoxon, Fisher epidermis/dermis tropism,
  Kruskal-Wallis program divergence). All on the `nested_subclone` axis.
- **Section 3 — Cohort overview** — CNV-subclone arm profiles, prevalence of ≥2 subclones (vs the
  paper's 84%), per-donor CNV heatmaps, per-subclone program-score heatmap.

**Runs CPU-light off the cached arm-CNV matrix** — no inferCNV recompute (all eligible donors are
cached). **Out of scope** (no data): CITE-seq surface gating, WGS/WES, drug/*S. aureus* assays, Numbat
haplotype phasing. Arm-level `infercnvpy` CNV is coarser than Numbat — focal 9p21/17p are blunted.

```python  # cell 1
# ============================================================
# §0  Parameters
# ============================================================
import os
# cap BLAS/OpenMP threads before numpy is imported (Cell 2) so a single op can't grab all cores
# and spike load average on the shared login node.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import sys
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

# ---- inputs ----
OBJ  = OUT_DIR / "skin_T_tcr_malig_v2.h5ad"                 # nb21 object; carries tcr_malignant_alice + X_mrvi_u
ARM  = OUT_DIR / "skin_T_arm_cnv_res0.5.parquet"            # cached per-cell 41-arm inferCNV matrix (nb20)
MALIG_PARQUET = OUT_DIR / "skin_T_malignancy.parquet"       # nb14 final CNV call: cnv_malig_cluster (bool)
GMT  = NB_DIR.parent / "lib1_immune.gmt"                    # hallmark + C2 gene sets (for GSEA)
# heavy fallback (only if a donor is missing from the arm cache)
GTF           = NB_DIR / "data" / "cache" / "Homo_sapiens.GRCh38.110.chr.gtf.gz"
INTEGRATED_H5 = NB_DIR / "data" / "Integrated_CTCL_skincellatlas_final_portal_tags.h5ad"

# ---- knobs ----
SEED      = 0
MIN_MAL   = 200        # min malignant cells/donor to attempt subclone detection
K_MAX     = 4          # max subclones/donor (paper median 3)
MIN_SUB   = 50         # min cells per subclone
MIN_LAYER_CELLS = 30   # min epidermis+dermis cells for a subclone to be layer-testable (§1 menu, §2 tropism)
SIL_MIN   = 0.15       # min silhouette to accept a k>1 split
MIN_ARM_DELTA = 0.03   # a split must differ by >= this on >=1 arm (CNA-scale, not state noise)
Z_THR, EPS = 2.5, 0.01 # arm gain/loss call vs benign baseline (nb20 cell-15)
BRANCH_DELTA = 0.03    # arm counts as a branch (divergent) event if subclone centroids differ by >= this
AUC_THR, SIL_THR = 0.65, 0.10   # major-vs-minor transcriptional-separation thresholds (weak in practice)
LATENT = "X_mrvi_u"    # trained MRVI latent (mrvi_joint_skin, sample-unaware) for §2 transcriptional sep
LEIDEN_RES  = 0.2      # §2 MAJOR: low-resolution Leiden on the mu embedding (few A/B/C txn clusters)
N_NEIGHBORS = 15       # §2 kNN graph on the mu embedding (per sample) for Leiden + UMAP

# ---- outputs ----
SUBCLONE_PARQUET = OUT_DIR / "subclones_v2.parquet"
SUMMARY_CSV      = OUT_DIR / "subclone_summary_v2.csv"
TRUNKBRANCH_CSV  = OUT_DIR / "subclone_trunk_branch_v2.csv"
ELIG_CSV         = OUT_DIR / "subclone_donor_eligibility_v2.csv"   # per-donor include/exclude + reason
DE_DIR           = OUT_DIR / "subclone_de"; DE_DIR.mkdir(exist_ok=True)
```

```
NB_DIR = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
```

```python  # cell 2
import importlib
import numpy as np, pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

sys.path.insert(0, str(NB_DIR))
import subclone_helpers as S
import alice_helpers as A
import skin_T_cnv_helpers as H
for _m in (S, A, H):
    importlib.reload(_m)
np.random.seed(SEED)
sc.settings.verbosity = 1
plt.rcParams["figure.dpi"] = 120
```

```
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
```

```python  # cell 3
# ============================================================
# §0  Load: malignant compartment + cached arm-CNV matrix
# ============================================================
# ⚠️ Loads ~6 GB. Run on a GPU/compute kernel — NOT the login node (fragile, shared).
if not os.environ.get("LSB_JOBID"):
    print("⚠️  No LSB_JOBID detected — you may be on the login node; loading here can crash it.")
adata = sc.read_h5ad(OBJ)
print("loaded", OBJ.name, adata.shape)
# X and the 'counts' layer are never used (Cell 19 rebuilds X from raw_counts); free them now to
# cut the resident atlas from ~6 GB (3 sparse matrices) to ~2 GB and keep peak RAM low.
adata.X = None
if "counts" in adata.layers:
    del adata.layers["counts"]

# Malignancy assignment = TCR-ALICE only (nb21: dominant founder + ALICE <=1-aa beta-variant family).
mal = adata.obs["tcr_malignant_alice"].astype(bool).to_numpy()
adata.obs["is_malig"] = mal

# Attach nb14's FINAL per-cell CNV call (cnv_malig_cluster, per-cnv-cluster GMM). Used ONLY for the
# CNV-side work: the §3 diploid-baseline hygiene + a §1 concordance QC. It does NOT define malignancy.
# (HC donors were held out of nb14's CNV query -> NaN -> False, which is correct: HC are benign.)
assert MALIG_PARQUET.exists(), f"{MALIG_PARQUET} missing — run 14_skin_T_tcr_cnv_malignancy.ipynb first"
_cnv = pd.read_parquet(MALIG_PARQUET)
adata.obs["cnv_malig_cluster"] = (_cnv["cnv_malig_cluster"].reindex(adata.obs_names)
                                  .fillna(False).astype(bool).to_numpy())
del _cnv
print("cnv_malig_cluster cells (nb14):", int(adata.obs["cnv_malig_cluster"].sum()))

# ---- sample selection (requirement a): who is eligible for subclone detection, and WHY not ----
# The loaded object is already the nb21 canonical cohort (atlas -> 34 donors); nb23 adds one gate:
# >= MIN_MAL malignant cells (KMeans needs enough cells; MIN_SUB=50, k up to K_MAX).
elig_tbl = (adata.obs.assign(_m=mal)
            .groupby("donor", observed=True)
            .agg(study=("study", "first"), disease=("disease", "first"),
                 n_cells=("_m", "size"), n_malignant=("_m", "sum"))
            .reset_index())
elig_tbl["n_malignant"] = elig_tbl["n_malignant"].astype(int)


def _exclude_reason(row):
    if str(row["disease"]) == "HC":
        return "healthy control — no malignant compartment"
    if row["n_malignant"] == 0:
        return "no ALICE-malignant cells (no dominant TCR clone)"
    if row["n_malignant"] < MIN_MAL:
        return f"<{MIN_MAL} malignant cells — too few for subclone KMeans"
    return ""


elig_tbl["reason_excluded"] = elig_tbl.apply(_exclude_reason, axis=1)
elig_tbl["eligible"] = elig_tbl["reason_excluded"].eq("")
elig_tbl = elig_tbl.sort_values(["eligible", "n_malignant"], ascending=[False, False]).reset_index(drop=True)
elig_tbl.to_csv(ELIG_CSV, index=False)

ELIG = sorted(elig_tbl.loc[elig_tbl["eligible"], "donor"])
print(f"malignant cells (TCR-ALICE): {int(mal.sum())} | cohort donors: {len(elig_tbl)} "
      f"| eligible (>= {MIN_MAL} malignant): {len(ELIG)} | excluded: {int((~elig_tbl['eligible']).sum())}")
print("upstream (nb21) reduced the atlas to these 34 donors: dropped D1__P303 (duplicate of D5__MFIVB),"
      " MF_gamma_delta / CD8_aggressive_epidermotropic_CTCL entities (non-ab-CD4), and <300-TCR-cell samples.")
print("\nexcluded from subclone detection (reason per donor):")
print(elig_tbl.loc[~elig_tbl["eligible"], ["donor", "study", "disease", "n_malignant", "reason_excluded"]]
      .to_string(index=False))
print("wrote", ELIG_CSV)

arm_df = pd.read_parquet(ARM).set_index("obs_name")
ARM_COLS = S.arm_order([c for c in arm_df.columns if c != "donor"])
arm_df = arm_df.reindex(adata.obs_names)                    # align; NaN where a cell has no arm-CNV row
miss = [d for d in ELIG if d not in set(arm_df["donor"].dropna().astype(str))]
print("arms:", len(ARM_COLS), "| eligible donors missing from arm cache:", miss or "NONE")
assert not miss, "recompute arm CNV for missing donors via H.compute_arm_cnv_per_cell (HEAVY, GPU)"
```

```
loaded skin_T_tcr_malig_v2.h5ad (242959, 49683)
cnv_malig_cluster cells (nb14): 155115
malignant cells (TCR-ALICE): 68620 | cohort donors: 34 | eligible (>= 200 malignant): 24 | excluded: 10
upstream (nb21) reduced the atlas to these 34 donors: dropped D1__P303 (duplicate of D5__MFIVB), MF_gamma_delta / CD8_aggressive_epidermotropic_CTCL entities (non-ab-CD4), and <300-TCR-cell samples.

excluded from subclone detection (reason per donor):
   donor           study    disease  n_malignant                                    reason_excluded
  H__SS4     herrera2021         MF          176 <200 malignant cells — too few for subclone KMeans
D3__P105     brunner2024 CTCL_other           66 <200 malignant cells — too few for subclone KMeans
  H__SS1     herrera2021         MF           62 <200 malignant cells — too few for subclone KMeans
  H__SS2     herrera2021         MF           22 <200 malignant cells — too few for subclone KMeans
H__MFIV1     herrera2021         MF           20 <200 malignant cells — too few for subclone KMeans
  H__SS3     herrera2021         MF           15 <200 malignant cells — too few for subclone KMeans
D1__P115 chennareddy2025         HC            0         healthy control — no malignant compartment
D1__P116 chennareddy2025         HC            0         healthy control — no malignant compartment
 D3__P94     brunner2024 CTCL_other            0   no ALICE-malignant cells (no dominant TCR clone)
  H__HC1     herrera2021         HC            0         healthy control — no malignant compartment
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/subclone_donor_eligibility_v2.csv
arms: 41 | eligible donors missing from arm cache: NONE
```

## §1 — Calculation

Everything computed once here — **no plots** (figures start in Section 1). Produces `cnv_subclone`,
the transcriptomic MAJOR + CNV MINOR tracks, the downstream **`nested_subclone`** axis, functional
program scores, skin-layer tropism, and the epidermis/dermis sample menu; persists
`subclones_v2.parquet` + `subclone_summary_v2.csv`.

```python  # cell 5
# Per-donor TCR founder + ALICE <=1-aa variant family (for the §4 reconciliation, TCR-only).
# founder = dominant-clone TRB; family = <=1-substitution connected component around the founder.
# Mirror nb21's PT35 exception (seed on the top-2 largest CD4 clones) so these founder/family sets
# match the tcr_malignant_alice call that defined the compartment.
PT35 = "Li2024_atlas__PT35"
mal_obs = adata.obs[mal].copy()
clono = A.clonotype_table(mal_obs, group="donor")           # donor, cdr3, n_cells, is_founder
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
    family_sets[d]  = A.founder_family(sub, seeds=seeds) if seeds else set()
n_var = {d: len(family_sets[d] - founder_sets[d]) for d in founder_sets}
print("donors with >=1 ALICE <=1-aa TCR-beta variant:",
      sum(v > 0 for v in n_var.values()), "/", len(n_var))
```

```
donors with >=1 ALICE <=1-aa TCR-beta variant: 7 / 30
```

```python  # cell 6
sub_label = pd.Series("", index=adata.obs_names, dtype=object)   # cell_id -> "{donor}_s{c}"
summary_rows, centroids = [], {}
for d in ELIG:
    cells = adata.obs_names[mal & (adata.obs["donor"].astype(str).values == d)]
    Ad = arm_df.reindex(cells)[ARM_COLS]
    lab, meta = S.detect_subclones(Ad, k_max=K_MAX, min_sub=MIN_SUB, seed=SEED,
                                   sil_min=SIL_MIN, min_arm_delta=MIN_ARM_DELTA)
    sub_label.loc[cells] = [f"{d}_s{c}" for c in lab]
    cen = meta["centroids"]; cen.index = [f"{d}_s{i}" for i in range(len(cen))]
    centroids[d] = cen
    # frac_cnv_confirmed = QC: fraction of this donor's TCR-malignant compartment that nb14's
    # independent CNV call (cnv_malig_cluster) also flags malignant. Cross-check only; not a filter.
    summary_rows.append({"donor": d, "n_malig": len(cells), "k": meta["k"],
                         "silhouette": meta["silhouette"], "max_arm_delta": meta["max_arm_delta"],
                         "centroid_spread": meta["centroid_spread"],
                         "frac_cnv_confirmed": round(float(adata.obs.loc[cells, "cnv_malig_cluster"].mean()), 3),
                         "sizes": ",".join(map(str, meta["sizes"])), "n_tcr_variants": n_var.get(d, 0)})

adata.obs["cnv_subclone"] = sub_label.values
# dominant subclone = largest per donor (paper: dominant ~75% of the malignant population)
vc = adata.obs.loc[sub_label.values != "", "cnv_subclone"].value_counts()
dom_sub = {d: vc[[s for s in vc.index if s.rsplit("_s", 1)[0] == d]].idxmax() for d in ELIG}
adata.obs["is_dominant_subclone"] = adata.obs["cnv_subclone"].isin(set(dom_sub.values())).values

summary = pd.DataFrame(summary_rows)
meta_cols = ["study", "disease", "disease_stage", "entity"]
summary = summary.merge(adata.obs[["donor", *meta_cols]].astype(str).drop_duplicates("donor"),
                        on="donor", how="left")
n_multi = int((summary["k"] > 1).sum())
print(f">= 2 CNV subclones: {n_multi}/{len(summary)} = {n_multi/len(summary):.0%} donors (paper: 84%)")
print("median k (multi-subclone donors):", int(summary.loc[summary['k']>1, 'k'].median() or 0), "(paper: 3)")
print(f"TCR/CNV concordance (mean frac_cnv_confirmed over eligible donors): "
      f"{summary['frac_cnv_confirmed'].mean():.2f}")
summary.sort_values("k", ascending=False)
```

```
>= 2 CNV subclones: 23/24 = 96% donors (paper: 84%)
median k (multi-subclone donors): 2 (paper: 3)
TCR/CNV concordance (mean frac_cnv_confirmed over eligible donors): 0.72

                  donor  n_malig  k  silhouette  max_arm_delta  \
1              D1__P204     1070  4      0.2033         0.0696   
4               D1__P76     5815  4      0.3267         0.0670   
12  Li2024_atlas__CTCL3     8068  4      0.2960         0.0697   
6              D3__P192     3363  3      0.2997         0.0631   
16  Li2024_atlas__CTCL7     3062  3      0.2351         0.0797   
11  Li2024_atlas__CTCL2     1824  3      0.2018         0.0762   
9             D5__MFIVB     1081  3      0.4619         0.0543   
23   Li2024_atlas__PT56     1116  3      0.1562         0.0710   
7              D3__P196      515  2      0.2955         0.0702   
5               D1__PGS     3267  2      0.3314         0.0628   
2              D1__P311     3776  2      0.1918         0.0655   
3              D1__P312      848  2      0.2844         0.0603   
0              D1__P171     5559  2      0.2522         0.0605   
13  Li2024_atlas__CTCL4      718  2      0.2427         0.0406   
10  Li2024_atlas__CTCL1     6932  2      0.2532         0.0818   
8               D3__P87     1147  2      0.2781         0.0656   
19   Li2024_atlas__PT35     2474  2      0.1692         0.0508   
14  Li2024_atlas__CTCL5     6379  2      0.3255         0.0893   
18   Li2024_atlas__PT11     1137  2      0.1870         0.0692   
17  Li2024_atlas__CTCL8     5250  2      0.3096         0.0769   
21   Li2024_atlas__PT50      369  2      0.2975         0.0726   
20   Li2024_atlas__PT47      227  2      0.2229         0.0401   
22   Li2024_atlas__PT53     1185  2      0.1576         0.0626   
15  Li2024_atlas__CTCL6     3077  1         NaN         0.0000   

    centroid_spread  frac_cnv_confirmed              sizes  n_tcr_variants  \
1            0.0869               0.711     414,52,449,155               0   
4            0.0709               0.328  3624,1662,186,343               0   
12           0.0642               0.877  260,5940,638,1230               0   
6            0.0675               0.180       467,2652,244               0   
16           0.0826               1.000       2449,440,173               0   
11           0.0838               0.770        803,771,250               0   
9            0.0612               0.154         199,814,68               1   
23           0.0742               0.697         92,509,515               0   
7            0.0709               1.000             432,83               2   
5            0.0891               0.951           2534,733               0   
2            0.0657               0.678          2713,1063               0   
3            0.0855               0.653            599,249               0   
0            0.0746               0.877          1574,3985               0   
13           0.0808               0.170            168,550               0   
10           0.0823               0.863          4312,2620               0   
8            0.0658               1.000           129,1018               0   
19           0.0819               0.751          1333,1141               0   
14           0.0894               0.960          5011,1368               0   
18           0.0917               0.828            434,703               2   
17           0.1102               0.743          3343,1907               1   
21           0.1004               0.610            177,192              12   
20           0.0661               0.850             171,56               2   
22           0.0650               0.885            679,506               9   
15           0.0000               0.864               3077               0   

              study     disease disease_stage                      entity  
1   chennareddy2025          MF      advanced                  MF_classic  
4   chennareddy2025          MF      advanced                  MF_classic
... [1672 more chars truncated]
```

```python  # cell 7
# §2 — per-sample transcriptomic MAJOR (Leiden on the MRVI mu embedding) + CNV MINOR (§1) tracks,
# the substrate for the nested subclones. Per-donor loop lifted to S.major_minor_tracks.
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
print(f"MAJOR = Leiden (res={LEIDEN_RES}) on {_lat}; MINOR = §1 CNV subclones "
      f"| clones with both tracks split: {n_both}/{len(mm)}")
```

```
/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/subclone_helpers.py:179: FutureWarning: In the future, the default backend for leiden will be igraph instead of leidenalg.

 To achieve the future defaults please pass: flavor="igraph" and n_iterations=2.  directed must also be False to work with igraph's implementation.
  sc.tl.leiden(sub, resolution=leiden_res, random_state=seed, key_added="_leiden")

MAJOR = Leiden (res=0.2) on X_mrvi_u; MINOR = §1 CNV subclones | clones with both tracks split: 23/24
```

```python  # cell 8
# cohort-wide NESTED subclones (transcriptomic MAJOR -> CNV MINOR within each major).
# nested_subclone = "{donor}_{Letter}{n}" (or "{donor}_{Letter}" if a major has no CNV substructure).
# This is the subclone axis used in Sections 1-2.
nested, split_summary = S.nested_subclone_labels(
    umap_by_donor, arm_df, ARM_COLS, ELIG,
    k_max=K_MAX, min_sub=MIN_SUB, seed=SEED, sil_min=SIL_MIN, min_arm_delta=MIN_ARM_DELTA)
adata.obs["nested_subclone"] = nested.reindex(adata.obs_names).fillna("").astype(str).values
n_split = int((split_summary["k"] > 1).sum())
print(f"nested subclones: {nested.nunique()} across {len(ELIG)} eligible donors "
      f"| majors that split further on CNV: {n_split}/{len(split_summary)}")
```

```
nested subclones: 150 across 24 eligible donors | majors that split further on CNV: 55/80
```

```python  # cell 9
# malignant cells, log-normalised from raw counts (scoring + DE) + nested-subclone bookkeeping + panels.
mal_ad = adata[adata.obs["cnv_subclone"].astype(str).values != ""].copy()
mal_ad.X = mal_ad.layers["raw_counts"].copy()
sc.pp.normalize_total(mal_ad, target_sum=1e4); sc.pp.log1p(mal_ad)
mal_ad.obs["nested_subclone"] = adata.obs.loc[mal_ad.obs_names, "nested_subclone"].astype(str).values

# dominant nested subclone / donor = largest; donors with >=2 nested subclones = multi_nested.
vc = mal_ad.obs["nested_subclone"].value_counts(); vc = vc[vc.index != ""]
by_donor = {}
for s, n in vc.items():
    by_donor.setdefault(s.rsplit("_", 1)[0], []).append((s, n))
dom_nested = {d: max(v, key=lambda t: t[1])[0] for d, v in by_donor.items()}
multi_nested = sorted(d for d, v in by_donor.items() if len(v) >= 2)
mal_ad.obs["is_dominant_nested"] = mal_ad.obs["nested_subclone"].isin(set(dom_nested.values())).values

# paper functional-marker panels (5 categories), filtered to genes present in the object.
panels = S.present_panels(mal_ad, S.PAPER_PANELS)
print("malignant cells for DE/scoring:", mal_ad.shape, "| donors:", mal_ad.obs["donor"].nunique())
print(f"nested subclones: {vc.shape[0]} | donors with >=2 nested subclones: {len(multi_nested)}")
print("panel coverage:",
      {cat: f"{len(panels.get(cat, []))}/{len(S.PAPER_PANELS[cat])}" for cat in S.PAPER_PANELS})
```

```
WARNING: adata.X seems to be already log-transformed.
malignant cells for DE/scoring: (68259, 49683) | donors: 24
nested subclones: 150 | donors with >=2 nested subclones: 24
panel coverage: {'homing': '15/15', 'spatial': '8/8', 'cytokine': '10/10', 'metabolism': '15/15', 'signaling': '16/16', 'protease': '12/12'}
```

```python  # cell 10
# functional-program scores + skin-layer tropism (nested axis) + epidermis/dermis sample menu, then persist.
added = S.score_programs(mal_ad, prefix="prog_", cell_cycle=True, seed=SEED)
prog_cols = [c for c in added if c.startswith("prog_")]

# skin layer per cell + per-subclone tropism (epidermal / dermal / mixed) on the nested axis.
layer = mal_ad.obs["tissue"].astype(str).str.lower()
is_epi = layer.str.contains("epiderm"); is_derm = layer.str.contains("derm") & ~is_epi
mal_ad.obs["skin_layer"] = np.where(is_epi, "epidermis", np.where(is_derm, "dermis", "other"))
trop_tbl, cell_trop = S.classify_subclone_tropism(mal_ad.obs, "nested_subclone", min_cells=MIN_LAYER_CELLS)
mal_ad.obs["tropism"] = cell_trop.values

# menu of samples with epidermis/dermis separation (both layers + >=2 nested subclones) for Sections 1-2.
MENU = S.layer_split_donors(mal_ad.obs, "nested_subclone", min_cells=MIN_LAYER_CELLS)

# persist declared §0 outputs.
adata.obs[["donor", "cnv_subclone", "nested_subclone", "is_dominant_subclone"]].to_parquet(SUBCLONE_PARQUET)
summary.to_csv(SUMMARY_CSV, index=False)

print(f"tropism classes: {trop_tbl['tropism'].value_counts().to_dict()}")
print(f"eligible donors: {len(ELIG)} | donors with >=2 nested: {len(multi_nested)} "
      f"| samples with epidermis/dermis separation: {len(MENU)}")
print("wrote", SUBCLONE_PARQUET.name, "+", SUMMARY_CSV.name)
```

```
tropism classes: {'unclassified': 87, 'epidermal': 31, 'dermal': 20, 'mixed': 12}
eligible donors: 24 | donors with >=2 nested: 24 | samples with epidermis/dermis separation: 8
wrote subclones_v2.parquet + subclone_summary_v2.csv
```

```python  # cell 11
# Export gene signatures for downstream SPATIAL scoring (nb25 DestVI-deconvolved Visium).
# {signature_name -> [genes]}, three kinds:
#   subclone__{donor}_{X} = within-donor Wilcoxon markers per nested subclone (DONOR-SPECIFIC —
#                           only meaningful within the same patient; kept for completeness).
#   malignant_overall     = malignant-vs-benign T markers (cross-patient overall tumour signal).
#   program__* / panel__*  = the paper's functional-divergence axes (CROSS-patient; the meaningful
#                           "subclonal divergence" annotation for an unrelated Visium cohort).
import json

SIGN_JSON = OUT_DIR / "subclone_signatures_v2.json"
TOP_SUB, TOP_MAL = 30, 50
signatures = {}

# --- per-nested-subclone (within-donor Wilcoxon; all multi-subclone donors) ---
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

# --- overall malignant signature: malignant vs benign T (balanced subsample for speed) ---
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
del mvb

# --- curated functional-divergence axes (cross-patient; verbatim, deduped) ---
for cat, genes in S.PAPER_PANELS.items():
    signatures[f"panel__{cat}"] = list(dict.fromkeys(genes))
for prog, genes in S.PROGRAM_SETS.items():
    signatures[f"program__{prog}"] = list(dict.fromkeys(genes))

SIGN_JSON.write_text(json.dumps(signatures, indent=1))
print(f"wrote {SIGN_JSON.name}: {len(signatures)} signatures "
      f"({sum(k.startswith('subclone__') for k in signatures)} per-subclone, "
      f"{sum(k.startswith('program__') for k in signatures)} programs, "
      f"{sum(k.startswith('panel__') for k in signatures)} panels) "
      f"| median genes/sig: {int(np.median([len(v) for v in signatures.values()]))}")
```

```
WARNING: adata.X seems to be already log-transformed.
wrote subclone_signatures_v2.json: 161 signatures (144 per-subclone, 10 programs, 6 panels) | median genes/sig: 30
```

## Section 1 — Per-sample subclone structure (UMAP + inferCNV)

The first cell lists the **selectable samples** (donors with epidermis/dermis separation). Copy one
`donor` id into `SAMPLE` in the next cell; `S.plot_nested_sample` then draws its per-MAJOR UMAP panels
(nested minors highlighted), the tissue (cell-location) UMAP, and the per-cell arm-CNV heatmap grouped
by the nested subclone label.

```python  # cell 13
# selectable samples — set SAMPLE (Section 1) / SAMPLES (Section 2) to any 'donor' below.
# columns: n_subclones = # nested subclones | n_epi / n_derm = layer-resolved cells | subclones = short labels
print(f"{len(MENU)} samples with epidermis/dermis separation:\n")
print(MENU.to_string(index=False))
```

```
8 samples with epidermis/dermis separation:

              donor  n_subclones  n_epi  n_derm                              subclones
Li2024_atlas__CTCL1           13   2173    4759 A1,A2,B1,B2,B3,B4,C1,C2,D1,D2,E1,E2,E3
Li2024_atlas__CTCL7           10   2812     250          A1,A2,A3,A4,B1,B2,B3,B4,C1,C2
Li2024_atlas__CTCL2           10   1121     703           A1,A2,B1,B2,B3,C1,C2,D1,D2,E
Li2024_atlas__CTCL5            9   4407    1972             A1,A2,B1,B2,C1,C2,D1,D2,D3
Li2024_atlas__CTCL3            9   2825    5243              A1,A2,A3,B1,B2,B3,C1,C2,D
Li2024_atlas__CTCL8            7    527    4723                    A1,A2,B1,B2,C1,C2,D
Li2024_atlas__CTCL4            3    675      43                                  A,B,C
Li2024_atlas__CTCL6            2   2822     255                                    A,B
```

```python  # cell 14
SAMPLE = "Li2024_atlas__CTCL8"      # <- pick ONE 'donor' from the list above
if SAMPLE not in set(MENU["donor"]):
    SAMPLE = MENU["donor"].iloc[0] if len(MENU) else ELIG[0]
print("selected:", SAMPLE)
S.plot_nested_sample(adata, SAMPLE, umap_by_donor, arm_df, ARM_COLS, mal, fig_dir=FIG_DIR)
```

```
selected: Li2024_atlas__CTCL8

<Figure size 1920x768 with 6 Axes>
```

## Section 2 — Subclonal functional divergence (dot plots + per-sample stats)

Pick **several** samples from the menu. Following the paper (Cancer Discovery 2025, Fig 4–5), markers
are grouped into five functional categories — **homing**, **spatial** (epidermotropism vs egress),
**cytokine**, **metabolism** (OXPHOS / glycolysis), **signaling** — and subclonal divergence is read
from descriptive **dot plots** (colour = mean expression, size = % expressing), then tested per sample.

Statistics use what the data supports (no biological replicates within a patient): within-donor
single-cell **Wilcoxon**, **Fisher** epidermis/dermis tropism, and **Kruskal-Wallis** program
divergence — all on the `nested_subclone` axis.

```python  # cell 16
# pick MULTIPLE samples from the epidermis/dermis menu for the functional comparison:
print(MENU.to_string(index=False))
SAMPLES = ["Li2024_atlas__CTCL8"]      # <- one or more donors from the menu above
SAMPLES = [d for d in SAMPLES if d in set(mal_ad.obs["donor"].astype(str))]
n_panel_genes = sum(len(v) for v in panels.values())

# (1) per-sample descriptive marker dot plot across each sample's nested subclones (paper Fig 4 style).
#     colour = mean expression (standard-scaled per gene), dot size = % of cells expressing.
for smp in SAMPLES:
    sub = mal_ad[mal_ad.obs["donor"].astype(str).values == smp].copy()
    subs = sorted(s for s in sub.obs["nested_subclone"].unique() if s)
    sub = sub[sub.obs["nested_subclone"].isin(subs)].copy()
    sub.obs["nested_subclone"] = pd.Categorical(sub.obs["nested_subclone"].astype(str), categories=subs)
    S.subclone_dotplot(sub, "nested_subclone", panels,
                       title=f"{smp}: subclonal marker expression",
                       save=FIG_DIR / f"subclone_dotplot_{smp}.png",
                       figsize=(min(26, max(8, 0.3 * n_panel_genes)), 1.0 + 0.45 * len(subs)))

# (2) cohort tropism dot plot: epidermal vs mixed vs dermal nested subclones.
groups = [g for g in ["epidermal", "mixed", "dermal"] if (mal_ad.obs["tropism"] == g).any()]
subt = mal_ad[mal_ad.obs["tropism"].isin(groups)].copy()
subt.obs["tropism"] = pd.Categorical(subt.obs["tropism"].astype(str), categories=groups)
S.subclone_dotplot(subt, "tropism", panels,
                   title="Subclone tropism: epidermal vs mixed vs dermal",
                   save=FIG_DIR / "subclone_tropism_dotplot.png",
                   figsize=(min(26, max(8, 0.3 * n_panel_genes)), 1.2 + 0.6 * len(groups)))
```

```
donor  n_subclones  n_epi  n_derm                              subclones
Li2024_atlas__CTCL1           13   2173    4759 A1,A2,B1,B2,B3,B4,C1,C2,D1,D2,E1,E2,E3
Li2024_atlas__CTCL7           10   2812     250          A1,A2,A3,A4,B1,B2,B3,B4,C1,C2
Li2024_atlas__CTCL2           10   1121     703           A1,A2,B1,B2,B3,C1,C2,D1,D2,E
Li2024_atlas__CTCL5            9   4407    1972             A1,A2,B1,B2,C1,C2,D1,D2,D3
Li2024_atlas__CTCL3            9   2825    5243              A1,A2,A3,B1,B2,B3,C1,C2,D
Li2024_atlas__CTCL8            7    527    4723                    A1,A2,B1,B2,C1,C2,D
Li2024_atlas__CTCL4            3    675      43                                  A,B,C
Li2024_atlas__CTCL6            2   2822     255                                    A,B

<Figure size 2736x498 with 6 Axes>
<Figure size 2736x360 with 6 Axes>
```

```python  # cell 17
# per-sample statistics for the chosen SAMPLES (all on the nested_subclone axis).
from scipy.stats import fisher_exact, kruskal
stat_samples = [d for d in SAMPLES if d in multi_nested]
print("samples with >=2 nested subclones (testable):", stat_samples or "NONE")

# --- (a) within-donor single-cell Wilcoxon DE (panel genes + full transcriptome; CSV per donor) ---
panel_genes = sorted({g for v in panels.values() for g in v})
gene2cat = {}
for cat, v in panels.items():
    for g in v:
        gene2cat.setdefault(g, cat)   # first category wins (homing before spatial on shared genes)
panel_hits = []
for d in stat_samples:
    sub = mal_ad[mal_ad.obs["donor"].astype(str).values == d].copy()
    if sub.obs["nested_subclone"].nunique() < 2:
        continue
    de = S.subclone_markers_wilcoxon(sub, "nested_subclone", n_genes=sub.n_vars)
    de.to_csv(DE_DIR / f"subclone_de_nested_{d}.csv", index=False)
    pg = de[de["gene"].isin(panel_genes)].copy(); pg.insert(0, "donor", d)
    panel_hits.append(pg)
panel_de = pd.concat(panel_hits, ignore_index=True) if panel_hits else pd.DataFrame()
if len(panel_de):
    panel_de["category"] = panel_de["gene"].map(gene2cat)
    panel_de.to_csv(OUT_DIR / "subclone_panel_de_nested.csv", index=False)
    sig = panel_de.query("pval_adj < 0.05 & log2fc > 0")
    print(f"\nWilcoxon panel-gene up-calls in a subclone (padj<0.05): {len(sig)} across {sig['donor'].nunique()} donors")
    print(sig.reindex(sig["score"].abs().sort_values(ascending=False).index)
          .head(15)[["donor", "subclone", "gene", "category", "log2fc", "pval_adj"]].round(3).to_string(index=False))
else:
    print("no chosen sample has >=2 nested subclones for Wilcoxon")

# --- (b) epidermis/dermis tropism per nested subclone (Fisher exact) ---
trop_rows = []
for d in stat_samples:
    o = mal_ad.obs[(mal_ad.obs["donor"].astype(str).values == d) &
                   mal_ad.obs["skin_layer"].isin(["epidermis", "dermis"]) &
                   (mal_ad.obs["nested_subclone"].astype(str) != "")]
    if o["skin_layer"].nunique() < 2 or o["nested_subclone"].nunique() < 2:
        continue
    tot_epi = int((o["skin_layer"] == "epidermis").sum()); tot_derm = int((o["skin_layer"] == "dermis").sum())
    for s, g in o.groupby("nested_subclone"):
        a = int((g["skin_layer"] == "epidermis").sum()); b = int((g["skin_layer"] == "dermis").sum())
        odds, p = fisher_exact([[a, b], [tot_epi - a, tot_derm - b]])
        trop_rows.append({"donor": d, "subclone": s, "n_epi": a, "n_derm": b,
                          "epi_frac": a / max(1, a + b), "fisher_or": odds, "fisher_p": p})
tropism = pd.DataFrame(trop_rows)
if len(tropism):
    print(f"\nFisher tropism — subclones with layer skew (p<0.05): {int((tropism['fisher_p']<0.05).sum())}/{len(tropism)}")
    print(tropism.sort_values("fisher_p").head(10).round(3).to_string(index=False))
else:
    print("\nno chosen sample has both skin layers across >=2 nested subclones")

# --- (c) functional-program divergence across subclones (Kruskal-Wallis) ---
div_rows = []
for d in stat_samples:
    o = mal_ad.obs[(mal_ad.obs["donor"].astype(str).values == d) &
                   (mal_ad.obs["nested_subclone"].astype(str) != "")]
    if o["nested_subclone"].nunique() < 2:
        continue
    for c in prog_cols:
        vals = [g[c].dropna().values for _, g in o.groupby("nested_subclone")]
        if all(len(v) > 5 for v in vals):
            try: p = kruskal(*vals)[1]
            except Exception: p = np.nan
            div_rows.append({"donor": d, "program": c.replace("prog_", ""), "kruskal_p": p})
div = pd.DataFrame(div_rows)
if len(div):
    sig = div[div["kruskal_p"] < 0.05]
    print(f"\nKruskal program divergence (donor x program, p<0.05): {len(sig)}/{len(div)}")
    print((div.assign(sig=div["kruskal_p"] < 0.05).groupby("program")["sig"].mean()
           .sort_values(ascending=False)).round(2).to_string())
```

```
samples with >=2 nested subclones (testable): ['Li2024_atlas__PT35']

Wilcoxon panel-gene up-calls in a subclone (padj<0.05): 34 across 1 donors
             donor              subclone    gene   category  log2fc  pval_adj
Li2024_atlas__PT35  Li2024_atlas__PT35_B    ENO1 metabolism   1.429       0.0
Li2024_atlas__PT35  Li2024_atlas__PT35_B     PKM metabolism   0.954       0.0
Li2024_atlas__PT35  Li2024_atlas__PT35_B   COX6C metabolism   0.932       0.0
Li2024_atlas__PT35  Li2024_atlas__PT35_B ATP5MC1 metabolism   1.329       0.0
Li2024_atlas__PT35  Li2024_atlas__PT35_B    CYCS metabolism   0.810       0.0
Li2024_atlas__PT35  Li2024_atlas__PT35_B    PGK1 metabolism   0.799       0.0
Li2024_atlas__PT35 Li2024_atlas__PT35_A1    IL22   cytokine   0.789       0.0
Li2024_atlas__PT35 Li2024_atlas__PT35_A1    KLF2     homing   0.956       0.0
Li2024_atlas__PT35  Li2024_atlas__PT35_B ATP5F1A metabolism   0.738       0.0
Li2024_atlas__PT35 Li2024_atlas__PT35_A2    CCR7     homing   0.543       0.0
Li2024_atlas__PT35  Li2024_atlas__PT35_B   TIMP1   protease   0.742       0.0
Li2024_atlas__PT35  Li2024_atlas__PT35_B  NDUFB1 metabolism   0.705       0.0
Li2024_atlas__PT35 Li2024_atlas__PT35_A2    KLF2     homing   0.704       0.0
Li2024_atlas__PT35 Li2024_atlas__PT35_C1    ENO1 metabolism   0.868       0.0
Li2024_atlas__PT35 Li2024_atlas__PT35_A1 TNFAIP3  signaling   0.411       0.0

no chosen sample has both skin layers across >=2 nested subclones

Kruskal program divergence (donor x program, p<0.05): 10/10
program
cytokine         1.0
glycolysis       1.0
oxphos           1.0
prolif           1.0
protease         1.0
recirc_homing    1.0
skin_homing      1.0
th1              1.0
th17             1.0
th2              1.0
```

## Section 3 — Cohort overview

Cohort-wide summaries: CNV-subclone arm profiles and the prevalence of ≥2 subclones (vs the paper's
84%) on the `cnv_subclone` detection axis, plus the per-subclone functional program-score heatmap on
the `nested_subclone` axis.

```python  # cell 19
# CNV-subclone arm profiles (cohort) + prevalence of >=2 subclones per donor.
S.plot_cohort_arm_profiles(centroids, ELIG, ARM_COLS, save=FIG_DIR / "subclone_arm_profiles_cohort.png")
S.plot_subclone_prevalence(summary, k_max=K_MAX, n_multi=n_multi, save=FIG_DIR / "subclone_prevalence.png")
```

```
<Figure size 1320x974.4 with 2 Axes>
<Figure size 600x360 with 1 Axes>
```

```python  # cell 20
# per-subclone functional program-score heatmap (nested axis).
S.plot_program_heatmap(mal_ad.obs, "nested_subclone", prog_cols,
                       donor_order=multi_nested, save=FIG_DIR / "subclone_programs.png")
```

```
<Figure size 600x2880 with 2 Axes>
```
