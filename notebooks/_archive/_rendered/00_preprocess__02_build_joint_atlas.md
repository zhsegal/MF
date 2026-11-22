# 00_preprocess/02_build_joint_atlas.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/00_preprocess/02_build_joint_atlas.ipynb` and is never run._

# Build the joint CTCL/MF/SS multi-tissue atlas + TCR clone table

Joins the **Li 2024 skin atlas** (`CTCL_all`, 419k cells) with the standalone GEO cohorts
(`D1 D3 H D5 D6 B1 B2 B4`) into one full-compartment atlas, with a harmonized `obs` schema and
a cross-platform **TCR clone table** (CDR3-based, TRB-primary).

**Excluded:** `B3` (453-gene targeted panel + no CDR3, only a paired-chain boolean), `D2` (no data yet).

Schema + crosswalk: [`data/atlas_joint/metadata_schema.md`](data/atlas_joint/metadata_schema.md).
All logic lives in `atlas_join_helpers.py`.

**Run order.** Steps 1-4 are GPU-kernel jobs on `neural_nmf_env`; heavy outputs are cached to disk
(re-runs skip). Set the `FORCE_*` flags to rebuild.

| Step | Output |
|---|---|
| 1 | `data/<LABEL>/processed/standardized.h5ad` |
| 2 | `data/atlas_joint/tcr_clones.parquet` |
| 3 | `data/atlas_joint/joint_raw.h5ad` |
| 4 | `data/atlas_joint/joint_integrated.h5ad` |
| 5 | `figures/joint_umap_*.png` + `data/atlas_joint/README.md` |

```python  # cell 1
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad


def _resolve_nb_dir():
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(start)


NB_DIR = _resolve_nb_dir()
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR / "helpers"))
import importlib
import atlas_join_helpers as H
importlib.reload(H)   # pick up edits without restarting the kernel

SEED = 0
np.random.seed(SEED)
sc.settings.verbosity = 1
OUT = NB_DIR / "data" / "atlas_joint"
OUT.mkdir(exist_ok=True)
REG = H.dataset_registry(NB_DIR)
EXPR = [k for k, v in REG.items() if v.get("in_expression")]
print("NB_DIR:", NB_DIR)
print("expression datasets:", EXPR)
```

```
NB_DIR: /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
expression datasets: ['Li2024_atlas', 'D1', 'D3', 'H', 'D5', 'D6', 'B1', 'B2', 'B4', 'B5']
```

## Step 1 - per-dataset standardized h5ads

Each source -> one AnnData with exactly `CANON_OBS` columns, `X` = raw counts, gene-symbol
`var_names`, and a `cell_id` consistent with the TCR table. Reuses the existing `concat.h5ad`
for D1/D3; rebuilds H (ECCITE, all 14 samples) and D5/D6/B1/B2/B4 from `raw/`; maps the atlas.

```python  # cell 3
# Per-paper QC (QC_CONFIG) is applied inside build_standardized: per-cell
# min/max-genes + %mito cutoffs, min_cells_per_gene=3, per-sample Scrublet.
# li24 is prefiltered (atlas already QC'd) -> metrics only.
#
# CACHING: each standardized.h5ad is written as soon as its dataset finishes, and
# FORCE_STD=False reuses it -> a rerun resumes after a crash (only redoes missing
# datasets), instead of rebuilding all 9 (~45 min). Set FORCE_STD=True only to
# force a full rebuild. The loop is crash-resilient: a failing dataset is recorded
# and skipped so the rest still complete.
FORCE_STD = False
RUN_SCRUBLET = True
std_paths, failed = {}, {}
for label in EXPR:
    try:
        std_paths[label] = H.build_standardized(label, NB_DIR, force=FORCE_STD,
                                                run_scrublet=RUN_SCRUBLET)
    except Exception as e:  # keep the already-built datasets; report the failure
        failed[label] = repr(e)
        print(f"  [{label}] FAILED: {e!r}")
if failed:
    print("\nFAILED datasets (rerun this cell after fixing):", list(failed))
print("built:", list(std_paths))
std_paths
```

```
[Li2024_atlas] cached (QC) -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/atlas_standardized.h5ad
[D1] cached (QC) -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/D1_chennareddy2025/processed/standardized.h5ad
[D3] cached (QC) -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/D3_brunner2024/processed/standardized.h5ad
[H] cached (QC) -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/H_herrera2021/processed/standardized.h5ad
[D5] cached (QC) -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/D5_rindler2021frontimm_multitissue/processed/standardized.h5ad
[D6] cached (QC) -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/D6_gaydosik2019_skin/processed/standardized.h5ad
[B1] cached (QC) -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/B1_borcherding2019_blood/processed/standardized.h5ad
[B2] cached (QC) -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/B2_borcherding2023_blood/processed/standardized.h5ad
[B4] cached (QC) -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/B4_geskin2026_dupilumab_blood/processed/standardized.h5ad
  [B5] PT01_Blood: GEX (3732, 36601)  ADT 127
  [B5] PT02_Blood: GEX (6159, 36601)  ADT 127
  [B5] PT04_Blood: GEX (801, 36601)  ADT 127
  [B5] PT05_Blood: GEX (5676, 36601)  ADT 127
  [B5] PT06_Blood: GEX (6478, 36601)  ADT 127
  [B5] PT07_Blood: GEX (2947, 36601)  ADT 127
  [B5] PT08_Day0_Blood: GEX (2004, 36601)  ADT 127
  [B5] PT08_Day266_Blood: GEX (3100, 36601)  ADT 127
  [B5] PT09_Skin_Erythroderma_Dermis_A: GEX (7175, 36601)  ADT 127
  [B5] PT09_Skin_Plaque_Dermis_B: GEX (6527, 36601)  ADT 127
  [B5] PT09_Skin_Erythroderma_Epidermis_A: GEX (7181, 36601)  ADT 127
  [B5] PT09_Skin_Plaque_Epidermis_B: GEX (5415, 36601)  ADT 127
  [B5] PT09_Blood: GEX (4765, 36601)  ADT 127
  [B5] PT10_Blood: GEX (2128, 36601)  ADT 127
  [B5] PT13_Day294_Blood: GEX (638, 36601)  ADT 102
  [B5] PT14_Day0_Blood: GEX (4074, 36601)  ADT 97
  [B5] PT14_Day43_Blood: GEX (4058, 36601)  ADT 97
  [B5] PT15_Blood: GEX (638, 36601)  ADT 101
  [B5] PT16_Blood: GEX (5118, 36601)  ADT 127
  [B5] PT18_Blood: GEX (4611, 36601)  ADT 127
  [B5] PT21_Day0_Blood: GEX (2714, 36601)  ADT 127
  [B5] PT21_Day126_Blood: GEX (3711, 36601)  ADT 127
  [B5] PT22_Skin_Erythroderma_Dermis_A.1: GEX (2677, 36601)  ADT 126
  [B5] PT22_Skin_Erythroderma_Dermis_A.2: GEX (2434, 36601)  ADT 126
  [B5] PT22_Skin_Erythroderma_Dermis_B.1: GEX (1635, 36601)  ADT 126
  [B5] PT22_Skin_Erythroderma_Dermis_B.2: GEX (1964, 36601)  ADT 126
  [B5] PT22_Skin_Erythroderma_Epidermis_A.1: GEX (2384, 36601)  ADT 126
  [B5] PT22_Skin_Erythroderma_Epidermis_A.2: GEX (2226, 36601)  ADT 126
  [B5] PT22_Skin_Erythroderma_Epidermis_B.1: GEX (791, 36601)  ADT 126
  [B5] PT22_Skin_Erythroderma_Epidermis_B.2: GEX (1021, 36601)  ADT 126
  [B5] PT22_Blood: GEX (2944, 36601)  ADT 126
  [B5] PT23_Skin_Erythroderma_Dermis_A.1: GEX (1924, 36601)  ADT 126
  [B5] PT23_Skin_Erythroderma_Dermis_A.2: GEX (2036, 36601)  ADT 126
  [B5] PT23_Skin_Erythroderma_Dermis_B.1: GEX (4862, 36601)  ADT 126
  [B5] PT23_Skin_Erythroderma_Dermis_B.2: GEX (3993, 36601)  ADT 126
  [B5] PT23_Skin_Erythroderma_Epidermis_A.1: GEX (357, 36601)  ADT 126
  [B5] PT23_Skin_Erythroderma_Epidermis_A.2: GEX (337, 36601)  ADT 126
  [B5] PT23_Skin_Erythroderma_Epidermis_B.1: GEX (812, 36601)  ADT 126
  [B5] PT23_Skin_Erythroderma_Epidermis_B.2: GEX (1187, 36601)  ADT 126
  [B5] PT23_Blood: GEX (8364, 36601)  ADT 126
  [B5] FAILED: ValueError('shape mismatch: value array of shape (638,102) could not be broadcast to indexing result of shape (638,127)')

FAILED datasets (rerun this cell after fixing): ['B5']
built: ['Li2024_atlas', 'D1', 'D3', 'H', 'D5', 'D6', 'B1', 'B2', 'B4']

{'Li2024_atlas': PosixPath('/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/atlas_standardized.h5ad'),
 'D1': Posix
... [1106 more chars truncated]
```

### QC audit

Per-dataset cells in → kept (from `uns['qc']`), with the threshold source. Lets you confirm the
filtering matches each paper and see how much `geskin26`/`gaydosik19` shrink toward their reported
counts. (For Scrublet auto-thresholding, `%pip install scikit-image`; otherwise a fixed 0.25 cutoff
is used.)

```python  # cell 5
rows = []
for label, p in std_paths.items():
    u = sc.read_h5ad(p, backed="r").uns.get("qc", {})
    n_in, n_kept = u.get("n_in", 0), u.get("n_kept", 0)
    rows.append(dict(dataset=H.DATASET_NAME.get(label, label), n_in=n_in, n_kept=n_kept,
                     pct_kept=round(100 * n_kept / max(1, n_in), 1),
                     doublets=u.get("n_doublet", 0), source=u.get("source", "")))
qc_audit = pd.DataFrame(rows).set_index("dataset")
print(qc_audit.to_string())
qc_audit
```

```
n_in  n_kept  pct_kept  doublets                                                                                    source
dataset                                                                                                                                    
li24           419579  419579     100.0         0  Li 2024 NatImmunol (atlas already QC'd: >=400 genes, >=1000 counts, <20% mito, Scrublet)
chennareddy25  294318  191147      64.9      2589                                               proxy brunner24 (Vienna; Appendix S1 gated)
brunner24      137409   63436      46.2      1187                                 Brunner 2024 JACI (>500 genes, <10% mito, scran doublets)
herrera21       37819   32002      84.6       167                                default (Herrera 2021 reports only Scrublet0.2 + HTODemux)
rindler21       12754    8612      67.5       122                                     Rindler 2021 FrontImmunol (200-4000 genes, <12% mito)
gaydosik19      44842   39260      87.6       215                                         default skin (Gaydosik 2019 CCR supplement gated)
borcherding19   15149   14253      94.1       381                                           Borcherding 2019 CCR (200-3500 genes, <9% mito)
borcherding23   46677   42642      91.4       578                                       proxy borcherding19 (same lab; 2023 Methods sparse)
geskin26       270992  260455      96.1      5198                              Geskin 2026 CIR (>=200 genes; mito was regressed -> cap 20%)

                 n_in  n_kept  pct_kept  doublets  \
dataset                                             
li24           419579  419579     100.0         0   
chennareddy25  294318  191147      64.9      2589   
brunner24      137409   63436      46.2      1187   
herrera21       37819   32002      84.6       167   
rindler21       12754    8612      67.5       122   
gaydosik19      44842   39260      87.6       215   
borcherding19   15149   14253      94.1       381   
borcherding23   46677   42642      91.4       578   
geskin26       270992  260455      96.1      5198   

                                                          source  
dataset                                                           
li24           Li 2024 NatImmunol (atlas already QC'd: >=400 ...  
chennareddy25        proxy brunner24 (Vienna; Appendix S1 gated)  
brunner24      Brunner 2024 JACI (>500 genes, <10% mito, scra...  
herrera21      default (Herrera 2021 reports only Scrublet0.2...  
rindler21      Rindler 2021 FrontImmunol (200-4000 genes, <12...  
gaydosik19     default skin (Gaydosik 2019 CCR supplement gated)  
borcherding19    Borcherding 2019 CCR (200-3500 genes, <9% mito)  
borcherding23  proxy borcherding19 (same lab; 2023 Methods sp...  
geskin26       Geskin 2026 CIR (>=200 genes; mito was regress...
```

## Step 2 - unified TCR clone table

CDR3 exact key, **TRB-primary** (TRA dropout does not split a clone). 10x VDJ: D1/D3/D5/B2/B4;
ECCITE membership matrix: H. Per-sample dominant clone (>=5% of VDJ cells **and** >=2x the 2nd
clone) -> `is_malignant` (generalizes `07_d1`). B1/D6 have no TCR; B3 has no CDR3.

```python  # cell 7
FORCE_TCR = True   # B5(buus25) add: force so the new cohort enters tcr_clones.parquet
tcr_parquet = H.build_tcr_table(NB_DIR, force=FORCE_TCR)
tcr = pd.read_parquet(tcr_parquet)
tcr.groupby("dataset", observed=True).agg(
    cells=("cell_id", "size"), clones=("clone_id", "nunique"),
    malignant=("is_malignant", "sum"))
```

```
[D1] TCR cells=96578  samples=22  malignant=47242
[D3] TCR cells=27070  samples=11  malignant=13566
[D5] TCR cells=3183  samples=3  malignant=1504
[B2] TCR cells=41526  samples=5  malignant=37008
[B4] TCR cells=102890  samples=6  malignant=75291
[B5] TCR cells=71648  samples=40  malignant=46399
[H] TCR cells=17899  samples=14  malignant=6794
[TCR] wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/tcr_clones.parquet  cells=360794  clones=84464

          cells  clones  malignant
dataset                           
B2        41526    3342      37008
B4       102890   17385      75291
B5        71648   12836      46399
D1        96578   34914      47242
D3        27070    6614      13566
D5         3183    1039       1504
H         17899    8334       6794
```

### (optional) scirpy fuzzy clonotype merging

The exact TRB-CDR3 key above is the production path. scirpy can additionally merge 1-aa CDR3
variants into clonotype clusters. Opt-in (needs `%pip install scirpy`); validate against your
installed scirpy version before trusting the refined ids. Leave OFF to use the exact key.

```python  # cell 9
USE_SCIRPY = False
if USE_SCIRPY:
    try:
        import scirpy as ir
        print("scirpy", ir.__version__, "- refine clone_id per (dataset, donor) with ir_dist/define_clonotype_clusters")
        # NOTE: scirpy API varies across versions; implement + validate here before use.
    except ImportError:
        print("scirpy not installed; run %pip install scirpy")
else:
    print("Using exact TRB-CDR3 clone key (scirpy refinement disabled).")
```

```
Using exact TRB-CDR3 clone key (scirpy refinement disabled).
```

## Step 3 - dedup overlap + concatenate

`detect_duplicate_cells` reports the shared Brunner patients (D1==D3 reuse the same libraries;
`P112` is byte-identical). `concat_joint` keeps **D1's** copy of `P112/P115/P116/P121/P303/PGS`,
drops the D3 duplicates, outer-joins all datasets on gene symbols, and left-joins the TCR table.

```python  # cell 11
print("shared-Brunner cell counts per dataset:")
print(H.detect_duplicate_cells(std_paths))

FORCE_CONCAT = True   # B5(buus25) add: rebuild joint_raw with the new cohort + its ADT obsm
joint_path = H.concat_joint(NB_DIR, std_paths, tcr_parquet,
                            force=FORCE_CONCAT, brunner_keep="D1")

# cache the concatenated atlas in-session (joint_raw.h5ad is already on disk)
adata_raw = sc.read_h5ad(joint_path)
# friendly dataset names (author+year); relabels an older cache in place, no rebuild
_friendly = adata_raw.obs["dataset"].map(lambda d: H.DATASET_NAME.get(d, d)).astype("category")
if not adata_raw.obs["dataset"].astype(str).equals(_friendly.astype(str)):
    adata_raw.obs["dataset"] = _friendly
    adata_raw.write_h5ad(joint_path)
    print("relabeled dataset names ->", list(adata_raw.obs["dataset"].cat.categories))
print("joint_raw:", adata_raw.shape,
      "| TCR+ cells:", int(adata_raw.obs["has_tcr"].sum()),
      "| malignant:", int(adata_raw.obs["is_malignant"].sum()),
      "| ADT:", ("X_adt" in adata_raw.obsm))
adata_raw
```

```
shared-Brunner cell counts per dataset:
{'P112': {'D1': 4186, 'D3': 4186}, 'P115': {'D1': 5189, 'D3': 5189}, 'P116': {'D1': 3444, 'D3': 3444}, 'P121': {'D1': 3032, 'D3': 3032}, 'P303': {'D1': 4157, 'D3': 4157}, 'PGS': {'D1': 5545, 'D3': 5545}}
[concat] D3: dropped 25553 cells of shared Brunner donors

Failed to restart the Kernel. 

request to http://hgn26.wexac.weizmann.ac.il:8888/api/kernels/cef2f0ee-45f5-41e9-9986-e29358bed4fc/restart?1784616244543 failed, reason: connect ECONNREFUSED 132.76.221.2:8888. 

View Jupyter <a href='command:jupyter.viewOutput'>log</a> for further details.
```

```python  # cell 12
# B5(buus25) add: generate curated metadata rows for the new cohort from the freshly
# concatenated atlas so cell 13's annotate_from_csv finds every buus sample_id.
# Idempotent (replaces any existing B5 rows); no-op when buus25 is absent.
_FINAL = NB_DIR / "data" / "sample_metadata_final.csv"
_b = adata_raw.obs[adata_raw.obs["dataset"].astype(str) == "buus25"]
if len(_b):
    _cur = pd.read_csv(_FINAL, dtype=str)
    _CONST = dict(dataset="buus25", study="buus2025", n_donors="1", disease="SS",
                  disease_stage="", tech="10x_5p", sex="unknown", repo="GEO",
                  accession="GSE284075", malignant_call_method="TCR+ALICE",
                  tcr_available="yes", malignant_labeled="no", malignant_frac="",
                  tcr_recovery="", stage_class="advanced", stage_clean="", lineage="CD4",
                  entity="Sezary", fmf_status="not_applicable", lineage_resolve="no",
                  treatment_context="ex_vivo_baseline", blood_involvement="yes",
                  lesion_type="unknown", cohort_country="Denmark",
                  cohort_region="European", origin_resolve="no", east_asian="False")
    _rows = []
    for _sid, _sub in _b.groupby(_b["sample_id"].astype(str)):
        _comp = str(_sub["compartment"].iloc[0]); _tis = str(_sub["tissue"].iloc[0])
        _r = dict.fromkeys(_cur.columns, ""); _r.update(_CONST)
        _r.update(sample_id=_sid, donor=str(_sub["donor"].iloc[0]), compartment=_comp,
                  tissue=_tis, organ=("blood" if _comp == "Blood" else "skin"),
                  tissue_detail=_tis, n_cells=str(len(_sub)),
                  n_tcr=str(int(_sub["has_tcr"].sum())),
                  n_clones=str(_sub.loc[_sub["clone_id"].astype(str) != "", "clone_id"].nunique()),
                  n_malignant=str(int(_sub["is_malignant"].sum())))
        _rows.append(_r)
    _new = pd.DataFrame(_rows, columns=_cur.columns)
    _kept = _cur[~_cur["sample_id"].isin(_new["sample_id"])]
    pd.concat([_kept, _new], ignore_index=True).to_csv(_FINAL, index=False)
    print(f"[B5] wrote {len(_new)} buus25 metadata rows ({_new.donor.nunique()} donors) -> {_FINAL.name}")
else:
    print("[B5] no buus25 cells in atlas — skipping metadata append")
```

## Step 3b·a — apply final curated metadata (`sample_metadata_final.csv`) → cache

Rebuilds `joint_annotated.h5ad` from the pristine concat (`joint_raw.h5ad`) using the
**authoritative** `data/sample_metadata_final.csv`: `H.annotate_from_csv` broadcasts the curated
per-sample fields (`repo, accession, stage_class, lineage, entity, cohort_country/region,
origin_resolve, east_asian, …`) onto `obs` by `sample_id`, then `H.resolve_lineage` finalizes the
provisional lineages (`lineage_resolve != "no"`), per a priority order:

1. **D1 clinical** (`patients.tsv`) — authoritative. The Chennareddy `subset` diagnosis
   (Berti→CD8, gd_MF→γδ, CD4_MF→CD4) overrides expression, which mis-calls the γδ cases
   (γδ malignant cells under-express TRGC/TRDC; tiny malignant fractions give noisy gd ratios).
   Expression scores are still recorded and `lineage_disagree` flags where they differ.
2. **Literature-resolved** (`resolved_*`, e.g. Liu2022, incl. MF14=CD8) — kept as-is.
3. **Expression / inferCNV** (the TCR-less li24 `PT*`) — γδ if gd/(gd+ab)≥thresh else CD8 if
   CD8>CD4 else CD4, marked `needs_infercnv_verification` (malignant label not TCR-backed).
4. **Unresolvable** (no malignant cells) — left unchanged, `lineage_resolve="unresolved"`.

`entity` is synced to the resolved lineage (curated non-placeholder entities like Sezary /
MF/SS_leukemic preserved); `lineage_resolve` is rewritten to `resolved`/`unresolved`.

Starting from `joint_raw` (not the prior annotated cache) keeps the **per-cell** base columns —
notably per-cell `tissue` (Epidermis/Dermis) — intact. The cell validates sample coverage +
disease/compartment vocab + non-null lineage, asserts no stale `*_RESOLVE` placeholders and that
`entity`/`lineage_resolve`/`lineage_method` are mutually consistent, then re-writes the cache.

```python  # cell 14
# Merge the authoritative curated metadata (sample_metadata_final.csv) onto the atlas.
# Canonical path: start from joint_raw (pristine per-cell base obs, incl. per-cell tissue),
# broadcast the curated per-sample fields with annotate_from_csv, then resolve provisional lineages.
# D1 lineages come from the Chennareddy clinical table (patients.tsv) — authoritative — not from
# expression; entity is synced to the resolved lineage; lineage_resolve -> resolved/unresolved.
ANNOT = OUT / "joint_annotated.h5ad"
FINAL = NB_DIR / "data" / "sample_metadata_final.csv"
D1_CLIN = NB_DIR / "data" / "D1_chennareddy2025" / "meta" / "patients.tsv"

adata_raw = sc.read_h5ad(OUT / "joint_raw.h5ad")            # per-cell base columns intact
adata_raw, added = H.annotate_from_csv(adata_raw, FINAL)    # adds curated cols (repo ... east_asian)
report = H.resolve_lineage(adata_raw, min_cells=20, gd_frac_thresh=0.5, clinical_d1_tsv=D1_CLIN)
print("\nlineage resolution (lineage_resolve != 'no'):")
print(report.to_string(index=False))
flips = report[report["disagree"]]
if len(flips):
    print(f"\n{len(flips)} sample(s) where the expression call disagrees with the clinical diagnosis "
          "(clinical wins; expr scores retained for audit):")
    print(flips[["sample_id", "expr_call", "clinical", "final"]].to_string(index=False))

# make sure everything is in order
meta_ids = set(pd.read_csv(FINAL, dtype=str)["sample_id"])
missing = sorted(set(adata_raw.obs["sample_id"].astype(str).unique()) - meta_ids)
assert not missing, f"{len(missing)} obs samples absent from CSV: {missing[:10]}"
assert not (set(adata_raw.obs["disease"].unique()) - H.DISEASE_VOCAB), "bad disease vocab"
assert not (set(adata_raw.obs["compartment"].unique()) - H.COMPARTMENT_VOCAB), "bad compartment vocab"
assert adata_raw.obs["lineage"].isna().sum() == 0, "NaN lineage after resolve"
for c in ["cohort_country", "cohort_region", "origin_resolve", "east_asian"]:
    assert c in adata_raw.obs.columns, f"missing new column: {c}"

# post-fix validation: no stale *_RESOLVE placeholders; lineage_resolve↔lineage_method agree;
# entity consistent with lineage on every resolved row.
o = adata_raw.obs
for col in ("lineage", "entity", "lineage_resolve"):
    bad = int(o[col].astype(str).str.contains("RESOLVE").sum())
    assert bad == 0, f"{bad} cells still carry a *_RESOLVE placeholder in {col}"
assert not o["lineage_resolve"].astype(str).str.startswith("yes").any(), "stale 'yes_*' lineage_resolve"
res, meth = o["lineage_resolve"].astype(str), o["lineage_method"].astype(str)
assert meth[res == "unresolved"].str.startswith("unresolved").all(), "unresolved rows: method mismatch"
assert meth[res == "resolved"].str.startswith("resolved").all(), "resolved rows: method mismatch"
assert (meth[res == "no"] == "high_confidence").all(), "'no' rows: method mismatch"
ent_ok = {"CD8": {"MF_CD8", "CD8_aggressive_epidermotropic_CTCL_Berti"},
          "gamma_delta": {"MF_gamma_delta"}}
sub = o[(res == "resolved") & o["lineage"].astype(str).isin(ent_ok)]
for lin, allowed in ent_ok.items():
    got = set(sub.loc[sub["lineage"].astype(str) == lin, "entity"].astype(str).unique())
    assert got <= allowed, f"lineage {lin} maps to inconsistent entity {got - allowed}"
print("\nlineage × entity (non-empty rows):")
print(pd.crosstab(o["lineage"], o["entity"]).loc[lambda d: d.sum(1) > 0].to_string())

adata_raw.write_h5ad(ANNOT)
print(f"\n[merge] added {len(added)} curated cols: {added}")
print(f"[merge] wrote {ANNOT}  shape={adata_raw.shape}  obs cols={adata_raw.obs.shape[1]}")
```

```python  # cell 15
# Refine the cached lineage: D1 clinical override + hold the TCR-less PT* as provisional, then re-cache.
# Runs after the H.resolve_lineage cell (operates on the in-memory adata_raw) and writes back to
# joint_annotated.h5ad so the export cell below (which reloads from disk) sees it.
import pandas as pd

obs = adata_raw.obs

# columns we edit -> plain strings so re-assignment doesn't hit categorical-membership errors
for c in ['lineage', 'entity', 'lineage_method', 'lineage_resolve']:
    obs[c] = obs[c].astype(str)

# per-cell patient key = the part of sample_id after the dataset prefix. For D1 this is
# `sample_id_in_raw` (see CANON build), so match patients.tsv on that column, no suffix stripping.
pat = obs['sample_id'].astype(str).str.split('__').str[-1]

# ---- 1. D1 (Chennareddy): clinical diagnosis overrides the expression call ----
clin   = pd.read_csv(NB_DIR / 'data' / 'D1_chennareddy2025' / 'meta' / 'patients.tsv', sep='\t')
PATCOL = 'sample_id_in_raw'                       # matches the atlas sample_id suffix
sub2   = {'gd_MF': ('gamma_delta', 'MF_gamma_delta'),
          'Berti': ('CD8', 'CD8_aggressive_epidermotropic_CTCL')}
clin_map = {p: sub2[s] for p, s in zip(clin[PATCOL], clin['subset']) if s in sub2}

expr  = obs['lineage'].copy()                    # expression call, before override
is_d1 = obs['dataset'].eq('chennareddy25') & pat.isin(set(clin_map))
for p, (lin, ent) in clin_map.items():
    m = is_d1 & pat.eq(p)
    obs.loc[m, 'lineage'] = lin
    obs.loc[m, 'entity']  = ent
obs['lineage_disagreement'] = is_d1 & obs['lineage'].ne(expr) & ~expr.str.contains('RESOLVE')
obs.loc[is_d1, 'lineage_method']  = 'clinical(patients.tsv)'
obs.loc[is_d1, 'lineage_resolve'] = 'resolved_clinical'

# ---- 2. PT* atlas (no TCR -> inferCNV-based): hold provisional, don't lock CD8/gd ----
pt = obs['dataset'].eq('li24') & pat.isin(['PT11', 'PT53', 'PT55'])
obs.loc[pt, 'lineage']         = 'unresolved_provisional'   # scores retained in lineage_score_*
obs.loc[pt, 'entity']          = 'MF_unresolved'
obs.loc[pt, 'lineage_method']  = 'inferCNV_state_no_TCR(unverified)'
obs.loc[pt, 'lineage_resolve'] = 'provisional_inferCNV_unverified'

# ---- 3. drop the broken collapse columns only if they leaked into obs ----
for c in ['is_malignant', 'is_dominant_clone']:
    if c in obs and obs[c].astype(str).eq('2 values').any():
        obs.drop(columns=c, inplace=True)

# restore categoricals + persist so the export cell (reloads from disk) picks up the refinement
for c in ['lineage', 'entity', 'lineage_method', 'lineage_resolve']:
    obs[c] = obs[c].astype('category')
adata_raw.write_h5ad(OUT / "joint_annotated.h5ad")

# sanity check
print(obs.loc[is_d1 | pt, ['lineage', 'entity', 'lineage_resolve']].value_counts())
```

```python  # cell 16
# Step 3b - read & print the final cached annotated atlas + re-export per-sample metadata (for comparison)
import re

adata_raw = sc.read_h5ad(OUT / "joint_annotated.h5ad")
print(adata_raw)

obs = adata_raw.obs
# sanity: per-cell booleans (is_malignant, is_dominant_clone, ...) must stay per-cell in obs,
# never the literal "N values" / "N unique" collapse string.
_collapsed = re.compile(r"^\d+ (values|unique)$")
leaked = [c for c in obs.columns
          if obs[c].dtype == object and obs[c].astype(str).str.match(_collapsed).any()]
assert not leaked, f"obs columns hold a collapsed-aggregate string: {leaked}"

def _agg_col(s):            # per-sample summary that never yields a bare "N values"
    if pd.api.types.is_bool_dtype(s):
        return round(float(s.mean()), 4)               # per-cell boolean -> fraction True
    if s.name == "tissue":                             # legitimately multi-valued -> canonical join
        return "|".join(sorted({H._norm_tissue(x) for x in s.dropna()}))
    u = pd.unique(s.dropna())
    if len(u) == 1:
        return u[0]                                     # genuine per-sample scalar
    if pd.api.types.is_numeric_dtype(s):
        return round(float(s.mean()), 4)               # varying numeric (scores) -> mean
    return f"{len(u)} unique"                           # varying per-cell categorical (cell_id, cdr3)

# aggregate *every* obs column (complete quality check), then append derived counts/fractions.
g = obs.groupby("sample_id", observed=True)
sample_meta = g.agg(**{c: (c, _agg_col) for c in obs.columns if c != "sample_id"})
for name, col, fn in [("n_donors", "donor", lambda s: s.nunique()),
                      ("n_cells", "cell_id", lambda s: s.size),
                      ("n_tcr", "has_tcr", lambda s: int(s.sum())),
                      ("n_clones", "clone_id", lambda s: s[s != ""].nunique()),
                      ("n_malignant", "is_malignant", lambda s: int(s.sum()))]:
    if col in obs.columns:
        sample_meta[name] = g[col].agg(fn)
sample_meta = sample_meta.reset_index()
sort_cols = [c for c in ["dataset", "sample_id"] if c in sample_meta.columns]
sample_meta = sample_meta.sort_values(sort_cols)

# (a) no "N values" anywhere; (b) entity consistent with lineage; (c) lineage_resolve↔lineage_method
assert not sample_meta.astype(str).apply(lambda c: c.str.match(r"^\d+ values$")).to_numpy().any(), \
    "a column still collapsed to a bare 'N values' string"
ent_ok = {"CD8": {"MF_CD8", "CD8_aggressive_epidermotropic_CTCL_Berti"},
          "gamma_delta": {"MF_gamma_delta"}}
for lin, allowed in ent_ok.items():
    got = set(sample_meta.loc[sample_meta["lineage"].astype(str) == lin, "entity"].astype(str))
    assert got <= allowed, f"per-sample lineage {lin} maps to inconsistent entity {got - allowed}"
agree = sample_meta.apply(
    lambda r: (str(r["lineage_method"]).startswith(str(r["lineage_resolve"]))
               if r["lineage_resolve"] in ("resolved", "unresolved")
               else str(r["lineage_method"]) == "high_confidence"), axis=1)
assert agree.all(), f"lineage_resolve/lineage_method disagree:\n{sample_meta.loc[~agree, ['sample_id','lineage_resolve','lineage_method']]}"

# write to a *separate* file so the authoritative input (sample_metadata.csv) is untouched
csv_path = OUT / "sample_metadata_check.csv"
sample_meta.to_csv(csv_path, index=False)
print(f"wrote {csv_path}  ({len(sample_meta)} samples, {sample_meta.shape[1]} cols)  -- checks: no 'N values', entity↔lineage, resolve↔method all pass")
sample_meta
```

## Step 4 — Raw UMAP (baseline, before MrVI)

Loads the cached annotated atlas and computes a **standard UMAP straight from raw counts**
(HVG → normalize → log1p → PCA → neighbors → UMAP) over all ~1.34M cells — no integration. This is
the un-corrected baseline (expect visible `study`/`compartment` batch structure) to compare against
the MrVI `X_mrvi_u` embedding in Step 6.

**HEAVY:** run on the GPU kernel. UMAP coordinates are cached to
`data/atlas_joint/joint_umap_raw.npy` so re-runs just reload.

```python  # cell 18
# Step 4 (pre-UMAP) - atlas overview: headline counts + obs summary tables
ANNOT = OUT / "joint_annotated.h5ad"
if "adata_raw" not in globals():
    adata_raw = sc.read_h5ad(ANNOT)
obs = adata_raw.obs


def _col(name):
    return obs[name] if name in obs.columns else None


print("=" * 60)
print("JOINT CTCL/MF/SS ATLAS - overview")
print("=" * 60)
print(f"cells        : {adata_raw.n_obs:,}")
print(f"genes        : {adata_raw.n_vars:,}")
for label, c in [("studies", "study"), ("datasets", "dataset"), ("samples", "sample_id"),
                 ("donors", "donor"), ("real donors", "real_donor"),
                 ("cell types", "cell_type"), ("entities", "entity")]:
    s = _col(c)
    if s is not None:
        print(f"{label:<12} : {s.nunique():,}")

# TCR / malignancy headline
if "has_tcr" in obs:
    n_tcr = int(obs["has_tcr"].sum())
    print(f"TCR+ cells   : {n_tcr:,}  ({100 * n_tcr / adata_raw.n_obs:.1f}%)")
if "is_malignant" in obs and not obs["is_malignant"].astype(str).eq("2 values").any():
    n_mal = int(obs["is_malignant"].sum())
    print(f"malignant    : {n_mal:,}  ({100 * n_mal / adata_raw.n_obs:.1f}%)")
if "clone_id" in obs:
    print(f"TCR clones   : {obs.loc[obs['clone_id'].astype(str) != '', 'clone_id'].nunique():,}")


def summary_table(col, top=None):
    """Per-category cell counts + % + #samples, sorted by cell count."""
    if col not in obs.columns:
        return None
    g = obs.groupby(col, observed=True)
    df = pd.DataFrame({
        "cells": g.size(),
        "pct": (100 * g.size() / adata_raw.n_obs).round(1),
    })
    if "sample_id" in obs.columns:
        df["samples"] = g["sample_id"].nunique()
    if "study" in obs.columns:
        df["studies"] = g["study"].nunique()
    df = df.sort_values("cells", ascending=False)
    return df.head(top) if top else df


for col in ["study", "disease", "stage_class", "compartment", "tissue",
            "lineage", "sex", "tech"]:
    df = summary_table(col)
    if df is not None:
        print(f"\n--- by {col} ---")
        print(df.to_string())

print("\n--- top 15 cell types ---")
ct = summary_table("cell_type", top=15)
if ct is not None:
    print(ct.to_string())

# disease x compartment cross-tab (where the biology vs batch confound lives)
if {"disease", "compartment"} <= set(obs.columns):
    print("\n--- disease x compartment (cells) ---")
    print(pd.crosstab(obs["disease"], obs["compartment"]).to_string())
```

```python  # cell 19
# Step 4 - load cached annotated atlas + baseline UMAP from raw counts (no integration)
import numpy as np

ANNOT = OUT / "joint_annotated.h5ad"
if "adata_raw" not in globals():
    adata_raw = sc.read_h5ad(ANNOT)
print(adata_raw)

np.random.seed(SEED)
UMAP_NPY = OUT / "joint_umap_raw.npy"
if UMAP_NPY.exists():
    adata_raw.obsm["X_umap"] = np.load(UMAP_NPY)
    print("loaded cached UMAP", adata_raw.obsm["X_umap"].shape)
else:
    # HEAVY (GPU kernel): standard pipeline on a 2k-HVG copy (keeps adata_raw genes intact)
    hvg = sc.pp.highly_variable_genes(
        adata_raw, n_top_genes=2000, flavor="seurat_v3",
        layer="raw_counts", inplace=False, subset=False,
    )
    ad_vis = adata_raw[:, hvg["highly_variable"].to_numpy()].copy()
    ad_vis.X = ad_vis.layers["raw_counts"].copy()
    sc.pp.normalize_total(ad_vis, target_sum=1e4)
    sc.pp.log1p(ad_vis)
    sc.pp.scale(ad_vis, max_value=10)
    sc.pp.pca(ad_vis, n_comps=50, random_state=SEED)
    sc.pp.neighbors(ad_vis, random_state=SEED)
    sc.tl.umap(ad_vis, random_state=SEED)
    adata_raw.obsm["X_umap"] = ad_vis.obsm["X_umap"]
    np.save(UMAP_NPY, adata_raw.obsm["X_umap"])
    print("computed + cached UMAP ->", UMAP_NPY)
```

```python  # cell 20
adata_raw.obs.head()
```

```python  # cell 21
# Step 4b - baseline UMAP colored by annotation (before MrVI)
sc.pl.umap(
    adata_raw,
    color=[ "study", "disease", "compartment",'disease_stage'],
    ncols=2, frameon=False, size=2, wspace=0.25,
)
```

## Step 6 — MrVI sample-level model

Fits **MrVI** (`scvi.external.MRVI`) on the full joint atlas → a sample-aware latent space plus
sample-level comparative analyses (sample distances, differential abundance, DE).

- `sample_key="sample_id"` (109 samples), `batch_key="study"` (9 cohorts), `layer="raw_counts"`.
  Per the MrVI paper, the sample/donor identifier is the natural target covariate and the
  *study of origin* is the textbook cross-study nuisance.
- Input HVG-subset to 10k genes (`seurat_v3`, batched on `study`), cached to
  `data/atlas_joint/joint_mrvi_input.h5ad`. Architecture/training use the MrVI defaults
  (`n_latent=30`, `n_latent_u=10`, `max_epochs=100` w/ early stopping, `batch_size=256`,
  `lr=2e-3`) — set "through hyperparameter sensitivity studies" in the paper.

**Caveat — study/disease confound.** This MRVI build accepts a *single* `batch_key`. In this atlas
`study` is partly confounded with `disease`/`compartment` (blood-only Sézary vs skin-only MF cohorts),
so the `study` nuisance can absorb biological signal. Interpret comparative DE/DA across `disease`
**within strata** where both groups co-occur (e.g. MF-skin stage contrasts); the confound-free
contrasts are **within-donor** (malignant-vs-benign pseudo-samples, see `jobs/run_mrvi_tcells_mb.py`).

### Launch (GPU, bsub)

Training on 1.34M cells is heavy — run as a job, **not** in this kernel:

```bash
bash jobs/run_mrvi_joint.sh                       # full atlas, 15k HVG, max_epochs=50
bash jobs/run_mrvi_joint.sh --subsample 300000    # faster: fit on 300k cells, transform all
bash jobs/run_mrvi_joint.sh --skip-train          # reuse models/mrvi_joint, just (re)cache latents/diagnostics
```

Monitor with `tail -f jobs/run_mrvi_joint.bsub.log`. Outputs: `models/mrvi_joint/`,
`data/atlas_joint/joint_X_mrvi{,_u}.npy`, `figures/mrvi_joint_{history.json,donor_distances.nc,da.nc}`.
The HVG-subset input is cached once to `data/atlas_joint/joint_mrvi_input.h5ad` (rebuild with `--force`).

```python  # cell 24
# Step 6a - submit the MrVI training job to the GPU queue (bsub returns immediately)
import subprocess

# --force rebuilds the HVG-input cache (joint_mrvi_input.h5ad) from the CURRENT
# joint_annotated.h5ad. REQUIRED after the atlas changes (e.g. adding buus25) — without it
# MrVI silently reuses a stale cache and trains on the old cell set.
res = subprocess.run(
    ["bash", str(NB_DIR / "jobs" / "run_mrvi_joint.sh"), "--force"],
    capture_output=True, text=True,
)
print(res.stdout, res.stderr)   # prints the submitted job id
# monitor:  !tail -n 40 jobs/run_mrvi_joint.bsub.log
# faster first pass:  add "--subsample", "300000" to the command above
```

```
Job <326106> is submitted to queue <gsla_high_gpu>.
 Please note that memory reservation is per task (-n X)
Memory reservation is (MB): 200000 
===Your total amount of memory reservation for this job is (MB): 200000 ===
```

```python  # cell 25
# Step 6b - load cached MrVI latents into obsm (after the bsub job finishes)
import numpy as np

# The heavy concat/annotate now runs as a batch job (jobs/run_build_joint.sh), not in this
# kernel -> load the annotated atlas here if starting from Step 6.
ANNOT = OUT / "joint_annotated.h5ad"
if "adata_raw" not in globals():
    adata_raw = sc.read_h5ad(ANNOT)
    print("loaded", adata_raw.shape, "from", ANNOT.name)

z = np.load(OUT / "joint_X_mrvi.npy")       # sample-aware
u = np.load(OUT / "joint_X_mrvi_u.npy")     # sample-unaware
assert z.shape[0] == u.shape[0] == adata_raw.n_obs, (
    f"MrVI latent rows ({u.shape[0]}) != atlas n_obs ({adata_raw.n_obs}) - "
    "stale npy vs current joint_annotated.h5ad; rerun run_mrvi_joint.sh --force")
adata_raw.obsm["X_mrvi_z"] = z
adata_raw.obsm["X_mrvi_u"] = u
print("X_mrvi_z", z.shape, "| X_mrvi_u", u.shape)

# HEAVY - run on the GPU kernel only: neighbors+UMAP on 1.34M cells.
# X_mrvi_u = integration view (cell state); X_mrvi_z = sample-aware view.
sc.pp.neighbors(adata_raw, use_rep="X_mrvi_u")
sc.tl.umap(adata_raw)
sc.pl.umap(adata_raw, color=["cell_type", "study", "disease", "compartment"], ncols=2)
```

```
loaded (1173694, 40821) from joint_annotated.h5ad

---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
Cell In[5], line 13
     11 z = np.load(OUT / "joint_X_mrvi.npy")       # sample-aware
     12 u = np.load(OUT / "joint_X_mrvi_u.npy")     # sample-unaware
---> 13 assert z.shape[0] == u.shape[0] == adata_raw.n_obs, (
     14     f"MrVI latent rows ({u.shape[0]}) != atlas n_obs ({adata_raw.n_obs}) - "
     15     "stale npy vs current joint_annotated.h5ad; rerun run_mrvi_joint.sh --force")
     16 adata_raw.obsm["X_mrvi_z"] = z
     17 adata_raw.obsm["X_mrvi_u"] = u

AssertionError: MrVI latent rows (1339241) != atlas n_obs (1173694) - stale npy vs current joint_annotated.h5ad; rerun run_mrvi_joint.sh --force
```

```python  # cell 26
# Step 6c - sample-distance + differential-abundance diagnostics (after the job finishes)
import xarray as xr
import matplotlib.pyplot as plt

FIG = NB_DIR / "figures"
dist = xr.open_dataset(FIG / "mrvi_joint_donor_distances.nc")
da = xr.open_dataset(FIG / "mrvi_joint_da.nc")
print(dist)
print(da)

# mean sample x sample distance (averaged over cell-type groups) - does it cluster by study?
dvar = next(iter(dist.data_vars))
arr = dist[dvar]
M = arr.mean(dim=arr.dims[0]).to_pandas()           # collapse the cell_type group dim
fig, ax = plt.subplots(figsize=(5, 3))
im = ax.imshow(M.values, cmap="viridis")
fig.colorbar(im, ax=ax, fraction=0.046)
ax.set_title("MrVI sample distances: check disease vs study clustering")
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
```

## Step 7 — MrVI (skin-only)

Re-runs the **same** pipeline on skin cells (`compartment == "Skin"`: 79 samples, ~866k cells,
6 studies) — smaller and less study/disease-confounded than the full atlas (drops the blood-only
Sézary cohorts + the single LN sample). Reuses `jobs/run_mrvi_joint.py` via
`--subset-compartment Skin --tag skin`; every output gets a `_skin` suffix
(`joint_mrvi_input_skin.h5ad`, `models/mrvi_joint_skin/`, `joint_X_mrvi{,_u}_skin.npy`,
`mrvi_joint_skin_*`) so it never collides with the full-atlas run. The submit cell sets
`JOB_TAG=skin` so the bsub job name + log are also distinct (`run_mrvi_joint_skin.bsub.log`).

```python  # cell 28
# Step 7a - submit the skin-only MrVI job (JOB_TAG gives it a distinct job name + log)
import os, subprocess

# --force rebuilds the skin HVG-input cache from the current atlas (see Step 6a note).
env = {**os.environ, "JOB_TAG": "skin", "WALL_H": "48"}
res = subprocess.run(
    ["bash", str(NB_DIR / "jobs" / "run_mrvi_joint.sh"),
     "--subset-compartment", "Skin", "--tag", "skin", "--force"],
    capture_output=True, text=True, env=env,
)
print(res.stdout, res.stderr)   # submitted job id
# monitor:  !tail -n 40 jobs/run_mrvi_joint_skin.bsub.log
```

```python  # cell 29
# Step 7b - load skin MrVI latents (after the skin job finishes)
import numpy as np

# load the atlas if the kernel is fresh (Step 6 may not have been run this session)
ANNOT = OUT / "joint_annotated.h5ad"
if "adata_raw" not in globals():
    adata_raw = sc.read_h5ad(ANNOT)
    print("loaded", adata_raw.shape, "from", ANNOT.name)

ad_skin = adata_raw[adata_raw.obs["compartment"] == "Skin"].copy()   # same row order as the job
u = np.load(OUT / "joint_X_mrvi_u_skin.npy")     # sample-unaware
assert u.shape[0] == ad_skin.n_obs, (
    f"skin MrVI rows ({u.shape[0]}) != skin cells ({ad_skin.n_obs}) - "
    "stale npy; rerun the skin MrVI job with --force")
ad_skin.obsm["X_mrvi_z"] = np.load(OUT / "joint_X_mrvi_skin.npy")     # sample-aware
ad_skin.obsm["X_mrvi_u"] = u
print("ad_skin", ad_skin.shape,
      "| X_mrvi_z", ad_skin.obsm["X_mrvi_z"].shape,
      "| X_mrvi_u", ad_skin.obsm["X_mrvi_u"].shape)

# HEAVY - GPU kernel: neighbors+UMAP on the skin subset (X_mrvi_u = integration view)
# sc.pp.neighbors(ad_skin, use_rep="X_mrvi_u")
# sc.tl.umap(ad_skin)
# sc.pl.umap(ad_skin, color=["cell_type", "study", "disease", "stage_class"], ncols=2)
```

```python  # cell 30
sc.pp.neighbors(ad_skin, use_rep="X_mrvi_u")
sc.tl.umap(ad_skin)
sc.pl.umap(ad_skin, color=["cell_type", "study", "disease", "stage_class"], ncols=2)
```
