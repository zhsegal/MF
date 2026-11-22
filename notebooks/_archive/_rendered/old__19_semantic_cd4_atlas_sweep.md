# old/19_semantic_cd4_atlas_sweep.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/19_semantic_cd4_atlas_sweep.ipynb` and is never run._

# SemanticSCVI hyperparameter sweep — CD4 CTCL atlas

Parallel to `four_way_benchmark_haniffa_cd8_genept_sweep.ipynb`, but for the **CD4 non-Treg atlas** model of `18_semantic_cd4_atlas_malignancy.ipynb` (geometric, **Geneformer** prior). Preprocessing is identical to nb18 (CD4 non-Treg subset, ribo drop, **Geneformer-vocab genes only**, HVG).

**Flow (like `17_mrvi_pseudosample_malignancy.ipynb`):** this notebook runs the light preprocessing on the interactive kernel, writes a slim input + config, then **submits a bsub job from a cell** (`jobs/run_cd4_sweep.sh` → `jobs/run_cd4_sweep.py`) that does the heavy GPU work (train every variant + projection benchmark + report). Poll from a cell; display the HTML report when done.

- Sweeps **5 axes**: `max_epochs`, `warmup_epochs`, `n_epochs_kl_warmup`, `coherence_weight`, `n_layers`. Edit `BASELINE` (fixed defaults) and `SWEEP` (per-axis alternatives). Each value in `SWEEP[axis]` makes one variant differing from baseline **only** in that axis; baseline always included; duplicates deduped.
- **Projections only** (no gene clustering); top-30 loadings per projection.
- LLM scoring with **Opus only**, judging each projection in the context of CD4 T cells spanning benign/malignant states in T-cell lymphoma.
- Per-projection **MSigDB** enrichment with `lib1_immune.gmt` + `lib2_cd4.gmt`.
- One-third subsample of CD4 cells for speed.

**Cells:** config → imports/harness → data prep (nb18-verbatim) → subsample → build `lib2_cd4.gmt` → **write job input** → **submit** → **poll** → **display report**.

```python  # cell 1
# ============================================================
# Parameters — TCR rule + SemanticSCVI sweep config (geometric Geneformer, from nb18).
# ============================================================
from pathlib import Path


def _resolve_nb_dir() -> Path:
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("notebooks/MF"), Path("scvi-tools-neural-nmf/notebooks/MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data starting from {start}")


NB_DIR = _resolve_nb_dir()
print(f"NB_DIR = {NB_DIR}")

# ---- paths (same atlas inputs as nb18) ----
OBJ            = NB_DIR / "data" / "atlas_joint" / "skin_T_annotated.h5ad"
TCR_OBJ        = NB_DIR / "data" / "atlas_joint" / "skin_T_tcr_annotated.h5ad"
LI_TCR         = NB_DIR / "data" / "Li2024_atlas" / "li2024_tcr_malignancy.parquet"
GENE_ID_SOURCE = NB_DIR / "data" / "cache" / "cnmf_malignant_counts.h5ad"   # gene_name -> gene_id
SEMANTIC_CACHE_GENEFORMER = NB_DIR / "data" / "mf_cd4_atlas_geneformer.pt"
MODEL_CACHE_DIR  = NB_DIR / "models" / ".model_cache_semantic_geom_cd4_atlas"
SEM_CACHE_PARENT = "semantic_scvi_sweep"
OUT_DIR          = NB_DIR / "benchmark_results" / "cd4_atlas_semantic_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# MSigDB libraries (in notebooks/, one level up from MF/)
LIB1_GMT = NB_DIR.parent / "lib1_immune.gmt"
LIB2_GMT = NB_DIR.parent / "lib2_cd4.gmt"
C7_SOURCE = Path("/home/projects/nyosef/zvise/metrics/pathways/c7.all.v2025.1.Hs.symbols.gmt")
PATHWAY_INDEX = NB_DIR.parent / "pathway_index.pkl"

# ---- bsub job handoff (slim input + config consumed by jobs/run_cd4_sweep.py) ----
# Job input lives in OUT_DIR/_job_input/ which the report-cleanup step preserves
# (alongside _llm_cache / _score_cache), so resubmits don't lose it.
JOBS_DIR      = NB_DIR / "jobs"
JOB_INPUT_DIR = OUT_DIR / "_job_input"
CD4_INPUT     = JOB_INPUT_DIR / "cd4_sweep_input.h5ad"
SEM_MAP_PATH  = JOB_INPUT_DIR / "cd4_sweep_semantic_map.pt"
JOB_CONFIG    = JOBS_DIR / "cd4_sweep_config.json"
JOB_SH        = JOBS_DIR / "run_cd4_sweep.sh"
JOB_LOG       = JOBS_DIR / "run_cd4_sweep.bsub.log"

# ---- TCR dominant-clone rule (relaxed: fold-change >= 1.33) ----
FRAC_THRESH, RATIO_THRESH, EXPANDED_MIN = 0.05, 1.33, 2

# ---- CD4 non-Treg selection (cell_type_T) ----
CD4_T_TYPES = ["CD4_Cm", "CD4_Th2", "CD4_cytotoxic", "CD4_mem"]

# ---- Preprocessing ----
HVG_TOP_N  = 2500
HVG_FLAVOR = "seurat_v3"
SUBSAMPLE_FRACTION = 1.0 / 3.0   # one-third of CD4 cells for the sweep

# ---- Shared model size / batch ----
N_LATENT  = 10
BATCH_KEY = "sample_id"

# ---- Projection scoring ----
PER_PROJECTION_N_TOP = 30

# ---- Sweep: baseline = nb18 geometric Geneformer config ----
BASELINE = {
    "max_epochs":         200,
    "warmup_epochs":       20,
    "n_epochs_kl_warmup": 100,
    "coherence_weight":  1000.0,
    "n_layers":             1,
}
# Per-axis alternatives. Each value -> one variant differing only in that axis.
SWEEP = {
    "max_epochs":         [],
    "warmup_epochs":      [],
    "n_epochs_kl_warmup": [],
    "coherence_weight":   [500.0, 2000.0,3000.0],
    "n_layers":           [],
}
# Fixed (non-swept) SemanticSCVI constructor kwargs (from nb18).
# coherence_weight / n_layers are swept; max_epochs/warmup/kl are passed explicitly.
FIXED_KWARGS = dict(
    loss_mode="geometric",
    n_gene_sample=1024,
    n_latent=N_LATENT,
    n_hidden=128,
    dropout_rate=0.1,
    gene_likelihood="nb",
    weights_positive=True,
    use_batch_norm=False,
)

# Per-variant cache_dir is keyed on a hash of the ABSOLUTE params (see _cache_slug
# in the next cell), so changing BASELINE never collides: same config -> cache hit,
# any change -> fresh train. No need to clear the cache when editing BASELINE.
```

```
NB_DIR = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
```

```python  # cell 2
import sys, gc, json, hashlib, subprocess, importlib
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))            # MF/ holds atlas_join_helpers, skin_T_cnv_helpers
if str(NB_DIR.parent) not in sys.path:
    sys.path.insert(0, str(NB_DIR.parent))     # notebooks/ holds benchmark_helpers

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import scvi

import atlas_join_helpers as H
import skin_T_cnv_helpers as C
importlib.reload(H); importlib.reload(C)
from benchmark_helpers import get_or_build_geneformer_map   # training/benchmark run in jobs/run_cd4_sweep.py

scvi.settings.seed = 0
np.random.seed(0)
sc.settings.verbosity = 1
print("scvi", scvi.__version__, "| scanpy", sc.__version__, "| CUDA:", torch.cuda.is_available())

# ---- generalized sweep harness (all axes in BASELINE are sweepable) ----
_AXES = tuple(BASELINE)
_AXIS_SHORT = {"max_epochs": "me", "warmup_epochs": "w", "n_epochs_kl_warmup": "kl",
               "coherence_weight": "cw", "n_layers": "nl"}


def _fmt_val(v):
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:g}".replace(".", "p")
    return str(v)


def _variant_name(variant, baseline):
    """Human-readable label = diff from baseline (used in the report)."""
    diffs = [k for k in _AXES if variant[k] != baseline[k]]
    if not diffs:
        return "sem_baseline"
    return "sem_" + "_".join(f"{_AXIS_SHORT[k]}{_fmt_val(variant[k])}" for k in diffs)


def _cache_slug(variant, n=10):
    """Stable 10-char hash of the ABSOLUTE params that define the trained model.
    cache_dir is keyed on this (not the diff-name), so changing BASELINE never
    collides: same config -> cache hit, any change -> fresh train."""
    blob = json.dumps(
        {
            "axes": {k: variant[k] for k in _AXES},
            "fixed": dict(sorted(FIXED_KWARGS.items())),
            "batch_key": BATCH_KEY,
            "labels_key": "cell_type_T",
            "hvg_top_n": HVG_TOP_N,
        },
        default=str, sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:n]


def _expand(baseline, sweep):
    for k in sweep:
        if k not in _AXES:
            raise ValueError(f"SWEEP key {k!r} not sweepable. Allowed: {_AXES}.")
    seen, variants = set(), []
    def _add(v):
        t = tuple(sorted((k, v[k]) for k in _AXES))
        if t not in seen:
            seen.add(t)
            variants.append(dict(v))
    _add(baseline)
    for axis, values in sweep.items():
        for val in values:
            v = dict(baseline); v[axis] = val
            _add(v)
    return variants


variants = _expand(BASELINE, SWEEP)
print(f"{len(variants)} variant(s):")
for v in variants:
    print(f"  {_variant_name(v, BASELINE):>20s}  [{_cache_slug(v)}]  " +
          "  ".join(f"{_AXIS_SHORT[k]}={v[k]}" for k in _AXES))
```

```
Seed set to 0

/tmp/ipykernel_2992269/1350882000.py:21: FutureWarning: `__version__` is deprecated, use `importlib.metadata.version('scanpy')` instead
  print("scvi", scvi.__version__, "| scanpy", sc.__version__, "| CUDA:", torch.cuda.is_available())

scvi 1.3.3 | scanpy 1.11.5 | CUDA: True
4 variant(s):
          sem_baseline  [8f2e45825c]  me=200  w=20  kl=100  cw=1000.0  nl=1
             sem_cw500  [b4acc56ec5]  me=200  w=20  kl=100  cw=500.0  nl=1
            sem_cw2000  [458e5a7f5c]  me=200  w=20  kl=100  cw=2000.0  nl=1
            sem_cw3000  [55dc715ca8]  me=200  w=20  kl=100  cw=3000.0  nl=1
```

```python  # cell 3
adata = C.build_or_load_tcr_object(OBJ, TCR_OBJ, LI_TCR, H)   # cached ~12 GB object
is_li = adata.obs["study"].astype(str).eq("li2024").to_numpy()
print("loaded:", adata.shape)
```

```
loaded cached TCR-complete object -> skin_T_tcr_annotated.h5ad (401566, 49683)

study:
 study
li2024             232365
chennareddy2025    128927
brunner2024         25778
gaydosik2019        11443
rindler2021_fi       2134
herrera2021           919
Name: count, dtype: int64

has_tcr by study:
                    size     sum
study                          
brunner2024       25778   12392
chennareddy2025  128927   69086
gaydosik2019      11443       0
herrera2021         919     659
li2024           232365  114640
rindler2021_fi     2134    1371
cached tumor_cell: 99802 / 401566
loaded: (401566, 49683)
```

```python  # cell 4
# unified TRB-primary clone key + per-donor dominant-clone malignancy (relaxed fold >= 1.33)
dom_tbl = C.recompute_dominant_clone(adata, H, is_li, FRAC_THRESH, RATIO_THRESH, EXPANDED_MIN)
print(dom_tbl.to_string(index=False))

clone_summary = C.clone_summary_table(adata, FRAC_THRESH, RATIO_THRESH)
print(f"\nthresholds: dom_frac >= {FRAC_THRESH}  AND  fold_change >= {RATIO_THRESH}")
print(f"malignant donors: {int(clone_summary['malignant'].sum())}/{len(clone_summary)}")
print("\nTCR-malignant cells by study:\n",
      adata.obs.groupby("study", observed=True)["tcr_is_malignant"].agg(["size", "sum"]))
clone_summary
```

```
donor  n_tcr                top_clone  top_n  dom_frac  ratio  is_dominant
           D1__P112    220      TRB:CASSQTGVINSPLHF     16     0.073   4.00         True
           D1__P115    583      TRA:CAVEDVKAAGNKLTF      8     0.014   1.14        False
           D1__P116    491   TRB:CASSQEIVDSEGTDTQYF     13     0.026   1.86        False
           D1__P121    245        TRB:CASSAGSQETQYF     19     0.078   2.71         True
           D1__P124    908  TRB:CSARPNWSAGRGNTGELFF     14     0.015   1.27        False
           D1__P127   2183       TRB:CSAHSGVGTDTQYF     34     0.016   1.21        False
           D1__P171   6800     TRB:CAISEAPSGLYNEQFF   6112     0.899 142.14         True
           D1__P195  10326        TRB:CASSLGQGNIQYF   5404     0.523 110.29         True
           D1__P202   3627       TRB:CASTALQSLYEQYF      7     0.002   1.17        False
           D1__P204   7407        TRB:CASSIRGNEKLFF   2623     0.354   4.34         True
           D1__P218   3839        TRB:CASSLGQGNIQYF   3468     0.903 385.33         True
           D1__P230    391      TRB:CASSYSSLAWGTQYF     54     0.138   2.84         True
           D1__P303   1371      TRB:CASSQDRALENTIYF    909     0.663   4.71         True
           D1__P311   5824        TRB:CASSFGGVSPLHF   4024     0.691   4.65         True
           D1__P312   2347      TRB:CASSVANRGTYEQYF    935     0.398   5.05         True
     D1__P318_thick   1388        TRB:CASSGTENSPLHF     11     0.008   1.10        False
      D1__P318_thin   1515        TRB:CASSLGQAYEQYF     15     0.010   1.36        False
            D1__P71    273     TRB:CASKETGYESVDEQYF     30     0.110   6.00         True
            D1__P76   8215    TRB:CASSLALSGGAYNEQFF   6209     0.756  66.05         True
            D1__P95   5192    TRB:CSAATSGTAGNNYGYTF   3040     0.586  84.44         True
            D1__P97    923          TRB:CASSPRGEQYF     12     0.013   1.33        False
            D1__PGS   5018    TRB:CASSFPRGASGANVLTF   3978     0.793  23.26         True
           D3__P105    598        TRB:CASSLTGGETQYF     77     0.129   2.08         True
           D3__P192   4841           TRB:CSVEVTEAFF   3656     0.755  26.11         True
           D3__P196    857    TRB:CASSEARPGLGTDTQYF    623     0.727  20.10         True
            D3__P87   1384     TRB:CASSYAGTGVGNTIYF   1208     0.873   7.06         True
            D3__P94   4712        TRB:CASSTGVFEKLFF    143     0.030   1.46        False
          D5__MFIVB   1371      TRB:CASSQDRALENTIYF    909     0.663   4.71         True
             H__HC1     19      TRB:CASSPGQGQGETQYF      3     0.158   1.50         True
           H__MFIV1     67      TRB:CSARFLRGGYNEQFF     23     0.343  23.00         True
             H__SS1    117      TRB:CASSRQGGSGNTIYF     65     0.556  13.00         True
             H__SS2    157     TRB:CASKLFMDREKGEAFF     27     0.172   1.69         True
             H__SS3     69        TRB:CASSTTGNTEAFF     23     0.333   7.67         True
             H__SS4    230     TRB:CASSRLGGAYNSPLHF    177     0.770  29.50         True
Li2024_atlas__CTCL1   8156        TRB:CSARTGSNQPQHF   7073     0.867  60.45         True
Li2024_atlas__CTCL2   4698   TRB:CASRASSNSGRASYEQYF   1837     0.391  27.42         True
Li2024_atlas__CTCL3  11490         TRB:CSARTGGYGYTF   8141     0.709  23.46         True
Li2024_atlas__CTCL4   9890       TRB:CASSVSGSNTEAFF    738     0.075   2.77         True
Li2024_atlas__CTCL5  12376      TRB:CASSPSTQSNEKLFF   6472     0.523  30.67         True
Li2024_atlas__CTCL6  13981     TRB:CASSQAFGPPHGTIYF   3092     0.221  31.55         True
Li2024_atlas__CTCL7  32803       TRB:CASSYRGPYNEQFF   3117     0.095   5.60         True
Li2024_atlas__CTCL8   8527     TRB:CASSLVMGVANTEAFF   5313     0.623  41.83         True
 Li2024_atlas__PT11   1814     TRB:CASSQDSWPYHSPLHF   1592     0.878 159.20         True
 Li2024_atlas__PT35   3183      TRB:CASSRLGGSGNTIYF   1456     0.457   1.03        False
 Li2024_a
... [8942 more chars truncated]
```

```python  # cell 5
# drop samples (donors) with no TCR at all — malignancy is undefined there
tcr_donors = adata.obs.loc[adata.obs["has_tcr"].astype(bool), "donor"].unique()
dropped = sorted(set(adata.obs["donor"].unique()) - set(tcr_donors))
print(f"donors with TCR: {len(tcr_donors)} | dropped (no TCR): {len(dropped)} -> {dropped}")
adata = adata[adata.obs["donor"].isin(tcr_donors)].copy()
print("after drop:", adata.shape)
print(adata.obs.groupby("study", observed=True)["has_tcr"].agg(["size", "sum"]))
```

```
donors with TCR: 50 | dropped (no TCR): 29 -> ['D6__MF2', 'D6__MF5', 'D6__NOR1', 'D6__NOR2', 'D6__NOR3', 'D6__NOR4', 'D6__SC157', 'D6__SC158', 'D6__SC205', 'Li2024_atlas__CTCL10', 'Li2024_atlas__CTCL11', 'Li2024_atlas__CTCL12', 'Li2024_atlas__CTCL13', 'Li2024_atlas__CTCL14', 'Li2024_atlas__CTCL15', 'Li2024_atlas__CTCL16', 'Li2024_atlas__CTCL17', 'Li2024_atlas__CTCL18', 'Li2024_atlas__CTCL9', 'Li2024_atlas__MF14', 'Li2024_atlas__MF15', 'Li2024_atlas__MF17', 'Li2024_atlas__MF18', 'Li2024_atlas__MF21', 'Li2024_atlas__MF22', 'Li2024_atlas__MF26', 'Li2024_atlas__MF27', 'Li2024_atlas__MF28', 'Li2024_atlas__MF30']
after drop: (322305, 49683)
                   size     sum
study                          
brunner2024       25778   12392
chennareddy2025  128927   69086
herrera2021         919     659
li2024           164547  114640
rindler2021_fi     2134    1371
```

```python  # cell 6
adata_cd4 = adata[adata.obs["cell_type_T"].isin(CD4_T_TYPES)].copy()

# drop ribosomal protein genes (RPS*/RPL*) — they dominate loadings and are uninformative here
ribo = adata_cd4.var_names.str.upper().str.startswith(("RPS", "RPL"))
print(f"removing {int(ribo.sum())} ribosomal protein genes (RPS*/RPL*)")
adata_cd4 = adata_cd4[:, ~ribo].copy()

adata_cd4.X = adata_cd4.layers["raw_counts"].copy()   # NB likelihood needs raw counts
print("CD4 non-Treg:", adata_cd4.shape)
print("cell_type_T:\n", adata_cd4.obs["cell_type_T"].value_counts())
print("\ntcr_is_malignant:", int(adata_cd4.obs["tcr_is_malignant"].astype(bool).sum()),
      "/", adata_cd4.n_obs)
print("by study:\n",
      adata_cd4.obs.groupby("study", observed=True)["tcr_is_malignant"].agg(["size", "sum"]))
del adata; gc.collect()
```

```
removing 105 ribosomal protein genes (RPS*/RPL*)
CD4 non-Treg: (222454, 49578)
cell_type_T:
 cell_type_T
CD4_Th2          125558
CD4_Cm            53650
CD4_mem           43005
CD4_cytotoxic       241
Name: count, dtype: int64

tcr_is_malignant: 71834 / 222454
by study:
                    size    sum
study                         
brunner2024       19845   4892
chennareddy2025   93859  30626
herrera2021         629    291
li2024           106560  35209
rindler2021_fi     1561    816

4297
```

```python  # cell 7
src = sc.read_h5ad(GENE_ID_SOURCE, backed="r")
sym2ens = dict(zip(src.var["gene_name"].astype(str), src.var["gene_id"].astype(str)))
adata_cd4.var["gene_id"] = [sym2ens.get(s, s) for s in adata_cd4.var_names.astype(str)]
n_mapped = int(sum(g.startswith("ENSG") for g in adata_cd4.var["gene_id"]))
print(f"gene_id mapped to Ensembl: {n_mapped}/{adata_cd4.n_vars}")
del src; gc.collect()
```

```
gene_id mapped to Ensembl: 9766/49578

0
```

```python  # cell 8
# Build the FULL-gene Geneformer map first (cached). Then HVG-subset adata + map together.
# Delete the .pt to force a rebuild.
semantic_map = get_or_build_geneformer_map(
    adata_cd4, SEMANTIC_CACHE_GENEFORMER, var_id_key="gene_id",
)
print("semantic_map (raw):", tuple(semantic_map.shape))

# Stale-cache guard: rebuild if the cached map rows don't match the current full-gene adata.
if semantic_map.shape[0] != adata_cd4.n_vars:
    print(f"shape mismatch ({semantic_map.shape[0]} vs {adata_cd4.n_vars}) — rebuilding")
    SEMANTIC_CACHE_GENEFORMER.unlink()
    semantic_map = get_or_build_geneformer_map(
        adata_cd4, SEMANTIC_CACHE_GENEFORMER, var_id_key="gene_id",
    )
    print("semantic_map (rebuilt):", tuple(semantic_map.shape))

# Keep only genes with a real Geneformer prior (non-zero embedding row). Out-of-vocab
# genes (~all non-coding RP11-*/AC0*/LINC*/MIR*) are zero-filled by the map builder and
# carry no semantic signal — drop them BEFORE HVG so HVG ranks variance only among
# semantically-grounded (effectively protein-coding) genes.
in_vocab = (semantic_map.norm(dim=1) > 0).cpu().numpy()
print(f"in-vocab (non-zero Geneformer row): {int(in_vocab.sum())}/{adata_cd4.n_vars}")
adata_cd4 = adata_cd4[:, in_vocab].copy()
semantic_map = semantic_map[torch.as_tensor(in_vocab)]

if HVG_TOP_N is not None:
    sc.pp.highly_variable_genes(adata_cd4, n_top_genes=HVG_TOP_N, flavor=HVG_FLAVOR, subset=False)
    keep = adata_cd4.var["highly_variable"].values
    adata_cd4 = adata_cd4[:, keep].copy()
    semantic_map = semantic_map[torch.as_tensor(keep)]
print("after HVG:", adata_cd4.shape, "| semantic_map:", tuple(semantic_map.shape))
```

```
Loading cached Geneformer map from /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/mf_cd4_atlas_geneformer.pt
semantic_map (raw): (49578, 1152)
in-vocab (non-zero Geneformer row): 9737/49578
after HVG: (222454, 2500) | semantic_map: (2500, 1152)
```

```python  # cell 9
# Subsample to one-third of CD4 cells for the sweep (seeded; shared across all variants).
if SUBSAMPLE_FRACTION is not None and SUBSAMPLE_FRACTION < 1.0:
    n_before = adata_cd4.n_obs
    sc.pp.subsample(adata_cd4, fraction=SUBSAMPLE_FRACTION, random_state=0)
    print(f"subsampled {n_before} -> {adata_cd4.n_obs} cells (fraction={SUBSAMPLE_FRACTION:.3f})")
    print("tcr_is_malignant after subsample:",
          int(adata_cd4.obs["tcr_is_malignant"].astype(bool).sum()), "/", adata_cd4.n_obs)
else:
    print(f"no subsampling (n_obs={adata_cd4.n_obs})")
```

```
subsampled 222454 -> 74151 cells (fraction=0.333)
tcr_is_malignant after subsample: 23962 / 74151
```

```python  # cell 10
# Build lib2_cd4.gmt once: curated HC_CD4_* block + CD4-filtered C7 IMMUNESIGDB sets.
import re

if LIB2_GMT.exists():
    print(f"lib2_cd4.gmt exists ({sum(1 for _ in LIB2_GMT.open())} sets) — skipping build")
else:
    HC_CD4 = {
        "HC_CD4_NAIVE":          ["CCR7", "SELL", "TCF7", "LEF1", "IL7R"],
        "HC_CD4_TH1":            ["TBX21", "IFNG", "CXCR3", "STAT1", "IL12RB2"],
        "HC_CD4_TH2":            ["GATA3", "IL4", "IL5", "IL13", "CCR4"],
        "HC_CD4_TH17":           ["RORC", "IL17A", "IL17F", "IL23R", "CCR6"],
        "HC_CD4_TREG":           ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF18"],
        "HC_CD4_TFH":            ["BCL6", "CXCR5", "PDCD1", "ICOS", "IL21"],
        "HC_CD4_CYTOTOXIC":      ["GZMB", "PRF1", "NKG7", "GNLY", "GZMK"],
        "HC_CD4_EXHAUSTION":     ["TOX", "LAG3", "TIGIT", "HAVCR2", "PDCD1"],
        "HC_CD4_ACTIVATION":     ["CD69", "IL2RA", "TNFRSF9", "CD40LG", "ICOS"],
        "HC_CD4_MALIGNANT_CTCL": ["TOX", "GATA3", "TWIST1", "KIR3DL2", "PLS3"],
    }
    KEEP = re.compile(r"CD4|TH1|TH2|TH17|TREG|TFH|TCONV|THELPER|T_HELPER|TFOLLICULAR")
    DROP_CD8 = re.compile(r"CD8")
    c7_lines = []
    for line in C7_SOURCE.open():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        name = parts[0]
        if not KEEP.search(name):
            continue
        if DROP_CD8.search(name) and "CD4" not in name:
            continue
        c7_lines.append(line.rstrip("\n"))
    with LIB2_GMT.open("w") as fh:
        for nm, genes in HC_CD4.items():
            fh.write("\t".join([nm, "NA", *genes]) + "\n")
        for line in c7_lines:
            fh.write(line + "\n")
    print(f"built {LIB2_GMT.name}: {len(HC_CD4)} curated + {len(c7_lines)} C7 = {len(HC_CD4)+len(c7_lines)} sets")
```

```
lib2_cd4.gmt exists (1415 sets) — skipping build
```

```python  # cell 11
# ---- write the slim job input (adata + semantic_map) + config for jobs/run_cd4_sweep.py ----
JOBS_DIR.mkdir(parents=True, exist_ok=True)
JOB_INPUT_DIR.mkdir(parents=True, exist_ok=True)

adata_cd4.write_h5ad(CD4_INPUT)
torch.save(semantic_map, SEM_MAP_PATH)
print(f"wrote input        -> {CD4_INPUT}  {adata_cd4.shape}")
print(f"wrote semantic_map -> {SEM_MAP_PATH}  {tuple(semantic_map.shape)}")

# name = report label (diff from baseline); cache_key = hash of absolute params.
job_variants = [
    {"name": _variant_name(v, BASELINE), "cache_key": _cache_slug(v), **{k: v[k] for k in _AXES}}
    for v in variants
]
notes = (
    f"SemanticSCVI sweep [Geneformer, CD4 CTCL atlas, projections-only]: {len(variants)} variant(s). "
    f"Fixed: n_latent={N_LATENT}, HVG_TOP_N={HVG_TOP_N}, SUBSAMPLE_FRACTION={SUBSAMPLE_FRACTION:.3f}, "
    f"PER_PROJECTION_N_TOP={PER_PROJECTION_N_TOP}. LLM judge: Opus only; MSigDB: lib1_immune + lib2_cd4."
)
cfg = {
    "input_h5ad":     str(CD4_INPUT),
    "semantic_map":   str(SEM_MAP_PATH),
    "model_cache_dir": str(MODEL_CACHE_DIR / SEM_CACHE_PARENT),
    "out_dir":        str(OUT_DIR),
    "lib1_gmt":       str(LIB1_GMT),
    "lib2_gmt":       str(LIB2_GMT),
    "pathway_index":  str(PATHWAY_INDEX),
    "batch_key":      BATCH_KEY,
    "labels_key":     "cell_type_T",
    "per_projection_n_top": PER_PROJECTION_N_TOP,
    "cell_type_context": ("CD4 T cells from cutaneous T-cell lymphoma (mycosis fungoides) skin, "
                          "spanning benign reactive and malignant clonal states"),
    "fixed_kwargs":   FIXED_KWARGS,
    "variants":       job_variants,
    "notes":          notes,
}
JOB_CONFIG.write_text(json.dumps(cfg, indent=2))
print(f"wrote job config   -> {JOB_CONFIG}  ({len(job_variants)} variants)")
```

```
wrote input        -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/benchmark_results/cd4_atlas_semantic_sweep/_job_input/cd4_sweep_input.h5ad  (74151, 2500)
wrote semantic_map -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/benchmark_results/cd4_atlas_semantic_sweep/_job_input/cd4_sweep_semantic_map.pt  (2500, 1152)
wrote job config   -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/jobs/cd4_sweep_config.json  (4 variants)
```

```python  # cell 12
# ---- submit the sweep bsub job (train all variants + projection benchmark + report on GPU) ----
# Guard against double-submit: skip only if a cd4_sweep job is already PEND/RUN.
_q = subprocess.run(["bjobs", "-J", "cd4_sweep"], capture_output=True, text=True).stdout
if any(s in _q for s in (" RUN ", " PEND ")):
    print("a cd4_sweep job is already active; not resubmitting:\n", _q)
else:
    r = subprocess.run(["bash", str(JOB_SH)], capture_output=True, text=True)
    print(r.stdout, end=""); print(r.stderr, end="")
    print("\nlog:", JOB_LOG)
    print("poll the next cell (re-run) until a new report HTML appears, then run the display cell")
```

```
Job <398023> is submitted to queue <gsla_high_gpu>.
Please note that memory reservation is per task (-n X)
Memory reservation is (MB): 128000 
===Your total amount of memory reservation for this job is (MB): 128000 ===

log: /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/jobs/run_cd4_sweep.bsub.log
poll the next cell (re-run) until a new report HTML appears, then run the display cell
```

```python  # cell 13
# ---- poll: re-run until the report HTML exists; shows queue status + per-variant cache + log tail ----
print("=== bjobs -a (cd4_sweep) ===")
print(subprocess.run(["bjobs", "-a", "-J", "cd4_sweep"], capture_output=True, text=True).stdout or "(no jobs)")

reports = sorted(OUT_DIR.glob("cd4_atlas_semantic_sweep_*.html"))
print(f"\nreport HTML: {'OK ' + reports[-1].name if reports else '... not yet'}")
for v in variants:
    name, key = _variant_name(v, BASELINE), _cache_slug(v)
    mp = MODEL_CACHE_DIR / SEM_CACHE_PARENT / key / "model.pt"
    print(f"  {'OK ' if mp.exists() else '...'} {name}  [{key}]")

if JOB_LOG.exists():
    print("\n=== tail", JOB_LOG.name, "===")
    print(subprocess.run(["tail", "-n", "30", str(JOB_LOG)], capture_output=True, text=True).stdout)
```

```
=== bjobs -a (cd4_sweep) ===
JOBID      USER    STAT  QUEUE      FROM_HOST   EXEC_HOST   JOB_NAME   SUBMIT_TIME
398023     zvise   DONE  gsla_high_ hgn39       dgn02       cd4_sweep  Jun 13 18:28


report HTML: OK cd4_atlas_semantic_sweep_20260613_190232.html
  OK  sem_baseline  [8f2e45825c]
  OK  sem_cw500  [b4acc56ec5]
  OK  sem_cw2000  [458e5a7f5c]
  OK  sem_cw3000  [55dc715ca8]

=== tail run_cd4_sweep.bsub.log ===
</home/projects/nyosef/zvise> was used as the home directory.
</home/projects/nyosef/zvise> was used as the working directory.
Started at Sat Jun 13 18:28:17 2026
Terminated at Sat Jun 13 19:02:34 2026
Results reported at Sat Jun 13 19:02:34 2026

Your job looked like:

------------------------------------------------------------
# LSBATCH: User input
/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/jobs/_run_cd4_sweep_inner.sh
------------------------------------------------------------

Successfully completed.

Resource usage summary:

    CPU time :                                   1954.00 sec.
    Max Memory :                                 3597 MB
    Average Memory :                             2823.84 MB
    Total Requested Memory :                     128000.00 MB
    Delta Memory :                               124403.00 MB
    Max Swap :                                   -
    Max Processes :                              7
    Max Threads :                                22
    Run time :                                   2062 sec.
    Turnaround time :                            2059 sec.

The output (if any) is above this job summary.
```

```python  # cell 14
# ---- display the report once the job has finished ----
reports = sorted(OUT_DIR.glob("cd4_atlas_semantic_sweep_*.html"))
if not reports:
    print("no report yet — submit the job and poll until the HTML appears")
else:
    report = reports[-1]
    print(f"report: {report}  ({report.stat().st_size / 1e6:.1f} MB)")
    from IPython.display import IFrame, display
    display(IFrame(str(report.relative_to(NB_DIR)), width="100%", height=700))
```

```
report: /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/benchmark_results/cd4_atlas_semantic_sweep/cd4_atlas_semantic_sweep_20260613_190232.html  (1.1 MB)

<IPython.lib.display.IFrame at 0x14800d2fd040>
```
