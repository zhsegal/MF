# old/25_delta_gene_axis.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/25_delta_gene_axis.ipynb` and is never run._

# 25 — the malignancy axis δ̂ on the v2 skin atlas: which genes build it, and what its 1-D projection can do

**Part A** builds the analysis object; **Parts 1 and 2** are the analysis. Part A is what is left of
`old/37_semantic_clonality_transfer` (`old/37_semantic_clonality_transfer.ipynb`), which fitted, in a 10-d latent, the
donor-averaged clonal direction

$$\hat\delta \;=\; \operatorname{mean}_d\Big[\ \overline{z}\,(\text{ALICE-clonal cells of } d)\ -\ \overline{z}\,(\text{benign anchors of } d)\ \Big]$$

over the TCR-labelled MF/SS skin donors and transferred malignancy onto the TCR-less ones. On the
**v1** cohort (70 donors, 31 labelled, 34 dark) that transfer scored leave-3-out per-donor
AUROC ≈ 0.94 against a shuffled-label null ≈ 0.51. This notebook runs the same machinery on the
**v2** atlas — `21_reannotation`'s re-annotation (539,916 T / 146 donors / 174 samples / 14 studies) and `23_malignancy_tcr_cnv`'s
v4 TCR call — and then asks the two questions `old/37_semantic_clonality_transfer` never did: what δ̂ *is* in gene space, and
whether the projection `s = z·δ̂/‖δ̂‖` works as an object. They are held to **different standards on
purpose**:

**Part 1 — what genes separate malignant from benign.** A discovery question: all-donor models,
every labelled donor, every diagnostic. Leakage is irrelevant because nothing here is a performance
claim. The gene map is exact rather than heuristic — these arms decode as
`px_scale = softmax_genes(W z + W_batch·onehot(batch))` with `W = softplus(·) ≥ 0`, so a step along
δ̂ shifts gene *g*'s logit by `(W δ̂)_g`. Two consequences drive everything below:

* **The softmax gauge.** Adding a constant to a column of `W` leaves the likelihood unchanged, so
  raw `A @ δ̂` carries a meaningless global offset and only gene *contrasts* are identifiable. Every
  score here is `A δ̂ − Σ_g p̄_g (A δ̂)_g` — the exact first-order `d log px_scale_g / dt`. Asserted in
  the loadings cell, and cross-checked against an assumption-free counterfactual decode in Fig 3.
* **Rank 10.** Ten factors, so the gene ranking has ten degrees of freedom, not 2500. Fig 1 and the
  factor table report that decomposition instead of implying more resolution than the model has.

**Part 2 — can the 1-D projection discriminate.** A performance claim, so it is leakage-free: each
donor is scored on the fold latent where *that donor was held out*, with δ̂ fitted only on that
fold's training donors. Raw `z` carries large donor offsets (the reason δ̂ is per-donor-differenced
in the first place), so separability is read *within* donor and the cost of the offset is measured
separately.

**Two batch arms.** `structural` is trained with `batch_key="study"`; `structural_donor` is the same
model with `batch_key="donor"`, so donor-level expression lands in the decoder's batch offset
instead of in the 10 latent dimensions. Donor-batch cannot be folded (a held-out donor's batch
category has no trained embedding), which is exactly why it belongs to part 1 only. Whether it helps
is reported as a number, not assumed.

## Part A — building the analysis object · HEAVY (compute kernel)

Everything below used to be `old/37_semantic_clonality_transfer` (`old/37_semantic_clonality_transfer.ipynb`). Only the data build
survives here: the cohort, the two label columns, the gene space + Geneformer map, the training
pool, the slim job input and the fold/job config. `old/37_semantic_clonality_transfer`'s five-arm benchmark
(`semantic` / `ldvae` / `scvi` / `structural_shuffled`) is **not** rebuilt — δ̂ needs `structural`
and `structural_donor`, and nothing else here reads the others.

**The v2 inputs.** `skin_T_tcr_annotated_v4.h5ad` is `23_malignancy_tcr_cnv`'s own Step 1 cache on the `21_reannotation` v2
re-annotation (539,916 T / 146 donors / 174 samples / 14 studies), with the Li2024 V(D)J folded into
`has_tcr` and raw counts in `layers["raw_counts"]`. Labels come from `alice_malignancy_v4.parquet`,
which carries **only `cell_id` and `tcr_clonal`** — so `tcr_is_expanded` is recomputed here on
`23_malignancy_tcr_cnv`'s own unified TRB-primary clone key rather than read from a parquet.

**The cohort gate is `23_malignancy_tcr_cnv`'s, not ours.** Keeping only cells present in the alice parquet inherits
every v2 gate in one line: `MF_gamma_delta`, `CD8_aggressive_epidermotropic_CTCL`, donors under 200
T cells, and the `D1__P303` duplicate of `D5__MFIVB` are already gone (492,653 of 539,916 rows / 114
donors). `CD4_Treg` stays out of the cohort for `23_malignancy_tcr_cnv`'s reason — reactive Tregs expand clonally too,
so a Treg dominant clone is not tumour evidence.

**Seven v2 cohorts have no V(D)J at all** (`brentuximab2026`, `lyp2026`, `rindler2021`, `alkon2024`,
`gaydosik2019`, `gaydosik2023`, `jonak2021`), and `23_malignancy_tcr_cnv` additionally withholds ALICE from any donor
whose CD4 repertoire is too thin to calibrate an OLGA null. For those donors `tcr_clonal` is False
because *nothing was tested*, not because the sample is clean — they must land in `dark` (the
transfer target), never in `no_clone` (the negative control). Cell A5 asserts exactly that.

```python  # cell 2
# ============================================================
# A1 — parameters: inputs, cohort, the two δ̂ arms, fold layout.
# ============================================================
import hashlib
import json
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
ATLAS = NB_DIR / "data" / "atlas_joint"

# ---- inputs (nb30 on the nb10b v2 re-annotation) ----
TCR_OBJ  = ATLAS / "skin_T_tcr_annotated_v4.h5ad"   # nb30 Step 1 cache (12 GB), layer raw_counts
ALICE    = ATLAS / "alice_malignancy_v4.parquet"    # cell_id, tcr_clonal — the only two columns
MALIG_V5 = ATLAS / "skin_T_malignancy_v5.parquet"   # optional: CNV columns, once nb30's v5 run lands
GENE_ID_SOURCE = NB_DIR / "data" / "cache" / "cnmf_malignant_counts.h5ad"   # symbol -> Ensembl
SEMANTIC_CACHE = NB_DIR / "data" / "mf_clonality_geneformer_v2.pt"          # full-gene Geneformer map
COUNTS = "raw_counts"          # nb30's cache renames nb10b's `counts` layer

# ---- outputs: a v2 tree, so the v1 (nb37-cohort) artifacts are never overwritten ----
OUT_DIR    = NB_DIR / "benchmark_results" / "delta_axis_v2"
JOB_INPUT  = OUT_DIR / "_job_input"
INPUT_H5AD = JOB_INPUT / "clonality_input.h5ad"     # written by A7; read back by Part 1 onward
MAP_PT     = JOB_INPUT / "clonality_semantic_map.pt"
CONFIG      = NB_DIR / "jobs" / "clonality_folds_config.json"
MODEL_CACHE = NB_DIR / "models" / ".model_cache_delta_axis_v2"
FIG_DIR     = NB_DIR / "figures"
for _d in (OUT_DIR, JOB_INPUT, MODEL_CACHE, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- cohort ----
CD4_TYPE = "CD4"                                    # cell_type_T; excludes CD4_Treg / CD8
# nb30's clone rule and ALICE cohort gate, reused verbatim: the benign anchors are defined by clone
# size, so they must be computed on the SAME clone key the malignant anchors came from, and a donor
# ALICE never ran on must not be read as ALICE-negative.
FRAC_THRESH, RATIO_THRESH, EXPANDED_MIN = 0.05, 1.33, 2
TCR_MIN_CELLS = 300                                 # nb30 gate 2: thinner repertoires cannot
TCR_EXEMPT_STUDIES = {"herrera2021"}                # calibrate an OLGA null (HC exempt too)

# ---- preprocessing / model (nb18 recipe, unchanged from the v1 run so the two are comparable) ----
HVG_TOP_N, HVG_FLAVOR = 2500, "seurat_v3"
N_LATENT   = 10
BATCH_KEY  = "study"        # NOT sample_id: a held-out donor's sample_id would be an unseen batch
LABELS_KEY = "cell_type_T"
MAX_EPOCHS, WARMUP_EPOCHS, KL_WARMUP = 100, 20, 100
GF_MIN_IN_VOCAB = 15_000

# ---- the STRUCTURAL decoder (Rounds A-E on haniffa_cd8) ----
# Loadings are W = softplus(s_k cos(S_g, alpha_k) - b_k), an explicit function of the gene embedding
# rather than a free matrix the semantic loss merely regularizes. Config comes from Round E's
# final_config.json when that sweep has run, else the frozen Round C default.
ROUND_E_CONFIG = (NB_DIR.parent / "benchmark_results" / "haniffa_cd8_roundE_sweep"
                  / "final_config.json")
_ROUND_C_STRUCT = dict(
    decoder_mode="structural", scale_mode="decoupled",
    logit_scale_init=15.0, logit_scale_min=5.0, logit_scale_max=30.0, learn_scale=True,
    b_mode="quantile", b_quantile=0.90, b_slack=0.0, b_detach=True,
    alpha_init="kmeans_expr", learn_alpha=True,
    missing_residual=False, usage_balance_weight=0.0, usage_balance_mode="entropy",
    n_latent=N_LATENT, n_layers=2, n_hidden=128, dropout_rate=0.1,
    gene_likelihood="nb", weights_positive=True, use_batch_norm=False,
)
if ROUND_E_CONFIG.exists():
    _re = json.loads(ROUND_E_CONFIG.read_text())
    STRUCT_KWARGS, STRUCT_WARMUP = dict(_re["kwargs"]), _re["warmup_epochs"]
    STRUCT_PROVENANCE = f"Round E '{_re['label']}'"
    if _re["embedding"] != "geneformer":
        raise ValueError(
            f"Round E selected the '{_re['embedding']}' embedding, but this notebook's semantic "
            f"map ({SEMANTIC_CACHE.name}) is Geneformer. Build the matching map first.")
else:
    STRUCT_KWARGS, STRUCT_WARMUP = dict(_ROUND_C_STRUCT), 0
    STRUCT_PROVENANCE = "Round C frozen default (Round E has not run)"

# The two arms δ̂ needs. `structural_donor` is the same model with batch_key="donor", so donor-level
# expression lands in the decoder's batch offset instead of in the 10 latent dimensions; it is
# all-donor ONLY, because a held-out donor's batch category would have no trained embedding.
MODEL_NAMES = ["structural", "structural_donor"]
MODELS = {
    "structural":       {"kwargs": STRUCT_KWARGS, "warmup_epochs": STRUCT_WARMUP},
    "structural_donor": {"kwargs": STRUCT_KWARGS, "warmup_epochs": STRUCT_WARMUP,
                         "batch_key": "donor"},
}
FOLDED_ARMS = ["structural"]        # arms that get per-fold runs on top of the all-donor reference

# ---- folds / scoring ----
SEED            = 0
N_NULL_DRAWS    = 20     # a shuffled δ̂ is a RANDOM DIRECTION in 10-d: sd ~0.37 per draw, so one
                         # draw is not a null. Read mean ± sd over draws, never a point estimate.
FOLD_SIZE       = 3      # leave-3-samples-out
TRAIN_FRAC      = 1 / 3  # per-donor training subsample (encoding always uses all cells)
TRAIN_MIN_CELLS = 500    # never subsample a donor below this
EVAL_MIN_CELLS  = 200    # holdout eligibility: per-donor metrics need enough of both classes
EVAL_MIN_CLONAL = 25
EVAL_MIN_ANCHOR = 25
print("structural config:", STRUCT_PROVENANCE)
```

```
NB_DIR = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
structural config: Round E 'geneformer_smax30'
```

```python  # cell 3
import gc
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import torch

for _p in (str(NB_DIR / "helpers"),):                  # MF/helpers
    if _p not in sys.path:
        sys.path.insert(0, _p)

import alice_helpers as A               # clonotype_table — nb30's ALICE cohort gate
import atlas_join_helpers as H          # clone_id_from_cdr3, used by recompute_dominant_clone
import semantic_clonality_helpers as SC # donor_label_table / make_triples / fit_delta
import skin_T_cnv_helpers as C          # recompute_dominant_clone — nb30's clone key, verbatim

np.random.seed(SEED)
sc.settings.verbosity = 1


def mem(tag=""):
    import resource
    print(f"[mem] {tag:<24} peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")
```

### A3 — cohort · HEAVY

Backed read → CD4 mask → `to_memory`, so the 12 GB object is never fully materialised.
`recompute_dominant_clone` runs on the **backed** object, before the subset: clone ids and sizes
must be counted over every TCR+ T cell (all lineages), and only the dominance test narrows to CD4.

```python  # cell 5
alice = pd.read_parquet(ALICE).set_index("cell_id")["tcr_clonal"].astype(bool)
print(f"alice baseline: {len(alice):,} cells | {int(alice.sum()):,} clonal")

assert TCR_OBJ.exists(), (
    f"{TCR_OBJ.name} is missing — run nb30 Step 1 first. It is the object that carries the Li2024 "
    "V(D)J fold-in; the plain skin_T_annotated.h5ad reports li2024 has_tcr = 0.074, not 0.491")
a = sc.read_h5ad(TCR_OBJ, backed="r")
is_cd4 = a.obs["cell_type_T"].astype(str).eq(CD4_TYPE).to_numpy()
is_li  = a.obs["study"].astype(str).eq("li2024").to_numpy()
print("cell_type_T:\n", a.obs["cell_type_T"].value_counts().to_string())

# nb30 cell 4, verbatim. Sets tcr_clone_id / tcr_clone_size / tcr_is_expanded /
# tcr_is_dominant_clone. The atlas ships its own `clone_size` built from each study's original
# clone key — using that would define the benign anchors on a different clone definition than the
# ALICE malignant anchors came from.
C.recompute_dominant_clone(a, H, is_li, FRAC_THRESH, RATIO_THRESH, EXPANDED_MIN, dom_mask=is_cd4)

# nb30 gate 2, recomputed here because it is NOT persisted in the alice parquet: a donor whose CD4
# repertoire is thinner than TCR_MIN_CELLS never had ALICE run, so its all-False `tcr_clonal` is
# "untested", not "clean". HC donors are exempt (they are the reference, never query patients).
clono_cd4 = A.clonotype_table(a.obs[is_cd4], group="donor")
n_tcr_cd4 = clono_cd4.groupby("donor", observed=True)["n_cells"].sum()
_dmeta = (a.obs[["donor", "study", "disease"]].astype(str)
          .groupby("donor", observed=True).agg("first"))
_exempt = _dmeta["study"].isin(TCR_EXEMPT_STUDIES) | _dmeta["disease"].eq("HC")
TCR_DONORS = sorted(_dmeta.index[(n_tcr_cd4.reindex(_dmeta.index).fillna(0) >= TCR_MIN_CELLS)
                                 | _exempt])
print(f"ALICE cohort: {len(TCR_DONORS)}/{len(_dmeta)} donors carry a testable CD4 repertoire")

keep = is_cd4 & a.obs_names.isin(alice.index)
print(f"{a.n_obs:,} T -> {int(keep.sum()):,} CD4 in nb30's gated cohort "
      f"({int((~a.obs_names.isin(alice.index)).sum()):,} cells belong to gated-out donors)")
adata = a[keep].to_memory()
a.file.close(); del a; gc.collect()

# NB likelihood wants raw counts. No .copy(): X takes the reference, then dropping the layer
# leaves one array instead of two — on this cohort the copy is several GB of nothing.
adata.X = adata.layers[COUNTS]
for _lay in list(adata.layers):
    del adata.layers[_lay]
gc.collect()
print(adata)
mem("after cohort")
```

```
alice baseline: 492,653 cells | 81,963 clonal
cell_type_T:
 cell_type_T
CD4         442982
CD8          55923
CD4_Treg     41011
ALICE cohort: 76/146 donors carry a testable CD4 repertoire
539,916 T -> 401,851 CD4 in nb30's gated cohort (47,263 cells belong to gated-out donors)
AnnData object with n_obs × n_vars = 401851 × 42347
    obs: 'cell_id', 'dataset', 'study', 'donor', 'real_donor', 'sample_id', 'sex', 'disease', 'disease_stage', 'compartment', 'tissue', 'tech', 'cell_type_orig', 'cell_type', 'n_genes', 'total_counts', 'pct_mito', 'doublet_score', 'predicted_doublet', 'patient_key', 'has_tcr', 'clone_id', 'tra_cdr3', 'trb_cdr3', 'clone_size', 'is_expanded', 'is_dominant_clone', 'is_malignant', 'repo', 'accession', 'organ', 'tissue_detail', 'malignant_call_method', 'tcr_available', 'malignant_labeled', 'malignant_frac', 'tcr_recovery', 'stage_class', 'stage_clean', 'lineage', 'entity', 'fmf_status', 'lineage_resolve', 'treatment_context', 'blood_involvement', 'lesion_type', 'cohort_country', 'cohort_region', 'origin_resolve', 'east_asian', 'lineage_orig', 'lineage_score_cd4', 'lineage_score_cd8', 'lineage_score_gd', 'lineage_score_ab', 'lineage_method', 'lineage_disagree', 'lineage_disagreement', 'leiden_broad', 'cell_type_broad', 'leiden_T', 'cell_type_T', '_li_clone_key', 'tcr_clone_id', 'tcr_clone_size', 'tcr_is_expanded', 'tcr_is_dominant_clone', 'tcr_is_malignant'
    obsm: 'X_mrvi_u', 'X_mrvi_z'
[mem] after cohort             peak RSS 64.4 GB
```

### A4 — the two label columns

```
is_clonal = tcr_clonal                                  # ALICE founder family (`23_malignancy_tcr_cnv` Step 3)
is_anchor = has_tcr & ~is_clonal & ~tcr_is_expanded      # TCR+, non-expanded, `23_malignancy_tcr_cnv`'s clone key
unlabelled = everything else                             # no TCR is NOT a negative
```

`has_tcr` is first narrowed to the ALICE cohort (`tcr_cohort`), with the raw flag kept as
`has_tcr_raw`. That single line is what keeps a donor ALICE never ran on out of the `no_clone`
negative-control group: with `has_tcr` False everywhere, `donor_label_table` counts `n_tcr = 0` and
files the donor under `dark`. Healthy controls are benign by definition, so their clonal calls are
cleared and their TCR+ cells become anchors.

```python  # cell 7
adata.obs["tcr_clonal"] = alice.reindex(adata.obs_names).to_numpy()
adata.obs["tcr_cohort"] = adata.obs["donor"].astype(str).isin(TCR_DONORS).to_numpy()
adata.obs["has_tcr_raw"] = adata.obs["has_tcr"].to_numpy(dtype=bool)
adata.obs["has_tcr"] = (adata.obs["has_tcr_raw"].to_numpy()
                        & adata.obs["tcr_cohort"].to_numpy())
_lost = int(adata.obs["has_tcr_raw"].sum() - adata.obs["has_tcr"].sum())
print(f"has_tcr: {int(adata.obs['has_tcr'].sum()):,} in the ALICE cohort "
      f"({_lost:,} TCR+ cells masked as untested)")

adata.obs["is_clonal"] = adata.obs["tcr_clonal"].to_numpy(dtype=bool)
adata.obs["is_anchor"] = (adata.obs["has_tcr"].to_numpy()
                          & ~adata.obs["is_clonal"].to_numpy()
                          & ~adata.obs["tcr_is_expanded"].to_numpy(dtype=bool))

hc = adata.obs["disease"].astype(str).eq("HC").to_numpy()
n_flip = int(adata.obs["is_clonal"].to_numpy()[hc].sum())
adata.obs.loc[hc, "is_clonal"] = False
adata.obs.loc[hc & adata.obs["has_tcr"].to_numpy(), "is_anchor"] = True

ic0 = adata.obs["is_clonal"].to_numpy(dtype=bool)
ia0 = adata.obs["is_anchor"].to_numpy(dtype=bool)
print(f"clonal {int(ic0.sum()):,} | anchor {int(ia0.sum()):,} | "
      f"unlabelled {int((~ic0 & ~ia0).sum()):,} (HC cells forced benign: {n_flip})")
```

```
has_tcr: 151,782 in the ALICE cohort (824 TCR+ cells masked as untested)
clonal 81,963 | anchor 34,883 | unlabelled 285,005 (HC cells forced benign: 0)
```

### A5 — donor roles

`labelled` / `labelled_small` fit δ̂, `no_clone` is the negative control, `dark` is the transfer
target. The assertion is the point of the cell: **no donor without a testable repertoire may be a
negative control.** On the v1 cohort every study had V(D)J, so this could not fire; on v2 seven
cohorts do not, and a silent misfile would turn "we never looked" into "we looked and found
nothing".

```python  # cell 9
donor_tbl = SC.donor_label_table(adata.obs, eval_min_cells=EVAL_MIN_CELLS,
                                 eval_min_clonal=EVAL_MIN_CLONAL,
                                 eval_min_anchor=EVAL_MIN_ANCHOR)
print(donor_tbl.groupby("group", observed=True)
      .agg(donors=("n_cells", "size"), cells=("n_cells", "sum"),
           eligible=("eligible", "sum")).to_string())

# a donor outside the ALICE cohort has no repertoire evidence either way -> it must be `dark`
_no_alice = sorted(set(donor_tbl.index.astype(str)) - set(TCR_DONORS))
_bad = [d for d in _no_alice if donor_tbl.loc[d, "group"] != "dark"]
assert not _bad, f"donors ALICE never ran on are not `dark`: {_bad[:8]}"
print(f"\n{len(_no_alice)} donors outside the ALICE cohort, all filed as `dark`")
print("\nper-study donor roles:\n",
      pd.crosstab(donor_tbl["study"], donor_tbl["group"]).to_string())

with pd.option_context("display.width", 200, "display.max_rows", 200):
    print(donor_tbl.to_string())
donor_tbl.to_csv(OUT_DIR / "donor_table.csv")
```

```
donors   cells  eligible
group                             
dark          70  181606         0
labelled      38  208178        38
no_clone       6   12067         0

62 donors outside the ALICE cohort, all filed as `dark`

per-study donor roles:
 group            dark  labelled  no_clone
study                                    
alkon2024          13         0         0
brentuximab2026     6         0         0
brunner2024         1         3         1
buus2025            0         2         1
chennareddy2025     0         7         3
gaydosik2019        5         0         0
gaydosik2022        5         3         0
gaydosik2023        2         0         0
herrera2021         0         1         0
il4ra2026           3         8         1
jonak2021           1         0         0
li2024             20        14         0
lyp2026             8         0         0
rindler2021_mc      6         0         0
                      n_cells  n_clonal  n_anchor  n_tcr            study     disease     group  clonal_frac  eligible
donor                                                                                                                 
D8__P65                  1241         0         0      0        alkon2024          MF      dark       0.0000     False
D8__P78                   356         0         0      0        alkon2024  CTCL_other      dark       0.0000     False
D8__P79                  1335         0         0      0        alkon2024  CTCL_other      dark       0.0000     False
D8__P88                   722         0         0      0        alkon2024  CTCL_other      dark       0.0000     False
D8__P90                   300         0         0      0        alkon2024          MF      dark       0.0000     False
D8__P102                 1212         0         0      0        alkon2024  CTCL_other      dark       0.0000     False
D8__P107                 1720         0         0      0        alkon2024          MF      dark       0.0000     False
D8__P138                 1241         0         0      0        alkon2024          MF      dark       0.0000     False
D8__P143                  724         0         0      0        alkon2024  CTCL_other      dark       0.0000     False
D8__P149                  591         0         0      0        alkon2024  CTCL_other      dark       0.0000     False
D8__P150                 3305         0         0      0        alkon2024          MF      dark       0.0000     False
D8__P163                 1298         0         0      0        alkon2024          MF      dark       0.0000     False
D8__P194                 1799         0         0      0        alkon2024  CTCL_other      dark       0.0000     False
D11__P1                   768         0         0      0  brentuximab2026          MF      dark       0.0000     False
D11__P2                 10536         0         0      0  brentuximab2026          MF      dark       0.0000     False
D11__P3                  9185         0         0      0  brentuximab2026          MF      dark       0.0000     False
D11__P4                   686         0         0      0  brentuximab2026          MF      dark       0.0000     False
D11__P5                  7920         0         0      0  brentuximab2026          MF      dark       0.0000     False
D11__P6                  5951         0         0      0  brentuximab2026          MF      dark       0.0000     False
D3__P87                   351         0         0      0      brunner2024  CTCL_other      dark       0.0000     False
D6__MF2                  1385         0         0      0     gaydosik2019          MF      dark       0.0000     False
D6__MF5                  5023         0         0      0     gaydosik2019          MF      dark       0.0000     False
D6__SC157                1384         0         0      0     gaydosik2019          MF      dark       0.0000     False
D6__SC158                 405         0         0      0     gaydosik2019          MF      dark       0.00
... [10722 more chars truncated]
```

### A6 — gene space + Geneformer map · HEAVY

Ribosomal genes out, symbols mapped to Ensembl (Geneformer's vocabulary is keyed by Ensembl id and
the atlas `var` is symbol-only), map built on the **full** gene panel, then subset to in-vocab genes
and to 2,500 HVGs. The map cache is `_v2.pt`: the v1 file has one row per v1 gene, and the
stale-cache guard below would rebuild it in place.

Coverage on the full ~42k-gene panel is capped at `vocab/n_vars ≈ 0.5` (most of the tail is
antisense/lncRNA/novel loci), so the builder's default `min_coverage=0.5` can never pass and is the
wrong guard. It is disabled, and the assertion is on the quantity that actually detects a key
mismatch — how much of Geneformer's 20,275-token vocabulary we hit.

```python  # cell 11
from benchmark_helpers import get_or_build_geneformer_map

src = sc.read_h5ad(GENE_ID_SOURCE, backed="r")
sym2ens = dict(zip(src.var["gene_name"].astype(str), src.var["gene_id"].astype(str)))
del src; gc.collect()

ribo = adata.var_names.str.upper().str.startswith(("RPS", "RPL"))
print(f"dropping {int(ribo.sum())} ribosomal protein genes")
adata = adata[:, ~ribo].copy()
adata.var["gene_id"] = [sym2ens.get(s, s) for s in adata.var_names.astype(str)]
adata.var["feature_name"] = adata.var_names.astype(str)
print(f"gene_id mapped to Ensembl: "
      f"{int(sum(g.startswith('ENSG') for g in adata.var['gene_id']))}/{adata.n_vars}")

GF_KW = dict(var_id_key="gene_id", symbol_key="feature_name", min_coverage=0.0)
# The builder stores the gene names alongside the map and raises on a mismatch rather than
# returning misaligned rows. Rebuild in that case: the gene set changing is expected here (the
# cohort gate moved), a silently misaligned map would not be.
try:
    semantic_map = get_or_build_geneformer_map(adata, SEMANTIC_CACHE, **GF_KW)
except ValueError as exc:
    print(f"stale cache ({exc}) — rebuilding")
    SEMANTIC_CACHE.unlink()
    semantic_map = get_or_build_geneformer_map(adata, SEMANTIC_CACHE, **GF_KW)
assert semantic_map.shape[0] == adata.n_vars, (semantic_map.shape, adata.n_vars)

in_vocab = (semantic_map.norm(dim=1) > 0).cpu().numpy()
print(f"in-vocab (non-zero Geneformer row): {int(in_vocab.sum())}/{adata.n_vars}")
assert int(in_vocab.sum()) >= GF_MIN_IN_VOCAB, (
    f"only {int(in_vocab.sum())} genes hit Geneformer's vocabulary — adata.var ids/symbols "
    "probably do not match it; a near-empty map silently disables the semantic prior")
adata = adata[:, in_vocab].copy()
semantic_map = semantic_map[torch.as_tensor(in_vocab)]

sc.pp.highly_variable_genes(adata, n_top_genes=HVG_TOP_N, flavor=HVG_FLAVOR, subset=False)
hv = adata.var["highly_variable"].to_numpy()
adata = adata[:, hv].copy()
semantic_map = semantic_map[torch.as_tensor(hv)]
print("after HVG:", adata.shape, "| map", tuple(semantic_map.shape))
gc.collect(); mem("after gene space")
```

```
dropping 105 ribosomal protein genes
gene_id mapped to Ensembl: 9766/42242
Building Geneformer map (will cache to /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/mf_clonality_geneformer_v2.pt)

Fetching 4 files: 100%|██████████| 4/4 [00:00<00:00, 128.48it/s]

Geneformer: 18228/42242 genes mapped (43.2%), 8491 via symbol->Ensembl, 24014 zero-filled
  unmapped (first 10): A1BG-AS1, A2M-AS1, A2ML1-AS1, AADACL2-AS1, AAED1, AARS, AATBC, AB015752.1, AB015752.3, AB019440.50
in-vocab (non-zero Geneformer row): 18228/42242
after HVG: (401851, 2500) | map (2500, 1152)
[mem] after gene space         peak RSS 67.4 GB
```

### A7 — training pool + the slim job input

The per-donor subsample only shrinks the *fit* set; every run encodes all cells, held-out and dark
included. The CNV columns join only if `23_malignancy_tcr_cnv`'s v5 run has landed — the arm-level dosage test in
Part 1 reads the GTF and needs none of them, so their absence costs one per-cell correlation.
Note the v3 → v5 rename: `cnv_malig_cluster` is gone; the v5 call is `cnv_arm_malignant` gated by
`cnv_callable`.

```python  # cell 13
rng = np.random.default_rng(SEED)
pool = np.zeros(adata.n_obs, dtype=bool)
donors = adata.obs["donor"].astype(str).to_numpy()
for d in np.unique(donors):
    idx = np.where(donors == d)[0]
    n = int(min(len(idx), max(TRAIN_MIN_CELLS, round(TRAIN_FRAC * len(idx)))))
    pool[rng.choice(idx, n, replace=False)] = True
adata.obs["train_pool"] = pool
print(f"train_pool {int(pool.sum()):,}/{adata.n_obs:,} cells ({pool.mean():.1%}) "
      f"across {len(np.unique(donors))} donors")

OBS_COLS = ["donor", "sample_id", "study", "disease", "entity", "cell_type_T",
            "has_tcr", "has_tcr_raw", "tcr_cohort", "tcr_is_expanded", "tcr_clonal",
            "is_clonal", "is_anchor", "train_pool"]
CNV_COLS = ["cnv_cell_score", "cnv_arm_malignant", "cnv_callable"]
if MALIG_V5.exists():
    # read whole-file, not columns=[...]: the parquet's index is `obs_name` and a column
    # projection can drop it, which would silently align the join by position
    _v5 = pd.read_parquet(MALIG_V5)
    assert set(CNV_COLS) <= set(_v5.columns), sorted(set(CNV_COLS) - set(_v5.columns))
    for c in CNV_COLS:
        adata.obs[c] = _v5[c].reindex(adata.obs_names).to_numpy()
    del _v5; gc.collect()
    OBS_COLS += CNV_COLS
    print(f"joined {MALIG_V5.name}: {CNV_COLS}")
else:
    print(f"{MALIG_V5.name} not on disk yet — the per-cell CNV correlation in Part 1 will be "
          "skipped (the arm-level dosage test does not need it)")

slim = sc.AnnData(
    X=adata.X.copy(),
    obs=adata.obs[[c for c in OBS_COLS if c in adata.obs.columns]].copy(),
    var=adata.var[["gene_id", "feature_name", "highly_variable"]].copy(),
)
for k in ("X_mrvi_u", "X_scVI"):
    if k in adata.obsm:
        slim.obsm[k] = np.asarray(adata.obsm[k], dtype=np.float32)
for c in ("donor", "sample_id", "study", "cell_type_T"):
    slim.obs[c] = slim.obs[c].astype(str).astype("category")

slim.write_h5ad(INPUT_H5AD)
torch.save(semantic_map, MAP_PT)
print("wrote", INPUT_H5AD, slim.shape, "| obsm:", list(slim.obsm))
print("wrote", MAP_PT, tuple(semantic_map.shape))
```

```
train_pool 144,737/401,851 cells (36.0%) across 114 donors
skin_T_malignancy_v5.parquet not on disk yet — the per-cell CNV correlation in Part 1 will be skipped (the arm-level dosage test does not need it)
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/benchmark_results/delta_axis_v2/_job_input/clonality_input.h5ad (401851, 2500) | obsm: ['X_mrvi_u']
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/benchmark_results/delta_axis_v2/_job_input/clonality_semantic_map.pt (2500, 1152)
```

### A8 — folds + job config

`make_triples` builds leave-3-out folds over the *eligible* labelled donors under one hard
constraint: no fold may hold out every donor of a `study`, because a held-out donor whose batch
category vanished from training has no trained batch embedding and cannot be encoded. The fold
count follows from the v2 cohort — it is not the v1 run's eight.

**The cache slug hashes the data, not just the knobs.** Checkpoint reuse is keyed on
`<model_cache_dir>/<arm>/<fold>` alone, so a v2 run with unchanged model kwargs would hash to the
v1 slug `81e1097a79`, find the v1 checkpoint, and skip training entirely — reporting v1 models on a
v2 cohort with no warning. That is the same filename-only-versioning trap that bit `23_malignancy_tcr_cnv`, so the
fingerprint below includes the cohort: shape, gene set, donor set and label counts.

```python  # cell 15
folds = SC.make_triples(donor_tbl, size=FOLD_SIZE, seed=SEED, batch_key=BATCH_KEY)
SC.check_batch_coverage(slim.obs, folds, batch_key=BATCH_KEY)
print(f"{len(folds)} folds of {sorted({len(f) for f in folds})} | "
      f"{len({d for f in folds for d in f})} donors tested once")
for k, f in enumerate(folds):
    print(f"  fold{k}: " + ", ".join(f"{d} ({donor_tbl.loc[d, 'n_clonal']}c/"
                                     f"{donor_tbl.loc[d, 'n_anchor']}a)" for d in f))

# δ̂ is fitted on every clone-bearing donor (fit_delta skips any that lack one of the classes);
# `eligible` — a strictly smaller set — governs who may be *held out*.
LABELLED_A = list(donor_tbl.index[donor_tbl["group"].isin(["labelled", "labelled_small"])])
DARK_A     = list(donor_tbl.index[donor_tbl["group"] == "dark"])
NO_CLONE_A = list(donor_tbl.index[donor_tbl["group"] == "no_clone"])
print(f"\nδ fitted on {len(LABELLED_A)} clone-bearing donors "
      f"({int(donor_tbl['eligible'].sum())} of them holdout-eligible) | "
      f"no_clone {len(NO_CLONE_A)} | dark {len(DARK_A)}")

# Latent-free fold gate — costs nothing, and catches the bugs worth catching before the GPU jobs.
_ic = slim.obs["is_clonal"].to_numpy(dtype=bool)
_ia = slim.obs["is_anchor"].to_numpy(dtype=bool)
_dn = slim.obs["donor"].astype(str).to_numpy()
held_all = [d for f in folds for d in f]
assert len(held_all) == len(set(held_all)), "a donor is held out by more than one fold"
assert set(held_all) <= set(donor_tbl.index[donor_tbl["eligible"]]), "an ineligible donor is held out"
for k, f in enumerate(folds):
    for h in f:
        lab = (_dn == h) & (_ic | _ia)
        y = _ic[lab]
        assert 0 < y.sum() < y.size, f"fold{k} {h}: labelled cells are one class ({y.sum()}/{y.size})"
print(f"OK: {len(folds)} disjoint folds, {len(held_all)} donors tested once, "
      "both classes present in every held-out donor")


def _cache_slug(n=10):
    """Stable hash of every param AND every cohort property a trained model depends on.

    Params alone are not enough: the cache path is <slug>/<arm>/<fold>, so an unchanged model
    config on a *different cohort* would reload the previous cohort's checkpoint. The v1 slug was
    81e1097a79; asserting against it below makes that failure loud instead of silent.
    """
    blob = json.dumps({
        "models": {k: MODELS[k] for k in sorted(MODELS)},
        "max_epochs": MAX_EPOCHS, "warmup_epochs": WARMUP_EPOCHS,
        "n_epochs_kl_warmup": KL_WARMUP, "hvg": HVG_TOP_N,
        "batch_key": BATCH_KEY, "labels_key": LABELS_KEY,
        "data": {
            "n_obs": int(slim.n_obs), "n_vars": int(slim.n_vars),
            "genes": hashlib.sha1(",".join(sorted(slim.var_names)).encode()).hexdigest(),
            "donors": hashlib.sha1(",".join(sorted(set(_dn))).encode()).hexdigest(),
            "n_clonal": int(_ic.sum()), "n_anchor": int(_ia.sum()),
        },
    }, default=str, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:n]


PARAM_SLUG = _cache_slug()
assert PARAM_SLUG != "81e1097a79", "slug collides with the v1 run — the fingerprint is not working"
print("\nparam slug:", PARAM_SLUG, "| model cache:", MODEL_CACHE / PARAM_SLUG)

cfg = {
    "input_h5ad": str(INPUT_H5AD),
    "semantic_map": str(MAP_PT),
    "model_cache_dir": str(MODEL_CACHE / PARAM_SLUG),
    "out_dir": str(OUT_DIR),
    "donor_key": SC.DONOR_KEY,
    "batch_key": BATCH_KEY,
    "labels_key": LABELS_KEY,
    "train_pool_key": "train_pool",
    "max_epochs": MAX_EPOCHS,
    "warmup_epochs": WARMUP_EPOCHS,
    "n_epochs_kl_warmup": KL_WARMUP,
    "models": MODELS,
    # one all-donor reference per arm, plus per-fold runs for the foldable arm only
    "runs": ([{"name": f"{m}_all", "cache_key": "all", "model": m, "held_out": []}
              for m in MODEL_NAMES]
             + [{"name": f"{m}_fold{k}", "cache_key": f"fold{k}", "model": m,
                 "held_out": list(held)}
                for m in FOLDED_ARMS for k, held in enumerate(folds)]),
    "folds": folds,
    "notes": (f"nb38 δ̂ gene axis (v2 skin atlas): leave-{FOLD_SIZE}-out over "
              f"{int(donor_tbl['eligible'].sum())} holdout-eligible donors ({len(folds)} folds) "
              f"+ an all-donor reference, x {len(MODEL_NAMES)} arms "
              f"({', '.join(MODEL_NAMES)}; structural_donor is all-donor only because a held-out "
              f"donor's batch category has no trained embedding). Geneformer geometric prior, "
              f"HVG={HVG_TOP_N}, n_latent={N_LATENT}, batch={BATCH_KEY}. "
              f"Cohort: {slim.n_obs} CD4 cells / {donor_tbl.shape[0]} donors from "
              f"alice_malignancy_v4 + skin_T_tcr_annotated_v4."),
}
CONFIG.write_text(json.dumps(cfg, indent=2))
print("wrote", CONFIG, f"| {len(cfg['runs'])} runs")
print(cfg["notes"])
```

```
make_triples: training-only (sole donor of its study): H__SS4 [herrera2021]
12 folds of [3, 4] | 37 donors tested once
  fold0: Li2024_atlas__PT50 (824c/362a), Li2024_atlas__CTCL3 (7456c/987a), Li2024_atlas__CTCL7 (3076c/6828a), D15__MF28 (30c/351a)
  fold1: Li2024_atlas__CTCL4 (736c/3684a), D3__P192 (2589c/205a), Li2024_atlas__CTCL1 (6728c/159a)
  fold2: D1__P312 (571c/376a), Li2024_atlas__CTCL5 (6347c/2219a), D15__MF26 (2380c/121a)
  fold3: D1__P204 (1992c/1814a), D13__MF18 (469c/375a), Li2024_atlas__CTCL8 (5196c/1180a)
  fold4: B5__PT09 (4967c/1210a), D15__MF36 (377c/675a), D1__PGS (1967c/169a)
  fold5: D1__P171 (4754c/190a), D3__P196 (335c/111a), Li2024_atlas__PT11 (1832c/101a)
  fold6: D1__P76 (4397c/804a), Li2024_atlas__PT47 (287c/63a), D15__MF35 (1628c/260a)
  fold7: D15__MF30 (243c/179a), Li2024_atlas__PT56 (1245c/1073a), B5__PT23 (3538c/185a)
  fold8: Li2024_atlas__PT53 (1235c/561a), D13__MF19 (184c/118a), D1__P311 (3456c/179a)
  fold9: D15__MF24 (129c/493a), Li2024_atlas__CTCL2 (1157c/1436a), Li2024_atlas__PT35 (3343c/50a)
  fold10: D13__MF17 (3226c/28a), D15__MF34 (78c/236a), D1__P303 (952c/46a)
  fold11: Li2024_atlas__CTCL6 (3104c/4264a), D15__MF48 (921c/130a), D3__P105 (56c/209a)

δ fitted on 38 clone-bearing donors (38 of them holdout-eligible) | no_clone 6 | dark 70
OK: 12 disjoint folds, 37 donors tested once, both classes present in every held-out donor

param slug: e969152b7a | model cache: /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/models/.model_cache_delta_axis_v2/e969152b7a
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/jobs/clonality_folds_config.json | 14 runs
nb38 δ̂ gene axis (v2 skin atlas): leave-3-out over 38 holdout-eligible donors (12 folds) + an all-donor reference, x 2 arms (structural, structural_donor; structural_donor is all-donor only because a held-out donor's batch category has no trained embedding). Geneformer geometric prior, HVG=2500, n_latent=10, batch=study. Cohort: 401851 CD4 cells / 114 donors from alice_malignancy_v4 + skin_T_tcr_annotated_v4.
```

### A9 — train the arms · HEAVY (bsub / GPU)

`run_clonality_folds.py` reads the config written above and saves `z_<run>.npy` (all cells, every
run) into `out_dir`. Nothing in this notebook trains anything.

```bash
cd notebooks/MF/jobs
./run_clonality_folds.sh --folds structural_all      # smoke test one run first
ARRAY=1 ./run_clonality_folds.sh                     # then everything, one bsub job per run
```

~6 GB and a few minutes of GPU per run. Part 1 below starts from `INPUT_H5AD` + those latents, so a
fresh kernel can begin at the next cell.

```python  # cell 17
import importlib, json, sys, warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# The kernel's cwd is wherever jupyter was started (often $HOME), so locate MF/helpers instead of
# assuming it. Everything downstream then comes from the helper module's own __file__.
_CAND = [*(p / "helpers" for p in [Path.cwd(), *Path.cwd().parents]),
         Path("/home/projects/nyosef/zvise/MF/helpers")]
_hit = next((p for p in _CAND if (p / "delta_axis_helpers.py").exists()), None)
if _hit is None:
    raise FileNotFoundError("cannot find MF/helpers/delta_axis_helpers.py from "
                            f"cwd={Path.cwd()} — cd into the MF tree and restart the kernel")
for p in (str(_hit.resolve()),):
    if p not in sys.path:
        sys.path.insert(0, p)

import delta_axis_helpers as DA
import semantic_clonality_helpers as SC
importlib.reload(SC); importlib.reload(DA)

NB_MF, NB = DA.NB_MF, DA.NB          # authoritative: derived from the helper module's __file__

SEED = 0
# Part A defines these; re-declared so a fresh kernel can start the analysis here.
N_NULL_DRAWS = globals().get("N_NULL_DRAWS", 20)
np.random.seed(SEED)
warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "axes.titlesize": 9,
                     "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7})

FIG = NB_MF / "figures"; FIG.mkdir(parents=True, exist_ok=True)
OUT = DA.OUT_DIR
print(f"NB_MF   = {NB_MF}\nfigures = {FIG}\noutputs = {OUT}")

adata, cfg, FOLDS, donor_tbl, LABELLED = DA.load_context()
donors_np, ic, ia = DA.label_arrays(adata.obs)
SYMBOL = adata.var["feature_name"].astype(str) if "feature_name" in adata.var else pd.Series(
    adata.var_names, index=adata.var_names)

HELD = sorted({d for f in FOLDS for d in f})
DARK = sorted(donor_tbl.index[donor_tbl["group"] == "dark"].astype(str))
print(f"\n{adata.n_obs:,} CD4 cells x {adata.n_vars} HVGs | {donor_tbl.shape[0]} donors")
print(f"delta-fitting donors: {len(LABELLED)} | held-out (part 2): {len(HELD)} in {len(FOLDS)} folds"
      f" | dark: {len(DARK)}")
print(f"clonal {int(ic.sum()):,} | anchor {int(ia.sum()):,} | unlabelled {int((~ic & ~ia).sum()):,}")
```

```
NB_MF   = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
figures = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures
outputs = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/benchmark_results/delta_axis_v2

401,851 CD4 cells x 2500 HVGs | 114 donors
delta-fitting donors: 38 | held-out (part 2): 37 in 12 folds | dark: 70
clonal 81,963 | anchor 34,883 | unlabelled 285,005
```

## Part 1 — what genes build δ̂

Load both batch arms' all-donor checkpoints and latents, and fit δ̂ (plus the per-donor `δ_d`
it averages) in each latent space. δ̂ lives in a specific model's latent, so it has to be refitted
per arm — it cannot be carried across. It is persisted here, per arm.

```python  # cell 19
ARMS = ["structural", "structural_donor"]
Z, MODELS, DELTAS, DELTAS_D = {}, {}, {}, {}

for arm in ARMS:
    try:
        z, model = DA.load_arm(adata, arm, "all", cfg=cfg)
    except FileNotFoundError as exc:
        print(f"SKIP {arm}: {exc}"); continue
    dd, dhat = DA.per_donor_deltas(z, adata.obs, LABELLED)
    Z[arm], MODELS[arm], DELTAS[arm], DELTAS_D[arm] = z, model, dhat, dd
    np.save(OUT / f"delta_{arm}_all.npy", dhat)
    print(f"  ||delta_hat|| = {np.linalg.norm(dhat):.3f} | per-donor ||delta_d||: "
          + ", ".join(f"{k}={v:.2f}" for k, v in
                      dd.apply(np.linalg.norm, axis=1).describe()[["mean", "std", "min", "max"]].items()))

if not Z:
    raise SystemExit("no arm available — submit jobs/run_clonality_folds.sh first")
HEAD = "structural_donor" if "structural_donor" in Z else "structural"
print(f"\nheadline arm for the gene signature: {HEAD}")
```

```
INFO     File                                                                                                      
         /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/models/.model_cache_delta_axis_v2/e969152b7
         a/structural/all/model.pt already downloaded                                                              
structural/all: batch=study n_batch=14 z=(401851, 10)
  ||delta_hat|| = 1.125 | per-donor ||delta_d||: mean=1.81, std=0.54, min=0.84, max=3.00
INFO     File                                                                                                      
         /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/models/.model_cache_delta_axis_v2/e969152b7
         a/structural_donor/all/model.pt already downloaded                                                        
structural_donor/all: batch=donor n_batch=114 z=(401851, 10)
  ||delta_hat|| = 1.065 | per-donor ||delta_d||: mean=1.42, std=0.53, min=0.70, max=2.58

headline arm for the gene signature: structural_donor
```

**Fig 0 — δ̂, to scale.** One picture of what Part 1 is about to decode into genes. Cells are centred
per donor (raw `z` carries large donor offsets — the reason δ̂ is a per-donor difference in the first
place) and projected onto the plane spanned by δ̂ and the leading direction orthogonal to it. That
makes the horizontal coordinate exactly `s = z·δ̂/‖δ̂‖` and the arrow exactly δ̂, at its true length —
which a UMAP could not do, since a straight arrow through a non-linear embedding means nothing.

Red = ALICE-clonal, blue = benign anchor, grey = the unlabelled CD4 cells δ̂ will eventually be used
to score. Part 1 asks what happens in **gene** space when a cell takes one step along that arrow.

```python  # cell 21
# ============================================================
# Fig 0 — δ̂ to scale in the plane it lives in. Explanatory; no number here is a result.
# ============================================================
PLOT_N = 120_000                      # scatter only — δ̂ is fitted on every cell

zh, dhat = Z[HEAD], DELTAS[HEAD]
u_hat = DA.unit(dhat)

_rng = np.random.default_rng(SEED)
SUB = np.sort(_rng.choice(adata.n_obs, min(PLOT_N, adata.n_obs), replace=False))
ics, ias, dns = ic[SUB], ia[SUB], donors_np[SUB]
oth = ~ics & ~ias

# donor-centred, then projected onto span(δ̂, leading orthogonal direction)
zc = zh[SUB] - pd.DataFrame(zh[SUB]).groupby(dns).transform("mean").to_numpy()
x = zc @ u_hat
r = zc - np.outer(x, u_hat)
w = np.linalg.svd(r - r.mean(0), full_matrices=False)[2][0]
y = zc @ w

C_CL, C_AN, C_BG, C_INK = "#c0392b", "#2c7fb8", "#e4e4e4", "#111111"
fig, ax = plt.subplots(figsize=(5.6, 4.8))
ax.axhline(0, lw=.5, c="#bbbbbb", zorder=1); ax.axvline(0, lw=.5, c="#bbbbbb", zorder=1)
ax.scatter(x[oth], y[oth], s=1.6, c=C_BG, lw=0, rasterized=True, zorder=2,
           label=f"unlabelled CD4 ({oth.sum():,})")
ax.scatter(x[ias], y[ias], s=2.4, c=C_AN, lw=0, alpha=.45, rasterized=True, zorder=3,
           label=f"benign anchor ({ias.sum():,})")
ax.scatter(x[ics], y[ics], s=2.4, c=C_CL, lw=0, alpha=.45, rasterized=True, zorder=4,
           label=f"ALICE-clonal ({ics.sum():,})")

L = float(np.linalg.norm(dhat))
ax.annotate("", xy=(L, 0), xytext=(0, 0), zorder=6,
            arrowprops=dict(arrowstyle="-|>,head_width=.28,head_length=.55", lw=2.6, color=C_INK))
ax.annotate(r"$\hat\delta$", xy=(L, 0), xytext=(4, 8), textcoords="offset points",
            fontsize=13, fontweight="bold", zorder=7)

_lim = lambda v, pad: (min(np.quantile(v, .002), 0) - pad, max(np.quantile(v, .998), 0) + pad)
ax.set_xlim(*_lim(np.append(x, L), .1 * L)); ax.set_ylim(*_lim(y, .1 * L))
ax.set_aspect("equal", adjustable="box")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_xlabel(r"along $\hat\delta$   ( $s = z\cdot\hat\delta/\|\hat\delta\|$, donor-centred )")
ax.set_ylabel(r"leading direction $\perp\ \hat\delta$")
ax.set_title(f"the malignancy axis: clonal cells sit further along $\\hat\\delta$\n"
             f"$\\|\\hat\\delta\\|$ = {L:.2f} over {len(LABELLED)} labelled donors  [{HEAD}]")
ax.legend(loc="upper left", markerscale=5, frameon=False, handletextpad=.2, borderpad=.1)
fig.tight_layout(); fig.savefig(FIG / "nb38_delta_schematic.png", dpi=200); plt.show()

print(f"mean s: anchor {x[ias].mean():+.3f} | clonal {x[ics].mean():+.3f} | "
      f"unlabelled {x[oth].mean():+.3f}   (donor-centred, {len(SUB):,} cells plotted)")
```

```
INFO:matplotlib.mathtext:Substituting symbol \perp from STIXGeneral
INFO:matplotlib.mathtext:Substituting symbol \perp from STIXGeneral
INFO:matplotlib.mathtext:Substituting symbol \perp from STIXGeneral
INFO:matplotlib.mathtext:Substituting symbol \perp from STIXGeneral
INFO:matplotlib.mathtext:Substituting symbol \perp from STIXGeneral

<Figure size 616x528 with 1 Axes>
mean s: anchor -0.512 | clonal +0.361 | unlabelled -0.042   (donor-centred, 120,000 cells plotted)
```

The loading matrix, the gauge fix, and the expression gate.

`use_batch_norm=False` for these arms, so `get_loadings()` is exactly `softplus(W) ≥ 0` — every sign
in `A @ δ̂` comes from δ̂ itself. `p̄` is the mean expressed fraction over the labelled cells: the
point the gene softmax is expanded at, and the weights the score is centred by. The gauge assertion
is the real check that the centring is right — adding a constant to any column of `A` is a likelihood
no-op and must leave the score alone.

**The expression gate is not cosmetic.** `gene_score` is `d log px_scale / dt`, a *relative* change,
so a gene sitting at a fraction of 1e-6 can top the ranking on nothing. That bites hardest under
`decoder_mode='structural'`, where `W` is a function of the frozen Geneformer embedding rather than
of expression: a gene this CD4 cohort never transcribes still gets a full loading row, so genes at
the extremes of the embedding geometry (keratins, neuroendocrine peptides) can carry large logit
swings that correspond to no detectable transcript. Two views are therefore reported side by side —
the relative score restricted to genes actually detected, and `p̄·s`, the change in *absolute*
expressed fraction, which downweights unexpressed genes by construction rather than by a threshold.

```python  # cell 23
P_BAR = DA.mean_fraction(adata.X, ic | ia)
MIN_DETECT = 0.05
EXPRESSED_M, DETECT = DA.expressed_mask(adata.X, ic | ia, min_detect=MIN_DETECT, p_bar=P_BAR)
EXPRESSED = pd.Series(EXPRESSED_M, index=SYMBOL.to_numpy())
DET = pd.Series(DETECT, index=SYMBOL.to_numpy())

A, SCORE, ABS = {}, {}, {}
for arm in Z:
    A[arm] = DA.loadings(MODELS[arm])
    assert A[arm].shape == (adata.n_vars, cfg["models"][arm]["kwargs"]["n_latent"]), A[arm].shape
    SCORE[arm] = DA.gene_score(A[arm], DELTAS[arm], P_BAR).rename(index=SYMBOL.to_dict())
    ABS[arm] = DA.abundance_change(SCORE[arm], P_BAR)
    resid = DA.gauge_residual(A[arm], DELTAS[arm], P_BAR)
    assert resid < 1e-6, f"{arm}: gauge residual {resid:.2e} — centring is wrong"
    print(f"{arm}: A{A[arm].shape} min={A[arm].values.min():.3g} | gauge residual {resid:.1e} | "
          f"score range [{SCORE[arm].min():+.3f}, {SCORE[arm].max():+.3f}]")
print(f"\np_bar sums to {P_BAR.sum():.6f}; top-expressed: "
      + ", ".join(SYMBOL.iloc[np.argsort(-P_BAR)[:6]].tolist()))
EXP_GENES = EXPRESSED.index[EXPRESSED.to_numpy()]
print(f"expressed gate: detected in >= {MIN_DETECT:.0%} of labelled cells -> "
      f"{int(EXPRESSED.sum())}/{len(EXPRESSED)} genes ({EXPRESSED.mean():.0%})")
```

```
structural: A(2500, 10) min=9.05e-05 | gauge residual 9.9e-09 | score range [-2.480, +3.669]
structural_donor: A(2500, 10) min=2.32e-07 | gauge residual 4.2e-09 | score range [-0.767, +4.608]

p_bar sums to 1.000000; top-expressed: VIM, FTL, MT-ATP6, S100A6, CRIP1, HSP90AA1
expressed gate: detected in >= 5% of labelled cells -> 1043/2500 genes (42%)
```

**Fig 1 — how many of the ten factors does δ̂ actually use?** `contribution_k = u_k · std_genes(A[:,k])`
is the gauge-invariant amount of gene-space movement δ̂ buys along factor *k*: the additive
per-column constant drops out of a standard deviation, and the reciprocal `column·c / z_k/c` rescaling
leaves the product fixed. A direction concentrated on one or two factors is a program; one spread
thinly over ten is the latent's overall geometry.

```python  # cell 25
# Contributions are computed for every arm (the CSV export in Part 1's last cell reads both);
# only the headline arm is plotted.
FC = {arm: DA.factor_contributions(A[arm], DELTAS[arm], Z[arm], module=MODELS[arm].module)
      for arm in Z}

fc = FC[HEAD].reindex(A[HEAD].columns)
pos = fc["contribution"] > 0
fig, ax = plt.subplots(figsize=(5, 3))
ax.axhline(0, lw=.8, c="#333333", zorder=3)
ax.bar(range(len(fc)), fc["contribution"], width=.68, zorder=2,
       color=np.where(pos, "#c0392b", "#2c7fb8"), edgecolor="white", linewidth=.6)
ax.set_xticks(range(len(fc)))
ax.set_xticklabels([c.replace("Z_", "") for c in fc.index])
ax.set_xlim(-.7, len(fc) - .3)
ax.grid(axis="y", lw=.4, c="#e6e6e6", zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "bottom"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="x", length=0)
ax.set_xlabel("latent factor")
ax.set_ylabel("contribution to $\\hat\\delta$\n$u_k\\cdot\\mathrm{std}_g A_{\\cdot k}$")
ax.set_title("factor contributions to $\\hat\\delta$")
fig.tight_layout(); fig.savefig(FIG / "nb38_factor_contributions.png", dpi=200); plt.show()

for arm, f in FC.items():
    print(f"\n{arm}{'  [plotted]' if arm == HEAD else ''} | top-2 factors carry "
          f"{f['abs_frac'].iloc[:2].sum():.0%} of the movement | "
          f"eff_n_factors = {f.attrs.get('eff_n_factors', float('nan')):.2f}")
    print(f[["delta_unit", "w_std", "contribution", "abs_frac", "amplitude"]].round(3).head(5).to_string())
```

```
<Figure size 550x330 with 1 Axes>

structural | top-2 factors carry 58% of the movement | eff_n_factors = 9.86
     delta_unit  w_std  contribution  abs_frac  amplitude
Z_4       0.578  0.469         0.271     0.299      0.392
Z_2      -0.747  0.341        -0.255     0.281      0.269
Z_0      -0.244  0.488        -0.119     0.131      0.324
Z_7      -0.147  0.496        -0.073     0.080      0.260
Z_8      -0.087  0.651        -0.057     0.062      0.285

structural_donor  [plotted] | top-2 factors carry 56% of the movement | eff_n_factors = 9.40
     delta_unit  w_std  contribution  abs_frac  amplitude
Z_2       0.929  0.383         0.356     0.453      0.284
Z_5      -0.226  0.379        -0.086     0.109      0.120
Z_0       0.184  0.409         0.075     0.096      0.178
Z_9      -0.088  0.746        -0.066     0.084      0.217
Z_3      -0.166  0.373        -0.062     0.079      0.120
```

Naming the programs δ̂ leans on — and checking they exist. Within one column the gauge constant is
shared by every gene, so the top-loading ranking is well defined even though the column's absolute
level is not. `+` = pushed up by malignancy, `−` = pushed down.

`support` is the dump-factor test: the mean detection rate of a factor's top-30 genes divided by the
cohort-wide mean. Under the structural decoder a factor can point at a corner of the embedding space
this cohort never transcribes — it will still look like a coherent program in the loadings while
carrying no transcript. A `support` well below 1 on a factor δ̂ leans on means that part of the gene
ranking is embedding geometry, not biology. The matched control for that failure mode is a
row-permuted embedding (`structural_shuffled` in the v1 run); it is not retrained here, so
`support_ratio` is the only guard in this notebook.

```python  # cell 27
fc = FC[HEAD]
SUPP = DA.factor_expression_support(A[HEAD], DETECT, n_top=30)
picks = fc.index[:4]
tbl = DA.top_genes_per_factor(A[HEAD], factors=picks, n=12).rename(columns={
    k: (f"{k} ({'+' if fc.loc[k, 'contribution'] > 0 else '-'}{abs(fc.loc[k, 'contribution']):.2f},"
        f" supp {SUPP.loc[k, 'support_ratio']:.2f})") for k in picks})
print(f"top-loading genes of the factors δ̂ moves most  [{HEAD}]")
print(tbl.replace(SYMBOL.to_dict()).to_string())

sup = SUPP.join(fc[["contribution", "abs_frac"]]).sort_values("abs_frac", ascending=False)
print("\nexpression support per factor (top-30 loading genes vs all genes):")
print(sup[["contribution", "abs_frac", "top_mean_detect", "all_gene_mean_detect",
           "support_ratio"]].round(3).to_string())
weak = sup[(sup["support_ratio"] < 0.5) & (sup["abs_frac"] > 0.05)]
if len(weak):
    print(f"\n!! δ̂ leans on {len(weak)} factor(s) whose top genes are barely expressed "
          f"({', '.join(weak.index)}) — {weak['abs_frac'].sum():.0%} of its movement. Read the "
          "expressed-gene panel of Fig 2, not the raw ranking.")
```

```
top-loading genes of the factors δ̂ moves most  [structural_donor]
   Z_2 (+0.36, supp 0.43) Z_5 (-0.09, supp 2.51) Z_0 (+0.08, supp 5.63) Z_9 (-0.07, supp 0.00)
1                   IL17F                   SELL                   MT1E                  LCE1B
2                   IL17A                   KLF2                   MT2A                 TRIM49
3                   FRMD5                    LTB                   NME1                  LCE3C
4                     IL9                  TXNIP                   MT1X                TRIM49C
5                 SLC28A3                 NBEAL2                 DCTPP1                  SPRR4
6                    IL21                  PGGHG                 LGALS1                  KRT37
7                    CCL1                  TRIB2                   MT1F                 MAGEB4
8                    IL22                 ATP2A3                   MCM5                TRIM49B
9                    EBI3                  NLRC5                  PRDX1               KRTAP4-3
10                 CXCL13                   CCR4                   ENO1              KRTAP21-2
11                   IL13                   CD27                   NME2                  LCE1A
12                  ZBED2                 MICAL1                 S100A6                  LCE2C

expression support per factor (top-30 loading genes vs all genes):
     contribution  abs_frac  top_mean_detect  all_gene_mean_detect  support_ratio
Z_2         0.356     0.453            0.035                 0.082          0.430
Z_5        -0.086     0.109            0.207                 0.082          2.513
Z_0         0.075     0.096            0.463                 0.082          5.630
Z_9        -0.066     0.084            0.000                 0.082          0.002
Z_3        -0.062     0.079            0.205                 0.082          2.495
Z_6        -0.055     0.071            0.240                 0.082          2.919
Z_8         0.042     0.053            0.035                 0.082          0.429
Z_1        -0.028     0.035            0.066                 0.082          0.804
Z_7        -0.014     0.018            0.251                 0.082          3.044
Z_4        -0.002     0.002            0.002                 0.082          0.021

!! δ̂ leans on 3 factor(s) whose top genes are barely expressed (Z_2, Z_9, Z_8) — 59% of its movement. Read the expressed-gene panel of Fig 2, not the raw ranking.
```

**Fig 2 — the malignancy gene signature, two views.** Left: relative change `d log px_scale / dt`
restricted to genes detected in ≥5% of labelled cells. Right: absolute change `p̄·s`, i.e. how much
expressed fraction each gene actually gains or loses — no threshold needed. A gene that is large in
both is a solid call; one large only on the left is a low-abundance gene, and one large only on the
right is a highly expressed gene moving by a small relative amount.

The unfiltered ranking is printed underneath rather than hidden, so the artifact the gate removes is
visible.

```python  # cell 29
n_show = 22
s_all = SCORE[HEAD]
s_exp = s_all[EXPRESSED.reindex(s_all.index).fillna(False).to_numpy()].sort_values()
ab = ABS[HEAD].sort_values()

fig, axes = plt.subplots(1, 2, figsize=(8.4, 6.4))
for ax, sel, xlab, ttl in (
        (axes[0], pd.concat([s_exp.head(n_show), s_exp.tail(n_show)]),
         "Δ log normalized expression per unit step along δ̂",
         f"relative change, expressed genes only\n({int(EXPRESSED.sum())} of {len(EXPRESSED)} genes)"),
        (axes[1], pd.concat([ab.head(n_show), ab.tail(n_show)]),
         "Δ expressed fraction  $\\bar p_g\\, s_g$",
         "absolute change, all genes\n(low-abundance genes downweighted by construction)")):
    ax.barh(range(len(sel)), sel.to_numpy(),
            color=["#2c7fb8" if v < 0 else "#c0392b" for v in sel])
    ax.set_yticks(range(len(sel))); ax.set_yticklabels(sel.index, fontsize=6.5)
    ax.axvline(0, lw=.6, c="k"); ax.set_ylim(-.8, len(sel) - .2)
    ax.set_xlabel(xlab); ax.set_title(ttl)
fig.suptitle(f"genes moved by the malignancy axis  [{HEAD}]", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "nb38_signature_top_genes.png"); plt.show()

print("EXPRESSED, relative  up  :", ", ".join(s_exp.tail(20).index[::-1]))
print("EXPRESSED, relative  down:", ", ".join(s_exp.head(20).index))
print("ABSOLUTE abundance   up  :", ", ".join(ab.tail(20).index[::-1]))
print("ABSOLUTE abundance   down:", ", ".join(ab.head(20).index))
print("\nfor contrast, the UNFILTERED relative ranking (what the gate removes):")
print("  up  :", ", ".join(s_all.nlargest(15).index),
      f"\n        median detection {DET.reindex(s_all.nlargest(15).index).median():.1%}")
print("  down:", ", ".join(s_all.nsmallest(15).index),
      f"\n        median detection {DET.reindex(s_all.nsmallest(15).index).median():.1%}")
```

```
<Figure size 924x704 with 2 Axes>
EXPRESSED, relative  up  : IL22, CXCL13, IL13, IL26, LMCD1, TNFSF11, TWIST1, IGFL2, ADAM12, CPM, ETV7, BCAT1, NTRK2, MEF2C, TNS3, PDLIM4, PTHLH, RDH10, TMPRSS3, P4HA2
EXPRESSED, relative  down: TXNIP, SELL, KLF2, ARHGAP15, DUSP1, SMCHD1, TCF7, CDC14A, FOS, PGGHG, CD7, LTB, LEF1, PDE3B, SORL1, PRKCH, NLRC5, CD69, SESN3, PASK
ABSOLUTE abundance   up  : CXCL13, GNLY, LGALS1, VIM, S100A6, IL22, GAPDH, MT2A, LGALS3, CRIP1, ENO1, PRDX1, PKM, LMNA, BATF, CD74, NME2, TNFRSF18, GZMB, S100A11
ABSOLUTE abundance   down: FTL, HLA-B, LTB, MT-ATP6, TXNIP, HLA-A, PTMA, FOS, CD7, MT-ND4, DUSP1, DNAJB1, HLA-C, CD69, DDIT4, SMCHD1, PABPC1, SELL, SAT1, KLF2

for contrast, the UNFILTERED relative ranking (what the gate removes):
  up  : IL17F, IL17A, FRMD5, SLC28A3, IL9, IL21, CCL1, IL22, EBI3, CXCL13, IL13, ZBED2, CHRNA6, IL26, IL5 
        median detection 1.6%
  down: TXNIP, SELL, KLF2, ARHGAP15, DUSP1, TRIM49C, LCE1B, TRIM49, SMCHD1, TCF7, CDC14A, FOS, PGGHG, CD7, SPRR4 
        median detection 23.8%
```

**Fig 3 — is reading the loadings legitimate?** The linear score is a first-order expansion. The
counterfactual decodes `μ_ben` and `μ_ben + δ̂` through the fitted `generative()` at each donor's own
batch code and takes the log2 ratio — no linearity assumption, and it absorbs the gene softmax, the
batch offset and any decoder curvature. High rank agreement means Fig 2 can be read as written; if
it is low, the signature must come from the decode route instead.

```python  # cell 31
CF = {arm: DA.counterfactual_lfc(MODELS[arm], Z[arm], adata.obs, DELTAS[arm], LABELLED)
           .rename(index=SYMBOL.to_dict()) for arm in Z}

cf, sc = CF[HEAD], SCORE[HEAD]
rho = spearmanr(sc.reindex(cf.index), cf).statistic
fig, ax = plt.subplots(figsize=(4, 3.4))
ax.scatter(sc.reindex(cf.index), cf, s=4, alpha=.35, lw=0, c="#34495e")
ax.axhline(0, lw=.5, c="grey"); ax.axvline(0, lw=.5, c="grey")
ax.set_xlabel("linear gauge-centred score  $A\\hat u - \\bar p^{\\top}A\\hat u$")
ax.set_ylabel("counterfactual $\\log_2$ FC\ndecode($\\mu_{ben}+\\hat\\delta$) / decode($\\mu_{ben}$)")
verdict = "linear reading is exact" if rho > .9 else "LINEAR READING FAILS — use the decode route"
ax.set_title(f"{verdict}  (Spearman ρ = {rho:.3f})")
fig.tight_layout(); fig.savefig(FIG / "nb38_linear_vs_counterfactual.png"); plt.show()

print(f"rho = {rho:.4f} over {len(cf)} genes | "
      f"{spearmanr(sc.reindex(EXP_GENES), cf.reindex(EXP_GENES)).statistic:.4f} over expressed | "
      f"top-50 overlap {len(set(sc.nlargest(50).index) & set(cf.nlargest(50).index))}/50")
if rho <= 0.9:
    print("!! Fig 2 is not trustworthy at this ||delta||; rebuild the signature from CF[HEAD].")
```

```
<Figure size 440x374 with 1 Axes>
rho = 0.9909 over 2500 genes | 1.0000 over expressed | top-50 overlap 49/50
```

**Fig 4 — does correcting for donor change what malignancy looks like?** Same model, same cells, same
δ̂ recipe; the only difference is whether the decoder absorbs donor-level expression (`batch=donor`)
or only study-level (`batch=study`). If the two signatures agree, the study-batch latent was not
smuggling donor identity into the malignancy axis.

```python  # cell 33
if len(SCORE) < 2:
    print("only one arm available — train structural_donor to run this comparison")
else:
    a, b = "structural", "structural_donor"
    common = SCORE[a].index.intersection(SCORE[b].index).intersection(EXP_GENES)
    x, y = SCORE[a].reindex(common), SCORE[b].reindex(common)
    rho = spearmanr(x, y).statistic
    ov = len(set(x.nlargest(50).index) & set(y.nlargest(50).index))
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.scatter(x, y, s=4, alpha=.35, lw=0, c="#34495e")
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, lw=.6, ls="--", c="grey")
    ax.set_xlabel(f"{a}  (batch = study)"); ax.set_ylabel(f"{b}  (batch = donor)")
    ax.set_title(f"donor-batch {'agrees with' if rho > .7 else 'DISAGREES with'} study-batch\n"
                 f"ρ = {rho:.3f} over {len(common)} expressed genes, top-50 up overlap {ov}/50")
    fig.tight_layout(); fig.savefig(FIG / "nb38_study_vs_donor_batch.png"); plt.show()
    print("only in donor-batch top-50 up:",
          ", ".join(sorted(set(y.nlargest(50).index) - set(x.nlargest(50).index))[:15]))
    print("only in study-batch top-50 up:",
          ", ".join(sorted(set(x.nlargest(50).index) - set(y.nlargest(50).index))[:15]))
```

```
<Figure size 418x374 with 1 Axes>
only in donor-batch top-50 up: CAMK1, CD70, CD9, COL6A3, CSF2, GNG4, HDAC9, IGFL2, IL13, IL22, INPP5F, KCNN4, KRT7, LAIR2, NRN1
only in study-batch top-50 up: ATF5, CDKN2A, CHN1, CTNNAL1, DNM3, GNAI1, HILPDA, HMOX1, LAYN, PDE7B, PDGFA, PLS3, PTH2R, PTPN13, RNF144B
```

**Fig 5 — is δ̂ one shared direction, or an average over donors that disagree?** δ̂ is a mean of many
`δ_d`; each of those gives its own gene signature. Correlating them across donors says whether
malignancy looks the same in every patient. Donors are ordered by clonal fraction, which is also the
sensitivity axis: three donors sit at 0.84–0.88 clonal with only ~110–160 anchor cells, so their
benign centroid — and under `batch=donor`, their batch offset — is estimated from very few cells.

```python  # cell 35
SD = {arm: DA.per_donor_gene_scores(A[arm], DELTAS_D[arm], P_BAR).rename(columns=SYMBOL.to_dict())
      for arm in Z}
SD = {arm: v[EXP_GENES] for arm, v in SD.items()}      # expressed genes only, as in Fig 2
frac = donor_tbl["clonal_frac"].reindex(SD[HEAD].index).astype(float)
order = frac.sort_values().index
C = SD[HEAD].loc[order].T.corr(method="spearman")

fig, ax = plt.subplots(figsize=(5.4, 4.6))
im = ax.imshow(C.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(order))); ax.set_xticklabels(
    [f"{d}  {frac[d]:.2f}" for d in order], rotation=90, fontsize=5)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=5)
off = C.to_numpy()[~np.eye(len(C), dtype=bool)]
ax.set_title(f"per-donor signatures agree: median pairwise ρ = {np.median(off):.2f}  [{HEAD}]\n"
             "donors ordered by clonal fraction (shown on x)")
fig.colorbar(im, ax=ax, shrink=.7, label="Spearman ρ")
fig.tight_layout(); fig.savefig(FIG / "nb38_per_donor_signature_corr.png"); plt.show()

LOW = [d for d in DELTAS_D[HEAD].index if frac.get(d, 1.0) <= 0.7]
_, d_low = DA.per_donor_deltas(Z[HEAD], adata.obs, LOW)
s_low = DA.gene_score(A[HEAD], d_low, P_BAR).rename(index=SYMBOL.to_dict())
print(f"sensitivity: refit on the {len(LOW)} donors with clonal_frac <= 0.7 -> "
      f"cos(δ̂, δ̂_low) = {float(DA.unit(DELTAS[HEAD]) @ DA.unit(d_low)):.3f}, "
      f"signature ρ = {spearmanr(SCORE[HEAD].reindex(EXP_GENES), s_low.reindex(EXP_GENES)).statistic:.3f}"
      " (expressed genes)")
worst = C.mean().nsmallest(4).round(2)
print("least-agreeing donors (mean ρ to the rest):")
print(pd.DataFrame({"mean_rho": worst, "clonal_frac": frac.reindex(worst.index),
                    "n_anchor": donor_tbl["n_anchor"].reindex(worst.index)}).to_string())
```

```
<Figure size 594x506 with 2 Axes>
sensitivity: refit on the 26 donors with clonal_frac <= 0.7 -> cos(δ̂, δ̂_low) = 0.992, signature ρ = 0.973 (expressed genes)
least-agreeing donors (mean ρ to the rest):
           mean_rho  clonal_frac  n_anchor
D15__MF26      0.06       0.7199       121
D13__MF17      0.13       0.8598        28
D3__P105       0.28       0.0885       209
D1__P311       0.31       0.7238       179
```

**Fig 6 — strong *and* reproducible.** The averaged δ̂ cannot tell a gene every donor moves a little
from a gene two donors move a lot. `t = mean_d / (sd_d/√n)` over the per-donor signatures can.
`frac_same_sign` is the nonparametric version, and the one to trust when a donor is an outlier.

```python  # cell 37
CONS = DA.consistency(SD[HEAD])                # SD is already restricted to expressed genes
CONS["score"] = SCORE[HEAD].reindex(CONS.index)
CONS["detect"] = DET.reindex(CONS.index)
top = pd.concat([CONS.nlargest(15, "t"), CONS.nsmallest(15, "t")])

fig, ax = plt.subplots(figsize=(4.2, 5.4))
ax.barh(range(len(top)), top["t"], color=["#2c7fb8" if v < 0 else "#c0392b" for v in top["t"]])
ax.set_yticks(range(len(top))); ax.set_yticklabels(
    [f"{g}  ({f:.0%})" for g, f in zip(top.index, top["frac_same_sign"])])
ax.axvline(0, lw=.6, c="k"); ax.set_ylim(-.8, len(top) - .2)
ax.set_xlabel(f"t across the {SD[HEAD].shape[0]} per-donor signatures")
ax.set_title("most reproducible malignancy genes\n(label = gene, % of donors agreeing on sign)")
fig.tight_layout(); fig.savefig(FIG / "nb38_consistency_top_genes.png"); plt.show()

print(f"rank agreement between |score| and |t|: ρ = "
      f"{spearmanr(CONS['score'].abs(), CONS['abs_t']).statistic:.3f}")
strong = CONS.loc[CONS["score"].abs().nlargest(100).index]
flaky = strong[strong["frac_same_sign"] < 0.8]
print(f"strong but inconsistent ({len(flaky)} of the top-100 |score| genes have "
      f"< 80% donor sign agreement): " + ", ".join(flaky.index[:12]))
```

```
<Figure size 462x594 with 1 Axes>
rank agreement between |score| and |t|: ρ = 0.849
strong but inconsistent (6 of the top-100 |score| genes have < 80% donor sign agreement): TMEM200A, TCF7, LEF1, NLRC5, SESN3, PASK
```

**Fig 7 — the model against the data.** Paired per-donor pseudobulk: each donor contributes
`log2(mean CP10K | clonal) − log2(mean CP10K | anchor)`, so the contrast is within-donor by
construction and donor composition cannot drive it. This is the model-free reference. The shuffled-δ̂
arm is the null: permuting clonal/anchor within each training donor must destroy the agreement.

```python  # cell 39
PB, PB_donor = DA.pseudobulk_lfc(adata.X, adata.obs, LABELLED, var_names=SYMBOL.to_numpy())
common = SCORE[HEAD].index.intersection(PB.index).intersection(EXP_GENES)
x, y = SCORE[HEAD].reindex(common), PB["logfc"].reindex(common)
rho = spearmanr(x, y).statistic
rho_unfiltered = spearmanr(SCORE[HEAD].reindex(PB.index), PB["logfc"]).statistic

_, d_null = DA.per_donor_deltas(Z[HEAD], adata.obs, LABELLED, shuffle_seed=SEED + 7)
s_null = DA.gene_score(A[HEAD], d_null, P_BAR).rename(index=SYMBOL.to_dict()).reindex(common)
rho_null = spearmanr(s_null, y).statistic

fig, ax = plt.subplots(figsize=(4.2, 3.6))
ax.scatter(x, y, s=4, alpha=.3, lw=0, c="#34495e")
for g in list(x.nlargest(6).index) + list(x.nsmallest(6).index):
    ax.annotate(g, (x[g], y[g]), fontsize=6, alpha=.9)
ax.axhline(0, lw=.5, c="grey"); ax.axvline(0, lw=.5, c="grey")
ax.set_xlabel("δ̂ gene score (model)")
ax.set_ylabel(f"paired pseudobulk $\\log_2$FC\nclonal vs anchor, {PB_donor.shape[0]} donors")
ax.set_title(f"the latent axis recovers the data-level contrast\n"
             f"ρ = {rho:.3f} over {len(common)} expressed genes   (shuffled-δ̂ null ρ = {rho_null:+.3f})")
fig.tight_layout(); fig.savefig(FIG / "nb38_score_vs_pseudobulk.png"); plt.show()

print(f"rho = {rho:.3f} (expressed) | {rho_unfiltered:.3f} (all {len(PB)} genes) | "
      f"shuffled-δ̂ null = {rho_null:+.3f}")
print(f"genes with q<0.05 in the paired test: {int((PB['qvalue'] < 0.05).sum())}/{len(PB)}")
```

```
<Figure size 462x396 with 1 Axes>
rho = 0.690 (expressed) | 0.539 (all 2500 genes) | shuffled-δ̂ null = -0.250
genes with q<0.05 in the paired test: 0/2500
```

Where the model and the data disagree — what the 10-d bottleneck adds and what it drops. The axis is
a rank-10 projection, so it denoises toward factor-coherent programs and cannot represent a gene
that moves on its own.

```python  # cell 41
rank_s = SCORE[HEAD].reindex(common).rank(pct=True)
rank_p = PB["logfc"].reindex(common).rank(pct=True)
gap = (rank_s - rank_p).rename("rank_gap")
tab = pd.DataFrame({"delta_score": SCORE[HEAD].reindex(common).round(3),
                    "pseudobulk_logfc": PB["logfc"].reindex(common).round(3),
                    "q": PB["qvalue"].reindex(common),
                    "frac_donors_up": PB["frac_up"].reindex(common).round(2),
                    "rank_gap": gap.round(3)})
print("model says UP, data does not (top rank_gap):")
print(tab.nlargest(12, "rank_gap").to_string())
print("\ndata says UP, model does not (most negative rank_gap):")
print(tab.nsmallest(12, "rank_gap").to_string())
```

```
model says UP, data does not (top rank_gap):
          delta_score  pseudobulk_logfc   q  frac_donors_up  rank_gap
IL2RA           0.491            -0.983 NaN            0.24     0.859
TNFRSF9         0.562            -0.451 NaN            0.34     0.741
GNA15           0.300            -0.640 NaN            0.39     0.717
GZMB            0.489            -0.418 NaN            0.34     0.706
NPDC1           0.258            -0.618 NaN            0.37     0.678
TNFRSF4         0.376            -0.436 NaN            0.42     0.677
LAG3            0.164            -1.201 NaN            0.21     0.671
HPGD            0.189            -0.821 NaN            0.29     0.654
HLA-DQA1        0.305            -0.374 NaN            0.16     0.614
CXCR6           0.114            -1.202 NaN            0.21     0.613
ADAM19          0.218            -0.511 NaN            0.37     0.602
LTA             0.480            -0.211 NaN            0.39     0.576

data says UP, model does not (most negative rank_gap):
        delta_score  pseudobulk_logfc   q  frac_donors_up  rank_gap
PNMA5        -0.249             0.504 NaN            0.63    -0.718
ESYT2        -0.193             0.558 NaN            0.79    -0.659
TNFSF9       -0.157             0.570 NaN            0.79    -0.616
MRPS7        -0.308             0.207 NaN            0.53    -0.613
KLRC1        -0.312             0.190 NaN            0.29    -0.606
MRPL52       -0.306             0.190 NaN            0.42    -0.600
PI16         -0.500             0.073 NaN            0.55    -0.586
RXRA         -0.143             0.531 NaN            0.74    -0.580
MRPS33       -0.277             0.151 NaN            0.45    -0.547
RHOB         -0.120             0.516 NaN            0.66    -0.544
SFMBT2       -0.178             0.320 NaN            0.66    -0.544
SPON2        -0.283             0.135 NaN            0.50    -0.541
```

**Fig 8 — a transcriptional program, or copy-number dosage?** MF/SS carries recurrent arm-level CNV
(8q/17q/7q gain, 10q/13q/17p/9p loss). If δ̂ were tracking dosage, its gene score would be shifted
coherently across whole chromosome arms; a program need not be. The permutation holds the score
distribution and the arm sizes fixed and only reassigns genes to arms, so the p-value asks exactly
whether *this* arm's genes are unusually shifted. The per-cell correlation with `23_malignancy_tcr_cnv`'s `cnv_cell_score`
is the same question asked of the projection rather than the loadings.

```python  # cell 43
ARM_OF = DA.arm_annotate(SYMBOL.to_numpy())
AE = DA.arm_enrichment(SCORE[HEAD].reindex(EXP_GENES), ARM_OF, n_perm=2000, seed=SEED)

top = AE.head(14).iloc[::-1]
fig, ax = plt.subplots(figsize=(4.4, 4))
ax.barh(range(len(top)), top["delta_vs_all"],
        color=["#c0392b" if k else "#95a5a6" for k in top["known_ctcl"].ne("")])
ax.set_yticks(range(len(top)))
ax.set_yticklabels([f"{a}  n={n}{'  ' + k if k else ''}"
                    for a, n, k in zip(top.index, top["n_genes"], top["known_ctcl"])])
ax.axvline(0, lw=.6, c="k")
sig_arms = AE.query("qvalue < 0.05")
ax.set_xlabel("mean δ̂ score of the arm's genes − genome-wide mean")
ax.set_title(f"{len(sig_arms)} of {len(AE)} arms shifted at q<0.05\n"
             "red = recurrent CTCL arm; a program need not be arm-coherent")
fig.tight_layout(); fig.savefig(FIG / "nb38_arm_shift.png"); plt.show()

print(AE.head(10).round(4).to_string())
s_cell = SC.project(Z[HEAD], DELTAS[HEAD])
cnv = adata.obs["cnv_cell_score"].to_numpy(dtype=float) if "cnv_cell_score" in adata.obs else None
if cnv is not None:
    ok = np.isfinite(cnv) & (ic | ia)
    print(f"\nper-cell rho(s, cnv_cell_score) over labelled cells = "
          f"{spearmanr(s_cell[ok], cnv[ok]).statistic:.3f}  (n={int(ok.sum()):,})")
else:
    print("\nno cnv_cell_score in the slim input — nb30's v5 CNV run had not landed when Part A "
          "was built. The arm-level test above is unaffected (it reads the GTF, not nb30).")
```

```
INFO:root:Extracted GTF attributes: ['gene_id', 'gene_version', 'gene_name', 'gene_source', 'gene_biotype', 'transcript_id', 'transcript_version', 'transcript_name', 'transcript_source', 'transcript_biotype', 'tag', 'transcript_support_level', 'exon_number', 'exon_id', 'exon_version', 'protein_id', 'protein_version', 'ccds_id']

WARNING: GTF file misses annotation for 1 genes in adata.
WARNING: Skipped 4 genes because of duplicate identifiers in GTF file.
arm annotation: 2489/2500 genes placed (99.6%)

<Figure size 484x440 with 1 Axes>
        n_genes  mean_score  delta_vs_all  pvalue  qvalue known_ctcl
arm                                                                 
chr7p        27      0.2562        0.2160  0.0040  0.1159           
chr12q       45      0.1474        0.1073  0.0340  0.3478           
chr20q       15      0.2341        0.1940  0.0360  0.3478           
chr2q        64     -0.0225       -0.0627  0.1659  0.7609           
chr12p       25      0.1487        0.1085  0.1164  0.7609           
chr15q       24      0.1317        0.0915  0.2099  0.7609           
chr1p        55     -0.0208       -0.0610  0.1969  0.7609           
chr11q       39     -0.0352       -0.0754  0.1859  0.7609           
chr19q       29     -0.0005       -0.0407  0.5577  0.8986           
chr10q       21      0.1083        0.0681  0.3818  0.8986       loss

no cnv_cell_score in the slim input — nb30's v5 CNV run had not landed when Part A was built. The arm-level test above is unaffected (it reads the GTF, not nb30).
```

Pathway enrichment of the two ends of the signature (hallmark + C2 immune + C7 IMMUNESIGDB),
universe = the 2500 HVGs the model was actually trained on.

```python  # cell 45
s_exp = SCORE[HEAD].reindex(EXP_GENES)
UNI = set(s_exp.index)                      # universe = expressed HVGs, matching the gene lists
for label, genes in (("UP along δ̂", s_exp.nlargest(100).index),
                     ("DOWN along δ̂", s_exp.nsmallest(100).index)):
    e = DA.enrichment(genes, UNI, top_n=6)
    print(f"\n=== {label} (top 100 genes) ===")
    print(e[["library", "gene_set", "ER", "qvalue", "overlap", "set_size"]].round(4).to_string(index=False)
          if len(e) else "  no enriched set")
```

```
=== UP along δ̂ (top 100 genes) ===
  library                                                 gene_set     ER  qvalue  overlap  set_size
C2_immune                 WP_CYTOKINECYTOKINE_RECEPTOR_INTERACTION 3.5853  0.0081       11        32
C2_immune              KEGG_CYTOKINE_CYTOKINE_RECEPTOR_INTERACTION 3.4767  0.0081       12        36
C2_immune WP_OVERVIEW_OF_PROINFLAMMATORY_AND_PROFIBROTIC_MEDIATORS 4.1720  0.1033        6        15
C2_immune         REACTOME_TNFS_BIND_THEIR_PHYSIOLOGICAL_RECEPTORS 4.3458  0.1592        5        12
        H                               HALLMARK_KRAS_SIGNALING_UP 2.9800  0.1872        6        21
        H               HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION 2.9204  0.1872        7        25

=== DOWN along δ̂ (top 100 genes) ===
  library                                                gene_set     ER  qvalue  overlap  set_size
       C7 GSE36476_CTRL_VS_TSST_ACT_40H_MEMORY_CD4_TCELL_YOUNG_UP 7.0556     0.0       23        34
       C7 GSE36476_CTRL_VS_TSST_ACT_16H_MEMORY_CD4_TCELL_YOUNG_UP 6.1353     0.0       20        34
C2_immune                        LEE_DIFFERENTIATING_T_LYMPHOCYTE 6.8196     0.0       17        26
       C7   GSE36476_CTRL_VS_TSST_ACT_40H_MEMORY_CD4_TCELL_OLD_UP 6.6752     0.0       16        25
       C7 GSE36476_CTRL_VS_TSST_ACT_72H_MEMORY_CD4_TCELL_YOUNG_UP 5.0660     0.0       17        35
       C7   GSE36476_CTRL_VS_TSST_ACT_16H_MEMORY_CD4_TCELL_OLD_UP 5.2150     0.0       15        30
```

### Part 1 readout

Read this against the two things that constrain how far the gene list can be pushed:

1. **The expression gate matters here.** On a subsampled dry run of `structural/all`, the unfiltered
   relative ranking came back as `CRYBA2, ZCCHC12, CNR1, GRP, PENK, TRH, NPY` up and
   `KRT84, KRTAP4-3, LCE2B, KRT28` down — neuroendocrine peptides and keratins, none of them CD4
   T-cell biology. They are genes at the extremes of the Geneformer embedding that this cohort barely
   transcribes; because `W` is a function of the embedding under `decoder_mode='structural'`, they
   still get full loading rows, and a relative change on a near-zero baseline ranks first. The
   expressed-gene panel and the abundance-weighted panel of Fig 2 are the interpretable ones, and the
   per-factor `support_ratio` above says how much of δ̂'s movement rides on that failure mode.
2. **Rank 10.** The ranking is a signed combination of ten factor programs, so a gene's position is
   determined by its loading profile across those ten — not by an independent per-gene fit.

Then compare against two external references:

* **The v1 run (`old/37_semantic_clonality_transfer` cell 30)** ran a plain Wilcoxon of predicted-clonal vs predicted-benign on
  the *dark* donors and got `CCR4, PHF19, COL6A3, LMNA, PLEC, RNF19A, VIM, ITM2A`. Overlap with the
  δ̂ signature is printed below — these are independent routes to the same axis (a naive DE on
  transferred calls vs the model's own decoder), so agreement is meaningful and disagreement is
  worth naming. It is a **v1-cohort** list, so it is a comparison, not an expectation.
* Known CTCL/MF malignant-CD4 markers.

```python  # cell 47
# the v1 run's naive Wilcoxon hits on the dark donors — a comparison, not an expectation
V1_WILCOXON = ["CCR4", "PHF19", "COL6A3", "LMNA", "PLEC", "RNF19A", "VIM", "ITM2A"]
CTCL_MARKERS = ["CCR4", "TOX", "GTSF1", "PLS3", "TWIST1", "KIR3DL2", "IL2RA", "ICOS", "TNFRSF8",
                "CD7", "CD26", "DPP4", "STAT5A", "MIR155HG", "TIGIT", "CTLA4"]
s = SCORE[HEAD].reindex(EXP_GENES)
ref = pd.DataFrame({"delta_score": s.round(3), "pctile_expressed": s.rank(pct=True).round(3),
                    "detect": DET.reindex(s.index).round(3),
                    "pseudobulk_logfc": PB["logfc"].reindex(s.index).round(3),
                    "q": PB["qvalue"].reindex(s.index), "t_consistency": CONS["t"].round(1)})
for label, genes in (("v1 Wilcoxon hits", V1_WILCOXON), ("known CTCL markers", CTCL_MARKERS)):
    have = [g for g in genes if g in ref.index]
    print(f"\n=== {label} ({len(have)}/{len(genes)} in the HVG set) ===")
    print(ref.loc[have].sort_values("delta_score", ascending=False).to_string())

sig = pd.DataFrame({f"score_{a}": SCORE[a] for a in SCORE})
for a in ABS:
    sig[f"abs_change_{a}"] = ABS[a]
sig["p_bar"] = pd.Series(P_BAR, index=SYMBOL.to_numpy())
sig["detect_frac"] = DET
sig["expressed"] = EXPRESSED
sig["consistency_t"] = CONS["t"]
sig["frac_same_sign"] = CONS["frac_same_sign"]
sig["counterfactual_lfc"] = CF[HEAD]
sig["pseudobulk_logfc"] = PB["logfc"]
sig["pseudobulk_q"] = PB["qvalue"]
sig["chrom_arm"] = ARM_OF.reindex(sig.index)
sig.sort_values(f"score_{HEAD}", ascending=False).to_csv(OUT / "delta_gene_signature.csv")
pd.concat({a: FC[a] for a in FC}, names=["arm_model"]).to_csv(OUT / "delta_factor_contributions.csv")
SUPP.join(FC[HEAD][["contribution", "abs_frac"]]).to_csv(OUT / "delta_factor_support.csv")
print(f"\nwrote delta_gene_signature.csv {sig.shape}, delta_factor_contributions.csv, "
      "delta_factor_support.csv")
```

```
=== v1 Wilcoxon hits (7/8 in the HVG set) ===
        delta_score  pctile_expressed  detect  pseudobulk_logfc   q  t_consistency
COL6A3        0.810             0.965   0.054             0.510 NaN           16.2
LMNA          0.475             0.908   0.594             0.562 NaN           12.3
PHF19         0.232             0.775   0.156             0.169 NaN            6.2
VIM           0.176             0.718   0.949             0.135 NaN            5.7
ITM2A        -0.095             0.399   0.507            -0.411 NaN           -2.4
RNF19A       -0.149             0.319   0.442            -0.083 NaN           -1.8
CCR4         -0.321             0.111   0.227            -0.600 NaN           -2.8

=== known CTCL markers (12/16 in the HVG set) ===
         delta_score  pctile_expressed  detect  pseudobulk_logfc   q  t_consistency
TWIST1         1.288             0.994   0.106             1.335 NaN           18.5
IL2RA          0.491             0.914   0.130            -0.983 NaN           10.1
GTSF1          0.407             0.893   0.207             1.332 NaN           13.2
TOX            0.238             0.781   0.346             0.992 NaN            4.4
PLS3           0.133             0.669   0.068             0.611 NaN            3.5
TIGIT         -0.017             0.498   0.259            -1.186 NaN            0.4
KIR3DL2       -0.029             0.477   0.089             0.991 NaN           -0.5
DPP4          -0.055             0.453   0.086            -0.885 NaN           -1.6
CTLA4         -0.057             0.451   0.250            -1.097 NaN           -0.3
CCR4          -0.321             0.111   0.227            -0.600 NaN           -2.8
ICOS          -0.373             0.072   0.204            -1.723 NaN           -5.0
CD7           -0.570             0.011   0.319            -3.943 NaN          -10.6

wrote delta_gene_signature.csv (2500, 13), delta_factor_contributions.csv, delta_factor_support.csv
```

## Part 2 — can `s = z·δ̂/‖δ̂‖` separate malignant from benign on a line?

Leakage-free from here on: donor *d* is scored on `z_structural_fold{k}.npy`, the latent of the model
trained **without** fold *k*'s donors, with δ̂ fitted on that fold's training donors only. The
`study`-batch arm is the only one that can be folded — under `batch=donor` a held-out donor's batch
category has no trained embedding.

The null is a *distribution* over `N_NULL_DRAWS` label permutations per fold, not a single draw:
one shuffled direction is shared by every held-out donor of its fold, and in 10 dimensions a random
direction has `cos` with δ̂ of order `1/√10`, which is plenty to look like signal. The v1 run hit
exactly this and read a 0.385 null as a bug.

**The null is the reference, not a v1 number.** The v1 cohort's headline (per-donor AUROC 0.94,
null 0.51) is printed alongside for orientation, but it is a *different cohort* — v2 adds seven
studies with no repertoire at all and roughly quadruples the dark set, so the two are not
comparable and nothing asserts they match. What the numbers below are read against is the
shuffled-δ̂ null, plus the decomposition around it: the 10-d ceiling, the pooled-vs-centred gap,
direction stability, and the histograms.

```python  # cell 49
AXIS_ARM = "structural"
PC, FOLD_DELTAS, NULL_DELTAS = DA.axis_scores(AXIS_ARM, FOLDS, adata.obs, LABELLED,
                                              seed=SEED, n_null=N_NULL_DRAWS)
MET, NULLS = DA.axis_metrics(PC, adata.obs, AXIS_ARM, FOLDS, LABELLED,
                             null_deltas=NULL_DELTAS, seed=SEED)
for k, d in FOLD_DELTAS.items():
    np.save(OUT / f"delta_{AXIS_ARM}_{k}.npy", d)

print(f"{len(MET)} held-out donors, {int(PC['labelled'].sum()):,} labelled cells scored")
print(f"1-D projection : per-donor AUROC {MET['auc_1d'].mean():.3f} +- {MET['auc_1d'].std():.3f}")
print(f"10-d logistic  : per-donor AUROC {MET['auc_10d'].mean():.3f} +- {MET['auc_10d'].std():.3f}")
print(f"shuffled-δ̂ null: {NULLS['auc'].mean():.3f} +- {NULLS['auc'].std():.3f} "
      f"({NULLS['draw'].nunique()} draws x {NULLS['donor'].nunique()} donors)")
print(f"pooled AUROC   : raw {DA.pooled_auc(PC)[0]:.3f} | "
      f"donor-median-centred {DA.pooled_auc(PC, center='median')[0]:.3f}")

# The v1 cohort for orientation only — a DIFFERENT cohort (70 donors / 31 labelled / 34 dark,
# every study with V(D)J), so this is not a regression anchor and nothing asserts a match.
V1_REFERENCE = {"1-D projection": 0.940, "shuffled-δ̂ null": 0.510}
for label, mine in (("1-D projection", MET["auc_1d"].mean()),
                    ("shuffled-δ̂ null", NULLS["auc"].mean())):
    print(f"  v1 cohort {label:<16} here {mine:.3f} vs v1 {V1_REFERENCE[label]:.3f}")
# The null is the gate: a 10-d random direction must not discriminate.
assert abs(NULLS["auc"].mean() - 0.5) < 0.05, (
    f"shuffled-δ̂ null at {NULLS['auc'].mean():.3f}, not ~0.5 over {NULLS['draw'].nunique()} draws "
    "— the permutation is not destroying the label")
```

```
37 held-out donors, 113,236 labelled cells scored
1-D projection : per-donor AUROC 0.916 +- 0.084
10-d logistic  : per-donor AUROC 0.938 +- 0.058
shuffled-δ̂ null: 0.472 +- 0.303 (20 draws x 37 donors)
pooled AUROC   : raw 0.917 | donor-median-centred 0.824
  v1 cohort 1-D projection   here 0.916 vs v1 0.940
  v1 cohort shuffled-δ̂ null here 0.472 vs v1 0.510
```

**Fig 9 — the line itself.** Where the labelled cells of a held-out donor land on `s`, clonal vs
anchor, for the best / median / worst donors by within-donor AUROC. The dashed line is that donor's
median `s` — a label-free reference, not a fitted threshold.

```python  # cell 51
S = PC[PC["labelled"].to_numpy(dtype=bool)]
o = MET.sort_values("auc_1d", ascending=False)
picks = list(o.head(2).donor) + list(o.iloc[[len(o) // 2, len(o) // 2 + 1]].donor) + list(o.tail(2).donor)

fig, axes = plt.subplots(2, 3, figsize=(8, 4))
for ax, d in zip(axes.ravel(), picks):
    sub = S[S["donor"] == d]
    y = sub["y"].to_numpy(dtype=bool)
    bins = np.linspace(sub["s"].min(), sub["s"].max(), 45)
    ax.hist(sub.loc[~y, "s"], bins=bins, color="#2c7fb8", alpha=.65, label="anchor (benign)")
    ax.hist(sub.loc[y, "s"], bins=bins, color="#c0392b", alpha=.65, label="ALICE-clonal")
    ax.axvline(sub["s"].median(), ls="--", lw=.7, c="k")
    ax.set_title(f"{d}  AUROC {float(o.set_index('donor').loc[d, 'auc_1d']):.3f}", fontsize=8)
    ax.set_xlabel("s = z·δ̂/‖δ̂‖"); ax.set_yticks([])
axes[0, 0].legend(loc="upper left", frameon=False)
fig.suptitle("the malignancy axis separates within donor (best / median / worst held-out donors)",
             fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "nb38_axis_histograms.png"); plt.show()
```

```
<Figure size 880x440 with 6 Axes>
```

**Fig 10 — what does collapsing 10-d → 1-D cost?** Within-donor AUROC for the single δ̂ projection,
the 10-d logistic discriminant (the ceiling any one direction competes against), and the shuffled-δ̂
null. Within a donor the logistic probability is monotone in `w·z`, so `auc_10d` needs no benign
centroid and is a fair comparison.

```python  # cell 53
groups = [("1-D  δ̂ projection", MET["auc_1d"].to_numpy()),
          ("10-d logistic", MET["auc_10d"].to_numpy()),
          ("shuffled-δ̂ null", NULLS.groupby("donor")["auc"].mean().to_numpy())]
fig, ax = plt.subplots(figsize=(4.4, 3.2))
rng = np.random.default_rng(SEED)
for i, (lab, v) in enumerate(groups):
    ax.boxplot(v, positions=[i], widths=.5, showfliers=False,
               medianprops=dict(color="k"), boxprops=dict(color="grey"))
    ax.scatter(np.full(len(v), i) + rng.normal(0, .06, len(v)), v, s=10, alpha=.6,
               c="#c0392b" if i == 0 else ("#34495e" if i == 1 else "#95a5a6"), lw=0)
ax.axhline(.5, ls=":", lw=.7, c="k")
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([f"{l}\n{v.mean():.3f}" for l, v in groups])
ax.set_ylabel("within-donor AUROC"); ax.set_ylim(.3, 1.02)
gap = MET["auc_10d"].mean() - MET["auc_1d"].mean()
ax.set_title(f"one direction costs {gap:+.3f} AUROC vs the full 10-d discriminant\n"
             f"({len(MET)} held-out donors, leakage-free)")
fig.tight_layout(); fig.savefig(FIG / "nb38_auroc_1d_vs_10d.png"); plt.show()

print(MET.round(3).to_string(index=False))
```

```
<Figure size 484x352 with 1 Axes>
 fold               donor  n_labelled  true_frac  auc_1d  auc_10d
   10           D15__MF34         314      0.248   0.999    0.997
    9 Li2024_atlas__CTCL2        2593      0.446   0.998    0.994
    4           D15__MF36        1052      0.358   0.995    0.999
   11 Li2024_atlas__CTCL6        7368      0.421   0.994    0.996
    6           D15__MF35        1888      0.862   0.987    0.994
    3            D1__P204        3806      0.523   0.983    0.979
    7            B5__PT23        3723      0.950   0.982    0.977
    1 Li2024_atlas__CTCL4        4420      0.167   0.978    0.968
    2            D1__P312         947      0.603   0.974    0.978
    0 Li2024_atlas__CTCL7        9904      0.311   0.969    0.970
   11           D15__MF48        1051      0.876   0.969    0.964
    2 Li2024_atlas__CTCL5        8566      0.741   0.955    0.981
    0 Li2024_atlas__CTCL3        8443      0.883   0.955    0.974
    8  Li2024_atlas__PT53        1796      0.688   0.954    0.921
    5            D1__P171        4944      0.962   0.953    0.969
    3 Li2024_atlas__CTCL8        6376      0.815   0.951    0.923
   10            D1__P303         998      0.954   0.949    0.914
    0           D15__MF28         381      0.079   0.948    0.989
    5            D3__P196         446      0.751   0.946    0.934
    1 Li2024_atlas__CTCL1        6887      0.977   0.939    0.981
    4            B5__PT09        6177      0.804   0.937    0.937
    0  Li2024_atlas__PT50        1186      0.695   0.937    0.954
    6  Li2024_atlas__PT47         350      0.820   0.934    0.930
    7           D15__MF30         422      0.576   0.928    0.883
    5  Li2024_atlas__PT11        1933      0.948   0.924    0.963
    9           D15__MF24         622      0.207   0.922    0.993
    7  Li2024_atlas__PT56        2318      0.537   0.921    0.926
    8           D13__MF19         302      0.609   0.913    0.915
    4             D1__PGS        2136      0.921   0.886    0.832
    8            D1__P311        3635      0.951   0.879    0.913
    1            D3__P192        2794      0.927   0.823    0.880
   10           D13__MF17        3254      0.991   0.813    0.952
    2           D15__MF26        2501      0.952   0.805    0.955
    6             D1__P76        5201      0.845   0.793    0.808
    9  Li2024_atlas__PT35        3393      0.985   0.751    0.788
   11            D3__P105         265      0.211   0.692    0.804
    3           D13__MF18         844      0.556   0.669    0.860
```

**Fig 11 — what donor offsets cost.** Pooling every held-out cell onto one global axis vs first
subtracting each donor's own median `s`. The centring uses no labels, so the gap is purely the donor
offset in `z` — the reason δ̂ is fitted per-donor-differenced and the reason a single global threshold
is the wrong rule.

```python  # cell 55
from sklearn.metrics import roc_curve

fig, ax = plt.subplots(figsize=(3.8, 3.4))
for center, lab, c in (("none", "raw s (one global axis)", "#95a5a6"),
                       ("median", "s − donor median (label-free)", "#c0392b")):
    auc, y, s = DA.pooled_auc(PC, center=center)
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, lw=1.4, c=c, label=f"{lab}  AUROC {auc:.3f}")
ax.plot([0, 1], [0, 1], ls=":", lw=.7, c="k")
ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
ax.legend(loc="lower right", frameon=False)
ax.set_title("donor offset, not weak separation, is what breaks a global threshold")
fig.tight_layout(); fig.savefig(FIG / "nb38_pooled_roc.png"); plt.show()
```

```
<Figure size 418x374 with 1 Axes>
```

**Fig 12 — is δ̂ a stable direction, and does it point where the discriminant does?** Eight
overlapping donor subsets should give near-parallel δ̂s; cosine far from 1 would mean δ̂ is a property
of which donors landed in the fit rather than of malignancy. The `logit` row asks whether the
mean-difference direction is close to the optimal separating direction.

```python  # cell 57
z_all = DA.load_latent(AXIS_ARM, "all", n_obs=adata.n_obs)
_, d_all = DA.per_donor_deltas(z_all, adata.obs, LABELLED)
w_all = DA.logistic_direction(z_all, adata.obs, LABELLED, seed=SEED)
COS = DA.direction_stability(FOLD_DELTAS, delta_all=d_all, extra={"logit(all)": w_all})

fig, ax = plt.subplots(figsize=(4.4, 3.8))
im = ax.imshow(COS.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(COS))); ax.set_xticklabels(COS.index, rotation=90)
ax.set_yticks(range(len(COS))); ax.set_yticklabels(COS.index)
for i in range(len(COS)):
    for j in range(len(COS)):
        ax.text(j, i, f"{COS.iloc[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
folds_only = [c for c in COS.index if c.startswith("fold")]
off = COS.loc[folds_only, folds_only].to_numpy()[~np.eye(len(folds_only), dtype=bool)]
ax.set_title(f"δ̂ is the same direction in every fold (median cos = {np.median(off):.3f})\n"
             f"cos(δ̂_all, logistic) = {COS.loc['all', 'logit(all)']:.3f}")
fig.colorbar(im, ax=ax, shrink=.75, label="cosine")
fig.tight_layout(); fig.savefig(FIG / "nb38_direction_stability.png"); plt.show()
```

```
<Figure size 484x418 with 2 Axes>
```

**Fig 13 — face validity on the TCR-less donors.** The same axis applied to the dark donors and
the healthy controls, each donor's `s` centred on its own median. On v2 the dark set is the point:
seven of the fourteen cohorts carry no V(D)J at all, so this is where the axis does work that no
repertoire-based call can. Nothing here is validated against external truth — a bimodal MF donor and
a unimodal healthy control is the expected shape, and a healthy control that looks bimodal is the
failure mode the v1 run flagged for `D1__P112`.

```python  # cell 59
s_cell_all = pd.Series(SC.project(z_all, d_all), index=adata.obs.index)
obs = adata.obs
grp = pd.Series("other", index=obs.index)
grp[obs[DA.DONOR_KEY].astype(str).isin(DARK).to_numpy()] = "dark (no TCR)"
grp[obs["disease"].astype(str).eq("HC").to_numpy()] = "healthy control"
grp[(ic | ia)] = "labelled (TCR)"
sc_ctr = s_cell_all - s_cell_all.groupby(
    obs[DA.DONOR_KEY].astype(str).to_numpy()).transform("median")

fig, ax = plt.subplots(figsize=(4.6, 3.2))
for lab, c in (("labelled (TCR)", "#34495e"), ("dark (no TCR)", "#c0392b"),
               ("healthy control", "#2c7fb8")):
    v = sc_ctr[grp == lab]
    if len(v) > 50:
        ax.hist(v, bins=80, histtype="step", density=True, lw=1.3, color=c,
                label=f"{lab}  n={len(v):,}")
ax.set_xlabel("s − donor median"); ax.set_yticks([])
ax.legend(frameon=False)
ax.set_title("dark donors spread along the axis like the labelled cohort;\nhealthy controls stay narrow")
fig.tight_layout(); fig.savefig(FIG / "nb38_dark_donor_axis.png"); plt.show()

spread = sc_ctr.groupby(obs[DA.DONOR_KEY].astype(str).to_numpy()).agg(
    iqr=lambda v: float(np.subtract(*np.percentile(v, [75, 25]))), n="size")
spread["group"] = donor_tbl["group"].reindex(spread.index)
spread["disease"] = donor_tbl["disease"].reindex(spread.index)
print(spread.query("n >= 200").groupby(["group", "disease"])["iqr"].describe()[["count", "mean", "50%"]].round(3).to_string())
```

```python  # cell 60
MET.to_csv(OUT / "delta_axis_1d_per_donor.csv", index=False)
NULLS.to_csv(OUT / "delta_axis_1d_null_draws.csv", index=False)
PC.rename_axis("obs_name").to_parquet(OUT / "delta_axis_percell.parquet")
print("wrote:", ", ".join(f.name for f in (
    OUT / "delta_axis_1d_per_donor.csv", OUT / "delta_axis_1d_null_draws.csv",
    OUT / "delta_axis_percell.parquet", OUT / "delta_gene_signature.csv",
    OUT / "delta_factor_contributions.csv")))
print("\nheadline numbers")
print(f"  gene signature  : {HEAD}, rho(model, paired pseudobulk) = "
      f"{spearmanr(SCORE[HEAD].reindex(common), PB['logfc'].reindex(common)).statistic:.3f}")
print(f"  1-D axis        : within-donor AUROC {MET['auc_1d'].mean():.3f} "
      f"vs 10-d {MET['auc_10d'].mean():.3f} vs null {NULLS['auc'].mean():.3f}")
print(f"  pooled          : raw {DA.pooled_auc(PC)[0]:.3f} -> "
      f"{DA.pooled_auc(PC, center='median')[0]:.3f} after label-free donor centring")
```

### Part 2 readout

- Can a single line separate malignant from benign? Read Fig 9 + Fig 10: the within-donor number is
  the honest one, and the gap to the 10-d discriminant is what the collapse to one dimension costs.
- A global threshold on `s` is not the same claim. Fig 11 separates "the axis is weak" from "donors
  sit at different offsets"; only the second is fixable without labels.
- δ̂ is only interpretable as *one* direction if it is stable — Fig 12.
