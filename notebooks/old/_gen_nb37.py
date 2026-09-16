"""Generate notebooks/MF/37_semantic_clonality_transfer.ipynb."""
import json
from pathlib import Path

C = []


def md(s):
    C.append({"cell_type": "markdown", "id": f"md{len(C):02d}", "metadata": {},
              "source": s.strip("\n").splitlines(keepends=True)})


def code(s):
    C.append({"cell_type": "code", "id": f"cd{len(C):02d}", "execution_count": None,
              "metadata": {}, "outputs": [],
              "source": s.strip("\n").splitlines(keepends=True)})


md(r"""
# 37 — Transferring TCR clonality to TCR-less skin samples: semantic vs LDVAE vs scVI

TCR exists for **36 of the 82 skin donors** (nb30). The rest — 21 li2024 + 8 gaydosik2019 donors —
have no clonality data *and* no published malignant label, so they are dark. This notebook asks
whether the malignant/benign split can be **transferred** into them through the latent, and which
latent does it best.

The mechanism is the δ-arithmetic from `notebooks/perturbation_trial.ipynb`, where a held-out
unit's latent state is recovered as `z_ctrl + δ` with `δ = mean(z|stim) − mean(z|ctrl)` fitted on
training units. Here:

$$\delta_d = \mathrm{mean}(z \mid \text{ALICE-clonal cells of } d) - \mathrm{mean}(z \mid \text{benign anchors of } d),
\qquad \hat\delta = \mathrm{mean}_d\, \delta_d$$

Per-donor differencing cancels donor offsets. For a donor with no TCR we do not know its benign
centroid, so it is **estimated** from that donor's own cells (2-component GMM on the $\hat\delta$
projection — the per-donor trick nb30 uses on `cnv_score`), and the clonal centroid is
**reconstructed** as $\hat\mu_{mal} = \hat\mu_{ben} + \hat\delta$. Each cell goes to the nearer
centroid.

**Validation is the point of this notebook.** Repeated **leave-3-samples-out** over the labelled
donors: each fold retrains the encoder with the 3 held-out donors' cells fully removed
(`perturbation_trial`'s protocol), encodes everything with that fold's model, and scores the
transfer on the 3 unseen donors. The applied call on the dark donors comes last and is explicitly
**not** validated against external truth — they have none.

**Three arms, one protocol.** Every arm sees the same folds, cells, HVGs, epochs, KL schedule,
batch/labels keys and seed — the only differences are the ones named:

| arm | model | difference |
|---|---|---|
| `semantic` | `SemanticSCVI` | geometric Geneformer prior, `coherence_weight=1000` |
| `ldvae` | `SemanticSCVI` | **`coherence_weight=0`** — architecture-identical no-prior ablation |
| `scvi` | `SCVI` | nonlinear decoder, no positivity constraint |

`ldvae` isolates the semantic prior with one knob: the ELBO, the positive-weight decoder, the
decorrelation term and the KL warm-up are byte-identical to `semantic`. `scvi` cannot be matched
that tightly — it has no `weights_positive`, hence no decorrelation penalty, and its decoder is
nonlinear by construction. Its batch norm is set to `"encoder"`, the faithful translation of the
LDVAE family's `use_batch_norm=False` (which disables *decoder* BN only).

> **HEAVY** cells are marked. The object is ~20 GB and the fold training is 27 GPU runs (3 arms ×
> 8 folds + 3 all-donor references) — run those on the GPU/compute kernel (`neural_nmf_env`) or via
> `jobs/run_clonality_folds.sh`, never on the login node.
""")

md("## Part 0 — parameters")

code(r'''
# ============================================================
# Parameters — cohort filters, the three arms' knobs (nb18 recipe), fold layout.
# ============================================================
import hashlib
import json
from pathlib import Path


def _resolve_nb_dir() -> Path:
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("notebooks/MF"), Path("scvi-tools-neural-nmf/notebooks/MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data from {start}")


NB_DIR = _resolve_nb_dir()
print("NB_DIR =", NB_DIR)

# ---- inputs ----
OBJ            = NB_DIR / "data" / "atlas_joint" / "skin_T_tcr_annotated_v3.h5ad"   # 82 donors, nb30
MALIG          = NB_DIR / "data" / "atlas_joint" / "skin_T_malignancy_v3.parquet"   # nb30 per-cell calls
GENE_ID_SOURCE = NB_DIR / "data" / "cache" / "cnmf_malignant_counts.h5ad"           # gene_name -> Ensembl
SEMANTIC_CACHE = NB_DIR / "data" / "mf_clonality_geneformer.pt"                     # full-gene Geneformer map

# ---- outputs ----
OUT_DIR   = NB_DIR / "benchmark_results" / "clonality_transfer"
JOB_INPUT = OUT_DIR / "_job_input"
INPUT_H5AD = JOB_INPUT / "clonality_input.h5ad"   # written by Part 2; read back by Part 3 onward
MAP_PT     = JOB_INPUT / "clonality_semantic_map.pt"
FIG_DIR   = NB_DIR / "figures"
CONFIG    = NB_DIR / "jobs" / "clonality_folds_config.json"
MODEL_CACHE = NB_DIR / "models" / ".model_cache_semantic_clonality"
FOLD_PARQUET  = NB_DIR / "data" / "atlas_joint" / "semantic_clonality_folds_v1.parquet"
TRANS_PARQUET = NB_DIR / "data" / "atlas_joint" / "semantic_clonality_transfer_v1.parquet"
for d in (OUT_DIR, JOB_INPUT, FIG_DIR, MODEL_CACHE):
    d.mkdir(parents=True, exist_ok=True)

# ---- cohort (nb18's rules) ----
CD4_TYPE      = "CD4"                                    # cell_type_T; excludes CD4_Treg / CD8 / NK
DROP_ENTITIES = {"MF_gamma_delta", "CD8_aggressive_epidermotropic_CTCL"}  # TCRb caller is blind to these
DROP_DONORS   = {"D1__P303"}                             # duplicate of D5__MFIVB
MIN_TRB_FRAC  = 0.0                                      # no extra recovery gate; nb30 already filtered

# ---- preprocessing / model (nb18 recipe; epochs halved, one run per fold + reference) ----
HVG_TOP_N, HVG_FLAVOR = 2500, "seurat_v3"
N_LATENT   = 10
BATCH_KEY  = "study"        # NOT sample_id: a held-out donor's sample_id would be an unseen batch
LABELS_KEY = "cell_type_T"
MAX_EPOCHS, WARMUP_EPOCHS, KL_WARMUP = 100, 20, 100
SEMANTIC_KWARGS = dict(
    loss_mode="geometric",
    coherence_weight=1000.0,
    n_gene_sample=1024,
    n_latent=N_LATENT,
    n_layers=1,
    n_hidden=128,
    dropout_rate=0.1,
    gene_likelihood="nb",
    weights_positive=True,
    use_batch_norm=False,
)

# The three arms. Everything outside `kwargs` (epochs, KL warm-up, batch/labels keys, folds, HVGs,
# train pool, seed) is shared, so each arm differs only by what is written here.
#   ldvae — SemanticSCVI with the semantic loss switched off. `SemanticLDVAE.loss` still computes
#           the coherence term but adds `scale * 0 * raw_loss`, so the gradient contribution is
#           exactly zero while the ELBO / positive decoder / decorrelation / KL schedule stay
#           identical to `semantic`. (`train_or_load_nonneg_ldvae` would NOT match: it trains bare,
#           leaving n_epochs_kl_warmup at scvi's default 400.)
#   scvi  — use_batch_norm="encoder" is the faithful translation of the LDVAE family's
#           use_batch_norm=False, which only disables the decoder BN (encoder BN is hard-coded on).
#           gene_likelihood must be given explicitly: SCVI defaults to zinb, not nb.
MODEL_NAMES = ["semantic", "ldvae", "scvi"]
MODELS = {
    "semantic": {"kwargs": SEMANTIC_KWARGS, "warmup_epochs": WARMUP_EPOCHS},
    "ldvae": {"kwargs": {**SEMANTIC_KWARGS, "coherence_weight": 0.0},
              "warmup_epochs": WARMUP_EPOCHS},
    "scvi": {"kwargs": dict(n_latent=N_LATENT, n_layers=1, n_hidden=128, dropout_rate=0.1,
                            gene_likelihood="nb", use_batch_norm="encoder")},
}

# ---- folds ----
SEED               = 0
FOLD_SIZE          = 3       # leave-3-samples-out
TRAIN_FRAC         = 1 / 3   # per-donor training subsample (nb19 sweep used the same)
TRAIN_MIN_CELLS    = 500     # never subsample a donor below this
EVAL_MIN_CELLS     = 200     # holdout eligibility: per-donor metrics need enough of both classes
EVAL_MIN_CLONAL    = 25
EVAL_MIN_ANCHOR    = 25


def _cache_slug(n=10):
    """Stable hash of every param that affects a trained model (shared by all folds)."""
    blob = json.dumps({"kwargs": dict(sorted(SEMANTIC_KWARGS.items())),
                       "max_epochs": MAX_EPOCHS, "warmup_epochs": WARMUP_EPOCHS,
                       "n_epochs_kl_warmup": KL_WARMUP, "hvg": HVG_TOP_N,
                       "batch_key": BATCH_KEY, "labels_key": LABELS_KEY},
                      default=str, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:n]


PARAM_SLUG = _cache_slug()
print("param slug:", PARAM_SLUG, "| model cache:", MODEL_CACHE / PARAM_SLUG)
''')

code(r'''
import gc
import importlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

for _p in (str(NB_DIR), str(NB_DIR.parent)):     # MF/ helpers, then notebooks/ helpers
    if _p not in sys.path:
        sys.path.insert(0, _p)

import semantic_clonality_helpers as SC
import semantic_malig_helpers as M

importlib.reload(M)
importlib.reload(SC)

np.random.seed(SEED)
sc.settings.verbosity = 1
plt.rcParams["figure.dpi"] = 120
''')

md(r"""
## Part 1 — cohort + labels · HEAVY (compute kernel)

Read the nb30 TCR object **backed**, keep only the CD4 non-Treg cells of the included donors, then
pull that subset into memory (the full file is ~20 GB). Join nb30's per-cell calls and build the two
label columns:

- `is_clonal` — ALICE malignant (`tcr_malignant_alice`), the positives.
- `is_anchor` — `has_tcr & ~ALICE & ~tcr_is_expanded`, the benign negatives (nb16/17's anchor rule).

Cells that are neither (TCR-negative cells of TCR-covered donors, and every cell of a dark donor)
have unknown status and are **excluded from scoring** — they still train the encoder.
""")

code(r'''
# Backed read -> mask -> to_memory: never materialise the whole 20 GB object.  HEAVY
ad = sc.read_h5ad(OBJ, backed="r")
obs_all = ad.obs

keep = (obs_all["cell_type_T"].astype(str) == CD4_TYPE).to_numpy()
keep &= ~obs_all["entity"].astype(str).isin(DROP_ENTITIES).to_numpy()
keep &= ~obs_all["donor"].astype(str).isin(DROP_DONORS).to_numpy()
print(f"CD4 non-Treg of included donors: {int(keep.sum())}/{ad.n_obs} cells, "
      f"{obs_all.loc[keep, 'donor'].nunique()} donors")

adata = ad[keep].to_memory()
del ad
gc.collect()
adata.X = adata.layers["raw_counts"].copy()          # NB likelihood wants raw counts
for lay in list(adata.layers):
    del adata.layers[lay]
gc.collect()
print(adata)
''')

code(r'''
# nb30 per-cell TCR/CNV calls -> the two label columns.
mal = pd.read_parquet(MALIG)
for c in ["tcr_malignant_alice", "has_tcr", "tcr_is_expanded", "cnv_malig_cluster"]:
    v = mal[c].reindex(adata.obs_names)
    adata.obs[c] = v.astype("boolean").fillna(False).to_numpy(dtype=bool)
adata.obs["cnv_cell_score"] = mal["cnv_cell_score"].reindex(adata.obs_names).to_numpy(float)

adata.obs["is_clonal"] = adata.obs["tcr_malignant_alice"].to_numpy()
adata.obs["is_anchor"] = (adata.obs["has_tcr"].to_numpy()
                          & ~adata.obs["is_clonal"].to_numpy()
                          & ~adata.obs["tcr_is_expanded"].to_numpy())
# HC donors are benign by definition (nb18 does the same)
hc = adata.obs["disease"].astype(str).eq("HC").to_numpy()
n_flip = int(adata.obs["is_clonal"].to_numpy()[hc].sum())
adata.obs.loc[hc, "is_clonal"] = False
adata.obs.loc[hc & adata.obs["has_tcr"].to_numpy(), "is_anchor"] = True

print(f"clonal {int(adata.obs.is_clonal.sum()):,} | anchor {int(adata.obs.is_anchor.sum()):,} | "
      f"unlabelled {int((~adata.obs.is_clonal & ~adata.obs.is_anchor).sum()):,} "
      f"(HC cells forced benign: {n_flip})")
''')

code(r'''
# Donor roles: labelled (delta + eval) / no_clone (training + negative control) / dark (transfer target).
donor_tbl = SC.donor_label_table(adata.obs, eval_min_cells=EVAL_MIN_CELLS,
                                 eval_min_clonal=EVAL_MIN_CLONAL, eval_min_anchor=EVAL_MIN_ANCHOR)
print(donor_tbl.groupby("group", observed=True)
      .agg(donors=("n_cells", "size"), cells=("n_cells", "sum"),
           eligible=("eligible", "sum")).to_string())
with pd.option_context("display.width", 200, "display.max_rows", 100):
    print(donor_tbl.to_string())
donor_tbl.to_csv(OUT_DIR / "donor_table.csv")
''')

md(r"""
## Part 2 — Geneformer map, HVG, training pool, slim job input · HEAVY

nb18's preprocessing exactly: map symbols → Ensembl, build the **full-gene** Geneformer map (cached),
drop out-of-vocab genes *before* HVG so HVG ranks variance only among semantically-grounded genes,
then HVG-subset adata and the map together.

`train_pool` marks the per-donor training subsample (≈1/3, floor `TRAIN_MIN_CELLS`). Held-out
donors are removed per fold *on top of* this pool, and **encoding always uses every cell** — the
subsample only shrinks what the encoder is fit on.
""")

code(r'''
# gene_name -> Ensembl (Geneformer's vocabulary is keyed by Ensembl id).
src = sc.read_h5ad(GENE_ID_SOURCE, backed="r")
sym2ens = dict(zip(src.var["gene_name"].astype(str), src.var["gene_id"].astype(str)))
del src
gc.collect()

ribo = adata.var_names.str.upper().str.startswith(("RPS", "RPL"))
print(f"dropping {int(ribo.sum())} ribosomal protein genes")
adata = adata[:, ~ribo].copy()
adata.var["gene_id"] = [sym2ens.get(s, s) for s in adata.var_names.astype(str)]
adata.var["feature_name"] = adata.var_names.astype(str)
print(f"gene_id mapped to Ensembl: {int(sum(g.startswith('ENSG') for g in adata.var['gene_id']))}"
      f"/{adata.n_vars}")
''')

code(r'''
import torch

from benchmark_helpers import get_or_build_geneformer_map

# Geneformer's vocabulary is 20,275 Ensembl tokens, so on this FULL-gene panel (~40.7k, most of it
# antisense/lncRNA/novel loci) coverage is capped at vocab/n_vars ~ 0.50 — the builder's default
# min_coverage=0.5 can never pass and is the wrong guard here. Disable it and assert instead on the
# quantity that actually detects a key mismatch: how much of the vocabulary we hit. A wrong id/symbol
# column collapses that to a few hundred genes; a healthy run recovers most of the 20,275.
# symbol_key gives the Ensembl lookup a real symbol to fall back on rather than reusing the id.
GF_MIN_IN_VOCAB = 15_000
GF_KW = dict(var_id_key="gene_id", symbol_key="feature_name", min_coverage=0.0)

semantic_map = get_or_build_geneformer_map(adata, SEMANTIC_CACHE, **GF_KW)
if semantic_map.shape[0] != adata.n_vars:            # stale-cache guard (nb18)
    print(f"map rows {semantic_map.shape[0]} != n_vars {adata.n_vars} — rebuilding")
    SEMANTIC_CACHE.unlink()
    semantic_map = get_or_build_geneformer_map(adata, SEMANTIC_CACHE, **GF_KW)

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
print("after HVG:", adata.shape, "| semantic_map", tuple(semantic_map.shape))
assert semantic_map.shape[0] == adata.n_vars
''')

code(r'''
# Per-donor training subsample (encoding always uses all cells; this only shrinks the fit set).
rng = np.random.default_rng(SEED)
pool = np.zeros(adata.n_obs, dtype=bool)
donors = adata.obs["donor"].astype(str).to_numpy()
for d in np.unique(donors):
    idx = np.where(donors == d)[0]
    n = int(min(len(idx), max(TRAIN_MIN_CELLS, round(TRAIN_FRAC * len(idx)))))
    pool[rng.choice(idx, n, replace=False)] = True
adata.obs["train_pool"] = pool
print(f"train_pool {int(pool.sum()):,}/{adata.n_obs:,} cells "
      f"({pool.mean():.1%}) across {len(np.unique(donors))} donors")
''')

code(r'''
# Write the slim job input. Keeps X_mrvi_u / X_scVI so the baselines and the CPU dry run are free.
slim = sc.AnnData(
    X=adata.X.copy(),
    obs=adata.obs[["donor", "sample_id", "study", "disease", "entity", "cell_type_T",
                   "has_tcr", "tcr_is_expanded", "tcr_malignant_alice",
                   "cnv_malig_cluster", "cnv_cell_score",
                   "is_clonal", "is_anchor", "train_pool"]].copy(),
    var=adata.var[["gene_id", "feature_name", "highly_variable"]].copy(),
)
for k in ("X_mrvi_u", "X_scVI"):
    if k in adata.obsm:
        slim.obsm[k] = np.asarray(adata.obsm[k], dtype=np.float32)
for c in ("study", "cell_type_T"):
    slim.obs[c] = slim.obs[c].astype(str).astype("category")

slim.write_h5ad(INPUT_H5AD)
torch.save(semantic_map, MAP_PT)
print("wrote", INPUT_H5AD, slim.shape, "| obsm:", list(slim.obsm))
print("wrote", MAP_PT, tuple(semantic_map.shape))
''')

md(r"""
## Part 3 — folds + job config

`SC.make_triples` deals the eligible labelled donors largest-first into the currently smallest fold,
under the hard constraint that **no fold may hold out every donor of a `study`** — a held-out donor
whose batch category vanished from training has no trained batch embedding and could not be encoded.
`SC.check_batch_coverage` re-asserts this on the actual cells, and the job asserts it again before
training.

`runs` = one entry per (arm, fold) plus one `all` per arm (no holdout) — the leakage reference and
the model used for the applied call.

The first cell re-reads the slim input if the kernel restarted, so the config can be regenerated
without re-running the ~20 GB Part 1–2.
""")

code(r'''
# Reload guard: Part 3 onward needs only the slim input, so a fresh kernel can start here.  HEAVY-ish
if "slim" not in dir():
    slim = sc.read_h5ad(INPUT_H5AD)
    donor_tbl = pd.read_csv(OUT_DIR / "donor_table.csv", index_col=0)
    print("reloaded slim input:", slim.shape)

folds = SC.make_triples(donor_tbl, size=FOLD_SIZE, seed=SEED, batch_key=BATCH_KEY)
SC.check_batch_coverage(slim.obs, folds, batch_key=BATCH_KEY)
print(f"{len(folds)} folds of {sorted({len(f) for f in folds})} | "
      f"{len({d for f in folds for d in f})} donors tested once")
print(f"submit with:  cd {NB_DIR / 'jobs'} && ARRAY=1 ./run_clonality_folds.sh")
for k, f in enumerate(folds):
    print(f"  fold{k}: " + ", ".join(f"{d} ({donor_tbl.loc[d, 'n_clonal']}c/"
                                     f"{donor_tbl.loc[d, 'n_anchor']}a)" for d in f))

# δ̂ is fitted on every clone-bearing donor (fit_delta skips any that lack one of the classes);
# `eligible` — a strictly smaller set — governs who may be *held out*.
LABELLED = list(donor_tbl.index[donor_tbl["group"].isin(["labelled", "labelled_small"])])
DARK = list(donor_tbl.index[donor_tbl["group"] == "dark"])
NO_CLONE = list(donor_tbl.index[donor_tbl["group"] == "no_clone"])
print(f"\nδ fitted on {len(LABELLED)} clone-bearing donors "
      f"({int(donor_tbl['eligible'].sum())} of them holdout-eligible) | "
      f"no_clone {len(NO_CLONE)} | dark {len(DARK)}")
''')

code(r'''
# Latent-free fold gate — costs nothing and catches the bugs worth catching before 18 GPU jobs.
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
      f"both classes present in every held-out donor")
''')

code(r'''
cfg = {
    "input_h5ad": str(INPUT_H5AD),
    "semantic_map": str(MAP_PT),
    "model_cache_dir": str(MODEL_CACHE / PARAM_SLUG),
    "out_dir": str(OUT_DIR),
    "donor_key": "donor",
    "batch_key": BATCH_KEY,
    "labels_key": LABELS_KEY,
    "train_pool_key": "train_pool",
    "max_epochs": MAX_EPOCHS,
    "warmup_epochs": WARMUP_EPOCHS,
    "n_epochs_kl_warmup": KL_WARMUP,
    "semantic_kwargs": SEMANTIC_KWARGS,          # kept for reference; the job reads "models"
    "models": MODELS,
    # one run per (arm, fold) plus one all-donor reference per arm. `cache_key` is the fold, so the
    # cache path is <slug>/<arm>/<fold> and `--folds 3` still selects fold 3 of every arm.
    "runs": [{"name": f"{m}_{key}", "cache_key": key, "model": m, "held_out": list(held)}
             for m in MODEL_NAMES
             for key, held in ([(f"fold{k}", f) for k, f in enumerate(folds)] + [("all", [])])],
    "folds": folds,
    "notes": (f"nb37 clonality transfer: leave-{FOLD_SIZE}-out over {len(LABELLED)} labelled "
              f"donors ({len(folds)} folds) + all-donor reference, x {len(MODEL_NAMES)} arms "
              f"({', '.join(MODEL_NAMES)}). Geneformer geometric prior, HVG={HVG_TOP_N}, "
              f"n_latent={N_LATENT}, batch={BATCH_KEY}."),
}
CONFIG.write_text(json.dumps(cfg, indent=2))
print("wrote", CONFIG, f"| {len(cfg['runs'])} runs")
print(cfg["notes"])
''')

md(r"""
## Part 4 — per-fold training of the three arms · HEAVY (bsub / GPU)

`3 × (len(folds) + 1)` runs: every arm × every fold, plus one all-donor reference per arm (the fold
count falls out of how many donors clear the eligibility gate — Part 3 prints it). Each run removes
its held-out donors' cells from training entirely, then encodes **all** cells (held-out and dark
included) with that run's model.

```bash
cd notebooks/MF/jobs
ARRAY=1 ./run_clonality_folds.sh          # one GPU job per run, in parallel
# or sequentially in one job:  ./run_clonality_folds.sh
# one run (smoke test):        ./run_clonality_folds.sh --folds scvi_fold5
# a fold across every arm:     ./run_clonality_folds.sh --folds 3
```

Models cache under `models/.model_cache_semantic_clonality/<slug>/<arm>/<fold>` and latents land in
`benchmark_results/clonality_transfer/z_<arm>_<fold>.npy`; a run whose latent exists is skipped, so
re-runs are free. The next cell just waits for the files.
""")

code(r'''
# Survive a kernel restart while the bsub jobs ran: everything below needs only the slim input.
if "slim" not in dir():
    slim = sc.read_h5ad(INPUT_H5AD)
    donor_tbl = pd.read_csv(OUT_DIR / "donor_table.csv", index_col=0)
    print("reloaded slim input:", slim.shape)
if "folds" not in dir():
    folds = json.loads(CONFIG.read_text())["folds"]
LABELLED = list(donor_tbl.index[donor_tbl["group"].isin(["labelled", "labelled_small"])])
DARK = list(donor_tbl.index[donor_tbl["group"] == "dark"])
NO_CLONE = list(donor_tbl.index[donor_tbl["group"] == "no_clone"])

# label arrays used by every cell below
donors_np = slim.obs["donor"].astype(str).to_numpy()
ic = slim.obs["is_clonal"].to_numpy(dtype=bool)
ia = slim.obs["is_anchor"].to_numpy(dtype=bool)

Z_PATHS = {(m, k): OUT_DIR / f"z_{m}_fold{k}.npy"
           for m in MODEL_NAMES for k in range(len(folds))}
Z_PATHS.update({(m, "all"): OUT_DIR / f"z_{m}_all.npy" for m in MODEL_NAMES})
missing = {k: p for k, p in Z_PATHS.items() if not p.exists()}
if missing:
    raise FileNotFoundError(
        f"{len(missing)}/{len(Z_PATHS)} latents missing: {sorted(p.name for p in missing.values())}\n"
        f"submit with:  cd {NB_DIR / 'jobs'} && ARRAY=1 ./run_clonality_folds.sh")

Z = {m: {k: np.load(Z_PATHS[(m, k)]) for k in list(range(len(folds))) + ["all"]}
     for m in MODEL_NAMES}
for m, zs in Z.items():
    for k, z in zs.items():
        assert z.shape[0] == slim.n_obs, f"{m} {k}: {z.shape} vs {slim.n_obs}"
print(f"loaded {len(Z_PATHS)} latents | "
      + " | ".join(f"{m} {Z[m]['all'].shape}" for m in MODEL_NAMES))
''')

md(r"""
## Part 5 — evaluate

Four rules × three arms, all per-fold, all on the same folds, donors, labels and cells.

- `delta_centroid` — the headline: nearer-centroid to $\hat\mu_{ben}$ / $\hat\mu_{ben}+\hat\delta$.
  Parameter-free at test time.
- `delta_threshold` — same axis, cut transferred from the training donors instead of the midpoint.
- `logistic_z` — supervised ceiling on donor-centred $z$.
- `delta_centroid_null` — labels shuffled within each training donor, **refit per arm per fold from
  that arm's own latent**, so each arm carries its own floor. `auc_raw` should sit at 0.5.

Read **`auc_raw`**, not `auc`: `M.binary_scores` reports the polarity-folded $\max(a, 1-a)$, which
floors a random direction well above 0.5 and would make the nulls look informative.
""")

code(r'''
arms = [(m, Z[m]) for m in MODEL_NAMES]      # per-fold latents; dict fold -> z, "all" ignored by run_cv

res_parts, pc_parts = [], []
for name, z in arms:
    r, pc = SC.run_cv(z, slim.obs, folds, latent_name=name, train_donors_all=LABELLED, seed=SEED)
    res_parts.append(r)
    pc_parts.append(pc)
    print(f"[{name}] {len(r)} rows")
res = pd.concat(res_parts, ignore_index=True)
percell = pd.concat(pc_parts)

summary = SC.summarize(res).sort_values("auc_raw_mean", ascending=False)
with pd.option_context("display.width", 220, "display.max_columns", None):
    print(summary.to_string())
summary.to_csv(OUT_DIR / "transfer_summary.csv")
res.to_csv(OUT_DIR / "transfer_per_donor.csv", index=False)
''')

code(r'''
# Gate + winner selection. Every assert here has caught a real bug in this pipeline's history.
for m in MODEL_NAMES:                       # a latent that is identically zero for some study makes
    for k, z in Z[m].items():               # every metric on those donors meaningless (see the old
        dead = int((np.abs(z).sum(1) == 0).sum())   # published X_scVI arm: 30% of cells were zero)
        assert dead == 0, f"{m} {k}: {dead:,} all-zero latent rows"

n_held = len({d for f in folds for d in f})
for m in MODEL_NAMES:
    sub = res[res.latent == m]
    assert len(sub) == n_held * len(SC.RULES), f"{m}: {len(sub)} rows, expected {n_held * len(SC.RULES)}"
    assert sub.groupby(["fold", "donor"])["n"].nunique().eq(1).all(), f"{m}: rules disagree on cells"

nulls = (res[res.rule == "delta_centroid_null"].groupby("latent")["auc_raw"].mean())
for m, a in nulls.items():
    print(f"[{m}] shuffled-δ null auc_raw = {a:.3f}" + ("" if 0.35 <= a <= 0.65 else "   <-- WARN"))

# best RULE per arm (nulls excluded), then the best arm — these drive the figures and Part 7-8
real = res[~res.rule.str.endswith("_null")]
mean_auc = real.groupby(["latent", "rule"])["auc_raw"].mean()
BEST_RULE = {m: mean_auc[m].idxmax() for m in MODEL_NAMES}
BEST_LATENT = max(MODEL_NAMES, key=lambda m: mean_auc[(m, BEST_RULE[m])])
print(f"\nbest rule per arm: " + ", ".join(f"{m}={BEST_RULE[m]} ({mean_auc[(m, BEST_RULE[m])]:.3f})"
                                          for m in MODEL_NAMES))
print(f"winner: {BEST_LATENT} / {BEST_RULE[BEST_LATENT]} -> used for the dark-donor transfer")
''')

md("## Part 6 — figures")

code(r'''
# Fig 1 — does the transfer work? Best rule per arm next to that arm's own shuffled-δ null.
ARM_COL = {"semantic": "#4c78a8", "ldvae": "#f58518", "scvi": "#888"}
piv = res.groupby(["latent", "rule"])["auc_raw"].agg(["mean", "std"])
x = np.arange(len(MODEL_NAMES))
fig, ax = plt.subplots(figsize=(5, 3))
# real bars carry the arm colour; the nulls are one neutral colour so the legend reads unambiguously
real_rule = [BEST_RULE[m] for m in MODEL_NAMES]
ax.bar(x - 0.19, [piv.loc[(m, r), "mean"] for m, r in zip(MODEL_NAMES, real_rule)], width=0.36,
       yerr=[piv.loc[(m, r), "std"] for m, r in zip(MODEL_NAMES, real_rule)], capsize=2.5,
       color=[ARM_COL[m] for m in MODEL_NAMES], label="best rule")
ax.bar(x + 0.19, [piv.loc[(m, "delta_centroid_null"), "mean"] for m in MODEL_NAMES], width=0.36,
       yerr=[piv.loc[(m, "delta_centroid_null"), "std"] for m in MODEL_NAMES], capsize=2.5,
       color="#d9d9d9", hatch="///", edgecolor="#999", label="shuffled-δ null")
ax.axhline(0.5, ls=":", lw=0.8, c="k")
ax.set_xticks(x)
ax.set_xticklabels([f"{m}\n{BEST_RULE[m]}" for m in MODEL_NAMES], fontsize=7)
ax.set_ylabel("AUROC vs ALICE (held-out donor)")
ax.set_ylim(0.3, 1.02)
_w = f"{BEST_LATENT} {piv.loc[(BEST_LATENT, BEST_RULE[BEST_LATENT]), 'mean']:.2f}"
ax.set_title(f"Clonality transfers to unseen samples — best: {_w}", fontsize=9)
ax.legend(fontsize=6.5, frameon=True, framealpha=0.9, edgecolor="none", loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "clonality_transfer_auroc.png", dpi=150, bbox_inches="tight")
plt.show()
''')

code(r'''
# Fig 2 — is the tumour BURDEN right? predicted vs true clonal fraction, one dot per held-out donor.
fig, ax = plt.subplots(figsize=(5, 3))
for m in MODEL_NAMES:
    d = res[(res.latent == m) & (res.rule == BEST_RULE[m])]
    r2 = SC.r2_latent(d["pred_frac"], d["true_frac"])
    ax.scatter(d["true_frac"], d["pred_frac"], s=22, alpha=0.8, color=ARM_COL[m],
               label=f"{m} / {BEST_RULE[m]} (R²={r2:.2f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("true ALICE clonal fraction")
ax.set_ylabel("predicted clonal fraction")
ax.set_title("Per-donor tumour burden on held-out samples", fontsize=9)
ax.legend(fontsize=6.5, frameon=False)
fig.tight_layout()
fig.savefig(FIG_DIR / "clonality_transfer_fraction.png", dpi=150, bbox_inches="tight")
plt.show()
''')

md(r"""
## Part 7 — negative controls

The rule must *not* fire where there is nothing to find:

1. **`no_clone` donors** — have TCR and *zero* ALICE-clonal cells. Predicted clonal fraction
   should be low. (Donors with a clone too small to be a labelled unit are `labelled_small`, not
   `no_clone` — they carry a clone and would be the wrong thing to assert cleanliness on.)
2. **HC donors** — healthy skin, benign by definition.
3. **CD8 cells** — read straight off the full object; MF malignancy is CD4, so these should be
   near-zero. Scored with the all-donor model's δ̂ (they were never in the CD4 training set).

Everything from here on uses the **winning arm and rule** chosen in Part 5, on that arm's all-donor
latent.
""")

code(r'''
z_win = Z[BEST_LATENT]["all"]
rule_win = BEST_RULE[BEST_LATENT]
delta_all, delta_tbl = SC.fit_delta(z_win, donors_np, ic, ia, LABELLED)
print(f"winner {BEST_LATENT} / {rule_win} | per-donor δ_d norms:",
      delta_tbl["norm"].describe().round(2).to_dict())

# delta_centroid needs nothing fitted; the other rules carry a cut learned on the labelled donors
fitted_win = (None if rule_win == "delta_centroid" else
              SC.fit_rules(z_win, donors_np, ic, ia, LABELLED, delta_all, seed=SEED)[rule_win])

ctrl_donors = NO_CLONE + [d for d in donor_tbl.index
                          if str(donor_tbl.loc[d, "disease"]) == "HC" and d not in NO_CLONE]
_, ctrl_tbl = SC.apply_to_donors(z_win, slim.obs, ctrl_donors, delta_all, seed=SEED,
                                 rule=rule_win, fitted=fitted_win)
ctrl_tbl = ctrl_tbl.merge(donor_tbl[["group", "n_clonal", "n_anchor"]], left_on="donor",
                          right_index=True, how="left")
print("\nnegative-control donors (predicted clonal fraction should be low):")
print(ctrl_tbl.sort_values("pred_clonal_frac", ascending=False).round(3).to_string(index=False))

lab_frac = res[(res.latent == BEST_LATENT) & (res.rule == rule_win)]["pred_frac"].mean()
print(f"\nmean predicted fraction — labelled held-out donors {lab_frac:.3f} "
      f"vs negative controls {ctrl_tbl.pred_clonal_frac.mean():.3f}")
''')

md(r"""
## Part 8 — apply to the dark donors

Winning arm's all-donor model, `δ̂` from every labelled donor, the winning rule per dark donor with
that donor's own GMM-estimated benign centroid.

**These donors have no TCR and no published malignant label**, so nothing below is validated
against external truth — the accuracy to quote is Part 5's held-out number, not anything here.
The checks available are internal: does the predicted burden track disease stage, and do the
predicted-clonal cells carry the MF tumour program.
""")

code(r'''
dark_percell, dark_tbl = SC.apply_to_donors(z_win, slim.obs, DARK, delta_all, seed=SEED,
                                           rule=rule_win, fitted=fitted_win)
dark_tbl = dark_tbl.merge(donor_tbl[["n_cells"]].rename(columns={"n_cells": "n_cd4"}),
                          left_on="donor", right_index=True, how="left")
print(dark_tbl.sort_values("pred_clonal_frac", ascending=False).round(3).to_string(index=False))
print(f"\n{len(dark_tbl)} dark donors | {len(dark_percell):,} cells | "
      f"predicted clonal {int(dark_percell.clonal_call.sum()):,} "
      f"({dark_percell.clonal_call.mean():.1%})")
''')

code(r'''
# Fig 3 — predicted burden per dark donor, next to the labelled donors' true burden for scale.
fig, ax = plt.subplots(figsize=(5, 3))
d = dark_tbl.sort_values("pred_clonal_frac", ascending=False)
_studies = list(dict.fromkeys(d["study"]))
_pal = dict(zip(_studies, ["#4c78a8", "#f58518", "#54a24b", "#c0392b", "#888", "#9467bd"] * 3))
ax.bar(np.arange(len(d)), d["pred_clonal_frac"], color=[_pal[s] for s in d["study"]], width=0.75)
ax.axhline(donor_tbl.loc[LABELLED, "clonal_frac"].median(), ls="--", lw=0.8, c="k",
           label="median true burden, labelled donors")
ax.set_xticks(np.arange(len(d)))
ax.set_xticklabels(d["donor"], rotation=75, ha="right", fontsize=4.5)
ax.set_ylabel("predicted clonal fraction")
ax.set_title(f"Projected tumour burden in the TCR-less samples — {BEST_LATENT} (unvalidated)",
             fontsize=9)
handles, labels = ax.get_legend_handles_labels()
handles += [plt.Line2D([], [], lw=0, marker="s", color=_pal[s]) for s in _studies]
ax.legend(handles, labels + _studies, fontsize=6, frameon=False)
fig.tight_layout()
fig.savefig(FIG_DIR / "clonality_dark_burden.png", dpi=150, bbox_inches="tight")
plt.show()
''')

code(r'''
# Do the predicted-clonal cells carry the MF tumour program? rank_genes_groups within the dark set.
ad_dark = slim[dark_percell.index].copy()
ad_dark.obs["pred_clonal"] = pd.Categorical(
    np.where(dark_percell["clonal_call"].to_numpy(), "clonal", "benign"),
    categories=["benign", "clonal"])
sc.pp.normalize_total(ad_dark, target_sum=1e4)
sc.pp.log1p(ad_dark)
sc.tl.rank_genes_groups(ad_dark, "pred_clonal", groups=["clonal"], reference="benign",
                        method="wilcoxon")
top = pd.DataFrame(ad_dark.uns["rank_genes_groups"]["names"])["clonal"].head(25).tolist()
print("top 25 markers of predicted-clonal cells (dark donors):")
print(", ".join(top))
del ad_dark
gc.collect()
''')

md("## Part 9 — persist")

code(r'''
# (a) held-out fold scores/calls, every arm x rule; (b) the applied call on the dark donors.
fold_out = percell.copy()
fold_out.index.name = "obs_name"
fold_out.to_parquet(FOLD_PARQUET)
print("wrote", FOLD_PARQUET, fold_out.shape)

trans = dark_percell.copy()
trans["study"] = slim.obs["study"].astype(str).reindex(trans.index)
trans["disease"] = slim.obs["disease"].astype(str).reindex(trans.index)
trans["model"] = f"{BEST_LATENT}_alldonor/{rule_win}"
trans.index.name = "obs_name"
trans.to_parquet(TRANS_PARQUET)
print("wrote", TRANS_PARQUET, trans.shape)

dark_tbl.to_csv(OUT_DIR / "dark_donor_predictions.csv", index=False)
print("wrote", OUT_DIR / "dark_donor_predictions.csv")
''')

md(r"""
### Reading the results

- **Headline** = the winning arm/rule in `transfer_summary.csv`: `auc_raw` vs ALICE on donors the
  encoder never saw. That is the number that licenses (or does not license) Part 8.
- **`auc_raw` of each arm's `delta_centroid_null`** should sit at ~0.5. If one does not, that arm's
  fold/label wiring is leaking — the result is a bug, not a finding.
- **`semantic` − `ldvae`** is the semantic prior's contribution, isolated: the two differ by
  `coherence_weight` alone. **`ldvae` − `scvi`** is the linear positive-weight decoder's
  contribution, though scvi also loses the decorrelation penalty (see the header).
- Rank by `auc_raw` and `balanced_acc`, **not `f1`**: most held-out donors are majority-clonal, so
  an all-positive call scores F1 ≈ 0.85 while being uninformative.
- `centroid_r2` answers the perturbation-trial question directly — does `μ̂_ben + δ̂` land on the real
  clonal centroid of an unseen donor — but note it stays high even for the shuffled-δ null, because
  `μ̂_mal ≈ μ̂_ben` and the donor offset dominates. Compare it across arms, never against 1.0.
""")

nb = {"cells": C,
      "metadata": {"kernelspec": {"display_name": "neural_nmf_env", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
p = Path("/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/"
         "37_semantic_clonality_transfer.ipynb")
p.write_text(json.dumps(nb, indent=1))
print("wrote", p, len(C), "cells")
