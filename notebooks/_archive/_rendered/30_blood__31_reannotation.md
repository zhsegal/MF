# 30_blood/31_reannotation.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/30_blood/31_reannotation.ipynb` and is never run._

# 31 — Blood (PBMC) re-annotation on the **v2** atlas

Runs the **`10b_skin_reannotation`** pipeline on the blood compartment of the v2 atlas
(21 cohorts, 2,157,693 cells → **811,024 blood**, 89 samples / 10 studies / 131 donor-units /
75 patients; v1 blood was 423,042 / 49 / 6). Blood carries `cell_type == "Unknown"` for 100 % of
its cells, so this is the only source of blood cell-type labels; `32_atlas_descriptive` gates on
the row count of what this notebook writes and reports blood as unannotated until it lands.

Same order as 10b: log-normalize → neighbors/Leiden on `X_mrvi_u` → subsampled UMAP views →
marker dot plot → hand-filled cluster→cell-type map → T/NK sub-clustering → re-cluster the
leftovers → merge → `cell_type_final`.

## Aligned to 10b

- **Same label names.** `CD4_Treg` (not `Tregs`), `Unk` (not `UNK`), `CD4` / `CD8` / `B` /
  `Plasma` / `Myeloid` / `Mast` / `T` / `NK` exactly as 10b writes them, so
  `skin_cell_type_final.csv` and `blood_cell_type_final.csv` concatenate into one vocabulary
  with no remapping.
- **Same marker genes** for every lineage the two compartments share — 10b's panels are pasted
  verbatim and only *extended*, never edited.
- **Same `plot_view()`** subsampled-UMAP helper (at `PLOT_N=300_000` here vs 10b's 150k, since
  blood is smaller): at 811k cells a full-object `sc.tl.umap` is hours of CPU and an unreadable
  blob. Leiden and every label still run on all cells.
- **Same two-pass hand-map workflow** and the same `cluster2ct` / `cluster2ct_T` /
  `cluster2ct_unk` names.

## Where it still differs, and why

1. **PBMC marker panels.** Skin's keratinocyte / fibroblast / melanocyte / endothelium panels are
   biologically absent here — kept verbatim from 10b as a *contamination tripwire*, not as callable
   lineages. Added: monocyte (CD14 vs CD16), cDC1/cDC2, pDC, platelet, erythroid, HSPC, MAIT, γδ,
   NK, and an explicit **Sézary** panel split into GAIN and LOSS directions.
2. **No Li2024 purity check.** Blood has no external labels, so 10b's
   `cell_type_broad × cell_type` crosstab has nothing to validate against. Replaced by per-cluster
   **TCR / clonality** tables — the only atlas-wide call that exists in blood — and by the
   healthy-blood composition reference in §14b.
3. **Frozen clusterings.** 10b recomputes Leiden every run and asserts only that the map's keys
   match the categories — but the categories are `"0".."N"` either way, so a renumbered Leiden
   passes while silently invalidating the hand map. Here every Leiden is persisted, reloaded on
   re-run, and each hand map additionally asserts the per-cluster **sizes** it was authored
   against.

### The dominant population is the tumour

Blood is 57 % SS + 24 % CTCL_other + 13 % MF (`disease`), and per-sample dominant-clone malignant
fractions run past 85 %. The prior on the largest CD4 cluster is therefore **malignant**, not
healthy memory CD4 — the inverse of the skin reading. Sézary cells are classically
CD4⁺CCR7⁺CD27⁺CD62L⁺, a *central-memory* surface phenotype, so `CCR7`/`SELL`/`TCF7` positivity
does **not** exclude tumour. Read §11b (per-cluster malignancy) **before** filling `cluster2ct_T`.

v2 also brings the healthy reference this notebook lacked: **47,679 HC blood cells across 7
donors** (`N1-N3` ren2023, `HB1-HB3` gaydosik2022, `H__HC1` herrera2021), against v1's single
4,481-cell sample. §14b uses all of them.

### Sections

| § | what |
|---|---|
| 0 | parameters, path constants, helpers (`plot_view`, `frozen_leiden`, `apply_hand_map`, `cluster_qc`) |
| 1 | **sample scope** — the 89 blood samples, printed and auditable (login-node safe) |
| 2 | load the blood subset + MrVI latents, with alignment asserts |
| 3 | log-normalize (raw counts kept in `layers["counts"]`) |
| 4 | subsampled UMAP views (plotting only) |
| 5 | broad Leiden (frozen to CSV) |
| 6 | UMAP overview |
| 7 | broad marker dot plot (10b panel + PBMC panels) + per-cluster QC |
| 8 | **hand map** → `cell_type_broad` |
| 9 | per-cluster TCR / clonality validation |
| 10–12 | T/NK sub-clustering → `cell_type_T` |
| 12.5 | re-cluster the still-unknown T cells |
| 13 | `sezary_like` (a separate column, *not* a cell-type level) |
| 14–15 | merge → `cell_type_final`, healthy-reference sanity check, final UMAPs |
| 16 | write `blood_cell_type_final.csv`, `blood_T_annotated.h5ad`, provenance JSON |
| 17 | how `12_atlas_descriptive` consumes this |

> **HEAVY — run on the GPU kernel** (`neural_nmf_env`). `sc.pp.neighbors` on 811k × 10 and the
> three `plot_view` embeddings are the expensive steps; everything caches.
>
> `jobs/run_mrvi_joint.py` hardcodes `SAMPLE_KEY="sample_id"`, and geskin's `sample_id` is an HTO
> **lane** pooling 5–6 donors — so the sample-aware `z` will not resolve geskin donors. Irrelevant
> here (this notebook uses `u` throughout); relevant to any later blood `z` work.

### Two-pass workflow

**Pass 1** — run everything with the three `cluster2ct*` dicts left empty. You get the dot plots,
the QC tables, the malignancy tables, and a printed `{"0": "", ...}` template plus the cluster
sizes. **Pass 2** — paste the filled dicts and the `CLUSTER_SIZES_*` locks, re-run. Every heavy
step is cached, so pass 2 takes minutes.

**The v1 maps below are commented out, not deleted.** Cluster IDs renumber when the atlas and the
MrVI latent change, so a v1 map applied to v2 clusters is silently wrong — exactly what the
`CLUSTER_SIZES_*` locks now refuse to allow. They are kept only as a reading aid.

```python  # cell 1
# ============================================================================
# §0  Parameters, paths, helpers
# ============================================================================
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import gc, json, sys, warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from natsort import natsorted

warnings.filterwarnings("ignore")


def _resolve_nb_dir() -> Path:
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data from {start}")


NB_DIR = _resolve_nb_dir()
sys.path.insert(0, str(NB_DIR / "helpers"))
OUT = NB_DIR / "data" / "atlas_joint"
FIG = NB_DIR / "figures" / "blood_reannotation"; FIG.mkdir(parents=True, exist_ok=True)
TAB = NB_DIR / "tables"; TAB.mkdir(exist_ok=True)

SEED = 0
np.random.seed(SEED)
sc.settings.verbosity = 1
mpl.rcParams["figure.dpi"] = 110
mpl.rcParams["savefig.bbox"] = "tight"
print("NB_DIR =", NB_DIR)

# ---------------------------------------------------------------- switches
LATENT_SCOPE = "blood"   # "blood" = blood-native MrVI (--tag blood) | "joint" = slice the full-atlas latent
INCLUDE_LN = False       # LN (1,142 cells) is a different tissue; would contaminate a PBMC panel
FULL_GENES = False       # True -> joint_annotated.h5ad (42,347 genes) instead of the 10k HVG
ALLOW_JOINTLAT_LABELS = False   # guard on writing canonical labels from the joint latent

# --- v2 atlas contract (docs/ATLAS_V2_DEDUP_GATE.md sec 4c) --------------------------------
# v1 was 1,173,694 / Blood 423,042. Every cache below is invalidated by the row count, and the
# v1 caches stay on disk under their old names because every v2 cache carries CACHE_V.
N_TOTAL, N_BLOOD, N_LN, N_SKIN = 2_157_693, 811_024, 1_142, 1_345_527
V2_N_SAMPLES_BLOOD, V2_N_STUDIES_BLOOD = 89, 10
# Blood cohorts that exist only in the v2 build. Their presence is what distinguishes v2 from the
# v1 atlas at the same paths — run_build_joint.py rebuilds these files IN PLACE.
V2_BLOOD_DATASETS = {"ren23", "harro23", "dorando26", "gaydosik22"}
CACHE_V = "v2"

if LATENT_SCOPE == "blood":
    SRC_H5AD = OUT / "joint_mrvi_input_blood.h5ad"      # already blood-only: 811,024 rows
    MRVI_U = OUT / "joint_X_mrvi_u_blood.npy"
    MRVI_Z = OUT / "joint_X_mrvi_blood.npy"
elif LATENT_SCOPE == "joint":
    SRC_H5AD = OUT / "joint_mrvi_input.h5ad"            # full atlas: 2,157,693 rows
    MRVI_U = OUT / "joint_X_mrvi_u.npy"
    MRVI_Z = OUT / "joint_X_mrvi.npy"
else:
    raise ValueError(LATENT_SCOPE)

# FULL_GENES swaps the GENE space only. The row space is then the full atlas, and §2 reconciles
# that against whichever latent was chosen by row count — a blood latent (811,024) is matched to
# the blood rows of the full atlas, whose order is identical because joint_mrvi_input_blood.h5ad
# was itself built by subsetting joint_annotated.h5ad in file order (§2 asserts this on cell_id).
BLOOD_HVG_H5AD = OUT / "joint_mrvi_input_blood.h5ad"    # cell_id reference for that assert
if FULL_GENES:
    SRC_H5AD = OUT / "joint_annotated.h5ad"             # 42,347 genes, full-atlas rows

# Genes the hand annotation cannot be done without. HVG selection is data-driven, so a gene can
# vanish: in the v2 blood-native 10k set CD4 itself is absent (blood is dominated by CD4
# malignancy, so CD4 has little variance across cells and misses the top 10k). §2 says what to do.
REQUIRED_MARKERS = ["CD3D", "CD3E", "TRAC", "CD8A", "CD8B", "IL7R", "CD40LG", "FOXP3", "NKG7",
                    "GNLY", "MS4A1", "CD14", "LYZ", "FCGR3A", "PPBP", "HBB", "KIR3DL2", "CD7",
                    "DPP4"]

# Both latent modes yield 811,024 rows, so a row-count cache guard CANNOT tell them apart:
# every cache filename carries the scope as well as the build version.
SUF = f"_{CACHE_V}" if LATENT_SCOPE == "blood" else f"_jointlat_{CACHE_V}"

# ---------------------------------------------------------------- caches (scope-specific)
LEIDEN_CSV = OUT / f"blood_leiden_broad{SUF}.csv"
T_LEIDEN_CSV = OUT / f"blood_T_leiden{SUF}.csv"
UNK_LEIDEN_CSV = OUT / f"blood_T_unk_leiden{SUF}.csv"
# UMAP caches are named by plot_view(): blood_umap_<key><SUF>_<parent_n>_<sub_n>.npy
# ---------------------------------------------------------------- canonical deliverables (no SUF)
SCOPE_CSV = OUT / "blood_scope_roster.csv"
CLUSTER_QC_CSV = OUT / "blood_cluster_qc.csv"
T_CLUSTER_QC_CSV = OUT / "blood_T_cluster_qc.csv"
CT_FINAL_CSV = OUT / "blood_cell_type_final.csv"        # <- what nb32 gates on
T_ANNOT = OUT / "blood_T_annotated.h5ad"
PROV_JSON = OUT / "blood_annotation_provenance.json"

# ---------------------------------------------------------------- resolutions
# 10b uses 0.7 for the broad pass. Blood keeps 1.0: pDC / cDC1 / HSPC are 0.1-2% of PBMC and a
# rare-lineage merge at the broad pass is unrecoverable downstream. T and unknown match 10b.
LEIDEN_RES = 1.0
T_LEIDEN_RES = 1.0      # == 10b's T_LEIDEN_RES
UNK_LEIDEN_RES = 0.5    # == 10b's UNK_LEIDEN_RES
UNK_LABELS = {"Unk", "UNK", "Unknown", "unknown", "", "nan", "None"}   # 10b's set, + case variants
T_SUBSET_LABELS = {"T", "NK"}   # NK joins the T pass: NK/CD8-EMRA boundaries resolve best together

# UMAP is a visualization artifact only — Leiden and every label run on all cells. umap-learn is
# CPU-only here, and embedding all 811k points costs hours. 10b uses 150k out of 1.35M (11%);
# 300k out of 811k is 37% of blood, so rare clusters are comfortably sampled.
PLOT_N = 300_000

# ---------------------------------------------------------------- label vocabulary (10b names)
# 10b's broad vocabulary is the base. Skin-only lineages are dropped (they are a contamination
# tripwire here, not callable); PBMC lineages skin has no equivalent for are added.
VOCAB_BROAD = {
    # --- shared with 10b, spelled identically
    "T", "B", "Plasma", "Myeloid", "Mast", "Unk",
    # --- blood-only
    "NK", "Mono_CD14", "Mono_CD16", "cDC", "pDC",
    "Platelet", "Erythroid", "HSPC", "Prolif", "LowQC", "Doublet",
}
# 10b's cell_type_T vocabulary is {CD4, CD8, CD4_Treg, Unk}; blood adds the innate/unconventional
# subsets that skin does not resolve.
VOCAB_T = {"CD4", "CD8", "CD4_Treg", "gdT", "MAIT", "NK", "Prolif", "LowQC", "Doublet", "Unk"}

print(f"LATENT_SCOPE={LATENT_SCOPE!r}  INCLUDE_LN={INCLUDE_LN}  FULL_GENES={FULL_GENES}  "
      f"CACHE_V={CACHE_V!r}")
print("source h5ad :", SRC_H5AD.name, "(exists)" if SRC_H5AD.exists() else "(MISSING)")
print("latent u    :", MRVI_U.name, "(exists)" if MRVI_U.exists() else "(MISSING)")
print("latent z    :", MRVI_Z.name, "(exists)" if MRVI_Z.exists() else "(MISSING)")
assert MRVI_U.stat().st_mtime > SRC_H5AD.stat().st_mtime, (
    "blood MrVI latents are OLDER than the HVG input they should have been trained on; "
    "rerun jobs/run_mrvi_joint.sh with --subset-compartment Blood --tag blood --force")
```

```
NB_DIR = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
LATENT_SCOPE='blood'  INCLUDE_LN=False  FULL_GENES=False  CACHE_V='v2'
source h5ad : joint_mrvi_input_blood.h5ad (exists)
latent u    : joint_X_mrvi_u_blood.npy (exists)
latent z    : joint_X_mrvi_blood.npy (exists)
```

```python  # cell 2
# ============================================================================
# §0b  Helpers: booleans, subsampled UMAP views, frozen Leiden, hand maps, QC tables
# ============================================================================
def _bool(s) -> np.ndarray:
    """Coerce a bool / object / categorical / string column to a real bool array.

    Load-bearing: several of these columns round-trip through the h5ad as strings, and
    `np.asarray(["False"]).astype(bool)` is `True`.
    """
    return pd.Series(np.asarray(s)).astype(str).isin(["True", "true", "1", "1.0"]).to_numpy()


def plot_view(a, key, n=PLOT_N, seed=SEED):
    """A subsampled copy of `a` carrying a UMAP, for plotting only.  (10b sec 3, verbatim logic)

    Never use the result to compute labels — it is a random subset. The embedding caches as
    blood_umap_<key><SUF>_<parent_n>_<sub_n>.npy; the parent row count is in the name so a
    re-annotation that changes the subset cannot silently reuse the old embedding, and SUF carries
    the latent scope so the two same-shape latents can never collide. Cells are drawn uniformly:
    at PLOT_N=300k out of 811k (37%) a cluster holding 0.1% of cells still keeps ~300 points.
    """
    if a.n_obs <= n:
        sub, idx = a.copy(), np.arange(a.n_obs)
    else:
        idx = np.sort(np.random.default_rng(seed).choice(a.n_obs, n, replace=False))
        sub = a[idx].copy()
    cache = OUT / f"blood_umap_{key}{SUF}_{a.n_obs}_{sub.n_obs}.npy"
    if cache.exists():
        sub.obsm["X_umap"] = np.load(cache)
        print(f"plot_view[{key}]: {sub.n_obs:,}/{a.n_obs:,} cells, cached UMAP")
    else:
        sc.pp.neighbors(sub, use_rep="X_mrvi_u", random_state=seed)
        sc.tl.umap(sub, random_state=seed)
        np.save(cache, sub.obsm["X_umap"])
        print(f"plot_view[{key}]: {sub.n_obs:,}/{a.n_obs:,} cells, computed UMAP ->", cache.name)
    sub.uns["plot_idx"] = idx
    return sub


def sync_view(view, parent, *cols):
    """Copy obs columns from the parent onto a plot_view subsample, preserving dtype.

    plot_view() snapshots obs at the time it is called, so any column created later (leiden,
    cell_type_*) is absent from the view. A bare `.to_numpy()[idx]` assignment would also drop the
    categorical dtype, and save_umap() needs a categorical to draw a legend rather than a colourbar.
    """
    idx = view.uns["plot_idx"]
    for c in cols:
        s = parent.obs[c]
        view.obs[c] = (pd.Categorical.from_codes(s.cat.codes.to_numpy()[idx], dtype=s.dtype)
                       if isinstance(s.dtype, pd.CategoricalDtype)
                       else s.to_numpy()[idx])
    return view


def coerce_latents(a):
    """pynndescent's numba kernels reject read-only / non-contiguous / non-float32 arrays."""
    for k in ("X_mrvi_u", "X_mrvi_z"):
        if k not in a.obsm:
            continue
        arr = a.obsm[k]
        if not (arr.flags.writeable and arr.flags.c_contiguous and arr.dtype == np.float32):
            a.obsm[k] = np.ascontiguousarray(arr, dtype=np.float32).copy()
            print(f"  {k}: coerced to a writable C-contiguous float32 array")


def frozen_leiden(a, csv: Path, key: str, res: float):
    """Reload a persisted Leiden if it covers this object, else compute it and freeze it.

    Why this and not 10b's recompute-every-run: the hand-filled cluster maps below are authored
    against specific cluster IDs, and Leiden IDs are not stable across runs (the pynndescent kNN
    graph is thread-nondeterministic). 10b only asserts the *names* match, which they always do.
    """
    ids = a.obs["cell_id"].astype(str)
    if csv.exists():
        s = pd.read_csv(csv, dtype=str).set_index("cell_id")[key]
        if len(s) == a.n_obs and ids.isin(s.index).all():
            vals = s.reindex(ids).to_numpy()
            a.obs[key] = pd.Categorical(vals, categories=natsorted(pd.unique(vals)))
            print(f"reloaded FROZEN {key} from {csv.name}: {a.obs[key].nunique()} clusters")
            return
        raise AssertionError(
            f"{csv.name} does not cover the current scope ({len(s):,} rows vs {a.n_obs:,}).\n"
            f"Delete it to recluster — but then the hand map for {key} MUST be re-authored.")
    if "neighbors" not in a.uns:
        sc.pp.neighbors(a, use_rep="X_mrvi_u", random_state=SEED)
    sc.tl.leiden(a, resolution=res, random_state=SEED, key_added=key,
                 flavor="igraph", n_iterations=2, directed=False)
    pd.DataFrame({"cell_id": ids.to_numpy(),
                  key: a.obs[key].astype(str).to_numpy()}).to_csv(csv, index=False)
    print(f"computed {key} (res={res}): {a.obs[key].nunique()} clusters -> froze to {csv.name}")


def apply_hand_map(a, key: str, mapping: dict, vocab, authored_sizes: dict, out_col: str) -> bool:
    """Assign `out_col` from a hand-filled cluster map, with all four guards."""
    cats = [str(c) for c in a.obs[key].cat.categories]
    now = {c: int((a.obs[key].astype(str) == c).sum()) for c in cats}
    if (not mapping) or any(v == "" for v in mapping.values()):
        print(f"--- {out_col} UNFILLED: read the dot plot + the QC table, then paste ---")
        print("{" + ", ".join(f'"{c}": ""' for c in cats) + "}")
        print(f"\nCLUSTER_SIZES = {now!r}")
        print(f"\nallowed labels: {sorted(vocab)}")
        return False
    assert set(mapping) == set(cats), (
        f"map keys != {key} categories\n"
        f"  missing: {natsorted(set(cats) - set(mapping))}\n"
        f"  extra  : {natsorted(set(mapping) - set(cats))}")
    bad = set(mapping.values()) - set(vocab)
    assert not bad, f"labels outside the vocabulary: {sorted(bad)}\n  allowed: {sorted(vocab)}"
    if authored_sizes:
        assert now == {str(k): int(v) for k, v in authored_sizes.items()}, (
            f"{key} has changed since this map was authored — the cluster IDs no longer mean what "
            f"the map says.\n  authored: {authored_sizes}\n  now     : {now}")
    else:
        print(f"!! CLUSTER_SIZES_* is empty — paste `{now!r}` to lock the map to this clustering")
    a.obs[out_col] = pd.Categorical(a.obs[key].astype(str).map(mapping))
    print(a.obs[out_col].value_counts(dropna=False).to_string())
    return True


def add_tcr_flags(a):
    """Materialise the boolean TCR/malignancy flags + the VDJ-availability mask."""
    for src, dst in [("has_tcr", "_tcr"), ("is_malignant", "_mal"),
                     ("is_dominant_clone", "_dom"), ("is_expanded", "_exp")]:
        levels = set(a.obs[src].astype(str).unique())
        assert levels <= {"True", "False", "nan", "", "None"}, \
            f"{src} has unexpected levels {sorted(levels)[:8]} — _bool() would mislabel them"
        a.obs[dst] = _bool(a.obs[src])
    a.obs["_vdj_ok"] = ~a.obs["sample_id"].astype(str).isin(NO_VDJ_SAMPLES).to_numpy()
    print(f"TCR+ {a.obs['_tcr'].sum():,} ({100*a.obs['_tcr'].mean():.1f}%) | "
          f"dominant-clone malignant {a.obs['_mal'].sum():,} ({100*a.obs['_mal'].mean():.1f}%) | "
          f"in a VDJ-bearing sample {a.obs['_vdj_ok'].sum():,} "
          f"({100*a.obs['_vdj_ok'].mean():.1f}%)")


def cluster_qc(a, key: str) -> pd.DataFrame:
    """Per-cluster TCR / clonality / QC / provenance table.

    This replaces 10b's `cell_type_broad x cell_type` purity check, which has no counterpart in
    blood (cell_type == "Unknown" for 100% of blood cells).
    """
    o = a.obs
    g = o.groupby(key, observed=True)
    gv = o[o["_vdj_ok"]].groupby(key, observed=True)
    gt = o[o["_tcr"]].groupby(key, observed=True)
    idx = g.size().index

    def _top_clone(s):
        s = s[s.astype(str).ne("") & s.notna()]
        return round(float(s.value_counts(normalize=True).iloc[0]), 3) if len(s) else np.nan

    t = pd.DataFrame(index=idx)
    t["n"] = g.size()
    t["pct_of_obj"] = (100 * g.size() / a.n_obs).round(2)
    t["pct_tcr"] = (100 * g["_tcr"].mean()).round(1)
    t["pct_tcr_vdjok"] = (100 * gv["_tcr"].mean()).reindex(idx).round(1)
    t["pct_mal"] = (100 * g["_mal"].mean()).round(1)
    t["pct_mal_vdjok"] = (100 * gv["_mal"].mean()).reindex(idx).round(1)
    t["pct_mal_of_tcr"] = (100 * gt["_mal"].mean()).reindex(idx).round(1)
    t["pct_dom"] = (100 * g["_dom"].mean()).round(1)
    t["pct_expanded"] = (100 * g["_exp"].mean()).round(1)
    t["n_tcr"] = gt.size().reindex(idx).fillna(0).astype(int)
    t["n_clones"] = g["clone_id"].apply(lambda s: s[s.astype(str).ne("")].nunique())
    t["top_clone_frac"] = g["clone_id"].apply(_top_clone)
    t["med_clone_size"] = g["clone_size"].median().round(0)
    t["med_genes"] = g["n_genes"].median().astype(int)
    t["med_mito"] = g["pct_mito"].median().round(2)
    t["med_doublet"] = g["doublet_score"].median().round(3)
    t["n_donors"] = g["donor"].nunique()
    t["n_patients"] = g["patient_key"].nunique()
    t["n_samples"] = g["sample_id"].nunique()
    t["n_studies"] = g["study"].nunique()
    t["top_donor_frac"] = g["donor"].apply(
        lambda s: round(float(s.value_counts(normalize=True).iloc[0]), 3))
    t["top_study"] = g["study"].apply(lambda s: s.value_counts().idxmax())
    t["flag"] = np.select(
        [(t["n_tcr"] < 50),
         (t["med_genes"] < 800) | (t["med_mito"] > 10),
         (t["top_donor_frac"] > 0.8) & (t["pct_mal_of_tcr"] >= 50)],
        ["unscorable (<50 TCR+)", "LowQC candidate", "donor-private TUMOUR (expected)"],
        default="")
    return t.reindex(natsorted(t.index.astype(str)))


QC_READING = """
How to read the cluster QC table
  pct_mal_of_tcr >= 50 + high top_clone_frac + CD4 markers  -> Sezary (malignant CD4)
  pct_mal ~ 0  AND  pct_tcr_vdjok ~ 0                       -> genuine non-T lineage
  pct_mal ~ 0  AND  pct_tcr_vdjok high                      -> benign reactive T
  pct_mal low only because the cluster is zero-VDJ-dominated -> uninformative; read pct_mal_vdjok
  n_tcr < 50                                                -> UNSCORABLE, not benign
A donor-private cluster is NOT automatically a batch artefact: clonal tumours are donor-private by
construction. Low top-donor entropy PLUS low pct_mal_of_tcr PLUS few donors is what indicates batch.
"""


def filter_panel(panel: dict, present: set, name: str) -> dict:
    """Filter a marker panel to genes present — LOUDLY.

    10b's silent `[g for g in v if g in present]` hides which markers vanished. The blood-native
    HVG set differs from the skin one, so the drop list must be computed at runtime.
    """
    kept, dropped = {}, {}
    for k, v in panel.items():
        have = [g for g in v if g in present]
        miss = [g for g in v if g not in present]
        if have:
            kept[k] = have
        else:
            print(f"  !! panel {k!r} is EMPTY — all of {v} absent; that lineage CANNOT be called "
                  "from RNA in this gene space (try FULL_GENES=True)")
        if miss:
            dropped[k] = miss
    print(f"{name}: {len(kept)}/{len(panel)} panels usable, "
          f"{sum(len(v) for v in kept.values())} genes")
    for k, v in dropped.items():
        print(f"  dropped from {k}: {v}")
    return kept


def save_umap(a, colors, fname, size=2, legend_loc="right margin", legend_fontsize=None,
              figsize=None, **kw):
    """One figure per colour — never a grid. Saves to FIG/<fname stem>_<colour>.png.

    Every categorical gets a legend. Rather than dropping the legend on a high-cardinality column
    (131 donor-units, 89 samples), the font is scaled down and the canvas widened so it still fits;
    booleans are promoted to a two-level categorical so they get a legend instead of a colourbar.
    Titles are the column name alone — the subsample size belongs in the caption, not on the plot.
    """
    stem = Path(fname).stem
    written = []
    for c in [c for c in colors if c in a.obs or c in a.var_names]:
        s = a.obs[c] if c in a.obs else None
        col, cat, ncat = c, False, 0
        if s is not None and s.dtype == bool:          # -> legend, not a 0/1 colourbar
            col = f"_pl_{c}"
            a.obs[col] = pd.Categorical(np.where(s.to_numpy(), "True", "False"),
                                        categories=["False", "True"])
            cat, ncat = True, 2
        elif s is not None and isinstance(s.dtype, pd.CategoricalDtype):
            cat, ncat = True, int(s.cat.categories.size)
        fs = legend_fontsize or (9 if ncat <= 15 else 7 if ncat <= 40 else 5 if ncat <= 90 else 4)
        fw = figsize or (6 + min(6.0, 0.8 + 0.05 * ncat), 6)
        with mpl.rc_context({"figure.figsize": fw}):
            fig = sc.pl.umap(a, color=col, frameon=False, size=size, show=False, return_fig=True,
                             legend_loc=(legend_loc if cat else None),
                             legend_fontsize=fs, title=c, **kw)
        # the extra width is for the legend, not the scatter: without this the UMAP is stretched
        # into a flat rectangle. savefig.bbox="tight" (set in §0) trims the leftover whitespace.
        fig.axes[0].set_box_aspect(1)
        for ax in fig.axes:
            for coll in ax.collections:
                coll.set_rasterized(True)
        out = FIG / f"{stem}_{c}.png"
        fig.savefig(out, dpi=200)
        plt.show(); plt.close(fig)
        if col != c:
            del a.obs[col]
        written.append(out)
        print(f"saved {out.name}  ({ncat} categories)" if cat else f"saved {out.name}  (continuous)")
    return written


def save_umap_grid(a, colors, fname, ncols=4, size=3, **kw):
    """The one case a grid beats separate files: many marker genes, compared side by side."""
    colors = [c for c in colors if c in a.obs or c in a.var_names]
    fig = sc.pl.umap(a, color=colors, ncols=ncols, frameon=False, wspace=0.25, size=size,
                     show=False, return_fig=True, **kw)
    for ax in fig.axes:
        for coll in ax.collections:
            coll.set_rasterized(True)
    fig.savefig(FIG / fname, dpi=200)
    plt.show(); plt.close(fig); print("saved", FIG / fname)


def save_dotplot(a, panel, groupby, fname, **kw):
    dp = sc.pl.dotplot(a, panel, groupby=groupby, standard_scale="var", return_fig=True, **kw)
    dp.savefig(FIG / fname, dpi=200)
    dp.show()
    print("saved", FIG / fname)


print("helpers ready")
```

```
helpers ready
```

## 1 — Sample scope: which blood samples go in

Login-node safe: reads only the `obs` group of the blood h5ad (no `X`, no layers) and prints a
roster of all **89** v2 blood samples before anything heavy runs.

Self-contained on purpose. v1 read `12_atlas_descriptive`'s `atlas_obs_full.parquet`, which made `31_reannotation` depend on `12_atlas_descriptive`
having been run first — and `12_atlas_descriptive` in turn gates on `31_reannotation`'s output. The two harmonized columns it
needed (`entity_h`, `blood_involvement_eff`) are cheap to derive here, so the loop is cut.

**`NO_VDJ_SAMPLES` is now derived, not hardcoded.** v1 pinned a 9-name list; v2 has 11 zero-VDJ
samples (170,945 cells) because four cohorts are new. For these samples `is_malignant` /
`clone_id` are structurally absent, so `pct_mal == 0` is **absence of evidence, not evidence of
absence** — every malignancy table below therefore reports a VDJ-available denominator next to
the raw rate. They are not excluded by default.

```python  # cell 4
# ============================================================================
# §1  Sample scope — decided here, printed here, never hidden
# ============================================================================
from anndata.io import read_elem

SAMPLE_SCOPE = "all"        # "all" (89 samples / 811,024 cells) | "vdj_only" | "curated"

QC_COLS = ["cell_id", "study", "dataset", "sample_id", "donor", "real_donor", "patient_key",
           "disease", "entity", "disease_stage", "stage_clean", "tech", "treatment_context",
           "blood_involvement", "cohort_country", "sex", "lineage", "compartment",
           "n_genes", "total_counts", "pct_mito", "doublet_score",
           "has_tcr", "is_malignant", "is_dominant_clone", "clone_id", "clone_size", "is_expanded"]

with h5py.File(SRC_H5AD, "r") as h:
    OBS = read_elem(h["obs"])
OBS = OBS[[c for c in QC_COLS if c in OBS.columns]].copy()
BL = OBS[OBS["compartment"].astype(str).isin(["Blood"] + (["LN"] if INCLUDE_LN else []))].copy()
assert len(BL) == N_BLOOD + (N_LN if INCLUDE_LN else 0), \
    f"{SRC_H5AD.name} gives {len(BL):,} blood cells, expected v2's {N_BLOOD:,} — this is the v1 build"
_missing = V2_BLOOD_DATASETS - set(BL["dataset"].astype(str).unique())
assert not _missing, f"v2 blood cohorts absent — this is the v1 build. missing: {sorted(_missing)}"

# --- the two harmonized columns, same maps as nb32 sec 1 (v2 vocabularies double-count raw) ---
ENTITY_H = {"MF": "MF_classic", "SS": "Sezary", "healthy": "healthy_control"}
LEUKEMIC_ENT = {"Sezary", "MF/SS_leukemic", "erythrodermic_CTCL(eMF/SS)"}
BL["entity_h"] = BL["entity"].astype(str).replace(ENTITY_H)
_stg, _cur = BL["stage_clean"].astype(str), BL["blood_involvement"].astype(str)
BL["blood_involvement_eff"] = np.select(
    [_cur.isin(["yes", "B2_yes_by_def"]), _cur.eq("no"), BL["entity_h"].isin(LEUKEMIC_ENT),
     _stg.isin(["IV", "IVA", "IVA1"]), _stg.eq("IIIB"), _stg.isin(["IVA2", "IVB"])],
    ["yes(curated)", "no(HC)", "yes(leukemic entity)", "yes(stage IV/B2)",
     "likely(stage IIIB/B1)", "stage IV non-blood (IVA2/IVB)"], default="unknown")

for src, dst in [("has_tcr", "_tcr"), ("is_malignant", "_mal"),
                 ("is_dominant_clone", "_dom"), ("is_expanded", "_exp")]:
    BL[dst] = _bool(BL[src])

roster = (BL.groupby("sample_id", observed=True)
   .agg(study=("study", "first"),
        n_donors=("donor", "nunique"), n_patients=("patient_key", "nunique"),
        real_donor=("real_donor", lambda s: ", ".join(natsorted(set(s.astype(str)))[:3])),
        disease=("disease", "first"), entity=("entity_h", "first"), stage=("stage_clean", "first"),
        tech=("tech", "first"), treatment=("treatment_context", "first"),
        blood_inv=("blood_involvement_eff", "first"), country=("cohort_country", "first"),
        n_cells=("cell_id", "size"),
        pct_tcr=("_tcr", lambda s: round(100 * s.mean(), 1)),
        pct_mal=("_mal", lambda s: round(100 * s.mean(), 1)),
        pct_dom=("_dom", lambda s: round(100 * s.mean(), 1)),
        n_clones=("clone_id", lambda s: s[s.astype(str).ne("")].nunique()),
        med_genes=("n_genes", lambda s: int(np.median(s))),
        med_counts=("total_counts", lambda s: int(np.median(s))),
        med_mito=("pct_mito", lambda s: round(float(np.median(s)), 2)),
        max_dbl=("doublet_score", lambda s: round(float(np.max(s)), 3)))
   .sort_values(["study", "n_cells"], ascending=[True, False]))

# DERIVED, not hardcoded: v1 pinned a 9-name list and v2 has 11 such samples.
NO_VDJ_SAMPLES = sorted(roster.index[roster["pct_tcr"] == 0].astype(str))
PROVISIONAL_META = list(NO_VDJ_SAMPLES)

_SCOPES = {
    "all":      [],                       # <- DEFAULT
    "vdj_only": NO_VDJ_SAMPLES,
    "curated":  NO_VDJ_SAMPLES,           # add named low-complexity libraries here if needed
}
EXCLUDE_SAMPLES = set(_SCOPES[SAMPLE_SCOPE])
roster["NO_VDJ"] = roster.index.isin(NO_VDJ_SAMPLES)
roster["PROVISIONAL"] = roster.index.isin(PROVISIONAL_META)
roster["EXCLUDED"] = roster.index.isin(EXCLUDE_SAMPLES)
roster.to_csv(SCOPE_CSV)

print(f"SAMPLE_SCOPE={SAMPLE_SCOPE!r}  INCLUDE_LN={INCLUDE_LN}  -> {SCOPE_CSV.name}")
print(f"v2 blood confirmed: {len(BL):,} cells | {roster.shape[0]} samples | "
      f"{BL['study'].nunique()} studies | {BL['donor'].nunique()} donor-units | "
      f"{BL['patient_key'].nunique()} patients\n")
assert roster.shape[0] == V2_N_SAMPLES_BLOOD and BL["study"].nunique() == V2_N_STUDIES_BLOOD
print(roster.to_string())

kept = roster[~roster["EXCLUDED"]]
kept_cells = BL[~BL["sample_id"].astype(str).isin(EXCLUDE_SAMPLES)]
print(f"\nKEPT    : {len(kept)} samples | {kept['n_cells'].sum():,} cells | "
      f"{kept_cells['donor'].nunique()} donor-units | {kept_cells['patient_key'].nunique()} "
      f"patients | {kept_cells['study'].nunique()} studies")
print(f"DROPPED : {int(roster['EXCLUDED'].sum())} samples | "
      f"{int(roster.loc[roster['EXCLUDED'], 'n_cells'].sum()):,} cells")
_nv = int(kept.loc[kept["NO_VDJ"], "n_cells"].sum())
print(f"\nzero-VDJ samples ({len(NO_VDJ_SAMPLES)}, derived): {NO_VDJ_SAMPLES}")
print(f"  {_nv:,} cells ({100 * _nv / max(kept['n_cells'].sum(), 1):.1f}% of the kept scope) -> "
      "for these, pct_mal == 0 is ABSENCE OF EVIDENCE. Every malignancy table below reports a "
      "VDJ-available denominator alongside the raw rate.")

print(f"\ndisease: {kept_cells['disease'].value_counts().to_dict()}")
print(f"entity : {kept_cells['entity_h'].value_counts().to_dict()}")
_hc = kept_cells[kept_cells["disease"].astype(str) == "HC"]
print(f"\nHEALTHY REFERENCE: {len(_hc):,} HC blood cells / {_hc['real_donor'].nunique()} donors "
      f"{sorted(_hc['real_donor'].astype(str).unique())} — v1 had one 4,481-cell sample. "
      "sec 14b checks the annotation against these.")

_dps = kept_cells.groupby("sample_id", observed=True)["donor"].nunique()
print(f"\nsamples pooling >1 donor (HTO lanes): {_dps[_dps > 1].to_dict()}")
print("  -> `sample_id` is a LANE for these; donor-level statistics group by `donor`.")
_dis = kept_cells[kept_cells["disease"].astype(str) != "HC"]
print(f"\nlineage among diseased blood cells: {_dis['lineage'].value_counts().to_dict()}")
print("  -> a SAMPLE-level clinical annotation, constant across cells. It carries zero per-cell "
      "information and is never used to validate a CD4/CD8 cluster call.")
del OBS, BL, _dis, _hc, kept_cells; gc.collect()
```

```
SAMPLE_SCOPE='all'  INCLUDE_LN=False  -> blood_scope_roster.csv
v2 blood confirmed: 811,024 cells | 89 samples | 10 studies | 131 donor-units | 75 patients

                                  study  n_donors  n_patients                       real_donor     disease           entity stage    tech                        treatment                      blood_inv  country  n_cells  pct_tcr  pct_mal  pct_dom  n_clones  med_genes  med_counts  med_mito  max_dbl  NO_VDJ  PROVISIONAL  EXCLUDED
sample_id                                                                                                                                                                                                                                                                                                                                
B1__P5_a                borcherding2019         1           1                           B1__P5          SS           Sezary    NA     10x                 none/unspecified                   yes(curated)      USA     4328      0.0      0.0      0.0         0       1476        5058      3.85    0.234    True         True     False
B1__P5_b                borcherding2019         1           1                           B1__P5          SS           Sezary    NA     10x                 none/unspecified                   yes(curated)      USA     3612      0.0      0.0      0.0         0       1289        4714      4.41    0.242    True         True     False
B1__P4_a                borcherding2019         1           1                           B1__P4          SS           Sezary    NA     10x                 none/unspecified                   yes(curated)      USA     3292      0.0      0.0      0.0         0       2158        7255      2.60    0.236    True         True     False
B1__P4_b                borcherding2019         1           1                           B1__P4          SS           Sezary    NA     10x                 none/unspecified                   yes(curated)      USA     3021      0.0      0.0      0.0         0       2444        8542      3.50    0.229    True         True     False
B2__CTCL2               borcherding2023         1           1                        B2__CTCL2          SS           Sezary    NA  10x_5p  photopheresis/vorinostat_course                   yes(curated)      USA    33743     94.3     90.4     90.4       922        855        1929      5.05    0.244   False        False     False
B2__CTCL6               borcherding2023         1           1                        B2__CTCL6          SS           Sezary    NA  10x_5p  photopheresis/vorinostat_course                   yes(curated)      USA     3448     40.8     15.1     15.1       815       1578        4320      4.20    0.242   False        False     False
B2__CTCL3               borcherding2023         1           1                        B2__CTCL3          SS           Sezary    NA  10x_5p  photopheresis/vorinostat_course                   yes(curated)      USA     3034     59.6     12.2     12.2       975       1993        5255      4.09    0.229   False        False     False
B2__CTCL5               borcherding2023         1           1                        B2__CTCL5          SS           Sezary    NA  10x_5p  photopheresis/vorinostat_course                   yes(curated)      USA     1572     50.4     31.4     31.4       233       1008        2590      3.98    0.197   False        False     False
B2__CTCL4               borcherding2023         1           1                        B2__CTCL4          SS           Sezary    NA  10x_5p  photopheresis/vorinostat_course                   yes(curated)      USA      845     26.9     13.1     13.1       107       1106        3800      4.14    0.221   False        False     False
B5__PT23_Blood                 buus2025         1           1                         B5__PT23          SS           Sezary    NA  10x_5p                 ex_vivo_baseline                   yes(curated)  Denmark   
... [27587 more chars truncated]
```

## 2 — Load the blood subset + MrVI latents  ⚠️ HEAVY

811,024 × 10,000 HVG as float32 is ~6 GB plus the `counts` layer §3 makes. The latent's row space
is reconciled against the source's **by row count**, so all four (`LATENT_SCOPE` × `FULL_GENES`)
combinations work rather than only the two matched ones — and a stale `.npy` that matches neither
raises instead of being silently mis-aligned.

```python  # cell 6
# ============================================================================
# §2  Header read -> blood mask -> latent slice -> full read -> attach   (HEAVY)
# ============================================================================
with h5py.File(SRC_H5AD, "r") as h:
    _c = h["obs"]["compartment"]
    _cats = [c.decode() if isinstance(c, bytes) else str(c) for c in _c["categories"][:]]
    _codes = _c["codes"][:]
    _cids = h["obs"]["cell_id"].asstr()[:]
N_SRC = int(_codes.shape[0])
WANT_COMPARTMENT = ["Blood", "LN"] if INCLUDE_LN else ["Blood"]
_want_codes = [_cats.index(w) for w in WANT_COMPARTMENT if w in _cats]
assert _want_codes, f"none of {WANT_COMPARTMENT} in compartment categories {_cats}"
blood_mask = np.isin(_codes, _want_codes)
N_KEEP = int(blood_mask.sum())
print(f"{SRC_H5AD.name}: {N_SRC:,} rows -> {N_KEEP:,} in {WANT_COMPARTMENT}")

U_ALL = np.load(MRVI_U, mmap_mode="r")
Z_ALL = np.load(MRVI_Z, mmap_mode="r")
print(f"latents: u {U_ALL.shape} {U_ALL.dtype} | z {Z_ALL.shape} {Z_ALL.dtype}")

N_LAT = int(U_ALL.shape[0])
assert Z_ALL.shape[0] == N_LAT, f"u ({N_LAT:,}) and z ({Z_ALL.shape[0]:,}) latents disagree"

# Reconcile the latent's row space against the source's by ROW COUNT.
if N_LAT == N_KEEP:
    # latent already restricted to this scope -> mask the source only, slice nothing.
    # np.asarray() on an mmap_mode="r" array returns the READ-ONLY memmap itself, and
    # pynndescent's numba kernels only accept writable arrays -> force a real copy.
    U_BLOOD = np.array(U_ALL, dtype=np.float32)
    Z_BLOOD = np.array(Z_ALL, dtype=np.float32)
    print(f"latent is pre-sliced to the scope ({N_LAT:,} rows) — no latent slicing")
elif N_LAT == N_SRC:
    assert N_SRC == N_TOTAL, f"source h5ad has {N_SRC:,} rows, expected v2's {N_TOTAL:,}"
    assert N_KEEP == N_BLOOD + (N_LN if INCLUDE_LN else 0), \
        f"mask selects {N_KEEP:,}, expected {N_BLOOD + (N_LN if INCLUDE_LN else 0):,}"
    U_BLOOD = np.array(U_ALL[blood_mask], dtype=np.float32)
    Z_BLOOD = np.array(Z_ALL[blood_mask], dtype=np.float32)
    print(f"sliced the full-atlas latent {N_LAT:,} -> {N_KEEP:,} rows")
else:
    raise AssertionError(
        f"latent rows ({N_LAT:,}) match neither the in-scope cells ({N_KEEP:,}) nor the source "
        f"rows ({N_SRC:,}) -> STALE or mismatched npy; refusing to guess an alignment.\n"
        f"  latent: {MRVI_U.name}\n  source: {SRC_H5AD.name}\n"
        "(the v1 latents were moved to data/atlas_joint/_stale_v1_latents/.)\n"
        "Rebuild: JOB_TAG=blood QUEUE=short-gpu MEM_MB=120000 WALL_H=6 ./run_mrvi_joint.sh "
        "--subset-compartment Blood --tag blood --skip-diag --force")

# When the gene space came from the full atlas but the latent is blood-native, the alignment rests
# on the two files ordering blood rows identically. Prove it on cell_id rather than assume it.
if FULL_GENES and N_LAT == N_KEEP and BLOOD_HVG_H5AD.exists():
    with h5py.File(BLOOD_HVG_H5AD, "r") as h:
        _bcids = h["obs"]["cell_id"].asstr()[:]
    assert len(_bcids) == N_KEEP and (_bcids == _cids[blood_mask]).all(), (
        "blood row ORDER differs between joint_annotated.h5ad and joint_mrvi_input_blood.h5ad — "
        "the blood-native latent cannot be used with FULL_GENES=True")
    print("verified: full-atlas blood rows are ordered identically to the blood HVG file")

assert U_BLOOD.shape == (N_KEEP, 10), U_BLOOD.shape
assert Z_BLOOD.shape == (N_KEEP, 30), Z_BLOOD.shape
assert np.isfinite(U_BLOOD).all() and np.isfinite(Z_BLOOD).all(), "NaN/inf in the MrVI latent"
del U_ALL, Z_ALL; gc.collect()

print(f"\nreading {SRC_H5AD.name} ...")
_adata = sc.read_h5ad(SRC_H5AD)
assert _adata.n_obs == N_SRC, "h5ad changed between the header read and the full read"
assert (_adata.obs["cell_id"].astype(str).to_numpy() == _cids).all(), \
    "obs row order differs between the header read and the full read — the mask is invalid"

ad = _adata[blood_mask].copy() if N_KEEP != N_SRC else _adata
del _adata; gc.collect()
ad.obsm["X_mrvi_u"] = U_BLOOD
ad.obsm["X_mrvi_z"] = Z_BLOOD
coerce_latents(ad)
assert ad.n_obs == ad.obsm["X_mrvi_u"].shape[0]
assert (ad.obs["cell_id"].astype(str).to_numpy() == _cids[blood_mask]).all()
assert ad.obs["cell_id"].is_unique, "duplicate cell_id — every merge below keys on cell_id"

if EXCLUDE_SAMPLES:
    _keep = ~ad.obs["sample_id"].astype(str).isin(EXCLUDE_SAMPLES).to_numpy()
    print(f"scope filter: {ad.n_obs:,} -> {int(_keep.sum()):,} cells")
    ad = ad[_keep].copy()
for _c in ("study", "sample_id", "donor", "real_donor", "patient_key", "dataset", "disease",
           "entity", "tech", "treatment_context", "compartment"):
    if str(ad.obs[_c].dtype) == "category":
        ad.obs[_c] = ad.obs[_c].cat.remove_unused_categories()

print(ad)
print("\ncells per dataset:", ad.obs["dataset"].value_counts().to_dict())
print("cells per study  :", ad.obs["study"].value_counts().to_dict())
add_tcr_flags(ad)

# ---------------------------------------------------------------- gene-space check
_absent = [g for g in REQUIRED_MARKERS if g not in set(ad.var_names)]
print(f"\ngene space: {ad.n_vars:,} genes | required markers present "
      f"{len(REQUIRED_MARKERS) - len(_absent)}/{len(REQUIRED_MARKERS)}")
if _absent:
    print(f"  !! ABSENT: {_absent}")
    print("  HVG selection is data-driven, so this is expected rather than an error. Known case:")
    print("  CD4 is NOT in the v2 blood-native 10k HVG set — blood is dominated by CD4 malignancy,")
    print("  so CD4 has little variance across cells and misses the top 10k. The CD4 call")
    print("  therefore rests on IL7R + CD40LG plus CD8A/CD8B-negativity, which is standard")
    print("  practice anyway (CD4 mRNA is poorly captured in 10x data even when it IS present).")
    print("  Set FULL_GENES=True to annotate in the 42,347-gene space instead — costs a much")
    print("  larger read and T-object write, and does NOT change the latent or the clustering.")
```

```
joint_mrvi_input_blood.h5ad: 811,024 rows -> 811,024 in ['Blood']
latents: u (811024, 10) float32 | z (811024, 30) float32
latent is pre-sliced to the scope (811,024 rows) — no latent slicing

reading joint_mrvi_input_blood.h5ad ...
AnnData object with n_obs × n_vars = 811024 × 10000
    obs: 'cell_id', 'dataset', 'study', 'donor', 'real_donor', 'sample_id', 'sex', 'disease', 'disease_stage', 'compartment', 'tissue', 'tech', 'cell_type_orig', 'cell_type', 'n_genes', 'total_counts', 'pct_mito', 'doublet_score', 'predicted_doublet', 'patient_key', 'has_tcr', 'clone_id', 'tra_cdr3', 'trb_cdr3', 'clone_size', 'is_expanded', 'is_dominant_clone', 'is_malignant', 'repo', 'accession', 'organ', 'tissue_detail', 'malignant_call_method', 'tcr_available', 'malignant_labeled', 'malignant_frac', 'tcr_recovery', 'stage_class', 'stage_clean', 'lineage', 'entity', 'fmf_status', 'lineage_resolve', 'treatment_context', 'blood_involvement', 'lesion_type', 'cohort_country', 'cohort_region', 'origin_resolve', 'east_asian', 'lineage_orig', 'lineage_score_cd4', 'lineage_score_cd8', 'lineage_score_gd', 'lineage_score_ab', 'lineage_method', 'lineage_disagree', 'lineage_disagreement'
    var: 'highly_variable', 'highly_variable_rank', 'means', 'variances', 'variances_norm', 'highly_variable_nbatches'
    uns: 'adt_panel', 'hvg'
    obsm: 'X_adt', 'X_scVI', 'X_scVI_MDE', '_scvi_extra_categorical_covs', 'X_mrvi_u', 'X_mrvi_z'
    layers: 'raw_counts'

cells per dataset: {'geskin26': 260455, 'ren23': 140165, 'harro23': 130829, 'dorando26': 85847, 'buus25': 72378, 'borcherding23': 42642, 'gaydosik22': 31141, 'herrera21': 30132, 'borcherding19': 14253, 'rindler21': 3182}
cells per study  : {'geskin2026': 260455, 'ren2023': 140165, 'harro2023': 130829, 'dorando2026': 85847, 'buus2025': 72378, 'borcherding2023': 42642, 'gaydosik2022': 31141, 'herrera2021': 30132, 'borcherding2019': 14253, 'rindler2021_fi': 3182}
TCR+ 476,653 (58.8%) | dominant-clone malignant 288,721 (35.6%) | in a VDJ-bearing sample 640,079 (78.9%)

gene space: 10,000 genes | required markers present 19/19
```

## 3 — Log-normalize for marker viz (raw counts kept in `layers["counts"]`)

10b normalizes in place on the CSR data array and keeps **no** counts layer, because §13 re-streams
raw counts from the atlas for the T cells. Here the counts layer is kept: §16 writes
`blood_T_annotated.h5ad` straight from this object rather than re-reading the 112 GB atlas, and
`32_malignancy_tcr_cnv`'s CNV arm needs raw counts in it.

```python  # cell 8
# ============================================================================
# §3  Log-normalize
# ============================================================================
_chk = ad.X[:2000]
assert float(np.abs(_chk.data - np.round(_chk.data)).max()) == 0.0, \
    "X is not raw counts — normalize_total would DOUBLE-normalize"
assert "counts" not in ad.layers, "layers['counts'] already exists — X may already be normalized"
del _chk

ad.layers["counts"] = ad.X.copy()
sc.pp.normalize_total(ad, target_sum=1e4)
sc.pp.log1p(ad)
_mx = float(ad.X.max())
assert _mx < 20, f"X.max()={_mx} — does not look log1p-normalized"
print(f"X is log1p-normalized (max {_mx:.2f}); raw counts preserved in layers['counts']")
```

```
X is log1p-normalized (max 9.15); raw counts preserved in layers['counts']
```

## 4 — Subsampled UMAP views (plotting only)

10b §3, unchanged. At 811k cells a full-object `sc.tl.umap` is hours of single-threaded CPU and
draws an unreadable blob; `plot_view()` embeds a uniform random **300,000-cell** subsample (37 % of
blood) instead. **Leiden and every label below still run on all cells** — the view is only ever
passed to `sc.pl.*`.

Cache stem: `blood_umap_<key>_v2_<parent_n>_<sub_n>.npy`. The parent row count is in the name, so
a re-annotation that changes the subset cannot silently reuse an old embedding — and because
`PLOT_N` is part of that name, the 150k embeddings from an earlier run stay on disk untouched.

Every UMAP below is written as **its own figure**, one file per colour, named
`<stem>_<colour>.png`, with a right-margin legend. Titles carry the column name only.

```python  # cell 10
# ============================================================================
# §4  Broad plotting view + the full-object kNN graph   (HEAVY)
# ============================================================================
# Order matters: build the view BEFORE the graph, so the 300k copy does not have to slice an
# 811k x 811k obsp it is about to throw away and recompute for itself.
adv = plot_view(ad, "broad")
print("plotting view:", adv.shape)

# The full-object kNN graph is what Leiden clusters on (§5) — it is NOT used for a UMAP, and it is
# only needed when the clustering has to be computed. On a pass-2 re-run leiden_broad is reloaded
# from CSV, and building this graph on 811k cells would be ~10 min of pure waste.
if not LEIDEN_CSV.exists():
    sc.pp.neighbors(ad, use_rep="X_mrvi_u", random_state=SEED)
    print("full-object neighbors built on X_mrvi_u:", f"{ad.n_obs:,}", "cells")
else:
    print(f"{LEIDEN_CSV.name} exists -> skipping the full-object kNN graph "
          "(delete the CSV to recluster)")
```

```
plot_view[broad]: 150,000/811,024 cells, computed UMAP -> blood_umap_broad_v2_811024_150000.npy
plotting view: (150000, 10000)
full-object neighbors built on X_mrvi_u: 811,024 cells
```

## 5 — Broad Leiden (frozen to CSV)  ⚠️ HEAVY

`res=1.0`, above 10b's `0.7`: pDC / cDC1 / HSPC are 0.1–2 % of PBMC and a rare-lineage merge at
the broad pass is unrecoverable downstream. The sweep below prints the cluster count at four
resolutions on the already-built graph (cheap) so the choice is visible before it is frozen.

Once `blood_leiden_broad_v2.csv` exists it is **reloaded**, never recomputed — the hand map in §8
is authored against these exact cluster IDs. Delete the CSV to recluster, and then re-author the
map.

```python  # cell 12
# ============================================================================
# §5  Broad Leiden (frozen)   (HEAVY)
# ============================================================================
if not LEIDEN_CSV.exists():
    # cheap: the kNN graph is already built, so each extra resolution is just the partition
    for _r in (0.5, 0.7, 1.0, 1.2):
        sc.tl.leiden(ad, resolution=_r, random_state=SEED, key_added="_sweep",
                     flavor="igraph", n_iterations=2, directed=False)
        print(f"  res={_r}: {ad.obs['_sweep'].nunique()} clusters")
    del ad.obs["_sweep"]

frozen_leiden(ad, LEIDEN_CSV, "leiden_broad", LEIDEN_RES)
print(ad.obs["leiden_broad"].value_counts().reindex(
    natsorted(ad.obs["leiden_broad"].cat.categories)).to_string())
sync_view(adv, ad, "leiden_broad")
```

```
res=0.5: 15 clusters
  res=0.7: 19 clusters
  res=1.0: 24 clusters
  res=1.2: 28 clusters
computed leiden_broad (res=1.0): 24 clusters -> froze to blood_leiden_broad_v2.csv
leiden_broad
0       5073
1      49277
2       4430
3      20026
4      24354
5      26413
6       7347
7       8976
8      19095
9      84848
10    104723
11      4434
12       626
13      3023
14     17112
15    121131
16     59442
17    115064
18    109031
19        58
20      4704
21      2584
22     18504
23       749

AnnData object with n_obs × n_vars = 150000 × 10000
    obs: 'cell_id', 'dataset', 'study', 'donor', 'real_donor', 'sample_id', 'sex', 'disease', 'disease_stage', 'compartment', 'tissue', 'tech', 'cell_type_orig', 'cell_type', 'n_genes', 'total_counts', 'pct_mito', 'doublet_score', 'predicted_doublet', 'patient_key', 'has_tcr', 'clone_id', 'tra_cdr3', 'trb_cdr3', 'clone_size', 'is_expanded', 'is_dominant_clone', 'is_malignant', 'repo', 'accession', 'organ', 'tissue_detail', 'malignant_call_method', 'tcr_available', 'malignant_labeled', 'malignant_frac', 'tcr_recovery', 'stage_class', 'stage_clean', 'lineage', 'entity', 'fmf_status', 'lineage_resolve', 'treatment_context', 'blood_involvement', 'lesion_type', 'cohort_country', 'cohort_region', 'origin_resolve', 'east_asian', 'lineage_orig', 'lineage_score_cd4', 'lineage_score_cd8', 'lineage_score_gd', 'lineage_score_ab', 'lineage_method', 'lineage_disagree', 'lineage_disagreement', '_tcr', '_mal', '_dom', '_exp', '_vdj_ok', 'leiden_broad'
    var: 'highly_variable', 'highly_variable_rank', 'means', 'variances', 'variances_norm', 'highly_variable_nbatches'
    uns: 'adt_panel', 'hvg', 'log1p', 'neighbors', 'umap', 'plot_idx'
    obsm: 'X_adt', 'X_scVI', 'X_scVI_MDE', '_scvi_extra_categorical_covs', 'X_mrvi_u', 'X_mrvi_z', 'X_umap'
    layers: 'raw_counts', 'counts'
    obsp: 'distances', 'connectivities'
```

## 6 — UMAP overview

```python  # cell 14
# ============================================================================
# §6  UMAP overview  (on the subsampled view)
# ============================================================================
print(f"plotting {adv.n_obs:,} of {ad.n_obs:,} cells (uniform subsample)")
# leiden_broad is what §8 is annotated from; study / stage / disease are the metadata axes.
# `compartment` is constant ("Blood") here, so it is nb32's panel, not one of these.
sync_view(adv, ad, "study", "stage_clean", "disease")
save_umap(adv, ["leiden_broad", "study", "stage_clean", "disease"], "umap_v2.png")
```

```
<Figure size 1760x1056 with 4 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/blood_reannotation/umap_overview_v2.png

<Figure size 1760x1056 with 6 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/blood_reannotation/umap_qc_tcr_v2.png

<Figure size 1760x1056 with 4 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/blood_reannotation/umap_batch_v2.png
```

## 7 — Broad marker dot plot  ⚠️ HEAVY

**10b's `markers` dict is pasted verbatim as `MARKERS_10B` and never edited** — every lineage the
two compartments share is called on exactly the same genes, so a CD4 cluster here means what a CD4
cluster means in skin. Blood then makes three changes on top:

1. `CD8/NK` is **split** into `CD8` and `NK`. In skin, NK cells are rare enough that 10b folds them
   into the CD8 panel; in PBMC they are 5–15 % and must be separable, or every NK cell lands in CD8.
2. `Myeloid/DC` is **kept** and additionally resolved into `Mono CD14` / `Mono CD16` / `cDC1` /
   `cDC2` / `pDC`. If a cluster cannot be resolved past the shared panel, label it `Myeloid` — that
   is a valid 10b-compatible answer.
3. The four skin-stromal panels (`Keratinocyte`, `Fibroblast`, `Vascular endo`, `Lymphatic endo`,
   `Melanocyte`) are kept **verbatim, as a contamination tripwire**. They are not callable
   lineages in blood: signal there means ambient or a doublet, not a new cell type.

Added with no skin counterpart: MAIT/NKT, γδ, B-naive vs B-memory, platelet, erythroid, HSPC, an
IFN-response axis, and the ambient/stress warning panels.

`CD4` is absent from the v2 blood 10k HVG set (see §2), so the CD4 row rests on `IL7R` + `CD40LG`.

```python  # cell 16
# ============================================================================
# §7  Broad marker dot plot   (HEAVY)
# ============================================================================
# --- 10b's panel, verbatim. Do not edit: shared lineages must be called on identical genes. -----
MARKERS_10B = {
    "T":              ["CD3D", "CD3E", "TRAC"],
    "CD4":            ["CD4", "IL7R"],
    "CD8/NK":         ["CD8A", "GZMB", "NKG7", "GNLY"],
    "Treg":           ["FOXP3", "CTLA4"],
    "B":              ["MS4A1", "CD79A", "CD19"],
    "Plasma":         ["MZB1", "JCHAIN", "IGHG1"],
    "Myeloid/DC":     ["LYZ", "CD68", "ITGAX", "CD14"],
    "Mast":           ["TPSAB1", "CPA3", "KIT"],
    "Keratinocyte":   ["KRT14", "KRT5", "KRT1"],
    "Fibroblast":     ["COL1A1", "DCN", "LUM"],
    "Vascular endo":  ["PECAM1", "VWF", "CLDN5"],
    "Lymphatic endo": ["LYVE1", "PROX1"],
    "Melanocyte":     ["MLANA", "PMEL", "TYR"],
}
SKIN_ONLY = ["Keratinocyte", "Fibroblast", "Vascular endo", "Lymphatic endo", "Melanocyte"]

markers = {
    # ---- 10b keys, 10b genes (CD40LG appended to CD4: CD4 itself is not in the blood HVG set)
    "T core":      MARKERS_10B["T"] + ["TRBC2", "IL32", "CD247"],
    "CD4":         MARKERS_10B["CD4"] + ["CD40LG"],
    "CD8":         ["CD8A", "CD8B", "GZMK"],
    "Treg":        MARKERS_10B["Treg"] + ["IL2RA", "IKZF2", "TNFRSF18"],
    "B naive":     MARKERS_10B["B"] + ["CD79B", "IGHD", "TCL1A"],
    "B memory":    ["CD27", "IGHM", "CR2"],
    "Plasma":      MARKERS_10B["Plasma"] + ["XBP1", "DERL3", "TNFRSF17"],
    "Myeloid/DC":  MARKERS_10B["Myeloid/DC"] + ["HLA-DRA", "CD74", "CST3"],
    "Mast/Baso":   MARKERS_10B["Mast"] + ["MS4A2", "HDC", "GATA2"],
    # ---- innate lymphoid: 10b folds NK into CD8/NK; PBMC needs them separable
    "NK":          ["NCAM1", "KLRD1", "KLRF1", "GNLY", "NKG7", "PRF1", "FCGR3A", "TYROBP"],
    "MAIT/NKT":    ["SLC4A10", "KLRB1", "TRAV1-2", "ZBTB16", "NCR3"],
    "gdT":         ["TRDC", "TRGC1", "TRGC2", "TRDV1", "TRDV2"],
    # ---- the myeloid split 10b's single Myeloid/DC panel cannot make
    "Mono CD14":   ["CD14", "LYZ", "S100A8", "S100A9", "VCAN", "FCN1"],
    "Mono CD16":   ["FCGR3A", "MS4A7", "CDKN1C", "LST1", "AIF1"],
    "cDC1":        ["CLEC9A", "XCR1", "IDO1"],
    "cDC2":        ["CD1C", "FCER1A", "CLEC10A"],
    "pDC":         ["LILRA4", "CLEC4C", "IL3RA", "SERPINF1", "TCF4", "IRF7", "PLD4"],
    # ---- non-lymphoid PBMC constituents that always appear and are always mistaken
    "Platelet":    ["PPBP", "PF4", "ITGA2B", "GP9", "TUBB1", "NRGN"],
    "Erythroid":   ["HBB", "HBA1", "HBA2", "ALAS2", "AHSP", "GYPA"],
    "HSPC":        ["CD34", "SPINK2", "PRSS57", "MPO"],
    # ---- state axes, not lineages
    "Prolif":      ["MKI67", "TOP2A", "STMN1", "PCLAF", "TYMS"],
    "IFN response": ["ISG15", "IFI6", "MX1", "IFIT3", "OAS1", "LY6E"],
    # ---- WARNINGS: signal here means ambient / LowQC / Doublet, NOT a new lineage
    "Ambient warn": ["HBB", "PPBP", "LYZ", "MALAT1"],
    "Stress warn":  ["HSPA1A", "HSPA1B", "DNAJB1", "JUN", "FOS", "JUNB", "EGR1"],
    # 10b's five skin-stromal panels, verbatim, as a contamination tripwire
    "Non-PBMC warn": [g for k in SKIN_ONLY for g in MARKERS_10B[k]],
}
markers = filter_panel(markers, set(ad.var_names), "broad panel")

sc.tl.dendrogram(ad, groupby="leiden_broad", use_rep="X_mrvi_u")
assert "dendrogram_leiden_broad" in ad.uns, "dendrogram missing — dotplot would trigger a PCA"
save_dotplot(ad, markers, "leiden_broad", "dotplot_broad_v2.png", dendrogram=True, figsize=(24, 8))
```

```
broad panel: 25/25 panels usable, 138 genes
  dropped from CD4: ['CD4']
WARNING: Groups are not reordered because the `groupby` categories and the `var_group_labels` are different.
categories: 0, 1, 2, etc.
var_group_labels: T core, CD4, CD8, etc.

<Figure size 2640x880 with 6 Axes>
<Figure size 2640x880 with 6 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/blood_reannotation/dotplot_broad_v2.png
```

## 8 — Map clusters → `cell_type_broad`  (manual)

Same two-pass workflow and the same `cluster2ct` name as 10b §7. Run once with `{}` to print the
template plus the cluster sizes, read the §7 dot plot and the §7b QC table, then paste both the
map **and** `CLUSTER_SIZES_BROAD` — the size lock is what stops a re-clustered Leiden from being
silently reinterpreted through an old map.

**Allowed labels** (`VOCAB_BROAD`, defined in §0). Shared with 10b, spelled identically:
`T` · `B` · `Plasma` · `Myeloid` · `Mast` · `Unk`. Blood-only: `NK` · `Mono_CD14` · `Mono_CD16` ·
`cDC` · `pDC` · `Platelet` · `Erythroid` · `HSPC` · `Prolif` · `LowQC` · `Doublet`.

Use `Myeloid` when a cluster is clearly myeloid but the CD14/CD16/DC split is not readable — that
is a valid answer, not a failure. Do **not** invent labels: `apply_hand_map` rejects anything
outside the vocabulary, because `12_atlas_descriptive` concatenates this file with the skin one.

```python  # cell 18
# ============================================================================
# §8  leiden_broad -> cell_type_broad   (HAND-FILLED — pass 2)
# ============================================================================
# --- FILL from the §7 dot plot + §7b QC + §9 malignancy tables ---
cluster2ct = {
    "0":"Mono_CD16",
    "1":"Mono_CD14",
    "2":"Platelet",
    "3":"Mono_CD14",
    "4":"cDC",
    "5":"T",
    "6":"Erythroid",
    "7":"Erythroid",
    "8":"T",
    "9":"T",
    "10":"T",
    "11":"T",
    "12":"T",
    "13":"T",
    "14":"T",
    "15":"T",
    "16":"T",
    "17":"T",
    "18":"T",
    "19":"T",
    "20":"Plasma",
    "21":"T",
    "22":"B",
    "23":"cDC",
  
}

# paste the printed sizes here to lock the map to this clustering
CLUSTER_SIZES_BROAD = {'0': 5073, '1': 49277, '2': 4430, '3': 20026, '4': 24354, '5': 26413, '6': 7347, '7': 8976, '8': 19095, '9': 84848, '10': 104723, '11': 4434, '12': 626, '13': 3023, '14': 17112, '15': 121131, '16': 59442, '17': 115064, '18': 109031, '19': 58, '20': 4704, '21': 2584, '22': 18504, '23': 749}



_ok = apply_hand_map(ad, "leiden_broad", cluster2ct, VOCAB_BROAD,
                     CLUSTER_SIZES_BROAD, "cell_type_broad")
```

```
cell_type_broad
T            667584
Mono_CD14     69303
cDC           25103
B             18504
Erythroid     16323
Mono_CD16      5073
Plasma         4704
Platelet       4430
```

## 8b — UMAP with the new broad annotations

```python  # cell 20
# ============================================================================
# §8b  UMAP with the broad annotations
# ============================================================================
if _ok:
    _n = ad.obs["cell_type_broad"].value_counts()
    print(_n.to_string())
    sync_view(adv, ad, "cell_type_broad")
    save_umap(adv, ["cell_type_broad"], "umap_celltype_broad_v2.png")
    # compare against umap_overview_v2_leiden_broad.png from §6 — same view, same subsample, so a
    # mislabelled cluster shows up as a shape that changes colour between the two files
else:
    print("cell_type_broad not filled yet — fill cluster2ct in §8 first")
```

```
cell_type_broad
T            667584
Mono_CD14     69303
cDC           25103
B             18504
Erythroid     16323
Mono_CD16      5073
Plasma         4704
Platelet       4430

<Figure size 704x528 with 1 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/blood_reannotation/umap_celltype_broad_v2.png

<Figure size 1760x528 with 2 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/blood_reannotation/umap_celltype_broad_vs_leiden_v2.png
```

## 9 — Validation: per-cluster TCR / clonality

10b validates its broad map against the Li2024 taxonomy (`cell_type_broad × cell_type` purity).
Blood has no external labels at all — `cell_type == "Unknown"` for 100 % of it — so there is
nothing to cross-tabulate against. This is the replacement: the TCR/clonality calls are the only
atlas-wide per-cell annotation blood carries.

Read `QC_READING` below before interpreting a rate. The trap is that 11 samples (170,945 cells)
have zero V(D)J, so a cluster dominated by them shows `pct_mal ≈ 0` for reasons that have nothing
to do with biology — always read `pct_mal_vdjok` and `pct_mal_of_tcr` beside it.

```python  # cell 22
# ============================================================================
# §9  Per-cluster TCR / clonality (replaces 10b's Li2024 purity table)
# ============================================================================
qc = cluster_qc(ad, "leiden_broad")
if "cell_type_broad" in ad.obs:
    qc["label"] = ad.obs.groupby("leiden_broad", observed=True)["cell_type_broad"].apply(
        lambda s: s.astype(str).value_counts().idxmax()).reindex(qc.index)
qc.to_csv(CLUSTER_QC_CSV)
print(qc.to_string())
print(QC_READING)
print("wrote", CLUSTER_QC_CSV.name)
```

```
n  pct_of_obj  pct_tcr  pct_tcr_vdjok  pct_mal  pct_mal_vdjok  pct_mal_of_tcr  pct_dom  pct_expanded   n_tcr  n_clones  top_clone_frac  med_clone_size  med_genes  med_mito  med_doublet  n_donors  n_patients  n_samples  n_studies  top_donor_frac    top_study                   flag      label
leiden_broad                                                                                                                                                                                                                                                                                                          
0               5073        0.63      0.2            0.2      0.1            0.2            87.5      0.1           0.2       8         5           0.250             0.0       2275      4.71        0.044       111          60         68         10           0.162  dorando2026  unscorable (<50 TCR+)  Mono_CD16
1              49277        6.08      0.2            0.2      0.1            0.1            35.9      0.1           0.1      78        64           0.051             0.0       1419      3.23        0.047       119          63         75         10           0.086   geskin2026                         Mono_CD14
2               4430        0.55     17.4           19.4      6.1            6.9            35.3      6.1          10.3     771       467           0.127             0.0       1800      3.16        0.068       122          71         82         10           0.120     buus2025                          Platelet
3              20026        2.47     18.8           38.7      8.5           17.5            45.4      8.5          11.6    3769      1828           0.240             0.0       2675      3.20        0.106       124          68         81         10           0.072   geskin2026                         Mono_CD14
4              24354        3.00      0.3           34.2      0.1            9.9            28.9      0.1           0.2      76        57           0.118             0.0       1789      3.57        0.104        77          34         35          8           0.642   geskin2026                               cDC
5              26413        3.26     51.1           51.7     47.8           48.4            93.6     47.8          51.1   13493       177           0.162            89.0        921      7.65        0.051       102          66         77         10           0.128   geskin2026                                 T
6               7347        0.91      1.3            1.9      0.6            0.9            46.3      0.6           0.8      95        62           0.200             0.0        408      0.38        0.049       102          47         49          8           0.154   geskin2026        LowQC candidate  Erythroid
7               8976        1.11      1.4            1.5      1.1            1.2            83.1      1.1           1.4     124        24           0.621             0.0        290      0.10        0.022        74          33         34          8           0.176   geskin2026        LowQC candidate  Erythroid
8              19095        2.35     22.8           42.3      0.6            1.2             2.8      0.6           8.6    4360      3610           0.013             0.0       1137      7.69        0.038       125          70         80          9           0.106   geskin2026                                 T
9              84848       10.46     39.1           58.1      1.2            1.7             3.0      1.2          27.9   33147     12652           0.021             0.0       1539      4.19        0.044       131          75         89         10           0.077   geskin2026                                 T
10            104723       12.91     63.6           81.8      1.4            1.8             2.2      1.4          12.0   66561     59743           0.011             1.0       1492      3.82        0.072       131          75         89         10           0.155    harro2023           
... [4764 more chars truncated]
```

```python  # cell 23
# ============================================================================
# §9b  Top clones on the UMAP — the one clonality panel worth its own figure.
#      Per-cluster _mal / _dom / _exp rates live in the §9 QC table, not in a plot.
# ============================================================================
_top = ad.obs.loc[ad.obs["clone_id"].astype(str).ne(""), "clone_id"].value_counts().head(8).index
ad.obs["_top_clone"] = pd.Categorical(
    np.where(ad.obs["clone_id"].astype(str).isin(_top), ad.obs["clone_id"].astype(str), "other"))
sync_view(adv, ad, "_top_clone")
save_umap(adv, ["_top_clone"], "umap_top_clones_v2.png", size=3)
```

```
<Figure size 1760x1056 with 5 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/blood_reannotation/umap_malignancy_v2.png

<Figure size 704x528 with 1 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/blood_reannotation/umap_top_clones_v2.png
```

# T / NK sub-clustering

## 10 — Subset T + NK; neighbors + UMAP + Leiden on the subset  ⚠️ HEAVY

10b §9 subsets `cell_type_broad == "T"` only. Blood also pulls in `NK`: the NK / CD8-EMRA boundary
is a continuum of the same cytotoxic program and resolves far better when both are clustered
together than when NK is frozen at the broad pass. `T_SUBSET_LABELS` in §0 controls this.

```python  # cell 26
# ============================================================================
# §10  T/NK subset: plotting view + Leiden   (HEAVY)
# ============================================================================
assert "cell_type_broad" in ad.obs, "fill cluster2ct in §8 first (pass 2)"
_m = ad.obs["cell_type_broad"].astype(str).isin(T_SUBSET_LABELS).to_numpy()
adt = ad[_m].copy()
adt.uns.pop("neighbors", None)                           # inherited from ad;
for _k in list(adt.obsp):                                # rebuilt below only if needed
    del adt.obsp[_k]
print(f"T/NK subset: {adt.n_obs:,} cells ({100 * _m.mean():.1f}% of blood) | "
      f"{adt.obs['cell_type_broad'].value_counts().to_dict()}")
print("T/NK per dataset:", adt.obs["dataset"].value_counts().to_dict())

coerce_latents(adt)
adtv = plot_view(adt, "T")
if not T_LEIDEN_CSV.exists():
    sc.pp.neighbors(adt, use_rep="X_mrvi_u", random_state=SEED)
frozen_leiden(adt, T_LEIDEN_CSV, "leiden_T", T_LEIDEN_RES)
print(adt.obs["leiden_T"].value_counts().reindex(
    natsorted(adt.obs["leiden_T"].cat.categories)).to_string())
sync_view(adtv, adt, "leiden_T")
```

```
T/NK subset: 667,584 cells (82.3% of blood) | {'T': 667584}
T/NK per dataset: {'geskin26': 167599, 'ren23': 135396, 'harro23': 129793, 'dorando26': 77703, 'buus25': 61215, 'borcherding23': 36630, 'gaydosik22': 22799, 'herrera21': 20886, 'borcherding19': 14170, 'rindler21': 1393}
plot_view[T]: 150,000/667,584 cells, computed UMAP -> blood_umap_T_v2_667584_150000.npy
computed leiden_T (res=1.0): 18 clusters -> froze to blood_T_leiden_v2.csv
leiden_T
0     103430
1      68849
2      19356
3      11483
4       2453
5         65
6     118262
7       4538
8      78904
9     101200
10     23820
11       883
12     40948
13     77656
14     10215
15      3180
16      1736
17       606

AnnData object with n_obs × n_vars = 150000 × 10000
    obs: 'cell_id', 'dataset', 'study', 'donor', 'real_donor', 'sample_id', 'sex', 'disease', 'disease_stage', 'compartment', 'tissue', 'tech', 'cell_type_orig', 'cell_type', 'n_genes', 'total_counts', 'pct_mito', 'doublet_score', 'predicted_doublet', 'patient_key', 'has_tcr', 'clone_id', 'tra_cdr3', 'trb_cdr3', 'clone_size', 'is_expanded', 'is_dominant_clone', 'is_malignant', 'repo', 'accession', 'organ', 'tissue_detail', 'malignant_call_method', 'tcr_available', 'malignant_labeled', 'malignant_frac', 'tcr_recovery', 'stage_class', 'stage_clean', 'lineage', 'entity', 'fmf_status', 'lineage_resolve', 'treatment_context', 'blood_involvement', 'lesion_type', 'cohort_country', 'cohort_region', 'origin_resolve', 'east_asian', 'lineage_orig', 'lineage_score_cd4', 'lineage_score_cd8', 'lineage_score_gd', 'lineage_score_ab', 'lineage_method', 'lineage_disagree', 'lineage_disagreement', '_tcr', '_mal', '_dom', '_exp', '_vdj_ok', 'leiden_broad', 'cell_type_broad', '_top_clone', 'leiden_T'
    var: 'highly_variable', 'highly_variable_rank', 'means', 'variances', 'variances_norm', 'highly_variable_nbatches'
    uns: 'adt_panel', 'hvg', 'log1p', '_sweep', 'leiden_broad', 'dendrogram_leiden_broad', 'leiden_broad_colors', 'neighbors', 'umap', 'plot_idx'
    obsm: 'X_adt', 'X_scVI', 'X_scVI_MDE', '_scvi_extra_categorical_covs', 'X_mrvi_u', 'X_mrvi_z', 'X_umap'
    layers: 'raw_counts', 'counts'
    obsp: 'distances', 'connectivities'
```

## 11 — T / NK marker dot plot

**10b's `markers_T` is pasted verbatim as `MARKERS_T_10B`** and every one of its keys survives, so
a `CD4` / `CD8` / `CD4_Treg` call here is made on the same genes as in skin. Blood extends it in
four directions:

- **the differentiation axis is resolved**, because PBMC has real naive and EMRA compartments that
  skin does not: 10b's `Naive/mem` becomes `Naive/CM` + `EM` + `EMRA`.
- **unconventional T**: γδ and MAIT, which 10b does not attempt.
- **Th1 / Tfh** alongside 10b's `Th2/CTCL` and `Th17/Tc17`.
- **an explicit Sézary panel, split into GAIN and LOSS.** These read in *opposite* directions:
  the tumour signature is GAIN-high **and** LOSS-low. `CD7` and `DPP4` (CD26) loss is the classic
  flow-cytometry Sézary call and is the single most useful row in the plot.

⚠️ Sézary cells are CD4⁺CCR7⁺CD27⁺CD62L⁺ — a **central-memory** surface phenotype. `CCR7` / `SELL`
/ `TCF7` positivity does **not** exclude tumour, and marker-only annotation will merge Sézary into
normal CM CD4. §11b is what separates them; read it before filling the map.

```python  # cell 28
# ============================================================================
# §11  T/NK marker dot plot
# ============================================================================
# --- 10b's T panel, verbatim. Do not edit. ------------------------------------------------------
MARKERS_T_10B = {
    "T core":     ["CD3D", "CD3E", "TRAC"],
    "CD4":        ["CD4", "IL7R"],
    "CD8":        ["CD8A", "CD8B"],
    "Naive/mem":  ["CCR7", "SELL", "TCF7"],
    "Cytotoxic":  ["GZMK", "GZMB", "PRF1", "NKG7", "GNLY"],
    "Treg":       ["FOXP3", "CTLA4", "IL2RA"],
    "Th2/CTCL":   ["GATA3", "CCR4"],
    "Th17/Tc17":  ["RORC", "IL17A", "CCR6"],
    "Exhaustion": ["PDCD1", "TOX", "LAG3", "TIGIT", "HAVCR2"],
    "NK":         ["NCAM1", "KLRD1", "KLRF1"],
    "Prolif":     ["MKI67", "TOP2A"],
}

markers_T = {
    "T core":          MARKERS_T_10B["T core"] + ["CD247"],
    "CD4":             MARKERS_T_10B["CD4"] + ["CD40LG"],
    "CD8":             MARKERS_T_10B["CD8"],
    "gdT":             ["TRDC", "TRGC1", "TRGC2", "TRDV1", "TRDV2"],
    # ---- differentiation axis: 10b's Naive/mem, resolved into naive -> CM -> EM -> EMRA
    "Naive/CM":        MARKERS_T_10B["Naive/mem"] + ["LEF1", "MAL", "NOSIP", "ACTN1"],
    "EM":              ["GZMK", "CXCR3", "S100A4", "CD27", "CD28", "ITGB1"],
    "EMRA":            ["GZMB", "GZMH", "GNLY", "PRF1", "NKG7", "KLRG1", "CX3CR1",
                        "FGFBP2", "ADGRG1", "ZEB2"],
    "Cytotoxic":       MARKERS_T_10B["Cytotoxic"] + ["GZMA", "GZMH", "CTSW", "HOPX"],
    # ---- CD4 helper polarisation (10b's Th2/CTCL + Th17/Tc17, plus Th1 / Tfh)
    "Th1":             ["TBX21", "IFNG", "CXCR3"],
    "Th2/CTCL":        MARKERS_T_10B["Th2/CTCL"] + ["IL4", "IL5", "IL13"],
    "Th17/Tc17":       MARKERS_T_10B["Th17/Tc17"] + ["IL23R"],
    "Tfh":             ["CXCL13", "CXCR5", "BCL6", "ICOS"],
    "Treg":            MARKERS_T_10B["Treg"] + ["IKZF2", "TNFRSF18", "CCR8", "IL1R2", "LRRC32"],
    "MAIT":            ["SLC4A10", "KLRB1", "TRAV1-2", "ZBTB16", "RORA", "NCR3"],
    "NK":              MARKERS_T_10B["NK"] + ["KLRC1", "TYROBP", "FCER1G"],
    "Exhaustion/dysf": MARKERS_T_10B["Exhaustion"] + ["TOX2", "ENTPD1", "BATF", "EOMES"],
    "IFN response":    ["ISG15", "IFI6", "MX1", "IFIT1", "IFIT3", "OAS1", "STAT1", "IRF7"],
    # ---- Sezary / malignant CD4: GAIN and LOSS read in OPPOSITE directions
    "Sezary GAIN":     ["KIR3DL2", "CADM1", "TWIST1", "PLS3", "DNM3", "TNFRSF8", "TNFRSF1B",
                        "GATA3", "CCR4", "TOX", "MIR155HG", "IL2RB", "IL26", "SATB1",
                        "SESN3", "AHI1", "LAIR2", "ITGB1"],
    "Sezary LOSS":     ["CD7", "DPP4", "CD27", "SELL", "TCF7", "LEF1", "CD5", "CD6"],
    "Prolif":          MARKERS_T_10B["Prolif"] + ["STMN1"],
    "Stress warn":     ["HSPA1A", "JUN", "FOS", "MALAT1"],
    "Ambient warn":    ["HBB", "PPBP", "LYZ"],
}
markers_T = filter_panel(markers_T, set(adt.var_names), "T/NK panel")

sc.tl.dendrogram(adt, groupby="leiden_T", use_rep="X_mrvi_u")
save_dotplot(adt, markers_T, "leiden_T", "dotplot_T_v2.png", dendrogram=True, figsize=(26, 8))

# narrow Sezary read: GAIN-high AND LOSS-low is the tumour signature
_sez = {k: markers_T[k] for k in ("CD4", "CD8", "Naive/CM", "Sezary GAIN", "Sezary LOSS")
        if k in markers_T}
save_dotplot(adt, _sez, "leiden_T", "dotplot_T_sezary_v2.png", dendrogram=True, figsize=(14, 8))

# leiden_T is what §12 is annotated from; _mal is what separates Sezary from normal CM CD4.
sync_view(adtv, adt, "_mal")
save_umap(adtv, ["leiden_T", "_mal"], "umap_T_v2.png", size=3)
```

```
T/NK panel: 22/22 panels usable, 132 genes
  dropped from CD4: ['CD4']
  dropped from Th2/CTCL: ['IL5']
  dropped from Sezary GAIN: ['TNFRSF8']
WARNING: Groups are not reordered because the `groupby` categories and the `var_group_labels` are different.
categories: 0, 1, 2, etc.
var_group_labels: T core, CD4, CD8, etc.

<Figure size 2860x880 with 6 Axes>
<Figure size 2860x880 with 6 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/blood_reannotation/dotplot_T_v2.png
WARNING: Groups are not reordered because the `groupby` categories and the `var_group_labels` are different.
categories: 0, 1, 2, etc.
var_group_labels: CD4, CD8, Naive/CM, etc.

<Figure size 1540x880 with 6 Axes>
<Figure size 1540x880 with 6 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/blood_reannotation/dotplot_T_sezary_v2.png

<Figure size 1760x1056 with 4 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/blood_reannotation/umap_T_overview_v2.png
```

```python  # cell 29
# ============================================================================
# §11b  Per-cluster malignancy on the T/NK subset
#       READ THIS BEFORE FILLING cluster2ct_T IN §12.
#       Sezary cells are CD4+ CCR7+ CD27+ CD62L+ — a CENTRAL-MEMORY surface phenotype — so
#       CCR7/SELL/TCF7 positivity does NOT exclude tumour and marker-only annotation will
#       merge Sezary with normal CM CD4. The malignancy/clonality axis is what separates them.
# ============================================================================
qc_T = cluster_qc(adt, "leiden_T")
qc_T.to_csv(T_CLUSTER_QC_CSV)
print(qc_T.to_string())
print(QC_READING)
print("wrote", T_CLUSTER_QC_CSV.name)
```

```
n  pct_of_obj  pct_tcr  pct_tcr_vdjok  pct_mal  pct_mal_vdjok  pct_mal_of_tcr  pct_dom  pct_expanded  n_tcr  n_clones  top_clone_frac  med_clone_size  med_genes  med_mito  med_doublet  n_donors  n_patients  n_samples  n_studies  top_donor_frac   top_study                   flag
leiden_T                                                                                                                                                                                                                                                                                             
0         103430       15.49     63.7           82.2      1.5            1.9             2.3      1.5          13.0  65847     58445           0.012             1.0       1491      3.89        0.071       131          75         89         10           0.144   harro2023                       
1          68849       10.31     36.8           54.1      1.2            1.8             3.3      1.2          30.0  25336      6980           0.028             0.0       1485      4.21        0.037       131          75         89         10           0.076  geskin2026                       
2          19356        2.90     24.8           47.4      0.6            1.2             2.4      0.6           9.0   4803      4017           0.012             0.0       1161      7.84        0.039       126          71         81         10           0.107  geskin2026                       
3          11483        1.72     44.9           71.9      1.1            1.7             2.4      1.1          17.4   5152      3933           0.018             0.0       3403      3.33        0.064       129          73         85         10           0.044  geskin2026                       
4           2453        0.37     61.4           75.7      1.6            2.0             2.7      1.6           3.4   1505      1462           0.023             1.0       1688      3.25        0.050       103          54         59          9           0.118  geskin2026                       
5             65        0.01     90.8           93.7      0.0            0.0             0.0      0.0          12.3     59        58           0.034             1.0       1480      3.87        0.044        16          16         16          7           0.354   harro2023                       
6         118262       17.71     58.2           88.1      0.0            0.1             0.1      0.0           2.7  68829     67704           0.000             1.0       1418      3.75        0.058       130          75         89         10           0.183  geskin2026                       
7           4538        0.68     85.5           86.6     80.7           81.7            94.3     80.7          85.4   3881       127           0.073          3277.0       1928      1.73        0.051        93          62         73         10           0.073    buus2025                       
8          78904       11.82     86.1           88.6     81.4           83.9            94.6     81.4          86.0  67899       261           0.122          6432.0       1505      4.24        0.088        93          62         75         10           0.108     ren2023                       
9         101200       15.16     92.6           95.3     87.4           89.9            94.4     87.4          92.5  93677       244           0.097          6432.0       1770      3.44        0.090        96          65         77         10           0.091     ren2023                       
10         23820        3.57     50.9           51.6     47.7           48.3            93.7     47.7          50.9  12132       169           0.177            80.0        911      7.84        0.049       101          66         77         10           0.131  geskin2026                       
11           883        0.13      2.7            2.7      1.2            1.2            45.8      1.2           1.8     24        16           0.333             0.0       1072      0.54        
... [2564 more chars truncated]
```

## 12 — Map T/NK clusters → `cell_type_T`  (manual)

Same name and same two-pass workflow as 10b §11. **`VOCAB_T` uses 10b's spellings**: `CD4`, `CD8`,
`CD4_Treg` (not `Tregs`), `Unk` (not `UNK`) — plus `gdT`, `MAIT`, `NK`, `Prolif`, `LowQC`,
`Doublet`, which skin does not resolve.

10b resolves its T compartment into just `CD4` / `CD8` / `CD4_Treg` and pushes everything doubtful
to `Unk` for the §12.5 second pass. Do the same here: a cluster you cannot call is `Unk`, not a
guess.

```python  # cell 31
# ============================================================================
# §12  leiden_T -> cell_type_T   (HAND-FILLED — pass 2)
# ============================================================================
# --- FILL from the §11 dot plots + the §11b malignancy table ---
cluster2ct_T = {
    "0": "CD4",
    "1": "CD8",
    "2": "Unk",
    "3": "CD8",
    "4": "Unk",
    "5": "CD4",
    "6": "Unk",
    "7": "CD4",
    "8": "CD4",
    "9": "CD4",
    "10": "CD4",
    "11": "Unk",
    "12": "CD4",
    "13": "CD4",
    "14": "CD4",
    "15": "CD4",
    "16": "CD4",
    "17": "NK"
}

CLUSTER_SIZES_T = {'0': 103430, '1': 68849, '2': 19356, '3': 11483, '4': 2453, '5': 65, '6': 118262, '7': 4538, '8': 78904, '9': 101200, '10': 23820, '11': 883, '12': 40948, '13': 77656, '14': 10215, '15': 3180, '16': 1736, '17': 606}

# --- v1 reference only (different clustering; kept to show the resolved level set) -------------
#   CD4 dominated (0,1,2,4,6,7,8,9,17) · CD8 (10) · NK (15) · the rest went to the second pass.
#   v1 named Tregs/UNK; v2 uses 10b's CD4_Treg/Unk.

_ok_T = apply_hand_map(adt, "leiden_T", cluster2ct_T, VOCAB_T, CLUSTER_SIZES_T, "cell_type_T")
```

```
cell_type_T
CD4    445692
Unk    140954
CD8     80332
NK        606
```

## 12.5 — Re-cluster the still-unknown T/NK cells  ⚠️ HEAVY

10b §11.5, unchanged in intent: whatever §12 left as `Unk` gets its own graph at `res=0.5` and a
second reading of the same T panel. In v1 blood this recovered ~98k cells that the first pass
could not call.

```python  # cell 33
# ============================================================================
# §12.5  Second pass on the still-unknown cells   (HEAVY)
# ============================================================================
assert "cell_type_T" in adt.obs, "fill cluster2ct_T in §12 first (pass 2)"
unk_mask = adt.obs["cell_type_T"].astype(str).isin(UNK_LABELS).to_numpy()
print(f"still-unknown T/NK cells: {int(unk_mask.sum()):,} / {adt.n_obs:,} "
      f"({100 * unk_mask.mean():.1f}% of the T/NK subset)")

if unk_mask.sum() == 0:
    print("nothing left unknown — skipping the second pass")
    adu = aduv = None
else:
    adu = adt[unk_mask].copy()
    adu.uns.pop("neighbors", None)
    for _k in list(adu.obsp):
        del adu.obsp[_k]
    coerce_latents(adu)
    aduv = plot_view(adu, "unk")
    if not UNK_LEIDEN_CSV.exists():
        sc.pp.neighbors(adu, use_rep="X_mrvi_u", random_state=SEED)
    frozen_leiden(adu, UNK_LEIDEN_CSV, "leiden_unk", UNK_LEIDEN_RES)
    print(adu.obs["leiden_unk"].value_counts().reindex(
        natsorted(adu.obs["leiden_unk"].cat.categories)).to_string())
    sync_view(aduv, adu, "leiden_unk")
```

```
still-unknown T/NK cells: 140,954 / 667,584 (21.1% of the T/NK subset)
plot_view[unk]: 140,954/140,954 cells, computed UMAP -> blood_umap_unk_v2_140954_140954.npy
computed leiden_unk (res=0.5): 9 clusters -> froze to blood_T_unk_leiden_v2.csv
leiden_unk
0    23088
1    19659
2    13967
3    26400
4    30725
5     1552
6     5084
7    19593
8      886
```

```python  # cell 34
# ============================================================================
# §12.5b  Unknown subset: dot plot + malignancy table  (reuses the §11 T panel, as 10b does)
# ============================================================================
if adu is not None:
    _mu = filter_panel(dict(markers_T), set(adu.var_names), "UNK panel")
    sc.tl.dendrogram(adu, groupby="leiden_unk", use_rep="X_mrvi_u")
    save_dotplot(adu, _mu, "leiden_unk", "dotplot_T_unk_v2.png", dendrogram=True, figsize=(26, 6))
    save_umap(aduv, ["leiden_unk"], "umap_T_unk_v2.png", size=4)
    print(cluster_qc(adu, "leiden_unk").to_string())
```

```
UNK panel: 22/22 panels usable, 132 genes
WARNING: Groups are not reordered because the `groupby` categories and the `var_group_labels` are different.
categories: 0, 1, 2, etc.
var_group_labels: T core, CD4, CD8, etc.

<Figure size 2860x660 with 6 Axes>
<Figure size 2860x660 with 6 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/blood_reannotation/dotplot_T_unk_v2.png

<Figure size 1760x1056 with 4 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/blood_reannotation/umap_T_unk_v2.png
                n  pct_of_obj  pct_tcr  pct_tcr_vdjok  pct_mal  pct_mal_vdjok  pct_mal_of_tcr  pct_dom  pct_expanded  n_tcr  n_clones  top_clone_frac  med_clone_size  med_genes  med_mito  med_doublet  n_donors  n_patients  n_samples  n_studies  top_donor_frac   top_study                   flag
leiden_unk                                                                                                                                                                                                                                                                                            
0           23088       16.38     59.3           85.3      0.0            0.1             0.1      0.0           4.4  13697     13530           0.000             1.0       1383      3.89        0.063       120          69         80          9           0.199  geskin2026                       
1           19659       13.95     63.4           85.1      0.1            0.1             0.1      0.1           3.6  12471     12393           0.001             1.0       1530      3.80        0.072       124          73         84         10           0.225   harro2023                       
2           13967        9.91     59.3           87.2      0.0            0.0             0.0      0.0           2.6   8285      8254           0.000             1.0       1489      3.22        0.061       112          66         74          9           0.204  geskin2026                       
3           26400       18.73     68.3           91.7      0.0            0.0             0.0      0.0           2.2  18025     17958           0.000             1.0       1314      3.92        0.056       119          70         81         10           0.210   harro2023                       
4           30725       21.80     51.2           88.5      0.0            0.1             0.1      0.0           1.7  15744     15693           0.000             1.0       1472      3.88        0.050       125          73         83         10           0.129  geskin2026                       
5            1552        1.10     61.4           81.6      2.6            3.4             4.2      2.6           5.3    953       909           0.036             1.0       1634      2.86        0.067        93          50         54          9           0.082  geskin2026                       
6            5084        3.61     20.1           89.6      0.0            0.0             0.0      0.0           0.7   1020      1017           0.002             0.0       1409      2.87        0.035       102          59         66          9           0.118  geskin2026                       
7           19593       13.90     25.2           47.9      0.6            1.2             2.4      0.6           9.0   4944      4153           0.012             0.0       1164      7.78        0.040       126          71         82         10           0.107  geskin2026                       
8             886        0.63      2.5            2.5      1.0            1.0            40.9      1.0           1.5     22        14           0.364             0.0       1070      0.54        0.040        16          14         15          4           0.907    buus2025  unscorable (<50 TCR+)
```

### Annotate the unknown clusters → merge back into `cell_type_T`

```python  # cell 36
# ============================================================================
# §12.5c  leiden_unk -> cell_type_T (merge back by cell_id)   (HAND-FILLED — pass 2)
# ============================================================================
# --- FILL from the §12.5b dot plot ---
cluster2ct_unk = {"0": "CD4", "1": "CD4", "2": "CD4", "3": "CD4", "4": "CD8", "5": "CD8", "6": "CD4", "7": "CD4", "8": "CD4"}

CLUSTER_SIZES_UNK = {'0': 23088, '1': 19659, '2': 13967, '3': 26400, '4': 30725, '5': 1552, '6': 5084, '7': 19593, '8': 886}

if adu is None:
    print("no unknown subset — nothing to merge")
elif apply_hand_map(adu, "leiden_unk", cluster2ct_unk, VOCAB_T,
                    CLUSTER_SIZES_UNK, "cell_type_unk"):
    new_by_id = pd.Series(adu.obs["cell_type_unk"].astype(str).to_numpy(),
                          index=adu.obs["cell_id"].astype(str).to_numpy())
    ids = adt.obs["cell_id"].astype(str).to_numpy()
    repl = new_by_id.reindex(ids[unk_mask])
    assert repl.notna().all(), \
        f"{int(repl.isna().sum())} unknown cells missing a new label — cell_id mismatch"
    ct = adt.obs["cell_type_T"].astype(str).to_numpy()
    ct[unk_mask] = repl.to_numpy()
    adt.obs["cell_type_T"] = pd.Categorical(ct)
    print("\ncell_type_T after the second pass:")
    print(adt.obs["cell_type_T"].value_counts(dropna=False).to_string())
```

```
cell_type_unk
CD4    108677
CD8     32277

cell_type_T after the second pass:
cell_type_T
CD4    554369
CD8    112609
NK        606
```

## 13 — `sezary_like`: a separate column, **not** a cell-type level

10b's skin vocabulary has no tumour level either — malignancy in this atlas has four
non-equivalent definitions (`23_malignancy_tcr_cnv`/`26_transcriptome_malignancy_holdout`) and belongs in its own column, not folded into
`cell_type_final`. Putting "Sezary" in the cell-type vocabulary would also break the concatenation
with the skin labels that `12_atlas_descriptive` does.

The vote is per-Leiden-cluster and computed on the **TCR+ subset only**. Clusters with fewer than
`MIN_TCR_TO_SCORE` TCR+ cells are left `NaN` → `False`: they are **unscorable, not benign**.

```python  # cell 38
# ============================================================================
# §13  sezary_like — cluster-level vote on the TCR+ subset only
# ============================================================================
MIN_TCR_TO_SCORE = 50
SEZARY_FRAC = 0.5

_t = adt.obs[adt.obs["_tcr"]]
frac = _t.groupby("leiden_T", observed=True)["_mal"].mean()
n_tcr = _t.groupby("leiden_T", observed=True).size()
frac = frac.where(n_tcr >= MIN_TCR_TO_SCORE)          # unscorable -> NaN, NOT False

vote = pd.DataFrame({"n_tcr": n_tcr.reindex(frac.index).fillna(0).astype(int),
                     "mal_frac_of_tcr": frac.round(3),
                     "sezary_like": frac >= SEZARY_FRAC})
vote = vote.reindex(natsorted(vote.index.astype(str)))
print(vote.to_string())
print(f"\nclusters with <{MIN_TCR_TO_SCORE} TCR+ cells are NOT scored (NaN -> False): they are "
      "UNSCORABLE, not benign.")

adt.obs["sezary_like"] = adt.obs["leiden_T"].map(frac >= SEZARY_FRAC).fillna(False).astype(bool)
print(f"\nsezary_like: {adt.obs['sezary_like'].sum():,} / {adt.n_obs:,} cells "
      f"({100 * adt.obs['sezary_like'].mean():.1f}% of T/NK)")
if "cell_type_T" in adt.obs:
    print(pd.crosstab(adt.obs["cell_type_T"], adt.obs["sezary_like"]).to_string())
    print("\nper-patient Sezary fraction (top 20 by cell count):")
    _gg = adt.obs.groupby("patient_key", observed=True)
    _pp = pd.DataFrame({
        "n": _gg.size(), "disease": _gg["disease"].first(),
        "pct_sezary_like": (100 * _gg["sezary_like"].mean()).round(1),
        "pct_mal_of_tcr": (100 * _gg["_mal"].mean()).round(1),
        }).sort_values("n", ascending=False).head(20)
    print(_pp.to_string())
sync_view(adtv, adt, "cell_type_T", "sezary_like")
save_umap(adtv, ["cell_type_T", "sezary_like"], "umap_T_sezary_v2.png", size=3)
```

```
n_tcr  mal_frac_of_tcr  sezary_like
leiden_T                                     
0         65847            0.023        False
1         25336            0.033        False
2          4803            0.024        False
3          5152            0.024        False
4          1505            0.027        False
5            59            0.000        False
6         68829            0.001        False
7          3881            0.943         True
8         67899            0.946         True
9         93677            0.944         True
10        12132            0.937         True
11           24              NaN        False
12        37001            0.946         True
13        71250            0.950         True
14         9277            0.988         True
15         2741            0.960         True
16         1566            0.912         True
17           13              NaN        False

clusters with <50 TCR+ cells are NOT scored (NaN -> False): they are UNSCORABLE, not benign.

sezary_like: 342,197 / 667,584 cells (51.3% of T/NK)
sezary_like   False   True 
cell_type_T                
CD4          212172  342197
CD8          112609       0
NK              606       0

per-patient Sezary fraction (top 20 by cell count):
                     n     disease  pct_sezary_like  pct_mal_of_tcr
patient_key                                                        
other/B7:MF1     42688          MF              0.0             0.0
washu:CTCL2      31441          SS             91.8            93.6
pittsburgh:SZ38  24940          SS             98.8            90.7
pittsburgh:SZ5   23391          SS             92.2            84.2
washu:1084       23116  CTCL_other             97.9            86.1
washu:1126       21654  CTCL_other             89.0            84.5
other/B7:MF2     21447          MF              0.0             0.0
pittsburgh:PF50  20691          SS              0.2             0.0
other/B7:SS3     19634          SS             98.4            95.8
pittsburgh:PF47  17267          SS              0.1             0.0
pittsburgh:PF49  17189          SS              0.0             0.0
other/B7:SS1     16935          SS             89.1            81.8
pittsburgh:PF46  16858          SS              0.1             0.0
other/B7:MF3     12959          MF              0.0             0.0
other/B6:N2      12554          HC              0.0             0.0
other/B6:P2      12184  CTCL_other             56.8            47.9
other/B6:P6      12065  CTCL_other             88.3             0.0
other/B6:P4      11927  CTCL_other             65.1            61.0
pittsburgh:PF48  11284          SS              0.0             0.0
other/B6:N1      11258          HC              0.0             0.0

<Figure size 1760x528 with 2 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/blood_reannotation/umap_T_sezary_v2.png
```

## 14 — Merge broad + T/NK → `cell_type_final`

10b §12.5, with `T` **and** `NK` replaced by their subtype instead of `T` alone.

`CT_FINAL_VOCAB` is derived from `VOCAB_BROAD` and `VOCAB_T` rather than retyped, so it cannot
drift from the vocabularies the hand maps were validated against.

`cell_type_ct12` collapses the blood levels onto `12_atlas_descriptive`'s harmonized skin vocabulary
(`Myeloid` absorbs the monocyte/DC split, `NK_ILC` absorbs NK/γδ/MAIT, `other` takes the
non-lymphoid PBMC constituents). With the §0 renames — `CD4_Treg`, `Unk` — every remaining level
is already a skin level, so `cell_type_final` itself concatenates cleanly and `cell_type_ct12` is
now only a convenience for older notebooks.

```python  # cell 40
# ============================================================================
# §14  Merge -> cell_type_final + the skin-compatible collapse
# ============================================================================
# nb10b v2 skin vocabulary — what cell_type_ct12 must land inside (plus NK_ILC / other).
CT_SKIN = ["CD4", "CD8", "CD4_Treg", "B", "Plasma", "Myeloid", "Mast",
           "Keratinocyte", "Fibroblast", "Vascular", "Melanocyte", "Unk"]
# derived, never retyped: exactly the labels the two hand maps could have produced
CT_FINAL_VOCAB = (VOCAB_BROAD - set(T_SUBSET_LABELS)) | VOCAB_T

BLOOD_TO_CT12 = {
    # already skin levels — identity
    "CD4": "CD4", "CD8": "CD8", "CD4_Treg": "CD4_Treg", "B": "B", "Plasma": "Plasma",
    "Mast": "Mast", "Myeloid": "Myeloid", "Unk": "Unk",
    # blood myeloid split -> skin's single Myeloid level
    "Mono_CD14": "Myeloid", "Mono_CD16": "Myeloid", "cDC": "Myeloid", "pDC": "Myeloid",
    # innate / unconventional lymphoid: no skin level exists
    "NK": "NK_ILC", "gdT": "NK_ILC", "MAIT": "NK_ILC",
    # non-lymphoid PBMC constituents and state-only levels
    "Platelet": "other", "Erythroid": "other", "HSPC": "other", "Prolif": "other",
    "LowQC": "Unk", "Doublet": "Unk",
}
assert set(BLOOD_TO_CT12) == CT_FINAL_VOCAB, \
    f"BLOOD_TO_CT12 vs CT_FINAL_VOCAB mismatch: {CT_FINAL_VOCAB ^ set(BLOOD_TO_CT12)}"
assert set(BLOOD_TO_CT12.values()) <= set(CT_SKIN) | {"NK_ILC", "other"}

ids = ad.obs["cell_id"].astype(str).to_numpy()
final = ad.obs["cell_type_broad"].astype(str).to_numpy().astype(object)
sub = adt.obs.set_index(adt.obs["cell_id"].astype(str))["cell_type_T"].astype(str)
is_sub = np.isin(final, list(T_SUBSET_LABELS))
repl = sub.reindex(ids[is_sub])
assert repl.notna().all(), \
    f"{int(repl.isna().sum())} T/NK cells missing a subtype after the merge — cell_id mismatch"
final[is_sub] = repl.to_numpy()

ad.obs["cell_type_final"] = pd.Categorical(final.astype(str))
bad = set(ad.obs["cell_type_final"].astype(str).unique()) - CT_FINAL_VOCAB
assert not bad, f"cell_type_final has levels outside CT_FINAL_VOCAB: {sorted(bad)}"
ad.obs["cell_type_ct12"] = pd.Categorical(
    ad.obs["cell_type_final"].astype(str).map(BLOOD_TO_CT12))
assert ad.obs["cell_type_ct12"].notna().all()

sez = adt.obs.set_index(adt.obs["cell_id"].astype(str))["sezary_like"]
ad.obs["sezary_like"] = pd.Series(
    sez.reindex(ids).to_numpy(), index=ad.obs.index).fillna(False).astype(bool)

print("cell_type_final:")
print(ad.obs["cell_type_final"].value_counts().to_string())
print("\ncell_type_ct12:")
print(ad.obs["cell_type_ct12"].value_counts().to_string())
print(f"\nsezary_like: {ad.obs['sezary_like'].sum():,} cells "
      f"({100 * ad.obs['sezary_like'].mean():.1f}% of blood)")
_shared = sorted(set(ad.obs["cell_type_final"].astype(str)) & set(CT_SKIN))
print(f"\nlevels shared verbatim with nb10b skin: {_shared}")
print(f"blood-only levels: {sorted(set(ad.obs['cell_type_final'].astype(str)) - set(CT_SKIN))}")
```

```
cell_type_final:
cell_type_final
CD4          554369
CD8          112609
Mono_CD14     69303
cDC           25103
B             18504
Erythroid     16323
Mono_CD16      5073
Plasma         4704
Platelet       4430
NK              606

cell_type_ct12:
cell_type_ct12
CD4        554369
CD8        112609
Myeloid     99479
other       20753
B           18504
Plasma       4704
NK_ILC        606

sezary_like: 342,197 cells (42.2% of blood)

levels shared verbatim with nb10b skin: ['B', 'CD4', 'CD8', 'Plasma']
blood-only levels: ['Erythroid', 'Mono_CD14', 'Mono_CD16', 'NK', 'Platelet', 'cDC']
```

```python  # cell 41
# ============================================================================
# §14b  Composition sanity — the healthy blood donors are the internal reference
#       v2 has 47,679 HC blood cells across 7 donors (N1-N3 ren2023, HB1-HB3 gaydosik2022,
#       H__HC1 herrera2021); v1 had one 4,481-cell sample. This check depends on neither the
#       marker panel nor is_malignant: if healthy blood does not look like normal PBMC, the
#       annotation is wrong.
# ============================================================================
comp = (100 * pd.crosstab(ad.obs["real_donor"], ad.obs["cell_type_final"],
                          normalize="index")).round(1)
# crosstab of two categoricals returns CategoricalIndex on BOTH axes, and DataFrame.join does
# `right[:]` internally -> InvalidIndexError. Flatten to plain string Index before any join/.loc.
comp.index = comp.index.astype(str)
comp.columns = comp.columns.astype(str)
_hc_donors = sorted(ad.obs.loc[ad.obs["disease"].astype(str) == "HC", "real_donor"]
                    .astype(str).unique())
if _hc_donors:
    print(f"INTERNAL HEALTHY REFERENCE — {len(_hc_donors)} donors: {_hc_donors}")
    hc = comp.loc[[d for d in comp.index if str(d) in _hc_donors]]
    print("\n% of each healthy donor's cells:")
    print(hc.loc[:, (hc > 0).any()].to_string())
    print("\nmean across healthy donors:")
    print(hc.mean().sort_values(ascending=False).head(12).round(1).to_string())
    print("\nExpect a recognizable normal PBMC mix: CD4, CD8, NK, B, Mono_CD14 all present and no "
          "single level above ~40%. A healthy donor that is 80% one T level means the broad or "
          "the T map is wrong — go back to §8 / §12 before writing anything.")
    print(f"\nsezary_like among healthy donors: "
          f"{100 * ad.obs.loc[ad.obs['real_donor'].astype(str).isin(_hc_donors), 'sezary_like'].mean():.2f}% "
          "(should be ~0)")
else:
    print("!! no HC blood in scope — the healthy-reference sanity check is UNAVAILABLE. "
          "Do not exclude ren2023 / gaydosik2022 / herrera2021.")

_g = ad.obs.groupby("real_donor", observed=True)
_meta = pd.DataFrame({
    "disease": _g["disease"].first().astype(str), "study": _g["study"].first().astype(str),
    "n": _g.size(),
    "pct_mal_of_tcr": (100 * _g["_mal"].mean()).round(1),
    "pct_sezary_like": (100 * _g["sezary_like"].mean()).round(1)})
_meta.index = _meta.index.astype(str)
print("\nper-donor composition (%) + malignancy:")
print(_meta.join(comp).sort_values("n", ascending=False).to_string())
```

```
INTERNAL HEALTHY REFERENCE — 7 donors: ['HB1', 'HB2', 'HB3', 'H__HC1', 'N1', 'N2', 'N3']

% of each healthy donor's cells:
cell_type_final    B   CD4   CD8  Erythroid  Mono_CD14  Mono_CD16  Plasma  Platelet  cDC
real_donor                                                                              
HB1              4.2  47.4  35.8        0.3        7.0        1.6     1.7       0.2  1.7
HB2              6.5  53.3  31.8        0.3        3.6        1.4     1.3       0.4  1.4
HB3              7.3  48.5  31.4        0.1        7.5        2.2     1.2       0.4  1.5
H__HC1           7.2  34.3  30.3        0.0       20.1        1.5     2.7       2.0  1.8
N1               0.2  75.4  23.0        0.0        0.1        0.2     0.3       0.6  0.2
N2               0.3  78.2  21.0        0.0        0.0        0.0     0.0       0.4  0.0
N3               0.2  79.8  19.2        0.0        0.1        0.2     0.2       0.3  0.0

mean across healthy donors:
cell_type_final
CD4          59.6
CD8          27.5
Mono_CD14     5.5
B             3.7
Plasma        1.1
Mono_CD16     1.0
cDC           0.9
Platelet      0.6
Erythroid     0.1
NK            0.0

Expect a recognizable normal PBMC mix: CD4, CD8, NK, B, Mono_CD14 all present and no single level above ~40%. A healthy donor that is 80% one T level means the broad or the T map is wrong — go back to §8 / §12 before writing anything.

sezary_like among healthy donors: 0.01% (should be ~0)

per-donor composition (%) + malignancy:
               disease            study      n  pct_mal_of_tcr  pct_sezary_like     B   CD4   CD8  Erythroid  Mono_CD14  Mono_CD16   NK  Plasma  Platelet   cDC
real_donor                                                                                                                                                     
MF1                 MF        harro2023  42968             0.0              0.0   0.0  76.9  22.5        0.0        0.0        0.0  0.0     0.1       0.4   0.0
B2__CTCL2           SS  borcherding2023  33743            90.4             85.6   0.1  90.5   2.7        0.0        5.9        0.3  0.0     0.2       0.3   0.0
1084        CTCL_other      dorando2026  23229            85.8             97.4   0.0  99.3   0.2        0.0        0.1        0.0  0.0     0.0       0.3   0.0
1126        CTCL_other      dorando2026  22054            83.1             87.4   0.0  93.0   5.2        0.0        0.5        0.8  0.0     0.2       0.1   0.1
MF2                 MF        harro2023  21964             0.0              0.0   0.2  76.8  20.8        0.0        0.1        0.0  0.0     0.5       1.5   0.1
SS3                 SS        harro2023  19648            95.8             98.3   0.0  99.7   0.2        0.0        0.0        0.0  0.0     0.0       0.0   0.0
PF48_HTO2           SS       geskin2026  19171             0.0              0.0   0.4   7.9   4.4        0.1        4.7        0.1  0.4     0.3       0.1  81.6
SS1                 SS        harro2023  16978            81.6             88.9   0.1  98.0   1.7        0.0        0.0        0.0  0.0     0.1       0.0   0.0
1181        CTCL_other      dorando2026  15588             4.9              5.0   0.2  50.0  15.9        0.2       27.6        5.3  0.0     0.1       0.1   0.6
MF3                 MF        harro2023  13102             0.0              0.0   0.5  59.4  39.5        0.0        0.0        0.0  0.0     0.2       0.4   0.1
N2                  HC          ren2023  12657             0.0              0.0   0.3  78.2  21.0        0.0        0.0        0.0  0.0     0.0       0.4   0.0
P2          CTCL_other          ren2023  12358            47.6             56.0   0.8  86.7  11.9        0.0        0.0        0.0  0.0     0.2       0.3   0.0
P4          CTCL_other          ren2023  12091            60.3             64.2   0.0  92.5   6.1        0.0        0.3        0.0  0.0     0.0       0.2   0.9
P6          CTCL_other          ren2023  12077             0.0             88.2   0.0  93.3   6.6        0.0        0.
... [18761 more chars truncated]
```

```python  # cell 42
# ============================================================================
# §15  Final UMAPs   (10b §12.6)
# ============================================================================
sync_view(adv, ad, "cell_type_final", "cell_type_ct12", "sezary_like")

print(f"blood view {adv.n_obs:,}/{ad.n_obs:,} | T/NK view {adtv.n_obs:,}/{adt.n_obs:,}")
save_umap(adv, ["cell_type_final"], "umap_celltype_v2.png")
save_umap(adtv, ["cell_type_T"], "umap_T_celltype_v2.png", size=3)
# 10b §12 plots GATA3/CCR4/CXCL13/TCF7/SELL/GZMB/GNLY/KIR3DL2 on its T view; same set + the
# Sezary LOSS pair (CD7/DPP4). Genes stay a GRID — the point is reading them against each other.
save_umap_grid(adtv, ["GATA3", "CCR4", "CXCL13", "TCF7", "SELL", "GZMB", "GNLY", "KIR3DL2",
                      "CADM1", "TOX", "CD7", "DPP4", "FOXP3"], "umap_T_markers_v2.png")
```

```
<Figure size 704x528 with 1 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/blood_reannotation/umap_celltype_final_v2.png

<Figure size 704x528 with 1 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/blood_reannotation/umap_celltype_ct12_v2.png

<Figure size 704x528 with 1 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/blood_reannotation/umap_T_celltype_v2.png

<Figure size 3520x2112 with 26 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/blood_reannotation/umap_T_markers_v2.png
```

## 16 — Write the labels, the T object, and the provenance record  ⚠️ HEAVY

`blood_cell_type_final.csv` is what `32_atlas_descriptive` §1 gates on: it joins the file only if
it covers >99 % of the v2 blood cells, so a stale v1 sidecar is rejected rather than half-joined.
Once this cell has run, re-run `12_atlas_descriptive` §1 (delete `atlas_obs_full_v2.parquet` first) and blood stops
being reported as unannotated.

```python  # cell 44
# ============================================================================
# §16  Write labels + T object + provenance   (HEAVY)
# ============================================================================
assert LATENT_SCOPE == "blood" or ALLOW_JOINTLAT_LABELS, (
    "refusing to write the canonical blood_cell_type_final.csv from the joint-atlas latent; "
    "set ALLOW_JOINTLAT_LABELS=True if that is really intended")
assert ad.n_obs == N_BLOOD or EXCLUDE_SAMPLES, (
    f"about to write {ad.n_obs:,} labels but v2 blood is {N_BLOOD:,} and no samples were "
    "excluded — nb32 will reject a sidecar that does not cover the compartment")

lab = pd.DataFrame({
    "cell_id": ad.obs["cell_id"].astype(str).to_numpy(),
    "cell_type_broad": ad.obs["cell_type_broad"].astype(str).to_numpy(),
    "cell_type_final": ad.obs["cell_type_final"].astype(str).to_numpy(),
    "cell_type_ct12": ad.obs["cell_type_ct12"].astype(str).to_numpy(),
    "sezary_like": ad.obs["sezary_like"].to_numpy(),
    "leiden_broad": ad.obs["leiden_broad"].astype(str).to_numpy(),
})
_lt = adt.obs.set_index(adt.obs["cell_id"].astype(str))["leiden_T"].astype(str)
lab["leiden_T"] = _lt.reindex(lab["cell_id"]).to_numpy()
assert lab["cell_id"].is_unique and len(lab) == ad.n_obs
lab.to_csv(CT_FINAL_CSV, index=False)
print(f"wrote {CT_FINAL_CSV.name}: {len(lab):,} rows x {lab.shape[1]} cols "
      f"(v1 had 423,042 — nb32 gates on this row set covering v2's {N_BLOOD:,} blood cells)")

for _c in adt.obs.columns:
    if adt.obs[_c].dtype == object:
        adt.obs[_c] = adt.obs[_c].astype(str)
adt.write_h5ad(T_ANNOT)
print(f"wrote {T_ANNOT.name}: {adt.shape} ({T_ANNOT.stat().st_size / 1e9:.1f} GB)")
print("X log1p-normalized:", float(adt.X.max()) < 20,
      "| counts layer raw:", float(adt.layers["counts"].max()) > 30)

PROV_JSON.write_text(json.dumps({
    "notebook": "33_blood_reannotation.ipynb",
    "atlas_build": "v2", "cache_v": CACHE_V,
    "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "latent_scope": LATENT_SCOPE, "include_ln": INCLUDE_LN, "full_genes": FULL_GENES,
    "source_h5ad": SRC_H5AD.name, "latent_u": MRVI_U.name, "latent_z": MRVI_Z.name,
    "sample_scope": SAMPLE_SCOPE, "excluded_samples": sorted(EXCLUDE_SAMPLES),
    "no_vdj_samples": NO_VDJ_SAMPLES, "seed": SEED, "plot_n": PLOT_N,
    "n_cells": int(ad.n_obs), "n_genes": int(ad.n_vars), "n_t_nk": int(adt.n_obs),
    "n_samples": int(ad.obs["sample_id"].nunique()),
    "n_patients": int(ad.obs["patient_key"].nunique()),
    "leiden_res": {"broad": LEIDEN_RES, "T": T_LEIDEN_RES, "unk": UNK_LEIDEN_RES},
    "t_subset_labels": sorted(T_SUBSET_LABELS),
    "vocab_broad": sorted(VOCAB_BROAD), "vocab_T": sorted(VOCAB_T),
    "sezary_vote": {"min_tcr": MIN_TCR_TO_SCORE, "frac": SEZARY_FRAC},
    "cluster2ct": cluster2ct, "cluster2ct_T": cluster2ct_T, "cluster2ct_unk": cluster2ct_unk,
    "cluster_sizes_broad": CLUSTER_SIZES_BROAD, "cluster_sizes_T": CLUSTER_SIZES_T,
    "cluster_sizes_unk": CLUSTER_SIZES_UNK,
    "label_counts": {k: int(v) for k, v in
                     ad.obs["cell_type_final"].value_counts().to_dict().items()},
}, indent=1))
print("wrote", PROV_JSON.name)

for _p in (SCOPE_CSV, CLUSTER_QC_CSV, T_CLUSTER_QC_CSV, CT_FINAL_CSV):
    if _p.exists():
        (TAB / _p.name).write_bytes(_p.read_bytes())
print("mirrored the CSV tables into", TAB)
```

```
wrote blood_cell_type_final.csv: 811,024 rows x 7 cols (v1 had 423,042 — nb32 gates on this row set covering v2's 811,024 blood cells)
wrote blood_T_annotated.h5ad: (667584, 10000) (16.4 GB)
X log1p-normalized: True | counts layer raw: True
wrote blood_annotation_provenance.json
mirrored the CSV tables into /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/tables
```

## 17 — How `12_atlas_descriptive` consumes this

`10_atlas/12_atlas_descriptive.ipynb` §1 already contains the join; nothing there needs editing. It reads
`blood_cell_type_final.csv`, checks that its `cell_id`s cover **>99 % of the v2 blood cells**, and
either concatenates it with `skin_cell_type_final.csv` or prints why it was skipped. The v1
sidecar (423,042 rows) fails that check, which is why blood currently shows as
`NONE (unannotated)`.

To pick these labels up:

```
rm data/atlas_joint/atlas_obs_full_v2.parquet     # force `12_atlas_descriptive` §1 to rebuild
# re-run 10_atlas/12_atlas_descriptive.ipynb
```

`label_source` then gains `reannotated (`31_reannotation`)`, `NONE (unannotated)` shrinks to the 1,142 LN
cells, and the §3 overview UMAP colours blood by cell type instead of grey.

Because §0 uses 10b's spellings (`CD4_Treg`, `Unk`, `Myeloid`), the two sidecars share every level
they have in common and `12_atlas_descriptive` needs no remapping table. `12_atlas_descriptive`'s `CT_H_ORDER` already lists the
blood-only levels (`gdT`, `MAIT`, `NK`, `Mono_CD14`, `Mono_CD16`, `cDC`, `pDC`, `Platelet`,
`Erythroid`, `HSPC`, `Prolif`, `LowQC`, `Doublet`), so no level falls outside the categorical.

### The scientific payoff

v2 has **75 blood patients** against v1's 49 samples, and the patients with paired blood *and*
skin are listed by `12_atlas_descriptive` §2c. Those are the substrate for the first within-patient blood-vs-skin
tumour-state comparison in this atlas — impossible while blood had no cell types.

Downstream consumers of `blood_T_annotated.h5ad`: `32_malignancy_tcr_cnv` (blood CNV / malignancy), `33_subclone_tcr_signaling` (blood
subclone + TCR signaling), `13_skin_vs_blood_subclone_signaling` (skin vs blood subclone signaling). All three were built against
the v1 object and will need their own cache-version bump — the same trap documented for `23_malignancy_tcr_cnv` v5.


