# 20_skin/21_reannotation.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/20_skin/21_reannotation.ipynb` and is never run._

# 21 — Skin re-annotation on the **v2** atlas

Re-runs the **`old/13_skin_lowres_annotation`** annotation *manner* on the retrained skin MrVI, now over the **v2 atlas**
(21 cohorts, 2,157,693 cells → **1,345,527 skin**, 174 samples / 14 studies; v1 was ~749k skin):
log-normalize → neighbors/UMAP/Leiden on `X_mrvi_u` → marker dot plot → cluster→cell-type map →
T-cell sub-clustering → merge → `cell_type_final`. Supersedes `13_skin_lowres_annotation` / `13n_*`
for the v2 atlas; writes the canonical caches (`skin_leiden_broad.csv`,
`skin_cell_type_final.csv`, `skin_T_annotated.h5ad`) so `old/13n_skin_T_reannotation` / `old/14_skin_T_tcr_cnv_malignancy` / `old/21_tcr_alice_neighborhood` / `old/23_subclonal_evolution` pick up the new
labels.

**Input is `joint_mrvi_input_skin.h5ad`, not `joint_annotated.h5ad`.** The annotated atlas is 112 GB
(CSR float64/int64, nnz 3.4e9, plus a raw_counts layer identical to X) and OOM-kills the kernel on
read. The MrVI HVG input is the same Skin slice in the same row order — verified cell-id-identical —
so it stays aligned with the latents; §1 additionally asserts the v2-only cohorts are present, since
`run_build_joint.py` rebuilds these files **in place** and the path alone doesn't identify the build.

**Cluster IDs renumber** when the atlas + MrVI change, so neither `old/13_skin_lowres_annotation`'s nor v1's hardcoded maps can
be copied — the v1 maps are kept below each cell as commented reference only. Read the dot plots,
then hand-fill `cluster2ct` (broad), `cluster2ct_T` (T subtypes) and `cluster2ct_unk`. The full skin
annotation is then the broad label with T cells replaced by their subtype. Run on the GPU kernel.

```python  # cell 1
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from anndata.io import read_elem, sparse_dataset


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

SEED = 0
np.random.seed(SEED)
sc.settings.verbosity = 1
OUT = NB_DIR / "data" / "atlas_joint"

# Do NOT sc.read_h5ad(joint_annotated.h5ad) for the clustering pass — 112 GB on disk: CSR float64
# data + int64 indices (nnz 3.4e9) is ~55 GB, and it carries a raw_counts layer identical to X, so a
# full read is ~110 GB resident before the Skin slice is even taken. That is what OOM-kills the
# kernel. SKIN_INPUT is the same Skin slice MrVI trained on — verified identical cell_id order,
# 1,345,527 rows — HVG-subset to 10k genes, ~5 GB as float32. §13 streams the full gene space back
# out of the atlas for the T cells only, which is what inferCNV (nb30) needs.
SKIN_INPUT = OUT / "joint_mrvi_input_skin.h5ad"
ANNOT = OUT / "joint_annotated.h5ad"       # full 42,347-gene space — streamed row-wise in §13 only
MRVI_U = OUT / "joint_X_mrvi_u_skin.npy"   # sample-unaware (integration view) -> neighbors/UMAP/Leiden
MRVI_Z = OUT / "joint_X_mrvi_skin.npy"     # sample-aware (kept for later substructure work)
LEIDEN_CSV = OUT / "skin_leiden_broad.csv"
T_LEIDEN_CSV = OUT / "skin_T_leiden.csv"
UNK_LEIDEN_CSV = OUT / "skin_T_unk_leiden.csv"
T_ANNOT = OUT / "skin_T_annotated.h5ad"
CT_FINAL_CSV = OUT / "skin_cell_type_final.csv"   # full-skin merged labels (broad + T subtype)

# UMAP is a *visualization artifact only* — Leiden and every label run on all cells. umap-learn is
# CPU-only here (no cuml/rapids_singlecell in this env), and 1.35M points costs hours to embed and
# is an unreadable blob when plotted. So embed a random subsample instead; see plot_view() in §3.
PLOT_N = 150_000
# igraph's Leiden is single-threaded but linear here: measured 4.0 s @100k, 7.1 s @200k, 14.6 s
# @400k on this latent (~36 s per million), so the full 1.35M skin graph is ~50 s. Exact clustering
# is therefore the default; drop LEIDEN_MAX_EXACT below a compartment's size to switch it to the
# seeded + kNN-propagated path in §4 (see leiden_labels).
LEIDEN_MAX_EXACT = 2_000_000
LEIDEN_SEED_N = 250_000

# --- v2 atlas contract -------------------------------------------------------------------
# Skin cohorts that exist only in the v2 build (data/append_v2_metadata.py: D7-D11, D13-D15).
# Their presence is what distinguishes v2 from the v1 (buus-extended) atlas at the same paths.
V2_SKIN_DATASETS = {"rindler21mc", "alkon24", "lyp26", "jonak21",
                    "brentuximab26", "gaydosik22", "gaydosik23", "il4ra26"}
V2_N_OBS_SKIN = 1_345_527   # v1 skin was ~749k -> every cache below is invalidated by row count


def _mtime(p):
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if p.exists() else "MISSING"


# --- staleness: fingerprint the build, not the mtime --------------------------------------
# mtime is not a staleness signal in this tree. jobs/apply_atlas_v2_fixups.py rewrites obs in
# place (it ran 2026-09-15), which restamps every h5ad without touching one matrix byte -- so the
# old `MRVI_U newer than SKIN_INPUT` assert failed on latents that are perfectly valid. Hash what
# MrVI actually consumed instead: shape + gene names + the whole indptr. Same construction as
# fingerprint() in 11_atlas_v2_qc §1; obs is deliberately not part of it, which is the point.
MRVI_PROV = OUT / "joint_X_mrvi_skin.provenance.json"


def build_fingerprint(p):
    """Identify an h5ad build from its matrix structure alone. Reads no matrix values.

    var_names are joined as bytes, not `.tobytes()`-ed: h5py hands back a variable-length string
    array whose numpy buffer holds *pointers*, so .tobytes() hashes addresses and the digest
    changes every process. (The same construction in 11_atlas_v2_qc §1 has that bug; it only ever
    displays the digest, so nothing there compares two runs.)
    """
    h = hashlib.blake2b(digest_size=16)
    with h5py.File(p, "r") as f:
        h.update(np.asarray(f["X"].attrs["shape"], dtype=np.int64).tobytes())
        names = f["var"][f["var"].attrs["_index"]][:]
        h.update(b"\x00".join(x if isinstance(x, bytes) else str(x).encode() for x in names))
        h.update(np.ascontiguousarray(f["X"]["indptr"][:]).tobytes())
    return h.hexdigest()


print("NB_DIR:", NB_DIR)
print(f"skin input {SKIN_INPUT.name}  mtime={_mtime(SKIN_INPUT)}  ({SKIN_INPUT.stat().st_size / 1e9:.1f} GB)")
print(f"latents    {MRVI_U.name}  mtime={_mtime(MRVI_U)}  |  {MRVI_Z.name}  mtime={_mtime(MRVI_Z)}")

_fp = build_fingerprint(SKIN_INPUT)
if MRVI_PROV.exists():
    _rec = json.loads(MRVI_PROV.read_text())
    assert _rec["skin_input_fp"] == _fp, (
        f"{SKIN_INPUT.name} is NOT the build the skin MrVI latents were trained on\n"
        f"  recorded: {_rec['skin_input_fp']}  ({_rec.get('recorded_utc')})\n"
        f"  now     : {_fp}\n"
        "rerun jobs/run_mrvi_joint.sh with --subset_compartment Skin --force")
    print(f"build fp   {_fp}  matches {MRVI_PROV.name}  (recorded {_rec.get('recorded_utc')})")
else:
    MRVI_PROV.write_text(json.dumps(
        {"skin_input_fp": _fp,
         "n_obs": V2_N_OBS_SKIN,
         "latent_rows": int(np.load(MRVI_U, mmap_mode="r").shape[0]),
         "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "note": "obs-only fixups (jobs/apply_atlas_v2_fixups.py) restamp the h5ad without "
                 "changing X -- that is why this gate is a fingerprint and not an mtime"},
        indent=1))
    print(f"build fp   {_fp}  RECORDED -> {MRVI_PROV.name}  (first run; bootstraps the gate)")
```

```
NB_DIR: /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
skin input joint_mrvi_input_skin.h5ad  mtime=2026-09-05 22:12  (16.4 GB)
latents    joint_X_mrvi_u_skin.npy  mtime=2026-09-06 00:24  |  joint_X_mrvi_skin.npy  mtime=2026-09-06 00:25
```

## 1 — Load the skin MrVI input + skin MrVI latents

Reads `joint_mrvi_input_skin.h5ad` (1,345,527 × 10k HVG), **not** `joint_annotated.h5ad` — the full
atlas is 112 GB and OOM-kills the kernel. Same rows, same order, same build; the only cost is the
gene space (10k HVGs instead of 42,347, which drops `RORC` from the §10 Th17 panel).

**Staleness is checked by fingerprint, not mtime.** `jobs/apply_atlas_v2_fixups.py` patches `obs`
in place across every v2 object, which restamps the h5ad without changing a single matrix byte —
so an mtime comparison flags valid latents as stale. §1 instead hashes what MrVI actually
consumed (shape + gene names + the whole `X/indptr`) and compares it against
`joint_X_mrvi_skin.provenance.json`, written next to the latents on first run.

```python  # cell 3
def read_h5ad_x_obs_var(path):
    """X + obs + var only, CSR cast to float32 during the HDF5 read.

    Skips the raw_counts layer (byte-identical to X here — both are raw integer counts) and the
    stale obsm carried over from the source atlas. float64 -> float32 halves the data array, so the
    1,345,527 x 10,000 skin matrix lands at ~5 GB instead of ~15 GB with the layer.
    """
    with h5py.File(path, "r") as f:
        g = f["X"]
        assert g.attrs["encoding-type"] == "csr_matrix", g.attrs["encoding-type"]
        shape = tuple(int(x) for x in g.attrs["shape"])
        data = np.empty(g["data"].shape[0], dtype=np.float32)
        g["data"].read_direct(data)          # HDF5 does the float64->float32 cast on the way out
        X = sp.csr_matrix((data, g["indices"][:], g["indptr"][:]), shape=shape)
        X.sort_indices()                     # stored unsorted; scanpy slicing/plotting wants sorted
        return AnnData(X=X, obs=read_elem(f["obs"]), var=read_elem(f["var"]))


ad = read_h5ad_x_obs_var(SKIN_INPUT)

missing = V2_SKIN_DATASETS - set(ad.obs["dataset"].astype(str).unique())
assert not missing, f"v2 skin cohorts absent — this is the v1 build. missing: {sorted(missing)}"
assert ad.n_obs == V2_N_OBS_SKIN, f"skin input is {ad.n_obs} rows, expected v2's {V2_N_OBS_SKIN}"
assert "total_counts" in ad.obs, "obs.total_counts missing — §2 needs it to normalize over all genes"

ad.obsm["X_mrvi_u"] = np.load(MRVI_U)
ad.obsm["X_mrvi_z"] = np.load(MRVI_Z)
assert ad.n_obs == ad.obsm["X_mrvi_u"].shape[0], (
    f"skin input ({ad.n_obs}) != skin latent rows ({ad.obsm['X_mrvi_u'].shape[0]}) — "
    "stale skin npy vs joint_mrvi_input_skin.h5ad; rerun the skin MrVI with --force")
print(ad)
print("X_mrvi_u", ad.obsm["X_mrvi_u"].shape, "| X_mrvi_z", ad.obsm["X_mrvi_z"].shape)
print(f"\nv2 skin confirmed: {ad.n_obs} cells, {ad.obs['sample_id'].nunique()} samples, "
      f"{ad.obs['study'].nunique()} studies, {ad.n_vars} HVGs")
print("datasets in skin:", ad.obs["dataset"].value_counts().to_dict())
print("v2-only skin cohorts:",
      {d: int((ad.obs["dataset"].astype(str) == d).sum()) for d in sorted(V2_SKIN_DATASETS)})
```

```
AnnData object with n_obs × n_vars = 1345527 × 10000
    obs: 'cell_id', 'dataset', 'study', 'donor', 'real_donor', 'sample_id', 'sex', 'disease', 'disease_stage', 'compartment', 'tissue', 'tech', 'cell_type_orig', 'cell_type', 'n_genes', 'total_counts', 'pct_mito', 'doublet_score', 'predicted_doublet', 'patient_key', 'has_tcr', 'clone_id', 'tra_cdr3', 'trb_cdr3', 'clone_size', 'is_expanded', 'is_dominant_clone', 'is_malignant', 'repo', 'accession', 'organ', 'tissue_detail', 'malignant_call_method', 'tcr_available', 'malignant_labeled', 'malignant_frac', 'tcr_recovery', 'stage_class', 'stage_clean', 'lineage', 'entity', 'fmf_status', 'lineage_resolve', 'treatment_context', 'blood_involvement', 'lesion_type', 'cohort_country', 'cohort_region', 'origin_resolve', 'east_asian', 'lineage_orig', 'lineage_score_cd4', 'lineage_score_cd8', 'lineage_score_gd', 'lineage_score_ab', 'lineage_method', 'lineage_disagree', 'lineage_disagreement'
    var: 'highly_variable', 'highly_variable_rank', 'means', 'variances', 'variances_norm', 'highly_variable_nbatches'
    obsm: 'X_mrvi_u', 'X_mrvi_z'
X_mrvi_u (1345527, 10) | X_mrvi_z (1345527, 30)

v2 skin confirmed: 1345527 cells, 174 samples, 14 studies, 10000 HVGs
datasets in skin: {'li24': 419579, 'chennareddy25': 191147, 'brentuximab26': 133677, 'alkon24': 113307, 'rindler21mc': 104721, 'lyp26': 87300, 'il4ra26': 75834, 'gaydosik22': 60483, 'buus25': 55483, 'gaydosik19': 39260, 'brunner24': 37883, 'gaydosik23': 20630, 'jonak21': 4353, 'herrera21': 1870}
v2-only skin cohorts: {'alkon24': 113307, 'brentuximab26': 133677, 'gaydosik22': 60483, 'gaydosik23': 20630, 'il4ra26': 75834, 'jonak21': 4353, 'lyp26': 87300, 'rindler21mc': 104721}
```

## 2 — Log-normalize for marker viz

Scaled by `obs.total_counts` (the **full 42,347-gene** library size, verified equal to the row sums
of the annotated atlas), *not* by the row sums of this 10k-HVG matrix — that reproduces exactly what
`sc.pp.normalize_total(target_sum=1e4)` gave on the full atlas. In place on the CSR buffer, and with
**no `counts` layer**: §13 re-streams raw counts from the atlas for the cells that need them, so a
second copy here would only double every downstream `.copy()`.

```python  # cell 5
assert float(ad.X.data.max()) > 30 and np.allclose(ad.X.data[:10_000], np.round(ad.X.data[:10_000])), \
    "X does not look like raw counts — §2 must not run twice"

# No counts layer: §13 re-streams raw counts for the T cells straight from the atlas, so keeping a
# second 5 GB copy here only slows every .copy() downstream (adt/adu each duplicate every layer).
sf = (1e4 / ad.obs["total_counts"].to_numpy()).astype(np.float32)   # full-gene library size
ad.X.data *= np.repeat(sf, np.diff(ad.X.indptr))
np.log1p(ad.X.data, out=ad.X.data)

print("X is log1p-normalized:", float(ad.X.data.max()) < 20, "| dtype:", ad.X.dtype)
print(f"X in memory: {(ad.X.data.nbytes + ad.X.indices.nbytes + ad.X.indptr.nbytes) / 1e9:.1f} GB")
```

```
X is log1p-normalized: True | dtype: float32
X in memory: 5.0 GB
```

## 3 — Subsampled UMAP views (plotting only)

`sc.tl.umap` on 1.35M points costs hours on CPU (no `cuml`/`rapids_singlecell` in this env) and
draws an unreadable blob. `plot_view()` embeds a random `PLOT_N`-cell subsample instead, with its
own neighbors graph, cached per subset. Labels never come from it — Leiden in §4 runs on the full
graph, and every crosstab/purity number below is computed on all cells.

```python  # cell 7
def plot_view(a, key, n=PLOT_N, seed=SEED):
    """A subsampled copy of `a` carrying a UMAP, for plotting only.

    Never use the result to compute labels — it is a random subset. The embedding caches as
    skin_umap_<key>_<parent_n>_<sub_n>.npy; the parent row count is in the name so a re-annotation
    that changes the subset (e.g. more T cells) cannot silently reuse the old embedding. Cells are
    drawn uniformly: at PLOT_N=150k out of 1.35M (11%) a cluster holding 0.1% of cells still keeps
    ~150 points, which is enough to see it.
    """
    if a.n_obs <= n:
        sub, idx = a.copy(), np.arange(a.n_obs)
    else:
        idx = np.sort(np.random.default_rng(seed).choice(a.n_obs, n, replace=False))
        sub = a[idx].copy()
    cache = OUT / f"skin_umap_{key}_{a.n_obs}_{sub.n_obs}.npy"
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


print("plot_view ready — the full-graph neighbors/UMAP are never built on all cells")
```

```
plot_view ready — the full-graph neighbors/UMAP are never built on all cells
```

## 4 — Low-resolution Leiden (broad lineages)

`leiden_labels()` builds the neighbor graph if it isn't there yet and clusters **all** cells.
Timings measured on this latent: leiden itself is linear at ~36 s per million (4.0 s @100k,
7.1 s @200k, 14.6 s @400k), so 1.35M is **~50 s** plus a few minutes for `sc.pp.neighbors` — if it
seems stuck, it is probably nearly done. Cluster count is stable across scales (20 @100k, 21 @200k,
24 @400k), so the structure is not an artifact of size.

If it does stall, lower `LEIDEN_MAX_EXACT` in §1 to switch to the seeded path: cluster a
`LEIDEN_SEED_N` subsample, then propagate by kNN vote in `X_mrvi_u`, with hold-out agreement
printed so you can see what the shortcut cost.

```python  # cell 9
LEIDEN_RES = 0.7


def leiden_labels(a, res, key, seed=SEED, max_exact=LEIDEN_MAX_EXACT, n_seed=LEIDEN_SEED_N):
    """Leiden labels for every cell of `a`, written to a.obs[key]. Reports its own timings.

    Measured on this latent (igraph flavor, single-threaded): 4.0 s @100k, 7.1 s @200k, 14.6 s
    @400k — linear, ~36 s per million, so the full 1.35M skin graph is ~50 s and 20-24 clusters.
    If it runs much longer than that, the cell is not stuck on the algorithm; check the kernel's
    memory headroom (bjobs -o "max_mem memlimit") before assuming it hung.

    Above `max_exact` cells the seeded path clusters an `n_seed` subsample on its own graph and
    propagates by kNN vote in X_mrvi_u — the same 10-d space the graph is built from. It prints
    hold-out agreement so the approximation is auditable. Exact is the default; lower
    LEIDEN_MAX_EXACT in §1 to force the shortcut.
    """
    t0 = time.time()
    if a.n_obs <= max_exact:
        if "neighbors" not in a.uns:
            sc.pp.neighbors(a, use_rep="X_mrvi_u", random_state=seed)
            print(f"  neighbors on {a.n_obs:,} cells: {time.time() - t0:.0f}s")
        t = time.time()
        sc.tl.leiden(a, resolution=res, random_state=seed, key_added=key,
                     flavor="igraph", n_iterations=2, directed=False)
        print(f"leiden[{key}]: exact on {a.n_obs:,} cells -> {a.obs[key].nunique()} clusters "
              f"({time.time() - t:.0f}s leiden, {time.time() - t0:.0f}s total)")
        return a.obs[key]

    from sklearn.neighbors import KNeighborsClassifier

    rng = np.random.default_rng(seed)
    U = a.obsm["X_mrvi_u"]
    idx = np.sort(rng.choice(a.n_obs, n_seed, replace=False))
    seed_ad = AnnData(obs=pd.DataFrame(index=idx.astype(str)), obsm={"X_mrvi_u": U[idx]})
    sc.pp.neighbors(seed_ad, use_rep="X_mrvi_u", random_state=seed)
    sc.tl.leiden(seed_ad, resolution=res, random_state=seed, key_added=key,
                 flavor="igraph", n_iterations=2, directed=False)
    cats = seed_ad.obs[key].cat.categories
    lab = seed_ad.obs[key].to_numpy()

    hold = rng.permutation(n_seed)[: max(1, n_seed // 10)]        # audit the propagation
    train = np.setdiff1d(np.arange(n_seed), hold)
    probe = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)
    probe.fit(U[idx[train]], lab[train])
    acc = float((probe.predict(U[idx[hold]]) == lab[hold]).mean())

    knn = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1).fit(U[idx], lab)
    out = np.empty(a.n_obs, dtype=object)
    out[idx] = lab
    rest = np.setdiff1d(np.arange(a.n_obs), idx)
    for s in range(0, rest.size, 250_000):
        blk = rest[s : s + 250_000]
        out[blk] = knn.predict(U[blk])
        print(f"  propagated {min(s + 250_000, rest.size):,}/{rest.size:,}", end="\r")
    a.obs[key] = pd.Categorical(out.astype(str), categories=cats)
    print(f"\nleiden[{key}]: {len(cats)} clusters from a {n_seed:,}-cell seed, propagated to "
          f"{a.n_obs:,} | hold-out agreement {acc:.3f} ({time.time() - t0:.0f}s total)")
    return a.obs[key]


leiden_labels(ad, LEIDEN_RES, "leiden_broad")
pd.DataFrame({"cell_id": ad.obs["cell_id"].astype(str).to_numpy(),
              "leiden_broad": ad.obs["leiden_broad"].astype(str).to_numpy()}) \
    .to_csv(LEIDEN_CSV, index=False)
print(f"Leiden (res={LEIDEN_RES}) cached ->", LEIDEN_CSV)
print(ad.obs["leiden_broad"].value_counts().sort_index().to_dict())
```

```
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm

  neighbors on 1,345,527 cells: 204s
leiden[leiden_broad]: exact on 1,345,527 cells -> 26 clusters (69s leiden, 273s total)
Leiden (res=0.7) cached -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/skin_leiden_broad.csv
{'0': 155987, '1': 15206, '2': 212, '3': 20468, '4': 32, '5': 38280, '6': 67742, '7': 15422, '8': 41414, '9': 105589, '10': 8470, '11': 12925, '12': 14205, '13': 26329, '14': 187463, '15': 195505, '16': 156948, '17': 28610, '18': 5389, '19': 143006, '20': 3695, '21': 12091, '22': 2194, '23': 26, '24': 8315, '25': 80004}
```

## 5 — UMAP overview

```python  # cell 11
adv = plot_view(ad, "broad")
sc.pl.umap(adv, color=["leiden_broad", "cell_type", "study", "disease"],
           ncols=2, frameon=False, size=3, wspace=0.25)
```

```
plot_view[broad]: 150,000/1,345,527 cells, cached UMAP

<Figure size 1600x960 with 4 Axes>
```

## 6 — Marker dot plot (broad lineages)

```python  # cell 13
markers = {
    "T": ["CD3D", "CD3E", "TRAC"],
    "CD4": ["CD4", "IL7R"],
    "CD8/NK": ["CD8A", "GZMB", "NKG7", "GNLY"],
    "Treg": ["FOXP3", "CTLA4"],
    "B": ["MS4A1", "CD79A", "CD19"],
    "Plasma": ["MZB1", "JCHAIN", "IGHG1"],
    "Myeloid/DC": ["LYZ", "CD68", "ITGAX", "CD14"],
    "Mast": ["TPSAB1", "CPA3", "KIT"],
    "Keratinocyte": ["KRT14", "KRT5", "KRT1"],
    "Fibroblast": ["COL1A1", "DCN", "LUM"],
    "Vascular endo": ["PECAM1", "VWF", "CLDN5"],
    "Lymphatic endo": ["LYVE1", "PROX1"],
    "Melanocyte": ["MLANA", "PMEL", "TYR"],
}
present = set(ad.var_names)
markers = {k: [g for g in v if g in present] for k, v in markers.items()}
markers = {k: v for k, v in markers.items() if v}
sc.pl.dotplot(ad, markers, groupby="leiden_broad", standard_scale="var", dendrogram=True)
```

```
WARNING: dendrogram data not found (using key=dendrogram_leiden_broad). Running `sc.tl.dendrogram` with default parameters. For fine tuning it is recommended to run `sc.tl.dendrogram` independently.

/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/scanpy/tools/_dendrogram.py:138: UserWarning: You’re trying to run this on 10000 dimensions of `.X`, if you really want this, set `use_rep='X'`.
         Falling back to preprocessing with `sc.pp.pca` and default params.
  _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)

WARNING: Groups are not reordered because the `groupby` categories and the `var_group_labels` are different.
categories: 0, 1, 2, etc.
var_group_labels: T, CD4, CD8/NK, etc.

<Figure size 1636x1010 with 6 Axes>
```

## 7 — Map clusters → `cell_type_broad`  (manual)

**Read the §6 dot plot**, then fill `cluster2ct` (`{"<cluster>": "<label>"}`) for every
`leiden_broad` cluster. Run once with `cluster2ct = {}` to print the template for the current
cluster count. The v1 (749k-skin) map is kept below as reference only — the v2 atlas is 1.8× larger
and its cluster IDs do **not** correspond.

```python  # cell 15
# --- FILL from the §6 dot plot. Run once with {} to print the leiden_broad template. ---
cluster2ct = {"0":"Keratinocyte",
 "1":"Keratinocyte",
 "2":"Keratinocyte",
 "3":"Keratinocyte",
 "4":"Fibroblast",
 "5":"Unk",
 "6":"Keratinocyte",
 "7":"Melanocyte",
 "8":"Fibroblast",
 "9":"Myeloid",
 "10":"Vascular",
 "11":"Keratinocyte",
 "12":"Vascular",
 "13":"Myeloid",
 "14":"T",
 "15":"T",
 "16":"T",
 "17":"B",
 "18":"Plasma",
 "19":"Fibroblast",
 "20":"Mast",
 "21":"Myeloid",
 "22":"Vascular",
 "23":"Keratinocyte",
 "24":"Plasma",
 "25":"Vascular",}




if not cluster2ct or any(v == "" for v in cluster2ct.values()):
    print("cluster2ct unfilled — read the §6 dot plot, then fill this template:")
    print({c: "" for c in ad.obs["leiden_broad"].cat.categories})
    print("cluster sizes:", ad.obs["leiden_broad"].value_counts().sort_index().to_dict())
else:
    assert set(cluster2ct) == set(ad.obs["leiden_broad"].cat.categories), \
        "keys must match leiden_broad clusters: " + str(sorted(ad.obs["leiden_broad"].cat.categories))
    ad.obs["cell_type_broad"] = ad.obs["leiden_broad"].map(cluster2ct).astype("category")
    print(ad.obs["cell_type_broad"].value_counts(dropna=False).to_string())
```

```
cell_type_broad
T               539916
Keratinocyte    272566
Fibroblast      184452
Myeloid         144009
Vascular        104873
Unk              38280
B                28610
Melanocyte       15422
Plasma           13704
Mast              3695
```

```python  # cell 16
sc.pl.umap(adv, color=["cell_type_broad", "cell_type", "study", "disease"],
           ncols=2, frameon=False, size=3, wspace=0.25)
```

```
<Figure size 1600x960 with 4 Axes>
```

# T-cell sub-clustering

## 9 — Subset T cells; neighbors + UMAP + Leiden on the T subset

```python  # cell 19
adt = ad[ad.obs["cell_type_broad"] == "T"].copy()
print("T cells:", f"{adt.n_obs:,}")
print("T per dataset:", adt.obs["dataset"].value_counts().to_dict())

T_LEIDEN_RES = 1.0
leiden_labels(adt, T_LEIDEN_RES, "leiden_T")
pd.DataFrame({"cell_id": adt.obs["cell_id"].astype(str).to_numpy(),
              "leiden_T": adt.obs["leiden_T"].astype(str).to_numpy()}).to_csv(T_LEIDEN_CSV, index=False)
print(f"T Leiden (res={T_LEIDEN_RES}) cached ->", T_LEIDEN_CSV)
print(adt.obs["leiden_T"].value_counts().sort_index().to_dict())
```

```
T cells: 539,916
T per dataset: {'li24': 233430, 'chennareddy25': 83679, 'brentuximab26': 40964, 'lyp26': 36899, 'buus25': 36576, 'rindler21mc': 28555, 'il4ra26': 26049, 'alkon24': 18927, 'gaydosik19': 10925, 'gaydosik22': 10010, 'brunner24': 9132, 'jonak21': 2799, 'gaydosik23': 1147, 'herrera21': 824}
leiden[leiden_T]: exact on 539,916 cells -> 18 clusters (22s leiden, 22s total)
T Leiden (res=1.0) cached -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/skin_T_leiden.csv
{'0': 65537, '1': 2051, '2': 21124, '3': 43153, '4': 31764, '5': 41011, '6': 57983, '7': 23872, '8': 45996, '9': 17241, '10': 58792, '11': 16272, '12': 6139, '13': 66240, '14': 13954, '15': 28446, '16': 295, '17': 46}
```

## 10 — T-cell marker dot plot

```python  # cell 21
markers_T = {
    "T core": ["CD3D", "CD3E", "TRAC"],
    "CD4": ["CD4", "IL7R"],
    "CD8": ["CD8A", "CD8B"],
    "Naive/mem": ["CCR7", "SELL", "TCF7"],
    "Cytotoxic": ["GZMK", "GZMB", "PRF1", "NKG7", "GNLY"],
    "Treg": ["FOXP3", "CTLA4", "IL2RA"],
    "Th2/CTCL": ["GATA3", "CCR4"],
    "Th17/Tc17": ["RORC", "IL17A", "CCR6"],
    "Exhaustion": ["PDCD1", "TOX", "LAG3", "TIGIT", "HAVCR2"],
    "NK": ["NCAM1", "KLRD1", "KLRF1"],
    "Prolif": ["MKI67", "TOP2A"],
}
present = set(adt.var_names)
markers_T = {k: [g for g in v if g in present] for k, v in markers_T.items()}
markers_T = {k: v for k, v in markers_T.items() if v}
sc.pl.dotplot(adt, markers_T, groupby="leiden_T", standard_scale="var", dendrogram=True)
```

```
WARNING: dendrogram data not found (using key=dendrogram_leiden_T). Running `sc.tl.dendrogram` with default parameters. For fine tuning it is recommended to run `sc.tl.dendrogram` independently.
WARNING: Groups are not reordered because the `groupby` categories and the `var_group_labels` are different.
categories: 0, 1, 2, etc.
var_group_labels: T core, CD4, CD8, etc.

<Figure size 1414x730 with 6 Axes>
```

## 11 — Map T clusters → `cell_type_T`  (manual)

Same as §7. Run once with `cluster2ct_T = {}` to print the `leiden_T` template, then fill each
cluster from the §10 dot plot. `old/13_skin_lowres_annotation`'s old-atlas map kept as reference (IDs differ).

```python  # cell 23
# --- FILL from the §10 dot plot. Run once with {} to print the leiden_T template. ---
cluster2ct_T = {
    "0": "CD4",
    "1": "CD4",
    "2": "CD4",
    "3": "CD4",
    "4": "CD4",
    "5": "CD4_Treg",
    "6": "CD4",
    "7": "CD4",
    "8": "CD4",
    "9": "CD4",
    "10": "UNK",
    "11": "CD4",
    "12": "CD4",
    "13": "CD4",
    "14": "CD4",
    "15": "CD4",
    "16": "CD4",
    "17": "CD8"
}


if not cluster2ct_T or any(v == "" for v in cluster2ct_T.values()):
    print("cluster2ct_T unfilled — read the §10 dot plot, then fill this template:")
    print({c: "" for c in adt.obs["leiden_T"].cat.categories})
else:
    assert set(cluster2ct_T) == set(adt.obs["leiden_T"].cat.categories), \
        "keys must match leiden_T clusters: " + str(sorted(adt.obs["leiden_T"].cat.categories))
    adt.obs["cell_type_T"] = adt.obs["leiden_T"].map(cluster2ct_T).astype("category")
    print(adt.obs["cell_type_T"].value_counts(dropna=False).to_string())

    known = adt.obs[adt.obs["cell_type"] != "Unknown"]
    if len(known):
        cmp = pd.crosstab(known["cell_type_T"], known["cell_type"])
        frac = cmp.div(cmp.sum(1).clip(lower=1), axis=0)
        print("\ndominant Li2024 label per cell_type_T (purity):")
        print(pd.DataFrame({"top_li2024": frac.idxmax(1), "purity": frac.max(1).round(2),
                            "n_known": cmp.sum(1)}).to_string())
```

```
cell_type_T
CD4         440067
UNK          58792
CD4_Treg     41011
CD8             46

dominant Li2024 label per cell_type_T (purity):
             top_li2024  purity  n_known
cell_type_T                             
CD4          tumor_cell    0.51   183245
CD4_Treg           Treg    0.72    20141
CD8                  Tc    0.95       44
UNK                  Tc    0.90    30000
```

## 11.5 — Re-cluster the still-unknown T cells

Whatever clusters you labelled unknown in §11 (`UNK_LABELS`) get the same treatment on their own:
neighbors → UMAP → Leiden on `X_mrvi_u` (this subset only) → dot plot → hand-fill `cluster2ct_unk`
→ merged back into `cell_type_T`. Skip cleanly if no unknowns remain.

```python  # cell 25
# labels from §11 that count as "still unknown" -> re-cluster these
UNK_LABELS = {"UNK", "Unknown", "unknown", ""}

assert "cell_type_T" in adt.obs, "fill cluster2ct_T (§11) first"
unk_mask = adt.obs["cell_type_T"].astype(str).isin(UNK_LABELS).to_numpy()
print(f"still-unknown T cells: {int(unk_mask.sum()):,} / {adt.n_obs:,}")

adu = adt[unk_mask].copy()
if adu.n_obs == 0:
    print("no unknowns left — skip §11.5, go to §12.")
else:
    UNK_LEIDEN_RES = 0.5
    leiden_labels(adu, UNK_LEIDEN_RES, "leiden_unk")
    pd.DataFrame({"cell_id": adu.obs["cell_id"].astype(str).to_numpy(),
                  "leiden_unk": adu.obs["leiden_unk"].astype(str).to_numpy()}).to_csv(UNK_LEIDEN_CSV, index=False)
    print(f"UNK Leiden (res={UNK_LEIDEN_RES}) cached ->", UNK_LEIDEN_CSV)
    print(adu.obs["leiden_unk"].value_counts().sort_index().to_dict())
```

```
still-unknown T cells: 58,792 / 539,916
leiden[leiden_unk]: exact on 58,792 cells -> 8 clusters (2s leiden, 2s total)
UNK Leiden (res=0.5) cached -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/skin_T_unk_leiden.csv
{'0': 11432, '1': 3058, '2': 7310, '3': 2685, '4': 10416, '5': 11080, '6': 12581, '7': 230}
```

```python  # cell 26
# dot plot for the unknown subset (reuses the §10 T marker panel)
if adu.n_obs:
    present = set(adu.var_names)
    markers_unk = {k: [g for g in v if g in present] for k, v in markers_T.items()}
    markers_unk = {k: v for k, v in markers_unk.items() if v}
    aduv = plot_view(adu, "unk")
    sc.pl.umap(aduv, color=["leiden_unk", "study", "sample_id"], ncols=3,
               frameon=False, size=4, wspace=0.25, legend_loc="on data")
    sc.pl.dotplot(adu, markers_unk, groupby="leiden_unk", standard_scale="var", dendrogram=True)
```

```
plot_view[unk]: 58,792/58,792 cells, computed UMAP -> skin_umap_unk_58792_58792.npy

<Figure size 2400x480 with 3 Axes>
WARNING: dendrogram data not found (using key=dendrogram_leiden_unk). Running `sc.tl.dendrogram` with default parameters. For fine tuning it is recommended to run `sc.tl.dendrogram` independently.
WARNING: Groups are not reordered because the `groupby` categories and the `var_group_labels` are different.
categories: 0, 1, 2, etc.
var_group_labels: T core, CD4, CD8, etc.

<Figure size 1414x380 with 6 Axes>
```

### Annotate the unknown clusters → merge back into `cell_type_T`

Fill `cluster2ct_unk` from the §11.5 dot plot. Cells left unknown here keep their original label.

```python  # cell 28
# --- FILL from the §11.5 dot plot. Run once with {} to print the leiden_unk template. ---
cluster2ct_unk = {
    "0":"CD8",
    "1":"CD8",
    "2":"CD8",
    "3":"CD4",
    "4":"CD8",
    "5":"CD8", 
    "6":"CD8",
"7":"CD4",
}


if adu.n_obs == 0:
    print("no unknowns — nothing to merge.")
elif not cluster2ct_unk or any(v == "" for v in cluster2ct_unk.values()):
    print("cluster2ct_unk unfilled — read the §11.5 dot plot, then fill this template:")
    print({c: "" for c in adu.obs["leiden_unk"].cat.categories})
else:
    assert set(cluster2ct_unk) == set(adu.obs["leiden_unk"].cat.categories), \
        "keys must match leiden_unk clusters: " + str(sorted(adu.obs["leiden_unk"].cat.categories))
    new_by_id = pd.Series(adu.obs["leiden_unk"].map(cluster2ct_unk).astype(str).to_numpy(),
                          index=adu.obs["cell_id"].astype(str).to_numpy())

    ct = adt.obs["cell_type_T"].astype(str).to_numpy()
    ids = adt.obs["cell_id"].astype(str).to_numpy()
    ct[unk_mask] = new_by_id.reindex(ids[unk_mask]).to_numpy()
    adt.obs["cell_type_T"] = pd.Categorical(ct)
    print("refined cell_type_T:")
    print(adt.obs["cell_type_T"].value_counts(dropna=False).to_string())
```

```
refined cell_type_T:
cell_type_T
CD4         442982
CD8          55923
CD4_Treg     41011
```

## 12 — T-cell UMAP + key CTCL markers

```python  # cell 30
adtv = plot_view(adt, "T")
sc.pl.umap(adtv, color=["cell_type_T", "leiden_T", "study", "sample_id"],
           ncols=2, frameon=False, size=3, wspace=0.25, legend_loc="on data")
sc.pl.umap(adtv, color=["GATA3", "CCR4", "CXCL13", "TCF7", "SELL", "GZMB", "GNLY", "KIR3DL2"],
           ncols=2, frameon=False, size=3, wspace=0.25)
```

```
plot_view[T]: 150,000/539,916 cells, computed UMAP -> skin_umap_T_539916_150000.npy

<Figure size 1600x960 with 4 Axes>
<Figure size 1600x1920 with 16 Axes>
```

# Merge broad + T subtypes

## 12.5 — Full-skin `cell_type_final`

Combine the two annotations: every skin cell keeps its broad label, except T cells, which take their
fine `cell_type_T` subtype. Cached to `skin_cell_type_final.csv` (`cell_id → cell_type_broad`,
`cell_type_final`) for the downstream notebooks.

```python  # cell 32
# Merge: full-skin label = cell_type_broad, with T cells replaced by their cell_type_T subtype.
assert "cell_type_broad" in ad.obs, "fill cluster2ct (§7) first"
assert "cell_type_T" in adt.obs, "fill cluster2ct_T (§11) first"

ids = ad.obs["cell_id"].astype(str).to_numpy()
final = ad.obs["cell_type_broad"].astype(str).to_numpy()
t_sub = adt.obs.set_index(adt.obs["cell_id"].astype(str))["cell_type_T"].astype(str)
is_t = final == "T"
final[is_t] = t_sub.reindex(ids[is_t]).to_numpy()
assert pd.notna(final).all(), "T cells missing a subtype after merge (cell_id mismatch)"

ad.obs["cell_type_final"] = pd.Categorical(final)
print(ad.obs["cell_type_final"].value_counts(dropna=False).to_string())

# cache full-skin merged labels for nb14 / nb21 / nb23
pd.DataFrame({"cell_id": ids,
              "cell_type_broad": ad.obs["cell_type_broad"].astype(str).to_numpy(),
              "cell_type_final": final}).to_csv(CT_FINAL_CSV, index=False)
print("cached full-skin labels ->", CT_FINAL_CSV)
```

```
cell_type_final
CD4             442982
Keratinocyte    272566
Fibroblast      184452
Myeloid         144009
Vascular        104873
CD8              55923
CD4_Treg         41011
Unk              38280
B                28610
Melanocyte       15422
Plasma           13704
Mast              3695
cached full-skin labels -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/skin_cell_type_final.csv
```

## 12.6 — Final annotated UMAPs

Full skin UMAP by `cell_type_final` and the T-only UMAP by the refined `cell_type_T`, before caching.

```python  # cell 34
adv = plot_view(ad, "broad")
adtv = plot_view(adt, "T")
sc.pl.umap(adv, color="cell_type_final", frameon=False, size=3,
           legend_loc="on data", legend_fontsize=6,
           title=f"skin — cell_type_final ({adv.n_obs:,} of {ad.n_obs:,} shown)")
sc.pl.umap(adtv, color="cell_type_T", frameon=False, size=4,
           legend_loc="on data", legend_fontsize=6,
           title=f"T cells — cell_type_T ({adtv.n_obs:,} of {adt.n_obs:,} shown)")
```

```
plot_view[broad]: 150,000/1,345,527 cells, cached UMAP
plot_view[T]: 150,000/539,916 cells, cached UMAP

<Figure size 640x480 with 1 Axes>
<Figure size 640x480 with 1 Axes>
```

## 13 — Cache the annotated T-cell object (full 42,347-gene space)

§1–§12 run on 10k HVGs, which is fine for clustering and marker dot plots but **not** for inferCNV:
`23_malignancy_tcr_cnv` needs genome-wide coverage per chromosome arm, and an HVG panel is biased toward variable
genes and leaves whole arms with almost no support. So this step streams the atlas' full gene space
back in — for the ~400k T cells only — without ever loading the 112 GB file.

Contiguous CSR slabs, filtered to the T rows: ~0.13 GB/s measured, so ~5 min and ~15 GB peak.
(Row-fancy-indexing looks tidier but degrades to ~6 h when the wanted rows are sparse in the file.)
Output matches the v1 contract — `X` log1p-normalized, raw counts in `layers["counts"]` — so
`old/13n_skin_T_reannotation` / `old/14_skin_T_tcr_cnv_malignancy` / `old/21_tcr_alice_neighborhood` / `old/23_subclonal_evolution` and `23_malignancy_tcr_cnv`'s CNV chain read it unchanged.

```python  # cell 36
def fullgene_counts(atlas, cell_ids, slab=50_000):
    """Raw counts over all 42,347 atlas genes for `cell_ids`, streamed in contiguous slabs.

    Row-fancy-indexing (`Xd[rows]`) is only fast when the wanted rows are dense in the file:
    measured 6.8 s per 10k rows when they sit 1-in-3, but 557 s per 10k when scattered over the
    2.16M — a 6-hour extrapolation. Contiguous slabs read at a steady ~0.13 GB/s (~5 min for the
    whole atlas) whatever the distribution, so read slabs and drop the rows we do not want.
    """
    with h5py.File(atlas, "r") as f:
        atlas_ids = np.asarray(read_elem(f["obs/cell_id"])).astype(str)
        pos = pd.Series(np.arange(atlas_ids.size), index=atlas_ids)
        rows = pos.reindex(cell_ids).to_numpy()
        assert not pd.isna(rows).any(), "some T cell_ids are absent from the atlas"
        order = np.argsort(rows.astype(np.int64), kind="stable")
        srows = rows.astype(np.int64)[order]

        Xd = sparse_dataset(f["X"])
        n_atlas, blocks, done = atlas_ids.size, [], 0
        for a in range(0, n_atlas, slab):
            b = min(a + slab, n_atlas)
            lo, hi = np.searchsorted(srows, a), np.searchsorted(srows, b)
            if hi == lo:
                continue                                   # slab holds no wanted cell — skip the read
            keep = Xd[a:b][srows[lo:hi] - a].astype(np.float32)
            keep.indices = keep.indices.astype(np.int32)    # 42,347 genes fits int32
            blocks.append(keep)
            done += hi - lo
            print(f"  streamed {done:,}/{srows.size:,} cells (atlas row {b:,}/{n_atlas:,})", end="\r")
        X = sp.vstack(blocks, format="csr")
        del blocks
        var = read_elem(f["var"])

    if not (np.diff(order) == 1).all():                     # restore the caller's cell order
        X = X[np.argsort(order, kind="stable")]
    print(f"\nstreamed {X.shape[0]:,} x {X.shape[1]:,}  nnz={X.nnz:,}  "
          f"({(X.data.nbytes + X.indices.nbytes) / 1e9:.1f} GB)")
    return X, var


t_ids = adt.obs["cell_id"].astype(str).to_numpy()
counts, var_full = fullgene_counts(ANNOT, t_ids)
assert counts.shape[0] == adt.n_obs

adt_full = AnnData(X=counts.copy(), obs=adt.obs.copy(), var=var_full,
                   obsm={k: adt.obsm[k] for k in ("X_mrvi_u", "X_mrvi_z")})
adt_full.layers["counts"] = counts    # exact: integer counts are lossless in float32 below 2^24
del counts

sf = (1e4 / adt_full.obs["total_counts"].to_numpy()).astype(np.float32)   # same as §2
adt_full.X.data *= np.repeat(sf, np.diff(adt_full.X.indptr))
np.log1p(adt_full.X.data, out=adt_full.X.data)

adt_full.write_h5ad(T_ANNOT)
print("wrote", T_ANNOT, "->", adt_full.shape, f"({T_ANNOT.stat().st_size / 1e9:.1f} GB)")
print("X log1p-normalized:", float(adt_full.X.data.max()) < 20,
      "| counts layer raw:", float(adt_full.layers["counts"].data.max()) > 30)
print("feeds nb13n (cell_type_T2 CD4/CD8 refinement) / nb14 / nb21 / nb23 / nb30 inferCNV")
```

```
streamed 539,916/539,916 cells (atlas row 2,157,693/2,157,693)
streamed 539,916 x 42,347  nnz=755,862,568  (6.0 GB)
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/skin_T_annotated.h5ad -> (539916, 42347) (12.4 GB)
X log1p-normalized: True | counts layer raw: True
feeds nb13n (cell_type_T2 CD4/CD8 refinement) / nb14 / nb21 / nb23 / nb30 inferCNV
```
