# old/09_h_semantic_geom_cd4.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/09_h_semantic_geom_cd4.ipynb` and is never run._

# SemanticSCVI (geometric) on H (Herrera 2021, GSE171811) — **CD4** cells

Replicates `06_semantic_geom_cd4.ipynb` on the Herrera *Blood* 2021 ECCITE-seq
cohort (`GSE171811`). Same SemanticSCVI config as the ref: Geneformer prior,
geometric loss, 10 latents, 200 epochs, `coherence_weight=2000`,
`batch_key=donor`.

**Cohort filter (advanced MF skin):**
- Keep skin biopsies from tumor/plaque-stage MF (advanced cutaneous MF) and
  any HC skin samples; drop blood / PBMC and Sézary-syndrome samples.
- Vocabulary is dataset-specific — Step 2 prints the GEO metadata and writes
  a starter `patients.tsv` with a heuristic `is_advanced_mf_skin` flag; edit
  the TSV before re-running if the heuristic mislabels anything.

**Malignant CD4 calling — same method as the ref (no V(D)J).**
Herrera deposited V(D)J alongside RNA, but we stay faithful to the ref's
manual cluster-pick recipe (`06_semantic_geom_cd4` cells 4–8, also used in
`08_d2_semantic_geom_cd4`): cluster CD4 T cells on a standard scVI latent,
dot plot CD4 / CD8 / FOXP3 + composition by MF-vs-HC origin, hand-pick the
clusters dominated by MF samples as malignant.

**Workflow**
1. Download GEO supplementary files + SOFT metadata.
2. Inspect metadata, write `patients.tsv` with `is_advanced_mf_skin` flag.
3. Concat samples, filter, QC, identify T cells.
4. Inspect cluster dot plot, fill `CD4_MAL_CLUSTERS`.
5. Run the rest top-to-bottom on the `neural_nmf_env` GPU kernel.

```python  # cell 1
# ============================================================
# Parameters — knobs match 06_semantic_geom_cd4.ipynb (the ref)
# ============================================================
import hashlib
import json
from pathlib import Path


def _resolve_nb_dir() -> Path:
    start = Path(__file__).parent.resolve() if "__file__" in globals() else Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("notebooks/MF"), Path("scvi-tools-neural-nmf/notebooks/MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data starting from {start}")


def _semantic_cache_slug(kwargs, max_epochs, warmup_epochs, n_epochs_kl_warmup, hvg_top_n, n=10):
    blob = json.dumps(
        {"kwargs": dict(sorted(kwargs.items())),
         "max_epochs": max_epochs, "warmup_epochs": warmup_epochs,
         "n_epochs_kl_warmup": n_epochs_kl_warmup, "hvg_top_n": hvg_top_n},
        default=str, sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:n]


NB_DIR = _resolve_nb_dir()
print(f"NB_DIR = {NB_DIR}")

# ---- H paths ----
GSE_ID    = "GSE171811"
GEO_BASE  = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE171nnn/{GSE_ID}"

H_DIR     = NB_DIR / "data" / "H_herrera2021"
RAW_DIR   = H_DIR / "raw"
META_DIR  = H_DIR / "meta"
META_TSV  = META_DIR / "patients.tsv"
SOFT_GZ   = META_DIR / f"{GSE_ID}_family.soft.gz"
CONCAT_H5 = H_DIR / "processed" / "concat.h5ad"
CD4_H5    = H_DIR / "processed" / "h_cd4_combined.h5ad"
SEMANTIC_CACHE_GENEFORMER = NB_DIR / "data" / "h_cd4_geneformer.pt"
MODEL_CACHE_DIR = NB_DIR / "models" / ".model_cache_h_semantic_geom_cd4"
FIG_DIR   = NB_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR = NB_DIR / "tables";  TABLE_DIR.mkdir(exist_ok=True)
for d in (RAW_DIR, META_DIR, CONCAT_H5.parent):
    d.mkdir(parents=True, exist_ok=True)

# ---- CD4 selection ----
TUMOR_LEIDEN_RESOLUTION = 0.5

# ---- Preprocessing (ref values) ----
HVG_TOP_N  = 2500
HVG_FLAVOR = "seurat_v3"
SUBSAMPLE_N = None

# ---- Model size + batch ----
N_LATENT  = 10
BATCH_KEY = "donor"

# ---- SemanticSCVI (Geneformer prior) — identical to ref ----
SEMANTIC_GENEFORMER_MAX_EPOCHS = 200
SEMANTIC_GENEFORMER_WARMUP_EPOCHS = 20
SEMANTIC_GENEFORMER_N_EPOCHS_KL_WARMUP = 100
SEMANTIC_GENEFORMER_KWARGS = dict(
    loss_mode="geometric",
    coherence_weight=2000.0,
    n_gene_sample=1024,
    n_latent=N_LATENT,
    n_layers=2, n_hidden=128, dropout_rate=0.1,
    gene_likelihood="nb",
    weights_positive=True,
    use_batch_norm=False,
)
SEMANTIC_GENEFORMER_CACHE_DIR = MODEL_CACHE_DIR / "semantic_scvi" / _semantic_cache_slug(
    {**SEMANTIC_GENEFORMER_KWARGS, "batch_key": BATCH_KEY},
    SEMANTIC_GENEFORMER_MAX_EPOCHS,
    SEMANTIC_GENEFORMER_WARMUP_EPOCHS,
    SEMANTIC_GENEFORMER_N_EPOCHS_KL_WARMUP,
    HVG_TOP_N,
)
print(f"semantic_geom cache_dir: {SEMANTIC_GENEFORMER_CACHE_DIR}")
FORCE_TRAIN_SEMANTIC_GENEFORMER = False

PER_PROJECTION_N_TOP = 30
CLUSTER_N_TOP        = 500
CLUSTER_MAX_K        = 20
TOP_PER_CLUSTER      = 25
```

```python  # cell 2
import sys
if str(NB_DIR.parent) not in sys.path:
    sys.path.insert(0, str(NB_DIR.parent))

import gc, gzip
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import scvi

from benchmark_helpers import get_or_build_geneformer_map, train_or_load_semantic_scvi

scvi.settings.seed = 0
sc.settings.verbosity = 1
print("scvi", scvi.__version__, "| scanpy", sc.__version__)
print("CUDA available:", torch.cuda.is_available())
```

## Step 1 — Download GEO supplementary files + SOFT metadata

Mirrors `notebooks/MF/herrera_GSE171811_guide.md`. Idempotent — re-running
when `raw/` already has files just lists what's there. The SOFT file
(`GSE171811_family.soft.gz`) carries per-sample `characteristics_ch1`
(tissue, disease, stage…) that we mine in Step 2.

```python  # cell 4
import subprocess
import urllib.request

# 1. List what's in the suppl/ dir (sanity check)
suppl_url = f"{GEO_BASE}/suppl/"
print(f"listing {suppl_url}")
try:
    html = urllib.request.urlopen(suppl_url, timeout=30).read().decode()
    listed = sorted({
        tok.split('href="', 1)[1].split('"', 1)[0]
        for tok in html.split() if 'href="' in tok and 'href="?' not in tok and 'href="/' not in tok
    })
    for n in listed[:30]:
        print(" ", n)
except Exception as e:
    print("listing failed:", e)
    listed = []

# 2. Download supplementary archive(s) — skip if raw/ already has data
existing = [p for p in RAW_DIR.iterdir()
            if p.suffix in {".h5", ".h5ad", ".rds", ".mtx", ".tsv", ".csv", ".gz", ".tar"}]
if existing:
    print(f"\nraw/ already populated ({len(existing)} files) — skipping download.")
else:
    print(f"\ndownloading into {RAW_DIR}")
    bundle_url = f"{GEO_BASE}/suppl/{GSE_ID}_RAW.tar"
    rc = subprocess.call(["wget", "-nc", "-q", "-P", str(RAW_DIR), bundle_url])
    bundle = RAW_DIR / f"{GSE_ID}_RAW.tar"
    if rc == 0 and bundle.exists():
        print(f"untarring {bundle.name}")
        subprocess.check_call(["tar", "-xvf", str(bundle), "-C", str(RAW_DIR)])
        bundle.unlink()
    else:
        # Fallback: mirror everything in suppl/
        print("bundle missing — falling back to recursive wget of suppl/")
        subprocess.check_call([
            "wget", "-r", "-np", "-nd", "-nc", "-q", "-e", "robots=off",
            "-A", "tar,gz,csv,mtx,tsv,h5,h5ad,rds",
            "-P", str(RAW_DIR), suppl_url,
        ])

print("\nfiles now in raw/:")
for p in sorted(RAW_DIR.iterdir())[:30]:
    print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")

# 3. SOFT family file for sample metadata
if not SOFT_GZ.exists():
    soft_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE171nnn/{GSE_ID}/soft/{GSE_ID}_family.soft.gz"
    print(f"\ndownloading SOFT: {soft_url}")
    urllib.request.urlretrieve(soft_url, SOFT_GZ)
print(f"SOFT: {SOFT_GZ}  ({SOFT_GZ.stat().st_size / 1e6:.1f} MB)")
```

## Step 2 — Inspect GEO metadata, write `patients.tsv`

Parses `characteristics_ch1` from the SOFT file to surface per-sample
`tissue / disease / stage / source_name`. Writes a starter `patients.tsv`
keyed by `sample_id` (GSM accession) with an `is_advanced_mf_skin`
heuristic.

**Heuristic:** `is_advanced_mf_skin = True` if `tissue ~ skin/biopsy/lesion`
AND NOT `blood|PBMC|sezary|SS` AND `stage ~ IIB|III|IV|tumor|plaque|advanced|late`.

Inspect the printed table; if the heuristic mislabels any row, **edit
`patients.tsv` directly** and re-run from Step 3.

```python  # cell 6
import re

# Parse SOFT directly (lightweight — no GEOparse dep needed)
sample_rows: list[dict] = []
cur: dict | None = None
with gzip.open(SOFT_GZ, "rt", errors="replace") as fh:
    for line in fh:
        line = line.rstrip("\n")
        if line.startswith("^SAMPLE = "):
            if cur is not None:
                sample_rows.append(cur)
            cur = {"sample_id": line.split("= ", 1)[1].strip(), "characteristics": []}
        elif cur is not None:
            if line.startswith("!Sample_title"):
                cur["title"] = line.split("= ", 1)[1].strip()
            elif line.startswith("!Sample_source_name_ch1"):
                cur["source_name"] = line.split("= ", 1)[1].strip()
            elif line.startswith("!Sample_characteristics_ch1"):
                cur["characteristics"].append(line.split("= ", 1)[1].strip())
            elif line.startswith("!Sample_library_strategy"):
                cur["library_strategy"] = line.split("= ", 1)[1].strip()
            elif line.startswith("!Sample_supplementary_file"):
                cur.setdefault("supp", []).append(line.split("= ", 1)[1].strip())
if cur is not None:
    sample_rows.append(cur)

# Pivot 'key: value' characteristics into columns
def _kv(s: str):
    if ":" not in s:
        return None
    k, v = s.split(":", 1)
    return k.strip().lower().replace(" ", "_"), v.strip()

records = []
for r in sample_rows:
    flat = {"sample_id": r["sample_id"], "title": r.get("title"),
            "source_name": r.get("source_name"),
            "library_strategy": r.get("library_strategy")}
    for c in r.get("characteristics", []):
        kv = _kv(c)
        if kv:
            flat[kv[0]] = kv[1]
    records.append(flat)
meta = pd.DataFrame(records)
print(f"{GSE_ID}: {len(meta)} samples\n")
print("columns:", list(meta.columns))
print("\nfull sample table:")
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 220):
    print(meta.to_string(index=False))

# Unique values per characteristic (shows stage / tissue vocabulary)
print("\nunique values per characteristic column:")
for col in meta.columns:
    if col in {"sample_id", "title"}:
        continue
    vals = meta[col].dropna().astype(str).unique()
    if len(vals) <= 40:
        print(f"  {col}: {sorted(vals)}")
    else:
        print(f"  {col}: ({len(vals)} unique)")

# Heuristic flag
def _has(s, pat):
    return bool(re.search(pat, str(s or ""), flags=re.IGNORECASE))

text_cols = [c for c in meta.columns if c not in {"sample_id", "supp"}]
blobs = meta.apply(lambda row: " | ".join(str(row.get(c) or "") for c in text_cols), axis=1)

is_skin     = blobs.apply(lambda b: _has(b, r"\bskin\b|biopsy|lesion"))
is_blood    = blobs.apply(lambda b: _has(b, r"\bblood\b|PBMC|peripheral"))
is_sezary   = blobs.apply(lambda b: _has(b, r"sezary|\bSS\b"))
is_advanced = blobs.apply(lambda b: _has(b, r"\bIIB\b|\bIIIA?\b|\bIIIB\b|\bIVA?\b|\bIVB\b|tumou?r|plaque|advanced|late"))

meta["disease"] = np.where(blobs.str.contains(r"healthy|normal|control", case=False), "HC", "MF")
meta["is_advanced_mf_skin"] = (is_skin & ~is_blood & ~is_sezary & is_advanced).astype(bool)

# Best-effort disease_stage column (some Herrera samples may use 'stage'/'tnmb_stage'/…)
if "disease_stage" not in meta.columns:
    for cand in ("stage", "tumor_stage", "tnm_stage", "tnmb_stage", "patient_stage", "disease_status"):
        if cand in meta.columns:
            meta["disease_stage"] = meta[cand]
            break
    else:
        meta["disease_stage"] = ""

print("\nheuristic assignment (review before continuing):")
print(meta[["sample_id", "title", "source_name", "disease", "disease_stage", "is_advanced_mf_skin"]].to_string(index=False))

if META_TSV.exists():
    print(f"\n{META_TSV} already exists — NOT overwriting. Delete it to regenerate.")
else:
    meta.to_csv(META_TSV, sep="\t", index=False)
    print(f"\nwrote starter {META_TSV}")
    print("→ inspect / edit `is_advanced_mf_skin` column, then continue from Step 3.")
```

## Step 3 — Concatenate samples

Handles four deposit formats (auto-detected):
1. **Integrated `.h5ad`**.
2. **Integrated `.rds`** (Seurat v4) — convert to `.h5ad` via `sceasy` in R first.
3. **Per-sample 10x** (`*filtered_feature_bc_matrix.h5` or `matrix.mtx` triplets).
4. **Per-sample ECCITE-seq dense TSV** (Herrera): `GSM<id>_<donor>_<tissue>_GEX.tsv.gz`,
   gene-rows × cell-cols. Prefilters on `patients.tsv` (skips SS / blood-only samples
   to save RAM); parses `donor` + `tissue` from the filename.

Sample matching: filename → GSM id → `patients.tsv` row.

```python  # cell 8
import re
import shutil

patients = pd.read_csv(META_TSV, sep="\t")
print(f"loaded {len(patients)} patient rows from {META_TSV}")
print("advanced-MF-skin samples:",
      patients.loc[patients["is_advanced_mf_skin"], "sample_id"].tolist())
print("HC samples:",
      patients.loc[patients["disease"] == "HC", "sample_id"].tolist())

CONCAT_H5.parent.mkdir(parents=True, exist_ok=True)

h5ad_candidates = sorted(RAW_DIR.glob("*.h5ad"))
rds_candidates  = sorted(RAW_DIR.glob("*.rds"))
per_sample_h5   = sorted(RAW_DIR.glob("*filtered_feature_bc_matrix.h5"))
per_sample_mtx  = sorted(RAW_DIR.glob("*matrix.mtx*"))
per_sample_tsv  = sorted(RAW_DIR.glob("*_GEX.tsv.gz"))   # Herrera ECCITE-seq

def _gsm_from(name: str) -> str | None:
    m = re.match(r"(GSM\d+)", name)
    return m.group(1) if m else None

if CONCAT_H5.exists():
    print(f"\nusing cached {CONCAT_H5}")
    adata = sc.read_h5ad(CONCAT_H5)
elif h5ad_candidates:
    src = h5ad_candidates[0]
    print(f"\nloading integrated h5ad: {src}")
    adata = sc.read_h5ad(src)
    if "raw_counts" not in adata.layers:
        adata.layers["raw_counts"] = adata.X.copy()
elif rds_candidates:
    raise RuntimeError(
        f"Found .rds in {RAW_DIR}. Convert to .h5ad in R first:\n"
        "  library(sceasy); sceasy::convertFormat(<rds>, from='seurat', to='anndata', outFile='<...>.h5ad')"
    )
elif per_sample_h5:
    print("\nper-sample format (10x .h5) — concatenating")
    adatas = {}
    for h5 in per_sample_h5:
        sid = _gsm_from(h5.name) or h5.stem.replace("_filtered_feature_bc_matrix", "")
        a = sc.read_10x_h5(h5)
        a.var_names_make_unique()
        a.obs["sample_id"] = sid
        adatas[sid] = a
        print(f"  {sid}: {a.shape}")
    adata = sc.concat(adatas, axis=0, join="outer", index_unique="-")
    adata.layers["raw_counts"] = adata.X.copy()
elif per_sample_tsv:
    # Herrera GSE171811: per-sample dense GEX TSV (gene-rows × cell-cols),
    # filename = GSM<id>_<donor>_<tissue>_GEX.tsv.gz. Prefilter on patients.tsv
    # so we never load SS samples we'd drop in Step 5 anyway.
    print(f"\nper-sample format (ECCITE-seq TSV GEX) — {len(per_sample_tsv)} files")
    relevant_sids = set(
        patients.loc[
            patients["is_advanced_mf_skin"] | (patients["disease"] == "HC"),
            "sample_id",
        ].astype(str)
    )
    print(f"  relevant cohort (advanced MF skin OR HC): {sorted(relevant_sids)}")

    adatas = {}
    for tsv in per_sample_tsv:
        stem = tsv.name.replace("_GEX.tsv.gz", "")
        parts = stem.split("_", 2)   # ['GSM5234587', 'MFIV1', 'Skin']
        sid        = parts[0]
        donor_lbl  = parts[1] if len(parts) > 1 else sid
        tissue_lbl = parts[2] if len(parts) > 2 else ""
        if relevant_sids and sid not in relevant_sids:
            print(f"  skip {sid} ({donor_lbl}/{tissue_lbl}) — not in cohort")
            continue
        # NOTE: index_col=0 puts gene names in the index BEFORE dtype is applied to
        # data columns, but pandas applies a single dtype to ALL columns (incl. the
        # index column) when reading — so we can't pass dtype=int here. Cast after.
        df = pd.read_csv(tsv, sep="\t", index_col=0).astype(np.int32, copy=False)
        # genes × cells → transpose to cells × genes
        a = sc.AnnData(
            X=sp.csr_matrix(df.T.values),
            obs=pd.DataFrame(index=df.columns.astype(str)),
            var=pd.DataFrame(index=df.index.astype(str)),
        )
        a.var_names_make_unique()
        a.obs["sample_id"]   = sid
        a.obs["donor_label"] = donor_lbl
        a.obs["tissue"]      = tissue_lbl
        adatas[sid] = a
        print(f"  {sid} ({donor_lbl}/{tissue_lbl}): {a.shape}")
        del df
        gc.collect()
    if not adatas:
        raise RuntimeError(
            "No relevant samples found. Inspect patients.tsv — none matched "
            "is_advanced_mf_skin OR disease=='HC'. Edit the TSV and re-run."
        )
    adata = sc.concat(adatas, axis=0, join="outer", index_unique="-")
    adata.layers["raw_counts"] = adata.X.copy()
elif per_sample_mtx:
    print("\nper-sample format (10x mtx) — concatenating")
    adatas = {}
    seen = set()
    for mtx in per_sample_mtx:
        sid = _gsm_from(mtx.name)
        if sid is None or sid in seen:
            continue
        seen.add(sid)
        stem = mtx.name.split("matrix.mtx", 1)[0]
        feats = next(iter(RAW_DIR.glob(f"{stem}*features.tsv*")), None) \
             or next(iter(RAW_DIR.glob(f"{stem}*genes.tsv*")), None)
        bcs   = next(iter(RAW_DIR.glob(f"{stem}*barcodes.tsv*")), None)
        if feats is None or bcs is None:
            print(f"  skip {sid}: missing features/barcodes ({stem})"); continue
        tmp = RAW_DIR / "_staged" / sid
        tmp.mkdir(parents=True, exist_ok=True)
        for src_p, dst_n in [(mtx, "matrix.mtx.gz"),
                             (feats, "features.tsv.gz"),
                             (bcs, "barcodes.tsv.gz")]:
            dst = tmp / dst_n
            if not dst.exists():
                if src_p.suffix == ".gz":
                    shutil.copy(src_p, dst)
                else:
                    with src_p.open("rb") as s, gzip.open(dst, "wb") as d:
                        d.write(s.read())
        a = sc.read_10x_mtx(tmp, var_names="gene_symbols", make_unique=True)
        a.obs["sample_id"] = sid
        adatas[sid] = a
        print(f"  {sid}: {a.shape}")
    if not adatas:
        raise RuntimeError("no per-sample mtx triplets resolved")
    adata = sc.concat(adatas, axis=0, join="outer", index_unique="-")
    adata.layers["raw_counts"] = adata.X.copy()
else:
    raise RuntimeError(
        f"no recognizable deposit format in {RAW_DIR} "
        f"(saw: {[p.name for p in sorted(RAW_DIR.iterdir())][:10]})"
    )

# Merge patient metadata
if "sample_id" not in adata.obs.columns:
    raise RuntimeError(
        "Need adata.obs['sample_id'] aligning with patients.tsv. "
        f"Current obs columns: {list(adata.obs.columns)}"
    )
adata.obs = adata.obs.merge(patients, on="sample_id", how="left", suffixes=("", "_meta"))
# Prefer human-readable donor label parsed from filename when available
adata.obs["donor"] = (
    adata.obs["donor_label"].astype(str)
    if "donor_label" in adata.obs.columns
    else adata.obs["sample_id"].astype(str)
)
if "raw_counts" not in adata.layers:
    adata.layers["raw_counts"] = adata.X.copy()
if not CONCAT_H5.exists():
    adata.write_h5ad(CONCAT_H5)
    print(f"\nwrote {CONCAT_H5}  | shape {adata.shape}")
print("\nsample counts by disease:")
print(adata.obs["disease"].value_counts() if "disease" in adata.obs else "(no disease col)")
if "disease_stage" in adata.obs:
    print("\nsample counts by disease_stage:")
    print(adata.obs["disease_stage"].value_counts())
print("\ncells per donor:")
print(adata.obs["donor"].value_counts())
```

```
loaded 14 patient rows from /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/H_herrera2021/meta/patients.tsv
advanced-MF-skin samples: []
HC samples: ['GSM5234576', 'GSM5234577']

per-sample format (ECCITE-seq TSV GEX) — 14 files
  relevant cohort (advanced MF skin OR HC): ['GSM5234576', 'GSM5234577']
  GSM5234576 (HC1/Blood): (4503, 33660)
  GSM5234577 (HC1/Skin): (50, 33660)
  skip GSM5234578 (SS1/Blood) — not in cohort
  skip GSM5234579 (SS1/Skin) — not in cohort
  skip GSM5234580 (SS2/Blood) — not in cohort
  skip GSM5234581 (SS2/Skin) — not in cohort
  skip GSM5234582 (SS3/Blood) — not in cohort
  skip GSM5234583 (SS3/Skin) — not in cohort
  skip GSM5234584 (SS4/Blood) — not in cohort
  skip GSM5234585 (SS4/Skin) — not in cohort
  skip GSM5234586 (MFIV1/Blood) — not in cohort
  skip GSM5234587 (MFIV1/Skin) — not in cohort
  skip GSM5234588 (SS5/Blood) — not in cohort
  skip GSM5234589 (SS6/Blood) — not in cohort

/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/functools.py:912: ImplicitModificationWarning: Transforming to str index.
  return dispatch(args[0].__class__)(*args, **kw)


wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/H_herrera2021/processed/concat.h5ad  | shape (4553, 33660)

sample counts by disease:
disease
HC    4553
Name: count, dtype: int64

sample counts by disease_stage:
Series([], Name: count, dtype: int64)

cells per donor:
donor
HC1    4553
Name: count, dtype: int64
```

```python  # cell 9
adata.obs.shape
```

```
(4553, 13)
```

## Step 4 — Locate author annotations (T-cell labels, malignancy if any)

Print obs columns; if any look like author cell-type / cluster labels,
unify them under `cell_type_author`. Otherwise we'll annotate by markers
in Step 6.

```python  # cell 11
print("obs columns in concatenated AnnData:")
for c in sorted(adata.obs.columns):
    n_unique = adata.obs[c].nunique() if adata.obs[c].dtype != float else "(float)"
    sample_vals = list(adata.obs[c].dropna().astype(str).unique()[:5])
    print(f"  {c:30s}  n_unique={n_unique}  sample={sample_vals}")

# CANDIDATES to look for and unify under "cell_type_author":
CANDIDATE_TYPE_COLS = [
    "cell_type", "celltype", "CellType", "Cell_Type",
    "seurat_clusters", "cluster", "Cluster", "annotation",
    "cell_id", "manual_annotation",
]
type_col = next((c for c in CANDIDATE_TYPE_COLS if c in adata.obs.columns), None)
if type_col is None:
    print("\n⚠️  no obvious cell-type column — will need to annotate from scratch via markers.")
else:
    print(f"\nusing '{type_col}' as author cell-type label.")
    adata.obs["cell_type_author"] = adata.obs[type_col].astype(str)
    print(adata.obs["cell_type_author"].value_counts())
```

## Step 5 — Filter to advanced MF skin + HC; basic QC

```python  # cell 13
keep_mask = adata.obs["is_advanced_mf_skin"] | (adata.obs["disease"] == "HC")
adata = adata[keep_mask].copy()
print("after sample filter:", adata.shape)
print(adata.obs["sample_id"].value_counts())

adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
keep = (
    (adata.obs["pct_counts_mt"] < 15)
    & (adata.obs["n_genes_by_counts"] > 200)
    & (adata.obs["n_genes_by_counts"] < 6000)
)
adata = adata[keep].copy()
print("post-QC:", adata.shape)
adata.layers["raw_counts"] = adata.X.copy()
```

## Step 6 — Identify CD4⁺ T cells by marker expression

```python  # cell 15
adata.layers["counts"] = adata.layers["raw_counts"].copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

def _expr(a, gene):
    if gene not in a.var_names:
        return np.zeros(a.n_obs, dtype=np.float32)
    v = a[:, gene].X
    return (v.toarray().ravel() if sp.issparse(v) else np.asarray(v).ravel()).astype(np.float32)

adata.obs["score_CD3"]   = np.maximum.reduce([_expr(adata, g) for g in ["CD3D", "CD3E", "CD3G"]])
adata.obs["score_CD4"]   = _expr(adata, "CD4")
adata.obs["score_CD8"]   = np.maximum(_expr(adata, "CD8A"), _expr(adata, "CD8B"))
adata.obs["score_FOXP3"] = _expr(adata, "FOXP3")

is_tcell = adata.obs["score_CD3"] > 0.5
is_cd4   = (adata.obs["score_CD4"] > 0) & (adata.obs["score_CD4"] >= adata.obs["score_CD8"])
not_treg = adata.obs["score_FOXP3"] < 0.5
adata.obs["is_cd4_tcell"] = (is_tcell & is_cd4 & not_treg).astype(bool)

print("T cell count:", int(is_tcell.sum()))
print("CD4 helper T cells:", int(adata.obs["is_cd4_tcell"].sum()))
print("by sample:")
print(adata.obs.groupby("sample_id")["is_cd4_tcell"].sum().to_string())

adata_cd4 = adata[adata.obs["is_cd4_tcell"]].copy()
adata_cd4.X = adata_cd4.layers["raw_counts"].copy()
print("\nCD4 subset:", adata_cd4.shape)
del adata; gc.collect()
```

## Step 7 — Cluster CD4 T cells on standard scVI latent (manual malignant-pick)

Herrera has no author-provided malignant T-cell label, so we use the ref's
manual cluster-pick approach (`06_semantic_geom_cd4` cells 4–8). The dot
plot + per-cluster MF/HC composition below guides the pick.

```python  # cell 17
scvi.model.SCVI.setup_anndata(adata_cd4, batch_key="donor", layer="raw_counts")
scvi_clust = scvi.model.SCVI(adata_cd4, n_layers=2, n_latent=10, gene_likelihood="nb")
scvi_clust.train(max_epochs=50, early_stopping=True, accelerator="auto")
adata_cd4.obsm["X_scVI"] = scvi_clust.get_latent_representation()

sc.pp.neighbors(adata_cd4, use_rep="X_scVI", random_state=0)
sc.tl.umap(adata_cd4, random_state=0)
sc.tl.leiden(adata_cd4, resolution=TUMOR_LEIDEN_RESOLUTION, random_state=0, key_added="cd4_leiden")
print("leiden cluster sizes:", adata_cd4.obs["cd4_leiden"].value_counts().to_dict())

sc.pl.umap(
    adata_cd4,
    color=["cd4_leiden", "disease", "disease_stage", "sample_id"],
    ncols=2, wspace=0.3, frameon=False, save=None,
)
```

### Dot plot CD4 / CD8 / FOXP3 + per-cluster MF/HC composition

```python  # cell 19
import matplotlib.pyplot as plt

markers = [g for g in ["CD4", "CD8A", "CD8B", "FOXP3", "CD3D"] if g in adata_cd4.var_names]
dp = sc.pl.dotplot(
    adata_cd4, var_names=markers, groupby="cd4_leiden",
    standard_scale="var", return_fig=True,
)
dp.savefig(FIG_DIR / "h_cd4_tumor_dotplot.png", dpi=200, bbox_inches="tight")
print("saved", FIG_DIR / "h_cd4_tumor_dotplot.png")

cluster_stats = (
    adata_cd4.obs.groupby("cd4_leiden")
    .agg(
        n_cells=("disease", "size"),
        frac_HC=("disease", lambda s: (s == "HC").mean()),
        frac_MF=("disease", lambda s: (s == "MF").mean()),
        n_samples=("sample_id", "nunique"),
    )
    .sort_values("frac_MF", ascending=False)
)
print("\nper-cluster MF/HC composition:")
print(cluster_stats.round(3).to_string())
```

## 🛑 Step 8 — Hand-pick malignant CD4 clusters

Set `CD4_MAL_CLUSTERS` from the dot plot + composition table above.
Pick clusters that are (a) CD4+ / CD8- / FOXP3- by markers AND (b)
dominated by MF samples (high `frac_MF`, low `frac_HC`). Mixed clusters
→ benign by default.

```python  # cell 21
# EDIT THIS LIST after inspecting the dot plot above.
CD4_MAL_CLUSTERS: list[str] = []   # e.g. ["0", "1", "3"]

assert CD4_MAL_CLUSTERS, "Inspect the dot plot, then fill CD4_MAL_CLUSTERS"

adata_cd4.obs["is_malignant"] = adata_cd4.obs["cd4_leiden"].isin(CD4_MAL_CLUSTERS).astype(bool)
adata_cd4.obs["status"]    = np.where(adata_cd4.obs["is_malignant"], "tumor_cd4", "benign_cd4")
adata_cd4.obs["cell_type"] = adata_cd4.obs["status"]

print(f"selected clusters: {CD4_MAL_CLUSTERS}")
print("status counts:", adata_cd4.obs["status"].value_counts().to_dict())
print("\nper-sample status:")
print(adata_cd4.obs.groupby("sample_id")["status"].value_counts().unstack(fill_value=0).to_string())
adata_cd4.write_h5ad(CD4_H5)
print("wrote", CD4_H5)
```

## Step 9 — Attach Ensembl gene_id + build Geneformer semantic map, HVG-subset

```python  # cell 23
# If the deposit didn't include ENSG IDs, attach them from any 10x features.tsv
# in the raw dir, or fall back to symbol-only matching (Geneformer accepts both
# but Ensembl is preferred).
if "gene_id" not in adata_cd4.var.columns:
    features_files = list(RAW_DIR.rglob("features.tsv*")) + list(RAW_DIR.rglob("*features.tsv*"))
    if features_files:
        ft = pd.read_csv(
            features_files[0], header=None, sep="\t",
            names=["gene_id", "gene_name", "feature_type"],
        )
        sym2ens = dict(zip(ft["gene_name"].astype(str), ft["gene_id"].astype(str)))
        adata_cd4.var["gene_id"] = [sym2ens.get(s, s) for s in adata_cd4.var_names.astype(str)]
    else:
        # Fallback: use symbol as the key (Geneformer will treat as missing for non-ENSG)
        adata_cd4.var["gene_id"] = adata_cd4.var_names

n_mapped = int(sum(str(g).startswith("ENSG") for g in adata_cd4.var["gene_id"]))
print(f"gene_id mapped to Ensembl: {n_mapped}/{adata_cd4.n_vars}")

semantic_map = get_or_build_geneformer_map(adata_cd4, SEMANTIC_CACHE_GENEFORMER, var_id_key="gene_id")
print("semantic_map (raw):", tuple(semantic_map.shape))

if HVG_TOP_N is not None:
    sc.pp.highly_variable_genes(adata_cd4, n_top_genes=HVG_TOP_N, flavor=HVG_FLAVOR, subset=False)
    keep = adata_cd4.var["highly_variable"].values
    adata_cd4 = adata_cd4[:, keep].copy()
    semantic_map = semantic_map[torch.as_tensor(keep)]
print("after HVG:", adata_cd4.shape, "| semantic_map:", tuple(semantic_map.shape))
```

## Step 10 — Train SemanticSCVI (geometric, Geneformer prior) — GPU

```python  # cell 25
model = train_or_load_semantic_scvi(
    adata_cd4,
    semantic_map,
    cache_dir=SEMANTIC_GENEFORMER_CACHE_DIR,
    force_train=FORCE_TRAIN_SEMANTIC_GENEFORMER,
    max_epochs=SEMANTIC_GENEFORMER_MAX_EPOCHS,
    warmup_epochs=SEMANTIC_GENEFORMER_WARMUP_EPOCHS,
    n_epochs_kl_warmup=SEMANTIC_GENEFORMER_N_EPOCHS_KL_WARMUP,
    batch_key=BATCH_KEY,
    **SEMANTIC_GENEFORMER_KWARGS,
)
print("trained:", model.is_trained)
```

## Step 11 — Projection analysis (mirrors 08 / 07 / 06)

```python  # cell 27
z = np.asarray(model.get_latent_representation())
ad_emb = sc.AnnData(z, obs=model.adata.obs.copy())
sc.pp.neighbors(ad_emb, use_rep="X", random_state=0)
sc.tl.umap(ad_emb, random_state=0)

color = [c for c in ["status", "disease", "sample_id", "disease_stage"] if c in ad_emb.obs]
sc.pl.umap(ad_emb, color=color, ncols=2, wspace=0.3, frameon=False)
```

```python  # cell 28
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import fisher_exact

is_mal = (ad_emb.obs["status"].astype(str).values == "tumor_cd4").astype(int)
n_factors = z.shape[1]
MAX_PER_CLASS = 15000
rng = np.random.default_rng(0)
idx_mal = np.where(is_mal == 1)[0]; idx_ben = np.where(is_mal == 0)[0]
if len(idx_mal) > MAX_PER_CLASS: idx_mal = rng.choice(idx_mal, MAX_PER_CLASS, replace=False)
if len(idx_ben) > MAX_PER_CLASS: idx_ben = rng.choice(idx_ben, MAX_PER_CLASS, replace=False)
plot_idx = np.concatenate([idx_mal, idx_ben])

aucs = {}
fig, axes = plt.subplots((n_factors + 1) // 2, 2, figsize=(11, 2.2 * ((n_factors + 1) // 2)), sharex=False)
axes = axes.flatten()
for k in range(n_factors):
    ax = axes[k]
    auc = roc_auc_score(is_mal, z[:, k]); aucs[f"Z_{k}"] = max(auc, 1 - auc)
    sub_vals = z[plot_idx, k]; sub_mal = is_mal[plot_idx]
    order = np.argsort(sub_vals); vals = sub_vals[order]; mal = sub_mal[order]
    rank = np.arange(len(vals))
    ax.scatter(rank[mal == 1], vals[mal == 1], s=2, c="tab:red", alpha=0.35, rasterized=True, linewidths=0)
    ax.scatter(rank[mal == 0], vals[mal == 0], s=2, c="tab:blue", alpha=0.35, rasterized=True, linewidths=0)
    ax.set_title(f"Z_{k}  |  AUROC = {aucs[f'Z_{k}']:.3f}", fontsize=10)
    ax.set_xlabel(f"cell rank (n={MAX_PER_CLASS}/class)", fontsize=8); ax.set_ylabel(f"Z_{k}", fontsize=8)
    ax.tick_params(labelsize=7)
for j in range(n_factors, len(axes)): axes[j].axis("off")
fig.tight_layout()
out = FIG_DIR / "h_semantic_geom_cd4_factor_separation.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out); plt.show()

print("\nAUROC (symmetric) per factor:")
print(pd.Series(aucs).sort_values(ascending=False).to_string())

# Top-3 most-separating factors → 3-D scatter
top_factors = sorted(aucs.items(), key=lambda kv: kv[1], reverse=True)[:3]
FX3, FY3, FZ3 = [int(k.split("_")[1]) for k, _ in top_factors]
print(f"\ntop-3 factors → 3-D scatter: Z_{FX3}, Z_{FY3}, Z_{FZ3}")
```

```python  # cell 29
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

x3 = z[:, FX3]; y3 = z[:, FY3]; zZ3 = z[:, FZ3]
mu_x3, sd_x3 = x3.mean(), x3.std()
mu_y3, sd_y3 = y3.mean(), y3.std()
mu_z3, sd_z3 = zZ3.mean(), zZ3.std()
xz3 = (x3 - mu_x3) / sd_x3; yz3 = (y3 - mu_y3) / sd_y3; zz3 = (zZ3 - mu_z3) / sd_z3
s3 = xz3 + yz3 + zz3
y_true = is_mal
median_s3 = float(np.median(s3))
fpr3, tpr3, thrs3 = roc_curve(y_true, s3)
best_thr3 = float(thrs3[np.argmax(tpr3 - fpr3)])
auc_s3 = max(roc_auc_score(y_true, s3), 1 - roc_auc_score(y_true, s3))

rng3 = np.random.default_rng(0); shuf3 = rng3.permutation(len(plot_idx))
xx3 = x3[plot_idx][shuf3]; yy3 = y3[plot_idx][shuf3]; zz3p = zZ3[plot_idx][shuf3]
ss3 = s3[plot_idx][shuf3]
cc3 = np.where(is_mal[plot_idx][shuf3] == 1, "tab:red", "tab:blue")
s_vmax3 = float(np.quantile(np.abs(s3), 0.99))

def s_plane(c, xg, yg):
    return sd_z3 * (c - (xg - mu_x3) / sd_x3 - (yg - mu_y3) / sd_y3) + mu_z3

xg = np.linspace(xx3.min(), xx3.max(), 20)
yg = np.linspace(yy3.min(), yy3.max(), 20)
XG, YG = np.meshgrid(xg, yg)
ZG_med  = s_plane(median_s3, XG, YG); ZG_best = s_plane(best_thr3, XG, YG)

fig = plt.figure(figsize=(14, 7))
axL = fig.add_subplot(1, 2, 1, projection="3d")
axL.scatter(xx3, yy3, zz3p, c=cc3, s=3, alpha=0.3, depthshade=False)
axL.plot_surface(XG, YG, ZG_med,  color="black", alpha=0.10, edgecolor="none")
axL.plot_surface(XG, YG, ZG_best, color="black", alpha=0.20, edgecolor="none")
axL.set_xlabel(f"Z_{FX3}"); axL.set_ylabel(f"Z_{FY3}"); axL.set_zlabel(f"Z_{FZ3}")
axL.set_title(f"colored by status   AUROC(s3) = {auc_s3:.3f}", fontsize=10)

axR = fig.add_subplot(1, 2, 2, projection="3d")
sc3 = axR.scatter(xx3, yy3, zz3p, c=ss3, cmap="RdBu_r", vmin=-s_vmax3, vmax=s_vmax3, s=3, alpha=0.55, depthshade=False)
axR.plot_surface(XG, YG, ZG_med,  color="black", alpha=0.10, edgecolor="none")
axR.plot_surface(XG, YG, ZG_best, color="black", alpha=0.20, edgecolor="none")
axR.set_xlabel(f"Z_{FX3}"); axR.set_ylabel(f"Z_{FY3}"); axR.set_zlabel(f"Z_{FZ3}")
axR.set_title(f"colored by s3 = z(Z_{FX3})+z(Z_{FY3})+z(Z_{FZ3})", fontsize=10)
fig.colorbar(sc3, ax=axR, fraction=0.03, pad=0.10).set_label("s3 value")
fig.suptitle(f"H 3-D scatter Z_{FX3}, Z_{FY3}, Z_{FZ3} with diagonal score s3", fontsize=12, y=1.02)
fig.tight_layout()
out = FIG_DIR / f"h_semantic_geom_cd4_scatter3d_Z{FX3}_Z{FY3}_Z{FZ3}_with_s3.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out); plt.show()
```

## Step 12 — Gene loadings + hierarchical clustering

```python  # cell 31
W = model.get_loadings().reindex(adata_cd4.var_names)
adata_cd4.varm["W_h_semantic_geom_cd4"] = W.values
print("W (loadings):", W.shape)

top_per_factor = {
    f"proj_{col}": W[col].nlargest(PER_PROJECTION_N_TOP).index.tolist() for col in W.columns
}
top_df = pd.DataFrame(top_per_factor)
top_df.index = [f"top_{i + 1}" for i in range(PER_PROJECTION_N_TOP)]
top_df.to_csv(TABLE_DIR / "h_semantic_geom_cd4_top_genes_per_factor.tsv", sep="\t")
print("\ntop genes per factor (head):")
with pd.option_context("display.max_columns", None, "display.width", 250, "display.max_colwidth", 30):
    print(top_df.head(15))
```

```python  # cell 32
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import seaborn as sns

max_vals = W.abs().max(axis=1)
top_idx = max_vals.sort_values(ascending=False).head(CLUSTER_N_TOP).index
subset = W.loc[top_idx]; subset = subset.loc[subset.values.std(axis=1) > 0]
top_idx = subset.index
dists = pdist(subset.values, metric="correlation"); dist_matrix = squareform(dists)
Z = linkage(dists, method="average")

best_k, best_score = 2, -1.0
for k in range(2, CLUSTER_MAX_K + 1):
    labels = fcluster(Z, t=k, criterion="maxclust")
    try:
        s = silhouette_score(dist_matrix, labels, metric="precomputed")
    except ValueError:
        continue
    if s > best_score: best_score, best_k = s, k
final_labels = fcluster(Z, t=best_k, criterion="maxclust")
print(f"hierarchical clusters: k={best_k} (silhouette={best_score:.3f}) over {len(top_idx)} genes")

clust = pd.DataFrame({"Cluster": final_labels, "max_loading": max_vals.loc[top_idx].values}, index=top_idx)
for c in sorted(clust["Cluster"].unique()):
    members = clust[clust["Cluster"] == c].sort_values("max_loading", ascending=False)
    print(f"\n=== cluster {c}  (n={len(members)}) ===")
    print(", ".join(members.head(TOP_PER_CLUSTER).index.tolist()))

corr = np.corrcoef(subset.values)
g = sns.clustermap(
    pd.DataFrame(corr, index=top_idx, columns=top_idx),
    row_linkage=Z, col_linkage=Z, cmap="vlag", center=0,
    xticklabels=False, yticklabels=False, figsize=(8, 8),
)
g.fig.suptitle(f"H semantic_geom CD4: top-{len(top_idx)} gene loading correlation (k={best_k})", y=1.01)
out = FIG_DIR / "h_semantic_geom_cd4_gene_hierarchy.png"
g.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
```
