# old/03_spatial_visium_download.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/03_spatial_visium_download.ipynb` and is never run._

# CTCL (MF) atlas — Visium spatial download & ingest

First step of the spatial arm. Fetches the **Li/Strobl/Poyner 2024** 10x Visium sections,
reads them with `scanpy.read_visium`, concatenates into one AnnData, and writes a clean,
**resolVI-ready** object (`data/Li2024_atlas/visium/ctcl_visium.h5ad`).

Data provenance (see `SPATIAL_data_map.md`): there is **no processed Visium h5ad** published;
both accessions host per-section **spaceranger output tarballs** on EMBL-EBI BioStudies:
- **E-MTAB-13614** — 8 CTCL sections `CTCL1..8_spaceranger_output.tar`
- **E-MTAB-14559** — 15 healthy sections `WS_D_SKNsp<id>.tar.gz`

> ⚠️ Heavy steps (download ~8.5 GB, extract, full read) run on the **GPU/compute kernel**, not the
> login node. `sc.read_visium` per section is light; concatenation of 23 sections is the memory
> step. resolVI training is a **later** notebook.

```python  # cell 1
from pathlib import Path
import json, tarfile, subprocess, urllib.request
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

np.random.seed(0)


def _resolve_nb_dir():
    start = Path(__file__).parent.resolve() if "__file__" in globals() else Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data starting from {start}")

NB_DIR   = _resolve_nb_dir()
DATA_DIR = NB_DIR / "data"
FIG_DIR  = NB_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)
VISIUM_DIR = DATA_DIR / "Li2024_atlas" / "visium"; VISIUM_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR    = VISIUM_DIR / "raw";     RAW_DIR.mkdir(exist_ok=True)     # downloaded tarballs
EXTRACT_DIR= VISIUM_DIR / "sections"; EXTRACT_DIR.mkdir(exist_ok=True) # untarred spaceranger outs
VISIUM_H5AD = VISIUM_DIR / "ctcl_visium.h5ad"                          # output

sc.settings.set_figure_params(dpi=80, facecolor="white", figsize=(5, 3))
sc.settings.figdir = str(FIG_DIR)

# BioStudies accessions (verified 2026-07-12; no processed h5ad exists)
ACC_CTCL    = "E-MTAB-13614"   # 8 CTCL Visium sections (ENA ERP155847)
ACC_HEALTHY = "E-MTAB-14559"   # 15 healthy Visium sections (ENA ERP165340)
FILES_URL    = "https://www.ebi.ac.uk/biostudies/files/{acc}/{fname}"          # downloads
FILELIST_URL = "https://www.ebi.ac.uk/biostudies/api/v1/files/{acc}?length=1000"  # manifest

print("scanpy", sc.__version__)
print("NB_DIR     ", NB_DIR)
print("VISIUM_DIR ", VISIUM_DIR)
```

```
scanpy 1.11.5
NB_DIR      /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
VISIUM_DIR  /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/Li2024_atlas/visium

/tmp/ipykernel_3307108/2973417336.py:37: FutureWarning: `__version__` is deprecated, use `importlib.metadata.version('scanpy')` instead
  print("scanpy", sc.__version__)
```

## 1. Resolve the file manifest

Query the BioStudies API for each accession's file list (authoritative). We keep only the
per-section spaceranger tarballs (`.tar` / `.tar.gz`), dropping the `.sdrf`/`.idf` metadata files
(fetched separately below).

```python  # cell 3
def fetch_manifest(acc):
    with urllib.request.urlopen(FILELIST_URL.format(acc=acc), timeout=60) as r:
        d = json.load(r)
    return [(f["path"], int(f.get("Size", 0) or 0)) for f in d["data"]]

def section_tarballs(acc):
    return [(p, sz) for p, sz in fetch_manifest(acc)
            if p.endswith(".tar") or p.endswith(".tar.gz")]

ctcl_files    = section_tarballs(ACC_CTCL)
healthy_files = section_tarballs(ACC_HEALTHY)

def _section_id(fname):
    # CTCL3_spaceranger_output.tar -> CTCL3 ; WS_D_SKNsp10264994.tar.gz -> WS_D_SKNsp10264994
    return fname.split("_spaceranger")[0].replace(".tar.gz", "").replace(".tar", "")

MANIFEST = (
    [dict(acc=ACC_CTCL, fname=p, size=sz, section=_section_id(p), condition="CTCL")    for p, sz in ctcl_files] +
    [dict(acc=ACC_HEALTHY, fname=p, size=sz, section=_section_id(p), condition="healthy") for p, sz in healthy_files]
)
manifest = pd.DataFrame(MANIFEST)
manifest["size_MB"] = (manifest["size"] / 1e6).round(0).astype(int)
print(f"{len(ctcl_files)} CTCL + {len(healthy_files)} healthy = {len(manifest)} sections, "
      f"{manifest['size_MB'].sum()/1000:.1f} GB total")
manifest[["condition", "section", "fname", "size_MB"]]
```

```
8 CTCL + 15 healthy = 23 sections, 8.5 GB total

   condition             section                         fname  size_MB
0       CTCL               CTCL7  CTCL7_spaceranger_output.tar      233
1       CTCL               CTCL1  CTCL1_spaceranger_output.tar      232
2       CTCL               CTCL6  CTCL6_spaceranger_output.tar      347
3       CTCL               CTCL8  CTCL8_spaceranger_output.tar      295
4       CTCL               CTCL5  CTCL5_spaceranger_output.tar      323
5       CTCL               CTCL4  CTCL4_spaceranger_output.tar      376
6       CTCL               CTCL3  CTCL3_spaceranger_output.tar      237
7       CTCL               CTCL2  CTCL2_spaceranger_output.tar      214
8    healthy  WS_D_SKNsp12767488     WS_D_SKNsp12767488.tar.gz      457
9    healthy  WS_D_SKNsp11418116     WS_D_SKNsp11418116.tar.gz      293
10   healthy  WS_D_SKNsp12398171     WS_D_SKNsp12398171.tar.gz      438
11   healthy  WS_D_SKNsp10264994     WS_D_SKNsp10264994.tar.gz      392
12   healthy  WS_D_SKNsp12767487     WS_D_SKNsp12767487.tar.gz      445
13   healthy  WS_D_SKNsp12767482     WS_D_SKNsp12767482.tar.gz      476
14   healthy  WS_D_SKNsp11418108     WS_D_SKNsp11418108.tar.gz      289
15   healthy  WS_D_SKNsp12767486     WS_D_SKNsp12767486.tar.gz      457
16   healthy  WS_D_SKNsp12767483     WS_D_SKNsp12767483.tar.gz      503
17   healthy  WS_D_SKNsp12767489     WS_D_SKNsp12767489.tar.gz      441
18   healthy  WS_D_SKNsp12767484     WS_D_SKNsp12767484.tar.gz      557
19   healthy  WS_D_SKNsp12767485     WS_D_SKNsp12767485.tar.gz      509
20   healthy  WS_D_SKNsp11418112     WS_D_SKNsp11418112.tar.gz      343
21   healthy  WS_D_SKNsp10921574     WS_D_SKNsp10921574.tar.gz      331
22   healthy  WS_D_SKNsp10264998     WS_D_SKNsp10264998.tar.gz      273
```

## 2. Download the tarballs

`wget -c` (resume-safe) into `raw/`. **~8.5 GB across 23 files** — run on the compute node.
Set `FETCH` to control scope: start with CTCL only (`"CTCL"`, ~2.4 GB), then `"all"` once verified.
Already-complete files are skipped.

```python  # cell 5
FETCH = "all"   # "CTCL" | "healthy" | "all"

def _select(df, which):
    if which == "all": return df
    return df[df["condition"] == ("CTCL" if which == "CTCL" else "healthy")]

def download(row):
    url = FILES_URL.format(acc=row.acc, fname=row.fname)
    dest = RAW_DIR / row.fname
    if dest.exists() and dest.stat().st_size >= row.size:
        print(f"  skip (have) {row.fname}"); return dest
    print(f"  wget {row.fname} ({row.size_MB} MB)")
    subprocess.run(["wget", "-q", "-c", "-O", str(dest), url], check=True)
    return dest

todo = _select(manifest, FETCH)
print(f"Downloading {len(todo)} files ({todo['size_MB'].sum()/1000:.1f} GB) -> {RAW_DIR}")
for row in todo.itertuples():
    download(row)
print("done")
```

```
Downloading 23 files (8.5 GB) -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/Li2024_atlas/visium/raw
  skip (have) CTCL7_spaceranger_output.tar
  skip (have) CTCL1_spaceranger_output.tar
  skip (have) CTCL6_spaceranger_output.tar
  skip (have) CTCL8_spaceranger_output.tar
  skip (have) CTCL5_spaceranger_output.tar
  skip (have) CTCL4_spaceranger_output.tar
  skip (have) CTCL3_spaceranger_output.tar
  skip (have) CTCL2_spaceranger_output.tar
  skip (have) WS_D_SKNsp12767488.tar.gz
  skip (have) WS_D_SKNsp11418116.tar.gz
  skip (have) WS_D_SKNsp12398171.tar.gz
  skip (have) WS_D_SKNsp10264994.tar.gz
  skip (have) WS_D_SKNsp12767487.tar.gz
  skip (have) WS_D_SKNsp12767482.tar.gz
  skip (have) WS_D_SKNsp11418108.tar.gz
  skip (have) WS_D_SKNsp12767486.tar.gz
  skip (have) WS_D_SKNsp12767483.tar.gz
  skip (have) WS_D_SKNsp12767489.tar.gz
  skip (have) WS_D_SKNsp12767484.tar.gz
  skip (have) WS_D_SKNsp12767485.tar.gz
  skip (have) WS_D_SKNsp11418112.tar.gz
  skip (have) WS_D_SKNsp10921574.tar.gz
  skip (have) WS_D_SKNsp10264998.tar.gz
done
```

### Section-level metadata (SDRF)

Small text files with per-section disease / sex / age. Parsed for annotation.

```python  # cell 7
def load_sdrf(acc):
    url = FILES_URL.format(acc=acc, fname=f"{acc}.sdrf.txt")
    df = pd.read_csv(url, sep="\t")
    keep = {"Source Name": "section"}
    for c in df.columns:
        cl = c.lower()
        if "characteristics[disease]" in cl: keep[c] = "disease"
        elif "characteristics[sex]" in cl:   keep[c] = "sex"
        elif "characteristics[age]" in cl:    keep[c] = "age"
    out = df[list(keep)].rename(columns=keep).drop_duplicates("section")
    return out

sdrf = pd.concat([load_sdrf(ACC_CTCL), load_sdrf(ACC_HEALTHY)], ignore_index=True)
sdrf = sdrf.set_index("section")
sdrf
```

```
age     sex                    disease
section                                                   
CTCL1                37  female  Cutaneous T-cell lymphoma
CTCL2                67    male  Cutaneous T-cell lymphoma
CTCL3                74  female  Cutaneous T-cell lymphoma
CTCL4                57    male  Cutaneous T-cell lymphoma
CTCL5                56    male  Cutaneous T-cell lymphoma
CTCL6                67    male  Cutaneous T-cell lymphoma
CTCL7                59  female  Cutaneous T-cell lymphoma
CTCL8                61    male  Cutaneous T-cell lymphoma
WS_D_SKNsp10264994   58    male                        NaN
WS_D_SKNsp10264998   61  female                        NaN
WS_D_SKNsp10921574   82    male                        NaN
WS_D_SKNsp11418108   41  female                        NaN
WS_D_SKNsp11418112   58    male                        NaN
WS_D_SKNsp11418116   61  female                        NaN
WS_D_SKNsp12398171   82    male                        NaN
WS_D_SKNsp12767482   56    male                        NaN
WS_D_SKNsp12767483   62  female                        NaN
WS_D_SKNsp12767484   62  female                        NaN
WS_D_SKNsp12767485   70  female                        NaN
WS_D_SKNsp12767486   56  female                        NaN
WS_D_SKNsp12767487   62  female                        NaN
WS_D_SKNsp12767488   62  female                        NaN
WS_D_SKNsp12767489   70  female                        NaN
```

## 3. Extract + read each section

Untar into `sections/<id>/`, locate the spaceranger `outs` dir (the folder containing `spatial/`),
and read with `sc.read_visium`. `library_id = section` so `uns['spatial']` keys stay unique for
plotting after concatenation.

```python  # cell 9
def extract(row):
    tar_path = RAW_DIR / row.fname
    out_dir = EXTRACT_DIR / row.section
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        with tarfile.open(tar_path, "r:*") as t:
            t.extractall(out_dir)
    return out_dir

def find_visium_dir(root):
    # spaceranger outs contains spatial/scalefactors_json.json + filtered_feature_bc_matrix(.h5)
    hits = list(root.rglob("spatial/scalefactors_json.json"))
    if not hits:
        raise FileNotFoundError(f"no spaceranger spatial/ under {root}")
    return hits[0].parent.parent  # the 'outs'-level dir

def read_section(row):
    vdir = find_visium_dir(extract(row))
    h5 = "filtered_feature_bc_matrix.h5"
    ad = sc.read_visium(vdir, count_file=h5 if (vdir / h5).exists() else None,
                        library_id=row.section)
    ad.var_names_make_unique()
    ad.obs["section"] = row.section
    ad.obs["condition"] = row.condition
    return ad

FETCHED = sorted({p.name for p in RAW_DIR.glob("*.tar*")})
sections = manifest[manifest["fname"].isin(FETCHED)]
print(f"reading {len(sections)} downloaded sections")
adatas = {row.section: read_section(row) for row in sections.itertuples()}
for sid, ad in adatas.items():
    print(f"  {sid:20s} {ad.n_obs:6d} spots x {ad.n_vars} genes")
```

```
reading 23 downloaded sections

/tmp/ipykernel_3307108/2336214416.py:20: FutureWarning: Use `squidpy.read.visium` instead.
  ad = sc.read_visium(vdir, count_file=h5 if (vdir / h5).exists() else None,
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/anndata/_core/anndata.py:1808: UserWarning: Variable names are not unique. To make them unique, call `.var_names_make_unique`.
  utils.warn_names_duplicates("var")
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/anndata/_core/anndata.py:1808: UserWarning: Variable names are not unique. To make them unique, call `.var_names_make_unique`.
  utils.warn_names_duplicates("var")
/tmp/ipykernel_3307108/2336214416.py:20: FutureWarning: Use `squidpy.read.visium` instead.
  ad = sc.read_visium(vdir, count_file=h5 if (vdir / h5).exists() else None,
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/anndata/_core/anndata.py:1808: UserWarning: Variable names are not unique. To make them unique, call `.var_names_make_unique`.
  utils.warn_names_duplicates("var")
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/anndata/_core/anndata.py:1808: UserWarning: Variable names are not unique. To make them unique, call `.var_names_make_unique`.
  utils.warn_names_duplicates("var")
/tmp/ipykernel_3307108/2336214416.py:20: FutureWarning: Use `squidpy.read.visium` instead.
  ad = sc.read_visium(vdir, count_file=h5 if (vdir / h5).exists() else None,
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/anndata/_core/anndata.py:1808: UserWarning: Variable names are not unique. To make them unique, call `.var_names_make_unique`.
  utils.warn_names_duplicates("var")
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/anndata/_core/anndata.py:1808: UserWarning: Variable names are not unique. To make them unique, call `.var_names_make_unique`.
  utils.warn_names_duplicates("var")
/tmp/ipykernel_3307108/2336214416.py:20: FutureWarning: Use `squidpy.read.visium` instead.
  ad = sc.read_visium(vdir, count_file=h5 if (vdir / h5).exists() else None,
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/anndata/_core/anndata.py:1808: UserWarning: Variable names are not unique. To make them unique, call `.var_names_make_unique`.
  utils.warn_names_duplicates("var")
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/anndata/_core/anndata.py:1808: UserWarning: Variable names are not unique. To make them unique, call `.var_names_make_unique`.
  utils.warn_names_duplicates("var")
/tmp/ipykernel_3307108/2336214416.py:20: FutureWarning: Use `squidpy.read.visium` instead.
  ad = sc.read_visium(vdir, count_file=h5 if (vdir / h5).exists() else None,
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/anndata/_core/anndata.py:1808: UserWarning: Variable names are not unique. To make them unique, call `.var_names_make_unique`.
  utils.warn_names_duplicates("var")
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/anndata/_core/anndata.py:1808: UserWarning: Variable names are not unique. To make them unique, call `.var_names_make_unique`.
  utils.warn_names_duplicates("var")
/tmp/ipykernel_3307108/2336214416.py:20: FutureWarning: Use `squidpy.read.visium` instead.
  ad = sc.read_visium(vdir, count_file=h5 if (vdir / h5).exists() else None,
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/anndata/_core/anndata.py:1808: UserWarning: Variable names are not unique. To make them unique, call `.var_names_make_unique`.
  utils.warn_names_duplicates("var")
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/annd
... [13006 more chars truncated]
```

## 4. Concatenate + annotate

Inner-join on genes (identical spaceranger reference), keep per-section `uns['spatial']` (images +
scalefactors) so `sc.pl.spatial` still works, and merge SDRF metadata.

```python  # cell 11
import anndata as ad_

merged_spatial = {}
for adx in adatas.values():
    merged_spatial.update(adx.uns.get("spatial", {}))

# section/condition already on each obs; concat a list and keep obsm/var aligned
adata = ad_.concat(list(adatas.values()), join="inner", index_unique="-", merge="unique")
adata.uns["spatial"] = merged_spatial

# section-level metadata
meta = sdrf.reindex(adata.obs["section"].values)
for c in meta.columns:
    adata.obs[c] = meta[c].values
adata.obs["section"] = adata.obs["section"].astype("category")
adata.obs["condition"] = adata.obs["condition"].astype("category")

print(adata)
print("\nsections:"); print(adata.obs["section"].value_counts())
```

```
AnnData object with n_obs × n_vars = 79911 × 18085
    obs: 'in_tissue', 'array_row', 'array_col', 'section', 'condition', 'age', 'sex', 'disease'
    var: 'gene_ids', 'feature_types', 'genome'
    uns: 'spatial'
    obsm: 'spatial'

sections:
section
WS_D_SKNsp12398171    4992
WS_D_SKNsp11418112    4992
WS_D_SKNsp11418116    4992
WS_D_SKNsp11418108    4992
WS_D_SKNsp10921574    4992
WS_D_SKNsp10264998    4992
WS_D_SKNsp10264994    4992
WS_D_SKNsp12767486    4992
WS_D_SKNsp12767483    4992
WS_D_SKNsp12767484    4992
WS_D_SKNsp12767485    4992
WS_D_SKNsp12767488    4992
WS_D_SKNsp12767487    4992
WS_D_SKNsp12767489    4992
WS_D_SKNsp12767482    4992
CTCL5                 1040
CTCL6                  811
CTCL7                  712
CTCL4                  658
CTCL8                  629
CTCL2                  487
CTCL3                  389
CTCL1                  305
Name: count, dtype: int64
```

## 5. QC per section

Spots, median counts/genes, mito fraction per section; `sc.pl.spatial` of one CTCL section
(sanity of coords + image).

```python  # cell 13
adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None)

qc = (adata.obs.groupby("section", observed=True)
      .agg(spots=("total_counts", "size"),
           median_counts=("total_counts", "median"),
           median_genes=("n_genes_by_counts", "median"),
           pct_mt=("pct_counts_mt", "median"))
      .round(1))
print(qc)

_ctcl = [s for s in adata.obs["section"].cat.categories if str(s).startswith("CTCL")]
if _ctcl:
    s0 = _ctcl[0]
    sub = adata[adata.obs["section"] == s0].copy()
    sc.pl.spatial(sub, library_id=s0, color="total_counts", size=1.4,
                  title=f"{s0}: total counts/spot", show=False)
    plt.tight_layout(); plt.savefig(FIG_DIR / f"visium_qc_{s0}.png", dpi=120); plt.show()
```

```
spots  median_counts  median_genes  pct_mt
section                                                       
CTCL1                 305        22501.0        8281.0     2.6
CTCL2                 487         8719.0        4811.0     2.3
CTCL3                 389        15072.0        6217.0     2.2
CTCL4                 658        14484.5        6610.0     2.4
CTCL5                1040         4142.0        2514.0     2.2
CTCL6                 811         7245.0        3747.0     2.3
CTCL7                 712         4422.0        3032.0     3.4
CTCL8                 629        12390.0        5484.0     3.5
WS_D_SKNsp10264994   4992          697.0         490.0     9.3
WS_D_SKNsp10264998   4992          258.0         194.0     5.4
WS_D_SKNsp10921574   4992          601.5         429.0     9.5
WS_D_SKNsp11418108   4992          324.0         258.0     7.5
WS_D_SKNsp11418112   4992          107.0          83.5    10.7
WS_D_SKNsp11418116   4992          140.0         119.0     5.3
WS_D_SKNsp12398171   4992          123.0         104.0     8.9
WS_D_SKNsp12767482   4992          174.0         144.0     7.8
WS_D_SKNsp12767483   4992          199.0         162.0     8.4
WS_D_SKNsp12767484   4992          208.0         161.0    10.4
WS_D_SKNsp12767485   4992          303.0         231.0     8.2
WS_D_SKNsp12767486   4992           95.0          82.0     7.8
WS_D_SKNsp12767487   4992           75.0          64.0     9.1
WS_D_SKNsp12767488   4992          147.0         113.5    10.7
WS_D_SKNsp12767489   4992          168.0         133.0     9.2

/tmp/ipykernel_3307108/225161191.py:16: FutureWarning: Use `squidpy.pl.spatial_scatter` instead.
  sc.pl.spatial(sub, library_id=s0, color="total_counts", size=1.4,

<Figure size 400x240 with 2 Axes>
```

## 6. resolVI-ready object + write

`scvi.external.RESOLVI` needs, per section: **raw counts** in a layer, **`obsm['X_spatial']`**
(default `spatial_rep`), and a **section batch key** for per-slice neighbor graphs. read_visium
gives raw counts in `.X` and coords in `.obsm['spatial']`; we stash both explicitly.

⚠️ resolVI targets *single-cell-resolved* spatial data; Visium spots are multi-cell, so its
neighbor-diffusion denoising is a partial fit — documented, not resolved here.

```python  # cell 15
adata.layers["counts"] = adata.X.copy()          # raw counts (read_visium X is raw)
adata.obsm["X_spatial"] = adata.obsm["spatial"].astype(float)

assert adata.layers["counts"].max() == int(adata.layers["counts"].max()), "counts not integer"
assert adata.obsm["X_spatial"].shape[1] == 2
assert adata.obs["section"].dtype.name == "category"

adata.write_h5ad(VISIUM_H5AD)
print("wrote", VISIUM_H5AD)
print(adata)
```

```
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/Li2024_atlas/visium/ctcl_visium.h5ad
AnnData object with n_obs × n_vars = 79911 × 18085
    obs: 'in_tissue', 'array_row', 'array_col', 'section', 'condition', 'age', 'sex', 'disease', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts', 'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt'
    var: 'gene_ids', 'feature_types', 'genome', 'mt', 'n_cells_by_counts', 'mean_counts', 'log1p_mean_counts', 'pct_dropout_by_counts', 'total_counts', 'log1p_total_counts'
    uns: 'spatial'
    obsm: 'spatial', 'X_spatial'
    layers: 'counts'
```

## 7. resolVI setup dry-check (no training)

Confirm the object is resolVI-consumable: run `setup_anndata` on **one small section** so the
per-section neighbor graph builds and `index_neighbor` / `distance_neighbor` populate. **No
training here** — the real fit is a later GPU notebook.

```python  # cell 17
from scvi.external import RESOLVI

s0 = adata.obs["section"].cat.categories[0]
probe = adata[adata.obs["section"] == s0].copy()
RESOLVI.setup_anndata(probe, layer="counts", batch_key="section", prepare_data=True,
                      prepare_data_kwargs={"spatial_rep": "X_spatial"})
print("index_neighbor  ", probe.obsm["index_neighbor"].shape)
print("distance_neighbor", probe.obsm["distance_neighbor"].shape)
print("resolVI setup OK on", s0)
```

```
INFO     Generating sequential column names                                                                        
INFO     Generating sequential column names                                                                        
index_neighbor   (305, 10)
distance_neighbor (305, 10)
resolVI setup OK on CTCL1

/home/projects/nyosef/zvise/scvi-tools-neural-nmf/src/scvi/data/fields/_dataframe_field.py:187: UserWarning: Category 0 in adata.obs['_scvi_ind_x'] has fewer than 3 cells. Models may not train properly.
  categorical_mapping = _make_column_categorical(

The Kernel crashed while executing code in the current cell or a previous cell. 

Please review the code in the cell(s) to identify a possible cause of the failure. 

Click <a href='https://aka.ms/vscodeJupyterKernelCrash'>here</a> for more info. 

View Jupyter <a href='command:jupyter.viewOutput'>log</a> for further details.
```

## Summary

- **Sections:** 8 CTCL (`E-MTAB-13614`) + 15 healthy (`E-MTAB-14559`) 10x Visium — spaceranger
  tarballs from BioStudies (no processed h5ad exists upstream).
- **Output:** `data/Li2024_atlas/visium/ctcl_visium.h5ad` — raw counts in `layers['counts']`,
  coords in `obsm['spatial']`/`obsm['X_spatial']`, per-section images in `uns['spatial']`,
  `section` / `condition` / disease-sex-age obs.
- **resolVI-ready:** `setup_anndata(layer='counts', batch_key='section',
  prepare_data_kwargs={'spatial_rep':'X_spatial'})` verified on one section.
- **Gaps:** no per-spot cell-type labels (`labels_key`) — Visium spots are multi-cell; labels
  would come from deconvolution later. resolVI resolution mismatch noted above.

**Next:** resolVI training notebook (GPU), and/or join to the scRNA atlas for niche analysis.
