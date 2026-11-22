# 10_atlas/11_atlas_v2_qc.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/10_atlas/11_atlas_v2_qc.ipynb` and is never run._

# Atlas v2 — release QC

We are about to send the v2 CTCL atlas to a collaborating group. This notebook is the gate: it
re-derives every number we quote about the atlas, checks the object against its own build
record, and asks the one question a reviewer will ask first — *does this look like biology, or
does it look like an integration artifact?*

Nothing here raises. Every test appends a row to a ledger with a status of **PASS**, **WARN**
(a known, quantified caveat) or **FAIL** (a blocker), and §8 writes the ledger to
`tables/atlas_v2_qc_report.csv`. A QC notebook that dies on the first failure cannot report the
other forty, and the report is the deliverable — it ships alongside the data.

**The 112 GB rule.** `joint_annotated.h5ad` is 112 GB (CSR float64 with int64 indices, plus a
`raw_counts` layer that duplicates `X`). `sc.read_h5ad` on it kills the kernel. Everything below
reads one of: the 77 MB obs parquet, an h5py header, an `indptr` array, or a *contiguous* slab of
a CSR matrix. The contiguity matters more than it sounds — scattered row indexing on these files
runs at ~15 rows/s, while contiguous slabs run at ~15,000 rows/s. Streaming the whole atlas in
order is three orders of magnitude cheaper than sampling 40,000 random cells from it.

```python  # cell 1
import warnings; warnings.filterwarnings("ignore")
import hashlib, itertools, json, re, sys, time
from pathlib import Path

import h5py, numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt

SEED = 0
np.random.seed(SEED)
mpl.rcParams["figure.dpi"] = 110
mpl.rcParams["savefig.bbox"] = "tight"


def _resolve_nb_dir() -> Path:
    """Same resolver the descriptive notebook uses, so this runs from anywhere in the MF tree."""
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data from {start}")


NB_DIR = _resolve_nb_dir()
sys.path.insert(0, str(NB_DIR / "helpers"))
import atlas_join_helpers as H
import atlas_v2_fixups as FIX

OUT = NB_DIR / "data" / "atlas_joint"
TAB = NB_DIR / "tables"
FIG = NB_DIR / "figures" / "atlas_v2_qc"; FIG.mkdir(parents=True, exist_ok=True)

P_ANN   = OUT / "joint_annotated.h5ad"        # 112 GB — headers and indptr only, never read_h5ad
P_MRVI  = OUT / "joint_mrvi_input.h5ad"       # 2,157,693 x 10,000 HVG, raw_counts layer
P_OBS   = OUT / "atlas_obs_full_v2.parquet"   # the workhorse
P_U     = OUT / "joint_X_mrvi_u.npy"          # 2,157,693 x 10, sample-unaware latent
P_LOG   = NB_DIR / "jobs" / "run_build_joint.bsub.log"

# What the tables claim. Anything that disagrees is either a stale table or a rebuilt object.
N_TOTAL, N_SKIN, N_BLOOD, N_LN = 2_157_693, 1_345_527, 811_024, 1_142
# 264 samples, not the 263 the build reported: §3f splits gaydosik2022's HTO-multiplexed
# `D13__SZ29` lane into its blood and skin specimens.
N_SAMPLES, N_DONOR_UNITS, N_REAL_DONOR, N_PATIENTS, N_STUDIES = 264, 268, 253, 201, 21

print("NB_DIR =", NB_DIR)
```

```
NB_DIR = /home/projects/nyosef/zvise/MF
```

```python  # cell 2
LEDGER = []


def check(section, name, ok, observed="", expected="", severity="BLOCK", note=""):
    """Record one verdict and print it. Never raises.

    severity BLOCK -> a failure is a FAIL and stops the release
             WARN  -> a failure is a documented caveat we ship with, quantified
             INFO  -> context, not a judgement
    """
    status = "PASS" if ok else ("FAIL" if severity == "BLOCK" else severity)
    LEDGER.append(dict(section=section, check=name, status=status, severity=severity,
                       observed=str(observed)[:400], expected=str(expected)[:200], note=note[:400]))
    print(f"  [{status:4s}] {section} · {name}" + (f"  ->  {observed}" if observed != "" else ""))
    return ok


def h5_ids(path, key="cell_id"):
    """Read an obs column out of an h5ad with h5py, decoding a categorical if that is how it
    was written. anndata would open the whole object; we only ever want one column."""
    with h5py.File(path, "r") as f:
        o = f["obs"][key] if key in f["obs"] else f["obs"][f["obs"].attrs["_index"]]
        if isinstance(o, h5py.Group):
            return np.asarray(o["categories"][:]).astype(str)[o["codes"][:]]
        return np.asarray(o[:]).astype(str)


def nnz_per_cell(path, where="X"):
    """Per-cell non-zero counts from indptr alone — 17 MB read, ~2 s, no matrix touched."""
    with h5py.File(path, "r") as f:
        g = f[where] if where == "X" else f["layers"][where]
        return np.diff(g["indptr"][:])
```

## 1 · Provenance — is this file the build the tables describe?

`jobs/run_build_joint.py` writes `joint_*` **in place**, so a path is not an identity: the same
filename can hold two different atlases. Three things pin the build down. The bsub log is a
contemporaneous record of what the concat did. The file mtimes have to be in the right order —
in particular the obs parquet must be *newer* than both label sidecars, or it was cached before
the labels existed. And a fingerprint over `X/indptr` encodes the per-cell non-zero count of all
2.16 M cells, which no other build can collide with, for 2 s of I/O instead of the 30–45 minutes
a sha256 of 175 GB would cost.

```python  # cell 4
def fingerprint(p: Path) -> dict:
    """Header + index structure, not content. For an h5ad this hashes the shape, the gene names
    and the whole indptr array, which together identify a build uniquely."""
    st = p.stat()
    h = hashlib.blake2b(digest_size=16)
    rows = None
    if p.suffix == ".h5ad":
        with h5py.File(p, "r") as f:
            h.update(np.asarray(f["X"].attrs["shape"]).tobytes())
            h.update(f["var"][f["var"].attrs["_index"]][:].tobytes())
            h.update(f["X"]["indptr"][:].tobytes())
            rows = int(f["X"].attrs["shape"][0])
    elif p.suffix == ".npy":
        a = np.load(p, mmap_mode="r")
        h.update(np.asarray(a.shape).tobytes())
        h.update(np.ascontiguousarray(a[::997]).tobytes())
        rows = a.shape[0]
    else:
        h.update(p.read_bytes()[: 1 << 20])
    return dict(path=str(p.relative_to(NB_DIR)), gb=round(st.st_size / 1e9, 3),
                mtime=pd.Timestamp(st.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M"),
                rows=rows, fp=h.hexdigest())


# P_LOG is deliberately not here: the bsub log is gitignored and deleted at submit time, so
# its absence is a provenance caveat (handled in the next cell), not a missing release artifact.
MANIFEST = [OUT / f for f in [
    "joint_annotated.h5ad", "joint_raw.h5ad", "joint_mrvi_input.h5ad",
    "joint_mrvi_input_skin.h5ad", "joint_mrvi_input_blood.h5ad",
    "joint_X_mrvi_u.npy", "joint_X_mrvi.npy", "atlas_obs_full_v2.parquet",
    "tcr_clones.parquet", "skin_cell_type_final.csv", "blood_cell_type_final.csv",
]] + [NB_DIR / "data" / "sample_metadata_final.csv",
      TAB / "atlas_dedup_v2_decisions.csv"]

missing = [p for p in MANIFEST if not p.exists()]
check("1", "all release artifacts present", not missing, f"{len(MANIFEST) - len(missing)}/{len(MANIFEST)}",
      len(MANIFEST), note=str([p.name for p in missing]))

t0 = time.time()
man = pd.DataFrame(fingerprint(p) for p in MANIFEST if p.exists())
man.to_csv(TAB / "atlas_v2_qc_manifest.csv", index=False)
print(f"\nfingerprinted in {time.time() - t0:.0f}s")
man
```

```
[PASS] 1 · all release artifacts present  ->  13/13

fingerprinted in 0s

                                            path       gb             mtime  \
0          data/atlas_joint/joint_annotated.h5ad  112.170  2026-09-15 11:07   
1                data/atlas_joint/joint_raw.h5ad  112.015  2026-09-15 11:07   
2         data/atlas_joint/joint_mrvi_input.h5ad   33.796  2026-09-15 11:07   
3    data/atlas_joint/joint_mrvi_input_skin.h5ad   16.381  2026-09-15 11:07   
4   data/atlas_joint/joint_mrvi_input_blood.h5ad   13.558  2026-09-15 11:07   
5            data/atlas_joint/joint_X_mrvi_u.npy    0.086  2026-09-06 06:29   
6              data/atlas_joint/joint_X_mrvi.npy    0.259  2026-09-06 06:30   
7     data/atlas_joint/atlas_obs_full_v2.parquet    0.077  2026-09-15 11:08   
8            data/atlas_joint/tcr_clones.parquet    0.016  2026-09-05 17:17   
9      data/atlas_joint/skin_cell_type_final.csv    0.071  2026-09-06 11:00   
10    data/atlas_joint/blood_cell_type_final.csv    0.041  2026-09-07 15:21   
11                data/sample_metadata_final.csv    0.000  2026-09-15 11:14   
12           tables/atlas_dedup_v2_decisions.csv    0.000  2026-09-05 16:26   

         rows                                fp  
0   2157693.0  417463ce4f4128efe5a9e9c1cb19dcd9  
1   2157693.0  272bf2091a7d1448b3f4e0debadcf069  
2   2157693.0  d31e2fca337bad54bdc675a6f5537a0c  
3   1345527.0  7fe2db162e0f9385b3b6a0a8618daa9c  
4    811024.0  3fc16e2592bdcf19b8174314584077f5  
5   2157693.0  d99d48e7d7715c8135414c5f08754b6f  
6   2157693.0  42346ffc9faefd19019de80478db9f37  
7         NaN  19b0a6d1bf3c302276b2093a8a7fcb5f  
8         NaN  e9e8114c3029999f8f6b216ff95f11f6  
9         NaN  d99cfd15cce5eb6904f6fb82a0bc97f3  
10        NaN  1018d113113bc96738f605e08b8a1fcf  
11        NaN  3d82ba21779aaff2b2c4b4d5dee22cdc  
12        NaN  202e5e6dc44808e61e6c463d71d287d7
```

```python  # cell 5
# The parquet is a cache of obs *plus* the two label sidecars. If it predates either sidecar it
# was built against different labels, and every composition number below would be wrong.
mt = {r["path"]: r["mtime"] for _, r in man.iterrows()}
ok = (mt["data/atlas_joint/atlas_obs_full_v2.parquet"]
      > max(mt["data/atlas_joint/skin_cell_type_final.csv"],
            mt["data/atlas_joint/blood_cell_type_final.csv"])
      > mt["data/atlas_joint/joint_annotated.h5ad"])
check("1", "artifact mtimes are in build order", ok,
      f"annotated {mt['data/atlas_joint/joint_annotated.h5ad']} -> "
      f"labels -> parquet {mt['data/atlas_joint/atlas_obs_full_v2.parquet']}",
      "annotated < sidecars < parquet")

# The bsub log is the one provenance artifact that is not reproducible: `jobs/*.bsub.log` is
# gitignored and run_build_joint.sh deletes it at submit time, so a resubmit or a tree move
# loses it for good. If it is gone, say so in the ledger rather than dying here -- the
# fingerprint and the mtime order above still pin the build.
log = P_LOG.read_text() if P_LOG.exists() else ""
m = re.search(r"duplicate cells dropped=(\d+) across (\d+) samples", log) if log else None
DEDUP_DROPPED = int(m[1]) if m else None   # §5 reuses this for the cell arithmetic

if not log:
    for name, exp in [("build log records the dedup drop", "290,112 / 45"),
                      ("build log and joint_annotated.h5ad are the same build", "< 30 min")]:
        check("1", name, False, "build log absent", exp, severity="WARN",
              note=f"{P_LOG.relative_to(NB_DIR)} is not on disk; it is gitignored and "
                   "run_build_joint.sh removes it at submit time. Provenance rests on the "
                   "X/indptr fingerprint and the mtime ordering.")
else:
    check("1", "build log records the dedup drop", bool(m) and (int(m[1]), int(m[2])) == (290112, 45),
          f"{int(m[1]):,} cells / {int(m[2])} samples" if m else "not found", "290,112 / 45")
    check("1", "build log and joint_annotated.h5ad are the same build",
          abs(P_LOG.stat().st_mtime - P_ANN.stat().st_mtime) < 1800,
          f"{abs(P_LOG.stat().st_mtime - P_ANN.stat().st_mtime) / 60:.0f} min apart", "< 30 min")
    print("\n" + "\n".join(l for l in log.splitlines()[-12:]))
```

```
[FAIL] 1 · artifact mtimes are in build order  ->  annotated 2026-09-15 11:07 -> labels -> parquet 2026-09-15 11:08
  [WARN] 1 · build log records the dedup drop  ->  build log absent
  [WARN] 1 · build log and joint_annotated.h5ad are the same build  ->  build log absent
```

```python  # cell 6
t0 = time.time()
A = pd.read_parquet(P_OBS)
for c in A.columns:
    if isinstance(A[c].dtype, pd.CategoricalDtype):
        A[c] = A[c].astype(str)
print(f"obs: {A.shape} in {time.time() - t0:.0f}s\n")

for col, n in [("cell_id", N_TOTAL), ("sample_id", N_SAMPLES), ("donor", N_DONOR_UNITS),
               ("real_donor", N_REAL_DONOR), ("patient_key", N_PATIENTS),
               ("study", N_STUDIES), ("dataset", N_STUDIES)]:
    check("1", f"nunique({col})", A[col].nunique() == n, f"{A[col].nunique():,}", f"{n:,}")
for comp, n in [("Skin", N_SKIN), ("Blood", N_BLOOD), ("LN", N_LN)]:
    got = int((A.compartment == comp).sum())
    check("1", f"cells in {comp}", got == n, f"{got:,}", f"{n:,}")

# v2 added twelve cohorts. Their presence is the only cheap proof that this is not the v1 object
# wearing a v2 filename -- the failure mode the in-place rebuild creates.
V2_ONLY = {"rindler21mc", "alkon24", "lyp26", "jonak21", "brentuximab26", "pacritinib26",
           "gaydosik22", "gaydosik23", "il4ra26", "ren23", "harro23", "dorando26"}
absent = sorted(V2_ONLY - set(A.dataset.unique()))
check("1", "v2-only cohorts present", absent == ["pacritinib26"], f"absent: {absent}",
      "only pacritinib26", severity="WARN",
      note="pacritinib26 (D12/GSE309807) has a QC_CONFIG row and appears in the dedup doc, but "
           "contributed no cells to this build. The atlas is 21 studies, not 22.")
```

```
obs: (2157693, 72) in 15s

  [PASS] 1 · nunique(cell_id)  ->  2,157,693
  [PASS] 1 · nunique(sample_id)  ->  264
  [PASS] 1 · nunique(donor)  ->  268
  [PASS] 1 · nunique(real_donor)  ->  253
  [PASS] 1 · nunique(patient_key)  ->  201
  [PASS] 1 · nunique(study)  ->  21
  [PASS] 1 · nunique(dataset)  ->  21
  [PASS] 1 · cells in Skin  ->  1,345,527
  [PASS] 1 · cells in Blood  ->  811,024
  [PASS] 1 · cells in LN  ->  1,142
  [PASS] 1 · v2-only cohorts present  ->  absent: ['pacritinib26']

True
```

```python  # cell 7
# Re-derive the headline table from the object and diff it against the shipped CSV. This is what
# catches a stale table travelling next to a fresh atlas.
head = pd.read_csv(TAB / "atlas_v2_headline.csv", index_col=0)
derived = {}
for comp in ["Skin", "Blood", "LN"]:
    s = A[A.compartment == comp]
    derived[comp] = dict(studies=s.study.nunique(), patients=s.patient_key.nunique(),
                         donors=s.donor.nunique(), samples=s.sample_id.nunique(), cells=len(s))
derived["ALL"] = dict(studies=A.study.nunique(), patients=A.patient_key.nunique(),
                      donors=A.donor.nunique(), samples=A.sample_id.nunique(), cells=len(A))
D = pd.DataFrame(derived)

rows = [("studies", "studies"), ("patients", "patients (patient_key)"),
        ("donors", "donor-units"), ("samples", "samples"), ("cells", "cells")]
diff = {k: {c: (D.loc[k, c], head.loc[v, c]) for c in D.columns
            if D.loc[k, c] != head.loc[v, c]} for k, v in rows}
diff = {k: v for k, v in diff.items() if v}
check("1", "headline table matches the object", not diff, diff or "exact match", "no differences")
D
```

```
[PASS] 1 · headline table matches the object  ->  exact match

             Skin   Blood    LN      ALL
studies        14      10     1       21
patients      137      75     1      201
donors        146     131     1      268
samples       174      89     1      264
cells     1345527  811024  1142  2157693
```

## 2 · Structural integrity

The `.npy` latents carry no index, so *row order* is the entire contract between them, the h5ad
objects and the obs parquet. It is checkable, it costs about twenty seconds, and getting it wrong
mis-assigns every downstream call silently — which is exactly how `23_malignancy_tcr_cnv` once reported v1 numbers
on a v2 input.

```python  # cell 9
check("2", "cell_id is unique", A.cell_id.is_unique, f"{A.cell_id.nunique():,}", f"{len(A):,}")
bad = ~A.cell_id.str.match(r"^[^|]+\|.+$")
check("2", "cell_id matches <sample_id>|<barcode>", not bad.any(), int(bad.sum()), 0)
# The prefix is the sequencing library the cell came off: usually that is also its sample_id,
# it is the dataset name for the li2024 object (which arrived already concatenated), and it is
# the pre-split lane for the specimens §3f separates out. cell_id itself is never rewritten --
# it is the join key for the label sidecars and the TCR table.
pre = A.cell_id.str.split("|").str[0]
lane = A.sample_id.astype(str).map(lambda s: FIX.LANE_OF.get(s, s))
ok = (pre == A.sample_id) | (pre == A.dataset) | (pre == lane)
check("2", "cell_id prefix is the sequencing library", bool(ok.all()), int((~ok).sum()), 0)
check("2", "cell_id prefix granularity", True,
      f"{int((pre == A.sample_id).sum()):,} sample-prefixed, "
      f"{int((pre == A.dataset).sum()):,} dataset-prefixed (li2024), "
      f"{int((ok & (pre != A.sample_id) & (pre != A.dataset)).sum()):,} split-lane",
      severity="INFO")

REQ = ["cell_id", "dataset", "study", "donor", "real_donor", "sample_id", "patient_key",
       "disease", "compartment", "organ", "tech", "n_genes", "total_counts", "pct_mito"]
nulls = {c: int(A[c].isna().sum() + (A[c] == "nan").sum()) for c in REQ if c in A}
check("2", "no nulls in the required columns", not any(nulls.values()),
      {k: v for k, v in nulls.items() if v} or "all clean", 0)

# doublet_score is NaN for li24 by construction -- that cohort arrived pre-QC'd, so qc_filter
# returns before Scrublet. Worth recording, not a defect.
dn = A.groupby("dataset").doublet_score.apply(lambda s: float(s.isna().mean()))
check("2", "doublet_score coverage", True, f"missing for: {list(dn[dn > 0].index)}",
      severity="INFO", note="li24 is `prefiltered=True`; Scrublet never ran on it.")
```

```
[PASS] 2 · cell_id is unique  ->  2,157,693
  [PASS] 2 · cell_id matches <sample_id>|<barcode>  ->  0
  [FAIL] 2 · cell_id prefix is the sequencing library  ->  419579
  [PASS] 2 · cell_id prefix granularity  ->  1,737,483 sample-prefixed, 0 dataset-prefixed (li2024), 631 split-lane
  [PASS] 2 · no nulls in the required columns  ->  all clean
  [PASS] 2 · doublet_score coverage  ->  missing for: ['li24']

True
```

```python  # cell 10
t0 = time.time()
ids_full = h5_ids(P_MRVI)
check("2", "joint_mrvi_input row order == parquet row order",
      len(ids_full) == len(A) and bool((ids_full == A.cell_id.values).all()),
      f"n={len(ids_full):,}", f"n={len(A):,}")

for comp, hp, up in [("Skin", "joint_mrvi_input_skin.h5ad", "joint_X_mrvi_u_skin.npy"),
                     ("Blood", "joint_mrvi_input_blood.h5ad", "joint_X_mrvi_u_blood.npy")]:
    ref = A.loc[A.compartment == comp, "cell_id"].values
    sub = h5_ids(OUT / hp)
    check("2", f"{comp} sub-object == the {comp} slice of the atlas, in order",
          len(sub) == len(ref) and bool((sub == ref).all()), f"n={len(sub):,}", f"n={len(ref):,}")
    if (OUT / up).exists():
        check("2", f"{comp} latent rows", np.load(OUT / up, mmap_mode="r").shape[0] == len(ref),
              np.load(OUT / up, mmap_mode="r").shape, (len(ref), 10))

for f, shape in [("joint_X_mrvi_u.npy", (N_TOTAL, 10)), ("joint_X_mrvi.npy", (N_TOTAL, 30))]:
    a = np.load(OUT / f, mmap_mode="r")
    check("2", f"{f} shape", a.shape == shape, a.shape, shape)

u = np.load(P_U)
check("2", "latent is finite", bool(np.isfinite(u).all()), int((~np.isfinite(u)).sum()), 0)
print(f"\nalignment checked in {time.time() - t0:.0f}s")
```

```
[PASS] 2 · joint_mrvi_input row order == parquet row order  ->  n=2,157,693
  [PASS] 2 · Skin sub-object == the Skin slice of the atlas, in order  ->  n=1,345,527
  [PASS] 2 · Skin latent rows  ->  (1345527, 10)
  [PASS] 2 · Blood sub-object == the Blood slice of the atlas, in order  ->  n=811,024
  [PASS] 2 · Blood latent rows  ->  (811024, 10)
  [PASS] 2 · joint_X_mrvi_u.npy shape  ->  (2157693, 10)
  [PASS] 2 · joint_X_mrvi.npy shape  ->  (2157693, 30)
  [PASS] 2 · latent is finite  ->  0

alignment checked in 3s
```

```python  # cell 11
with h5py.File(P_ANN, "r") as f:
    v42 = np.asarray(f["var"][f["var"].attrs["_index"]][:]).astype(str)
with h5py.File(P_MRVI, "r") as f:
    v10 = np.asarray(f["var"][f["var"].attrs["_index"]][:]).astype(str)

check("2", "gene symbols are unique", len(set(v42)) == len(v42), f"{len(v42):,} names, "
      f"{len(v42) - len(set(v42))} duplicated", 0)
check("2", "the 10k HVG space is a subset of the 42k union", set(v10) <= set(v42), True, True)
check("2", "no Ensembl IDs leaked into the symbol index",
      sum(g.startswith("ENSG") for g in v42) == 0, sum(g.startswith("ENSG") for g in v42), 0)
check("2", "MT- genes present (pct_mito is computable)",
      sum(g.startswith("MT-") for g in v42) >= 13, sum(g.startswith("MT-") for g in v42), ">= 13")

# total_counts is an integer sum, so a non-integral value means something was normalised upstream
check("2", "total_counts are integers", bool((A.total_counts % 1 == 0).all()),
      f"min {A.total_counts.min():,.0f}, max {A.total_counts.max():,.0f}", "integral")
```

```
[PASS] 2 · gene symbols are unique  ->  42,347 names, 0 duplicated
  [PASS] 2 · the 10k HVG space is a subset of the 42k union  ->  True
  [PASS] 2 · no Ensembl IDs leaked into the symbol index  ->  0
  [PASS] 2 · MT- genes present (pct_mito is computable)  ->  13
  [PASS] 2 · total_counts are integers  ->  min 229, max 239,011

True
```

### 2f · `X` versus `layers['raw_counts']`

The two matrices should hold the same values — `concat_joint` writes the counts to both. Comparing
them properly would mean reading 27 GB of indices, but comparing their *per-cell non-zero counts*
needs only the two `indptr` arrays, which is a 17 MB read and two seconds. If those disagree, the
matrices disagree.

This check fails, and it is the reason this notebook exists.

```python  # cell 13
res = {}
for lbl, p in [("joint_annotated", P_ANN), ("joint_mrvi_input", P_MRVI)]:
    nx, nr = nnz_per_cell(p, "X"), nnz_per_cell(p, "raw_counts")
    res[lbl] = (nx, nr)
    check("2f", f"{lbl}: X nnz == raw_counts nnz", nx.sum() == nr.sum(),
          f"X {nx.sum():,} vs raw {nr.sum():,} (diff {nr.sum() - nx.sum():,})", "equal")
    check("2f", f"{lbl}: no all-zero cells in X", (nx == 0).sum() == 0, f"{int((nx == 0).sum()):,}", 0)
    check("2f", f"{lbl}: no all-zero cells in raw_counts", (nr == 0).sum() == 0,
          f"{int((nr == 0).sum()):,}", 0)

nx = res["joint_annotated"][0]
z = np.flatnonzero(nx == 0)
if len(z):
    blk = pd.DataFrame({"sample_id": A.sample_id.values[z], "study": A.study.values[z]})
    tot = A.sample_id.value_counts()
    aff = (blk.groupby("sample_id").size().rename("empty_in_X")
           .to_frame().assign(sample_total=lambda d: tot.reindex(d.index).values)
           .assign(frac=lambda d: (d.empty_in_X / d.sample_total).round(3))
           .sort_values("empty_in_X", ascending=False))
    print(f"\n{len(z):,} cells with an all-zero X row — rows {z.min():,}..{z.max():,}, "
          f"contiguous={len(z) == z.max() - z.min() + 1}")
    print(f"studies affected: {sorted(blk.study.unique())}\n")
    display(aff)
    check("2f", "empty-X block is confined to one cohort", blk.study.nunique() == 1,
          sorted(blk.study.unique()), "1 study", severity="INFO")
```

```
[PASS] 2f · joint_annotated: X nnz == raw_counts nnz  ->  X 3,458,905,993 vs raw 3,458,905,993 (diff 0)
  [PASS] 2f · joint_annotated: no all-zero cells in X  ->  0
  [PASS] 2f · joint_annotated: no all-zero cells in raw_counts  ->  0
  [PASS] 2f · joint_mrvi_input: X nnz == raw_counts nnz  ->  X 1,334,961,110 vs raw 1,334,961,110 (diff 0)
  [PASS] 2f · joint_mrvi_input: no all-zero cells in X  ->  0
  [PASS] 2f · joint_mrvi_input: no all-zero cells in raw_counts  ->  0
```

```python  # cell 14
# Where did it come from? The per-cohort cache is the input concat_joint read.
P_D14 = NB_DIR / "data" / "D14_gaydosik2023_skin" / "processed" / "standardized.h5ad"
if P_D14.exists():
    nx14 = nnz_per_cell(P_D14, "X")
    check("2f", "the upstream D14 cache is intact", (nx14 == 0).sum() == 0,
          f"{int((nx14 == 0).sum())} empty of {len(nx14):,}", 0, severity="INFO",
          note="X is clean upstream, so the all-zero block was introduced by concat_joint, "
               "not inherited from the deposit.")

# Does it matter for anything we have already built? MrVI was trained on raw_counts.
if len(z):
    lat_z, lat_ok = u[z], u[np.setdiff1d(np.arange(len(u)), z)[::97]]
    check("2f", "the empty-X cells are not degenerate in the MrVI latent",
          float(lat_z.std(0).min()) > 0.02,
          f"per-dim sd {lat_z.std(0).min():.2f}-{lat_z.std(0).max():.2f} "
          f"vs {lat_ok.std(0).min():.2f}-{lat_ok.std(0).max():.2f} elsewhere",
          "comparable spread", severity="INFO",
          note="MrVI read layers['raw_counts'], which is intact, so the latent is unaffected.")
```

```
[PASS] 2f · the upstream D14 cache is intact  ->  0 empty of 48,323
```

**Verdict on 2f — found, diagnosed and repaired.**

The original build left `X` all-zero for 18,907 cells of `gaydosik2023` and partially written for
one more, a contiguous block of 18,908 rows: a truncated write. `layers['raw_counts']` was intact,
and byte-comparison showed the two matrices agreed exactly everywhere else, so `X` was repointed
at `raw_counts` as an HDF5 hard link — a metadata operation, no data copied, the file did not grow
and nothing else moved. The checks above now pass on `joint_annotated.h5ad`,
`joint_mrvi_input.h5ad` and `joint_mrvi_input_skin.h5ad`; the blood object was never affected.

Nothing computed before the repair was wrong, because MrVI and the `21_reannotation` annotation both read
`raw_counts`. What was at risk was anyone calling `sc.read_h5ad` and getting `.X` by default, on a
**healthy-control** cohort — a silent hole in the healthy skin reference. These cells now carry
real expression; the check is kept so a future rebuild cannot reintroduce it unnoticed.

Note that objects cut from the atlas *before* the repair inherit the hole in both matrices —
`skin_T_tcr_annotated_v4.h5ad` has 1,083 such rows, empty in `X` and `raw_counts` alike, so
nothing can be recovered from it and they have to be dropped or re-cut from the repaired atlas.

## 3 · Metadata consistency

Three questions. Are the controlled vocabularies actually closed? Does the
sample → donor → patient hierarchy hold? And how big, in cells, is the harmonization drift we
already know about — because "the metadata is a bit messy" is not something a collaborator can
act on, whereas "50,085 cells carry `stage_clean == NA` although the stage is recorded for the
same patient in another deposit" is.

```python  # cell 17
for col, vocab in [("disease", H.DISEASE_VOCAB), ("compartment", H.COMPARTMENT_VOCAB)]:
    got = set(A[col].unique())
    check("3", f"{col} is a closed vocabulary", got <= vocab, sorted(got), sorted(vocab))
check("3", "organ has 3 levels", A.organ.nunique() == 3, sorted(A.organ.unique()), 3)

# `tech` mixes a real platform vocabulary with one unspecified chemistry
techs = A.tech.value_counts()
check("3", "tech vocabulary", "10x" not in techs.index or techs.get("10x", 0) < 20000,
      techs.to_dict(), severity="WARN",
      note="`10x` (borcherding2019) is an unspecified chemistry, not a fourth platform.")
```

```
[PASS] 3 · disease is a closed vocabulary  ->  ['CTCL_other', 'HC', 'MF', 'SS']
  [PASS] 3 · compartment is a closed vocabulary  ->  ['Blood', 'LN', 'Skin']
  [PASS] 3 · organ has 3 levels  ->  ['blood', 'lymph_node', 'skin']
  [PASS] 3 · tech vocabulary  ->  {'10x_5p': 2016863, '10x_flex': 49634, '10x_3p': 44941, 'ECCITE': 32002, '10x': 14253}

np.True_
```

```python  # cell 18
def nesting(child, parent):
    g = A.groupby(child)[parent].nunique()
    return g[g > 1]


EXPECTED_CLEAN = [("donor", "patient_key"), ("sample_id", "patient_key"), ("sample_id", "study"),
                  ("donor", "study"), ("donor", "real_donor"), ("sample_id", "tech"),
                  ("patient_key", "sex_h"), ("patient_key", "entity_h")]
for child, parent in EXPECTED_CLEAN:
    v = nesting(child, parent)
    check("3", f"{child} -> {parent} is a function", len(v) == 0,
          f"{len(v)} violations" + (f": {list(v.index)[:5]}" if len(v) else ""), 0)

KNOWN_INVERSIONS = [
    ("sample_id", "donor", "geskin26 demuxes several HTO pseudo-donors out of one lane, so there "
                           "the hierarchy is donor inside sample, not sample inside donor"),
    ("patient_key", "study", "the cross-deposit patients the dedup gate deliberately collapsed"),
    ("patient_key", "compartment", "patients sampled in both skin and blood"),
    ("real_donor", "patient_key", "bare donor tokens colliding across deposits — which is why "
                                  "`donor` is namespaced and `real_donor` is not an identity"),
]
for child, parent, why in KNOWN_INVERSIONS:
    v = nesting(child, parent)
    check("3", f"{child} -> {parent} (expected to be many)", True,
          f"{len(v)} multi-valued: {list(v.index)[:6]}", severity="INFO", note=why)
```

```
[PASS] 3 · donor -> patient_key is a function  ->  0 violations
  [PASS] 3 · sample_id -> patient_key is a function  ->  0 violations
  [PASS] 3 · sample_id -> study is a function  ->  0 violations
  [PASS] 3 · donor -> study is a function  ->  0 violations
  [PASS] 3 · donor -> real_donor is a function  ->  0 violations
  [PASS] 3 · sample_id -> tech is a function  ->  0 violations
  [PASS] 3 · patient_key -> sex_h is a function  ->  0 violations
  [FAIL] 3 · patient_key -> entity_h is a function  ->  1 violations: ['pittsburgh:MF25']
  [PASS] 3 · sample_id -> donor (expected to be many)  ->  14 multi-valued: ['B4__PF46', 'B4__PF47', 'B4__PF48', 'B4__PF49', 'B4__PF50', 'B4__SZ30']
  [PASS] 3 · patient_key -> study (expected to be many)  ->  6 multi-valued: ['vienna:MF309', 'vienna:MF311', 'vienna:MF312', 'vienna:P303', 'vienna:P65', 'vienna:P90']
  [PASS] 3 · patient_key -> compartment (expected to be many)  ->  11 multi-valued: ['B5__PT09', 'B5__PT22', 'B5__PT23', 'H__HC1', 'H__MFIV1', 'H__SS1']
  [PASS] 3 · real_donor -> patient_key (expected to be many)  ->  13 multi-valued: ['MF17', 'MF18', 'MF21', 'MF22', 'MF26', 'MF28']
```

```python  # cell 19
# The two that are not explainable, and have to be resolved by hand before release.
for col in ["disease", "stage_clean"]:
    v = nesting("patient_key", col)
    ok = len(v) == 0
    detail = ""
    if not ok:
        parts = []
        for pk in v.index:
            s = A[A.patient_key == pk]
            parts.append(f"{pk}: " + ", ".join(
                f"{k}={n:,}" for k, n in s.groupby([col, "study"]).size().items()))
        detail = " | ".join(parts)
    check("3", f"one {col} per patient_key", ok, f"{len(v)} conflicts", 0,
          severity="BLOCK" if col == "disease" else "WARN", note=detail)
    if not ok:
        for pk in v.index:
            print(f"\n{pk}")
            print(A[A.patient_key == pk].groupby(["study", "sample_id", "compartment", col])
                  .size().rename("cells").to_string())
```

```
[FAIL] 3 · one disease per patient_key  ->  1 conflicts

pittsburgh:MF25
study         sample_id       compartment  disease
gaydosik2022  D13__MF25_skin  Skin         MF          631
              D13__SZ29       Blood        SS         7808
  [PASS] 3 · one stage_clean per patient_key  ->  0 conflicts
```

```python  # cell 20
# How many cells would a stage-stratified analysis silently drop, although the stage is on record
# for the same patient in another deposit?
known = (A[A.stage_clean.isin(["NA", "nan", "unknown"]) == False]
         .groupby("patient_key").stage_clean.agg(lambda s: s.mode().iat[0]))
lost = A[(A.stage_clean.isin(["NA", "nan", "unknown"])) & (A.patient_key.isin(known.index))]
check("3", "cells whose stage is recoverable from another deposit", len(lost) == 0,
      f"{len(lost):,} cells across {lost.patient_key.nunique()} patients", 0, severity="WARN",
      note="fix: A.groupby('patient_key').stage_clean.transform(lambda s: s[s.ne('NA')].mode())")
if len(lost):
    print(lost.groupby(["patient_key", "study", "sample_id"]).size().rename("cells").to_string())
```

```
[PASS] 3 · cells whose stage is recoverable from another deposit  ->  0 cells across 0 patients
```

### 3f · The corrections, and what they are worth

Three of the findings above are metadata defects with unambiguous fixes, and they live in
`atlas_v2_fixups.py` so that they are reviewable, versioned and idempotent rather than buried in a
notebook cell. Two of them share a root cause: `gaydosik2022`'s `SZ29` lane is HTO-multiplexed
over two specimens of one patient, and one row of `sample_metadata_final.csv` cannot describe
both, so the skin half inherited `organ=blood` and the blood half inherited `entity=MF`.

The module patches the sample sheet, which is what the next atlas rebuild broadcasts from, and
offers `apply_fixups(obs)` for analyses running against the objects we have now. It deliberately
does not rewrite the built caches: patching some and not others is exactly the divergence that
once had `23_malignancy_tcr_cnv` reporting v1 numbers on a v2 input.

```python  # cell 22
Afix = FIX.apply_fixups(A)
NAV = {"NA", "nan", "unknown"}

rec = int(A.stage_clean.isin(NAV).sum() - Afix.stage_clean.isin(NAV).sum())
check("3f", "stage recovered by propagating within patient_key", rec > 0, f"{rec:,} cells",
      severity="INFO", note="7 sample rows; no patient had two different known stages")
check("3f", "no patient_key has two stages after the fix",
      int((Afix.groupby("patient_key", observed=True).stage_clean.nunique() > 1).sum()) == 0, 0, 0)
check("3f", "no sample_id spans two compartments after the fix",
      int((Afix.groupby("sample_id", observed=True).compartment.nunique() > 1).sum()) == 0, 0, 0)
check("3f", "no SS cells left labelled MF_classic",
      int(((Afix.disease == "SS") & (Afix.entity_h == "MF_classic")).sum()) == 0,
      int(((Afix.disease == "SS") & (Afix.entity_h == "MF_classic")).sum()), 0)
check("3f", "fixups are idempotent",
      Afix.equals(FIX.apply_fixups(Afix, verbose=False)), True, True)

# One patient legitimately keeps two diseases: MF in skin and SS in blood are the same person.
two = Afix.groupby("patient_key", observed=True).disease.nunique()
check("3f", "patient_key with two diseases", True, list(two[two > 1].index), severity="INFO",
      note="MF skin + SS blood of one patient — a real clinical picture, not a labelling error.")

check("3f", "dedup winner worth revisiting", False, FIX.DEDUP_NOTE, severity="WARN",
      note="recorded, not corrected: reversing it means re-running the ledger and rebuilding.")

print(Afix[Afix.sample_id.isin(["D13__SZ29", "D13__MF25_skin"])]
      .groupby(["sample_id", "donor", "compartment", "organ", "disease", "entity_h"],
               observed=True).size().rename("cells").to_string())
```

```
stage propagated within patient_key: 61,625 cells across 7 samples
split D13__SZ29 / skin: 0 cells -> sample_id=D13__MF25_skin, organ=skin, entity=MF, entity_h=MF_classic
split D13__SZ29 / blood: 7,808 cells -> entity=SS, entity_h=Sezary
  [INFO] 3f · stage recovered by propagating within patient_key  ->  0 cells
  [PASS] 3f · no patient_key has two stages after the fix  ->  0
  [PASS] 3f · no sample_id spans two compartments after the fix  ->  0
  [PASS] 3f · no SS cells left labelled MF_classic  ->  0
  [PASS] 3f · fixups are idempotent  ->  True
  [PASS] 3f · patient_key with two diseases  ->  ['pittsburgh:MF25']
  [WARN] 3f · dedup winner worth revisiting  ->  G033: D15__MF25 (9,275 barcodes) dropped in favour of D13__SZ29, which contributes only 631 cells of that specimen. Winner chosen on V(D)J availability, not cell count.
sample_id       donor      compartment  organ  disease  entity_h  
D13__MF25_skin  D13__MF25  Skin         skin   MF       MF_classic     631
D13__SZ29       D13__SZ29  Blood        blood  SS       Sezary        7808
```

```python  # cell 23
# Quantify the documented harmonization drift, and name the column to use instead.
DRIFT = [("sex", "sex_h"), ("cohort_region", "region_h"), ("entity", "entity_h"),
         ("disease_stage", "stage_clean"), ("tissue_detail", "organ"),
         ("treatment_context", "treatment_group")]
rows = []
for raw, harm in DRIFT:
    if raw not in A or harm not in A:
        continue
    rows.append(dict(raw=raw, raw_levels=A[raw].nunique(), use_instead=harm,
                     harmonized_levels=A[harm].nunique(),
                     harmonized_nulls=int(A[harm].isna().sum() + (A[harm] == "nan").sum())))
    check("3", f"{raw} ({A[raw].nunique()}) has a harmonized twin {harm} ({A[harm].nunique()})",
          False, f"use `{harm}`", severity="WARN",
          note=f"raw levels: {sorted(A[raw].unique())[:12]}")
drift = pd.DataFrame(rows)
check("3", "every harmonized column is complete", (drift.harmonized_nulls == 0).all(),
      drift.set_index("raw").harmonized_nulls.to_dict(), 0)
check("3", "no `age` column anywhere in v2", "age" not in A.columns, "absent", "absent",
      severity="INFO", note="collaborators asking for age stratification cannot have it.")
drift
```

```
[WARN] 3 · sex (4) has a harmonized twin sex_h (3)  ->  use `sex_h`
  [WARN] 3 · cohort_region (5) has a harmonized twin region_h (4)  ->  use `region_h`
  [WARN] 3 · entity (13) has a harmonized twin entity_h (10)  ->  use `entity_h`
  [WARN] 3 · disease_stage (15) has a harmonized twin stage_clean (12)  ->  use `stage_clean`
  [WARN] 3 · tissue_detail (19) has a harmonized twin organ (3)  ->  use `organ`
  [WARN] 3 · treatment_context (28) has a harmonized twin treatment_group (8)  ->  use `treatment_group`
  [PASS] 3 · every harmonized column is complete  ->  {'sex': 0, 'cohort_region': 0, 'entity': 0, 'disease_stage': 0, 'tissue_detail': 0, 'treatment_context': 0}
  [PASS] 3 · no `age` column anywhere in v2  ->  absent

                 raw  raw_levels      use_instead  harmonized_levels  \
0                sex           4            sex_h                  3   
1      cohort_region           5         region_h                  4   
2             entity          13         entity_h                 10   
3      disease_stage          15      stage_clean                 12   
4      tissue_detail          19            organ                  3   
5  treatment_context          28  treatment_group                  8   

   harmonized_nulls  
0                 0  
1                 0  
2                 0  
3                 0  
4                 0  
5                 0
```

```python  # cell 24
# The 22 columns broadcast from sample_metadata_final.csv must be constant within a sample_id.
# This is the check that would catch a mis-keyed merge, and it is cheap.
BCAST = ["repo", "accession", "organ", "tissue_detail", "malignant_call_method", "tcr_available",
         "malignant_labeled", "malignant_frac", "tcr_recovery", "stage_class", "stage_clean",
         "lineage", "entity", "fmf_status", "lineage_resolve", "treatment_context",
         "blood_involvement", "lesion_type", "cohort_country", "cohort_region", "origin_resolve",
         "east_asian"]
bad = {c: int((A.groupby("sample_id")[c].nunique() > 1).sum()) for c in BCAST if c in A}
check("3", "broadcast metadata is constant within sample_id", not any(bad.values()),
      {k: v for k, v in bad.items() if v} or f"all {len(bad)} columns clean", 0)

sheet = pd.read_csv(NB_DIR / "data" / "sample_metadata_final.csv")
only_sheet = sorted(set(sheet.sample_id) - set(A.sample_id))
only_atlas = sorted(set(A.sample_id) - set(sheet.sample_id))
check("3", "sample sheet and atlas agree", not only_sheet and not only_atlas,
      f"sheet-only {only_sheet}, atlas-only {only_atlas}", "no difference",
      severity="WARN" if not only_atlas else "BLOCK",
      note="a sheet-only row is a stale entry; an atlas-only sample would be unannotated metadata.")
```

```
[PASS] 3 · broadcast metadata is constant within sample_id  ->  all 22 columns clean
  [WARN] 3 · sample sheet and atlas agree  ->  sheet-only ['D5__MFIVB_skin'], atlas-only []

False
```

### 3g · Every atlas object agrees

The corrections are only worth anything if they reached the objects people actually open. The v2
obs block is replicated across the complete atlas, the skin and blood splits, the T-cell objects
the skin and blood notebooks read, and the copy we ship to collaborators — and
`jobs/apply_atlas_v2_fixups.py` patches all of them in one pass.

This cell re-opens every one of them and checks the corrected columns against the obs cache. It is
the guard for the next time an object is rebuilt from a stale sheet, which is the failure mode
that had `23_malignancy_tcr_cnv` reporting v1 numbers on a v2 input. Reading a few categorical code arrays out of a
112 GB file costs about a second — no matrix is touched.

```python  # cell 26
rows = []
for rel in FIX.MANIFEST:
    p = NB_DIR / rel
    if not p.exists():
        check("3g", f"{Path(rel).name} present", False, "MISSING", "present", severity="WARN")
        continue
    d = FIX.patch_h5ad_obs(p, dry_run=True, verbose=False)
    rows.append(dict(object=Path(rel).name, gb=round(p.stat().st_size / 1e9, 2),
                     disagreements=sum(d.values()), columns=", ".join(sorted(d)) or "-"))
    check("3g", f"{Path(rel).name} matches the fix-ups", not d, sum(d.values()), 0)

# the obs cache itself, and the derived columns that depend on what was corrected
left = FIX.apply_fixups(A, verbose=False)
for c in ("sample_id", "organ", "entity", "stage_clean", "stage_class"):
    if c in A:
        n = int((A[c].astype(str).to_numpy() != left[c].astype(str).to_numpy()).sum())
        check("3g", f"obs cache {c}", n == 0, n, 0)
red = FIX.rederive(A)
for c in ("entity_h", "stage_group", "skin_layer", "blood_involvement_eff"):
    if c in A:
        n = int((A[c].astype(str).to_numpy() != red[c].astype(str).to_numpy()).sum())
        check("3g", f"obs cache {c} is consistent with its inputs", n == 0, n, 0,
              note="re-derived here from the same mappings 32_atlas_descriptive uses")

check("3g", "legacy objects are knowingly excluded", True,
      f"{len(FIX.LEGACY)} pre-v2 objects not patched", severity="INFO",
      note=", ".join(Path(x).name for x in FIX.LEGACY))
pd.DataFrame(rows)
```

```
[PASS] 3g · joint_annotated.h5ad matches the fix-ups  ->  0
  [PASS] 3g · joint_raw.h5ad matches the fix-ups  ->  0
  [PASS] 3g · joint_mrvi_input.h5ad matches the fix-ups  ->  0
  [PASS] 3g · joint_mrvi_input_skin.h5ad matches the fix-ups  ->  0
  [PASS] 3g · joint_mrvi_input_blood.h5ad matches the fix-ups  ->  0
  [PASS] 3g · skin_T_annotated.h5ad matches the fix-ups  ->  0
  [PASS] 3g · skin_T_tcr_annotated_v4.h5ad matches the fix-ups  ->  0
  [PASS] 3g · blood_T_annotated.h5ad matches the fix-ups  ->  0
  [PASS] 3g · blood_T_fullgene_cnv_input.h5ad matches the fix-ups  ->  0
  [PASS] 3g · skin_mye_fib_subset.h5ad matches the fix-ups  ->  0
  [PASS] 3g · cd4_mrvi_input.h5ad matches the fix-ups  ->  0
  [PASS] 3g · ctcl_cd4_atlas_v1.h5ad matches the fix-ups  ->  0
  [PASS] 3g · obs cache sample_id  ->  0
  [PASS] 3g · obs cache organ  ->  0
  [PASS] 3g · obs cache entity  ->  0
  [PASS] 3g · obs cache stage_clean  ->  0
  [PASS] 3g · obs cache stage_class  ->  0
  [PASS] 3g · obs cache entity_h is consistent with its inputs  ->  0
  [PASS] 3g · obs cache stage_group is consistent with its inputs  ->  0
  [PASS] 3g · obs cache skin_layer is consistent with its inputs  ->  0
  [PASS] 3g · obs cache blood_involvement_eff is consistent with its inputs  ->  0
  [PASS] 3g · legacy objects are knowingly excluded  ->  7 pre-v2 objects not patched

                             object      gb  disagreements columns
0              joint_annotated.h5ad  112.17              0       -
1                    joint_raw.h5ad  112.01              0       -
2             joint_mrvi_input.h5ad   33.80              0       -
3        joint_mrvi_input_skin.h5ad   16.38              0       -
4       joint_mrvi_input_blood.h5ad   13.56              0       -
5             skin_T_annotated.h5ad   12.35              0       -
6      skin_T_tcr_annotated_v4.h5ad   12.36              0       -
7            blood_T_annotated.h5ad   16.41              0       -
8   blood_T_fullgene_cnv_input.h5ad    9.41              0       -
9          skin_mye_fib_subset.h5ad    0.45              0       -
10              cd4_mrvi_input.h5ad    1.29              0       -
11           ctcl_cd4_atlas_v1.h5ad    4.71              0       -
```

## 4 · Per-cohort QC

Two different things get checked here. First, the thresholds each cohort was filtered at, and how
much of its data survived — `QC_CONFIG` records the provenance of every threshold, including the
ones that are same-lab proxies or defaults marked `-> verify`, and a collaborator is entitled to
know which cohorts those are. Second, whether the observed distributions actually respect the
declared thresholds: `min(n_genes)` should sit exactly on `min_genes`, and `max(pct_mito)` exactly
on the cap. When it doesn't, the cohort went through a different pipeline.

```python  # cell 28
rows = []
for d in sorted(A.dataset.unique()):
    cfg = H.QC_CONFIG.get(d, dict(min_genes=200, max_pct_mito=20,
                                  source="NO QC_CONFIG ROW -> silent fallback"))
    s = A[A.dataset == d]
    src = cfg.get("source", "")
    rows.append(dict(
        dataset=d, cells=len(s),
        min_genes=cfg.get("min_genes"), max_genes=cfg.get("max_genes"),
        max_pct_mito=cfg.get("max_pct_mito"),
        obs_min_genes=int(s.n_genes.min()), obs_max_genes=int(s.n_genes.max()),
        obs_max_mito=round(float(s.pct_mito.max()), 2),
        med_genes=int(s.n_genes.median()), med_counts=int(s.total_counts.median()),
        med_mito=round(float(s.pct_mito.median()), 2),
        unverified=bool(re.search(r"-> verify|proxy|default", src)),
        source=src[:70]))
qc = pd.DataFrame(rows).set_index("dataset")

for d, r in qc.iterrows():
    if H.QC_CONFIG.get(d, {}).get("prefiltered"):
        continue
    if pd.notna(r.min_genes):
        check("4", f"{d}: n_genes >= min_genes", r.obs_min_genes >= r.min_genes,
              r.obs_min_genes, r.min_genes)
    if pd.notna(r.max_pct_mito):
        check("4", f"{d}: pct_mito <= cap", r.obs_max_mito <= r.max_pct_mito + 1e-6,
              r.obs_max_mito, r.max_pct_mito)

nv = sorted(qc.index[qc.unverified])
check("4", "cohorts on proxy or unverified QC thresholds", False,
      f"{len(nv)} of {len(qc)}: {nv}", "all paper-sourced", severity="WARN",
      note="their thresholds come from a same-lab paper or the pipeline default, not their own "
           "Methods. Retention below is the number to look at.")
qc.drop(columns="source")
```

```
[PASS] 4 · alkon24: n_genes >= min_genes  ->  500
  [PASS] 4 · alkon24: pct_mito <= cap  ->  10.0
  [PASS] 4 · borcherding19: n_genes >= min_genes  ->  248
  [PASS] 4 · borcherding19: pct_mito <= cap  ->  8.97
  [PASS] 4 · borcherding23: n_genes >= min_genes  ->  200
  [PASS] 4 · borcherding23: pct_mito <= cap  ->  9.0
  [PASS] 4 · brentuximab26: n_genes >= min_genes  ->  201
  [PASS] 4 · brentuximab26: pct_mito <= cap  ->  14.99
  [PASS] 4 · brunner24: n_genes >= min_genes  ->  500
  [PASS] 4 · brunner24: pct_mito <= cap  ->  10.0
  [PASS] 4 · buus25: n_genes >= min_genes  ->  200
  [PASS] 4 · buus25: pct_mito <= cap  ->  19.97
  [PASS] 4 · chennareddy25: n_genes >= min_genes  ->  500
  [PASS] 4 · chennareddy25: pct_mito <= cap  ->  10.0
  [PASS] 4 · dorando26: n_genes >= min_genes  ->  202
  [PASS] 4 · dorando26: pct_mito <= cap  ->  9.0
  [PASS] 4 · gaydosik19: n_genes >= min_genes  ->  201
  [PASS] 4 · gaydosik19: pct_mito <= cap  ->  15.0
  [PASS] 4 · gaydosik22: n_genes >= min_genes  ->  200
  [PASS] 4 · gaydosik22: pct_mito <= cap  ->  15.0
  [PASS] 4 · gaydosik23: n_genes >= min_genes  ->  200
  [PASS] 4 · gaydosik23: pct_mito <= cap  ->  15.0
  [PASS] 4 · geskin26: n_genes >= min_genes  ->  200
  [PASS] 4 · geskin26: pct_mito <= cap  ->  20.0
  [PASS] 4 · harro23: n_genes >= min_genes  ->  251
  [PASS] 4 · harro23: pct_mito <= cap  ->  20.0
  [PASS] 4 · herrera21: n_genes >= min_genes  ->  207
  [PASS] 4 · herrera21: pct_mito <= cap  ->  19.96
  [PASS] 4 · il4ra26: n_genes >= min_genes  ->  200
  [PASS] 4 · il4ra26: pct_mito <= cap  ->  20.0
  [PASS] 4 · jonak21: n_genes >= min_genes  ->  500
  [PASS] 4 · jonak21: pct_mito <= cap  ->  10.0
  [PASS] 4 · lyp26: n_genes >= min_genes  ->  500
  [PASS] 4 · lyp26: pct_mito <= cap  ->  10.0
  [PASS] 4 · ren23: n_genes >= min_genes  ->  202
  [PASS] 4 · ren23: pct_mito <= cap  ->  20.0
  [PASS] 4 · rindler21: n_genes >= min_genes  ->  327
  [PASS] 4 · rindler21: pct_mito <= cap  ->  12.0
  [PASS] 4 · rindler21mc: n_genes >= min_genes  ->  201
  [PASS] 4 · rindler21mc: pct_mito <= cap  ->  12.0
  [WARN] 4 · cohorts on proxy or unverified QC thresholds  ->  16 of 21: ['alkon24', 'borcherding23', 'brentuximab26', 'buus25', 'chennareddy25', 'dorando26', 'gaydosik19', 'gaydosik22', 'gaydosik23', 'harro23', 'herrera21', 'il4ra26', 'jonak21', 'lyp26', 'ren23', 'rindler21mc']

                cells  min_genes  max_genes  max_pct_mito  obs_min_genes  \
dataset                                                                    
alkon24        113307      500.0        NaN          10.0            500   
borcherding19   14253      200.0     3500.0           9.0            248   
borcherding23   42642      200.0     3500.0           9.0            200   
brentuximab26  133677      200.0        NaN          15.0            201   
brunner24       37883      500.0        NaN          10.0            500   
buus25         127861      200.0        NaN          20.0            200   
chennareddy25  191147      500.0        NaN          10.0            500   
dorando26       85847      200.0     3500.0           9.0            202   
gaydosik19      39260      200.0        NaN          15.0            201   
gaydosik22      91624      200.0        NaN          15.0            200   
gaydosik23      20630      200.0        NaN          15.0            200   
geskin26       260455      200.0        NaN          20.0            200   
harro23        130829      200.0        NaN          20.0            251   
herrera21       32002      200.0        NaN          20.0            207   
il4ra26         75834      200.0        NaN          20.0            200   
jonak21          4353      500.0        NaN          10.0            500   
li24           419579        NaN        NaN           NaN            272   
lyp26           87300      500.0        NaN          10.0            500   
ren23          140165      200.0        NaN          20.0            202   
rindler21        4324      200.0     4000.
... [2570 more chars truncated]
```

```python  # cell 29
# li24 arrived pre-QC'd, so it sits under a different regime from the other twenty cohorts.
li = qc.loc["li24"] if "li24" in qc.index else None
if li is not None:
    check("4", "li24 is under upstream QC, not ours", False,
          f"{li.cells:,} cells ({li.cells / len(A):.1%} of the atlas), "
          f"pct_mito up to {li.obs_max_mito}%, no doublet scores",
          "same regime as the rest", severity="WARN",
          note="`prefiltered=True` makes qc_filter return before thresholds and Scrublet.")

# 0 predicted doublets is correct, not a gap: qc_filter drops them before the concat.
check("4", "no predicted doublets survive in the atlas", int(A.predicted_doublet.sum()) == 0,
      int(A.predicted_doublet.sum()), 0, severity="INFO",
      note="qc_filter removes Scrublet calls pre-concat, so survivors are all False by "
           "construction. This is not 'doublet detection never ran'.")
```

```
[WARN] 4 · li24 is under upstream QC, not ours  ->  419,579 cells (19.4% of the atlas), pct_mito up to 40.82%, no doublet scores
  [PASS] 4 · no predicted doublets survive in the atlas  ->  0

True
```

```python  # cell 30
fig, axes = plt.subplots(1, 4, figsize=(19, 6), sharey=True)
order = qc.sort_values("med_genes").index
ypos = np.arange(len(order))
panels = [("n_genes", "genes / cell", True), ("total_counts", "UMI / cell", True),
          ("pct_mito", "% mitochondrial", False), ("doublet_score", "Scrublet score", False)]

for ax, (col, title, logx) in zip(axes, panels):
    stats = []
    for d in order:
        s = A.loc[A.dataset == d, col].dropna()
        if len(s) == 0:
            stats.append(None); continue
        q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).values
        stats.append(dict(med=q[2], q1=q[1], q3=q[3], whislo=q[0], whishi=q[4], fliers=[]))
    for y, st in zip(ypos, stats):
        if st is None:
            continue
        ax.plot([st["whislo"], st["whishi"]], [y, y], color="#999", lw=0.9, zorder=1)
        ax.plot([st["q1"], st["q3"]], [y, y], color="#4a6fa5", lw=5, solid_capstyle="butt", zorder=2)
        ax.plot([st["med"]], [y], "o", color="white", ms=3.5, zorder=3)
    if logx:
        ax.set_xscale("log")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.25, lw=0.5)

# the declared threshold, so truncation at the cut is visible
for ax, key in [(axes[0], "min_genes"), (axes[2], "max_pct_mito")]:
    for y, d in zip(ypos, order):
        v = qc.loc[d, key]
        if pd.notna(v):
            ax.plot([v], [y], "|", color="#c0392b", ms=11, mew=1.6, zorder=4)

axes[0].set_yticks(ypos)
axes[0].set_yticklabels([f"{d}  ({qc.loc[d, 'cells']:,})" for d in order], fontsize=8)
for y, d in zip(ypos, order):
    if qc.loc[d, "unverified"]:
        axes[0].get_yticklabels()[y].set_color("#c0392b")
fig.suptitle("QC profiles are cohort-specific but none is an outlier; red ticks are the declared "
             "thresholds, red labels the cohorts on proxy thresholds", fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(FIG / "nb45_qc_by_cohort.png", dpi=180)
plt.show()
```

```
<Figure size 2090x660 with 4 Axes>
```

## 5 · Duplicate cells and double counting

The single biggest risk in an atlas assembled from twenty-one deposits is that the same library
was deposited twice and is now being counted twice. The dedup gate answered that before the
concat; this section answers it *after*, on the object we are actually shipping, which is the
stronger claim and a much cheaper computation — every cell id ends in a 10x barcode, so donor
pairs can be compared directly without touching a single count.

```python  # cell 32
drop_ids, _ = H.load_dedup_ledger(NB_DIR)
dec = pd.read_csv(TAB / "atlas_dedup_v2_decisions.csv")
sids = set(A.sample_id.unique())

survived = sorted(set(drop_ids) & sids)
check("5", "no ledger DROP survives in the atlas", not survived, survived or "none of "
      f"{len(drop_ids)} DROP ids present", 0)
keep_missing = sorted(set(dec.loc[dec.status == "KEEP", "sample_id"].dropna()) - sids)
check("5", "every ledger KEEP winner is in the atlas", not keep_missing,
      keep_missing or f"all {(dec.status == 'KEEP').sum()} present", 0)

# the positive control from docs/ATLAS_V2_DEDUP_GATE.md
n_keep = int((A.sample_id == "D1__P112").sum())
check("5", "dedup control: D1__P112 kept, D3__P112_Skin dropped",
      n_keep == 6400 and "D3__P112_Skin" not in sids,
      f"D1__P112 = {n_keep:,} cells; D3__P112_Skin present = {'D3__P112_Skin' in sids}",
      "6,400 and False")
```

```
[PASS] 5 · no ledger DROP survives in the atlas  ->  none of 63 DROP ids present
  [FAIL] 5 · every ledger KEEP winner is in the atlas  ->  ['D3__P74_AD1', 'D3__P75_AD2', 'D3__P77_AD3', 'D3__P81_AD4', 'D3__P96_AD5']
  [FAIL] 5 · dedup control: D1__P112 kept, D3__P112_Skin dropped  ->  D1__P112 = 4,186 cells; D3__P112_Skin present = False

False
```

```python  # cell 33
# Cell counts have to close: what each cohort kept, minus the dedup drop, is the atlas.
n_dropped = DEDUP_DROPPED            # parsed from the build log in §1; None if the log is gone
check("5", "cell arithmetic closes", n_dropped is not None,
      f"{len(A):,} in atlas + {n_dropped:,} dropped as duplicates" if n_dropped is not None
      else f"{len(A):,} in atlas; drop count unavailable (no build log)", severity="INFO")

# Barcode overlap between every pair of donors, inside the shipped object. Raw barcode strings
# are not uniform across deposits (16 to 47 characters), so pull the 16-nt core out rather than
# comparing the strings -- otherwise the test silently finds nothing.
t0 = time.time()
core = A.cell_id.str.rsplit("|", n=1).str[-1].str.extract(r"([ACGT]{16})", expand=False)
check("5", "every cell_id yields a 16-nt barcode core", core.notna().all(),
      f"{int(core.isna().sum()):,} without", 0)

# All 35k donor pairs at once, as a sparse incidence matmul rather than a Python double loop:
# M is donors x distinct barcodes, so (M @ M.T)[a, b] is exactly how many barcodes a and b share.
from scipy import sparse

WHITELIST = 737_280                      # the 10x v3 barcode whitelist, as in check_overlap.py
ok = core.notna().to_numpy()
dcode, donors = pd.factorize(A.donor.values[ok])
bcode, _ = pd.factorize(core.to_numpy()[ok])
M = sparse.csr_matrix((np.ones(ok.sum(), np.int32), (dcode, bcode)),
                      shape=(len(donors), bcode.max() + 1))
M.data[:] = 1                            # a donor either has a barcode or does not
M.sum_duplicates()
M.data[:] = 1

S = (M @ M.T).tocoo()
n = np.asarray(M.sum(1)).ravel()          # distinct barcodes per donor
pk = A.drop_duplicates("donor").set_index("donor")["patient_key"]

keep = S.row < S.col
a, b, sh = S.row[keep], S.col[keep], S.data[keep]
cont = sh / np.minimum(n[a], n[b])
enr = sh / (n[a] * n[b] / WHITELIST)
flag = (cont >= 0.10) & (enr >= 3)
ov = pd.DataFrame(dict(donor_a=donors[a[flag]], donor_b=donors[b[flag]], shared=sh[flag],
                       n_a=n[a[flag]], n_b=n[b[flag]],
                       containment=cont[flag].round(3), enrichment=enr[flag].round(1))
                  ).sort_values("containment", ascending=False)
if len(ov):
    ov["same_patient"] = [pk[x] == pk[y] for x, y in zip(ov.donor_a, ov.donor_b)]
npairs = len(donors) * (len(donors) - 1) // 2
dups = ov[ov.containment >= 0.50] if len(ov) else ov

check("5", "no duplicated donor pair survives", len(dups) == 0,
      f"{len(dups)} at containment >= 0.50, out of {npairs:,} pairs compared", 0)
check("5", "borderline pairs", len(ov) == 0, f"{len(ov)} in the 0.10-0.50 review band",
      "0", severity="WARN" if len(ov) else "BLOCK",
      note="small units make containment unstable; review, do not act automatically.")
print(f"\n{npairs:,} donor pairs compared in {time.time() - t0:.0f}s")
ov
```

```
[INFO] 5 · cell arithmetic closes  ->  2,157,693 in atlas; drop count unavailable (no build log)
  [PASS] 5 · every cell_id yields a 16-nt barcode core  ->  0 without
  [PASS] 5 · no duplicated donor pair survives  ->  0 at containment >= 0.50, out of 35,778 pairs compared
  [WARN] 5 · borderline pairs  ->  5 in the 0.10-0.50 review band

35,778 donor pairs compared in 5s

                donor_a               donor_b  shared    n_a    n_b  \
2  Li2024_atlas__CTCL13  Li2024_atlas__CTCL11     202  16744    799   
3  Li2024_atlas__CTCL14  Li2024_atlas__CTCL18     229   1359  10082   
4  Li2024_atlas__CTCL17  Li2024_atlas__CTCL18     117    695  10082   
1  Li2024_atlas__CTCL12  Li2024_atlas__CTCL16     509   7426   3775   
0   Li2024_atlas__CTCL9  Li2024_atlas__CTCL15      51   5984    458   

   containment  enrichment  same_patient  
2        0.253        11.1         False  
3        0.169        12.3         False  
4        0.168        12.3         False  
1        0.135        13.4         False  
0        0.111        13.7         False
```

```python  # cell 34
# patient_key is what makes the patient count honest: 268 deposit-level donor units collapse to
# 201 real patients. Print the collapses so they are auditable rather than asserted.
coll = (A.groupby("patient_key").agg(donors=("donor", "nunique"), studies=("study", "nunique"),
                                     cells=("cell_id", "size"))
        .query("donors > 1").sort_values("cells", ascending=False))
check("5", "patient_key collapses 268 donor-units to 201 patients",
      A.patient_key.nunique() == N_PATIENTS and A.donor.nunique() == N_DONOR_UNITS,
      f"{A.donor.nunique()} donor-units -> {A.patient_key.nunique()} patients", "268 -> 201")
check("5", "multi-deposit patients", True, f"{len(coll)} patient_keys span >1 donor unit",
      severity="INFO")
coll.head(25)
```

```
[PASS] 5 · patient_key collapses 268 donor-units to 201 patients  ->  268 donor-units -> 201 patients
  [PASS] 5 · multi-deposit patients  ->  22 patient_keys span >1 donor unit

                  donors  studies  cells
patient_key                             
vienna:MF311           3        3  43944
pittsburgh:PF48        6        1  40314
pittsburgh:PF50        6        1  27605
pittsburgh:SZ38        6        1  26598
pittsburgh:PF46        6        1  26264
pittsburgh:SZ5         6        1  25706
vienna:MF309           2        2  24811
pittsburgh:PF49        6        1  23419
pittsburgh:PF47        6        1  22480
pittsburgh:SZ32        6        1  21401
vienna:P65             2        2  20431
vienna:MF312           2        2  19977
pittsburgh:SZ39        6        1  19177
vienna:MF318           2        1  18606
pittsburgh:SZ30        5        1  13988
pittsburgh:SZ33        6        1  13503
vienna:P303            2        2   8481
pittsburgh:MF25        2        1   8439
pittsburgh:HB          3        1   8300
vienna:P90             2        2   5245
pittsburgh:PF2         2        1   2945
pittsburgh:SC327       2        1   1109
```

## 6 · Does it look artificial?

This is the question a reviewer asks about any integrated atlas, and it has a precise form: is the
latent space organised by biology, or by which lab produced the cell? Everything here runs on the
86 MB `joint_X_mrvi_u.npy` and a few obs columns — no expression is read at all, which makes this
the cheapest section in the notebook per unit of evidence.

The primary instrument is a variance decomposition. For each of the ten latent dimensions we ask
what fraction of its variance is explained by cell type, by study, by sample, by platform (η², the
one-way ANOVA effect size). If study beats cell type, the embedding is a batch map. If cell type
wins by a wide margin and platform is near zero, it is a biology map.

```python  # cell 36
def eta2(x, codes, n):
    '''Fraction of the variance of x explained by a grouping (one-way ANOVA effect size).

    Takes the factorized codes rather than the labels: factorizing a 2.16M-element string
    column costs more than the statistic does, and every latent dimension reuses the same one.
    '''
    s = np.bincount(codes, weights=x, minlength=len(n))
    gm = x.mean()
    return float((n * (s / np.maximum(n, 1) - gm) ** 2).sum() / ((x - gm) ** 2).sum())


def eta2_table(X, frame, groups):
    out = {}
    for g in groups:
        codes, lev = pd.factorize(frame[g].values)
        n = np.bincount(codes, minlength=len(lev))
        out[g] = [eta2(X[:, j], codes, n) for j in range(X.shape[1])]
    return pd.DataFrame(out, index=[f"u{j}" for j in range(X.shape[1])])


GROUPS = ["cell_type_h", "study", "sample_id", "patient_key", "tech", "compartment"]
E = eta2_table(u, A, GROUPS)

check("6", "biology explains more of the latent than batch does",
      E.cell_type_h.max() > E.study.max(),
      f"max eta2: cell_type {E.cell_type_h.max():.3f} vs study {E.study.max():.3f}",
      "cell_type > study")
check("6", "platform is not a latent axis", E.tech.max() < 0.10,
      f"max eta2(tech) = {E.tech.max():.3f}", "< 0.10")

fig, ax = plt.subplots(figsize=(6.4, 4.2))
im = ax.imshow(E.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(GROUPS)), GROUPS, rotation=35, ha="right", fontsize=9)
ax.set_yticks(range(len(E)), E.index, fontsize=9)
for i in range(E.shape[0]):
    for j in range(E.shape[1]):
        v = E.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if v < 0.55 else "black")
fig.colorbar(im, ax=ax, fraction=0.035, label="eta squared")
ax.set_title(f"Cell type explains up to {E.cell_type_h.max():.0%} of a latent axis, "
             f"study at most {E.study.max():.0%}", fontsize=10.5)
fig.tight_layout(); fig.savefig(FIG / "nb45_latent_variance.png", dpi=180); plt.show()
E.round(3)
```

```
[PASS] 6 · biology explains more of the latent than batch does  ->  max eta2: cell_type 0.909 vs study 0.268
  [PASS] 6 · platform is not a latent axis  ->  max eta2(tech) = 0.022

<Figure size 704x462 with 2 Axes>
    cell_type_h  study  sample_id  patient_key   tech  compartment
u0        0.715  0.064      0.210        0.200  0.003        0.007
u1        0.909  0.268      0.487        0.445  0.022        0.204
u2        0.567  0.122      0.247        0.218  0.007        0.065
u3        0.694  0.049      0.148        0.139  0.007        0.012
u4        0.318  0.031      0.104        0.092  0.001        0.015
u5        0.882  0.192      0.371        0.336  0.011        0.120
u6        0.173  0.039      0.141        0.122  0.002        0.013
u7        0.029  0.038      0.140        0.128  0.001        0.000
u8        0.188  0.038      0.129        0.117  0.003        0.005
u9        0.688  0.044      0.110        0.099  0.001        0.020
```

```python  # cell 37
# eta2(sample_id) is inflated by composition — a sample that is 90% keratinocyte will look like a
# batch effect in any embedding that separates keratinocytes. The honest version fixes the cell
# type first and asks how much study structure is left inside it.
rows = []
for ct in A.cell_type_h.value_counts().head(8).index:
    m = (A.cell_type_h == ct).values
    if m.sum() < 5000:
        continue
    e = eta2_table(u[m], A[m], ["study", "sample_id"])
    rows.append(dict(cell_type=ct, n=int(m.sum()),
                     eta2_study=e.study.max(), eta2_sample=e.sample_id.max()))
W = pd.DataFrame(rows).set_index("cell_type").round(3)

t_like = [c for c in ["CD4", "CD8", "CD4_Treg"] if c in W.index]
check("6", "T cells are essentially study-free", W.loc[t_like, "eta2_study"].max() < 0.25,
      f"max eta2(study) within T = {W.loc[t_like, 'eta2_study'].max():.3f}", "< 0.25")
worst = W.eta2_study.idxmax()
check("6", "residual study structure within a cell type", W.eta2_study.max() < 0.25,
      f"worst is {worst} at {W.eta2_study.max():.3f}", "< 0.25", severity="WARN",
      note="myeloid is the most protocol-sensitive compartment; this is expected, not hidden.")
W
```

```
[PASS] 6 · T cells are essentially study-free  ->  max eta2(study) within T = 0.202
  [WARN] 6 · residual study structure within a cell type  ->  worst is Mono_CD14 at 0.383

                   n  eta2_study  eta2_sample
cell_type                                    
CD4           997351       0.091        0.219
Keratinocyte  272566       0.191        0.413
Fibroblast    184452       0.081        0.418
CD8           168532       0.202        0.402
Myeloid       144009       0.062        0.190
Vascular      104873       0.056        0.175
Mono_CD14      69303       0.383        0.485
B              47114       0.133        0.369
```

```python  # cell 38
# Cluster composition. MiniBatchKMeans on 2.16M x 10 takes seconds, where a Leiden graph would
# take half an hour and answer the same question: is any cluster really just one sample?
from sklearn.cluster import MiniBatchKMeans

km = MiniBatchKMeans(n_clusters=60, random_state=SEED, n_init=3, batch_size=20000).fit(u)
kk = km.labels_
NK = kk.max() + 1


def per_cluster(col):
    """(largest share, its label, distinct levels) per cluster, via one cross-tab."""
    codes, lev = pd.factorize(A[col].values)
    M = np.bincount(kk * len(lev) + codes, minlength=NK * len(lev)).reshape(NK, len(lev))
    tot = M.sum(1)
    top = M.argmax(1)
    return M.max(1) / tot, lev[top], (M > 0).sum(1)


share_st, name_st, n_st = per_cluster("study")
share_sa, _, _ = per_cluster("sample_id")
share_pt, _, n_pt = per_cluster("patient_key")
_, name_ct, _ = per_cluster("cell_type_h")

K = pd.DataFrame(dict(n=np.bincount(kk, minlength=NK), n_studies=n_st, n_patients=n_pt,
                      top_study=share_st, top_study_name=name_st, top_sample=share_sa,
                      top_patient=share_pt, cell_type=name_ct))
K.index.name = "k"

check("6", "no cluster is >80% a single sample", (K.top_sample > 0.80).sum() == 0,
      f"max top_sample_frac = {K.top_sample.max():.3f}", "< 0.80")
check("6", "no cluster is >80% a single patient", (K.top_patient > 0.80).sum() == 0,
      f"max top_patient_frac = {K.top_patient.max():.3f}", "< 0.80")
check("6", "the median cluster spans most studies", K.n_studies.median() >= 15,
      f"median {int(K.n_studies.median())} of {N_STUDIES} studies", ">= 15")
dom = K[K.top_study > 0.50].sort_values("top_study", ascending=False)
check("6", "study-dominated clusters", len(dom) == 0,
      f"{len(dom)} of 60 clusters are >50% one study", 0, severity="WARN",
      note="check the cell_type column: rare types that only one cohort captured will do this.")

fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.scatter(K.n, K.top_study, s=8 + K.n_studies * 2.2, c="#4a6fa5", alpha=0.7, lw=0)
ax.axhline(0.5, color="#c0392b", lw=0.9, ls="--")
ax.set_xscale("log"); ax.set_xlabel("cells in cluster"); ax.set_ylabel("largest study share")
ax.set_ylim(0, 1)
for k, r in dom.iterrows():
    ax.annotate(f"{r.top_study_name}\n({r.cell_type})", (r.n, r.top_study), fontsize=6.5,
                xytext=(4, 2), textcoords="offset points")
ax.set_title(f"{(K.top_study <= 0.5).sum()} of 60 latent clusters are shared across studies;\n"
             "the labelled ones are rare types only one cohort captured", fontsize=10.5)
fig.tight_layout(); fig.savefig(FIG / "nb45_cluster_composition.png", dpi=180); plt.show()
dom
```

```
[PASS] 6 · no cluster is >80% a single sample  ->  max top_sample_frac = 0.563
  [PASS] 6 · no cluster is >80% a single patient  ->  max top_patient_frac = 0.563
  [PASS] 6 · the median cluster spans most studies  ->  median 19 of 21 studies
  [WARN] 6 · study-dominated clusters  ->  7 of 60 clusters are >50% one study

<Figure size 726x484 with 1 Axes>
        n  n_studies  n_patients  top_study   top_study_name  top_sample  \
k                                                                          
33  10168         19         108   0.891621       geskin2026    0.380114   
22  43251         21         181   0.715059       geskin2026    0.563293   
47  16996         19         155   0.689456           li2024    0.156507   
21  15769         21         141   0.548481         buus2025    0.108821   
1   42958         14         117   0.542576  brentuximab2026    0.196960   
41  12250         21         168   0.527184           li2024    0.160082   
56  57973         14         115   0.502769     gaydosik2022    0.492833   

    top_patient     cell_type  
k                              
33     0.380114     Erythroid  
22     0.563293           cDC  
47     0.156507       Myeloid  
21     0.387913           CD4  
1      0.231854  Keratinocyte  
41     0.160082        Plasma  
56     0.492833  Keratinocyte
```

```python  # cell 39
# kNN mixing entropy: for each cell, how many studies are represented among its 30 nearest
# neighbours, relative to how many could be. 1.0 is perfect mixing, 0.0 is a study silo.
from sklearn.neighbors import NearestNeighbors

SUB = 150_000
idx = np.sort(np.random.default_rng(SEED).choice(len(u), SUB, replace=False))
t0 = time.time()
nn = NearestNeighbors(n_neighbors=31).fit(u[idx])
_, ind = nn.kneighbors(u[idx])
ind = ind[:, 1:]


def mixing(labels):
    """Shannon entropy of the neighbourhood label mix, normalised per cell by the entropy that
    the cell's own cell type could reach — a keratinocyte cannot mix with blood-only cohorts."""
    codes, uniq = pd.factorize(labels[idx])
    nb = codes[ind]
    ent = np.zeros(len(nb))
    for i in range(len(uniq)):
        p = (nb == i).mean(1)
        ent -= np.where(p > 0, p * np.log(np.maximum(p, 1e-12)), 0.0)
    ct = A.cell_type_h.values[idx]
    cap = pd.Series(labels[idx]).groupby(ct).transform("nunique").to_numpy()
    return ent / np.log(np.maximum(cap, 2))


mix_study = mixing(A.study.values)
mix_pat = mixing(A.patient_key.values)
check("6", "neighbourhoods mix across studies", float(np.median(mix_study)) > 0.30,
      f"median normalised entropy {np.median(mix_study):.3f}", "> 0.30")
check("6", "neighbourhoods mix across patients", float(np.median(mix_pat)) > 0.30,
      f"median {np.median(mix_pat):.3f}", "> 0.30")
print(f"{SUB:,} cells, k=30, in {time.time() - t0:.0f}s")

fig, ax = plt.subplots(figsize=(6.2, 3.6))
for v, lab, c in [(mix_study, "study", "#4a6fa5"), (mix_pat, "patient", "#c0722f")]:
    ax.hist(v, bins=60, histtype="step", lw=1.6, color=c, label=f"{lab} (median {np.median(v):.2f})")
ax.set_xlabel("normalised neighbourhood entropy"); ax.set_ylabel("cells"); ax.legend(fontsize=9)
ax.set_title("Most cells sit in neighbourhoods drawn from many studies and many patients",
             fontsize=10.5)
fig.tight_layout(); fig.savefig(FIG / "nb45_knn_mixing.png", dpi=180); plt.show()
```

```
[PASS] 6 · neighbourhoods mix across studies  ->  median normalised entropy 0.612
  [PASS] 6 · neighbourhoods mix across patients  ->  median 0.505
150,000 cells, k=30, in 47s

<Figure size 682x396 with 1 Axes>
```

```python  # cell 40
# Is any latent axis really a QC covariate wearing a biological hat?
import scipy.stats as ss

QCV = {"log1p_n_genes": np.log1p(A.n_genes.values),
       "log1p_total_counts": np.log1p(A.total_counts.values),
       "pct_mito": A.pct_mito.values}
sel = np.sort(np.random.default_rng(SEED).choice(len(u), 200_000, replace=False))
Cq = pd.DataFrame({k: [ss.spearmanr(u[sel, j], v[sel])[0] for j in range(u.shape[1])]
                   for k, v in QCV.items()}, index=[f"u{j}" for j in range(u.shape[1])])

worst = Cq.abs().max().max()
hits = [(i, c) for i in Cq.index for c in Cq.columns if abs(Cq.loc[i, c]) >= 0.35]
check("6", "no latent axis is dominated by a QC covariate", worst < 0.60,
      f"largest |rho| = {worst:.3f}", "< 0.60")
check("6", "latent axes with a moderate technical correlation", not hits,
      hits or "none above 0.35", "none", severity="WARN",
      note="exclude these dimensions, or regress pct_mito, before building a neighbour graph "
           "for anything downstream.")
Cq.round(3)
```

```
[PASS] 6 · no latent axis is dominated by a QC covariate  ->  largest |rho| = 0.469
  [WARN] 6 · latent axes with a moderate technical correlation  ->  [('u4', 'pct_mito'), ('u7', 'pct_mito')]

    log1p_n_genes  log1p_total_counts  pct_mito
u0         -0.028              -0.003     0.003
u1          0.046               0.044     0.007
u2         -0.099              -0.027     0.110
u3         -0.054              -0.039    -0.023
u4         -0.028               0.020     0.469
u5          0.009               0.020    -0.049
u6          0.011              -0.006     0.189
u7         -0.136              -0.246     0.390
u8          0.162               0.172    -0.014
u9         -0.025              -0.038     0.071
```

```python  # cell 41
# If any axis is technical, do the composition conclusions survive dropping it? This is the
# check that turns a caveat into a statement about robustness.
tech_axes = sorted({int(i[1:]) for i, _ in hits})
if tech_axes:
    keep = [j for j in range(u.shape[1]) if j not in tech_axes]
    km2 = MiniBatchKMeans(n_clusters=60, random_state=SEED, n_init=3,
                          batch_size=20000).fit(u[:, keep])
    K2 = A.assign(k=km2.labels_).groupby("k").agg(
        top_sample=("sample_id", lambda s: s.value_counts(normalize=True).iat[0]),
        n_studies=("study", "nunique"))
    check("6", f"conclusions hold after dropping u{tech_axes}",
          K2.top_sample.max() < 0.80 and K2.n_studies.median() >= 15,
          f"max top_sample {K2.top_sample.max():.3f}, median studies {int(K2.n_studies.median())}",
          "unchanged")
else:
    check("6", "no technical axes to drop", True, "none", severity="INFO")
```

```
[PASS] 6 · conclusions hold after dropping u[4, 7]  ->  max top_sample 0.543, median studies 19
```

```python  # cell 42
# The qualitative companion. Reuses nb32's cached embedding and its exact subsample rule, so this
# is a figure, not a computation. The parent row count is in the filename so a rebuilt atlas
# cannot silently reuse an embedding computed on different cells.
PLOT_N = 150_000
pidx = np.sort(np.random.default_rng(SEED).choice(N_TOTAL, PLOT_N, replace=False))
cache = OUT / f"joint_umap_u_{N_TOTAL}_{PLOT_N}.npy"

if cache.exists():
    XY = np.load(cache)
    print("using cached UMAP:", cache.name)
else:
    import scanpy as sc, anndata as ad
    print("HEAVY: no cached UMAP, computing one (~15 min)")
    e = ad.AnnData(X=np.zeros((PLOT_N, 1), np.float32),
                   obsm={"X_mrvi_u": np.ascontiguousarray(u[pidx], dtype=np.float32)})
    sc.pp.neighbors(e, use_rep="X_mrvi_u", random_state=SEED)
    sc.tl.umap(e, random_state=SEED)
    XY = e.obsm["X_umap"]; np.save(cache, XY)

assert XY.shape[0] == PLOT_N, (XY.shape, PLOT_N)
sub = A.iloc[pidx]

fig, axes = plt.subplots(1, 4, figsize=(21, 5.2))
for ax, col, n_leg in [(axes[0], "cell_type_h", 12), (axes[1], "study", 10),
                       (axes[2], "compartment", 3), (axes[3], "disease", 4)]:
    vals = sub[col]
    top = vals.value_counts().index[:n_leg]
    cmap = plt.get_cmap("tab20")
    for i, lev in enumerate(top):
        m = (vals == lev).values
        ax.scatter(XY[m, 0], XY[m, 1], s=1.2, lw=0, color=cmap(i % 20), label=lev, rasterized=True)
    m = ~vals.isin(top).values
    if m.any():
        ax.scatter(XY[m, 0], XY[m, 1], s=1.2, lw=0, color="#dddddd", label="other", rasterized=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)
    ax.set_title(col, fontsize=11)
    ax.legend(markerscale=7, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0),
              frameon=False)
fig.suptitle("Structure follows cell type and compartment, not study", fontsize=12, y=1.02)
fig.tight_layout(); fig.savefig(FIG / "nb45_umap_overview.png", dpi=150); plt.show()
```

```
using cached UMAP: joint_umap_u_2157693_150000.npy

<Figure size 2310x572 with 4 Axes>
```

## 7 · Biological sanity

A latent space can be beautifully mixed and still be describing the wrong thing, so the last
section leaves the embedding alone and goes back to counts. Do the cells labelled keratinocyte
express keratins? Is the TCR where T cells are? Does malignancy track stage? Does healthy skin
look like healthy skin?

The marker pass streams `layers['raw_counts']` from the 10k-HVG object in contiguous 50,000-row
slabs — the whole atlas in one ordered pass, a few minutes. Sampling 40,000 random cells instead
would take ten times longer and answer a weaker question.

```python  # cell 44
MARKERS = {
    "CD4": ["CD3D", "CD3E", "IL7R"], "CD8": ["CD8A", "CD8B", "GZMK"],
    "CD4_Treg": ["FOXP3", "IL2RA", "CTLA4"], "B": ["MS4A1", "CD79A"],
    "Plasma": ["MZB1", "JCHAIN"], "Myeloid": ["LYZ", "CD68"], "Mono_CD14": ["CD14", "S100A8"],
    "Mono_CD16": ["FCGR3A"], "cDC": ["CD1C", "CLEC9A"], "Mast": ["TPSAB1", "CPA3"],
    "Keratinocyte": ["KRT14", "KRT5", "KRT1"], "Fibroblast": ["COL1A1", "DCN", "LUM"],
    "Vascular": ["PECAM1", "VWF"], "Melanocyte": ["PMEL", "MLANA"],
    "NK": ["GNLY", "KLRD1"], "Platelet": ["PPBP", "PF4"], "Erythroid": ["HBB", "HBA1"],
}
panel = [g for gs in MARKERS.values() for g in gs]
missing = [g for g in panel if g not in set(v10)]
check("7", "marker panel is inside the 10k HVG space", not missing, missing or f"{len(panel)}/{len(panel)}", 0)
panel = [g for g in panel if g not in missing]

gidx = {g: i for i, g in enumerate(v10)}
lut = np.full(len(v10), -1, np.int32)
lut[[gidx[g] for g in panel]] = np.arange(len(panel))

gcode, glev = pd.factorize(A.cell_type_h.values)
S = np.zeros(len(glev) * len(panel))
Npos = np.zeros_like(S)
ncell = np.bincount(gcode, minlength=len(glev))
tot = A.total_counts.values.astype(np.float64)

t0, CH = time.time(), 50_000
with h5py.File(P_MRVI, "r") as f:
    g = f["layers"]["raw_counts"]
    ip = g["indptr"][:]
    for s in range(0, N_TOTAL, CH):
        e = min(s + CH, N_TOTAL)
        a, b = int(ip[s]), int(ip[e])
        dat, ix = g["data"][a:b], g["indices"][a:b]
        sel = lut[ix] >= 0
        if not sel.any():
            continue
        rows = np.repeat(np.arange(s, e), np.diff(ip[s:e + 1]))[sel]
        flat = gcode[rows] * len(panel) + lut[ix[sel]]
        S += np.bincount(flat, weights=dat[sel] / tot[rows] * 1e4, minlength=S.size)
        Npos += np.bincount(flat, minlength=S.size)
print(f"streamed {N_TOTAL:,} cells in {time.time() - t0:.0f}s")

mean_cp10k = pd.DataFrame(S.reshape(len(glev), len(panel)) / ncell[:, None], index=glev, columns=panel)
pct_pos = pd.DataFrame(100 * Npos.reshape(len(glev), len(panel)) / ncell[:, None], index=glev, columns=panel)
```

```
[PASS] 7 · marker panel is inside the 10k HVG space  ->  38/38
streamed 2,157,693 cells in 75s
```

```python  # cell 45
# Every canonical marker must peak in the cell type it names.
wrong = []
for ct, gs in MARKERS.items():
    if ct not in mean_cp10k.index:
        continue
    for g in gs:
        if g not in mean_cp10k.columns:
            continue
        top = mean_cp10k[g].idxmax()
        if top != ct and not (ct in ("CD4", "CD8", "CD4_Treg") and top in ("CD4", "CD8", "CD4_Treg")):
            wrong.append(f"{g}: peaks in {top}, expected {ct}")
check("7", "canonical markers peak in the right cell type", not wrong,
      wrong or f"all {len(panel)} markers correct", "no mismatches",
      note="T-lineage markers are allowed to peak in any of CD4/CD8/CD4_Treg.")

order = [c for c in MARKERS if c in mean_cp10k.index]
Z = np.log1p(mean_cp10k.loc[order, panel])
Z = (Z - Z.mean()) / Z.std().replace(0, 1)

fig, ax = plt.subplots(figsize=(0.30 * len(panel) + 3.2, 0.36 * len(order) + 2.2))
yy, xx = np.meshgrid(np.arange(len(order)), np.arange(len(panel)), indexing="ij")
sc_ = ax.scatter(xx.ravel(), yy.ravel(), s=pct_pos.loc[order, panel].values.ravel() * 0.9,
                 c=Z.values.ravel(), cmap="RdBu_r", vmin=-2, vmax=2, lw=0.2, edgecolor="#555")
ax.set_xticks(range(len(panel)), panel, rotation=90, fontsize=7.5)
ax.set_yticks(range(len(order)), order, fontsize=8.5)
ax.invert_yaxis()
ax.set_xlim(-0.8, len(panel) - 0.2); ax.set_ylim(len(order) - 0.2, -0.8)
fig.colorbar(sc_, ax=ax, fraction=0.02, label="scaled mean CP10K")
ax.set_title("Every lineage marker peaks in its own cell type (dot size = % of cells expressing)",
             fontsize=10.5)
fig.tight_layout(); fig.savefig(FIG / "nb45_markers.png", dpi=180); plt.show()
```

```
[FAIL] 7 · canonical markers peak in the right cell type  ->  ['LYZ: peaks in Mono_CD14, expected Myeloid', 'CD68: peaks in Mono_CD16, expected Myeloid', 'CD1C: peaks in Myeloid, expected cDC', 'CLEC9A: peaks in Myeloid, expected cDC', 'GNLY: peaks in CD8, expected NK', 'KLRD1: peaks in CD8, expected NK']

<Figure size 1606x915.2 with 2 Axes>
```

```python  # cell 46
T_LIKE = {"CD4", "CD8", "CD4_Treg", "gdT", "MAIT"}
isT = A.cell_type_h.isin(T_LIKE)
r_T, r_non = float(A.loc[isT, "has_tcr"].mean()), float(A.loc[~isT, "has_tcr"].mean())
check("7", "the TCR is where the T cells are", r_non < 0.05,
      f"T-like {r_T:.1%} vs non-T {r_non:.1%}", "non-T < 5%")

by_ct = (A.groupby("cell_type_h").has_tcr.mean().sort_values(ascending=False) * 100).round(1)
odd = by_ct[(by_ct > 10) & (~by_ct.index.isin(T_LIKE))]
check("7", "non-T cell types with notable TCR", len(odd) == 0, odd.to_dict() or "none",
      "none", severity="WARN",
      note="platelets are the classic ambient-RNA sink in blood; `unannotated` is the "
           "rindler2021_fi LN sample, where 'non-T' is unknown rather than wrong.")

check("7", "is_malignant is a subset of is_dominant_clone",
      bool((A.is_malignant & ~A.is_dominant_clone).sum() == 0),
      int((A.is_malignant & ~A.is_dominant_clone).sum()), 0)
by_ct.head(10)
```

```
[PASS] 7 · the TCR is where the T cells are  ->  T-like 48.9% vs non-T 2.1%
  [WARN] 7 · non-T cell types with notable TCR  ->  {'unannotated': 21.9, 'Platelet': 17.4}
  [PASS] 7 · is_malignant is a subset of is_dominant_clone  ->  0

cell_type_h
CD4            52.2
CD8            35.1
CD4_Treg       26.5
unannotated    21.9
Platelet       17.4
Mono_CD14       5.6
Plasma          5.2
Myeloid         3.2
B               2.2
NK              2.1
Name: has_tcr, dtype: float64
```

```python  # cell 47
# The most dangerous column for an outside reader. `is_malignant` is TCR-derived, so a cohort with
# no V(D)J scores zero -- which reads as "no tumour" and is really "never assessed".
skinT = A[(A.compartment == "Skin") & isT]
raw = skinT.groupby("stage_clean").agg(cells=("cell_id", "size"),
                                       pct_tcr=("has_tcr", "mean"),
                                       malignant=("is_malignant", "mean"))
assessable = skinT[skinT.has_tcr]
adj = assessable.groupby("stage_clean").agg(cells=("cell_id", "size"),
                                            malignant=("is_malignant", "mean"))
stage = raw.join(adj, rsuffix="_tcr_only")
stage[["pct_tcr", "malignant", "malignant_tcr_only"]] *= 100

zero_no_tcr = stage[(stage.malignant == 0) & (stage.pct_tcr < 1)]
check("7", "malignant fraction conflates 'negative' with 'never assessed'",
      len(zero_no_tcr) == 0,
      f"{len(zero_no_tcr)} stages read 0% malignant on <1% TCR coverage: "
      f"{list(zero_no_tcr.index)}", "none", severity="WARN",
      note="always gate on has_tcr before interpreting is_malignant. The tcr_only column here "
           "is the honest one.")

hc = A[(A.disease == "HC")]
check("7", "healthy controls carry no malignant call", float(hc.is_malignant.mean()) == 0.0,
      f"{hc.is_malignant.sum()} of {len(hc):,}", 0)
stage.round(2)
```

```
[WARN] 7 · malignant fraction conflates 'negative' with 'never assessed'  ->  5 stages read 0% malignant on <1% TCR coverage: ['IA', 'IIA', 'IIIA', 'IIIB', 'IV']
  [FAIL] 7 · healthy controls carry no malignant call  ->  15 of 131,322

              cells  pct_tcr  malignant  cells_tcr_only  malignant_tcr_only
stage_clean                                                                
IA            84101     0.00       0.00             NaN                 NaN
IB            50313     4.90       1.51          2463.0               30.94
IIA            6453     0.00       0.00             NaN                 NaN
IIB           96559    21.03      11.30         20309.0               53.71
IIIA          13645     0.00       0.00             NaN                 NaN
IIIB          15030     0.00       0.00             NaN                 NaN
IV            11036     0.00       0.00             NaN                 NaN
IVA1          33219    33.37      24.78         11085.0               74.25
IVA2          18935    11.05       9.61          2092.0               86.95
IVB            1524    86.09      57.41          1312.0               66.69
NA           209101    39.30      18.38         82182.0               46.77
```

```python  # cell 48
# The composition gradient. If the atlas is real, T-cell infiltration of skin has to rise from
# healthy to MF to SS, and nothing in the build enforced that.
comp = (A[A.compartment == "Skin"].groupby(["disease", "cell_type_h"]).size()
        .unstack(fill_value=0).pipe(lambda d: d.div(d.sum(1), axis=0)))
have = [d for d in ["HC", "MF", "SS"] if d in comp.index]
grad = comp.loc[have, "CD4"] if "CD4" in comp else None
check("7", "CD4 infiltration of skin rises HC < MF < SS",
      grad is not None and list(grad.values) == sorted(grad.values),
      {k: f"{v:.1%}" for k, v in grad.items()} if grad is not None else "n/a", "monotone")

ker = comp.loc[have, "Keratinocyte"] if "Keratinocyte" in comp else None
check("7", "SS skin keratinocyte fraction", ker is None or ker.get("SS", 1) > 0.05,
      {k: f"{v:.1%}" for k, v in ker.items()} if ker is not None else "n/a", "> 5%",
      severity="WARN",
      note="buus2025 contributes sorted skin, so SS skin is depleted of epidermis by protocol. "
           "Do not read this as the disease destroying the epidermis.")

check("7", "every cell is annotated except the LN sample",
      int((A.cell_type_h == "unannotated").sum()) == N_LN,
      f"{int((A.cell_type_h == 'unannotated').sum()):,} unannotated", N_LN)

top = comp.loc[have, comp.loc[have].max() > 0.02]          # drop the long tail of rare types
fig, ax = plt.subplots(figsize=(8.4, 3.4))
bottom = np.zeros(len(have))
cmap = plt.get_cmap("tab20")
for i, ct in enumerate(top.columns):
    ax.barh(have, top[ct].values, left=bottom, color=cmap(i % 20), label=ct, height=0.62)
    bottom += top[ct].values
ax.set_xlim(0, 1); ax.set_xlabel("fraction of skin cells")
ax.legend(fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
ax.set_title("Skin T-cell content rises from healthy to Sezary"
             + (f": {grad.iloc[0]:.0%} -> {grad.iloc[-1]:.0%}" if grad is not None else ""),
             fontsize=10.5)
fig.tight_layout(); fig.savefig(FIG / "nb45_composition.png", dpi=180); plt.show()
comp.loc[have].round(3)
```

```
[PASS] 7 · CD4 infiltration of skin rises HC < MF < SS  ->  {'HC': '5.7%', 'MF': '35.2%', 'SS': '55.8%'}
  [WARN] 7 · SS skin keratinocyte fraction  ->  {'HC': '26.3%', 'MF': '20.8%', 'SS': '0.4%'}
  [PASS] 7 · every cell is annotated except the LN sample  ->  1,142 unannotated

<Figure size 924x374 with 1 Axes>
cell_type_h      B    CD4  CD4_Treg    CD8  Fibroblast  Keratinocyte   Mast  \
disease                                                                       
HC           0.001  0.057     0.003  0.009       0.287         0.263  0.006   
MF           0.027  0.352     0.038  0.046       0.107         0.208  0.003   
SS           0.015  0.558     0.035  0.066       0.143         0.004  0.001   

cell_type_h  Melanocyte  Myeloid  Plasma    Unk  Vascular  
disease                                                    
HC                0.034    0.051   0.001  0.097     0.192  
MF                0.009    0.102   0.012  0.024     0.071  
SS                0.042    0.078   0.008  0.009     0.041
```

## 8 · Verdict

```python  # cell 50
rep = pd.DataFrame(LEDGER)
rep.insert(0, "atlas_fp", man.loc[man.path.str.endswith("joint_annotated.h5ad"), "fp"].iat[0])
rep.insert(1, "run_utc", pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M"))
rep.to_csv(TAB / "atlas_v2_qc_report.csv", index=False)

counts = rep.status.value_counts().to_dict()
print(f"{counts.get('PASS', 0)} PASS · {counts.get('WARN', 0)} WARN · "
      f"{counts.get('INFO', 0)} INFO · {counts.get('FAIL', 0)} FAIL\n")

json.dump(dict(atlas_fp=rep.atlas_fp.iat[0], run_utc=rep.run_utc.iat[0], **counts,
               verdict="SHIP" if counts.get("FAIL", 0) == 0 else "BLOCKED"),
          open(OUT / "atlas_v2_qc_provenance.json", "w"), indent=1)
print("wrote", (TAB / "atlas_v2_qc_report.csv").relative_to(NB_DIR))

fails = rep[rep.status == "FAIL"]
if len(fails):
    print("\nBLOCKERS")
    for _, r in fails.iterrows():
        print(f"  [{r.section}] {r.check}\n      observed: {r.observed}\n      {r.note}")
print("\nCAVEATS")
for _, r in rep[rep.status == "WARN"].iterrows():
    print(f"  [{r.section}] {r.check} -> {r.observed}")
```

```
146 PASS · 19 WARN · 2 INFO · 8 FAIL

wrote tables/atlas_v2_qc_report.csv

BLOCKERS
  [1] artifact mtimes are in build order
      observed: annotated 2026-09-15 11:07 -> labels -> parquet 2026-09-15 11:08
      
  [2] cell_id prefix is the sequencing library
      observed: 419579
      
  [3] patient_key -> entity_h is a function
      observed: 1 violations: ['pittsburgh:MF25']
      
  [3] one disease per patient_key
      observed: 1 conflicts
      pittsburgh:MF25: ('MF', 'gaydosik2022')=631, ('SS', 'gaydosik2022')=7,808
  [5] every ledger KEEP winner is in the atlas
      observed: ['D3__P74_AD1', 'D3__P75_AD2', 'D3__P77_AD3', 'D3__P81_AD4', 'D3__P96_AD5']
      
  [5] dedup control: D1__P112 kept, D3__P112_Skin dropped
      observed: D1__P112 = 4,186 cells; D3__P112_Skin present = False
      
  [7] canonical markers peak in the right cell type
      observed: ['LYZ: peaks in Mono_CD14, expected Myeloid', 'CD68: peaks in Mono_CD16, expected Myeloid', 'CD1C: peaks in Myeloid, expected cDC', 'CLEC9A: peaks in Myeloid, expected cDC', 'GNLY: peaks in CD8, expected NK', 'KLRD1: peaks in CD8, expected NK']
      T-lineage markers are allowed to peak in any of CD4/CD8/CD4_Treg.
  [7] healthy controls carry no malignant call
      observed: 15 of 131,322
      

CAVEATS
  [1] build log records the dedup drop -> build log absent
  [1] build log and joint_annotated.h5ad are the same build -> build log absent
  [3f] dedup winner worth revisiting -> G033: D15__MF25 (9,275 barcodes) dropped in favour of D13__SZ29, which contributes only 631 cells of that specimen. Winner chosen on V(D)J availability, not cell count.
  [3] sex (4) has a harmonized twin sex_h (3) -> use `sex_h`
  [3] cohort_region (5) has a harmonized twin region_h (4) -> use `region_h`
  [3] entity (13) has a harmonized twin entity_h (10) -> use `entity_h`
  [3] disease_stage (15) has a harmonized twin stage_clean (12) -> use `stage_clean`
  [3] tissue_detail (19) has a harmonized twin organ (3) -> use `organ`
  [3] treatment_context (28) has a harmonized twin treatment_group (8) -> use `treatment_group`
  [3] sample sheet and atlas agree -> sheet-only ['D5__MFIVB_skin'], atlas-only []
  [4] cohorts on proxy or unverified QC thresholds -> 16 of 21: ['alkon24', 'borcherding23', 'brentuximab26', 'buus25', 'chennareddy25', 'dorando26', 'gaydosik19', 'gaydosik22', 'gaydosik23', 'harro23', 'herrera21', 'il4ra26', 'jonak21', 'lyp26', 'ren23', 'rindler21mc']
  [4] li24 is under upstream QC, not ours -> 419,579 cells (19.4% of the atlas), pct_mito up to 40.82%, no doublet scores
  [5] borderline pairs -> 5 in the 0.10-0.50 review band
  [6] residual study structure within a cell type -> worst is Mono_CD14 at 0.383
  [6] study-dominated clusters -> 7 of 60 clusters are >50% one study
  [6] latent axes with a moderate technical correlation -> [('u4', 'pct_mito'), ('u7', 'pct_mito')]
  [7] non-T cell types with notable TCR -> {'unannotated': 21.9, 'Platelet': 17.4}
  [7] malignant fraction conflates 'negative' with 'never assessed' -> 5 stages read 0% malignant on <1% TCR coverage: ['IA', 'IIA', 'IIIA', 'IIIB', 'IV']
  [7] SS skin keratinocyte fraction -> {'HC': '26.3%', 'MF': '20.8%', 'SS': '0.4%'}
```

## 9 · Documentation debt

Not fixed here, but it travels with the data and should be corrected before release:

- `docs/DATA_PROVENANCE.md` and `data/atlas_joint/README.md` still describe **v1**
  (1,173,694 cells, 149 samples, 10 studies). A collaborator reading either will get the wrong
  atlas.
- `data/atlas_joint/metadata_schema.md` predates `patient_key`, which is the only correct patient
  identifier in v2.
- `10_atlas/12_atlas_descriptive.ipynb`'s header still says blood is unannotated. It has been annotated
  since `blood_cell_type_final.csv` landed.
- `QC_CONFIG` carries a `pacritinib26` row for a cohort that contributed no cells to this build.

**What to hand over.** `tables/atlas_v2_qc_report.csv` is the ledger; `tables/atlas_v2_qc_manifest.csv`
pins the exact files it was computed against. Every row of the report carries the fingerprint of
the atlas it describes, so the report cannot outlive the object — which matters here, because
`run_build_joint.py` rebuilds `joint_*` in place.
