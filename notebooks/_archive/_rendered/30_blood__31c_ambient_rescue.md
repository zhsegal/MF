# 30_blood/31c_ambient_rescue.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/30_blood/31c_ambient_rescue.ipynb` and is never run._

# 31c — Blood ambient-soup rescue (geskin2026)

31b §5 labels ~25.8k blood non-T cells `Ambient_soup`. **91% come from one geskin2026 lane,
`B4__PF48`**. It is overloaded: 55k filtered barcodes, and HTO2 alone holds 19k cells. Most of the
rest come from `B4__SZ32`. The cells have singlet-like depth and a mixed lymphoid + myeloid
profile. The residual scVI uses `batch = study`, so it never sees lanes.

**Question.** Is each cell one of the following?
- (a) a lymphocyte coated in monocyte-derived ambient RNA → label it
- (b) a same-HTO T+monocyte doublet, which HashSolo cannot see → `Doublet`
- (c) neither → leave it `Ambient_soup` and **stop**

**Evidence the atlas lacks.** The raw 10x h5 files carry 20–27 ADTs (CD3/CD4/CD8; CD19 in the PF
lanes) and the HTO counts. The atlas `X_adt` is NaN for geskin.

**Method**
- §1 loads raw RNA + ADT + HTO and builds a soup profile per lane.
- §2 diagnoses each cell from contamination ρ, a kNN against synthetic doublets and synthetic
  ambient cells, HTO depth and CD3 protein. It ends in an explicit futility gate.
- §3 trains a soup-robust classifier on clean geskin cells. A label is accepted only when the
  protein agrees.

**Writes** `data/atlas_joint/blood_ambient_rescue.csv` (+ provenance). 31b §5d reads it, so 31b
stays the only writer of `blood_subtypes.csv`.

> CPU-only, no scVI: `jobs/run_notebook.sh notebooks/30_blood/31c_ambient_rescue.ipynb`.

```python  # cell 1
# ============================================================================
# §0  Setup
# ============================================================================
import warnings; warnings.filterwarnings("ignore")
import json
import sys
import time
from pathlib import Path

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import scipy.sparse as sp
from anndata.io import read_elem
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
import annot_utils as A

OUT_DIR = NB_DIR / "data" / "atlas_joint"
B4_DIR = NB_DIR / "data" / "B4_geskin2026_dupilumab_blood"
HTO_ARMS = NB_DIR / "data" / "clinical" / "raw" / "b4_hto_arms.tsv"
FIG_DIR = NB_DIR / "figures" / "blood_subtypes"; FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR = NB_DIR / "tables"

OBS_PQ = OUT_DIR / "atlas_obs_full_v2.parquet"
BROAD_CSV = OUT_DIR / "blood_cell_type_final.csv"      # nb31: CD4 / CD8 / NK for the T object
SUBTYPE_CSV = OUT_DIR / "blood_subtypes.csv"           # nb31b: non-T subtypes incl. Ambient_soup
RESID_H5AD = OUT_DIR / "blood_nonT_resid_v1.h5ad"      # nb31b §5: leiden_resid of the Unk pool
RESCUE_CSV = OUT_DIR / "blood_ambient_rescue.csv"
RESCUE_PROV = OUT_DIR / "blood_ambient_rescue_provenance.json"

SEED = 0
MIN_SOUP_LANE = 20        # lanes with fewer soup cells are left alone
N_REF_KNN = 5000          # real lane singlets per class in the kNN reference
N_SYN = 3000              # synthetic doublets and synthetic ambient cells, per lane
AMB_F = (0.1, 0.6)        # ambient fraction of the synthetic ambient cells
K_NN = 25
N_HVG_KNN, N_HVG_CLF, N_PC = 2000, 2000, 50
N_TRAIN_PER_CLASS = 6000
SOUP_MASS_DROP = 0.5      # classifier drops the top soup genes carrying this share of soup mass
P_ACCEPT = 0.8
GATE_RESOLVED, GATE_ADT, GATE_LABELLED = 0.40, 0.30, 0.30

# Genes a lymphocyte does not make: in a lymphocyte their counts are ambient by construction,
# which is what lets rho be read as a contamination fraction (SoupX's non-expressed-gene trick).
MYELOID = ["LYZ", "S100A8", "S100A9", "S100A12", "FCN1", "CST3", "CD14", "VCAN", "CTSS",
           "TYROBP", "FCER1G", "LST1", "AIF1"]
LYMPH = ["T_CD4", "T_CD8", "NK", "B"]
MYELO = ["Mono_classical", "Mono_nonclassical"]
rng = np.random.default_rng(SEED)
sc.settings.verbosity = 1
T0 = time.time()
print("NB_DIR =", NB_DIR)
```

```
NB_DIR = /home/projects/nyosef/zvise/MF
```

## §1 — Raw RNA + ADT + HTO, and a soup profile per lane

The soup profile comes from barcodes CellRanger called but the atlas dropped (HashSolo negatives,
multiplets, QC fails), taking their low-UMI half. A filtered matrix has no empty droplets, so this
is the closest substitute. When a lane has too few such barcodes the fallback is the lane
pseudobulk.

```python  # cell 3
# ============================================================================
# §1  Load
# ============================================================================
obs = (pl.read_parquet(OBS_PQ, columns=["cell_id", "study", "sample_id", "donor", "clin_arm",
                                         "clin_ex_vivo", "n_genes", "pct_mito",
                                         "doublet_score"])
       .filter(pl.col("study") == "geskin2026").to_pandas().set_index("cell_id"))
broad = (pl.read_csv(BROAD_CSV, columns=["cell_id", "cell_type_broad", "cell_type_final"])
         .filter(pl.col("cell_id").str.starts_with("B4__")).to_pandas().set_index("cell_id"))
subt = (pl.read_csv(SUBTYPE_CSV).filter(pl.col("cell_id").str.starts_with("B4__"))
        .to_pandas().set_index("cell_id"))
obs = obs.join(broad).join(subt[["subtype_fine", "annotation_round"]])

# Soup = what 31b left Ambient_soup PLUS what an earlier run of this notebook already rescued
# (31b relabels those, so without this a re-run would see a shrinking pool).
_r3 = obs["annotation_round"].astype(str).eq("round3_ambient")
obs["is_soup"] = obs["subtype_fine"].astype(str).eq("Ambient_soup") | _r3

NONT2REF = {"Mono_classical": "Mono_classical", "Mono_nonclassical": "Mono_nonclassical",
            "B_naive": "B", "B_memory": "B", "Plasma": "Plasma", "cDC1": "cDC", "cDC2": "cDC",
            "pDC": "pDC", "Erythroid": "Erythroid", "Platelet": "Platelet",
            "NK_CD56bright": "NK", "HSPC": "HSPC"}
T2REF = {"CD4": "T_CD4", "CD8": "T_CD8", "NK": "NK"}
ref = obs["subtype_fine"].map(NONT2REF)
ref = ref.where(obs["subtype_fine"].notna(), obs["cell_type_final"].map(T2REF))
ref[obs["is_soup"]] = np.nan                         # never train on the cells being rescued
obs["ref"] = ref
print(f"geskin2026 atlas cells: {len(obs):,} | soup: {int(obs.is_soup.sum()):,} | "
      f"clean reference: {int(obs.ref.notna().sum()):,}")
print(pd.crosstab(obs["sample_id"].astype(str), obs["ref"].fillna("-")).to_string())

smp = pd.read_csv(B4_DIR / "meta" / "samples.tsv", sep="\t")
arms = pd.read_csv(HTO_ARMS, sep="\t")
parts, prot, soup_raw, soup_meta, genes0 = [], [], {}, {}, None
for r in smp.itertuples():
    lane = f"B4__{r.sample_id}"
    a = sc.read_10x_h5(B4_DIR / "raw" / r.gex_file, gex_only=False)
    a.var_names_make_unique()
    a.obs_names = lane + "|" + a.obs_names.str.replace(r"-1$", "", regex=True)
    is_g = (a.var["feature_types"] == "Gene Expression").to_numpy()
    if genes0 is None:
        genes0 = a.var_names[is_g]
    assert a.var_names[is_g].equals(genes0), f"{lane}: gene order differs"
    X = sp.csr_matrix(a.X[:, is_g], dtype=np.float32)
    umi = np.asarray(X.sum(1)).ravel()
    inat = np.asarray(a.obs_names.isin(obs.index))
    out = ~inat
    if out.sum() >= 200:
        lo = out & (umi <= np.quantile(umi[out], 0.5))
        src = "non-atlas barcodes, low-UMI half"
    else:
        lo, src = inat, "lane pseudobulk (fallback)"
    soup_raw[lane] = np.asarray(X[lo].sum(0)).ravel()
    soup_meta[lane] = {"source": src, "n_barcodes": int(lo.sum()),
                       "median_umi": float(np.median(umi[lo])), "n_filtered": int(a.n_obs),
                       "n_atlas": int(inat.sum())}
    parts.append(ad.AnnData(X[inat], obs=pd.DataFrame(index=a.obs_names[inat]),
                            var=pd.DataFrame(index=genes0)))
    P = a[inat, ~is_g].X
    prot.append(pd.DataFrame(P.toarray() if sp.issparse(P) else np.asarray(P),
                             index=a.obs_names[inat], columns=a.var_names[~is_g]))
    print(f"{lane}: {a.n_obs:,} filtered barcodes, {inat.sum():,} in atlas | soup: {src}, "
          f"{lo.sum():,} barcodes, median UMI {np.median(umi[lo]):.0f}")
    del a, X

gex = ad.concat(parts, join="inner"); del parts
missing = obs.index.difference(gex.obs_names)
print(f"\natlas geskin cells not found in the raw h5: {len(missing):,}")
gex.obs = obs.reindex(gex.obs_names)
_nc = np.bincount(gex.X.indices, minlength=gex.n_vars)
keep = _nc >= 20
gex = gex[:, keep].copy()
G = pd.Index(gex.var_names)
soup = pd.DataFrame({k: v[keep] for k, v in soup_raw.items()}, index=G)
soup_p = soup / soup.sum(0)
prot = pd.concat(prot).reindex(gex.obs_names)
Xc = gex.X.tocsr()
o = gex.obs
o["umi"] = np.asarray(Xc.sum(1)).ravel()
lanes_arr = o["sample_id"].astype(str).to_numpy()

# HTO depth: total over the lane's own HTOs, and 2nd/1st ratio
HTO_COLS = [c for c in prot.columns if c.startswith("HTO")]
ADT_COLS = [c for c in prot.columns if c not in HTO_COLS]
H = prot[HTO_COLS].fillna(0).to_numpy()
hcol = {c: i for i, c in enumerate(HTO_COLS)}
hto_total, hto_r2 = np.zeros(gex.n_obs), np.zeros(gex.n_obs)
for lane, hl in arms.groupby("sample_id")["hto"].apply(list).items():
    m = lanes_arr == lane
    if not m.any():
        continue
    sub_h = H[m][:, [hcol[h] for h in hl if h in hcol]]
    srt = np.sort(sub_h, 1)
    hto_total[m] = sub_h.sum(1)
    hto_r2[m] = srt[:, -2] / np.maximum(srt[:, -1], 1) if sub_h.shape[1] > 1 else 0
o["hto_total"], o["hto_r2"] = hto_total, hto_r2

# ADT CLR per lane (log1p minus the lane mean, per antibody)
clr = pd.DataFrame(np.nan, index=prot.index, columns=ADT_COLS)
for lane in np.unique(lanes_arr):
    m = lanes_arr == lane
    L = np.log1p(prot.loc[m, ADT_COLS].to_numpy(float))
    clr.loc[m] = L - np.nanmean(L, 0)

mye = [g for g in MYELOID if g in G]
mi = G.get_indexer(mye)
o["mye_share"] = np.asarray(Xc[:, mi].sum(1)).ravel() / np.maximum(o["umi"].to_numpy(), 1)
s_mye = soup_p.loc[mye].sum(0)
o["rho"] = o["mye_share"] / o["sample_id"].astype(str).map(s_mye.to_dict()).to_numpy()

SOUP_LANES = [l for l, n in o.loc[o["is_soup"], "sample_id"].astype(str).value_counts().items()
              if n >= MIN_SOUP_LANE]
print(f"\n{gex.shape} after gene filter | ADTs: {len(ADT_COLS)} | HTOs: {len(HTO_COLS)}")
print("soup lanes (>= %d soup cells):" % MIN_SOUP_LANE,
      o.loc[o["is_soup"], "sample_id"].astype(str).value_counts().loc[SOUP_LANES].to_dict())
for lane in SOUP_LANES[:3] + ["B4__PF46"]:
    s = soup_p[lane].sort_values(ascending=False)
    print(f"\n{lane} soup ({soup_meta[lane]['source']}): myeloid share {s_mye[lane]:.3f}\n  top: "
          + ", ".join(f"{g} {v:.3f}" for g, v in s.head(20).items()))
print(f"\n[{(time.time() - T0) / 60:.1f} min]")
```

```
geskin2026 atlas cells: 260,455 | soup: 25,681 | clean reference: 201,359

ref            -     B  Erythroid  HSPC  Mono_classical  Mono_nonclassical   NK  Plasma  Platelet  T_CD4  T_CD8  cDC  pDC
sample_id                                                                                                                
B4__PF46    1078  2785         40    33            5444                  0   15       4         1   9611   7244    0    9
B4__PF47    2732  1627        754    37              41                  0  345       2         0   9864   7065    0   13
B4__PF48   26907   606          3    29            1463                  1   97       3         0   6885   4312    1    7
B4__PF49    2688  1979          3    33            1472                  0   19      13         0  10714   6474    0   24
B4__PF50    4418  1425         61    46             910                  0   59      15         0  10968   9672    0   31
B4__SZ30    3570   415        155    12               3                  0    4       4         0   5863   3962    0    0
B4__SZ32    6430   458       3939   198               8                  0    9       2         6   6392   3957    1    1
B4__SZ33    6473   356          8     1               3                  0    2       0         0   6484    176    0    0
B4__SZ38     364    49       1200     1              42                  0   50       0         0  24794     98    0    0
B4__SZ39    3526   173       6208    72              28                  0    9      30         0   5447   3683    0    1
B4__SZ5      910   460         90     5             810                  1   56      15         0  22054   1283    0   22

B4__SZ30: 19,972 filtered barcodes, 13,988 in atlas | soup: non-atlas barcodes, low-UMI half, 2,992 barcodes, median UMI 1206

B4__SZ5: 35,743 filtered barcodes, 25,706 in atlas | soup: non-atlas barcodes, low-UMI half, 5,019 barcodes, median UMI 13438

B4__SZ38: 36,722 filtered barcodes, 26,598 in atlas | soup: non-atlas barcodes, low-UMI half, 5,064 barcodes, median UMI 7397

B4__SZ39: 28,361 filtered barcodes, 19,177 in atlas | soup: non-atlas barcodes, low-UMI half, 4,592 barcodes, median UMI 2279

B4__SZ32: 33,729 filtered barcodes, 21,401 in atlas | soup: non-atlas barcodes, low-UMI half, 6,164 barcodes, median UMI 1537

B4__SZ33: 20,399 filtered barcodes, 13,503 in atlas | soup: non-atlas barcodes, low-UMI half, 3,448 barcodes, median UMI 1048

B4__PF46: 37,242 filtered barcodes, 26,264 in atlas | soup: non-atlas barcodes, low-UMI half, 5,489 barcodes, median UMI 3863

B4__PF47: 31,343 filtered barcodes, 22,480 in atlas | soup: non-atlas barcodes, low-UMI half, 4,433 barcodes, median UMI 6478

B4__PF48: 55,478 filtered barcodes, 40,314 in atlas | soup: non-atlas barcodes, low-UMI half, 7,583 barcodes, median UMI 3955

B4__PF49: 33,750 filtered barcodes, 23,419 in atlas | soup: non-atlas barcodes, low-UMI half, 5,167 barcodes, median UMI 5461

B4__PF50: 40,458 filtered barcodes, 27,605 in atlas | soup: non-atlas barcodes, low-UMI half, 6,427 barcodes, median UMI 5347


atlas geskin cells not found in the raw h5: 0


(260455, 25814) after gene filter | ADTs: 27 | HTOs: 10
soup lanes (>= 20 soup cells): {'B4__PF48': 23526, 'B4__SZ32': 1842, 'B4__SZ30': 147, 'B4__PF50': 54, 'B4__SZ33': 45, 'B4__PF49': 32}

B4__PF48 soup (non-atlas barcodes, low-UMI half): myeloid share 0.010
  top: FTH1 0.016, B2M 0.013, FTL 0.012, ACTB 0.012, MT-CO1 0.011, MALAT1 0.010, TMSB4X 0.009, EEF1A1 0.008, RPLP1 0.008, TPT1 0.008, RPL10 0.008, RPS12 0.006, MT-CO2 0.006, TMSB10 0.006, RPL41 0.006, CD74 0.006, RPL28 0.005, RPL13 0.005, LYZ 0.005, GAPDH 0.004

B4__SZ32 soup (non-atlas barcodes, low-UMI half): myeloid share 0.007
  top: HBB 0.164, FTL 0.061, FTH1 0.036, MT-CO2 0.014, MT-CO1 0.011, B2M 0.007, RPLP1 0.007, MALAT1 0.007, TMSB10 0.007, HBA2 0.006, MT-CO3 0.006, MT-ATP6 0.006, RPL41 0.006, ACTB 0.005, RPS12 0.005, TMSB4X 0.005, RPL28 0.005, HBA1 0.004, MTRNR2L12 0.004, RPL10 0.004

B4__SZ30 soup (non-atlas barc
... [662 more chars truncated]
```

## §2 — Diagnose: ambient-coated singlets, same-HTO doublets, or neither?

Each soup lane gets a kNN reference built from:
- real lane lymphocytes and monocytes
- synthetic **T+Mono doublets** (summed counts of random same-lane pairs)
- synthetic **ambient cells** (lymphocyte + Poisson soup at 10–60%)

Each soup cell gets the fraction of its neighbours in each class. Two controls run through the
same reference: held-out real lymphocytes, and 31b's own `Doublet` calls.

**Verdict per cell**
- `doublet`: kNN-doublet ≥ 0.5 **and** HTO or UMI ≥ 1.3× the lane singlets. A doublet carries two
  cells' hashtags and RNA.
- `ambient`: lymph + ambient neighbours ≥ 0.5, ρ ≤ 1.2 and HTO < 1.5×. The myeloid RNA fits
  inside the soup.
- `myeloid`: monocyte neighbours ≥ 0.5.
- `unresolved`: everything else.

**Futility gate:** resolved < 40% **and** CD3-protein-resolvable < 30%. If both hold, §3 is
skipped.

```python  # cell 5
# ============================================================================
# §2a  kNN against synthetic doublets / synthetic ambient cells, per soup lane
# ============================================================================
def lognorm(M):
    s = np.asarray(M.sum(1)).ravel(); s[s == 0] = 1
    return sp.csr_matrix(sp.csr_matrix(M).multiply(1e4 / s[:, None])).log1p()


def poisson_soup(p, n_add, chunk=500):
    out = []
    for s in range(0, len(n_add), chunk):
        out.append(sp.csr_matrix(rng.poisson(n_add[s:s + chunk, None] * p[None, :])
                                 .astype(np.float32)))
    return sp.vstack(out).tocsr()


umi_all = o["umi"].to_numpy()
ref_arr = o["ref"].astype(object).to_numpy()
is_lymph = o["ref"].isin(LYMPH).to_numpy()
is_mono = o["ref"].isin(MYELO).to_numpy()
is_soup = o["is_soup"].to_numpy()
is_31dbl = o["subtype_fine"].astype(str).eq("Doublet").to_numpy()

knn_rows, syn_dbl = [], {}
for lane in SOUP_LANES:
    inl = lanes_arr == lane
    lymph_i = np.flatnonzero(inl & is_lymph)
    mono_i = np.flatnonzero(inl & is_mono)
    note = ""
    if len(mono_i) < 50:
        mono_i, note = np.flatnonzero(is_mono), "  (monocytes borrowed from all geskin lanes)"
    perm = rng.permutation(lymph_i)
    n_ref = min(N_REF_KNN, int(0.8 * len(perm)))
    lymph_ref, lymph_hold = perm[:n_ref], perm[n_ref:n_ref + 2000]
    mono_ref = rng.choice(mono_i, size=min(N_REF_KNN, len(mono_i)), replace=False)

    a_i, b_i = rng.choice(lymph_i, N_SYN), rng.choice(mono_i, N_SYN)
    D = (Xc[a_i] + Xc[b_i]).tocsr()
    syn_dbl[lane] = pd.DataFrame({"umi": umi_all[a_i] + umi_all[b_i],
                                  "hto": hto_total[a_i] + hto_total[b_i]})
    c_i = rng.choice(lymph_i, N_SYN)
    f = rng.uniform(*AMB_F, N_SYN)
    S = (Xc[c_i] + poisson_soup(soup_p[lane].to_numpy(), umi_all[c_i] * f / (1 - f))).tocsr()

    R = sp.vstack([Xc[lymph_ref], Xc[mono_ref], D, S]).tocsr()
    lab = np.array(["lymph"] * len(lymph_ref) + ["mono"] * len(mono_ref)
                   + ["doublet"] * N_SYN + ["ambient"] * N_SYN)
    Lr = lognorm(R)
    hv = ad.AnnData(Lr[: len(lymph_ref) + len(mono_ref)], var=pd.DataFrame(index=G))
    sc.pp.highly_variable_genes(hv, n_top_genes=N_HVG_KNN, flavor="seurat")
    gi = np.union1d(np.flatnonzero(hv.var["highly_variable"].to_numpy()), mi)
    scl = StandardScaler().fit(Lr[:, gi].toarray())
    pca = PCA(N_PC, random_state=SEED).fit(scl.transform(Lr[:, gi].toarray()))
    emb = lambda M: pca.transform(scl.transform(lognorm(M)[:, gi].toarray()))
    nn = NearestNeighbors(n_neighbors=K_NN).fit(emb(R))

    for name, ids in [("soup", np.flatnonzero(inl & is_soup)), ("lymph_heldout", lymph_hold),
                      ("31b_Doublet", np.flatnonzero(inl & is_31dbl))]:
        if not len(ids):
            continue
        fr = {}
        for s0 in range(0, len(ids), 5000):
            _, ix = nn.kneighbors(emb(Xc[ids[s0:s0 + 5000]]))
            for k in ["lymph", "mono", "doublet", "ambient"]:
                fr.setdefault(k, []).append((lab[ix] == k).mean(1))
        knn_rows.append(pd.DataFrame({"cell_id": o.index[ids], "lane": lane, "group": name,
                                      **{f"knn_{k}": np.concatenate(v) for k, v in fr.items()}}))
    print(f"{lane}: ref lymph {len(lymph_ref):,} / mono {len(mono_ref):,} / "
          f"{N_SYN:,} synthetic doublets + ambient | soup {int((inl & is_soup).sum()):,}{note}")

knn = pd.concat(knn_rows).set_index("cell_id")

# Depth relative to the lane's own clean singlets
med_umi = o[is_lymph].groupby(o["sample_id"].astype(str)[is_lymph])["umi"].median().to_dict()
med_hto = (o[is_lymph | is_mono].groupby(o["donor"].astype(str)[is_lymph | is_mono])["hto_total"]
           .median().to_dict())
o["umi_ratio"] = o["umi"] / o["sample_id"].astype(str).map(med_umi).to_numpy()
o["hto_ratio"] = o["hto_total"] / o["donor"].astype(str).map(med_hto).to_numpy()

diag = knn.join(o[["umi", "umi_ratio", "hto_total", "hto_ratio", "hto_r2", "mye_share", "rho",
                   "donor", "clin_arm"]])
_lymph_like = diag["knn_lymph"] + diag["knn_ambient"]
_dbl = (diag["knn_doublet"] >= 0.5) & ((diag["hto_ratio"] >= 1.3) | (diag["umi_ratio"] >= 1.3))
_amb = (_lymph_like >= 0.5) & (diag["rho"] <= 1.2) & (diag["hto_ratio"] < 1.5)
_mye = diag["knn_mono"] >= 0.5
diag["verdict"] = np.select([_dbl, _amb, _mye], ["doublet", "ambient", "myeloid"], "unresolved")

print("\nverdict by group (rows sum to 1) -- the two controls calibrate the soup row:")
print(pd.crosstab([diag["lane"], diag["group"]], diag["verdict"], normalize="index")
      .round(3).to_string())
print("\nmedians by group:")
print(diag.groupby(["lane", "group"])[["umi_ratio", "hto_ratio", "hto_r2", "rho", "knn_lymph",
                                       "knn_ambient", "knn_doublet", "knn_mono"]]
      .median().round(3).to_string())
print(f"\n[{(time.time() - T0) / 60:.1f} min]")
```

```
B4__PF48: ref lymph 5,000 / mono 1,464 / 3,000 synthetic doublets + ambient | soup 23,526

B4__SZ32: ref lymph 5,000 / mono 5,000 / 3,000 synthetic doublets + ambient | soup 1,842  (monocytes borrowed from all geskin lanes)

B4__SZ30: ref lymph 5,000 / mono 5,000 / 3,000 synthetic doublets + ambient | soup 147  (monocytes borrowed from all geskin lanes)

B4__PF50: ref lymph 5,000 / mono 910 / 3,000 synthetic doublets + ambient | soup 54

B4__SZ33: ref lymph 5,000 / mono 5,000 / 3,000 synthetic doublets + ambient | soup 45  (monocytes borrowed from all geskin lanes)

B4__PF49: ref lymph 5,000 / mono 1,472 / 3,000 synthetic doublets + ambient | soup 32

verdict by group (rows sum to 1) -- the two controls calibrate the soup row:

verdict                 ambient  doublet  myeloid  unresolved
lane     group                                               
B4__PF48 31b_Doublet      0.054    0.769    0.012       0.165
         lymph_heldout    0.856    0.000    0.000       0.144
         soup             0.577    0.016    0.000       0.407
B4__PF49 31b_Doublet      0.003    0.912    0.015       0.071
         lymph_heldout    0.687    0.000    0.000       0.313
         soup             0.062    0.188    0.000       0.750
B4__PF50 31b_Doublet      0.038    0.769    0.005       0.188
         lymph_heldout    0.664    0.000    0.000       0.336
         soup             0.259    0.259    0.000       0.481
B4__SZ30 31b_Doublet      0.512    0.047    0.000       0.442
         lymph_heldout    0.804    0.000    0.000       0.196
         soup             0.367    0.306    0.082       0.245
B4__SZ32 31b_Doublet      0.350    0.100    0.100       0.450
         lymph_heldout    0.772    0.000    0.001       0.226
         soup             0.182    0.146    0.022       0.649
B4__SZ33 31b_Doublet      0.000    0.000    0.000       1.000
         lymph_heldout    0.868    0.000    0.001       0.131
         soup             0.311    0.022    0.022       0.644

medians by group:
                        umi_ratio  hto_ratio  hto_r2    rho  knn_lymph  knn_ambient  knn_doublet  knn_mono
lane     group                                                                                            
B4__PF48 31b_Doublet        2.647      3.742   0.033  1.438       0.00         0.00         0.92      0.00
         lymph_heldout      0.999      0.942   0.170  0.117       0.96         0.04         0.00      0.00
         soup               0.842      0.287   0.526  1.090       0.12         0.68         0.04      0.00
B4__PF49 31b_Doublet        3.727      3.385   0.058  5.278       0.00         0.00         1.00      0.00
         lymph_heldout      0.990      0.942   0.266  0.157       0.88         0.12         0.00      0.00
         soup               0.444      0.530   0.110  3.015       0.08         0.32         0.36      0.00
B4__PF50 31b_Doublet        3.317      2.771   0.084  4.101       0.00         0.00         0.96      0.00
         lymph_heldout      1.020      1.006   0.303  0.319       0.88         0.12         0.00      0.00
         soup               0.721      0.992   0.257  1.941       0.12         0.38         0.20      0.00
B4__SZ30 31b_Doublet        1.014      1.383   0.137  0.412       0.88         0.08         0.00      0.00
         lymph_heldout      1.006      0.994   0.137  0.060       0.96         0.04         0.00      0.00
         soup               0.735      2.035   0.132  0.875       0.24         0.08         0.16      0.00
B4__SZ32 31b_Doublet        0.990      1.175   0.201  0.162       0.56         0.14         0.06      0.00
         lymph_heldout      1.007      0.987   0.236  0.046       0.96         0.04         0.00      0.00
         soup               0.763      3.078   0.053  0.964       0.00         0.64         0.12      0.00
B4__SZ33 31b_Doublet        3.519      4.444   0.033  0.416       0.00         0.60         0.36      0.04
         lymph_heldout      0.971      0.990   0.196  0.043       0.96       
... [147 more chars truncated]
```

```python  # cell 6
# ============================================================================
# §2b  CD3 protein, the per-cluster table, and the futility gate
# ============================================================================
ADT_TESTS = {"CD3": ("CD3-TotalSeqC", ["T_CD4", "T_CD8"], MYELO + ["B"]),
             "CD4": ("CD4-TotalSeqC", ["T_CD4"], ["T_CD8"]),
             "CD8": ("CD8-TotalSeqC", ["T_CD8"], ["T_CD4"]),
             "CD19": ("CD19-TotalSeqC", ["B"], ["T_CD4", "T_CD8"])}
soupdf = diag[diag["group"] == "soup"].copy()
thr_rows = []
for lane in SOUP_LANES:
    inl = lanes_arr == lane
    m = (soupdf["lane"] == lane).to_numpy()
    for k, (cname, pos, neg) in ADT_TESTS.items():
        soupdf.loc[m, f"adt_{k}_state"] = "na"
        if cname not in clr:
            continue
        v = clr[cname].to_numpy()
        mp = np.nanmedian(v[inl & o["ref"].isin(pos).to_numpy()])
        mn = np.nanmedian(v[inl & o["ref"].isin(neg).to_numpy()])
        ok = bool(np.isfinite(mp) and np.isfinite(mn) and mp - mn > 0.3)
        thr_rows.append({"lane": lane, "adt": k, "med_pos_ref": mp, "med_neg_ref": mn,
                         "usable": ok})
        vv = clr.loc[soupdf.index[m], cname].to_numpy()
        soupdf.loc[m, f"adt_{k}"] = vv
        if ok:
            thr, band = (mp + mn) / 2, 0.25 * (mp - mn)
            soupdf.loc[m, f"adt_{k}_state"] = np.where(
                vv > thr + band, "pos", np.where(vv < thr - band, "neg", "amb"))
adt_thr = pd.DataFrame(thr_rows)
print("ADT reference medians (CLR) -- a test is usable only if pos and neg refs separate:")
print(adt_thr.round(2).to_string(index=False))

print("\nCD3 protein state of soup cells, by RNA verdict:")
print(pd.crosstab([soupdf["lane"], soupdf["verdict"]], soupdf["adt_CD3_state"], margins=True)
      .to_string())

with h5py.File(RESID_H5AD, "r") as f:
    _o = f["obs"]
    _rid = pd.Series(np.asarray(read_elem(_o["leiden_resid"])).astype(str),
                     index=np.asarray(read_elem(_o[_o.attrs["_index"]])).astype(str))
soupdf["leiden_resid"] = _rid.reindex(soupdf.index).to_numpy()
by_cl = (soupdf.groupby("leiden_resid")
         .agg(n=("lane", "size"), top_lane=("lane", lambda s: s.value_counts().index[0]),
              umi_ratio=("umi_ratio", "median"), hto_ratio=("hto_ratio", "median"),
              rho=("rho", "median"), knn_doublet=("knn_doublet", "mean"),
              knn_lymph_amb=("knn_lymph", "mean"), knn_mono=("knn_mono", "mean"),
              cd3_pos=("adt_CD3_state", lambda s: (s == "pos").mean()),
              cd3_neg=("adt_CD3_state", lambda s: (s == "neg").mean()))
         .join(pd.crosstab(soupdf["leiden_resid"], soupdf["verdict"], normalize="index"))
         .sort_values("n", ascending=False))
by_cl["knn_lymph_amb"] += soupdf.groupby("leiden_resid")["knn_ambient"].mean()
print("\nper residual cluster (31b leiden_resid):")
print(by_cl.round(2).to_string())

frac_resolved = float(soupdf["verdict"].isin(["ambient", "doublet", "myeloid"]).mean())
frac_adt = float(soupdf["adt_CD3_state"].isin(["pos", "neg"]).mean())
FUTILE = (frac_resolved < GATE_RESOLVED) and (frac_adt < GATE_ADT)
print("\n" + "=" * 100)
print(f"GATE: RNA/HTO-resolved {frac_resolved:.1%} (need >= {GATE_RESOLVED:.0%})  |  "
      f"CD3-protein-resolvable {frac_adt:.1%} (need >= {GATE_ADT:.0%})")
print("FUTILE -- leave them Ambient_soup; §3 is skipped" if FUTILE
      else "NOT futile -- continue to §3")
print("=" * 100)

TAB_DIR.mkdir(parents=True, exist_ok=True)
diag.to_csv(TAB_DIR / "nb31c_soup_diagnostics_cells.csv")
by_cl.to_csv(TAB_DIR / "nb31c_soup_diagnostics_by_cluster.csv")
adt_thr.to_csv(TAB_DIR / "nb31c_adt_reference.csv", index=False)

# Figure 1 -- depth: are they two cells' worth of RNA and hashtag, or one?
big = SOUP_LANES[0]
inl = lanes_arr == big
_s = soupdf[soupdf["lane"] == big]
h_med, u_med = _s["hto_ratio"].median(), _s["umi_ratio"].median()
fig, ax = plt.subplots(figsize=(5, 3))
for nm, x, y, c in [
        ("lane lymphocytes", umi_all[inl & is_lymph], hto_total[inl & is_lymph], "#4C72B0"),
        ("lane monocytes", umi_all[inl & is_mono], hto_total[inl & is_mono], "#55A868"),
        ("synthetic T+Mono doublets", syn_dbl[big]["umi"], syn_dbl[big]["hto"], "#C44E52"),
        ("soup cells", _s["umi"], _s["hto_total"], "#8172B2")]:
    x, y = np.asarray(x), np.asarray(y)
    k = rng.choice(len(x), size=min(3000, len(x)), replace=False)
    ax.scatter(np.log10(x[k] + 1), np.log10(y[k] + 1), s=2, alpha=0.3, c=c, label=nm,
               rasterized=True)
ax.set_xlabel("log10 RNA UMI"); ax.set_ylabel("log10 HTO counts")
ax.set_title(f"{big}: soup cells at {'singlet' if h_med < 1.3 else 'doublet'} hashtag depth "
             f"(HTO x{h_med:.2f}, UMI x{u_med:.2f})", fontsize=8)
ax.legend(fontsize=6, markerscale=4, frameon=False)
fig.tight_layout(); fig.savefig(FIG_DIR / "nb31c_depth_vs_hto.png", dpi=200); plt.show()

# Figure 2 -- how much of a soup cell is soup
oth = np.isin(lanes_arr, [l for l in np.unique(lanes_arr) if l not in SOUP_LANES])
r_s, r_l, r_o = _s["rho"].clip(0, 3), o["rho"][inl & is_lymph].clip(0, 3), o["rho"][oth & is_lymph].clip(0, 3)
fig, ax = plt.subplots(figsize=(5, 3))
for v, nm, c in [(r_o, "lymphocytes, other lanes", "#999999"),
                 (r_l, f"lymphocytes, {big}", "#4C72B0"), (r_s, f"soup cells, {big}", "#8172B2")]:
    ax.hist(v, bins=60, range=(0, 3), density=True, histtype="step", lw=1.5, color=c, label=nm)
ax.axvline(1, ls="--", c="k", lw=0.8)
ax.set_xlabel("rho = myeloid share / soup myeloid share"); ax.set_ylabel("density")
ax.set_title(f"Soup cells: median rho {r_s.median():.2f} vs {r_l.median():.2f} in {big} "
             f"lymphocytes (>1 = more myeloid RNA than soup explains)", fontsize=7)
ax.legend(fontsize=6, frameon=False)
fig.tight_layout(); fig.savefig(FIG_DIR / "nb31c_rho.png", dpi=200); plt.show()
```

```
ADT reference medians (CLR) -- a test is usable only if pos and neg refs separate:
    lane  adt  med_pos_ref  med_neg_ref  usable
B4__PF48  CD3        -0.00        -0.00   False
B4__PF48  CD4         0.00         0.00   False
B4__PF48  CD8         0.00         0.00   False
B4__PF48 CD19        -0.00        -0.00   False
B4__SZ32  CD3         0.00         0.00   False
B4__SZ32  CD4         0.00         0.00   False
B4__SZ32  CD8         0.00         0.00   False
B4__SZ32 CD19          NaN          NaN   False
B4__SZ30  CD3         0.13        -0.60    True
B4__SZ30  CD4        -0.02        -0.02   False
B4__SZ30  CD8        -0.00        -0.00   False
B4__SZ30 CD19          NaN          NaN   False
B4__PF50  CD3        -0.00        -0.00   False
B4__PF50  CD4         0.00         0.00   False
B4__PF50  CD8         0.00         0.00   False
B4__PF50 CD19        -0.00        -0.00   False
B4__SZ33  CD3         0.00         0.00   False
B4__SZ33  CD4         0.00         0.00   False
B4__SZ33  CD8         0.00         0.00   False
B4__SZ33 CD19          NaN          NaN   False
B4__PF49  CD3        -0.00        -0.00   False
B4__PF49  CD4         0.00         0.00   False
B4__PF49  CD8         0.00         0.00   False
B4__PF49 CD19        -0.00        -0.00   False

CD3 protein state of soup cells, by RNA verdict:
adt_CD3_state        amb     na  neg  pos    All
lane     verdict                                
B4__PF48 ambient       0  13569    0    0  13569
         doublet       0    377    0    0    377
         myeloid       0     11    0    0     11
         unresolved    0   9569    0    0   9569
B4__PF49 ambient       0      2    0    0      2
         doublet       0      6    0    0      6
         unresolved    0     24    0    0     24
B4__PF50 ambient       0     14    0    0     14
         doublet       0     14    0    0     14
         unresolved    0     26    0    0     26
B4__SZ30 ambient       9      0   23   22     54
         doublet       3      0    4   38     45
         myeloid       2      0    0   10     12
         unresolved    7      0    7   22     36
B4__SZ32 ambient       0    336    0    0    336
         doublet       0    269    0    0    269
         myeloid       0     41    0    0     41
         unresolved    0   1196    0    0   1196
B4__SZ33 ambient       0     14    0    0     14
         doublet       0      1    0    0      1
         myeloid       0      1    0    0      1
         unresolved    0     29    0    0     29
All                   21  25499   34   92  25646


per residual cluster (31b leiden_resid):
                 n  top_lane  umi_ratio  hto_ratio   rho  knn_doublet  knn_lymph_amb  knn_mono  cd3_pos  cd3_neg  ambient  doublet  myeloid  unresolved
leiden_resid                                                                                                                                           
16            2768  B4__PF48       0.75       0.28  0.90         0.07           0.93      0.00     0.00     0.00     0.81     0.00     0.00        0.19
14            2745  B4__PF48       0.98       0.30  1.36         0.25           0.74      0.00     0.00     0.00     0.28     0.05     0.00        0.67
13            2193  B4__PF48       0.84       0.28  1.15         0.15           0.85      0.00     0.00     0.00     0.54     0.00     0.00        0.45
4             2060  B4__PF48       0.85       0.28  1.24         0.28           0.72      0.00     0.00     0.00     0.37     0.02     0.00        0.61
12            1657  B4__PF48       0.83       0.27  1.03         0.20           0.80      0.00     0.00     0.00     0.64     0.01     0.00        0.35
2             1601  B4__SZ32       0.92       3.64  1.04         0.29           0.63      0.07     0.05     0.01     0.06     0.20     0.03        0.71
15            1465  B4__PF48       0.77       0.27  1.02         0.06           0.94      0.00     0.00     0.00     0.71     0.00     0.00        0.29
18            1222  B4__PF48   
... [3398 more chars truncated]
```

## §3 — Rescue: a soup-robust classifier, accepted only where protein agrees

**Features.** 2,000 HVGs from clean geskin cells, minus the top soup genes (those carrying 50% of
soup mass in any soup lane), the myeloid set, and MT-/RPS/RPL genes. CP10K is computed over the
feature genes only, so soup mass outside them cannot rescale a cell.

**Model.** StandardScaler → PCA 50 → multinomial logistic regression. It is validated
leave-one-lane-out first.

**Acceptance requires all three:**
- the label fits the §2 verdict: `ambient` → a lymphoid label, `myeloid` → a non-lymphoid label
- prob ≥ 0.8
- the CD3 protein agrees: T → CD3+, everything else → CD3−

CD4 vs CD8 is taken from protein where it is unambiguous. `doublet`-verdict cells become `Doublet`.

```python  # cell 8
# ============================================================================
# §3a  Classifier + leave-one-lane-out
# ============================================================================
if FUTILE:
    print("FUTILE at §2 -- skipped")
else:
    classes = [c for c, n in o["ref"].value_counts().items() if n >= 50]
    tr = np.concatenate([rng.choice(np.flatnonzero(o["ref"].eq(c).to_numpy()),
                                    size=min(N_TRAIN_PER_CLASS, int(o["ref"].eq(c).sum())),
                                    replace=False) for c in classes])
    drop = set(mye) | {g for g in G if g.startswith(("MT-", "RPS", "RPL"))}
    for lane in SOUP_LANES:
        s = soup_p[lane].sort_values(ascending=False)
        drop |= set(s.index[: int(np.searchsorted(s.cumsum().to_numpy(), SOUP_MASS_DROP)) + 1])
    hv = ad.AnnData(lognorm(Xc[tr]), var=pd.DataFrame(index=G))
    sc.pp.highly_variable_genes(hv, n_top_genes=N_HVG_CLF, flavor="seurat")
    feat = (hv.var.loc[~hv.var.index.isin(drop), "dispersions_norm"]
            .sort_values(ascending=False).index[:N_HVG_CLF])
    fi = G.get_indexer(feat)
    print(f"classes: {classes}\ntrain cells: {len(tr):,} | soup/myeloid/MT/ribo genes dropped: "
          f"{len(drop):,} | features: {len(feat):,}")
    for g in ["CD3E", "CD3D", "MS4A1", "NKG7", "GNLY", "CD8A", "CD4", "LYZ", "CD74", "HLA-DRA"]:
        print(f"  {g:8s} {'kept' if g in set(feat) else 'not a feature'}")

    def featurize(ids):
        return lognorm(Xc[ids][:, fi]).toarray()

    def make_clf():
        return make_pipeline(StandardScaler(), PCA(N_PC, random_state=SEED),
                             LogisticRegression(C=0.5, max_iter=2000))

    Ftr, ytr, ltr = featurize(tr), ref_arr[tr].astype(str), lanes_arr[tr]
    lolo = []
    for lane in np.unique(lanes_arr):
        mtr = ltr != lane
        ev = np.concatenate([rng.choice(ix, size=min(3000, len(ix)), replace=False)
                             for c in classes
                             if len(ix := np.flatnonzero((lanes_arr == lane) & (ref_arr == c)))])
        clf = make_clf().fit(Ftr[mtr], ytr[mtr])
        p, y = clf.predict(featurize(ev)), ref_arr[ev].astype(str)
        coarse = lambda z: np.where(np.isin(z, ["T_CD4", "T_CD8"]), "T", z)
        for c in np.unique(y):
            mm = y == c
            lolo.append({"lane": lane, "class": c, "n": int(mm.sum()),
                         "recall": float((p[mm] == c).mean()),
                         "recall_coarse": float((coarse(p[mm]) == coarse(np.array([c]))[0]).mean())})
    lolo = pd.DataFrame(lolo)
    print("\nleave-one-lane-out recall (fine):")
    print(lolo.pivot(index="class", columns="lane", values="recall").round(2).to_string())
    print("\nleave-one-lane-out recall (T_CD4/T_CD8 merged):")
    print(lolo.pivot(index="class", columns="lane", values="recall_coarse").round(2).to_string())
    _w = lolo.assign(w=lolo["n"] * lolo["recall_coarse"]).groupby("lane")[["w", "n"]].sum()
    print("\noverall coarse accuracy per held-out lane:", (_w["w"] / _w["n"]).round(3).to_dict())
    lolo.to_csv(TAB_DIR / "nb31c_lolo_recall.csv", index=False)

    clf = make_clf().fit(Ftr, ytr)
    ids = o.index.get_indexer(soupdf.index)
    Pr = np.vstack([clf.predict_proba(featurize(ids[s0:s0 + 5000]))
                    for s0 in range(0, len(ids), 5000)])
    soupdf["pred"] = clf.classes_[Pr.argmax(1)]
    soupdf["prob"] = Pr.max(1)
    print("\nsoup cells: RNA verdict x predicted class")
    print(pd.crosstab(soupdf["verdict"], soupdf["pred"], margins=True).to_string())
    print(f"\n[{(time.time() - T0) / 60:.1f} min]")
```

```
classes: ['T_CD4', 'T_CD8', 'Erythroid', 'B', 'Mono_classical', 'NK', 'HSPC', 'pDC', 'Plasma']
train cells: 31,328 | soup/myeloid/MT/ribo genes dropped: 400 | features: 2,000
  CD3E     not a feature
  CD3D     not a feature
  MS4A1    kept
  NKG7     not a feature
  GNLY     not a feature
  CD8A     kept
  CD4      not a feature
  LYZ      not a feature
  CD74     not a feature
  HLA-DRA  not a feature


leave-one-lane-out recall (fine):
lane            B4__PF46  B4__PF47  B4__PF48  B4__PF49  B4__PF50  B4__SZ30  B4__SZ32  B4__SZ33  B4__SZ38  B4__SZ39  B4__SZ5
class                                                                                                                      
B                   0.99      1.00      0.98      0.99      0.99      0.99      0.98      0.97      0.96      0.95     0.99
Erythroid           0.90      0.95      0.67      1.00      0.89      0.96      0.98      0.88      0.81      0.92     0.96
HSPC                0.97      0.97      1.00      0.97      0.98      0.92      0.98      1.00      1.00      0.99     1.00
Mono_classical      0.96      0.95      0.99      1.00      0.99      0.67      0.88      1.00      1.00      0.89     0.97
NK                  0.20      0.36      0.13      0.11      0.29      0.25      0.00      0.00      0.04      0.00     0.05
Plasma              0.25      0.00      0.00      0.15      0.00      0.25      0.00       NaN       NaN      0.13     0.20
T_CD4               0.74      0.66      0.71      0.75      0.60      0.87      0.71      0.95      0.98      0.52     0.93
T_CD8               0.83      0.81      0.77      0.62      0.84      0.94      0.85      0.76      0.63      0.79     0.94
pDC                 0.78      1.00      1.00      1.00      0.90       NaN      1.00       NaN       NaN      1.00     1.00

leave-one-lane-out recall (T_CD4/T_CD8 merged):
lane            B4__PF46  B4__PF47  B4__PF48  B4__PF49  B4__PF50  B4__SZ30  B4__SZ32  B4__SZ33  B4__SZ38  B4__SZ39  B4__SZ5
class                                                                                                                      
B                   0.99      1.00      0.98      0.99      0.99      0.99      0.98      0.97      0.96      0.95     0.99
Erythroid           0.90      0.95      0.67      1.00      0.89      0.96      0.98      0.88      0.81      0.92     0.96
HSPC                0.97      0.97      1.00      0.97      0.98      0.92      0.98      1.00      1.00      0.99     1.00
Mono_classical      0.96      0.95      0.99      1.00      0.99      0.67      0.88      1.00      1.00      0.89     0.97
NK                  0.20      0.36      0.13      0.11      0.29      0.25      0.00      0.00      0.04      0.00     0.05
Plasma              0.25      0.00      0.00      0.15      0.00      0.25      0.00       NaN       NaN      0.13     0.20
T_CD4               0.97      1.00      0.99      0.98      0.95      0.97      0.96      0.95      0.99      0.87     0.99
T_CD8               0.99      0.99      0.99      0.98      0.98      0.98      0.97      0.92      0.98      0.81     1.00
pDC                 0.78      1.00      1.00      1.00      0.90       NaN      1.00       NaN       NaN      1.00     1.00

overall coarse accuracy per held-out lane: {'B4__PF46': 0.977, 'B4__PF47': 0.966, 'B4__PF48': 0.979, 'B4__PF49': 0.983, 'B4__PF50': 0.966, 'B4__SZ30': 0.977, 'B4__SZ32': 0.971, 'B4__SZ33': 0.952, 'B4__SZ38': 0.931, 'B4__SZ39': 0.866, 'B4__SZ5': 0.978}


soup cells: RNA verdict x predicted class
pred          B  Erythroid  HSPC  Mono_classical     NK  Plasma  T_CD4  T_CD8  pDC    All
verdict                                                                                  
ambient     506        150    29            1693   5901       0    823   4883    4  13989
doublet       4         14     2             457    141       0     28     65    1    712
myeloid       2          7     0              53      2       0      0      1    0     65
unresolved  245        294    27 
... [157 more chars truncated]
```

```python  # cell 9
# ============================================================================
# §3b  Acceptance, protein concordance, composition check, marker blocks
# ============================================================================
FUTILE_RESCUE = True
if not FUTILE:
    T_pred = soupdf["pred"].isin(["T_CD4", "T_CD8"])
    cd3 = soupdf["adt_CD3_state"]
    conc = (T_pred & cd3.eq("pos")) | (~T_pred & cd3.eq("neg"))
    fits = ((soupdf["verdict"].eq("ambient") & soupdf["pred"].isin(LYMPH))
            | (soupdf["verdict"].eq("myeloid") & ~soupdf["pred"].isin(LYMPH)))
    conf = soupdf["prob"] >= P_ACCEPT
    acc = fits & conf & conc
    lab = soupdf["pred"].copy()
    lab[T_pred & soupdf["adt_CD4_state"].eq("pos") & soupdf["adt_CD8_state"].eq("neg")] = "T_CD4"
    lab[T_pred & soupdf["adt_CD8_state"].eq("pos") & soupdf["adt_CD4_state"].eq("neg")] = "T_CD8"
    soupdf["rescue_label"] = np.where(soupdf["verdict"].eq("doublet"), "Doublet",
                                      np.where(acc, lab, ""))

    print("acceptance funnel (non-doublet soup cells):")
    nd = ~soupdf["verdict"].eq("doublet")
    print(f"  {int(nd.sum()):,} cells -> verdict fits pred {int((nd & fits).sum()):,} -> "
          f"prob>={P_ACCEPT} {int((nd & fits & conf).sum()):,} -> CD3 concordant "
          f"{int((nd & acc).sum()):,}")
    print("\nprotein concordance BEFORE the filter (the validation): CD3 state by predicted class,"
          " confident fits only")
    print(pd.crosstab(soupdf.loc[fits & conf, "pred"], cd3[fits & conf], normalize="index")
          .round(3).to_string())
    if "adt_CD19_state" in soupdf:
        _b = fits & conf & soupdf["pred"].eq("B")
        print("\nCD19 protein on confident B predictions:",
              soupdf.loc[_b, "adt_CD19_state"].value_counts().to_dict())
    _t = acc & T_pred
    if _t.any():
        print("\nRNA CD4/CD8 call vs protein CD4/CD8 state on accepted T:")
        print(pd.crosstab(soupdf.loc[_t, "pred"],
                          soupdf.loc[_t, "adt_CD4_state"] + "/" + soupdf.loc[_t, "adt_CD8_state"])
              .to_string())

    frac_lab = float(soupdf["rescue_label"].ne("").mean())
    FUTILE_RESCUE = frac_lab < GATE_LABELLED
    print("\n" + "=" * 100)
    print(f"labelled {frac_lab:.1%} of {len(soupdf):,} soup cells (need >= {GATE_LABELLED:.0%})"
          + ("  -> FUTILE, nothing written" if FUTILE_RESCUE else ""))
    print(pd.crosstab(soupdf["lane"], soupdf["rescue_label"].replace("", "(stays Ambient_soup)"),
                      margins=True).to_string())
    print("=" * 100)

    # Composition per lane before / after. PF48 should move toward PF46/47/49/50.
    def coarse_of(fine_t, fine_nt):
        m = {"CD4": "T", "CD8": "T", "NK": "NK", "T_CD4": "T", "T_CD8": "T", "B_naive": "B",
             "B_memory": "B", "B": "B", "Mono_classical": "Mono", "Mono_nonclassical": "Mono",
             "Ambient_soup": "soup", "Doublet": "Doublet", "LowQC": "LowQC/Unk",
             "Unk": "LowQC/Unk", "Erythroid": "Erythroid"}
        lab_ = fine_nt.where(fine_nt.notna(), fine_t)
        return lab_.map(m).fillna("other")
    before = coarse_of(o["cell_type_final"], o["subtype_fine"].where(~_r3.reindex(o.index, fill_value=False), "Ambient_soup"))
    after = before.copy()
    _rl = soupdf["rescue_label"][soupdf["rescue_label"].ne("")]
    after.loc[_rl.index] = coarse_of(pd.Series(np.nan, index=_rl.index), _rl).to_numpy()
    comp = pd.concat({"before": pd.crosstab(lanes_arr, before.to_numpy(), normalize="index"),
                      "after": pd.crosstab(lanes_arr, after.to_numpy(), normalize="index")}, axis=1)
    print("\nlane composition (fraction of atlas cells):")
    print(comp.round(3).to_string())
    comp.to_csv(TAB_DIR / "nb31c_lane_composition.csv")

    # Marker blocks: rescued groups after subtracting the soup (lymphoid labels only) vs the
    # same lane's clean reference cells.
    PANEL = {"T": ["CD3E", "CD3D", "TRAC", "IL7R", "CD4", "CD8A", "CD8B"],
             "NK": ["NKG7", "GNLY", "KLRD1", "PRF1", "FCGR3A"],
             "B": ["MS4A1", "CD79A", "CD79B", "BANK1"],
             "mono": ["LYZ", "S100A8", "S100A9", "CD14", "FCN1", "CSF1R"],
             "DC": ["CD1C", "FCER1A", "CLEC10A", "HLA-DRA", "CD74"],
             "soup": ["HBB", "PPBP", "MALAT1"]}
    pg = [g for gs in PANEL.values() for g in gs if g in G]
    pgi = G.get_indexer(pg)
    sh = soupdf.index[soupdf["rescue_label"].ne("") | soupdf["verdict"].eq("unresolved")]
    grp_s = np.where(soupdf.loc[sh, "rescue_label"].ne(""),
                     "rescued:" + soupdf.loc[sh, "rescue_label"], "left:Ambient_soup")
    ref_ids = np.concatenate([rng.choice(ix, size=min(500, len(ix)), replace=False)
                              for c in LYMPH + MYELO
                              if len(ix := np.flatnonzero((lanes_arr == big) & (ref_arr == c)))])
    ii = np.concatenate([o.index.get_indexer(sh), ref_ids])
    grp = np.concatenate([grp_s, "ref:" + ref_arr[ref_ids].astype(str)])
    Cp = Xc[ii][:, pgi].toarray()
    u = umi_all[ii].astype(float)
    rho_c = np.clip(o["rho"].to_numpy()[ii], 0, 0.95)
    lymphoid = np.array([g.split(":")[-1] in LYMPH for g in grp])
    rho_c[~lymphoid] = 0
    sp_lane = soup_p.loc[pg].to_numpy()[:, [soup_p.columns.get_loc(l) for l in lanes_arr[ii]]].T
    Cd = np.clip(Cp - rho_c[:, None] * u[:, None] * sp_lane, 0, None)
    Xd = np.log1p(Cd / (u * (1 - rho_c))[:, None] * 1e4)
    blk = ad.AnnData(Xd.astype(np.float32), var=pd.DataFrame(index=pg),
                     obs=o.iloc[ii][["donor", "study", "n_genes", "pct_mito", "doublet_score"]]
                     .assign(grp=grp).set_axis(np.arange(len(ii)).astype(str)))
    mean_b, frac_b = A.cluster_expression(blk, "grp", pg)
    qc_b = A.cluster_qc(blk, "grp", donor_key="donor", study_key="study")
    print("\n" + A.paste_block(blk, "grp", PANEL, mean_b, frac_b, qc_b,
                               vocab=sorted(set(grp)),
                               title=f"rescued soup (soup-subtracted for lymphoid labels) vs "
                                     f"{big} clean reference"))

    # Figure 3 -- the protein view of the outcome
    _s = soupdf[soupdf["lane"] == big]
    yk = "adt_CD19" if "adt_CD19" in _s and _s["adt_CD19"].notna().any() else "adt_CD4"
    fig, ax = plt.subplots(figsize=(5, 3))
    top = _s["rescue_label"].replace("", "unresolved").value_counts().index[:7]
    for nm in top:
        mm = _s["rescue_label"].replace("", "unresolved").eq(nm).to_numpy()
        k = rng.choice(np.flatnonzero(mm), size=min(2000, int(mm.sum())), replace=False)
        ax.scatter(_s["adt_CD3"].to_numpy()[k], _s[yk].to_numpy()[k], s=2, alpha=0.4,
                   label=nm, c="#BBBBBB" if nm == "unresolved" else None, rasterized=True)
    ax.set_xlabel("CD3 ADT (CLR)"); ax.set_ylabel(f"{yk[4:]} ADT (CLR)")
    fl = _s["rescue_label"].ne("").mean()
    ax.set_title(f"{big}: {fl:.0%} of soup cells get a protein-concordant label", fontsize=8)
    ax.legend(fontsize=6, markerscale=4, frameon=False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "nb31c_adt_rescue.png", dpi=200); plt.show()
    print(f"\n[{(time.time() - T0) / 60:.1f} min]")
```

```
acceptance funnel (non-doublet soup cells):
  24,934 cells -> verdict fits pred 12,173 -> prob>=0.8 4,345 -> CD3 concordant 14

protein concordance BEFORE the filter (the validation): CD3 state by predicted class, confident fits only
adt_CD3_state     amb     na    neg    pos
pred                                      
B               0.000  1.000  0.000  0.000
Erythroid       0.000  0.500  0.000  0.500
Mono_classical  0.022  0.870  0.000  0.109
NK              0.000  1.000  0.000  0.000
T_CD4           0.000  0.913  0.087  0.000
T_CD8           0.001  0.987  0.002  0.011

CD19 protein on confident B predictions: {'na': 268}

RNA CD4/CD8 call vs protein CD4/CD8 state on accepted T:
col_0  na/na
pred        
T_CD8     14

====================================================================================================
labelled 2.8% of 25,646 soup cells (need >= 30%)  -> FUTILE, nothing written
rescue_label  (stays Ambient_soup)  Doublet  T_CD8    All
lane                                                     
B4__PF48                     23149      377      0  23526
B4__PF49                        26        6      0     32
B4__PF50                        40       14      0     54
B4__SZ30                        88       45     14    147
B4__SZ32                      1573      269      0   1842
B4__SZ33                        44        1      0     45
All                          24920      712     14  25646
====================================================================================================


lane composition (fraction of atlas cells):
         before                                                                 after                                                               
col_0         B Doublet Erythroid LowQC/Unk   Mono     NK      T  other   soup      B Doublet Erythroid LowQC/Unk   Mono     NK      T  other   soup
row_0                                                                                                                                               
B4__PF46  0.106   0.003     0.002     0.038  0.207  0.000  0.642  0.002  0.000  0.106   0.003     0.002     0.038  0.207  0.000  0.642  0.002  0.000
B4__PF47  0.072   0.086     0.034     0.035  0.002  0.015  0.753  0.003  0.001  0.072   0.086     0.034     0.035  0.002  0.015  0.753  0.003  0.001
B4__PF48  0.015   0.044     0.000     0.038  0.036  0.002  0.278  0.002  0.584  0.015   0.054     0.000     0.038  0.036  0.002  0.278  0.002  0.574
B4__PF49  0.085   0.072     0.000     0.042  0.063  0.000  0.734  0.004  0.001  0.085   0.072     0.000     0.042  0.063  0.000  0.734  0.004  0.001
B4__PF50  0.052   0.069     0.002     0.088  0.033  0.002  0.748  0.004  0.002  0.052   0.070     0.002     0.088  0.033  0.002  0.748  0.004  0.001
B4__SZ30  0.030   0.003     0.011     0.242  0.000  0.000  0.702  0.001  0.011  0.030   0.006     0.011     0.242  0.000  0.000  0.703  0.001  0.006
B4__SZ32  0.021   0.001     0.184     0.213  0.000  0.000  0.484  0.010  0.086  0.021   0.014     0.184     0.213  0.000  0.000  0.484  0.010  0.074
B4__SZ33  0.026   0.000     0.001     0.476  0.000  0.000  0.493  0.000  0.003  0.026   0.000     0.001     0.476  0.000  0.000  0.493  0.000  0.003
B4__SZ38  0.002   0.008     0.045     0.005  0.002  0.002  0.936  0.000  0.000  0.002   0.008     0.045     0.005  0.002  0.002  0.936  0.000  0.000
B4__SZ39  0.009   0.165     0.324     0.018  0.001  0.000  0.476  0.006  0.001  0.009   0.165     0.324     0.018  0.001  0.000  0.476  0.006  0.001
B4__SZ5   0.018   0.009     0.004     0.026  0.032  0.002  0.908  0.002  0.000  0.018   0.009     0.004     0.026  0.032  0.002  0.908  0.002  0.000


====================================================================================================
rescued soup (soup-subtracted for lymphoid labels) vs B4__PF48 clean reference — 13,704 cells, 9 clusters; values are `mean log1p CP10K / fraction detected`
allowed labels: ['left:Ambient_soup', 'ref:B', 'ref:Mono_classical', 'ref:Mon
... [6952 more chars truncated]
```

```python  # cell 10
# ============================================================================
# §4  Write the rescue table (31b §5d folds it into blood_subtypes.csv)
# ============================================================================
if FUTILE or FUTILE_RESCUE:
    print(f"not writing {RESCUE_CSV.name}: the soup stays Ambient_soup "
          f"(evidence in {TAB_DIR}/nb31c_*)")
    if RESCUE_CSV.exists():
        print(f"!! a {RESCUE_CSV.name} from an earlier run exists and 31b §5d will still read it"
              " -- delete it if this verdict stands")
else:
    cols = ["lane", "leiden_resid", "verdict", "rescue_label", "pred", "prob", "rho",
            "knn_lymph", "knn_ambient", "knn_doublet", "knn_mono", "umi_ratio", "hto_ratio",
            "hto_r2"] + [c for c in soupdf.columns if c.startswith("adt_")]
    frame = soupdf[cols].rename_axis("cell_id").reset_index()
    prov = {
        "notebook": "30_blood/31c_ambient_rescue.ipynb",
        "inputs": {"raw_h5": str(B4_DIR / "raw"), "subtypes": str(SUBTYPE_CSV),
                   "broad": str(BROAD_CSV), "obs": str(OBS_PQ)},
        "soup_lanes": SOUP_LANES, "soup_profile": soup_meta,
        "gate": {"frac_resolved": frac_resolved, "frac_adt": frac_adt,
                 "frac_labelled": frac_lab},
        "params": {"N_SYN": N_SYN, "AMB_F": AMB_F, "K_NN": K_NN, "P_ACCEPT": P_ACCEPT,
                   "SOUP_MASS_DROP": SOUP_MASS_DROP, "N_HVG_CLF": N_HVG_CLF, "seed": SEED},
        "lolo_coarse_acc": (_w["w"] / _w["n"]).round(4).to_dict(),
        "rescue_counts": frame["rescue_label"].replace("", "Ambient_soup").value_counts().to_dict(),
    }
    A.write_sidecar(frame, RESCUE_CSV, RESCUE_PROV, prov, expect_rows=len(frame))
print(f"[{(time.time() - T0) / 60:.1f} min total]")
```

```
not writing blood_ambient_rescue.csv: the soup stays Ambient_soup (evidence in /home/projects/nyosef/zvise/MF/tables/nb31c_*)
[3.9 min total]
```

## Verdict (run 2026-10-22): futile, so the soup stays `Ambient_soup`

- **Not doublets.** PF48 soup cells have 0.84× singlet UMI. 31b's own `Doublet` calls have
  2.6× UMI and 3.7× HTO, and the same test calls 77% of them `doublet`.
- **Background-only hashing.** In every arm, soup cells carry a flat ~340–500 HTO counts with no
  dominant tag: the 2nd/1st ratio is 0.48–0.69, against 0.04–0.39 for clean lymphocytes. HashSolo
  pushed them mostly onto HTO2 (15.6k) and HTO1 (5.4k), whose clean cells carry 1,000–1,400. The
  RNA matches PF48's HashSolo-negative barcodes (ρ ≈ 1 against that profile). These are droplets
  that should have been called Negative, so their arm assignment cannot be trusted.
- **No independent identity evidence.**
  - All 20 ADTs are zero in every lane except SZ30.
  - The PF lanes have no TCR.
- **RNA alone does not hold.**
  - Only 4,345 / 24,934 cells (17%) get a confident RNA call that agrees with the §2 verdict.
  - In SZ32, where TCR exists, the cells RNA called lymphocyte + ambient carry a TCR only 8% of the
    time. Lane CD4 T cells: 77%; CD8: 52%.
- **The geskin `NK` reference is unusable.** It has MALAT1 0.64 (≈5 elsewhere) and 6% mito, and it
  pulled in 10k soup predictions.

No `blood_ambient_rescue.csv` was written and 31b is unchanged. If these cells need a label that
says what they are, it is a hashing-QC label (e.g. `LowQC`), not a cell type.
