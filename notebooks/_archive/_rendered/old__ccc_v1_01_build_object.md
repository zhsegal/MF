# old/ccc_v1_01_build_object.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/ccc_v1_01_build_object.ipynb` and is never run._

# 34 · CCC — build the skin ligand–receptor object

First notebook of the MF/CTCL **cell–cell communication** analysis (34 build · 35 descriptive ·
36 headline figures). Mirrors the working pipeline at `MyelomaProject/ccc/` (nb01/02/05), with
the mouse ortholog machinery dropped and a streaming build in its place.

**Question this phase answers**: what does malignant CD4 signal to, and receive from, the skin
TME (CD8, myeloid, fibroblast, keratinocyte, vascular, B, plasma, Tregs)?

## Sources

| | |
|---|---|
| Expression | `data/atlas_joint/joint_annotated.h5ad` — 1,173,694 × **40,821** gene symbols, `X` ≡ `layers['raw_counts']` (float64 CSR) |
| Labels / metadata | `data/atlas_joint/atlas_obs_full.parquet` — the **only** place `cell_type_final`, `mal_tcr_alice` and `stage_group` exist; the h5ad obs does not carry them |
| Resource | `li.rs.select_resource("consensus")` — human, **no** ortholog translation |

**Not** `joint_mrvi_input_skin.h5ad`: the 10k-HVG object drops 9 LR genes (`MIF`, `PVR`,
`ICOSLG`, `CD44`, `LTBR`, `ITGB1`, `NOTCH1`, `IL4R`, `IL7`).

## The malignancy call is ALICE-TCR alone

`MALIG_SRC = "mal_tcr_alice"` — nb30 cell 19: the per-donor founder TRB clonotype plus its
ALICE-significant ≤1-aa CDR3 family (OLGA Pgen model), CD4 only.

**Not** `mal_combined`. That is `alice | cnv_malig_cluster`, and half of the malignant CD4 it
produced (77,624 of 153,973) were `cnv_only` — called from inferCNV's *smoothed regional
expression*, which is partly the same measurement LIANA then scores as ligand and receptor
abundance. ALICE is a TCR-sequence fact, so the call and the scored expression are independent.
`mal_combined` and `mal_cnv` remain as sensitivity arms in nb35 §13.

A CD4 cell with **no TRB CDR3** is ALICE-negative by absence of data, not by evidence, so it is
demoted to `CD4_unassessed` rather than diluting the reactive comparator. That gate
(`tcr_assessed_src=assessed_tcr_nb30`) applies to the primary only — `mal_cnv`'s missingness is
CNV-set membership and `tumor_cell`'s is li2024 membership, so gating those on CDR3 recovery
would mix unrelated mechanisms.

⚠️ `assessed_tcr_nb30`, not `assessed_tcr`: the latter is the nb10 10x-VDJ clone table, where
li2024 — the largest cohort — contributes **0** cells.

## Why the built object is only ~1,832 genes

`rank_aggregate` subsets to resource genes internally, so keeping only the 1,832 consensus
subunit genes present in the atlas is **lossless** — provided the library-size factor is summed
over all 40,821 genes **before** the columns are subset. Doing it the other way round rescales
every cell by the ratio of resource-gene counts to total counts (median **6.7×** here) and
produces plausible, wrong `lr_means`. §5 gates on exactly this by re-running `rank_aggregate`
over a 3-donor **full-gene** object and asserting the statistics match.

⚠️ `ccc_skin.h5ad` is a **CCC-only** object. Its var is the LR resource subset — it is not
valid for HVG selection, DE, or UMAP.

## Facts every downstream number must be read against

- **29 donors** (of 82) carry ≥25 ALICE-malignant CD4; the partner axes run at 17–29 donors.
- **190,142 CD4 are unassessed** and pool two protocol artifacts, never biology: 130,484 outside
  nb30's scored set (100 % `skin_layer == "whole"` — li2024 66,234, chennareddy2025 51,856,
  gaydosik2019 11,900, brunner2024 494) plus 59,658 inside it with no TRB CDR3 recovered. Own
  level, never pooled with malignant or reactive.
- **HC skin has no usable T cells** (1,524 CD4 / 300 CD8 across 9 donors), so there is no
  normal-skin baseline. `CD4_reactive` is the comparator, and it is *lesional* reactive CD4.
- ALICE calls **no CD8 and no Treg** malignant — it is a CD4-only caller. Both stay single
  levels. (`mal_combined` called 21,200 CD8 and 792 Tregs malignant, all `cnv_only` with zero
  TCR support; §1 quantifies this.)
- **40,447 of the 72,207 reactive CD4 carry a `cnv_only` call.** Under an ALICE-primary design
  that disagreement is deliberate, not an error — nb35 §13 sweeps it.

**Sections**: §0 setup · §1 obs-only audit (no h5ad opened) · §2 resource + panel coverage ·
§3 submit the build job · §4 load + invariants · §5 the equivalence gate.

```python  # cell 1
import warnings; warnings.filterwarnings("ignore")
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc
import matplotlib as mpl, matplotlib.pyplot as plt
import liana as li


def _resolve_nb_dir() -> Path:
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("notebooks/MF"), Path("scvi-tools-neural-nmf/notebooks/MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data from {start}")


NB_DIR = _resolve_nb_dir(); sys.path.insert(0, str(NB_DIR))
import ccc_data as cd
import ccc_helpers as C

sc.settings.verbosity = 1
mpl.rcParams["figure.dpi"] = 110
mpl.rcParams["savefig.bbox"] = "tight"
cd.TAB_DIR.mkdir(parents=True, exist_ok=True); cd.FIG_DIR.mkdir(parents=True, exist_ok=True)
print("NB_DIR =", NB_DIR)
print("liana", li.__version__, "| scanpy", sc.__version__)
```

## §1 · Obs-only audit — **no h5ad is opened here**

Everything in this section reads the 39 MB parquet. It decides which levels exist and which
axes may carry a claim *before* any expression is touched, so the build is not repeated because
a level turned out to be unusable.

```python  # cell 3
obs = C.load_ccc_obs()                      # 749,510 skin cells
obs = obs.set_index("cell_id", drop=False)

# donor vs real_donor: the paired designs and any donor-level inference must use the true
# biological individual. If these ever disagree, DONOR_KEY has to become real_donor.
dr = pd.crosstab(obs["donor"].astype(str), obs["real_donor"].astype(str))
n_multi = int((dr > 0).sum(axis=1).gt(1).sum())
print(f"donor levels {dr.shape[0]} | real_donor levels {dr.shape[1]} | "
      f"donors mapping to >1 real_donor: {n_multi}")
assert n_multi == 0, "donor and real_donor disagree -- switch cd.DONOR_KEY to 'real_donor'"
print("DONOR_KEY =", cd.DONOR_KEY)
```

```python  # cell 4
# how the primary (ALICE) call lands on each cell type. Only CD4 is ever True --
# ALICE is a CD4-only caller, so the CD8/Treg malignant levels simply do not exist here.
ct_mal = pd.crosstab(obs[cd.CELLTYPE_SRC].astype(str), obs[cd.MALIG_SRC], dropna=False)
print(f"primary call: {cd.MALIG_SRC}")
display(ct_mal)

ev = pd.crosstab(obs[cd.CELLTYPE_SRC].astype(str), obs[cd.EVIDENCE_SRC].astype(str))
display(ev)
print("\nWhat the old mal_combined primary would have added, by evidence class -- every one of")
print("these is cnv_only, i.e. zero TCR support, so under ALICE they carry no malignant call:")
print(pd.crosstab(obs[cd.CELLTYPE_SRC].astype(str), obs[cd.EVIDENCE_SRC].astype(str))
      .loc[["CD8", "Tregs"], ["cnv_only"]].to_string())

# the CDR3 gate, restricted to CD4: this is what separates reactive from unassessed
cd4 = obs[obs[cd.CELLTYPE_SRC].astype(str) == "CD4"]
gate = pd.crosstab(cd4[cd.TCR_ASSESSED_SRC].fillna(False).astype(bool),
                   cd4[cd.MALIG_SRC], dropna=False)
gate.index.name = f"{cd.TCR_ASSESSED_SRC} (ALICE could test the cell)"
print("\nCD4 only -- the ALICE-negative row splits on whether a TRB CDR3 was recovered.")
print("False & False is negative-by-absence-of-data; it goes to CD4_unassessed, not reactive:")
display(gate)
```

```python  # cell 5
label = C.build_ccc_celltype(obs)
obs[cd.GROUPBY] = label
o = obs[label.notna()].copy()
print(f"\n{len(o)} cells carry a CCC label ({len(obs) - len(o)} dropped: {cd.DROP_LEVELS})")
```

```python  # cell 6
# per-donor and per-sample cell counts, and what MIN_CELLS removes
cnt_donor, sum_donor = C.cell_count_audit(o, sample_key=cd.DONOR_KEY)
cnt_samp, sum_samp = C.cell_count_audit(o, sample_key=cd.SAMPLE_KEY)

print(f"per donor (n={cnt_donor.shape[1]}), MIN_CELLS={cd.MIN_CELLS}:")
display(sum_donor)
cnt_donor.to_csv(cd.TAB_DIR / "ccc_cell_counts_by_donor.csv")
cnt_samp.to_csv(cd.TAB_DIR / "ccc_cell_counts_by_sample.csv")
sum_donor.to_csv(cd.TAB_DIR / "ccc_cell_count_audit_donor.csv")
sum_samp.to_csv(cd.TAB_DIR / "ccc_cell_count_audit_sample.csv")
```

```python  # cell 7
fig, ax = plt.subplots(figsize=(5, 3.2))
s = sum_donor["frac_units_passing"].sort_values()
ax.barh(s.index, s.values, color="0.4")
ax.axvline(cd.MIN_SAMPLES / cnt_donor.shape[1], ls="--", c="crimson", lw=1)
ax.set_xlabel(f"fraction of {cnt_donor.shape[1]} donors with >= {cd.MIN_CELLS} cells")
ax.set_title("Mast and Tregs are the levels that limit what is claimable", fontsize=9)
fig.savefig(cd.FIG_DIR / "ccc_level_donor_coverage.png", dpi=150)
```

```python  # cell 8
# which axes may carry a claim. This is the gate every figure in nb36 obeys.
feas = C.axis_feasibility(cnt_donor)
feas.to_csv(cd.TAB_DIR / "ccc_axis_feasibility.csv", index=False)
display(feas)
print("\nnot claimable:")
print(feas[feas.verdict != "claim"].to_string(index=False))
```

```python  # cell 9
# ALICE is the primary; the other three definitions are only moderately concordant with it,
# which is why nb35 §13 runs a sensitivity pass rather than trusting any single call
audit = C.malignancy_definition_audit(obs)
audit.to_csv(cd.TAB_DIR / "ccc_malignancy_definition_audit.csv", index=False)

# mal_combined is a strict superset of ALICE by construction (it is `alice | cnv`), so the
# interesting number is the other direction: how much of it ALICE declines to call.
cd4 = obs[obs[cd.CELLTYPE_SRC].astype(str) == "CD4"]
a, c = cd4["mal_tcr_alice"].fillna(False).astype(bool), cd4["mal_combined"].fillna(False).astype(bool)
print(f"\nCD4 called malignant by mal_combined but NOT by ALICE: {int((c & ~a).sum())}")
print("all of them cnv_only:", bool((cd4.loc[c & ~a, cd.EVIDENCE_SRC].astype(str) == "cnv_only").all()))
print("ALICE-only calls mal_combined also carries:", int((a & ~c).sum()), "(must be 0)")
```

```python  # cell 10
# study confounding: printed so nobody later reads a stage or disease effect off this atlas.
mal_donors = o[o[cd.GROUPBY].astype(str) == cd.CD4_MALIGNANT]
per_donor = mal_donors.groupby(cd.DONOR_KEY, observed=True).size()
keep = per_donor[per_donor >= cd.MIN_CELLS].index
d = mal_donors[mal_donors[cd.DONOR_KEY].isin(keep)].drop_duplicates(cd.DONOR_KEY)
print(f"{len(d)} donors carry >= {cd.MIN_CELLS} malignant CD4\n")
for col in ["study", "stage_group", "disease", "skin_layer"]:
    print(f"--- {col} ---"); print(pd.crosstab(d["study"], d[col]).to_string(), "\n")
pd.crosstab(d["study"], d["stage_group"]).to_csv(cd.TAB_DIR / "ccc_study_confounding.csv")
for k, v in cd.FORBIDDEN_CONTRASTS.items():
    print(f"FORBIDDEN {k}: {v}")
```

## §2 · Resource and curated-panel coverage

Two distinct failures that look identical in a result table, so they get separate columns:
a pair **absent from the resource** can never be scored however well expressed; a pair
**absent from the data** is a real negative.

The consensus resource also does not spell interactions the way people write them — subunits
are sorted alphabetically, some ligands are complexes (`CSF1 → CSF1R` is stored
`CSF1_IL34 → CSF1R`), some receptors carry obligate extra chains (`IL7R → IL2RG_IL7R`), and a
few pairs are stored with the receptor in the ligand column (`TIGIT → PVR`).
`C.resolve_pairs` translates the readable spellings in `ccc_data.LR_PANEL` onto the resource's,
so a real interaction is never reported as absent because of a naming convention.

```python  # cell 12
var = C.read_source_var()                   # h5py header read only; no expression touched
print("source var:", var.shape, "| index e.g.", var.index[:4].tolist())
assert var.index.is_unique, "source var index is not unique"

resource, coverage = C.load_resource(var_names=var.index)
genes = C.resource_genes(resource)
keep_gene_mask = np.asarray(var.index.isin(genes))
print(f"\nCCC object will keep {keep_gene_mask.sum()} of {len(var)} genes")
pd.Series(coverage).to_frame("value").to_csv(cd.TAB_DIR / "ccc_resource_coverage.csv")
assert not coverage["missing_checks"], coverage["missing_checks"]
```

```python  # cell 13
pc = C.panel_coverage(resource, var_names=var.index)
pc.to_csv(cd.TAB_DIR / "ccc_panel_coverage.csv", index=False)
print(f"{pc.in_resource.sum()}/{len(pc)} curated panel rows resolve to a resource interaction; "
      f"{pc.in_data.sum()} have all genes in the data\n")
print("respelled by the resolver:")
print(pc[(pc.orientation == 'forward') &
         ((pc.ligand_in != pc.ligand_complex) | (pc.receptor_in != pc.receptor_complex))]
      [["ligand_in", "receptor_in", "ligand_complex", "receptor_complex"]].to_string(index=False))
print("\nstored with the direction inverted (read the biology the other way round):")
print(pc[pc.orientation == "flipped"][["ligand_in", "receptor_in",
                                       "ligand_complex", "receptor_complex"]].to_string(index=False))
print("\nabsent from the resource -- cannot be scored at all:")
print(pc[~pc.in_resource][["group", "ligand_in", "receptor_in"]].to_string(index=False))
```

```python  # cell 14
# controls resolved to the resource's spelling, with source/target swapped where the
# resource stores the pair inverted
pos = C.resolve_controls(cd.POSITIVE_CONTROLS, resource, verbose=True)
neg = C.resolve_controls(cd.NEGATIVE_CONTROLS, resource, verbose=True)
print("\npositives:"); [print(" ", p) for p in pos]
print("negatives:"); [print(" ", n) for n in neg]
```

## §3 · Build the object (LSF)

Pure sparse I/O, ~13–15 min for the main pass (measured: 2,000 rows in 2.2 s). **No GPU** —
`gsla_high_gpu` would only queue behind the 700 GB group reservation cap. Run the cell below
to print the command, then submit it from a shell.

```python  # cell 16
print(f"cd {NB_DIR / 'jobs'} && ./run_ccc_build.sh --force")
print("\nwatch:  tail -f", NB_DIR / "jobs" / "run_ccc_build.bsub.log")
print("\nexpected outputs:")
for p in [cd.CCC_ADATA, cd.PSEUDOBULK_PQ, cd.PSEUDOBULK_META, cd.TESTSET_H5AD, cd.MANIFEST]:
    print(f"  {'OK ' if p.exists() else '-- '}{p}")
print(f"\nexpected shape: ({len(o)}, {keep_gene_mask.sum()})")
```

Job record — fill in after submitting:

- job id: ``
- submitted: ``
- wall time: ``
- log: `jobs/run_ccc_build.bsub.log`

## §4 · Load and check the invariants

Reconciliation against §1 is the point: if a level count here differs from the parquet audit,
the streaming row mapping is wrong and everything downstream is silently mislabelled.

```python  # cell 19
adata = C.load_ccc_adata()
manifest = json.loads(cd.MANIFEST.read_text())
print("\nmanifest:", {k: manifest[k] for k in
                      ["n_obs", "n_vars", "n_genes_source", "n_donors", "n_samples", "liana_version"]})
```

```python  # cell 20
# structural checks: counts integral, X is lognorm and not a second copy of counts,
# every control gene present
C.assert_ccc_invariants(adata)

# reconciliation with the obs-only audit of section 1
got = adata.obs[cd.GROUPBY].value_counts()
want = o[cd.GROUPBY].value_counts()
recon = pd.concat([want.rename("audit_sec1"), got.rename("built_object")], axis=1)
recon["match"] = recon["audit_sec1"] == recon["built_object"]
display(recon)
assert recon["match"].all(), "level counts diverge from the parquet audit"
assert adata.n_obs == len(o), (adata.n_obs, len(o))
print("\nreconciled.")
```

```python  # cell 21
# the pseudobulk cube is a by-product of the same streaming pass; cross-check the two
# accumulators against each other before trusting either
cube = pd.read_parquet(cd.PSEUDOBULK_PQ)
pmeta = pd.read_parquet(cd.PSEUDOBULK_META)
print("cube", cube.shape, "| meta", pmeta.shape)

shared = [g for g in adata.var_names if g in cube.columns]
lab = (adata.obs[cd.GROUPBY].astype(str) + "||" + adata.obs[cd.SAMPLE_KEY].astype(str)).to_numpy()
import scipy.sparse as sp
lv = cube.index.to_numpy()
codes = pd.Categorical(lab, categories=lv).codes
ind = sp.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes), len(lv)))
from_obj = pd.DataFrame((ind.T @ adata[:, shared].layers[cd.COUNTS_LAYER]).toarray(),
                        index=lv, columns=shared)
assert np.allclose(from_obj.values, cube[shared].values), "pseudobulk cube and object disagree"
print("cube reconciles with the object over", len(shared), "shared genes")
display(pmeta.head())
```

## §5 · The equivalence gate

**Blocking.** `rank_aggregate` on the 3-donor **full-40,821-gene** test object versus the same
cells inside the 1,832-gene `ccc_skin.h5ad`. Every statistic must match. If it does not, the
library-size factor was computed on the wrong gene set and no number produced by nb35 or nb36
means anything.

```python  # cell 23
test = sc.read_h5ad(cd.TESTSET_H5AD)
print("full-gene testset:", test.shape, "| donors:", test.obs[cd.DONOR_KEY].unique().tolist())

subset = adata[adata.obs_names.isin(test.obs_names)].copy()
test = test[subset.obs_names].copy()
assert (test.obs_names == subset.obs_names).all()
print("same", subset.n_obs, "cells in both objects")

for a in (test, subset):
    a.layers[cd.LAYER] = a.X.copy()

pairs = C.build_groupby_pairs({"gate": ([cd.CD4_MALIGNANT], cd.TME_CORE)})
res_full = C.run_rank_aggregate(test, resource, groupby_pairs=pairs, key_added="gate", verbose=False)
res_sub = C.run_rank_aggregate(subset, resource, groupby_pairs=pairs, key_added="gate", verbose=False)
print("\nfull-gene:", res_full.shape, "| subset:", res_sub.shape)
```

```python  # cell 24
keys = ["source", "target", "ligand_complex", "receptor_complex"]
a = res_full.set_index(keys).sort_index()
b = res_sub.set_index(keys).sort_index()
assert a.index.equals(b.index), (
    f"row sets differ: full-only {len(a.index.difference(b.index))}, "
    f"subset-only {len(b.index.difference(a.index))}"
)
for col in ["lr_means", "magnitude_rank", "specificity_rank", "cellphone_pvals", "expr_prod"]:
    d = float(np.nanmax(np.abs(a[col].to_numpy() - b[col].to_numpy())))
    print(f"  {col:<18} max abs diff {d:.3e}")
    assert d < 1e-6, f"{col} differs -- the lognorm size factor is wrong"
print("\nEQUIVALENCE GATE PASSED: restricting var to the resource genes changes nothing.")
```

```python  # cell 25
# and the negative control for the gate itself: normalising AFTER subsetting (the bug this
# guards against) does change the answer, so the gate above is not vacuous
import scipy.sparse as sp
bad = subset.copy()
cnts = bad.layers[cd.COUNTS_LAYER].astype(float)
lib = np.maximum(np.asarray(cnts.sum(axis=1)).ravel(), 1.0)
bad.layers[cd.LAYER] = sp.diags(1e4 / lib).dot(cnts).log1p().tocsr()
res_bad = C.run_rank_aggregate(bad, resource, groupby_pairs=pairs, key_added="gate", verbose=False)
c = res_bad.set_index(keys).sort_index().reindex(a.index)
print("median |lr_means| shift from the subset-first bug:",
      f"{float(np.nanmedian(np.abs(c['lr_means'] - a['lr_means']))):.4f}")
print("library-size inflation this would cause (median full/resource ratio):",
      f"{float(np.median(np.asarray(test.layers[cd.COUNTS_LAYER].sum(axis=1)).ravel() / lib)):.2f}x")
assert not np.allclose(c["lr_means"], a["lr_means"]), "gate is vacuous -- investigate"
print("\ngate is discriminative.")
```

### Outcome

`data/ccc/ccc_skin.h5ad` is ready and verified. Continue in
**`35_ccc_descriptive.ipynb`** — everything from here runs interactively off the ~2 GB object.
