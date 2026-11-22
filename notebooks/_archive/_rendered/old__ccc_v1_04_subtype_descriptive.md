# old/ccc_v1_04_subtype_descriptive.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/ccc_v1_04_subtype_descriptive.ipynb` and is never run._

# 38 · CCC — descriptive LR map at myeloid / fibroblast **sub-level** resolution

nb35 ran the ligand–receptor map with `Myeloid` (93,435 cells) and `Fibroblast` (66,760) as single
pooled levels. This notebook re-runs it with those two lineages split into the states annotated by
`10c_skin_myeloid_fibro_reannotation.ipynb`, on a narrowed roster:

`CD4_malignant`, `CD4_reactive`, `CD8`, `B`, `Keratinocyte` + the myeloid and fibroblast sub-levels.
`CD4_unassessed`, `Tregs`, `Plasma`, `Vascular`, `Mast`, `Melanocyte` are **out of scope here** —
their nb35/36 results stand; dropping them bounds the multiple-testing surface so that splitting two
levels into ~13 does not arrive with a bigger grid than v1 had in total.

**Nothing about the malignancy call changes.** The `CD4_malignant` / `CD4_reactive` split is
byte-identical to v1's ALICE-TCR definition, and `CD8` / `B` / `Keratinocyte` are carried over
cell-for-cell. That is deliberate: it makes those axes a **regression test** (§12) — CXCL13→CXCR5,
CD40LG→CD40 and CD86→CD28 with B must reproduce, because neither side of them moved.

**The object is not rebuilt.** `ccc_skin.h5ad` already contains every myeloid and fibroblast cell;
only the grouping column and the roster change, so the label is attached by a `cell_id` join. The
one stale by-product is `ccc_skin_pseudobulk_full.parquet` (keyed by `ccc_celltype × sample`), which
feeds only the deferred donor-level differential phase — re-run `jobs/run_ccc_build.py` if that
phase starts.

**What splitting a level costs.** k sub-levels multiply the tested grid by k while each carries
fewer cells and fewer donors. A pair that was solid on pooled `Myeloid` can therefore look weaker on
every sub-level with no biology having changed. §12 reads each sub-level result against the v1
pooled result rather than on its own, and §0 prints the donor coverage that decides which sub-levels
may carry a claim at all.

```python  # cell 1
import warnings; warnings.filterwarnings("ignore")
import importlib
import sys
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
import ccc_data as v1          # the pooled-level config, for the regression comparison
import ccc_data_sub as cd      # this run's config
import ccc_helpers as C
# A long-lived kernel holds whichever version of these modules it imported first; reload so a
# helper added to ccc_helpers.py after the first import is actually visible here.
for _m in (v1, cd, C):
    importlib.reload(_m)
assert hasattr(C, "build_ccc_celltype_sub"), "stale ccc_helpers -- restart the kernel"

sc.settings.verbosity = 1
mpl.rcParams["figure.dpi"] = 110
mpl.rcParams["savefig.bbox"] = "tight"
cd.TAB_DIR.mkdir(parents=True, exist_ok=True); cd.FIG_DIR.mkdir(parents=True, exist_ok=True)
print("liana", li.__version__)
print("roster:", cd.KEEP_LEVELS)
print("\n" + cd.CAVEAT_BLOCK)
```

## §0 · Attach the sub-level label, narrow the roster, audit the coverage

This section absorbs the role nb34 played for v1 (build QC + feasibility), because no build job runs
here. Four gates before any liana call: the sidecar's levels match `ccc_data_sub.MYELOID_LEVELS` /
`FIBRO_LEVELS`; every pooled myeloid/fibroblast cell is either sub-labelled or dropped for a stated
reason; the five untouched levels are carried over cell-for-cell; and the object still passes
`assert_ccc_invariants`.

```python  # cell 3
# ---------------------------------------------------------------- the object, unchanged
# ccc_skin.h5ad is NOT rebuilt: it already holds every myeloid and fibroblast cell, and only the
# grouping column and the roster change. Load it with its own manifest asserts (which validate
# the v1 label), then attach the sub-level label and narrow.
adata = C.load_ccc_adata(verbose=False)
adata.layers[cd.LAYER] = adata.X          # liana reads layer=LAYER; X already is lognorm
C.assert_ccc_invariants(adata)
print(f"\nv1 object: {adata.n_obs:,} cells x {adata.n_vars:,} resource genes")
print(adata.obs[v1.GROUPBY].value_counts().to_string())

# ---------------------------------------------------------------- the sidecar
assert cd.SUBTYPE_CSV.exists(), (
    f"missing {cd.SUBTYPE_CSV} -- run 10c_skin_myeloid_fibro_reannotation.ipynb first "
    "(it does both lineages in one pass; section 9 writes this file)")
side = pd.read_csv(cd.SUBTYPE_CSV, dtype=str)
print(f"\nnb10c sidecar: {len(side):,} rows")
print(pd.crosstab(side["lineage"], side["subtype_ccc"]).to_string())

# The two lists in ccc_data_sub are the contract between nb10c and this notebook. Assert it here
# rather than discovering a typo as an empty dot plot ten cells down.
got_m = sorted(set(side.loc[side.lineage == "Myeloid", "subtype_ccc"].dropna()))
got_f = sorted(set(side.loc[side.lineage == "Fibroblast", "subtype_ccc"].dropna()))
assert got_m == sorted(cd.MYELOID_LEVELS), (
    f"MYELOID_LEVELS in ccc_data_sub.py is stale.\n  sidecar: {got_m}\n  config : "
    f"{sorted(cd.MYELOID_LEVELS)}\nPaste the lists printed by nb10c section 9.")
assert got_f == sorted(cd.FIBRO_LEVELS), (
    f"FIBRO_LEVELS in ccc_data_sub.py is stale.\n  sidecar: {got_f}\n  config : "
    f"{sorted(cd.FIBRO_LEVELS)}")
print("\nMYELOID_LEVELS / FIBRO_LEVELS match the sidecar")

# ---------------------------------------------------------------- the new grouping
obs = adata.obs.copy()
obs["cell_id"] = obs.index.astype(str)
adata.obs[cd.GROUPBY] = C.build_ccc_celltype_sub(
    obs, cd.SUBTYPE_CSV, cd.KEEP_LEVELS, key=cd.GROUPBY, verbose=True)

# Every pooled Myeloid/Fibroblast cell must either carry a sub-level or have been dropped for a
# stated reason (UNK / proliferating / pericyte contamination) -- never silently vanish.
pooled = adata.obs[v1.GROUPBY].astype(str)
sub_lab = adata.obs[cd.GROUPBY].astype(str)
for lineage, levels in [("Myeloid", cd.MYELOID_LEVELS), ("Fibroblast", cd.FIBRO_LEVELS)]:
    m = pooled == lineage
    n_lab = int(sub_lab[m].isin(levels).sum())
    n_side = int(side[(side.lineage == lineage) & side.subtype_ccc.notna()].shape[0])
    assert n_lab == n_side, (lineage, n_lab, n_side)
    print(f"{lineage}: {int(m.sum()):,} pooled -> {n_lab:,} sub-labelled "
          f"({int(m.sum()) - n_lab:,} dropped as UNK/prolif/contaminant)")
# the untouched levels must be carried over cell-for-cell -- this is what makes the B/CD8 axes a
# usable regression test against nb35/36
for lv in ["CD8", "B", "Keratinocyte", cd.CD4_MALIGNANT, cd.CD4_REACTIVE]:
    assert int((pooled == lv).sum()) == int((sub_lab == lv).sum()), lv
print("CD4_malignant / CD4_reactive / CD8 / B / Keratinocyte carried over unchanged")

sub_all = adata[adata.obs[cd.GROUPBY].notna()].copy()
sub_all.obs[cd.GROUPBY] = sub_all.obs[cd.GROUPBY].cat.remove_unused_categories()
sub_all.layers[cd.LAYER] = sub_all.X
del adata
print(f"\nroster object: {sub_all.n_obs:,} cells, {len(sub_all.obs[cd.GROUPBY].cat.categories)} levels")
print(sub_all.obs[cd.GROUPBY].value_counts().to_string())

resource, coverage = C.load_resource(var_names=sub_all.var_names)
```

```python  # cell 4
# ============================================================================
# §0b  Coverage and axis feasibility — which sub-levels may carry a claim
# ============================================================================
counts, summary = C.cell_count_audit(sub_all, groupby=cd.GROUPBY, sample_key=cd.DONOR_KEY,
                                     min_cells=cd.MIN_CELLS)
cov = C.coverage_table(sub_all, groupby=cd.GROUPBY, sample_key=cd.DONOR_KEY, levels=cd.CT_ORDER)
cov.to_csv(cd.tab("coverage_by_donor"))
display(cov)

# restricted to the window the runs actually use
ctcl = C.focal_window(sub_all, disease=cd.CTCL_DISEASES, groupby=cd.GROUPBY, verbose=False)
counts_ctcl, _ = C.cell_count_audit(ctcl, groupby=cd.GROUPBY, sample_key=cd.DONOR_KEY,
                                    min_cells=cd.MIN_CELLS)
feas = C.axis_feasibility(counts_ctcl, axes=cd.FOCAL_AXES, min_cells=cd.MIN_CELLS,
                          min_samples=cd.MIN_SAMPLES, thin_levels=[])
feas.to_csv(cd.tab("axis_feasibility"), index=False)
print(f"\naxis feasibility in the CTCL window (min_cells={cd.MIN_CELLS}, "
      f"min_samples={cd.MIN_SAMPLES}):")
display(feas.groupby("verdict").size().rename("n_axes").to_frame())
display(feas[feas.verdict != "claim"])

CLAIMABLE = sorted({lv for lv in cd.KEEP_LEVELS
                    if lv in counts_ctcl.index
                    and int((counts_ctcl.loc[lv] >= cd.MIN_CELLS).sum()) >= cd.MIN_SAMPLES})
print(f"\nclaimable levels ({len(CLAIMABLE)}/{len(cd.KEEP_LEVELS)}): {CLAIMABLE}")
REPORT_ONLY = [lv for lv in cd.KEEP_LEVELS if lv not in CLAIMABLE]
if REPORT_ONLY:
    print(f"report-only (below the donor gate): {REPORT_ONLY}")
    print("  These are still computed and plotted — absence goes on the record — but nb39 excludes"
          " them from the headline set. If a level you care about is here, revisit the collapse in"
          " nb10c section 8 rather than lowering the gate.")
```

```python  # cell 5
# ============================================================================
# §0c  Study confounding — a sub-level that is one study is a batch, not a state
# ============================================================================
ct = pd.crosstab(sub_all.obs[cd.GROUPBY].astype(str), sub_all.obs[cd.STUDY_KEY].astype(str))
frac = ct.div(ct.sum(1), axis=0)
conf = pd.DataFrame({
    "n_cells": ct.sum(1),
    "n_studies": (ct > 0).sum(1),
    "top_study": frac.idxmax(1),
    "top_study_frac": frac.max(1).round(3),
    "n_donors": sub_all.obs.groupby(sub_all.obs[cd.GROUPBY].astype(str),
                                    observed=True)[cd.DONOR_KEY].nunique(),
}).reindex([lv for lv in cd.CT_ORDER if lv in ct.index])
conf["flag"] = np.where(conf["top_study_frac"] > 0.8, "STUDY-DOMINATED", "")
conf.to_csv(cd.tab("study_confounding"))
display(conf)
print("A level whose cells are >80% one study cannot be distinguished from that study's chemistry."
      "\nliana is run on it anyway — the flag travels with the result, it does not gate it — but no"
      "\nclaim rests on such a level alone. Compare against the same level in nb35's pooled run.")
```

```python  # cell 6
# ============================================================================
# §0d  Balanced subsample — the default, as in nb35
# ============================================================================
# The v1 cap of 15,000 per level was set against 13 pooled levels. Here CD4_malignant (76,349)
# would sit an order of magnitude above pDC or F_apCAF, and liana's permutation p-value shrinks
# with group size, so the cap drops to 8,000 with a 1,000-per-donor-per-level ceiling that keeps
# li2024 (56% of skin cells) from dominating any level.
bal, sub_report = C.subsample_levels(
    sub_all, groupby=cd.GROUPBY,
    max_per_level=cd.SUBSAMPLE_MAX_PER_LEVEL,
    max_per_donor_per_level=cd.SUBSAMPLE_MAX_PER_DONOR_PER_LEVEL,
    donor_key=cd.DONOR_KEY, seed=cd.SUBSAMPLE_SEED)
sub_report.to_csv(cd.tab("subsample_report"), index=False)
bal.layers[cd.LAYER] = bal.X
print("\ndonors retained per level:")
print(bal.obs.groupby(bal.obs[cd.GROUPBY].astype(str), observed=True)[cd.DONOR_KEY]
      .nunique().sort_values(ascending=False).to_string())


def run(name, source, key_added, out_name):
    """One ANALYSES entry -> (result, sub, pairs). Writes tables/ccc_sub_<out_name>.csv."""
    s, pairs, spec = C.prepare_analysis(source, name, groupby=cd.GROUPBY,
                                        analyses=cd.ANALYSES, axes=cd.FOCAL_AXES)
    res = C.run_rank_aggregate(s, resource, groupby=cd.GROUPBY, groupby_pairs=pairs,
                               expr_prop=cd.EXPR_PROP, min_cells=cd.MIN_CELLS,
                               n_perms=cd.N_PERMS, n_jobs=cd.N_JOBS, seed=cd.SEED,
                               key_added=key_added)
    res.to_csv(cd.tab(out_name), index=False)
    print(f"{name}: {res.shape} -> {cd.tab(out_name).name}")
    return res, s, pairs
```

## §1 · Myeloid axis — malignant CD4 ↔ each myeloid state

The primary run. In nb35 this was two dot-plot rows (`malignant CD4 → Myeloid` and back); here it is
one row per state. The question is not only *which* pairs rank highest but whether a v1 edge
**concentrates** on one state or is shared across all of them — a shared edge is evidence the pooled
level was not hiding anything, and it is reported as such in §11 rather than quietly dropped.

```python  # cell 8
res_mye, sub_mye, pairs_mye = run("sub_myeloid", bal, "liana_mye", "liana_myeloid")
display(res_mye.sort_values("magnitude_rank").head(25))
```

```python  # cell 9
C.dotplot_axis(liana_res=res_mye, source_labels=[cd.CD4_MALIGNANT], top_n=cd.DOTPLOT_TOP_N,
               figure_size=(9, 8), title="malignant CD4 -> myeloid states")
```

```python  # cell 10
C.dotplot_axis(liana_res=res_mye, target_labels=[cd.CD4_MALIGNANT], top_n=cd.DOTPLOT_TOP_N,
               figure_size=(9, 8), title="myeloid states -> malignant CD4")
```

## §2 · Fibroblast axis — malignant CD4 ↔ each fibroblast state

`CXCL12 → CXCR4` is the named hypothesis: CAFs in mycosis fungoides have been shown to drive
malignant-cell migration and doxorubicin resistance through that axis, and v1 could only attribute it
to "Fibroblast". If it localises to `F_inflammatory`, the split has bought a testable statement; if
it is uniform across all fibroblast states, it is a fibroblast property rather than a CAF one.
`F_apCAF` exists to test li2024's MHC-II⁺-fibroblast claim directly — read against
`LR_CAVEATS['HLA-DRA']`.

```python  # cell 12
res_fib, sub_fib, pairs_fib = run("sub_fibro", bal, "liana_fib", "liana_fibro")
display(res_fib.sort_values("magnitude_rank").head(25))
```

```python  # cell 13
C.dotplot_axis(liana_res=res_fib, source_labels=[cd.CD4_MALIGNANT], top_n=cd.DOTPLOT_TOP_N,
               figure_size=(9, 8), title="malignant CD4 -> fibroblast states")
```

```python  # cell 14
C.dotplot_axis(liana_res=res_fib, target_labels=[cd.CD4_MALIGNANT], top_n=cd.DOTPLOT_TOP_N,
               figure_size=(9, 8), title="fibroblast states -> malignant CD4")
```

## §3 · Unchanged partners — CD8, B, keratinocyte

Neither sender nor receiver moved on these axes, so they are the control surface for the whole
re-run: any shift versus nb35 comes from the roster and the subsampling cap, not from the new labels.
§12 quantifies that.

```python  # cell 16
res_lym, sub_lym, pairs_lym = run("sub_lymphoid", bal, "liana_lym", "liana_lymphoid")
res_str, sub_str, pairs_str = run("sub_structural", bal, "liana_str", "liana_structural")
C.dotplot_axis(liana_res=res_lym, target_labels=[cd.CD4_MALIGNANT], top_n=cd.DOTPLOT_TOP_N,
               title="CD8 / B -> malignant CD4")
```

```python  # cell 17
C.dotplot_axis(liana_res=res_str, source_labels=[cd.CD4_MALIGNANT], top_n=cd.DOTPLOT_TOP_N,
               title="malignant CD4 <-> keratinocyte")
```

## §4 · Malignant CD4 ↔ reactive CD4

Sender and receiver share a lineage, so shared-gene and autocrine artifacts are maximal here: read
only pairs whose ligand is asymmetric between the two levels. §8 computes that asymmetry explicitly.

```python  # cell 19
res_rea, sub_rea, pairs_rea = run("sub_reactive", bal, "liana_rea", "liana_reactive")
C.dotplot_axis(liana_res=res_rea, top_n=cd.DOTPLOT_TOP_N,
               title="malignant CD4 <-> reactive CD4")
```

## §5 · The comparator, and the rank-delta

**THE** reading rule of this pipeline, unchanged from v1 and now applied per sub-level: no claim of
the form *"malignant CD4 signals X to state Y"* is reportable unless the same pair is absent or
weaker with **reactive** CD4 as the sender, against the **same** state, in the same tissue. A pair
that ranks well from both is a CD4 property.

```python  # cell 21
res_cmp_mye, _, _ = run("sub_comparator_myeloid", bal, "liana_cmp_mye", "liana_comparator_myeloid")
res_cmp_fib, _, _ = run("sub_comparator_fibro", bal, "liana_cmp_fib", "liana_comparator_fibro")
res_cmp_lym, _, _ = run("sub_comparator_lymphoid", bal, "liana_cmp_lym", "liana_comparator_lymphoid")
res_cmp_str, _, _ = run("sub_comparator_structural", bal, "liana_cmp_str",
                        "liana_comparator_structural")

RES_MAL = pd.concat([res_mye, res_fib, res_lym, res_str], ignore_index=True)
RES_REA = pd.concat([res_cmp_mye, res_cmp_fib, res_cmp_lym, res_cmp_str], ignore_index=True)
print(RES_MAL.shape, RES_REA.shape)
```

```python  # cell 22
delta_out = C.rank_delta(RES_MAL.query("source == @cd.CD4_MALIGNANT"),
                         RES_REA.query("source == @cd.CD4_REACTIVE"))
delta_out.to_csv(cd.tab("rank_delta_outgoing"), index=False)
print("OUTGOING -- most malignant-shifted (negative delta = better rank from malignant CD4):")
display(delta_out.head(30))
print("found only with malignant CD4 as sender:", int(delta_out["malignant_only"].sum()))
```

```python  # cell 23
delta_in = C.rank_delta(
    RES_MAL.query("target == @cd.CD4_MALIGNANT").rename(columns={"source": "target",
                                                                 "target": "source"}),
    RES_REA.query("target == @cd.CD4_REACTIVE").rename(columns={"source": "target",
                                                                "target": "source"}))
delta_in.to_csv(cd.tab("rank_delta_incoming"), index=False)
print("INCOMING -- TME pairs most shifted toward malignant CD4 as the receiver:")
display(delta_in.head(30))
```

## §6 · Controls

`POSITIVE_CONTROLS` are re-pointed at the sub-level each edge is *expected* to live on (CCL17/CCL22
at `cDC`/`DC_LAMP3`, CSF1 at `Mac_SPP1_TREM2`, CXCL12 at `F_inflammatory`). A control landing on a
different sub-level is **not** a failure — it is the result this re-run exists to produce — so §11
reports where each one actually landed, not merely whether it was found. `SPEC_MUST_HAVES` (the
three edges the replication spec names, all on the untouched B level) are held to the v1 bar.

```python  # cell 25
res_all = pd.concat([RES_MAL, RES_REA, res_rea], ignore_index=True)
ctrl, caveat_hits = C.control_report(res_all, positives=cd.POSITIVE_CONTROLS,
                                     negatives=cd.NEGATIVE_CONTROLS, caveats=cd.LR_CAVEATS,
                                     top_n=40, resource=resource)
ctrl.to_csv(cd.tab("control_report"), index=False)
caveat_hits.to_csv(cd.tab("caveat_hits"), index=False)
display(ctrl)
```

```python  # cell 26
spec_ctrl = C.resolve_controls(cd.SPEC_MUST_HAVES, resource)
keys = ["source", "target", "ligand_complex", "receptor_complex"]
found = ctrl.merge(pd.DataFrame(spec_ctrl, columns=keys), on=keys)
print("replication-spec must-haves (all on the UNTOUCHED B level -- these must reproduce):")
display(found)
if len(found) and "pct_of_tested" in found:
    print("in the top quartile of magnitude_rank:",
          int((found["pct_of_tested"] < 0.25).sum()), "/", len(found))
```

## §7 · Sensitivity: `expr_prop`, label shuffle, downsampling

Run on the myeloid axis, the one with the most levels and therefore the most exposure to
`expr_prop` — it is a *detection-rate* threshold, and a state with fewer cells crosses it less often
purely on n.

```python  # cell 28
sweep, stable = C.expr_prop_sweep(sub_mye, resource, props=cd.EXPR_PROP_SWEEP, top_n=cd.TOP_N,
                                  groupby=cd.GROUPBY, groupby_pairs=pairs_mye,
                                  min_cells=cd.MIN_CELLS, n_perms=cd.N_PERMS, verbose=False)
sweep.to_csv(cd.tab("expr_prop_sweep_myeloid"), index=False)
print(f"{len(stable)}/{cd.TOP_N} of the top-{cd.TOP_N} are stable across expr_prop "
      f"{cd.EXPR_PROP_SWEEP}")
display(pd.DataFrame(sorted(stable), columns=["source", "target", "ligand_complex",
                                              "receptor_complex"]))
```

```python  # cell 29
shuf, overlap = C.shuffle_control(sub_mye, resource, groupby=cd.GROUPBY, within=cd.DONOR_KEY,
                                  n_shuffles=cd.N_SHUFFLES, top_n=cd.TOP_N, seed=cd.SEED,
                                  reference=res_mye, groupby_pairs=pairs_mye,
                                  min_cells=cd.MIN_CELLS, n_perms=cd.N_PERMS, verbose=False)
overlap.to_csv(cd.tab("shuffle_control"), index=False)
print("\nmean overlap:", overlap["overlap_with_real"].mean(), "-- target <= 2 of", cd.TOP_N)
print("Shuffling WITHIN donor holds each donor's composition fixed, so what dissolves is "
      "sub-level specificity rather than donor identity. With 8 myeloid levels instead of 1, a "
      "high overlap here would mean the sub-levels are not transcriptionally distinct enough to "
      "carry separate CCC edges -- which would be a finding about nb10c, not about liana.")
```

```python  # cell 30
stab, tops = C.downsample_stability(sub_mye, resource, groupby=cd.GROUPBY,
                                    sizes=cd.DOWNSAMPLE_SIZES, top_n=cd.TOP_N, seed=cd.SEED,
                                    groupby_pairs=pairs_mye, min_cells=cd.MIN_CELLS,
                                    n_perms=cd.N_PERMS)
stab.to_csv(cd.tab("downsample_stability"), index=False)
display(stab)
```

## §8 · Forced curated panel, and the autocrine check

The curated `LR_PANEL` is plotted whether or not a pair cleared `expr_prop`, so a negative is
*reported* rather than silently dropped. `forced_panel_expression` then separates the two reasons a
pair can be missing: the gene is not detected in that sub-level (a real negative at this depth), or
it sits just under the threshold (a detection limit).

```python  # cell 32
for gene, note in cd.LR_CAVEATS.items():
    print(f"{gene}: {note}\n")
```

```python  # cell 33
panel_res = pd.concat(
    [C.filter_to_panel(r, panel=cd.LR_PANEL, resource=resource).assign(analysis=name)
     for name, r in [("myeloid", res_mye), ("fibro", res_fib), ("lymphoid", res_lym),
                     ("structural", res_str), ("reactive", res_rea),
                     ("comparator_myeloid", res_cmp_mye), ("comparator_fibro", res_cmp_fib),
                     ("comparator_lymphoid", res_cmp_lym)]],
    ignore_index=True)
panel_res.to_csv(cd.tab("curated_panel"), index=False)
print(f"{len(panel_res)} scored rows over {panel_res.group.nunique()} panel groups")
display(panel_res.groupby(["group", "analysis"]).size().rename("n_scored").to_frame())
```

```python  # cell 34
fpe = C.forced_panel_expression(bal, panel=cd.LR_PANEL, groupby=cd.GROUPBY, layer=cd.LAYER,
                                min_cells=cd.MIN_CELLS, resource=resource)
fpe.to_csv(cd.tab("forced_panel_expression"), index=False)

near = fpe[(fpe.expr_prop > 0.01) & (fpe.expr_prop < cd.EXPR_PROP)]
print(f"curated genes between 1% and expr_prop={cd.EXPR_PROP} -- their absence from a result is a "
      f"detection limit, not a negative:")
display(near.sort_values("expr_prop", ascending=False).head(30))
```

```python  # cell 35
# Autocrine check: a ligand at a similar proportion in both CD4 levels is a CD4 property, not a
# directional signal on the malignant <-> reactive axis.
piv = (fpe[fpe.level.isin([cd.CD4_MALIGNANT, cd.CD4_REACTIVE])]
       .pivot(index="gene", columns="level", values="expr_prop"))
piv["ratio_mal_over_rea"] = piv[cd.CD4_MALIGNANT] / piv[cd.CD4_REACTIVE].replace(0, np.nan)
print("curated genes most malignant-skewed within CD4:")
display(piv.sort_values("ratio_mal_over_rea", ascending=False).head(20))
print("curated genes symmetric between the two CD4 levels (autocrine artifact risk):")
display(piv[(piv.ratio_mal_over_rea.between(0.8, 1.25))
            & (piv[cd.CD4_MALIGNANT] > cd.EXPR_PROP)].head(20))
```

## §9 · Study diagnostic

`expr_prop` is a detection-rate threshold and chemistry tracks study, so a pair that passes in one
study and fails in another is a capture artifact rather than a biological difference. This matters
more here than in v1: the sub-levels are smaller, so per-study donor counts are thinner.

```python  # cell 37
per_study_donors = (bal.obs.groupby([bal.obs[cd.STUDY_KEY].astype(str),
                                     bal.obs[cd.GROUPBY].astype(str)],
                                    observed=True)[cd.DONOR_KEY].nunique().unstack(fill_value=0))
display(per_study_donors)
ok_studies = [s for s in per_study_donors.index
              if per_study_donors.loc[s, cd.CD4_MALIGNANT] >= 3
              and (per_study_donors.loc[s, [lv for lv in cd.MYELOID_LEVELS
                                            if lv in per_study_donors.columns]] >= 3).any()]
print("studies with >= 3 donors on malignant CD4 and on at least one myeloid state:", ok_studies)

by_study = C.run_by_group(sub_mye, resource, cd.STUDY_KEY, groups=ok_studies,
                          min_cells=cd.MIN_CELLS, groupby=cd.GROUPBY, groupby_pairs=pairs_mye,
                          n_perms=cd.N_PERMS, verbose=False)
by_study.to_csv(cd.tab("liana_by_study_myeloid"), index=False)
print(by_study.shape)
```

```python  # cell 38
if len(by_study):
    named = {s: by_study[by_study[cd.STUDY_KEY] == s] for s in ok_studies}
    ov = C.top_n_overlap_matrix(named, top_n=cd.TOP_N)
    print(f"top-{cd.TOP_N} overlap between studies (myeloid axis):")
    display(ov)
    ov.to_csv(cd.tab("study_topn_overlap"))
    keys = ["source", "target", "ligand_complex", "receptor_complex"]
    per = by_study.groupby(keys)[cd.STUDY_KEY].nunique().rename("n_studies")
    top = res_mye.sort_values("magnitude_rank").head(30).set_index(keys)
    print("\nhow many studies recover each of the top-30 pooled-over-studies pairs:")
    display(top.join(per).reset_index()[keys + ["magnitude_rank", "n_studies"]])
```

## §10 · Malignancy-definition sensitivity

Same arm as nb35 §13, re-run at sub-level resolution: does the myeloid axis survive swapping the
ALICE-TCR primary for `mal_combined` / `mal_cnv` / li2024's own `tumor_cell` label? The CDR3 gate is
switched off for the alternatives on purpose — it encodes "could ALICE test this cell", which is
meaningless for a CNV- or paper-label-derived call.

```python  # cell 40
alt_res, alt_counts = {cd.MALIG_SRC: res_mye}, {}
for defn in ["mal_combined", "mal_cnv", "tumor_cell"]:
    a = bal.copy()
    o = a.obs.copy(); o["cell_id"] = o.index.astype(str)
    if defn == "tumor_cell":
        # li2024's own label; "Unknown" outside li2024, so its CD4_unassessed absorbs every
        # non-li2024 CD4 -- which is why it is a sensitivity arm and never the primary.
        alt = o[cd.FINE_SRC].astype(str)
        flag = pd.Series(np.nan, index=o.index, dtype="object")
        flag[alt == "tumor_cell"] = True
        flag[(alt != "tumor_cell") & (alt != "Unknown")] = False
        o["_alt_malig"] = flag
        col = "_alt_malig"
    else:
        col = defn
    a.obs[cd.GROUPBY] = C.build_ccc_celltype_sub(
        o, cd.SUBTYPE_CSV, cd.KEEP_LEVELS, key=cd.GROUPBY,
        malig_src=col, tcr_assessed_src=None, verbose=False)
    a = a[a.obs[cd.GROUPBY].notna()].copy()
    a.obs[cd.GROUPBY] = a.obs[cd.GROUPBY].cat.remove_unused_categories()
    a.layers[cd.LAYER] = a.X
    alt_counts[defn] = a.obs[cd.GROUPBY].value_counts().to_dict()
    try:
        alt_res[defn] = C.run_rank_aggregate(a, resource, groupby=cd.GROUPBY,
                                             groupby_pairs=pairs_mye, min_cells=cd.MIN_CELLS,
                                             n_perms=cd.N_PERMS, key_added=f"liana_{defn}",
                                             verbose=False)
    except Exception as exc:                                        # noqa: BLE001
        print(f"{defn}: {exc}")
print(f"\nprimary = {cd.MALIG_SRC}; alternatives relabelled without the CDR3 gate\n")
print(pd.DataFrame(alt_counts).fillna(0).astype(int).to_string())
```

```python  # cell 41
ov_def = C.top_n_overlap_matrix(alt_res, top_n=cd.TOP_N)
ov_def.to_csv(cd.tab("malignancy_definition_topn_overlap"))
print(f"top-{cd.TOP_N} overlap across malignancy definitions (myeloid axis):")
display(ov_def)

keys = ["source", "target", "ligand_complex", "receptor_complex"]
n_def = (pd.concat([r.sort_values("magnitude_rank").head(cd.TOP_N).assign(defn=k)
                    for k, r in alt_res.items()])
         .groupby(keys)["defn"].nunique().rename("n_definitions"))
n_def.sort_values(ascending=False).to_csv(cd.tab("pair_definition_stability"))
print(f"\npairs in the top-{cd.TOP_N} of all {len(alt_res)} definitions:")
display(n_def[n_def == len(alt_res)].to_frame())
```

## §11 · Where does each named edge actually land?

**This is the section the whole re-run exists for.** For each edge in `RESOLUTION_TESTS`, rank every
candidate sub-level by `magnitude_rank` and report the winner, the runner-up and the gap. Three
outcomes, all of them results:

- **concentrated** on the expected state → the split resolved a v1 edge into a testable statement;
- **concentrated on a different state** → the v1 attribution was to the wrong cell, and this is the
  more interesting outcome;
- **uniform across states** → the pooled level was not hiding anything for that edge; say so.

An edge scored on *no* candidate level is reported separately — that is a coverage failure
(`min_cells` / `expr_prop`), not a biological negative, and the coverage table of §0b says which.

```python  # cell 43
KEYS = ["source", "target", "ligand_complex", "receptor_complex"]


def landing(res, spec, name):
    """Rank the candidate sub-levels for one named edge. Handles resource direction flips."""
    focal = spec.get("sender", spec.get("receiver"))
    as_sender = "sender" in spec
    ctrls = [(focal, lv, lig, rec) if as_sender else (lv, focal, lig, rec)
             for lv in spec["candidates"] for lig, rec in spec["pairs"]]
    want = pd.DataFrame(C.resolve_controls(ctrls, resource), columns=KEYS)
    got = want.merge(res, on=KEYS, how="left")
    got["level"] = got["target"] if as_sender else got["source"]
    got["test"] = name
    return got.sort_values("magnitude_rank")


rows, summary = [], []
for name, spec in cd.RESOLUTION_TESTS.items():
    got = landing(RES_MAL, spec, name)
    rows.append(got)
    scored = got[got["magnitude_rank"].notna()]
    print(f"\n=== {name}: {spec['pairs']}  ({spec['why']})")
    print(f"    expected on {spec['expected']}")
    if scored.empty:
        print("    NOT SCORED on any candidate level -- a coverage failure (min_cells / "
              "expr_prop), not a negative. Check the coverage table in section 0b.")
        summary.append({"test": name, "verdict": "not_scored", "winner": None,
                        "runner_up": None, "gap_rank_ratio": np.nan,
                        "as_expected": False, "n_levels_scored": 0,
                        "n_candidates": len(spec["candidates"])})
        continue
    display(scored[["level", "ligand_complex", "receptor_complex", "lr_means",
                    "magnitude_rank", "specificity_rank", "cellphone_pvals"]].head(12))
    # one row per level: the level's best rank for any of the pairs in the test
    per_level = (scored.groupby("level")["magnitude_rank"].min().sort_values())
    best_lv, best_rank = per_level.index[0], float(per_level.iloc[0])
    second_lv = per_level.index[1] if len(per_level) > 1 else None
    gap = float(per_level.iloc[1] / best_rank) if second_lv is not None and best_rank > 0 else np.nan
    n_lv = len(per_level)
    # "uniform" = most candidates scored AND the winner is not clearly ahead of the runner-up
    spread_out = n_lv >= max(3, 0.6 * len(spec["candidates"]))
    close = (not np.isnan(gap)) and gap < 3
    verdict = "uniform" if (spread_out and close) else "concentrated"
    summary.append({"test": name, "verdict": verdict, "winner": best_lv,
                    "runner_up": second_lv,
                    "gap_rank_ratio": None if np.isnan(gap) else round(gap, 2),
                    "as_expected": best_lv in spec["expected"],
                    "n_levels_scored": int(n_lv),
                    "n_candidates": len(spec["candidates"])})

resolution = pd.concat(rows, ignore_index=True)
resolution.to_csv(cd.tab("resolution_tests"), index=False)
summary = pd.DataFrame(summary)
summary.to_csv(cd.tab("resolution_summary"), index=False)
print("\n\nWHERE EACH NAMED EDGE LANDED")
display(summary)
print("""
Reading the verdict
  concentrated + as_expected      -> the split resolved a v1 edge onto the predicted state
  concentrated + NOT as_expected  -> the v1 attribution pointed at the wrong cell; the more
                                     interesting outcome, and the one to chase in nb39
  uniform                         -> the pooled level was not hiding anything for this edge; the
                                     sub-level split buys nothing HERE, which is worth stating
  not_scored                      -> coverage, not biology
gap_rank_ratio is runner-up / winner on magnitude_rank; near 1 means the two states are tied.
None of this is donor-level inference -- see CAVEAT_BLOCK.
""")
```

## §12 · Regression against the pooled v1 run

`CD4_malignant`, `CD4_reactive`, `CD8`, `B` and `Keratinocyte` are carried over cell-for-cell from
nb35, so any difference on those axes comes from the narrowed roster and the smaller subsample cap —
not from the new labels. If the three `SPEC_MUST_HAVES` move materially here, the sub-level results
elsewhere in this notebook inherit that instability and should be read against it.

The myeloid and fibroblast comparison is the other direction: each v1 pooled edge is matched against
its best-scoring sub-level, which is how a "weaker on every sub-level" result gets distinguished from
a real disappearance.

```python  # cell 45
V1 = {"core": v1.TAB_DIR / "ccc_liana_ctcl_all_core.csv",
      "extended": v1.TAB_DIR / "ccc_liana_ctcl_all_extended.csv"}
have_v1 = {k: p for k, p in V1.items() if p.exists()}
if not have_v1:
    print("nb35 tables not found -- run 35_ccc_descriptive.ipynb for the regression comparison.")
    print("Everything above stands on its own; only this section needs them.")
else:
    v1res = pd.concat([pd.read_csv(p).assign(v1_analysis=k) for k, p in have_v1.items()],
                      ignore_index=True)
    print("v1 tables:", {k: str(p.name) for k, p in have_v1.items()}, "->", v1res.shape)

    # (a) untouched levels: same pairs, same levels, ranks should track
    UNCH = [cd.CD4_MALIGNANT, cd.CD4_REACTIVE, "CD8", "B", "Keratinocyte"]
    a = v1res[v1res.source.isin(UNCH) & v1res.target.isin(UNCH)]
    b = pd.concat([RES_MAL, RES_REA], ignore_index=True)
    b = b[b.source.isin(UNCH) & b.target.isin(UNCH)]
    m = a.merge(b, on=KEYS, suffixes=("_v1", "_sub"))
    print(f"\n(a) untouched levels: {len(m)} pairs scored in both runs")
    if len(m) > 2:
        rho = m["magnitude_rank_v1"].rank().corr(m["magnitude_rank_sub"].rank(), method="spearman")
        print(f"    Spearman rho on magnitude_rank: {rho:.3f}  "
              f"(low rho means the roster/cap changed the ranking, not the labels)")
    m.to_csv(cd.tab("v1_regression_unchanged"), index=False)

    spec_keys = pd.DataFrame(C.resolve_controls(cd.SPEC_MUST_HAVES, resource), columns=KEYS)
    print("\n    the three replication-spec must-haves, v1 vs this run:")
    display(spec_keys.merge(m, on=KEYS, how="left")[
        KEYS + ["magnitude_rank_v1", "magnitude_rank_sub",
                "cellphone_pvals_v1", "cellphone_pvals_sub"]])

    # (b) split levels: each v1 pooled edge vs its best sub-level
    for pooled_lv, levels, res_sub in [("Myeloid", cd.MYELOID_LEVELS, res_mye),
                                       ("Fibroblast", cd.FIBRO_LEVELS, res_fib)]:
        av = v1res[(v1res.source == pooled_lv) | (v1res.target == pooled_lv)]
        av = av[(av.source == cd.CD4_MALIGNANT) | (av.target == cd.CD4_MALIGNANT)]
        bs = res_sub.copy()
        bs["level"] = np.where(bs.source == cd.CD4_MALIGNANT, bs.target, bs.source)
        bs["direction"] = np.where(bs.source == cd.CD4_MALIGNANT, "out", "in")
        av = av.assign(direction=np.where(av.source == cd.CD4_MALIGNANT, "out", "in"))
        pk = ["ligand_complex", "receptor_complex", "direction"]
        best = (bs.sort_values("magnitude_rank").groupby(pk, as_index=False)
                .first()[pk + ["level", "magnitude_rank", "lr_means", "cellphone_pvals"]])
        cmp_ = av[pk + ["magnitude_rank", "lr_means"]].merge(
            best, on=pk, how="left", suffixes=("_v1_pooled", "_best_sublevel"))
        cmp_["lost_at_sublevel"] = cmp_["level"].isna()
        cmp_.to_csv(cd.tab(f"v1_regression_{pooled_lv.lower()}"), index=False)
        n_lost = int(cmp_["lost_at_sublevel"].sum())
        print(f"\n(b) {pooled_lv}: {len(cmp_)} v1 pooled edges | {n_lost} not scored on ANY "
              f"sub-level")
        print("    top v1 edges and the sub-level that carries them:")
        display(cmp_.sort_values("magnitude_rank_v1_pooled").head(20))
        if n_lost:
            print(f"    the {n_lost} lost ones -- expected when a pooled level's cells split "
                  f"below expr_prop or min_cells; check against the coverage table, do NOT read "
                  f"as a disappearance:")
            display(cmp_[cmp_["lost_at_sublevel"]].head(15))
```

## §13 · Healthy-skin negative control

The only axis computable in HC skin, now at sub-level resolution: fibroblast state →
keratinocyte / myeloid state. `CD4_malignant` must be **0 cells** here. Most sub-levels will fail
`min_cells` in HC, and that failure is itself informative — it says the sub-level resolution is a
statement about the CTCL window, not about skin in general.

```python  # cell 47
hc, pairs_hc, spec_hc = C.prepare_analysis(sub_all, "hc_structural", groupby=cd.GROUPBY,
                                           analyses=cd.ANALYSES, axes=cd.FOCAL_AXES)
n_mal = int((hc.obs[cd.GROUPBY].astype(str) == cd.CD4_MALIGNANT).sum())
print(f"\nmalignant CD4 in HC skin: {n_mal} (expected 0)")
assert n_mal == 0, "a malignant CD4 call in healthy skin needs explaining before going further"

hc_cov = C.coverage_table(hc, groupby=cd.GROUPBY, sample_key=cd.DONOR_KEY)
hc_cov.to_csv(cd.tab("hc_coverage"))
display(hc_cov)

hc_bal, _ = C.subsample_levels(hc, groupby=cd.GROUPBY,
                               max_per_level=cd.SUBSAMPLE_MAX_PER_LEVEL,
                               max_per_donor_per_level=cd.SUBSAMPLE_MAX_PER_DONOR_PER_LEVEL,
                               donor_key=cd.DONOR_KEY, seed=cd.SUBSAMPLE_SEED, verbose=False)
hc_bal.layers[cd.LAYER] = hc_bal.X
res_hc = C.run_rank_aggregate(hc_bal, resource, groupby=cd.GROUPBY, groupby_pairs=pairs_hc,
                              min_cells=cd.MIN_CELLS, n_perms=cd.N_PERMS,
                              key_added="liana_hc", verbose=False)
res_hc.to_csv(cd.tab("liana_hc_structural"), index=False)
display(res_hc.sort_values("magnitude_rank").head(20))
```

### Outcome

Every table lands in `tables/ccc_sub_*.csv`; `39_ccc_subtype_headline_figures.ipynb` reads them and
builds the figure set.

**What is here.** The malignant-CD4 ↔ TME map at myeloid/fibroblast state resolution, its reactive-CD4
comparator and rank-delta, the full control and sensitivity block from nb35 re-run on the new levels,
the resolution tests of §11, and a regression against the pooled v1 run.

**What is deliberately absent.**
- *Donor-level inference.* LIANA's permutation p-values use the cell as the unit and are
  pseudoreplicated across donors. They size dots; they are not evidence. That is the deferred
  differential phase, and it will need `ccc_skin_pseudobulk_full.parquet` rebuilt for these levels.
- *Stage, disease and layer contrasts.* Unchanged from v1 — see `ccc_data.FORBIDDEN_CONTRASTS`.
  Splitting myeloid and fibroblast does nothing to fix a 7-vs-7-donor stage axis inside one study.
- *The dropped levels.* Vascular, Tregs, Plasma, Mast, Melanocyte and CD4_unassessed are out of
  scope here, not refuted; their nb35/36 results stand.
- *Composition differences between sub-levels across conditions.* That is a differential-abundance
  question, not a CCC one, and it needs the donor as the unit.
