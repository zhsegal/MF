# old/ccc_v1_05_subtype_headline_figures.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/ccc_v1_05_subtype_headline_figures.ipynb` and is never run._

# 39 · CCC — headline figures at myeloid / fibroblast sub-level resolution

The nb36 figure set, rebuilt on the roster of nb38. Two runs only — `pairs_mal` (malignant CD4 ↔
every partner, plus reactive CD4) and `pairs_rea` (reactive CD4 ↔ every partner) — then one panel
pair per partner group.

**The claim gate is the same and is applied first**: a level may appear in a headline figure only if
it clears `min_cells = 25` in at least `min_samples = 5` donors *inside the CTCL window*. Levels that
fail are computed and printed, never plotted as a claim.

**The new figure is §6**: the partner heatmap at sub-level resolution, and the "where did it land"
panel that says, for each named edge, which myeloid or fibroblast state carries it — the thing the
pooled v1 map could not answer.

```python  # cell 1
import warnings; warnings.filterwarnings("ignore")
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

sc.settings.verbosity = 1
mpl.rcParams["figure.dpi"] = 110
mpl.rcParams["savefig.bbox"] = "tight"
cd.TAB_DIR.mkdir(parents=True, exist_ok=True); cd.FIG_DIR.mkdir(parents=True, exist_ok=True)
print("liana", li.__version__)
print("roster:", cd.KEEP_LEVELS)
print("\n" + cd.CAVEAT_BLOCK)
```

## §0 · Load, label, gate, and recompute both axes

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
# §0b  The claim gate, then the two runs
# ============================================================================
ctcl = C.focal_window(sub_all, disease=cd.CTCL_DISEASES, groupby=cd.GROUPBY)
counts_ctcl, _ = C.cell_count_audit(ctcl, groupby=cd.GROUPBY, sample_key=cd.DONOR_KEY,
                                    min_cells=cd.MIN_CELLS)
donors_ok = (counts_ctcl >= cd.MIN_CELLS).sum(axis=1)
ALL_PARTNERS = [lv for lv in cd.KEEP_LEVELS if lv != cd.CD4_MALIGNANT]
CLAIMABLE = [lv for lv in ALL_PARTNERS
             if int(donors_ok.get(lv, 0)) >= cd.MIN_SAMPLES]
REPORT_ONLY = [lv for lv in ALL_PARTNERS if lv not in CLAIMABLE]

gate = pd.DataFrame({"n_cells": counts_ctcl.sum(axis=1),
                     "donors_ge_min_cells": donors_ok}).reindex(cd.CT_ORDER)
gate["verdict"] = np.where(gate.index.isin(CLAIMABLE), "claim", "report_only")
gate.to_csv(cd.tab("headline_claim_gate"))
display(gate)
print(f"claimable ({len(CLAIMABLE)}): {CLAIMABLE}")
print(f"report-only ({len(REPORT_ONLY)}): {REPORT_ONLY}")

bal, _ = C.subsample_levels(ctcl, groupby=cd.GROUPBY,
                            max_per_level=cd.SUBSAMPLE_MAX_PER_LEVEL,
                            max_per_donor_per_level=cd.SUBSAMPLE_MAX_PER_DONOR_PER_LEVEL,
                            donor_key=cd.DONOR_KEY, seed=cd.SUBSAMPLE_SEED, verbose=False)
bal.layers[cd.LAYER] = bal.X
print(f"\nbalanced CTCL window: {bal.n_obs:,} cells")
```

```python  # cell 5
pairs_mal = C.build_groupby_pairs({"mal": ([cd.CD4_MALIGNANT], ALL_PARTNERS)})
pairs_rea = C.build_groupby_pairs({"rea": ([cd.CD4_REACTIVE],
                                           [lv for lv in ALL_PARTNERS if lv != cd.CD4_REACTIVE])})

res_mal = C.run_rank_aggregate(bal, resource, groupby=cd.GROUPBY, groupby_pairs=pairs_mal,
                               expr_prop=cd.EXPR_PROP, min_cells=cd.MIN_CELLS,
                               n_perms=cd.N_PERMS, n_jobs=cd.N_JOBS, seed=cd.SEED,
                               key_added="liana_mal")
res_mal.to_csv(cd.tab("headline_malignant_axes"), index=False)

res_rea = C.run_rank_aggregate(bal, resource, groupby=cd.GROUPBY, groupby_pairs=pairs_rea,
                               expr_prop=cd.EXPR_PROP, min_cells=cd.MIN_CELLS,
                               n_perms=cd.N_PERMS, n_jobs=cd.N_JOBS, seed=cd.SEED,
                               key_added="liana_rea")
res_rea.to_csv(cd.tab("headline_reactive_axes"), index=False)
print(res_mal.shape, res_rea.shape)

cd.FINAL_FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(g, name):
    """plotnine figure -> figures/final/ccc_sub_<name>.svg"""
    out = cd.FINAL_FIG_DIR / f"{cd.FIG_PREFIX}{name}.svg"
    g.save(out, verbose=False)
    print("saved", out.name)
    return g


MYE_OK = [lv for lv in cd.MYELOID_LEVELS if lv in CLAIMABLE]
FIB_OK = [lv for lv in cd.FIBRO_LEVELS if lv in CLAIMABLE]
print("claimable myeloid:", MYE_OK)
print("claimable fibroblast:", FIB_OK)
```

## §1 · Figure 1 — malignant CD4 ↔ myeloid states

What was one dot-plot row in nb36 is now one row per state. Read the two panels together: an edge
present in only one direction is a directional signal; an edge present in both is usually a shared
gene.

```python  # cell 7
save_fig(C.dotplot_axis(liana_res=res_mal, source_labels=[cd.CD4_MALIGNANT], target_labels=MYE_OK,
                        top_n=cd.DOTPLOT_TOP_N, figure_size=(9, 9),
                        title="malignant CD4 -> myeloid states"), "fig1a_malignant_to_myeloid")
```

```python  # cell 8
save_fig(C.dotplot_axis(liana_res=res_mal, source_labels=MYE_OK, target_labels=[cd.CD4_MALIGNANT],
                        top_n=cd.DOTPLOT_TOP_N, figure_size=(9, 9),
                        title="myeloid states -> malignant CD4"), "fig1b_myeloid_to_malignant")
```

## §2 · Figure 2 — malignant CD4 ↔ fibroblast states

```python  # cell 10
save_fig(C.dotplot_axis(liana_res=res_mal, source_labels=[cd.CD4_MALIGNANT], target_labels=FIB_OK,
                        top_n=cd.DOTPLOT_TOP_N, figure_size=(9, 9),
                        title="malignant CD4 -> fibroblast states"), "fig2a_malignant_to_fibro")
```

```python  # cell 11
save_fig(C.dotplot_axis(liana_res=res_mal, source_labels=FIB_OK, target_labels=[cd.CD4_MALIGNANT],
                        top_n=cd.DOTPLOT_TOP_N, figure_size=(9, 9),
                        title="fibroblast states -> malignant CD4"), "fig2b_fibro_to_malignant")
```

## §3 · Figure 3 — the unchanged partners: CD8, B, keratinocyte

Included both because they are part of the roster and because they are the visual form of the
regression test: these panels should look like nb36's.

```python  # cell 13
UNCH_OK = [lv for lv in ["CD8", "B", "Keratinocyte"] if lv in CLAIMABLE]
save_fig(C.dotplot_axis(liana_res=res_mal, source_labels=[cd.CD4_MALIGNANT],
                        target_labels=UNCH_OK, top_n=cd.DOTPLOT_TOP_N,
                        title="malignant CD4 -> CD8 / B / keratinocyte"),
         "fig3a_malignant_to_unchanged")
```

```python  # cell 14
save_fig(C.dotplot_axis(liana_res=res_mal, source_labels=UNCH_OK,
                        target_labels=[cd.CD4_MALIGNANT], top_n=cd.DOTPLOT_TOP_N,
                        title="CD8 / B / keratinocyte -> malignant CD4"),
         "fig3b_unchanged_to_malignant")
```

## §4 · Figure 4 — the whole map, one panel per direction

```python  # cell 16
save_fig(C.dotplot_axis(liana_res=res_mal, source_labels=[cd.CD4_MALIGNANT],
                        target_labels=CLAIMABLE, top_n=30, figure_size=(11, 11),
                        title="malignant CD4 -> every claimable partner"), "fig4a_map_outgoing")
```

```python  # cell 17
save_fig(C.dotplot_axis(liana_res=res_mal, source_labels=CLAIMABLE,
                        target_labels=[cd.CD4_MALIGNANT], top_n=30, figure_size=(11, 11),
                        title="every claimable partner -> malignant CD4"), "fig4b_map_incoming")
```

## §5 · Figure 5 — the comparator and the rank-delta

Nothing in §1–§4 is claimable on its own. A pair earns a claim by ranking better from malignant CD4
than from reactive CD4 **against the same state**.

```python  # cell 19
save_fig(C.dotplot_axis(liana_res=res_rea, source_labels=[cd.CD4_REACTIVE],
                        target_labels=[lv for lv in CLAIMABLE if lv != cd.CD4_REACTIVE],
                        top_n=30, figure_size=(11, 11),
                        title="reactive CD4 -> every claimable partner (THE comparator)"),
         "fig5_comparator")
```

```python  # cell 20
delta_out = C.rank_delta(res_mal.query("source == @cd.CD4_MALIGNANT"),
                         res_rea.query("source == @cd.CD4_REACTIVE"))
delta_out.to_csv(cd.tab("headline_rank_delta_outgoing"), index=False)
delta_in = C.rank_delta(
    res_mal.query("target == @cd.CD4_MALIGNANT").rename(columns={"source": "target",
                                                                 "target": "source"}),
    res_rea.query("target == @cd.CD4_REACTIVE").rename(columns={"source": "target",
                                                                "target": "source"}))
delta_in.to_csv(cd.tab("headline_rank_delta_incoming"), index=False)
print("OUTGOING -- most malignant-shifted:")
display(delta_out.head(25))
print("\nINCOMING -- most shifted toward malignant CD4 as receiver:")
display(delta_in.head(25))
```

```python  # cell 21
# Figure 5b: the delta itself, top shifted pairs per partner state
d = delta_out.head(40).copy()
d["pair"] = d["ligand_complex"] + " -> " + d["receptor_complex"]
fig, ax = plt.subplots(figsize=(7, 0.28 * len(d) + 1.5))
ax.barh(range(len(d)), -d["delta_rank"].to_numpy(), color="#3b6ea5")
ax.set_yticks(range(len(d)))
ax.set_yticklabels((d["target"] + "  |  " + d["pair"]).to_numpy(), fontsize=7)
ax.invert_yaxis()
ax.axvline(0, color="0.4", lw=0.8)
ax.set_xlabel("rank improvement from malignant vs reactive CD4 as sender")
ax.set_title("Pairs the malignant clone sends that reactive CD4 in the same skin does not",
             fontsize=9)
fig.savefig(cd.FINAL_FIG_DIR / f"{cd.FIG_PREFIX}fig5b_rank_delta.svg")
plt.show(); plt.close(fig)
```

## §6 · Figure 6 — the partner heatmap, and where each named edge lands

The panel the pooled v1 map could not produce. The heatmap shows the top interactions across every
claimable state; the landing table below it answers, per named edge, **which state carries it** and
how far ahead of the runner-up.

```python  # cell 23
# partner_heatmap returns a matplotlib Figure (not a plotnine object), so it is saved directly.
hm = C.partner_heatmap(res_mal, sender=cd.CD4_MALIGNANT, partners=CLAIMABLE,
                       top_n=cd.HEATMAP_TOP_N, figsize=(13, 6),
                       title="malignant CD4 <-> claimable partner states")
hm.savefig(cd.FINAL_FIG_DIR / f"{cd.FIG_PREFIX}fig6a_partner_heatmap.svg", bbox_inches="tight")
plt.show()
print(C.coverage_table(bal, groupby=cd.GROUPBY, sample_key=cd.DONOR_KEY,
                       levels=CLAIMABLE).to_string())
print("\nGrey cells = the pair failed min_cells in that stratum. Read them against the coverage "
      "table above, not as biology.")
```

```python  # cell 24
KEYS = ["source", "target", "ligand_complex", "receptor_complex"]


def landing(res, spec, name):
    focal = spec.get("sender", spec.get("receiver"))
    as_sender = "sender" in spec
    ctrls = [(focal, lv, lig, rec) if as_sender else (lv, focal, lig, rec)
             for lv in spec["candidates"] for lig, rec in spec["pairs"]]
    want = pd.DataFrame(C.resolve_controls(ctrls, resource), columns=KEYS)
    got = want.merge(res, on=KEYS, how="left")
    got["level"] = got["target"] if as_sender else got["source"]
    got["test"] = name
    return got.sort_values("magnitude_rank")


land = pd.concat([landing(res_mal, spec, name)
                  for name, spec in cd.RESOLUTION_TESTS.items()], ignore_index=True)
land.to_csv(cd.tab("headline_resolution_tests"), index=False)

scored = land[land["magnitude_rank"].notna()]
if len(scored):
    piv = (scored.groupby(["test", "level"])["magnitude_rank"].min().unstack()
           .reindex(list(cd.RESOLUTION_TESTS)))
    fig, ax = plt.subplots(figsize=(0.55 * piv.shape[1] + 4, 0.5 * len(piv) + 2))
    im = ax.imshow(-np.log10(piv.to_numpy().astype(float) + 1e-6), cmap=cd.HEATMAP_CMAP,
                   aspect="auto")
    im.cmap.set_bad(cd.HEATMAP_BAD)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45, ha="right",
                                                           fontsize=8)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([f"{t}\n{'/'.join(l + '-' + str(r) for l, r in cd.RESOLUTION_TESTS[t]['pairs'][:2])}"
                        for t in piv.index], fontsize=7)
    ax.set_title("Which state carries each named edge  (-log10 magnitude_rank; grey = not scored)",
                 fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(cd.FINAL_FIG_DIR / f"{cd.FIG_PREFIX}fig6b_edge_landing.svg", bbox_inches="tight")
    plt.show(); plt.close(fig)
    display(piv.round(4))
else:
    print("no named edge was scored on any candidate level -- see the coverage table in nb38 §0b")
```

## §7 · Figure 7 — the forced curated panel, and its negatives

Plotted whether or not a pair cleared `expr_prop`, so a negative is reported rather than dropped.
The table underneath separates *absent from the resource* (can never be scored) from *absent from
the data* (a real negative at this depth).

```python  # cell 26
for gene, note in cd.LR_CAVEATS.items():
    print(f"{gene}: {note}\n")
```

```python  # cell 27
panel = C.filter_to_panel(res_mal, panel=cd.LR_PANEL, resource=resource)
panel.to_csv(cd.tab("headline_curated_panel"), index=False)
save_fig(C.dotplot_axis(liana_res=panel, source_labels=[cd.CD4_MALIGNANT],
                        target_labels=CLAIMABLE, top_n=40, figure_size=(11, 10),
                        title="curated panel: malignant CD4 -> partner states"),
         "fig7a_panel_malignant_out")
```

```python  # cell 28
save_fig(C.dotplot_axis(liana_res=panel, source_labels=CLAIMABLE,
                        target_labels=[cd.CD4_MALIGNANT], top_n=40, figure_size=(11, 10),
                        title="curated panel: partner states -> malignant CD4"),
         "fig7b_panel_malignant_in")
```

```python  # cell 29
# the negatives, with numbers: curated pairs NOT scored, and why
pc = C.panel_coverage(resource, panel=cd.LR_PANEL, var_names=bal.var_names)
scored_keys = set(map(tuple, panel[["ligand_complex", "receptor_complex"]].dropna().values))
pc["scored_here"] = [tuple(x) in scored_keys
                     for x in pc[["ligand_complex", "receptor_complex"]].fillna("").values]
neg = pc[~pc["scored_here"]].copy()
neg.to_csv(cd.tab("headline_panel_negatives"), index=False)
print(f"{len(neg)}/{len(pc)} curated pairs were not scored on any claimable axis.")
display(neg)
print("""
Three different reasons, and they are not interchangeable:
  absent from the RESOURCE  -> can never be scored, no matter how well expressed
  genes absent from the DATA-> the LR gene set is the resource subset; a gene outside it was
                               never in the object
  present but below expr_prop in every level -> a real negative AT THIS DEPTH. Th2 cytokine mRNA
                               (IL13, IL4) is poorly captured by 3' 10x, which is exactly why it
                               is in the forced panel -- see LR_CAVEATS.
""")
```

## §8 · Robustness of the headline set

```python  # cell 31
ctrl, caveat_hits = C.control_report(pd.concat([res_mal, res_rea]),
                                     positives=cd.POSITIVE_CONTROLS,
                                     negatives=cd.NEGATIVE_CONTROLS, caveats=cd.LR_CAVEATS,
                                     top_n=40, resource=resource)
ctrl.to_csv(cd.tab("headline_control_report"), index=False)
display(ctrl)
```

```python  # cell 32
spec = pd.DataFrame(C.resolve_controls(cd.SPEC_MUST_HAVES, resource),
                    columns=["source", "target", "ligand_complex", "receptor_complex"])
print("replication-spec must-haves (untouched B level -- the regression test):")
display(ctrl.merge(spec, on=list(spec.columns)))
```

```python  # cell 33
_, overlap = C.shuffle_control(bal, resource, groupby=cd.GROUPBY, within=cd.DONOR_KEY,
                               n_shuffles=cd.N_SHUFFLES, top_n=cd.TOP_N, seed=cd.SEED,
                               reference=res_mal, groupby_pairs=pairs_mal,
                               min_cells=cd.MIN_CELLS, n_perms=cd.N_PERMS, verbose=False)
overlap.to_csv(cd.tab("headline_shuffle_control"), index=False)
print("mean overlap with the real top-20:", overlap["overlap_with_real"].mean(), "-- target <= 2")
```

```python  # cell 34
# Evidence-strength survival: restrict malignant CD4 to the cells that ALSO carry a CNV call.
# A NESTED subset of CD4_malignant under the ALICE primary -- it asks whether the tcr_only cells
# drive anything, not whether a different definition would.
both = bal[(bal.obs[cd.GROUPBY].astype(str) != cd.CD4_MALIGNANT)
           | (bal.obs[cd.EVIDENCE_SRC].astype(str) == "both")].copy()
both.layers[cd.LAYER] = both.X
is_mal = bal.obs[cd.GROUPBY].astype(str) == cd.CD4_MALIGNANT
ev_both = bal.obs[cd.EVIDENCE_SRC].astype(str) == "both"
print(f"malignant CD4 in the subsample: {int(is_mal.sum())} -> evidence=='both': "
      f"{int((is_mal & ev_both).sum())}")
res_both = C.run_rank_aggregate(both, resource, groupby=cd.GROUPBY, groupby_pairs=pairs_mal,
                                min_cells=cd.MIN_CELLS, n_perms=cd.N_PERMS,
                                key_added="liana_both", verbose=False)
res_both.to_csv(cd.tab("headline_evidence_both"), index=False)
display(C.top_n_overlap_matrix({cd.MALIG_SRC: res_mal, "evidence_both": res_both},
                               top_n=cd.TOP_N))

KEYS = ["source", "target", "ligand_complex", "receptor_complex"]
surv = (res_mal.sort_values("magnitude_rank").head(40)[KEYS + ["magnitude_rank"]]
        .merge(res_both[KEYS + ["magnitude_rank"]], on=KEYS, how="left",
               suffixes=("_alice", "_both")))
surv["survives_cnv_restriction"] = surv["magnitude_rank_both"].notna()
surv.to_csv(cd.tab("headline_evidence_both_survival"), index=False)
display(surv)
```

## §9 · The reportable set

One table, one row per edge that survives every gate. This is what may be written up.

```python  # cell 36
KEYS = ["source", "target", "ligand_complex", "receptor_complex"]

rep = res_mal.copy()
rep["partner"] = np.where(rep.source == cd.CD4_MALIGNANT, rep.target, rep.source)
rep["direction"] = np.where(rep.source == cd.CD4_MALIGNANT, "malignant_out", "malignant_in")

# gate 1: the partner level is claimable
rep = rep[rep["partner"].isin(CLAIMABLE)]
# gate 2: beats the reactive comparator on the same partner and the same pair
d = pd.concat([delta_out.assign(direction="malignant_out"),
               delta_in.assign(direction="malignant_in")], ignore_index=True)
dk = [c for c in ["ligand_complex", "receptor_complex", "target", "direction"] if c in d.columns]
rep = rep.merge(d[dk + [c for c in ["delta_rank", "malignant_only"] if c in d.columns]]
                .rename(columns={"target": "partner"}),
                on=["ligand_complex", "receptor_complex", "partner", "direction"], how="left")
rep["beats_comparator"] = (rep["delta_rank"].fillna(0) < 0) | rep["malignant_only"].fillna(False)
# gate 3: not a known null-model failure
rep["caveated"] = rep[["ligand_complex", "receptor_complex"]].apply(
    lambda r: any(g in str(r.iloc[0]).split("_") + str(r.iloc[1]).split("_")
                  for g in cd.LR_CAVEATS), axis=1)
# gate 4: survives the evidence=='both' restriction
both_keys = set(map(tuple, res_both[KEYS].values))
rep["survives_evidence_both"] = [tuple(x) in both_keys for x in rep[KEYS].values]
# gate 5: significant under the (descriptive) permutation p
rep["p_ok"] = rep["cellphone_pvals"] < cd.PVAL_ALPHA

rep["reportable"] = (rep["beats_comparator"] & ~rep["caveated"]
                     & rep["survives_evidence_both"] & rep["p_ok"])
rep = rep.sort_values(["reportable", "magnitude_rank"], ascending=[False, True])
rep.to_csv(cd.tab("reportable_set"), index=False)

print(f"{int(rep['reportable'].sum())} of {len(rep)} scored edges pass every gate.")
display(rep[rep["reportable"]][
    ["direction", "partner", "ligand_complex", "receptor_complex", "lr_means",
     "magnitude_rank", "specificity_rank", "cellphone_pvals", "delta_rank"]].head(40))
print("\ndropped by each gate:")
display(pd.Series({
    "fails comparator": int((~rep["beats_comparator"]).sum()),
    "caveated ligand/receptor": int(rep["caveated"].sum()),
    "lost under evidence=='both'": int((~rep["survives_evidence_both"]).sum()),
    "p >= alpha": int((~rep["p_ok"]).sum()),
}).rename("n_edges").to_frame())
print("\n" + cd.CAVEAT_BLOCK)
```

### Outcome, and what is deliberately absent

`tables/ccc_sub_reportable_set.csv` is the deliverable; the figures are in
`figures/final/ccc_sub_*.svg`.

**The question this set answers that nb36's could not**: for each malignant-CD4 ↔ TME edge, *which
myeloid or fibroblast state carries it*. §6b is the panel that says so. Three outcomes are all
results — the edge concentrates on the expected state, it concentrates on a different one (the most
interesting case, and a correction to the v1 attribution), or it is uniform across states, in which
case the pooled v1 level was not hiding anything for that edge and the split buys nothing there.

**Absent by design.**
- *Donor-level statistics.* Every p-value here is LIANA's cell-unit permutation p, pseudoreplicated
  across donors. It sizes dots and gates §9; it is not inference. The donor-level phase needs
  `ccc_skin_pseudobulk_full.parquet` rebuilt for these levels.
- *Stage / disease / layer contrasts.* `ccc_data.FORBIDDEN_CONTRASTS`, unchanged.
- *Vascular, Tregs, Plasma, Mast, Melanocyte, CD4_unassessed.* Out of scope for this roster; their
  nb35/36 results stand and are not superseded.
- *A claim that the sub-level map replaces the pooled one.* It does not. §12 of nb38 exists so that
  every sub-level result is read against its v1 pooled counterpart — a level split k ways carries
  fewer cells and fewer donors, and a weaker rank can be arithmetic rather than biology.
