"""Shared machinery for the hand-annotation notebooks (22, 22b).

`22_myeloid_fibro_reannotation` and `22b_unassigned_reannotation` run the same loop: embed a
subset, Leiden it, dump a numeric block per cluster, hand-fill a cluster -> label dict, then
cache a sidecar keyed on `cell_id`. Everything in that loop that is not the biology lives here,
so the two notebooks differ only in their marker panels and their vocabulary.

The one rule worth stating: every hand-filled map is locked to the clustering it was authored
against by a cluster-size dict (`check_map`). Leiden is not stable across scanpy/igraph
versions or across a re-embedding, and a map whose keys still resolve but whose clusters have
moved relabels cells silently. The lock turns that into an AssertionError.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted

NB_DIR = Path(__file__).resolve().parent.parent
ATLAS_JOINT = NB_DIR / "data" / "atlas_joint"


# ================================================================ panels


def present_panels(var_names, panels, label="", verbose=True):
    """Filter each panel to genes in `var_names`, preserving order, reporting what was dropped.

    `subclone_helpers.present_panels` does the same filtering silently. Here the drop is the
    point: these notebooks run on a 10k-HVG matrix, and a marker panel that quietly loses its
    defining genes produces a cluster that cannot be called rather than an error.
    """
    vs = set(map(str, var_names))
    keep, dropped = {}, {}
    for cat, genes in panels.items():
        seen, ok, no = set(), [], []
        for g in genes:
            if g in seen:
                continue
            seen.add(g)
            (ok if g in vs else no).append(g)
        if ok:
            keep[cat] = ok
        if no:
            dropped[cat] = no
    if verbose and dropped:
        head = f"{label}: " if label else ""
        for cat, no in dropped.items():
            print(f"  {head}dropped from {cat}: {no}")
        gone = [c for c in panels if c not in keep]
        if gone:
            print(f"  {head}!! panels with NO genes left: {gone}")
    return keep, dropped


# ================================================================ coverage


def coverage(obs, col, donor_key="donor", min_cells=25, min_samples=5):
    """Per-level donor coverage: the claim gate, as a table rather than a boolean.

    A level that clears `min_cells` in fewer than `min_samples` donors cannot carry a claim.
    Returning the table rather than the verdict is deliberate -- the notebooks report levels
    that fail as explicit abstentions, which is not the same as dropping them.
    """
    ct = pd.crosstab(obs[col].astype(str), obs[donor_key].astype(str))
    out = pd.DataFrame({
        "n_cells": ct.sum(1),
        "n_donors_present": (ct > 0).sum(1),
        f"n_donors_ge{min_cells}": (ct >= min_cells).sum(1),
    })
    out["claimable"] = out[f"n_donors_ge{min_cells}"] >= min_samples
    return out.sort_values(f"n_donors_ge{min_cells}", ascending=False)


# ================================================================ the numeric dump


def cluster_qc(adata, key, donor_key="donor", study_key="study"):
    """Size / donor spread / study dominance / depth+mito+doublet medians, per cluster.

    `top_study_frac` is the column that matters: a cluster that is 98% one study is a batch
    artefact until proven otherwise, and no marker panel will tell you that.
    """
    obs = adata.obs
    g = obs[key].astype(str)
    rows = {}
    for c in natsorted(g.unique()):
        m = (g == c).to_numpy()
        sub = obs.loc[m]
        vc = sub[study_key].astype(str).value_counts()
        rows[c] = {
            "n": int(m.sum()),
            "n_donors": int(sub[donor_key].astype(str).nunique()),
            "n_studies": int(len(vc)),
            "top_study": vc.index[0] if len(vc) else "",
            "top_study_frac": round(float(vc.iloc[0] / m.sum()), 2) if len(vc) else np.nan,
            "med_genes": int(np.median(sub["n_genes"])) if "n_genes" in sub else -1,
            "med_mito": round(float(np.median(sub["pct_mito"])), 2) if "pct_mito" in sub else np.nan,
            "med_doublet": (round(float(np.median(sub["doublet_score"])), 3)
                            if "doublet_score" in sub else np.nan),
        }
    return pd.DataFrame(rows).T


def cluster_expression(adata, key, genes):
    """(mean log1p, detected fraction) per cluster x gene, both rounded to 2dp.

    Densifies only the `genes` columns -- the full matrix is 10k wide and does not need to be
    in memory for this.
    """
    genes = list(dict.fromkeys(g for g in genes if g in set(map(str, adata.var_names))))
    X = adata[:, genes].X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    grp = adata.obs[key].astype(str).to_numpy()
    mean = pd.DataFrame(X, columns=genes).groupby(grp).mean()
    frac = pd.DataFrame((X > 0).astype(np.float32), columns=genes).groupby(grp).mean()
    order = natsorted(mean.index)
    return mean.reindex(order).round(2), frac.reindex(order).round(2)


def paste_block(adata, key, panels, mean, frac, qc, vocab, title, extra_cols=(), top_n=None):
    """The copy-into-an-LLM block: every cluster's numbers, nothing thresholded away.

    A marker that is absent prints as `0.00/0.00`. That is evidence, and hiding it behind a
    filter is how a cluster gets called on the two genes that happened to survive.
    """
    lines = ["=" * 100,
             f"{title} — {adata.n_obs:,} cells, {len(mean)} clusters; "
             f"values are `mean log1p CP10K / fraction detected`",
             f"allowed labels: {vocab}",
             "=" * 100]
    for c in mean.index:
        r = qc.loc[c]
        lines.append(
            f"\n--- cluster {c}  n={int(r['n']):,}  donors={int(r['n_donors'])}  "
            f"studies={int(r['n_studies'])}  top_study={r['top_study']}({r['top_study_frac']})  "
            f"med_genes={int(r['med_genes'])}  med_mito={r['med_mito']}  "
            f"doublet={r['med_doublet']}")
        for col in extra_cols:
            if col in qc.columns:
                lines.append(f"    {col}: {r[col]}")
        for cat, genes in panels.items():
            cells = "  ".join(f"{g} {mean.loc[c, g]:.2f}/{frac.loc[c, g]:.2f}"
                              for g in genes if g in mean.columns)
            if cells:
                lines.append(f"    {cat:16s} {cells}")
        if top_n is not None and c in top_n:
            lines.append(f"    {'wilcoxon top':16s} {top_n[c]}")
    return "\n".join(lines)


# ================================================================ the hand-filled map


def check_map(adata, key, mapping, sizes, vocab, label, out_col=None):
    """Validate a hand-filled cluster -> label dict against the clustering it was authored on.

    Returns (ok, current_sizes). When `mapping` is empty or has any "" value it prints a
    paste-ready template and returns ok=False, so the notebook halts at the hand-filled step
    instead of running past it on a half-filled map.

    The `sizes` lock is the part that matters -- see the module docstring.
    """
    cats = [str(c) for c in adata.obs[key].astype("category").cat.categories]
    g = adata.obs[key].astype(str)
    now = {c: int((g == c).sum()) for c in cats}

    if (not mapping) or any(v == "" for v in mapping.values()):
        print(f"--- {label}: UNFILLED. Paste as the map: ---")
        print("    {" + ", ".join(f'"{c}": ""' for c in cats) + "}")
        print(f"--- and as the size lock: ---\n    {now!r}\n")
        return False, now

    missing, extra = natsorted(set(cats) - set(mapping)), natsorted(set(mapping) - set(cats))
    assert not missing and not extra, (
        f"{label}: map keys != clusters\n  missing: {missing}\n  extra  : {extra}")
    bad = sorted(set(mapping.values()) - set(vocab))
    assert not bad, f"{label}: labels outside the vocabulary: {bad}"
    if sizes:
        assert now == {str(k): int(v) for k, v in sizes.items()}, (
            f"{label}: the clustering changed since this map was authored — cluster IDs no "
            f"longer mean what the map says.\n  authored: {sizes}\n  now: {now}")
    else:
        print(f"!! {label}: size lock is empty — paste {now!r}")

    if out_col is not None:
        adata.obs[out_col] = pd.Categorical(g.map(mapping))
    return True, now


# ================================================================ plots


def save_umap(adata, color, path, title=None, size=3, legend_loc="on data", **kw):
    """UMAP saved at 200 dpi with the point collections rasterized.

    Rasterizing matters: these are 150k-330k point scatters, and a vector PDF of one is
    unopenable.
    """
    import matplotlib.pyplot as plt
    import scanpy as sc

    fig = sc.pl.umap(adata, color=color, frameon=False, size=size, legend_loc=legend_loc,
                     legend_fontsize=7, show=False, return_fig=True, title=title, **kw)
    for ax in fig.axes:
        for coll in ax.collections:
            coll.set_rasterized(True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return fig


def save_dotplot(adata, panels, groupby, path, title=None, figsize=(16, 5), **kw):
    import matplotlib.pyplot as plt
    import scanpy as sc

    dp = sc.pl.dotplot(adata, panels, groupby=groupby, standard_scale="var",
                       dendrogram=True, figsize=figsize, return_fig=True, title=title, **kw)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dp.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close("all")
    return dp


# ================================================================ sidecar


def write_sidecar(frame, csv_path, prov_path, provenance, expect_rows=None, id_col="cell_id"):
    """Write a cell-level sidecar + its provenance json, asserting the contract first."""
    assert frame[id_col].is_unique, f"duplicate {id_col} in the sidecar"
    if expect_rows is not None:
        assert len(frame) == expect_rows, (len(frame), expect_rows)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    provenance = dict(provenance)
    provenance.setdefault("written_utc",
                          datetime.now(timezone.utc).isoformat(timespec="seconds"))
    provenance.setdefault("n_rows", int(len(frame)))
    Path(prov_path).write_text(json.dumps(provenance, indent=2, default=str))
    print(f"wrote {csv_path}  {frame.shape}")
    print(f"wrote {prov_path}")
    return frame


def reconcile(frame, expect_counts, obs_parquet=None, id_col="cell_id",
              label_col="subtype_fine", where=None, verbose=True):
    """Join a sidecar back onto the atlas obs and assert it lands where it claims to.

    `expect_counts` is {label: n}. This is the contract downstream configs freeze as their
    roster; asserting it at write time is what stops a re-run silently changing the substrate
    of an analysis that was already reported.
    """
    import polars as pl

    obs_parquet = ATLAS_JOINT / "atlas_obs_full_v2.parquet" if obs_parquet is None else obs_parquet
    cols = [id_col, "compartment", "cell_type_final"]
    obs = pl.read_parquet(obs_parquet, columns=cols).to_pandas()
    obs[id_col] = obs[id_col].astype(str)

    f = frame.copy()
    f[id_col] = f[id_col].astype(str)
    j = f.merge(obs, on=id_col, how="left", validate="one_to_one")
    n_unmatched = int(j["compartment"].isna().sum())
    assert n_unmatched == 0, f"{n_unmatched} sidecar rows are not in {Path(obs_parquet).name}"

    if where is not None:
        tgt = obs.query(where)
        assert set(f[id_col]) == set(tgt[id_col].astype(str)), (
            f"sidecar is not exactly `{where}`: "
            f"sidecar {len(f):,} vs target {len(tgt):,}, "
            f"{len(set(f[id_col]) - set(tgt[id_col].astype(str))):,} extra / "
            f"{len(set(tgt[id_col].astype(str)) - set(f[id_col])):,} missing")

    got = f[label_col].astype(str).value_counts().to_dict()
    bad = {k: (expect_counts[k], got.get(k, 0)) for k in expect_counts
           if got.get(k, 0) != expect_counts[k]}
    assert not bad, f"level counts moved (expected, got): {bad}"

    if verbose:
        print(f"reconciled {len(j):,} rows against {Path(obs_parquet).name}; "
              f"{len(expect_counts)} level counts match")
    return j
