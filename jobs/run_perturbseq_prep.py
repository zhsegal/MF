#!/usr/bin/env python
"""Compact the GWCD4i DE estimates to the one condition the MF notebook uses.

Reads `data/external_ref/perturbseq_gwcd4i/GWCD4i.DE_stats.h5ad` (15.6 GB; 33,983 perturbation x
condition rows x 10,282 genes, dense DESeq2 layers) with h5py only, keeps the rows of one culture
condition (default Stim48hr) and writes `GWCD4i_DE_<condition>.h5ad` (~2 GB) with float32 layers:

    zscore          log_fc / lfcSE as shipped, clipped at |100| (the authors clip at 100)
    zscore_masked   the same with the on-target (cis) entry of every regulator replaced by that
                    regulator's column mean -- the authors' `mask_cis_effects`; this is the model input
    log_fc, lfcSE, adj_p_value, baseMean   as shipped

obs = the shipped per-perturbation QC (joined with `n_total_de_genes` etc. from
DE_stats.suppl_table.csv); var = gene symbol + Ensembl id + `mean_baseMean_<cond>` for every
condition (the authors' gene filter needs all three). Also writes tables/pseq29_perturbseq_qc.csv.

Run through run_perturbseq_prep.sh (bsub); ~40 GB RAM peak, ~15 min.
"""
import argparse, json, sys, time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import anndata as ad
from anndata.io import read_elem

ROOT = Path(__file__).resolve().parent.parent
PSEQ = ROOT / "data" / "external_ref" / "perturbseq_gwcd4i"
CONDS = ("Rest", "Stim8hr", "Stim48hr")
LAYERS = ("log_fc", "lfcSE", "adj_p_value", "baseMean", "zscore")
CHUNK = 2000


def read_rows(ds, rows, chunk=CHUNK):
    """Rows `rows` (sorted int array) of a dense 2-D h5py dataset, read in contiguous blocks."""
    out = np.empty((len(rows), ds.shape[1]), dtype=np.float32)
    rows = np.asarray(rows)
    pos = 0
    for start in range(0, ds.shape[0], chunk):
        stop = min(start + chunk, ds.shape[0])
        sel = rows[(rows >= start) & (rows < stop)]
        if sel.size:
            block = ds[start:stop]
            out[pos:pos + sel.size] = block[sel - start]
            pos += sel.size
    assert pos == len(rows)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", default="Stim48hr", choices=CONDS)
    p.add_argument("--src", default=str(PSEQ / "GWCD4i.DE_stats.h5ad"))
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = Path(a.out) if a.out else PSEQ / f"GWCD4i_DE_{a.condition}.h5ad"
    t0 = time.time()

    with h5py.File(a.src, "r") as f:
        obs = read_elem(f["obs"])
        var = read_elem(f["var"])
        print(f"source obs {obs.shape}, var {var.shape}; layers {list(f['layers'].keys())}", flush=True)
        for L in LAYERS:
            g = f["layers"][L]
            if isinstance(g, h5py.Group):
                sys.exit(f"layer {L} is sparse ({dict(g.attrs)}) -- extend read_rows for CSR before use")
        cond = obs["culture_condition"].astype(str).to_numpy()
        rows = np.where(cond == a.condition)[0]
        print(f"{a.condition}: {rows.size:,} regulators", flush=True)

        # per-condition mean baseMean over all perturbations of that condition (gene filter)
        for c in CONDS:
            r = np.where(cond == c)[0]
            bm = read_rows(f["layers"]["baseMean"], r)
            var[f"mean_baseMean_{c}"] = np.nanmean(bm, axis=0)
            var[f"n_regulators_{c}"] = r.size
            del bm
            print(f"  baseMean {c}: done ({time.time() - t0:.0f}s)", flush=True)

        layers = {}
        for L in LAYERS:
            layers[L] = read_rows(f["layers"][L], rows)
            print(f"  layer {L}: {layers[L].shape} ({time.time() - t0:.0f}s)", flush=True)
        varm = {k: read_elem(f["varm"][k]) for k in f["varm"].keys()} if "varm" in f else {}

    obs = obs.iloc[rows].copy()
    obs.index = obs.index.astype(str)
    if "gene_name" in var.columns:
        gene_name = var["gene_name"].astype(str).to_numpy()
    else:
        gene_name = var.index.astype(str).to_numpy()
    var["ensembl_id"] = var["gene_ids"].astype(str).to_numpy() if "gene_ids" in var.columns else var.index.astype(str)
    var.index = pd.Index(gene_name, name="gene_name")
    if not var.index.is_unique:
        dup = var.index.duplicated(keep=False)
        print(f"  {dup.sum()} duplicated gene symbols -> suffixed", flush=True)
        var.index = ad.utils.make_index_unique(var.index)

    # z-score: clip at |100| (authors clip at 100), then the cis mask
    z = np.clip(np.nan_to_num(layers["zscore"], nan=0.0), -100, 100).astype(np.float32)
    layers["zscore"] = z
    zm = z.copy()
    reg = obs["target_contrast_gene_name"].astype(str).to_numpy()
    col = {g: i for i, g in enumerate(var.index)}
    n_cis = 0
    for i, g in enumerate(reg):
        j = col.get(g)
        if j is not None:
            zm[i, j] = np.nan
            n_cis += 1
    colmean = np.nanmean(zm, axis=0)
    ii, jj = np.where(np.isnan(zm))
    zm[ii, jj] = colmean[jj]
    layers["zscore_masked"] = zm
    print(f"  cis entries masked: {n_cis:,} of {len(reg):,} regulators measured as genes", flush=True)

    # join the tabular QC (n_total_de_genes drives the authors' regulator filter)
    qc_csv = PSEQ / "suppl_tables" / "DE_stats.suppl_table.csv"
    if qc_csv.exists():
        qc = pd.read_csv(qc_csv, index_col=0)
        qc.index = qc.index.astype(str)
        extra = [c for c in qc.columns if c not in obs.columns]
        obs = obs.join(qc[extra], how="left")
        print(f"  joined {len(extra)} QC columns from DE_stats.suppl_table.csv", flush=True)
    obs["regulator"] = reg

    a_out = ad.AnnData(X=None, obs=obs, var=var, layers=layers, shape=(len(obs), len(var)))
    for k, v in varm.items():
        try:
            if hasattr(v, "index") and len(v) == a_out.n_vars:
                v = v.copy(); v.index = a_out.var_names
            a_out.varm[k] = v
        except Exception as e:  # varm frames are optional
            print(f"  skipped varm {k}: {type(e).__name__}")
    a_out.uns["source"] = str(a.src)
    a_out.uns["condition"] = a.condition
    a_out.uns["prep"] = json.dumps({"script": "jobs/run_perturbseq_prep.py", "clip_z": 100,
                                    "cis_mask": "column mean (authors' mask_cis_effects)"})
    out.parent.mkdir(parents=True, exist_ok=True)
    a_out.write_h5ad(out, compression="gzip")
    print(f"wrote {out} {a_out.shape} ({time.time() - t0:.0f}s)", flush=True)

    # QC table: regulators per tier
    tab = ROOT / "tables"; tab.mkdir(exist_ok=True)
    o = a_out.obs
    tiers = {
        "tested": np.ones(len(o), bool),
        "n_total_de_genes>=10 (authors' model filter)": (o["n_total_de_genes"] >= 10).to_numpy() if "n_total_de_genes" in o else np.zeros(len(o), bool),
        ">75 DE genes & >50 cells (authors' clustering tier)": ((o.get("n_total_de_genes", 0) > 75) & (o.get("n_cells_target", 0) > 50)).to_numpy(),
        "ontarget_significant": o["ontarget_significant"].astype(bool).to_numpy() if "ontarget_significant" in o else np.zeros(len(o), bool),
        "distal_offtarget_flag": o["distal_offtarget_flag"].astype(bool).to_numpy() if "distal_offtarget_flag" in o else np.zeros(len(o), bool),
    }
    qc_tab = pd.DataFrame({"condition": a.condition, "tier": list(tiers), "n_regulators": [int(v.sum()) for v in tiers.values()]})
    qc_tab.loc[len(qc_tab)] = [a.condition, "measured genes", a_out.n_vars]
    qc_tab.loc[len(qc_tab)] = [a.condition, f"genes with mean baseMean>0.1 in all 3 conditions",
                               int((var[[f"mean_baseMean_{c}" for c in CONDS]] > 0.1).all(axis=1).sum())]
    qc_tab.to_csv(tab / "pseq29_perturbseq_qc.csv", index=False)
    print(qc_tab.to_string(index=False))


if __name__ == "__main__":
    main()
