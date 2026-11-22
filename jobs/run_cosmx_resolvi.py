#!/usr/bin/env python
"""resolVI on the CosMx early-MF layer (GSE337336) for nb28e.

Segmentation on CosMx assigns a neighbour's transcripts to a cell, so an intraepidermal T cell
reads keratin and a fibroblast touching a T cell reads CD3E (28d F4 spillover band). resolVI
models each cell's counts as its own expression + diffusion from its spatial neighbours +
background, and decodes the first term. nb28e uses it for three things:
  1. re-calling lineages, above all the keratinocyte-called cells InSituType calls T (the
     candidate intraepidermal T cells), via the semi-supervised classifier;
  2. corrected expression of proteases in T cells vs stroma (F7) and of motility / retention
     genes in epidermal vs dermal T cells (F9), free of neighbour spillover;
  3. a latent space for the piece-specificity check of the malignancy transfer score (F8).

Inputs : data/S1_cosmx_earlymf_spatial/processed/cosmx_s1.h5ad (layer counts)
         data/S1_cosmx_earlymf_spatial/processed/cosmx_s1_cells_28d.parquet (lineage from 28f §P, formerly 28d §1)
Outputs: data/S1_cosmx_earlymf_spatial/processed/cosmx_s1_resolvi.h5ad
         (--export-only: loads the saved model, writes every panel gene at library 1e4 to
          processed/cosmx_s1_resolvi_all.h5ad for nb28f)
           X = corrected expression (library-normalised to the median nCount) for OUT_GENES,
           obs = lineage_28d, lab_resolvi (training label), lineage_resolvi + p_<lineage>,
           obsm X_resolvi = latent
         data/S1_cosmx_earlymf_spatial/processed/resolvi_model/
Labels: 28d lineage, with Unassigned and keratinocyte-called cells that InSituType calls T.cell
set to unlabeled, so the classifier -- not the Leiden call -- decides them.
Spatial neighbours are built within slide (batch_key=slide); slides share no coordinates.
"""
import argparse
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
NB_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NB_DIR / "helpers"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cosmx_spatial as CS  # noqa: E402
import degradome_data as cfg  # noqa: E402

MODEL_DIR = CS.S1_DIR / "processed" / "resolvi_model"
RESOLVI_ALL = CS.S1_DIR / "processed" / "cosmx_s1_resolvi_all.h5ad"
PROTEASE_FAMS = ["mmp", "cathepsin", "granzyme", "serine_other", "metallo_other", "inhibitor"]
EXTRA_GENES = (CS.RETENTION + CS.EGRESS + CS.LAMINA + CS.MALIG_APRIORI + CS.T_SPILL + CS.KER_SPILL
               + [g for pair in CS.AXIS_PAIRS for g in pair]
               + ["CD4", "CD8A", "CD8B", "AQP3", "FN1", "IL4R", "CD44", "VIM", "PFN1", "RAC2",
                  "ITGB1", "ITGB2", "ITGAL", "EZR", "S100A4", "COL1A1", "DCN", "CD68", "KRT10", "KRT1"])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--n-latent", type=int, default=10)
    p.add_argument("--n-samples", type=int, default=20, help="posterior samples for expression")
    p.add_argument("--force", action="store_true")
    p.add_argument("--export-only", action="store_true",
                   help="load the saved model; write every panel gene at library 1e4 to RESOLVI_ALL (nb28f)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.export_only and RESOLVI_ALL.exists() and not args.force:
        print(f"{RESOLVI_ALL} exists; --force to rebuild")
        return
    if not args.export_only and CS.RESOLVI_H5AD.exists() and not args.force:
        print(f"{CS.RESOLVI_H5AD} exists; --force to rebuild")
        return
    import scvi
    from scvi.external import RESOLVI

    scvi.settings.seed = 0
    t0 = time.time()
    adata = CS.load_s1(cache=True, verbose=False)
    cells = pd.read_parquet(CS.CELLS_28D)
    assert cells.index.equals(adata.obs_names), "28d cell table and cosmx_s1.h5ad differ; re-run 28f §P (REBUILD_LINEAGE = True)"
    adata.obs["lineage_28d"] = cells.lineage.astype(str).to_numpy()
    adata.X = adata.layers["counts"].astype(np.float32)
    keep = np.asarray(adata.layers["counts"].sum(1)).ravel() >= 5         # resolVI requirement
    adata = adata[keep].copy()
    print(f"cells: {adata.n_obs:,} (dropped {int((~keep).sum())} with < 5 counts) × {adata.n_vars} genes")

    lin = adata.obs.lineage_28d.to_numpy()
    kc_t = (lin == "Keratinocyte") & (adata.obs.insitutype.astype(str).to_numpy() == "T.cell")
    lab = np.where((lin == "Unassigned") | kc_t, "unknown", lin)
    adata.obs["lab_resolvi"] = pd.Categorical(lab)
    print("training labels:", pd.Series(lab).value_counts().to_dict())
    print(f"keratinocyte-called, InSituType T.cell -> unlabeled: {int(kc_t.sum()):,}")

    RESOLVI.setup_anndata(adata, layer="counts", batch_key="slide", labels_key="lab_resolvi",
                          unlabeled_category="unknown",
                          prepare_data_kwargs={"spatial_rep": "spatial", "n_neighbors": 10})
    if args.export_only:
        model = RESOLVI.load(str(MODEL_DIR), adata=adata)
        print(f"loaded {MODEL_DIR} ({time.time() - t0:.0f}s)")
    else:
        model = RESOLVI(adata, n_latent=args.n_latent, semisupervised=True)
        print(model)
        model.train(max_epochs=args.epochs)
        model.save(str(MODEL_DIR), overwrite=True)
        print(f"trained + saved in {time.time() - t0:.0f}s -> {MODEL_DIR}")

    Z = model.get_latent_representation()
    soft = model.predict(soft=True)
    hard = soft.idxmax(1).to_numpy()
    genes = [g for f in PROTEASE_FAMS for g in cfg.DEGRADOME_FAMILIES[f]]
    genes += [g for gs in CS.LINEAGE_MARKERS.values() for g in gs] + EXTRA_GENES
    genes = [g for g in dict.fromkeys(genes) if g in adata.var_names]
    lib = float(np.median(adata.obs.nCount_RNA))
    out_path = CS.RESOLVI_H5AD
    if args.export_only:
        genes, lib, out_path = list(adata.var_names), 1e4, RESOLVI_ALL
    E = model.get_normalized_expression(gene_list=genes, library_size=lib, n_samples=args.n_samples,
                                        return_mean=True)
    E = E[genes]

    import anndata as ad

    obs = adata.obs[["slide", "piece", "fov_key", "lineage_28d", "lab_resolvi", "insitutype"]].copy()
    obs["lineage_resolvi"] = pd.Categorical(hard)
    for c in soft.columns:
        obs[f"p_{c}"] = soft[c].to_numpy().astype(np.float32)
    out = ad.AnnData(X=E.to_numpy(np.float32), obs=obs, var=pd.DataFrame(index=pd.Index(genes)))
    out.obsm["X_resolvi"] = Z.astype(np.float32)
    out.uns["note"] = (f"resolVI corrected expression at library {lib:.0f}; semisupervised on 28d "
                       f"lineage; n_latent {args.n_latent}; {'nb28f (all genes)' if args.export_only else 'nb28e'} input")
    out.write_h5ad(out_path, compression="gzip")
    print(f"wrote {out_path}  {out.shape}")

    print("\n28d lineage × resolVI lineage:")
    print(pd.crosstab(obs.lineage_28d, obs.lineage_resolvi))
    print("\nkeratinocyte-called / InSituType T.cell cells, resolVI call:",
          pd.Series(hard[kc_t]).value_counts().to_dict())
    for g in ["MMP2", "MMP9", "MMP14", "CD3E", "KRT14"]:
        if g in E:
            print(f"median corrected {g} by resolVI lineage:",
                  E[g].groupby(hard).median().round(3).to_dict())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
