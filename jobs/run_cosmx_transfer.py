#!/usr/bin/env python
"""scRNA → CosMx malignancy transfer model for nb28e.

CosMx has no TCR and misses the clone markers (KIR3DL2, PLS3, CCR4, CD7, TNFRSF8), so a T cell
cannot be called malignant in situ. This fits, on skin CD4 T cells with a 26b cluster call
(`mal_cluster`, skin_T_malignancy_cluster_v1.parquet), a logistic model restricted to what
CosMx can read, and measures how well it separates held-out donors *at CosMx depth*:

  genes    : panel genes whose CosMx signal in T cells is not spillover -- T-cell mean >= half
             the top lineage's mean (28d lineages), T detection >= 5% -- minus TR/IG/MT/RP,
             CD8A/B and merged probes ("TPSAB1/B2"). No single keratinocyte/fibroblast gene can
             then drive the score of an intraepidermal T cell.
  depth    : scRNA counts over those panel genes are binomially thinned so each cell's library
             is drawn from the CosMx T-cell nCount distribution.
  features : log1p(count * median_T_lib / lib), z-scored WITHIN platform (train donors on
             scRNA; CosMx T cells in nb28e). Held-out folds are z-scored on their own cells,
             which is what CosMx scoring does, so the CV includes that step.
  CV       : 5-fold GroupKFold on donor; AUC overall, per donor (where both classes exist) and
             early stage only. nb28e only uses the score when held-out AUC >= 0.75.

The score is relative (within-platform z), so nb28e uses it as tertiles for group contrasts,
never as a per-cell call.

Outputs: tables/deg28e_malig_transfer_cv.csv, tables/deg28e_malig_transfer_coef.csv,
         data/S1_cosmx_earlymf_spatial/processed/malig_transfer_model.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("COSMX_TRANSFER_THREADS", "4"))
NB_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NB_DIR / "helpers"))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sparse  # noqa: E402

import cluster_malig_helpers as CM  # noqa: E402
import cosmx_spatial as CS  # noqa: E402

MODEL_JSON = CS.S1_DIR / "processed" / "malig_transfer_model.json"
CV_CSV = NB_DIR / "tables" / "deg28e_malig_transfer_cv.csv"
COEF_CSV = NB_DIR / "tables" / "deg28e_malig_transfer_coef.csv"
CALLED = {"tcr", "auto", "user"}          # call_source values that carry a real call
# post-re-annotation vocabulary; "CD4" is the malignant cells, which keep the old label
CD4_TYPES = {"CD4", "CD4_conv", "CD4_naive", "CD4_Treg"}
MIN_T_DET, MIN_REL = 0.05, 0.5
AUC_GATE = 0.75
SEED = 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--chunk", type=int, default=40_000, help="rows per h5 read")
    p.add_argument("--max-per-donor", type=int, default=4000, help="cells kept per donor × class")
    return p.parse_args()


def cosmx_side():
    """Spillover-safe panel genes and the CosMx T-cell library distribution."""
    adata = CS.load_s1(cache=True, verbose=False)
    cells = pd.read_parquet(CS.CELLS_28D)
    assert cells.index.equals(adata.obs_names), "28d cell table and cosmx_s1.h5ad differ"
    adata.obs["lineage"] = cells.lineage.astype(str).to_numpy()
    _, mean, det = CS.gene_gate(adata, list(adata.var_names))
    rel = mean.loc["T"] / mean.max().clip(lower=1e-9)
    ok = (rel >= MIN_REL) & (det.loc["T"] >= MIN_T_DET)
    genes = [g for g in mean.columns[ok] if "/" not in g]
    genes = [g for g, bad in zip(genes, CM.program_excluded(genes)) if not bad]
    t_lib = adata.obs.nCount_RNA.to_numpy()[adata.obs.lineage.to_numpy() == "T"]
    panel = [g for g in adata.var_names if "/" not in g]
    print(f"CosMx: {len(genes)} spillover-safe T genes of {adata.n_vars}; "
          f"T nCount median {np.median(t_lib):.0f} (IQR {np.percentile(t_lib, 25):.0f}-{np.percentile(t_lib, 75):.0f})")
    return genes, panel, t_lib, rel, det.loc["T"]


def load_scrna(panel, chunk):
    with h5py.File(CM.TCR_OBJ, "r") as h5:
        obs = CM.read_obs(h5, cols=["sample_id", "donor", "study", "disease", "stage_class", "cell_type_T"])
        genes = pd.Index(CM._read_elem(h5["var"][h5["var"].attrs["_index"]])).astype(str)
        calls = pd.read_parquet(CM.CALLS_PQ, columns=["cell_id", "call_source", "mal_cluster"]).set_index("cell_id")
        obs = obs.join(calls, how="left")
        print("disease:", obs.disease.astype(str).value_counts().to_dict())
        print("stage_class:", obs.stage_class.astype(str).value_counts().to_dict())
        use = (obs.cell_type_T.astype(str).isin(CD4_TYPES) & obs.call_source.isin(CALLED)).to_numpy()
        if (obs.disease.astype(str) == "MF").any():
            use &= (obs.disease.astype(str) == "MF").to_numpy()
        rows = np.flatnonzero(use)
        cols = genes.get_indexer([g for g in panel if g in genes])
        cols = cols[cols >= 0]
        g = h5["layers"][CM.COUNTS]
        parts = []
        for a in range(0, len(rows), chunk):
            parts.append(CM.read_csr_rows(g, rows[a:a + chunk])[:, cols])
            print(f"  read {min(a + chunk, len(rows)):,}/{len(rows):,}", flush=True)
    X = sparse.vstack(parts, format="csr")
    return X, genes[cols], obs.iloc[rows].copy()


def thin(X, t_lib, rng):
    """Binomial thinning of each row so its total is drawn from `t_lib` (never upsampled)."""
    X = X.tocsr().astype(np.int64)
    lib = np.asarray(X.sum(1)).ravel()
    target = rng.choice(t_lib, size=X.shape[0])
    p = np.minimum(1.0, target / np.maximum(lib, 1))
    X.data = rng.binomial(X.data, np.repeat(p, np.diff(X.indptr)))
    X.eliminate_zeros()
    return X


def features(X, feat_idx, scale):
    lib = np.asarray(X.sum(1)).ravel()
    F = X[:, feat_idx].toarray().astype(np.float32) * (scale / np.maximum(lib, 1))[:, None]
    return np.log1p(F)


def zscore(F):
    mu, sd = F.mean(0), F.std(0)
    return (F - mu) / np.where(sd > 0, sd, 1), mu, sd


def main():
    args = parse_args()
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    safe, panel, t_lib, rel, tdet = cosmx_side()
    X, sc_genes, obs = load_scrna(panel, args.chunk)
    print(f"scRNA CD4 with a call: {X.shape[0]:,} cells × {X.shape[1]} panel genes "
          f"({time.time() - t0:.0f}s)")

    y_all = obs.mal_cluster.astype(bool).to_numpy()
    donor = obs.donor.astype(str).to_numpy()
    # cap cells per donor × class so the largest donors do not own the fit
    keep = np.zeros(len(obs), bool)
    for (d, c), idx in pd.DataFrame({"d": donor, "c": y_all}).groupby(["d", "c"]).indices.items():
        keep[rng.permutation(idx)[:args.max_per_donor]] = True
    X, obs, y, donor = X[keep], obs[keep], y_all[keep], donor[keep]
    print(f"after ≤{args.max_per_donor} per donor × class: {X.shape[0]:,} cells, {len(set(donor))} donors, "
          f"malignant {y.mean():.2f}")

    Xt = thin(X, t_lib, rng)
    feat = [g for g in safe if g in set(sc_genes)]
    fidx = sc_genes.get_indexer(feat)
    scale = float(np.median(t_lib))
    F = features(Xt, fidx, scale)
    print(f"features: {len(feat)} genes; thinned library median {np.median(np.asarray(Xt.sum(1)).ravel()):.0f}")
    w = 1.0 / pd.Series(donor).map(pd.Series(donor).value_counts()).to_numpy()
    w = w / w.mean()
    stage = obs.stage_class.astype(str).to_numpy()

    rows, oof = [], np.full(len(y), np.nan)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(F, y, donor)):
        Ztr, _, _ = zscore(F[tr])
        Zte, _, _ = zscore(F[te])                              # own-fold z, as on CosMx
        m = CM.fit_model(pd.DataFrame(Ztr, columns=feat), y[tr], w[tr])
        oof[te] = m.predict_proba(Zte)[:, 1]
        rows.append({"fold": k, "n_test": len(te), "n_donors_test": len(set(donor[te])),
                     "auc": roc_auc_score(y[te], oof[te])})
    cv = pd.DataFrame(rows)
    per_donor = [roc_auc_score(y[donor == d], oof[donor == d]) for d in sorted(set(donor))
                 if len(set(y[donor == d])) == 2 and (donor == d).sum() >= 50]
    early = np.isin(stage, ["early", "Early", "IA", "IB", "IIA"])
    summary = {"auc_overall": roc_auc_score(y, oof), "auc_fold_mean": cv.auc.mean(),
               "auc_within_donor_median": float(np.median(per_donor)) if per_donor else np.nan,
               "n_donors_within": len(per_donor),
               "auc_early": roc_auc_score(y[early], oof[early]) if len(set(y[early])) == 2 else np.nan,
               "n_early": int(early.sum()), "n_genes": len(feat), "n_cells": len(y)}
    summary["passes_gate"] = bool(summary["auc_overall"] >= AUC_GATE)
    cv = pd.concat([cv, pd.DataFrame([{"fold": "summary", **summary}])], ignore_index=True)
    CV_CSV.parent.mkdir(parents=True, exist_ok=True)
    cv.to_csv(CV_CSV, index=False)
    print("\nheld-out donors at CosMx depth:")
    print(cv.round(3).to_string(index=False))

    Zall, _, _ = zscore(F)
    m = CM.fit_model(pd.DataFrame(Zall, columns=feat), y, w)
    sc, lr = m[0], m[-1]
    coef = pd.DataFrame({"gene": feat, "coef": lr.coef_[0] / sc.scale_,
                         "mean_mal": F[y].mean(0), "mean_ben": F[~y].mean(0),
                         "cosmx_T_rel": rel.reindex(feat).to_numpy(), "cosmx_T_det": tdet.reindex(feat).to_numpy()})
    prog = pd.read_csv(CM.PROGRAM_CSV).set_index("gene").direction
    coef["program_dir"] = coef.gene.map(prog).fillna("")
    coef = coef.reindex(coef.coef.abs().sort_values(ascending=False).index)
    coef.to_csv(COEF_CSV, index=False)
    print("\ntop coefficients (per z unit):")
    print(coef.head(25).round(3).to_string(index=False))

    MODEL_JSON.write_text(json.dumps({
        "genes": feat, "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
        "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]), "lib_scale": scale,
        "features": "log1p(count * lib_scale / nCount_RNA), then z within CosMx T cells, then scaler",
        "cv": {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in summary.items()},
        "auc_gate": AUC_GATE, "training": "26b mal_cluster, skin CD4 + CD4_Treg, call_source tcr/auto/user"},
        indent=1))
    print(f"\nwrote {MODEL_JSON}, {CV_CSV}, {COEF_CSV}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
