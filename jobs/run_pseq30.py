#!/usr/bin/env python
"""Heavy pipeline for notebooks/20_skin/30_perturbseq_per_donor.ipynb (per-donor pert2state).

Steps (each skipped when its cache exists; delete a cache or bump pseq30_helpers.VER to recompute):
  1. donor table (nb29 cohort, min(n_malignant, n_reactive) >= 600)      -> pseq30_donor_table_v1.csv
  2. CD4 cells (nb29 cache) -> testable genes, reactive composition       -> pseq30_gene_set_prelim_v1.txt, pseq30_reactive_composition_v1.csv
  3. per-donor pseudo-replicate z (pydeseq2 ~ arm) + 30-donor cohort DE, genes = prelim set -> pseq30_donor_z_v1.parquet, pseq30_cohort30_de_v1.csv
  4. final gene set; X -> standardize -> SVD(60), scrambled Z, LIC order  -> pseq30_gene_set_v1.txt, pseq30_svd_v1.npz, pseq30_lic_order_v1.txt
  5. frozen hyper-parameters (ElasticNetCV on the cohort signature)       -> pseq30_hyperparams_v1.json
  6. fits: cohort, 30 donors, 30 donors x scrambled, shared folds         -> tables/pseq30_model_fit_v1.csv, tables/pseq30_regulator_weights_v1.csv
Usage: python jobs/run_pseq30.py [--n-cpus 8]   (submit with jobs/run_pseq30.sh)
"""
import argparse
import gc
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "helpers"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import perturbseq_helpers as P  # noqa: E402
import pseq30_helpers as H  # noqa: E402

T0 = time.time()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def read_list(p):
    return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]


def write_list(items, p):
    Path(p).write_text("\n".join(map(str, items)) + "\n")


def main(n_cpus: int):
    # ------------------------------------------------------------------ 1. donors
    if H.DONOR_TAB.exists():
        tab = pd.read_csv(H.DONOR_TAB, index_col=0)
    else:
        tab = H.donor_table()
        tab.to_csv(H.DONOR_TAB)
    donors = tab.index[tab["retained"]].tolist()
    log(f"donors retained (min arm >= {H.MIN_CELLS_ARM}): {len(donors)} of {len(tab)} eligible")
    print(tab[tab["retained"]][["n_malignant", "n_reactive", "n_min", "study", "stage", "split"]].to_string())
    print("\nnext 5 below the cutoff:")
    print(tab[~tab["retained"]].head(5)[["n_malignant", "n_reactive", "n_min", "study"]].to_string())
    assert len(donors) == 30, f"expected 30 donors, got {len(donors)} -- revisit MIN_CELLS_ARM"

    # ------------------------------------------------------------------ 2-3. signatures from the CD4 cells
    need = [p for p in (H.GENE_SET_PRE, H.REACT_COMP, H.DONOR_Z, H.COHORT_DE) if not p.exists()]
    if need:
        import scanpy as sc
        log(f"reading {H.CD4_H5.name} (missing: {[p.name for p in need]})")
        a = sc.read_h5ad(H.CD4_H5)
        a = a[a.obs["donor"].astype(str).isin(donors)].copy()
        gc.collect()
        log(f"CD4 cells of the 30 donors: {a.shape}")
        if not H.GENE_SET_PRE.exists():
            nz = H.nonzero_cells_per_arm(a)
            testable = H.testable_genes(nz, donors)
            prelim = testable.intersection(H.pseq_genes())
            write_list(prelim, H.GENE_SET_PRE)
            log(f"testable in >= {H.GENE_FRAC_DONORS:.0%} donors: {len(testable):,}; "
                f"x Perturb-seq measured: {len(prelim):,}")
        if not H.REACT_COMP.exists():
            H.reactive_composition(a, donors).to_csv(H.REACT_COMP)
        # DE only on the prelim gene set (Perturb-seq-measured x testable): ~6.5x faster than 42k genes and
        # every downstream use (design, LIC order, downstream rule) is within this set anyway.
        prelim = [l.strip() for l in H.GENE_SET_PRE.read_text().splitlines() if l.strip()]
        a = a[:, prelim].copy()
        gc.collect()
        log(f"restricted to the prelim gene set for DE: {a.shape}")
        if not H.DONOR_Z.exists():
            log("per-donor pseudo-replicate DE (pydeseq2 ~ arm, 3 + 3 pseudobulks per donor)")
            Z, failed = H.pseudo_replicate_signatures(a, donors, n_cpus=n_cpus, part_dir=H.DONOR_Z_PARTS)
            for d, e in failed:
                print(f"  !! pydeseq2 FAILED for {d}: {e}")
            assert not failed, "per-donor DE failed for some donors (listed above)"
            Z.to_parquet(H.DONOR_Z)
            log(f"donor z: {Z.shape}")
        if not H.COHORT_DE.exists():
            log("30-donor cohort DE (pydeseq2 ~ donor + arm)")
            res = H.cohort_signature(a, donors, n_cpus=n_cpus)
            res.to_csv(H.COHORT_DE)
            log(f"cohort DE: {res.shape}; padj<0.01: {(res['padj'] < 0.01).sum():,}")
        del a
        gc.collect()

    prelim = pd.Index(read_list(H.GENE_SET_PRE))
    Zd = pd.read_parquet(H.DONOR_Z)
    cohort = pd.read_csv(H.COHORT_DE, index_col=0)
    assert list(Zd.columns) == donors, "donor_z columns != retained donors"

    # ------------------------------------------------------------------ 4. final gene set + design
    if H.GENE_SET.exists():
        genes = pd.Index(read_list(H.GENE_SET))
    else:
        finite = Zd.reindex(prelim).notna().all(axis=1) & cohort["stat"].reindex(prelim).notna()
        genes = prelim[finite.to_numpy()]
        write_list(genes, H.GENE_SET)
    log(f"final gene set: {len(genes):,} (prelim {len(prelim):,}; dropped {len(prelim) - len(genes)} with a NaN Wald stat)")
    nan_per_donor = Zd.reindex(prelim).isna().sum()
    print("NaN stat per donor on the prelim set (max):", int(nan_per_donor.max()))

    if H.SVD_NPZ.exists() and H.LIC_ORDER.exists():
        d = H.load_design()
    else:
        log("loading Perturb-seq resource and building the design (once)")
        pseq = P.load_pseq(H.CONDITION, H.PSEQ_H5, layers=("zscore_masked",))
        d = H.build_design(pseq, genes)
        Z_scr = H.scrambled_Z(d["X"], d, seed=H.SEED)
        H.save_design(d, H.SVD_NPZ, Z_scr=Z_scr)
        sig = cohort.index[cohort["padj"] < H.FDR_SIG].intersection(d["X"].index)
        write_list(H.lic_order(d["X"], sig), H.LIC_ORDER)
        log(f"X: {d['X'].shape}; SVD explained variance: {d['explained'].sum():.3f}; LIC genes: {len(sig):,}")
        del pseq, d["X"]
        gc.collect()
        d = H.load_design()
    assert list(d["genes"]) == list(genes), "design genes != gene set"
    regs = pd.Index(d["regulators"])
    Z, V, Z_scr = d["Z"], d["V"], d["Z_scr"]
    log(f"design: Z {Z.shape}, V {V.shape}, regulators {len(regs):,}")

    # ------------------------------------------------------------------ 5. hyper-parameters
    y_cohort = cohort["stat"].reindex(genes)
    assert y_cohort.notna().all()
    if H.HYPER.exists():
        hp = H.read_json(H.HYPER)
    else:
        log("freezing hyper-parameters (ElasticNetCV, 4-fold, authors' grids) on the cohort signature")
        hp = H.freeze_hyperparams(Z, P.prepare_y(y_cohort).to_numpy(), n_jobs=n_cpus)
        H.write_json(hp, H.HYPER)
    log(f"frozen: alpha = {hp['alpha']:.4g}, l1_ratio = {hp['l1_ratio']:.4g}")

    # ------------------------------------------------------------------ 6. fits
    folds = H.gene_folds(len(genes))
    fit = lambda Zm, y: H.fit_fixed(Zm, y, folds, V, hp["alpha"], hp["l1_ratio"])  # noqa: E731
    log("cohort fit")
    fc = fit(Z, y_cohort)
    rows, W, Wsem = [], {}, {}
    for i, dn in enumerate(donors):
        y = Zd[dn].reindex(genes)
        fr = fit(Z, y)
        fs = fit(Z_scr, y)
        W[dn], Wsem[dn] = fr["w_mean"], fr["w_sem"]
        rows.append({"donor": dn, "test_r": fr["test_r"], "test_r_se": fr["test_r_se"], "train_r": fr["train_r"],
                     "scrambled_r": fs["test_r"], "scrambled_r_se": fs["test_r_se"],
                     "converged": fr["converged"], "n_iter_max": fr["n_iter_max"]})
        log(f"  {i + 1:2d}/30 {dn:28s} r = {fr['test_r']:.3f} +- {fr['test_r_se']:.3f}  "
            f"scrambled {fs['test_r']:.3f}  train {fr['train_r']:.3f}{'' if fr['converged'] else '  !! not converged'}")
    W = pd.DataFrame(W, index=regs)
    W.index.name = "regulator"
    Wsem = pd.DataFrame(Wsem, index=regs)
    Wsem.index.name = "regulator"
    assert W.shape == (len(regs), 30) and not W.isna().any().any(), "weight matrix has NaN or wrong shape"

    mf = pd.DataFrame(rows).set_index("donor")
    mf = tab.loc[donors, ["study", "stage", "split", "n_malignant", "n_reactive", "n_min"]].join(mf)
    loo = {dn: np.corrcoef(W[dn], W.drop(columns=dn).mean(axis=1))[0, 1] for dn in donors}
    mf["r_vs_consensus_loo"] = pd.Series(loo)
    mf["is_cohort"] = False
    mf.loc["COHORT30"] = {"study": "all", "stage": "", "split": "", "n_malignant": tab.loc[donors, "n_malignant"].sum(),
                          "n_reactive": tab.loc[donors, "n_reactive"].sum(), "n_min": np.nan,
                          "test_r": fc["test_r"], "test_r_se": fc["test_r_se"], "train_r": fc["train_r"],
                          "scrambled_r": np.nan, "scrambled_r_se": np.nan, "converged": fc["converged"],
                          "n_iter_max": fc["n_iter_max"], "r_vs_consensus_loo": np.corrcoef(fc["w_mean"], W.mean(axis=1))[0, 1],
                          "is_cohort": True}
    mf.index.name = "donor"
    mf.to_csv(H.MODEL_FIT)

    out = W.copy()
    out["cohort_w"] = fc["w_mean"]
    out["mean_w"] = W.mean(axis=1)
    out["sd_w"] = W.std(axis=1)
    out["sem_mean"] = out["sd_w"] / np.sqrt(W.shape[1])
    out["frac_positive"] = (W > 0).mean(axis=1)
    out.sort_values("mean_w", ascending=False).to_csv(H.WEIGHTS)
    Wsem.to_parquet(H.W_SEM)

    log(f"cohort30 held-out r = {fc['test_r']:.3f} +- {fc['test_r_se']:.3f} (nb29 cohort: 0.61)")
    log(f"donor r: median {mf.loc[donors, 'test_r'].median():.3f}  range {mf.loc[donors, 'test_r'].min():.3f}-"
        f"{mf.loc[donors, 'test_r'].max():.3f};  scrambled median {mf.loc[donors, 'scrambled_r'].median():.3f}")
    flag = mf.loc[donors]
    for dn in flag.index[~flag["converged"]]:
        print(f"  !! {dn}: a fold hit max_iter")
    for dn in flag.index[flag["r_vs_consensus_loo"] < 0.1]:
        print(f"  !! {dn}: weights correlate {flag.loc[dn, 'r_vs_consensus_loo']:.2f} with the leave-one-out consensus")
    log(f"wrote {H.MODEL_FIT.name}, {H.WEIGHTS.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cpus", type=int, default=int(os.environ.get("LSB_DJOB_NUMPROC", 4)))
    args = ap.parse_args()
    main(args.n_cpus)
