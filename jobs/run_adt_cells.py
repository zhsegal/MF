#!/usr/bin/env python
"""Per-cell CITE-seq table for the protein-vs-transcript analysis (nb45).

    python run_adt_cells.py [--out-dir ...]

`blood_T_annotated.h5ad` carries a 128-marker ADT panel in `obsm["X_adt"]`, measured on the
61,215 buus2025 cells and NaN everywhere else. nb43's §7 compared protein and transcript at
donor level, 17 points per marker. That is enough to say whether the two agree and not nearly
enough to ask *why* they disagree -- which is the question: is a marker's surface level better
explained by its own transcript, or by the cell's trafficking machinery?

This writes the cell-level table that question needs, once, so the notebook is light:

    data/targets/blood_adt_counts.parquet   cell x 128 raw ADT counts, isotype controls kept
    data/targets/blood_adt_cells.parquet    cell x [mapped marker genes as log1p CP10K,
                                            11 trafficking module scores, lineage/QC covariates,
                                            arm, donor, dx]

The ADT matrix is written as raw counts, not normalised: CLR, isotype-background subtraction and
their variants are decisions nb45 makes and shows, not something baked in here. nb43's first
pass used `log1p(x / exp(mean(log1p(x))))`, which compresses the scale so far that most markers
came out with a malignant-minus-reactive delta of ~0.001 and were then counted as *contradicting*
their transcript.

Row-subset read via `CM.read_csr_rows`, so this touches ~9% of the matrix. Seconds, under 4 GB.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("TARGET_SCREEN_THREADS", "4"))

NB_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NB_DIR / "helpers"))

import h5py                              # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

import cluster_malig_helpers as CM       # noqa: E402
import malignancy as MAL                 # noqa: E402
import surface_targets as ST             # noqa: E402
import target_qc as TQ                   # noqa: E402

OUT_DIR = NB_DIR / "data" / "targets"
STRICT_MIN_COUNTS, STRICT_MIN_GENES = 2, 2


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--max-cells", type=int, default=None, help="smoke test only")
    return p.parse_args()


def main():
    args = parse_args()
    CM.use_tissue("blood")
    MAL.use_tissue("blood")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    obj = CM.CONFIG["blood"]["TCR_OBJ"]

    with h5py.File(obj, "r") as h5:
        var = pd.Index(CM._read_elem(h5["var"][h5["var"].attrs["_index"]])).astype(str)
        obs = CM.read_obs(h5, cols=["donor", "sample_id", "study", "disease", "cell_type_T",
                                    "has_tcr", "total_counts", "n_genes", "doublet_score"])
        panel = np.asarray(CM._read_elem(h5["uns"]["adt_panel"])).astype(str)
        adt = h5["obsm"]["X_adt"][:]
        log(f"{obj.name}: {len(obs):,} cells | ADT panel {len(panel)}")

        has_adt = np.isfinite(adt).any(1)
        rows = np.flatnonzero(has_adt)
        if args.max_cells:
            rows = rows[:args.max_cells]
            log(f"SMOKE TEST: first {len(rows):,} ADT cells only")
        log(f"ADT measured on {len(rows):,} cells ({has_adt.mean():.1%})")
        cov = (pd.DataFrame({"study": obs["study"].astype(str).to_numpy(),
                             "disease": obs["disease"].astype(str).to_numpy(), "adt": has_adt})
               .groupby(["study", "disease"], observed=True)["adt"].agg(["size", "sum"]))
        log("ADT coverage by study x disease:\n" + cov[cov["sum"] > 0].to_string())

        # ---- arms, on every cell, then subset. Same call and same HC rule as the screen.
        mal = MAL.load_malignancy(obs.index, has_tcr=obs["has_tcr"].astype(bool).to_numpy(),
                                  verbose=False)
        lvl = pd.Series(np.asarray(MAL.cd4_level(obs["cell_type_T"], mal,
                                                 call=MAL.HEADLINE)).astype(str), index=obs.index)
        try:
            import target_cohort as TC
            don = TC.blood_donor_table(obj=obj)
        except Exception as exc:                                   # noqa: BLE001
            log(f"target_cohort unavailable ({exc})")
            don = None
        donor = obs["donor"].astype(str)
        if don is not None:
            is_hc = donor.map(don["is_hc"]).fillna(False).to_numpy()
            lvl[(obs["cell_type_T"].astype(str) == "CD4") & is_hc] = "CD4_HC"

        # ---- genes: every ADT-mapped symbol, the trafficking modules, the loss markers
        want = sorted({*ST.ADT_TO_GENE.values(), *ST.TRAFFIC_GENES, *ST.LOSS_MARKERS,
                       *ST.POSITIVE_CONTROLS, *TQ.lineage_genes(include_nk=True)})
        genes = [g for g in want if g in set(var)]
        log(f"{len(genes):,} of {len(want):,} wanted genes present in the HVG var "
            f"({len(want) - len(genes)} absent)")

        X = CM.read_csr_rows(h5["layers"]["counts"], rows)
    log(f"read {X.shape} counts for the ADT cells")

    gi = var.get_indexer(genes)
    lib = pd.to_numeric(obs["total_counts"], errors="coerce").to_numpy(float)[rows]
    lib[~np.isfinite(lib) | (lib <= 0)] = np.asarray(X.sum(1)).ravel()[
        ~np.isfinite(lib) | (lib <= 0)]
    lib[lib <= 0] = 1.0
    C = np.asarray(X[:, gi].todense(), dtype=np.float32)
    L = np.log1p(C / lib[:, None] * 1e4).astype(np.float32)
    gpos = pd.Index(genes)

    # ---- per-cell module scores and contamination load, the same rule as the screen build
    cells = pd.DataFrame(index=pd.Index(obs.index[rows], name="cell_id"))
    cells["donor"] = donor.to_numpy()[rows]
    cells["study"] = obs["study"].astype(str).to_numpy()[rows]
    cells["arm"] = lvl.to_numpy()[rows]
    cells["cell_type_T"] = obs["cell_type_T"].astype(str).to_numpy()[rows]
    if don is not None:
        for c in ("dx", "dx2", "cohort_keep", "is_cultured"):
            cells[c] = cells["donor"].map(don[c]).to_numpy()
    cells["total_counts"] = lib
    cells["log_umi"] = np.log1p(lib)
    cells["n_genes"] = pd.to_numeric(obs["n_genes"], errors="coerce").to_numpy()[rows]
    cells["doublet_score"] = pd.to_numeric(obs["doublet_score"],
                                           errors="coerce").to_numpy()[rows]

    strict = [g for g in TQ.strict_genes() if g in set(genes)]
    S = C[:, gpos.get_indexer(strict)]
    cells["lineage_pos"] = ((S >= STRICT_MIN_COUNTS).any(1)
                            | ((S >= 1).sum(1) >= STRICT_MIN_GENES))
    for mod, spec in TQ.LINEAGE_MODULES.items():
        cols = [g for g in spec["genes"] if g in set(genes)]
        if cols:
            cells[f"score_{mod}"] = L[:, gpos.get_indexer(cols)].mean(1)
    log(f"lineage-positive among ADT cells: {cells['lineage_pos'].mean():.1%}")

    mods = {}
    for mod, mod_genes in ST.TRAFFIC_MODULES.items():
        cols = [g for g in mod_genes if g in set(genes)]
        mods[mod] = len(cols)
        if cols:
            cells[f"traffic_{mod}"] = L[:, gpos.get_indexer(cols)].mean(1)
    log("trafficking modules, genes present in the HVG var:\n"
        + pd.Series(mods).sort_values(ascending=False).to_string())

    # ---- the gene columns themselves, prefixed so they never collide with a covariate
    expr = pd.DataFrame(L, columns=[f"rna_{g}" for g in genes], index=cells.index)
    out_cells = pd.concat([cells, expr], axis=1)
    p_cells = args.out_dir / "blood_adt_cells.parquet"
    out_cells.reset_index().to_parquet(p_cells, index=False)
    log(f"wrote {p_cells.name}  {out_cells.shape}")

    A = pd.DataFrame(adt[rows], columns=panel, index=cells.index)
    p_adt = args.out_dir / "blood_adt_counts.parquet"
    A.reset_index().to_parquet(p_adt, index=False)
    log(f"wrote {p_adt.name}  {A.shape}")

    (args.out_dir / "blood_adt_manifest.json").write_text(json.dumps({
        "source": str(obj), "written": time.strftime("%Y-%m-%d %H:%M"),
        "n_cells": int(len(rows)), "n_panel": int(len(panel)),
        "n_genes": len(genes),
        "genes_absent_from_hvg": sorted(set(want) - set(genes)),
        "malignancy_call": MAL.HEADLINE,
        "arms": {k: int(v) for k, v in cells["arm"].value_counts().items()},
        "donors": int(cells["donor"].nunique()),
        "traffic_module_genes_present": mods,
        "clean_rule": {"strict_markers": strict, "min_counts": STRICT_MIN_COUNTS,
                       "min_genes": STRICT_MIN_GENES},
        "adt_normalisation": "none -- raw counts, nb45 normalises",
    }, indent=1) + "\n")
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
