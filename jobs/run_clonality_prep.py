#!/usr/bin/env python
"""Run nb37 Parts 0-3 headless to produce the fold job's input.

Parts 0-3 are the heavy CPU prep: backed read of the 20 GB nb30 TCR object, CD4 subset,
label join, Geneformer map, HVG, training pool, slim h5ad + semantic map, fold layout and
jobs/clonality_folds_config.json. The GPU fold jobs cannot start without them.

The notebook stays the single source of truth: this executes its actual code cells (Part 0
through the config cell) in one namespace rather than duplicating them. Everything from
Part 4 on is left for the notebook, which reloads the slim input and the saved latents.

Usage:  python run_clonality_prep.py            # submit via run_clonality_prep.sh
        python run_clonality_prep.py --dry      # list the cells that would run
"""
import argparse
import json
import sys
from pathlib import Path

JOB_DIR = Path(__file__).resolve().parent
NB_MF = JOB_DIR.parent
NOTEBOOK = NB_MF / "37_semantic_clonality_transfer.ipynb"

# Part 0 through the config-writing cell. Identified by a source prefix rather than an index
# so that inserting a markdown cell upstream cannot silently shift the range.
LAST_CELL_PREFIX = "cfg = {"


def cells_to_run(nb: dict) -> list[tuple[int, str]]:
    out = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        out.append((i, src))
        if src.lstrip().startswith(LAST_CELL_PREFIX):
            return out
    raise SystemExit(
        f"no cell starting with {LAST_CELL_PREFIX!r} in {NOTEBOOK.name} — "
        "the config cell moved or was renamed; fix LAST_CELL_PREFIX"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="list the cells, run nothing")
    args = ap.parse_args()

    nb = json.loads(NOTEBOOK.read_text())
    cells = cells_to_run(nb)
    print(f"{NOTEBOOK.name}: running {len(cells)} code cells "
          f"(nb indices {[i for i, _ in cells]})", flush=True)
    if args.dry:
        for i, src in cells:
            print(f"  [{i}] {src.strip().splitlines()[0][:90]}")
        return

    # the notebook's Part 0 locates MF/ by walking up from cwd looking for a dir named MF
    # that contains data/ — run from MF/ so that resolves without ambiguity
    import os
    os.chdir(NB_MF)

    ns: dict = {"__name__": "__main__", "__file__": str(NOTEBOOK)}
    for i, src in cells:
        print(f"\n{'=' * 70}\n[cell {i}] {src.strip().splitlines()[0][:90]}\n{'=' * 70}",
              flush=True)
        exec(compile(src, f"{NOTEBOOK.name}#cell{i}", "exec"), ns)

    cfg_path = Path(ns["CONFIG"])
    cfg = json.loads(cfg_path.read_text())
    n_folds = len(cfg["folds"])
    for k in ("input_h5ad", "semantic_map"):
        p = Path(cfg[k])
        if not p.exists():
            sys.exit(f"config points at a missing {k}: {p}")
        print(f"{k}: {p} ({p.stat().st_size / 1e9:.2f} GB)")
    print(f"\nprep done — {n_folds} folds + all-donor reference "
          f"({n_folds + 1} GPU runs)\nconfig: {cfg_path}", flush=True)


if __name__ == "__main__":
    main()
