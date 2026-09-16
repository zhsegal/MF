#!/usr/bin/env python
"""SemanticSCVI hyperparameter sweep — CD4 CTCL atlas (projections-only benchmark).

Trains every sweep variant (cached) on the slim CD4 input written by
old/19_semantic_cd4_atlas_sweep.ipynb, runs the projection-only SemanticBenchmark
(Opus LLM judge + MSigDB lib1/lib2 enrichment), and builds the combined HTML
report. No gene clustering.

Config + slim input (adata + semantic_map) are produced by the notebook's
"write job input" cell. Submit via jobs/run_cd4_sweep.sh.
"""
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

NB_MF = Path(__file__).resolve().parent.parent      # the MF project root
NB = NB_MF.parent                                     # notebooks
for _p in (str(NB_MF), str(NB)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scanpy as sc
import torch

from benchmark_helpers import (
    _ScviAdapter,
    build_report,
    plot_training_curves,
    train_or_load_semantic_scvi,
)
from benchmarking import SemanticBenchmark
from llm_scorers import ClaudeCLIScorer

CONFIG = NB_MF / "jobs" / "cd4_sweep_config.json"


def main():
    if not CONFIG.exists():
        sys.exit(f"missing {CONFIG}\nrun the 'write job input' cell of "
                 "old/19_semantic_cd4_atlas_sweep.ipynb first")
    cfg = json.loads(CONFIG.read_text())

    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    model_cache = Path(cfg["model_cache_dir"])

    adata = sc.read_h5ad(cfg["input_h5ad"])
    semantic_map = torch.load(cfg["semantic_map"], weights_only=False)
    print(f"loaded input {adata.shape} | semantic_map {tuple(semantic_map.shape)}", flush=True)

    # ---- train (or load) every sweep variant ----
    trained = {}
    for v in cfg["variants"]:
        name = v["name"]
        # cache_dir keyed on a hash of the ABSOLUTE params (computed in the notebook),
        # so changing BASELINE can never collide with a differently-configured model.
        cache_dir = model_cache / v.get("cache_key", name)
        print(f"\n=== {name} (cache_key={v.get('cache_key', name)}) === cache_dir={cache_dir}", flush=True)
        adata_v = adata.copy()                       # scvi UUID collision -> own copy
        model = train_or_load_semantic_scvi(
            adata_v, semantic_map,
            cache_dir=cache_dir,
            force_train=False,
            batch_key=cfg["batch_key"],
            labels_key=cfg["labels_key"],
            max_epochs=v["max_epochs"],
            warmup_epochs=v["warmup_epochs"],
            n_epochs_kl_warmup=v["n_epochs_kl_warmup"],
            coherence_weight=v["coherence_weight"],
            n_layers=v["n_layers"],
            **cfg["fixed_kwargs"],
        )
        trained[name] = (model, adata_v)
        plot_training_curves(model, name, out_dir / f"training_curves_{name}.png", semantic=True)

    # ---- projection-only benchmark (Opus judge + MSigDB; no clustering) ----
    models = {n: _ScviAdapter(m, a) for n, (m, a) in trained.items()}
    model_names = list(models.keys())

    bench = SemanticBenchmark(
        models, adata,
        pathway_index=cfg["pathway_index"],
        gene_mapping=None,                            # var_names already symbols
        out_dir=str(out_dir),
        cell_type_context=cfg["cell_type_context"],
        judges=[("claude_opus", ClaudeCLIScorer, {"model": "opus"})],
    )
    lib1 = cfg["lib1_gmt"] if cfg.get("lib1_gmt") and Path(cfg["lib1_gmt"]).exists() else None
    lib2 = cfg["lib2_gmt"] if cfg.get("lib2_gmt") and Path(cfg["lib2_gmt"]).exists() else None
    ntop = cfg["per_projection_n_top"]

    # Stage 1 — MSigDB / figures (no LLM):
    bench.benchmark_per_projection_biology(lib1_gmt=lib1, lib2_gmt=lib2, n_top=ntop, enable_llm=False)
    # Stage 2 — Opus grading (resumable from cache):
    bench.benchmark_per_projection_biology(
        lib1_gmt=lib1, lib2_gmt=lib2, n_top=ntop, enable_llm=True,
        llm_cache_dir=str(out_dir / "_llm_cache"),
    )

    # ---- report ----
    report_path = build_report(out_dir, model_names, tuple(adata.shape), notes=cfg.get("notes", ""))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final = out_dir / f"cd4_atlas_semantic_sweep_{ts}.html"
    report_path.rename(final)

    preserve = {"_llm_cache", "_score_cache", "_job_input"}
    for child in out_dir.iterdir():
        if child == final:
            continue
        if child.is_dir():
            if child.name not in preserve:
                shutil.rmtree(child)
        elif child.suffix != ".html":
            child.unlink()
    print(f"\nReport: {final}  ({final.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time() - t0:.1f}s", flush=True)
