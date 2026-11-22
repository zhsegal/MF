#!/usr/bin/env python
"""Populate data/degradome/cache so nb42 runs in seconds.

Writes ONLY cache files -- never the notebook -- so it is safe to run while nb42 is open in
Jupyter. Computes exactly what the notebook's `D.cached(...)` calls compute, under the same
names and the same mtime stamp, so the notebook finds them and skips straight past.
"""
import sys, time, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
import degradome_data as cfg          # noqa: E402
import degradome_helpers as D         # noqa: E402

t0 = time.time()
adata = D.load_degradome_adata()
D.cached("equivalence", lambda: {"max_abs_diff": D.assert_build_equivalence(adata)})
ctrl = D.cached("positive_controls", lambda: D.assert_positive_controls(adata, strict=False))
D.assert_controls_pass(ctrl)

cube = D.DegradomeCube(adata)
panel_cov, _ = D.panel_coverage(cube, verbose=False)
GENES = [g for g in cfg.panel_genes(False) if g in set(panel_cov.index[panel_cov["keep"]])]
D.cached("shuffle_null", lambda: D.shuffle_null(adata, cube, genes=GENES, verbose=False))

agg, dmeta = D.donor_pseudobulk()
D.cached("contrast_disease",
         lambda: D.disease_contrast(agg, dmeta, levels=cfg.DISEASE_TESTABLE,
                                    arm="within_study"))
D.cached("contrast_disease_pooled",
         lambda: D.disease_contrast(agg, dmeta, levels=cfg.DISEASE_TESTABLE,
                                    arm="pooled_sensitivity"))
D.cached("contrast_layer",
         lambda: D.layer_contrast(*D.donor_pseudobulk(extra_keys=(cfg.LAYER_KEY,))))
D.cached("contrast_stage", lambda: D.stage_contrast(agg, dmeta))

print(f"\ncache primed in {(time.time() - t0) / 60:.1f} min -> {cfg.CACHE_DIR}")
for f in sorted(cfg.CACHE_DIR.iterdir()):
    print(f"  {f.name}")
