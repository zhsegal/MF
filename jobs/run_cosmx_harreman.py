#!/usr/bin/env python
"""Hotspot + Harreman on the CosMx early-MF layer for nb28e (runs in the `pxlgn` env).

Model-free counterparts of 28d F4-F6 and 28e F10:
  A. Hotspot, pooled over slides (kNN graph within slide): local autocorrelation of every panel
     gene (DANB on counts, nCount_RNA as UMI), pairwise local correlation of the top spatial genes
     plus the focus genes (proteases, T genes, axis genes), gene modules and per-cell module
     scores. Question for F11: does any spatial module put T-cell genes with MMP9/MMP12/MMP2?
  B. Hotspot per slide on the focus genes only -> the ≥3/4-slides rule for protease × T-gene
     local correlation.
  C. Harreman cell-type-aware LR communication (CellChat human LR DB), twice: T cells split by
     compartment (epidermis / DEJ / dermis) and by malignancy-transfer tertile, each against
     Fibroblast / Myeloid / Keratinocyte, both directions.
     Harreman's own expression filter keeps a gene in a cell type at >= 20% detection; CosMx
     ligands in T cells sit at ~10-20% (28d §2), so it would drop nearly all of them. The filter
     is replaced by >= MIN_DET detection per cell type, written to uns before the gene pairs.

--resolvi (nb28f): Hotspot A + B only, on log1p resolVI expression (processed/cosmx_s1_resolvi_all.h5ad,
z-scored per gene within slide, model 'none'); tables get the suffix _rvi.

Inputs : cosmx_s1.h5ad (layer counts), cosmx_s1_cells_28e.parquet (lineage_final, compartment,
         mal_group -- from nb28e §R; run nb28e once before this job)
Outputs: tables/deg28e_hotspot_{autocorr,lcz,modules,focus_per_slide}.csv,
         tables/deg28e_harreman_{comp,mal}_{gp,m}.csv,
         data/S1_cosmx_earlymf_spatial/processed/cosmx_s1_hotspot_module_scores.parquet
"""
import os
import sys
import time
import traceback
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("COSMX_HARREMAN_THREADS", "4"))
NB_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NB_DIR / "helpers"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cosmx_spatial as CS  # noqa: E402
import degradome_data as cfg  # noqa: E402

TAB = NB_DIR / "tables"
MOD_PQ = CS.S1_DIR / "processed" / "cosmx_s1_hotspot_module_scores.parquet"
N_NEIGHBORS = 10
N_TOP_SPATIAL = 250
MIN_DET = 0.05
PARTNERS = ["Fibroblast", "Myeloid", "Keratinocyte"]
PROTEASE_FAMS = ["mmp", "cathepsin", "granzyme", "serine_other", "metallo_other", "inhibitor"]
T_GENES = ["CD3E", "CD3D", "CD2", "CD3G"]


RESOLVI_ALL = CS.S1_DIR / "processed" / "cosmx_s1_resolvi_all.h5ad"
SUFFIX = ""        # "_rvi" with --resolvi


def tab(name):
    return TAB / f"deg28e_{name}{SUFFIX}.csv"


def as_frame(m, genes):
    return m if isinstance(m, pd.DataFrame) else pd.DataFrame(np.asarray(m), index=genes, columns=genes)


def hotspot(hm, adata, genes_ac=None, genes_lc=None, modules=True, layer="counts", model="danb"):
    if genes_ac is not None:
        genes_ac = [g for g in genes_ac if g in adata.var_names]
    hm.tl.compute_knn_graph(adata, compute_neighbors_on_key="spatial", n_neighbors=N_NEIGHBORS,
                            sample_key="slide", weighted_graph=False)
    hm.hs.compute_local_autocorrelation(adata, layer_key=layer, model=model, genes=genes_ac,
                                        species="human", umi_counts_obs_key="nCount_RNA")
    ac = adata.uns["gene_autocorrelation_results"]
    if genes_lc is None:
        top = ac[ac.Z_FDR < 0.05].sort_values("Z", ascending=False).index[:N_TOP_SPATIAL]
        genes_lc = list(dict.fromkeys(list(top) + FOCUS))
    genes_lc = [g for g in genes_lc if g in ac.index]
    hm.hs.compute_local_correlation(adata, genes=genes_lc)
    lcz = as_frame(adata.uns["lc_zs"], genes_lc)
    if modules:
        hm.hs.create_modules(adata, min_gene_threshold=10, fdr_threshold=0.05)
        hm.hs.calculate_module_scores(adata)
    return ac, lcz


def ct_run(hm, base, key, tag):
    # Harreman 0.1.4 standardises counts over the cells of the tested cell types but indexes them
    # with positions over all cells (IndexError when any cell type is in no pair), so every cell
    # left in must belong to a tested type: Unassigned is dropped and the graph rebuilt.
    keep = base.obs[key].astype(str).isin([c for c in base.obs[key].astype(str).unique()
                                           if c.startswith("T_") or c in PARTNERS]).to_numpy()
    ad = base[keep].copy()
    ad.uns = {}
    ad.obs[key] = ad.obs[key].astype(str)
    hm.pp.extract_interaction_db(ad, species="human", database="LR", extracellular_only=True)
    db = ad.varm["database"]
    db_genes = db.index[(db != 0).any(axis=1)]
    print(f"[{tag}] {ad.n_obs:,} cells of tested types; LR DB: {db.shape[1]} interactions over {len(db_genes)} panel genes")
    hm.tl.compute_knn_graph(ad, compute_neighbors_on_key="spatial", n_neighbors=N_NEIGHBORS,
                            sample_key="slide", weighted_graph=False)
    hm.hs.compute_local_autocorrelation(ad, layer_key="counts", model="danb", genes=list(db_genes),
                                        species="human", umi_counts_obs_key="nCount_RNA")
    C = ad[:, db_genes].layers["counts"]
    lab = ad.obs[key].to_numpy()
    filt = {}
    for ct in np.unique(lab):
        det = np.asarray((C[lab == ct] > 0).mean(0)).ravel()
        filt[ct] = list(db_genes[det >= MIN_DET])
    ad.uns.update({"autocorrelation_filt": False, "expression_filt": False, "de_filt": False,
                   "filtered_genes": sorted(set().union(*filt.values())), "filtered_genes_ct": filt})
    t_groups = sorted(c for c in np.unique(lab) if c.startswith("T_"))
    pairs = [(a, b) for t in t_groups for p in PARTNERS for a, b in [(t, p), (p, t)]]
    hm.tl.compute_gene_pairs(ad, layer_key="counts", cell_type_key=key, cell_type_pairs=pairs,
                             ct_specific=True)
    print(f"[{tag}] gene pairs: {len(ad.uns['gene_pairs'])}; ct pairs with pairs: {len(ad.uns['cell_type_pairs'])}")
    try:
        hm.tl.compute_ct_cell_communication(ad, layer_key_p_test="counts", layer_key_np_test="counts",
                                            model="danb", cell_type_key=key, test="parametric")
    except ValueError as e:
        # 0.1.4 assembles its result table from a dense (ct pair × gene pair) array but lists only
        # the valid combinations; the statistics are already in uns, so the table is built here.
        print(f"[{tag}] Harreman table assembly failed ({str(e)[:80]}); building it from uns")
    for part, df in results_tables(ad).items():
        df.to_csv(tab(f"harreman_{tag}_{part}"), index=False)
        print(f"[{tag}] {part}: {len(df)} tests, {int((df.Z_FDR < 0.05).sum())} FDR<0.05")


def results_tables(ad):
    """(ct pair × gene pair) and (ct pair × LR) Z tables over valid combinations only, with BH
    recomputed over those (Harreman's FDR also counts the empty cells of the dense array)."""
    from statsmodels.stats.multitest import multipletests

    u, p = ad.uns, ad.uns["ct_ccc_results"]["p"]
    ctp, gps, per = u["cell_type_pairs"], u["gene_pairs"], u["gene_pairs_per_ct_pair_ind"]
    name = lambda g: "+".join(g) if isinstance(g, (list, tuple)) else g
    out = {}
    Z, C = np.asarray(p["gp"]["Z"]), np.asarray(p["gp"]["cs"])
    if Z.shape == (len(gps), len(ctp)):
        Z, C = Z.T, C.T
    rows = [{"Cell Type 1": a, "Cell Type 2": b, "Gene 1": name(gps[j][0]), "Gene 2": name(gps[j][1]),
             "C_p": C[i, j], "Z": Z[i, j]}
            for i, (a, b) in enumerate(ctp) for j in per.get((a, b), [])]
    out["gp"] = pd.DataFrame(rows)
    mets = list(u["gene_pair_dict"])
    Zm, Cm = np.asarray(p["m"]["Z"]), np.asarray(p["m"]["cs"])
    if Zm.shape == (len(mets), len(ctp)):
        Zm, Cm = Zm.T, Cm.T
    rows = [{"Cell Type 1": a, "Cell Type 2": b, "metabolite": m, "C_p": Cm[i, k], "Z": Zm[i, k]}
            for i, (a, b) in enumerate(ctp) for k, m in enumerate(mets)
            if set(u["gene_pair_dict"][m]) & set(per.get((a, b), []))]
    out["m"] = pd.DataFrame(rows)
    for df in out.values():
        df["Z_pval"] = norm_sf(df.Z.to_numpy(float))
        df["Z_FDR"] = multipletests(np.nan_to_num(df.Z_pval, nan=1.0), method="fdr_bh")[1]
    return out


def norm_sf(z):
    from scipy.stats import norm
    return norm.sf(z)


def load_resolvi():
    """cosmx_s1.h5ad obs/coordinates with layer 'rvi_z': log1p resolVI expression (library 1e4),
    z-scored per gene within slide, for Hotspot's 'none' model (input already standardised)."""
    import anndata as ad

    adata = CS.load_s1(cache=True, verbose=False)
    R = ad.read_h5ad(RESOLVI_ALL)
    assert R.obs_names.equals(adata.obs_names) and R.var_names.equals(adata.var_names), "resolVI export mismatch"
    L = np.log1p(np.asarray(R.X, np.float32))
    for s in adata.obs.slide.cat.categories:
        m = (adata.obs.slide == s).to_numpy()
        mu, sd = L[m].mean(0), L[m].std(0)
        L[m] = (L[m] - mu) / np.where(sd > 0, sd, 1)
    adata.layers["rvi_z"] = L
    adata.obs["lineage_resolvi"] = R.obs.lineage_resolvi.astype(str).to_numpy()
    return adata


def main_resolvi(hm):
    """Hotspot A + B on resolVI expression (nb28f). No Harreman part C."""
    t0 = time.time()
    adata = load_resolvi()
    print(adata)
    ac, lcz = hotspot(hm, adata, layer="rvi_z", model="none")
    ac.to_csv(tab("hotspot_autocorr"))
    lcz.to_csv(tab("hotspot_lcz"))
    mods = adata.uns["modules"].rename("module")
    mods.to_csv(tab("hotspot_modules"))
    print(f"A: {int((ac.Z_FDR < 0.05).sum())} autocorrelated genes; {lcz.shape[0]} in local correlation; modules:",
          mods[mods != -1].value_counts().sort_index().to_dict(), f"({time.time() - t0:.0f}s)")
    T = [g for g in T_GENES + ["CD4", "CD8A"] if g in lcz.index]
    for g in [g for g in ["MMP2", "MMP9", "MMP14"] if g in lcz.index]:
        z = lcz.loc[g].drop(g).sort_values(ascending=False)
        print(f"   {g}: top {z.head(5).round(1).to_dict()} | T genes (Z, rank/{len(z)}):",
              {t: (round(z[t], 1), int(z.index.get_loc(t)) + 1) for t in T if t in z})
    rows = []
    for s in adata.obs.slide.cat.categories:
        sub = adata[adata.obs.slide.to_numpy() == s].copy()
        g = [x for x in FOCUS + ["CD4", "CD8A"] if x in sub.var_names]
        try:
            _, z = hotspot(hm, sub, genes_ac=g, genes_lc=g, modules=False, layer="rvi_z", model="none")
        except Exception:
            traceback.print_exc(); continue
        for a in [x for x in PROTEASES if x in z.index]:
            for b in [x for x in T_GENES + ["CD4", "CD8A"] + AXIS if x in z.columns and x != a]:
                rows.append({"slide": s, "gene_a": a, "gene_b": b, "z": z.loc[a, b]})
    pd.DataFrame(rows).to_csv(tab("hotspot_focus_per_slide"), index=False)
    print(f"B: {len(rows)} per-slide protease × T/axis pairs; done in {time.time() - t0:.0f}s")


def main():
    global SUFFIX
    import harreman as hm

    if "--resolvi" in sys.argv[1:]:
        SUFFIX = "_rvi"
        return main_resolvi(hm)
    t0 = time.time()
    adata = CS.load_s1(cache=True, verbose=False)
    if not CS.CELLS_28E.exists():
        sys.exit(f"missing {CS.CELLS_28E}: run nb28e (through §R) first")
    cells = pd.read_parquet(CS.CELLS_28E)
    assert cells.index.equals(adata.obs_names), "28e cell table and cosmx_s1.h5ad differ"
    for c in ["lineage_final", "compartment", "mal_group"]:
        adata.obs[c] = cells[c].astype(str).to_numpy()
    adata.X = adata.layers["counts"]
    is_t = adata.obs.lineage_final.to_numpy() == "T"
    comp = adata.obs.compartment.replace({"no_epidermis": "dermis"}).to_numpy()
    adata.obs["ct_comp"] = np.where(is_t, "T_" + comp, adata.obs.lineage_final.to_numpy())
    adata.obs["ct_mal"] = np.where(is_t, "T_" + adata.obs.mal_group.to_numpy(), adata.obs.lineage_final.to_numpy())
    print(adata)
    print("ct_comp:", adata.obs.ct_comp.value_counts().to_dict())
    print("ct_mal:", adata.obs.ct_mal.value_counts().to_dict())

    # A. pooled Hotspot
    ac, lcz = hotspot(hm, adata)
    ac.to_csv(tab("hotspot_autocorr"))
    lcz.to_csv(tab("hotspot_lcz"))
    mods = adata.uns["modules"].rename("module")
    mods.to_csv(tab("hotspot_modules"))
    ms = adata.obsm["module_scores"]
    pd.DataFrame(np.asarray(ms), index=adata.obs_names,
                 columns=[f"module_{c}" for c in getattr(ms, "columns", range(np.shape(ms)[1]))]
                 ).to_parquet(MOD_PQ)
    print(f"A: {int((ac.Z_FDR < 0.05).sum())} autocorrelated genes; modules:",
          mods[mods != -1].value_counts().sort_index().to_dict(), f"({time.time() - t0:.0f}s)")
    for g in [g for g in ["MMP2", "MMP9", "MMP12", "MMP14"] if g in mods.index]:
        m = mods[g]
        print(f"   {g}: module {m}", "with", sorted(mods.index[(mods == m) & (m != -1)])[:25])

    # B. per slide, focus genes only
    rows = []
    for s in adata.obs.slide.cat.categories:
        sub = adata[adata.obs.slide.to_numpy() == s].copy()
        g = [x for x in FOCUS if x in sub.var_names]
        try:
            _, z = hotspot(hm, sub, genes_ac=g, genes_lc=g, modules=False)
        except Exception:
            traceback.print_exc(); continue
        for a in [x for x in PROTEASES if x in z.index]:
            for b in [x for x in T_GENES + AXIS if x in z.columns and x != a]:
                rows.append({"slide": s, "gene_a": a, "gene_b": b, "z": z.loc[a, b]})
    pd.DataFrame(rows).to_csv(tab("hotspot_focus_per_slide"), index=False)
    print(f"B: {len(rows)} per-slide protease × T/axis pairs ({time.time() - t0:.0f}s)")

    # C. Harreman, cell-type aware
    for key, tag in [("ct_comp", "comp"), ("ct_mal", "mal")]:
        if key == "ct_mal" and (adata.obs.mal_group[is_t] == "na").all():
            print("[mal] no transfer score in the 28e table; skipped"); continue
        try:
            ct_run(hm, adata, key, tag)
        except Exception:
            traceback.print_exc()
    print(f"done in {time.time() - t0:.0f}s")


PROTEASES = [g for f in PROTEASE_FAMS for g in cfg.DEGRADOME_FAMILIES[f]]
AXIS = sorted({g for pair in CS.AXIS_PAIRS for g in pair})
FOCUS = list(dict.fromkeys(PROTEASES + T_GENES + AXIS))

if __name__ == "__main__":
    main()
