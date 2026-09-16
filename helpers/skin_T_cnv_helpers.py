"""Helpers for the skin-T TCR + inferCNV malignancy notebooks (14 and 15).

Moved out of the notebooks to keep them short. The notebooks orchestrate; this module
holds the mechanical blocks (TCR fold-in, dominant-clone rule, inferCNV prep/run,
GMM caller, MRVI smoothing, chromosome heatmaps).

Shared by both:
- 14_skin_T_tcr_cnv_malignancy  — shared HC/healthy diploid reference.
- 15_skin_cd4_cd8ref_cnv_malignancy — per-sample CD8 reference, CD4 query
  (`prepare_cd4_cd8ref_inputs`, `run_per_sample_cd8ref_infercnv`).
The TCR / GMM-caller / MRVI-smoothing / strategy-agreement / heatmap blocks are generic
across both; the heatmap fns take `shared_ref=None` when the reference is already in `acnv`.
"""
from __future__ import annotations

import gc
import hashlib
from pathlib import Path

import anndata as ad
import infercnvpy as cnv
import numpy as np
import pandas as pd
import scanpy as sc

# default reference benign-T (49-type) labels for the external integrated atlas
BENIGN_T = ["Tc", "Th", "Treg", "Tc17_Th17", "Tc_IL13_IL22"]

# CELLxGENE cell_type labels counted as CD4 in the external healthy-PBMC reference (OneK1K).
ONEK1K_CD4 = ["CD4-positive, alpha-beta T cell",
              "CD4-positive, alpha-beta cytotoxic T cell",
              "central memory CD4-positive, alpha-beta T cell",
              "effector memory CD4-positive, alpha-beta T cell",
              "naive thymus-derived CD4-positive, alpha-beta T cell",
              "regulatory T cell"]


# ---------------------------------------------------------------- Step 1: load
def build_or_load_tcr_object(obj: Path, tcr_obj: Path, li_tcr: Path, H):
    """Load the TCR-complete T object, or build+cache it on first run.

    Build folds in Li2024/Haniffa TCR (parquet keyed by barcode after '|') then writes
    the ~12 GB cache. Returns adata with `cached_malignant` set.
    """
    if tcr_obj.exists():
        adata = sc.read_h5ad(tcr_obj)
        print("loaded cached TCR-complete object ->", tcr_obj.name, adata.shape)
    else:
        adata = sc.read_h5ad(obj)
        # nb10b (v2) writes the T cache with raw counts in layers['counts']; every consumer below
        # -- prepare_skin_cnv_inputs, the arm pass -- reads the atlas convention 'raw_counts'.
        if "raw_counts" not in adata.layers and "counts" in adata.layers:
            adata.layers["raw_counts"] = adata.layers["counts"]
            del adata.layers["counts"]
        assert "raw_counts" in adata.layers, f"no raw counts layer, only {list(adata.layers)}"
        bc_key = adata.obs_names.to_series().str.split("|", n=1).str[1]
        is_li = adata.obs["study"].astype(str).eq("li2024").to_numpy()
        li = pd.read_parquet(li_tcr)
        print("li2024 parquet:", li.shape, "| key overlap:",
              int(np.isin(bc_key[is_li], li.index).sum()), "/", int(is_li.sum()))
        li_bare = (li["tcr_clone_id"].where(li["has_tcr"].astype(bool), "").astype(str)
                   .str.split("::", n=1).str[-1])              # bare 'TRB:..' key
        adata.obs["_li_clone_key"] = ""
        adata.obs.loc[is_li, "_li_clone_key"] = bc_key[is_li].map(li_bare).fillna("").values
        has = adata.obs["has_tcr"].astype(bool).to_numpy().copy()
        has[is_li] = bc_key[is_li].map(li["has_tcr"].astype(bool)).fillna(False).values
        adata.obs["has_tcr"] = has
        adata.write_h5ad(tcr_obj)                              # HEAVY I/O (~12 GB)
        print("built + cached TCR-complete object ->", tcr_obj.name, adata.shape)

    adata.obs["cached_malignant"] = (adata.obs["cell_type"].astype(str) == "tumor_cell")
    print("\nstudy:\n", adata.obs["study"].value_counts())
    print("\nhas_tcr by study:\n",
          adata.obs.groupby("study", observed=True)["has_tcr"].agg(["size", "sum"]))
    print("cached tumor_cell:", int(adata.obs["cached_malignant"].sum()), "/", adata.n_obs)
    return adata


# ----------------------------------------------------- Step 2: TCR malignancy
def recompute_dominant_clone(adata, H, is_li, frac_thresh, ratio_thresh, expanded_min,
                             dom_mask=None):
    """Unified TRB-primary clone key + per-donor dominant-clone malignancy.

    ``dom_mask`` (bool array over obs) restricts the dominance test to a lineage --
    e.g. CD4 only, so a co-expanded reactive CD8 clone cannot compete for "top clone".
    Clone id / size / expansion always use all TCR+ cells; only the dominance decision
    narrows. ``None`` keeps the legacy pooled behaviour.

    Sets obs: tcr_clone_id, tcr_clone_size, tcr_is_expanded, tcr_is_dominant_clone,
    tcr_is_malignant. Returns the per-donor dominance table.
    """
    tra = adata.obs["tra_cdr3"].astype(str).fillna("").values
    trb = adata.obs["trb_cdr3"].astype(str).fillna("").values
    key_cdr3 = np.array([H.clone_id_from_cdr3(a, b) for a, b in zip(tra, trb)])
    clone_key = np.where(is_li, adata.obs["_li_clone_key"].astype(str).values, key_cdr3)
    clone_key[~adata.obs["has_tcr"].to_numpy()] = ""          # only TCR+ cells carry a key
    adata.obs["tcr_clone_id"] = clone_key

    obs = adata.obs
    tcr = obs["has_tcr"].to_numpy() & (clone_key != "")
    in_pool = np.ones(adata.n_obs, dtype=bool) if dom_mask is None else np.asarray(dom_mask, dtype=bool)
    df = pd.DataFrame({"donor": obs["donor"].astype(str).values,
                       "clone": clone_key, "pool": in_pool}, index=obs.index)[tcr]

    clone_size = pd.Series(0, index=obs.index, dtype=int)
    is_dom = pd.Series(False, index=obs.index)
    dom_rows = []
    for d, sub in df.groupby("donor", sort=False):
        sizes = sub["clone"].value_counts()
        clone_size.loc[sub.index] = sub["clone"].map(sizes).astype(int).values
        pool = sub[sub["pool"]]                       # dominance is judged on this subset only
        dom_sizes = pool["clone"].value_counts()
        if not len(dom_sizes):
            dom_rows.append({"donor": d, "n_tcr": len(sub), "n_dom_pool": 0, "top_clone": "",
                             "top_n": 0, "dom_frac": 0.0, "ratio": 0.0, "is_dominant": False})
            continue
        n = len(pool); top = dom_sizes.index[0]; top_n = int(dom_sizes.iloc[0])
        second_n = int(dom_sizes.iloc[1]) if len(dom_sizes) > 1 else 0
        dom_frac = top_n / max(1, n)
        ratio = top_n / second_n if second_n else np.inf
        is_dom_donor = (dom_frac >= frac_thresh) and (ratio >= ratio_thresh)
        if is_dom_donor:
            is_dom.loc[pool.index[pool["clone"] == top]] = True
        dom_rows.append({"donor": d, "n_tcr": len(sub), "n_dom_pool": n,
                         "top_clone": top, "top_n": top_n,
                         "dom_frac": round(dom_frac, 3),
                         "ratio": round(ratio, 2) if np.isfinite(ratio) else np.inf,
                         "is_dominant": is_dom_donor})

    adata.obs["tcr_clone_size"]        = clone_size.values
    adata.obs["tcr_is_expanded"]       = (clone_size >= expanded_min).values & tcr
    adata.obs["tcr_is_dominant_clone"] = is_dom.values
    adata.obs["tcr_is_malignant"]      = is_dom.values        # dominant clone == TCR malignant
    return pd.DataFrame(dom_rows).sort_values("donor")


def clone_summary_table(adata, frac_thresh, ratio_thresh, dom_mask=None):
    """Per-donor clone summary (largest clone, fold-change, malignant call).

    ``dom_mask`` mirrors ``recompute_dominant_clone`` so the ``malignant`` column stays
    consistent with ``obs["tcr_is_dominant_clone"]``.
    """
    obs = adata.obs
    has = obs["has_tcr"].astype(str).isin(["True", "1"]).to_numpy()
    tcr = has & (obs["tcr_clone_id"].astype(str).to_numpy() != "")
    in_pool = np.ones(adata.n_obs, dtype=bool) if dom_mask is None else np.asarray(dom_mask, dtype=bool)
    pc = pd.DataFrame({"donor": obs["donor"].astype(str).values,
                       "clone_id": obs["tcr_clone_id"].astype(str).values,
                       "pool": in_pool})[tcr]
    rows = []
    for sid, sub in pc.groupby("donor", sort=False):
        pool = sub[sub["pool"]]                       # n_tcr_cells stays all-T (cohort size gate)
        sizes = pool["clone_id"].value_counts()
        n = len(pool)
        top_n = int(sizes.iloc[0]) if len(sizes) else 0
        second_n = int(sizes.iloc[1]) if len(sizes) > 1 else 0
        dom_frac = top_n / max(1, n); ratio = top_n / second_n if second_n else np.inf
        rows.append({
            "donor": sid, "n_tcr_cells": len(sub), "n_dom_pool": n, "n_clones": int(len(sizes)),
            "largest_clone": top_n, "second_clone": second_n,
            "fold_change": ratio, "dom_frac": dom_frac,
            "pass_frac (>=%.2f)" % frac_thresh: dom_frac >= frac_thresh,
            "pass_ratio (>=%.1f)" % ratio_thresh: ratio >= ratio_thresh,
            "malignant": bool(top_n) and (dom_frac >= frac_thresh) and (ratio >= ratio_thresh),
        })
    return pd.DataFrame(rows).sort_values("largest_clone", ascending=False).reset_index(drop=True)


# --------------------------------------------------- Step 3: inferCNV inputs
def _clean_ref(a, label):
    a = a.copy()
    a.X = a.layers["raw_counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    out = ad.AnnData(X=a.X.copy(),
                     obs=pd.DataFrame({"cnv_ref": label, "donor": f"{label.upper()}_REF",
                                       "cell_type": a.obs["cell_type"].astype(str).values},
                                      index=a.obs_names.astype(str)),
                     var=pd.DataFrame(index=a.var_names.astype(str)))
    out.var_names_make_unique()
    return out


def _read_healthy_ref_rows(integrated_h5: Path, rng, n_healthy_ref, benign_t):
    """Gather only the selected benign-T reference rows' `raw_counts` directly via h5py.

    The integrated atlas is a ~886k-cell uncompressed CSR h5ad (~16 GB X + ~16 GB
    raw_counts). anndata's backed `rf[sel].to_memory()` materialises the *full* X and
    layer just to keep a few thousand rows — a ~32 GB read that stalls for ~1 h on the
    login node. Here we read the per-row CSR slices of `raw_counts` for the selected
    rows only (a few MB). Selection is identical to the old path (same string mask, same
    `rng.choice`) so the reference — and every downstream CNV score — is unchanged.
    """
    import h5py
    from scipy import sparse

    with h5py.File(integrated_h5, "r") as f:
        def _cat(col):
            g = f["obs"][col]
            return g["categories"].asstr()[:][g["codes"][:]]

        sample_type = _cat("sample_type")
        cell_type = _cat("cell_type")
        genes = f["var"]["genes"].asstr()[:]

        mask = (np.isin(sample_type, ["healthy_skin", "AD", "psoriasis"])
                & np.isin(cell_type, benign_t))
        sel = np.where(mask)[0]
        if sel.size > n_healthy_ref:
            sel = np.sort(rng.choice(sel, n_healthy_ref, replace=False))

        idx_key = f["obs"].attrs.get("_index", "_index")
        obs_names = f["obs"][idx_key].asstr()[sel]

        rc = f["layers"]["raw_counts"]
        n_genes = int(rc.attrs["shape"][1])
        indptr = rc["indptr"][:]
        data_ds, ind_ds = rc["data"], rc["indices"]
        data_parts, ind_parts, new_indptr = [], [], [0]
        for i in sel:
            a, b = int(indptr[i]), int(indptr[i + 1])
            data_parts.append(data_ds[a:b])
            ind_parts.append(ind_ds[a:b])
            new_indptr.append(new_indptr[-1] + (b - a))
        X = sparse.csr_matrix(
            (np.concatenate(data_parts) if data_parts else np.empty(0, data_ds.dtype),
             np.concatenate(ind_parts) if ind_parts else np.empty(0, ind_ds.dtype),
             np.asarray(new_indptr, dtype=np.int64)),
            shape=(sel.size, n_genes))
        ct_sel = cell_type[sel]

    return ad.AnnData(
        X=X,
        obs=pd.DataFrame({"cnv_ref": "healthy", "donor": "HEALTHY_REF", "cell_type": ct_sel},
                         index=pd.Index(obs_names.astype(str))),
        var=pd.DataFrame(index=pd.Index(genes.astype(str))))


def _h5_str_col(group, col):
    """Read an h5ad obs/var column as strings, whether categorical or a plain string dataset."""
    g = group[col]
    if hasattr(g, "keys") and "categories" in g:
        return g["categories"].asstr()[:][g["codes"][:]]
    return g.asstr()[:]


def _h5_csr_rows(g, sel):
    """Read only rows `sel` of a CSR-encoded h5ad matrix group. Returns scipy csr_matrix."""
    from scipy import sparse

    n_cols = int(g.attrs["shape"][1])
    indptr = g["indptr"][:]
    data_ds, ind_ds = g["data"], g["indices"]
    data_parts, ind_parts, new_indptr = [], [], [0]
    for i in sel:
        a, b = int(indptr[i]), int(indptr[i + 1])
        data_parts.append(data_ds[a:b])
        ind_parts.append(ind_ds[a:b])
        new_indptr.append(new_indptr[-1] + (b - a))
    return sparse.csr_matrix(
        (np.concatenate(data_parts) if data_parts else np.empty(0, data_ds.dtype),
         np.concatenate(ind_parts) if ind_parts else np.empty(0, ind_ds.dtype),
         np.asarray(new_indptr, dtype=np.int64)),
        shape=(len(sel), n_cols))


def _h5_csr_rows_blocked(g, sel, block=20000):
    """Read rows `sel` of a CSR h5ad matrix in contiguous file blocks. Returns csr in `sel` order.

    `_h5_csr_rows` issues two dataset reads per row, which costs minutes for a few thousand rows
    and hours for a few hundred thousand. Here each block of `block` consecutive *file* rows is
    read with one `data[a:b]` / `indices[a:b]` slice and then subset, so the number of reads
    scales with the file, not with `len(sel)`. Blocks holding no selected row are skipped.
    """
    from scipy import sparse

    n_rows, n_cols = (int(x) for x in g.attrs["shape"])
    indptr = g["indptr"][:]
    data_ds, ind_ds = g["data"], g["indices"]
    sel = np.asarray(sel)
    order = np.argsort(sel, kind="stable")
    ssel = sel[order]
    blocks = []
    for start in range(0, n_rows, block):
        stop = min(start + block, n_rows)
        lo, hi = np.searchsorted(ssel, start), np.searchsorted(ssel, stop)
        if hi <= lo:
            continue
        a, b = int(indptr[start]), int(indptr[stop])
        blk = sparse.csr_matrix(
            (data_ds[a:b].astype(np.float32, copy=False), ind_ds[a:b],
             (indptr[start:stop + 1] - a).astype(np.int64)),
            shape=(stop - start, n_cols))
        blocks.append(blk[ssel[lo:hi] - start])
    out = sparse.vstack(blocks, format="csr")
    inv = np.empty_like(order)
    inv[order] = np.arange(sel.size)
    return out[inv]


def build_fullgene_blood_cnv_object(joint_h5: Path, annotated_h5: Path, cache: Path, *,
                                    min_cells=20, block=20000, force=False):
    """Re-express the annotated blood-T object in the *full* gene space, for inferCNV.

    `blood_T_annotated.h5ad` carries only the 10k HVG set: ~7.5k of those genes have GTF
    positions, so with `window_size=250` one inferCNV window spans ~3 % of the genome, and HVGs
    are selected for exactly the state-driven variance inferCNV assumes away. This pulls the same
    cells out of `joint_annotated.h5ad` (40,821 genes) instead, keeping the annotated `obs`
    verbatim, so nothing upstream of the CNV step has to be recomputed.

    Genes are filtered to those detected in >= `min_cells` of the selected cells — an expression
    filter, not a variance filter. Rows are matched on `cell_id` (both files index by it) and the
    matrix is read block-wise via h5py; the 48 GB source is never opened with `sc.read_h5ad`.

    Writes `cache` with raw counts in both `X` and `layers['raw_counts']` (what
    `prepare_blood_pooled_cnv_inputs` expects) and returns it.
    """
    if cache.exists() and not force:
        out = sc.read_h5ad(cache)
        print(f"loaded cached full-gene CNV object ({cache.name}): {out.shape}")
        return out

    import h5py

    assert joint_h5.exists(), f"missing {joint_h5}"
    assert annotated_h5.exists(), f"missing {annotated_h5}"

    obs = sc.read_h5ad(annotated_h5, backed="r").obs.copy()      # backed: obs only, no matrix
    want = obs.index.astype(str).to_numpy()

    with h5py.File(joint_h5, "r") as f:
        joint_ids = f["obs"][f["obs"].attrs.get("_index", "cell_id")].asstr()[:]
        pos = pd.Index(joint_ids).get_indexer(want)
        missing = int((pos < 0).sum())
        assert not missing, f"{missing} of {want.size} cell_ids absent from {joint_h5.name}"
        genes = f["var"][f["var"].attrs.get("_index", "_index")].asstr()[:]
        print(f"reading {pos.size} rows x {genes.size} genes from {joint_h5.name} "
              f"(block={block}) ...")
        X = _h5_csr_rows_blocked(f["layers"]["raw_counts"], pos, block=block)

    n_cells_per_gene = np.bincount(X.indices, minlength=X.shape[1])
    keep = np.where(n_cells_per_gene >= min_cells)[0]
    print(f"genes detected in >= {min_cells} cells: {keep.size} / {X.shape[1]}")
    X = X[:, keep]

    out = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(genes[keep].astype(str))))
    out.var_names_make_unique()
    out.layers["raw_counts"] = out.X
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(cache)
    print(f"wrote full-gene CNV object -> {cache}: {out.shape}")
    return out


def build_healthy_pbmc_ref(onek1k_h5: Path, cache: Path, *, n_donors=300, n_per_donor=40,
                           seed=0, cd4_labels=ONEK1K_CD4, force=False):
    """Cache a small healthy-PBMC CD4 reference sampled across many donors (OneK1K).

    The blood atlas has exactly one healthy blood sample (`H__HC1_Blood`), so an in-atlas diploid
    reference is a single batch. This draws `n_donors` x `n_per_donor` CD4 cells from OneK1K
    (981 healthy donors, 10x 3' v2) — donor spread, not raw cell count, is what keeps any one
    batch from setting the inferCNV baseline.

    Reads only the selected CSR rows via h5py (as `_read_healthy_ref_rows` does): the source is a
    1.25M-cell / 35.5k-gene h5ad and `sc.read_h5ad` on it would stall the login node. CELLxGENE
    schema 7.1.0 puts raw counts in `X`; `var` is indexed by Ensembl id with symbols in
    `var["feature_name"]`, which is what the atlas gene space uses.

    Writes `cache` (h5ad, raw counts in X) and returns it. Download the source first with
    `jobs/download_onek1k.sh`.
    """
    if cache.exists() and not force:
        out = sc.read_h5ad(cache)
        print(f"loaded cached healthy-PBMC ref ({cache.name}): {out.n_obs} cells, "
              f"{out.obs['ref_donor'].nunique()} donors")
        return out

    import h5py

    assert onek1k_h5.exists(), f"missing {onek1k_h5} — run jobs/download_onek1k.sh first"
    rng = np.random.default_rng(seed)
    with h5py.File(onek1k_h5, "r") as f:
        cell_type = _h5_str_col(f["obs"], "cell_type")
        donor_id = _h5_str_col(f["obs"], "donor_id")
        is_cd4 = np.isin(cell_type, list(cd4_labels))
        assert is_cd4.any(), f"no cells matched cd4_labels; present: {sorted(set(cell_type))[:40]}"

        # sample donors first, then cells within each donor
        cd4_idx = np.where(is_cd4)[0]
        df = pd.DataFrame({"i": cd4_idx, "d": donor_id[cd4_idx]})
        groups = {d: s.to_numpy() for d, s in df.groupby("d", sort=True)["i"]}
        eligible = np.array([d for d, v in groups.items() if v.size >= n_per_donor])
        assert eligible.size, f"no donor has >= {n_per_donor} CD4 cells"
        pick = eligible if eligible.size <= n_donors else rng.choice(eligible, n_donors, replace=False)
        sel = np.sort(np.concatenate(
            [rng.choice(groups[d], n_per_donor, replace=False) for d in pick]))

        idx_key = f["obs"].attrs.get("_index", "_index")
        obs_names = f["obs"][idx_key].asstr()[sel]
        X = _h5_csr_rows(f["X"], sel)
        symbols = _h5_str_col(f["var"], "feature_name")
        ct_sel, donor_sel = cell_type[sel], donor_id[sel]

    out = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"cnv_ref": "healthy_pbmc", "donor": "HEALTHY_PBMC_REF",
                          "ref_donor": donor_sel, "cell_type": ct_sel},
                         index=pd.Index(obs_names.astype(str))),
        var=pd.DataFrame(index=pd.Index(symbols.astype(str))))
    out.var_names_make_unique()
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(cache)
    print(f"built healthy-PBMC ref -> {cache}: {out.n_obs} cells, {len(pick)} donors, "
          f"{out.n_vars} genes")
    return out


def prepare_infercnv_inputs(adata, gtf: Path, integrated_h5: Path, seed: int,
                            n_hc_ref=5000, n_healthy_ref=3000, benign_t=BENIGN_T,
                            use_external=True):
    """Build the inferCNV query + shared diploid reference.

    Query = annotated T cells (cell_type_T != UNK) of disease-bearing donors. Reference =
    within-donor non-clonal T (`nonclonal`) + atlas HC T (`hc_atlas`, preferred) + external
    healthy/AD/pso benign-T (`healthy`, fallback). Adds genomic positions from `gtf`.
    Returns (acnv, shared_ref, cnv_donors, hc_donors).
    """
    ct      = adata.obs["cell_type"].astype(str)
    ctT     = adata.obs["cell_type_T"].astype(str)
    disease = adata.obs["disease"].astype(str)
    donor   = adata.obs["donor"].astype(str)

    hc_donors = sorted(donor[disease.eq("HC")].unique())
    qmask     = (~donor.isin(hc_donors)) & ctT.ne("UNK")
    cnv_donors = sorted(donor[qmask].unique())
    print("HC donors held out:", len(hc_donors), "| CNV (query) donors:", len(cnv_donors),
          "| query cells:", int(qmask.sum()))

    # ---- query: T cells of disease donors, log-normalised ----
    acnv = adata[qmask.to_numpy()].copy()
    acnv.X = acnv.layers["raw_counts"].copy()
    sc.pp.normalize_total(acnv, target_sum=1e4)
    sc.pp.log1p(acnv)
    acnv.obs["cnv_ref"] = "query"
    nonclonal = (acnv.obs["has_tcr"].to_numpy()
                 & ~acnv.obs["tcr_is_dominant_clone"].to_numpy()
                 & (acnv.obs["cell_type"].astype(str) != "tumor_cell").to_numpy())
    acnv.obs.loc[nonclonal, "cnv_ref"] = "nonclonal"

    # ---- atlas-internal HC T reference (same platform; preferred) ----
    rng = np.random.default_rng(seed)
    hc_idx = np.where((disease.eq("HC") & ctT.ne("UNK")).to_numpy())[0]
    if hc_idx.size > n_hc_ref:
        hc_idx = np.sort(rng.choice(hc_idx, n_hc_ref, replace=False))
    assert hc_idx.size > 0, "no HC T cells found for the hc_atlas reference"
    hc_ref = _clean_ref(adata[hc_idx], "hc_atlas")
    print(f"hc_atlas reference: {hc_ref.n_obs} cells")
    refs = [hc_ref]

    # ---- external healthy/AD/pso benign-T reference (fallback) ----
    if use_external and integrated_h5.exists():
        healthy = _read_healthy_ref_rows(integrated_h5, rng, n_healthy_ref, benign_t)
        healthy.var_names_make_unique()
        sc.pp.normalize_total(healthy, target_sum=1e4)
        sc.pp.log1p(healthy)
        refs.append(healthy)
        print(f"external healthy reference: {healthy.n_obs} cells")

    # ---- common gene space + genomic positions; one shared reference ----
    common = acnv.var_names
    for r in refs:
        common = common.intersection(r.var_names)
    acnv = acnv[:, common].copy()
    cnv.io.genomic_position_from_gtf(gtf, adata=acnv, gtf_gene_id="gene_name")
    shared_ref = ad.concat([r[:, common] for r in refs], join="inner", index_unique=None)
    shared_ref.var = acnv.var.copy()                          # share genomic positions

    n_annot = int(acnv.var[["chromosome", "start", "end"]].notna().all(axis=1).sum())
    print(f"common genes: {len(common)} | with genomic position: {n_annot} / {acnv.n_vars}")
    assert n_annot > 8000, "too few genes annotated — check GTF gene_name / symbol intersection"
    print("query cells:", acnv.n_obs,
          "| within-donor nonclonal:", int((acnv.obs["cnv_ref"] == "nonclonal").sum()),
          "| shared ref:", shared_ref.n_obs, dict(shared_ref.obs["cnv_ref"].value_counts()))
    return acnv, shared_ref, cnv_donors, hc_donors


def prepare_skin_cnv_inputs(adata, gtf: Path, integrated_h5: Path, seed: int, *,
                            n_healthy_ref=10000, benign_t=BENIGN_T, hc_ref_frac=2 / 3,
                            cd8_ref_cap=2000, std_chr=None, min_positioned=12000,
                            hc_disease="HC"):
    """Skin CD4 inferCNV query + pooled diploid reference (nb30 v4).

    The v3 path (`prepare_infercnv_inputs` + the self-referential `call_per_donor`) called ~half of
    every tumour-free donor malignant by construction and consumed the HC donors as reference, so it
    had no held-out negative control. This mirrors the nb34 blood rebuild: three diploid categories
    whose biases differ and partly cancel, half of them held out as an explicit diploid null.

      cd8_ref    same-donor non-dominant CD8   right batch, wrong lineage   (capped per donor)
      hc_atlas   atlas HC T                    right tissue + lineage       (few donors)
      healthy    external healthy/AD/pso T     right lineage, many donors, wrong batch

    `nonclonal` (within-donor non-dominant, non-tumour T) is still *labelled* so the arm pass keeps
    those rows, but — unlike v3 — it is NOT a baseline: in a high-burden donor it carries tumour, so
    it is scored like any query cell. `cd8_ref` lives inside `acnv` per donor; `hc_atlas` / `healthy`
    come back in `shared_ref`, which `run_per_donor_infercnv` concatenates onto every donor block.

    CD8 is *capped* (unlike blood, which moved all CD8 into the baseline): skin's reactive-CD8
    infiltrate is analysed downstream, so surplus CD8 beyond `cd8_ref_cap` per donor stays in the
    query and still gets a CNV score. Donors below ~40 CD8 simply get no `cd8_ref` category.

    The atlas HC T cells are split: `hc_ref_frac` -> the `hc_atlas` reference, the remainder stays in
    the query as a held-out negative control (`hc_control_names`) — the run's acceptance test, since
    a working caller must leave it near-zero malignant. HC donors therefore appear in `cnv_donors`
    (their control cells need a per-donor inferCNV run to be scored).

    Returns (acnv, shared_ref, cnv_donors, hc_control_names).
    """
    if std_chr is None:
        std_chr = [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]]
    ctT     = adata.obs["cell_type_T"].astype(str)
    disease = adata.obs["disease"].astype(str)
    rng = np.random.default_rng(seed)

    is_T  = ctT.ne("UNK").to_numpy()
    is_hc = disease.eq(hc_disease).to_numpy()

    # ---- atlas HC T: hc_ref_frac -> hc_atlas reference, remainder -> held-out control ----
    hc_T = np.where(is_hc & is_T)[0]
    assert hc_T.size, f"no {hc_disease} T cells found for the hc_atlas reference"
    n_ref = int(round(hc_ref_frac * hc_T.size))
    ref_pos = np.sort(rng.choice(hc_T, n_ref, replace=False))
    hc_ref_names     = adata.obs_names[ref_pos]
    hc_control_names = adata.obs_names[np.setdiff1d(hc_T, ref_pos)]
    hc_ref = _clean_ref(adata[hc_ref_names], "hc_atlas")
    print(f"atlas HC T: {hc_ref.n_obs} -> hc_atlas reference | "
          f"{len(hc_control_names)} -> held-out control (stays in query)")

    # ---- query = disease T cells + HC held-out control (HC ref cells excluded), log-normalised ----
    qmask = (is_T & ~is_hc) | adata.obs_names.isin(hc_control_names)
    acnv = adata[qmask].copy()
    acnv.X = acnv.layers["raw_counts"].copy()
    sc.pp.normalize_total(acnv, target_sum=1e4)
    sc.pp.log1p(acnv)
    acnv.obs["cnv_ref"] = "query"
    is_control = acnv.obs_names.isin(hc_control_names)

    # ---- cd8_ref: up to cd8_ref_cap non-dominant CD8 per donor (control never a baseline) ----
    is_cd8 = acnv.obs["cell_type_T"].astype(str).eq("CD8").to_numpy()
    nondom = ~acnv.obs["tcr_is_dominant_clone"].to_numpy()
    cd8_pool = is_cd8 & nondom & ~is_control
    dser = acnv.obs["donor"].astype(str)
    cd8_ref_names = []
    for d in pd.unique(dser[cd8_pool]):
        idx = np.where(cd8_pool & (dser == d).to_numpy())[0]
        if idx.size > cd8_ref_cap:
            idx = np.sort(rng.choice(idx, cd8_ref_cap, replace=False))
        cd8_ref_names.extend(acnv.obs_names[idx])
    acnv.obs.loc[cd8_ref_names, "cnv_ref"] = "cd8_ref"

    # ---- within-donor non-dominant non-tumour T -> 'nonclonal' (labelled, but scored not baseline) ----
    is_q = (acnv.obs["cnv_ref"] == "query").to_numpy()
    nonclonal = (is_q
                 & acnv.obs["has_tcr"].to_numpy()
                 & ~acnv.obs["tcr_is_dominant_clone"].to_numpy()
                 & (acnv.obs["cell_type"].astype(str) != "tumor_cell").to_numpy())
    acnv.obs.loc[nonclonal, "cnv_ref"] = "nonclonal"
    acnv.obs.loc[hc_control_names, "cnv_ref"] = "query"       # control is scored, never a baseline

    # ---- external healthy/AD/pso benign-T reference ----
    healthy = _read_healthy_ref_rows(integrated_h5, rng, n_healthy_ref, benign_t)
    healthy.var_names_make_unique()
    sc.pp.normalize_total(healthy, target_sum=1e4)
    sc.pp.log1p(healthy)
    print(f"external healthy reference: {healthy.n_obs} cells")

    # ---- common gene space + genomic positions; restrict to chr1-22/X/Y; one shared reference ----
    refs = [hc_ref, healthy]
    common = acnv.var_names
    for r in refs:
        common = common.intersection(r.var_names)
    acnv = acnv[:, common].copy()
    cnv.io.genomic_position_from_gtf(gtf, adata=acnv, gtf_gene_id="gene_name")
    acnv = acnv[:, acnv.var["chromosome"].astype(str).isin(std_chr)].copy()
    shared_ref = ad.concat([r[:, acnv.var_names] for r in refs], join="inner", index_unique=None)
    shared_ref.var = acnv.var.copy()                          # share genomic positions

    # detection depth on the SHARED gene space — query and reference must be counted over the same
    # genes for the depth-corrected null in call_per_donor_null / arm_consensus_score to mean anything
    for a in (acnv, shared_ref):
        a.obs["n_genes_cnv"] = np.asarray((a.X > 0).sum(axis=1)).ravel()

    # every donor with a query/nonclonal cell is a calling unit — incl. HC donors (their control)
    qn = acnv.obs["cnv_ref"].isin(["query", "nonclonal"]).to_numpy()
    cnv_donors = sorted(acnv.obs.loc[qn, "donor"].astype(str).unique())
    hc_control_names = hc_control_names[hc_control_names.isin(acnv.obs_names)]

    n_annot = int(acnv.var[["chromosome", "start", "end"]].notna().all(axis=1).sum())
    print("cnv_ref:", dict(acnv.obs["cnv_ref"].value_counts()),
          "| shared ref:", shared_ref.n_obs, dict(shared_ref.obs["cnv_ref"].value_counts()),
          "| genes on chr1-22/X/Y:", acnv.n_vars, "| positioned:", n_annot)
    print("query donors:", len(cnv_donors), "| held-out HC control cells:", len(hc_control_names))
    assert acnv.n_vars > min_positioned, (
        f"only {acnv.n_vars} positioned genes (< {min_positioned}) — check the GTF gene_name / "
        "symbol intersection with the external reference")
    return acnv, shared_ref, cnv_donors, hc_control_names


# ----------------------------------------------- Step 5: per-donor inferCNV
def _split_null(rng, shared_ref, q, null_frac, null_cats, donor):
    """Hold out a diploid null for one donor: part of the shared reference + part of `null_cats`.

    Draw order is fixed (shared block, then the donor's own block) so that a second pass with the
    same seed — `compute_arm_cnv_per_cell` — holds out exactly the same cells.
    Returns (relabelled shared reference, bool mask over `q`).
    """
    hold = rng.random(shared_ref.n_obs) < null_frac
    ref = shared_ref.copy()
    ref.obs["cnv_ref"] = np.where(hold, "ref_null", ref.obs["cnv_ref"].astype(str).to_numpy())
    ref.obs_names = np.where(hold, f"{donor}|NULL|" + ref.obs_names.astype(str),
                             ref.obs_names.astype(str))
    q_hold = np.zeros(q.n_obs, dtype=bool)
    draw = rng.random(q.n_obs)                       # drawn unconditionally: fixed rng order
    qref = q.obs["cnv_ref"].astype(str).to_numpy()
    for cat in (null_cats or ()):
        eligible = qref == cat
        # only hold out if the category still clears the 20-cell bar as a baseline afterwards
        if eligible.sum() * (1 - null_frac) >= 20:
            q_hold |= eligible & (draw < null_frac)
    return ref, q_hold


def _donor_seed(seed, donor):
    """Per-donor RNG seed that does not move between processes (`hash()` of a str is salted)."""
    h = hashlib.md5(f"{seed}|{donor}".encode()).hexdigest()[:8]
    return int(h, 16)


def _pooled_reference_vector(sub, cats):
    """Mean expression over all cells in `cats` as a single (1, n_genes) reference row.

    Passing this to `cnv.tl.infercnv(reference=...)` keeps the estimator on plain mean
    subtraction. With >= 2 `reference_cat` entries infercnvpy instead switches to a *bounded*
    difference — anything inside [min, max] of the per-category means becomes logFC 0 — so the
    estimator silently changes with the number of reference categories, and a dead-band built
    from categories as heterogeneous as 3' PBMC / 5' CD8 / tumour-contaminated CD4 suppresses
    real signal while leaving per-cell noise untouched.
    """
    m = np.isin(sub.obs["cnv_ref"].astype(str).to_numpy(), list(cats))
    X = sub[m].X
    mu = np.asarray(X.mean(axis=0)).ravel()
    return mu[np.newaxis, :]


def run_per_donor_infercnv(acnv, shared_ref, cnv_donors, cache: Path, *, window=250,
                           topk_frac=0.10, leiden_res=2.0, n_jobs=8, chunk=2500,
                           ref_cats=("nonclonal", "hc_atlas", "healthy"), ref_mode="bounded",
                           dynamic_threshold=1.5, shuffle=False, null_frac=0.0,
                           null_cats=("cd8_ref",), keep_ref_scores=False,
                           depth_col="n_genes_cnv", seed=0, force=False):
    """Per-donor inferCNV vs the shared reference; focal per-cell score; cached to parquet.

    Fills acnv.obs: cnv_score (per cnv_leiden cluster), cnv_cell_score, cnv_focal_score,
    cnv_leiden. Reloads `cache` unless `force` or the cache is missing donors.

    `ref_cats` lists the candidate `cnv_ref` baseline categories in priority order; a category is
    used for a donor only if that donor's block carries >= 20 of its cells. `cd8_ref` and
    `nonclonal` live inside `acnv` per sample, `hc_atlas` / `healthy_pbmc` come from `shared_ref`.

    Defaults reproduce the nb22/nb30 behaviour exactly. nb34 overrides all four estimator knobs:

    ref_mode "mean"      one pooled reference vector (plain mean subtraction) instead of
                         infercnvpy's bounded multi-category dead-band.
    dynamic_threshold    `None` disables the `1.5 * std(chunk)` zeroing, whose threshold depends
                         on how each *cell chunk* happens to be composed — chunks follow obs
                         order, so a donor with few query cells has them pooled with reference
                         cells (low std, more noise survives) while a large donor gets
                         tumour-only chunks (high std). That alone makes scores incomparable
                         across donors.
    shuffle              permute cells before chunking so chunk composition is uniform.
    null_frac            hold out this fraction of the baseline from the baseline and score it
                         as query. Those cells are known diploid and traverse the identical
                         pipeline, so their score distribution is the empirical null for that
                         donor (see `call_per_donor_null`). They are written to the cache as
                         `cnv_ref == "ref_null"`; the shared-reference ones are renamed
                         `<donor>|NULL|<cell>` since the same reference cell recurs in every run.
    null_cats            in-`acnv` categories to *also* hold out (default same-sample CD8). A
                         null made only of shared-reference cells is anti-conservative: those
                         cells built the baseline they are scored against, so they sit closer to
                         it than any query cell of a different batch/chemistry can. Same-sample
                         CD8 is batch-matched and lineage-mismatched, the external PBMC is the
                         reverse, and pooling them lets the noisier of the two set the tail.
    keep_ref_scores      also write the baseline reference rows to the cache (diagnostics).
    """
    acnv.obs["cnv_score"]       = np.nan   # per cnv_leiden CLUSTER (heatmaps/diagnostics)
    acnv.obs["cnv_cell_score"]  = np.nan   # per CELL: genome-wide mean |X_cnv|
    acnv.obs["cnv_focal_score"] = np.nan   # per CELL: mean of top-K% |X_cnv| (drives the call)
    acnv.obs["cnv_leiden"]      = ""

    score_cols = ["cnv_score", "cnv_cell_score", "cnv_focal_score"]
    keep_depth = bool(depth_col) and depth_col in acnv.obs.columns
    use_cache = (not force) and cache.exists()
    if use_cache:
        cc = pd.read_parquet(cache).set_index("obs_name")
        use_cache = ({"cnv_cell_score", "cnv_focal_score"}.issubset(cc.columns)
                     and set(cnv_donors).issubset(set(cc["donor"].astype(str).unique()))
                     and (null_frac <= 0 or "cnv_ref" in cc.columns))
    if use_cache:
        for col in score_cols:
            acnv.obs[col] = cc[col].reindex(acnv.obs_names).to_numpy()
        acnv.obs["cnv_leiden"] = cc["cnv_leiden"].reindex(acnv.obs_names).fillna("").to_numpy()
        print(f"loaded cached inferCNV ({cache.name}) for {len(cnv_donors)} donors")
        return cc if (null_frac > 0 or keep_ref_scores) else None

    extra, in_acnv_null = [], []                              # ref_null / reference rows
    for i, d in enumerate(cnv_donors):
        rng = np.random.default_rng(_donor_seed(seed, d))     # stable across processes
        q, ref = acnv[acnv.obs["donor"] == d], shared_ref
        if null_frac > 0:
            ref, q_hold = _split_null(rng, shared_ref, q, null_frac, null_cats, d)
            if q_hold.any():
                q = q.copy()
                q.obs.loc[q.obs_names[q_hold], "cnv_ref"] = "ref_null"
                in_acnv_null.extend(q.obs_names[q_hold].tolist())
        sub = ad.concat([q, ref], join="inner", index_unique=None)
        sub.var = acnv.var.loc[sub.var_names].copy()          # restore genomic positions
        if shuffle:
            sub = sub[rng.permutation(sub.n_obs)].copy()      # uniform chunk composition
        cats = [c for c in ref_cats if int((sub.obs["cnv_ref"] == c).sum()) >= 20]
        assert cats, f"[{d}] no reference category reached 20 cells (ref_cats={ref_cats})"
        n_nc = int((q.obs["cnv_ref"] == "nonclonal").sum())
        print(f"[{d}] query={int((q.obs['cnv_ref'] == 'query').sum()):>6}  "
              f"nonclonal={n_nc:>5}  null={int((sub.obs['cnv_ref'] == 'ref_null').sum()):>5}  "
              f"ref={cats}")
        kw = (dict(reference=_pooled_reference_vector(sub, cats)) if ref_mode == "mean"
              else dict(reference_key="cnv_ref", reference_cat=cats))
        cnv.tl.infercnv(sub, window_size=window, n_jobs=n_jobs, chunksize=chunk,
                        dynamic_threshold=dynamic_threshold, **kw)
        cnv.tl.pca(sub)
        cnv.pp.neighbors(sub)
        cnv.tl.leiden(sub, resolution=leiden_res)             # finer CNV clusters
        cnv.tl.cnv_score(sub)                                 # per-cluster -> obs['cnv_score']
        Xc = sub.obsm["X_cnv"]
        Xc = np.abs(Xc.toarray() if hasattr(Xc, "toarray") else np.asarray(Xc))
        k  = max(1, int(round(topk_frac * Xc.shape[1])))
        sub.obs["cnv_cell_score"]  = Xc.mean(axis=1)
        sub.obs["cnv_focal_score"] = np.partition(Xc, Xc.shape[1] - k, axis=1)[:, -k:].mean(axis=1)
        del Xc
        sub.obs["cnv_leiden"] = d + "_" + sub.obs["cnv_leiden"].astype(str)
        keep = (sub.obs["donor"] == d).to_numpy()             # this donor's own cells
        for col in score_cols:
            acnv.obs.loc[sub.obs_names[keep], col] = sub.obs.loc[keep, col].to_numpy()
        acnv.obs.loc[sub.obs_names[keep], "cnv_leiden"] = sub.obs.loc[keep, "cnv_leiden"].to_numpy()
        # rows to append: shared-reference cells only — the in-acnv nulls already ride along in
        # `keep`, and appending them again would duplicate their obs_name in the cache
        take = ((sub.obs["cnv_ref"] == "ref_null").to_numpy() & ~keep) if not keep_ref_scores \
            else ~keep
        if take.any():
            cols = score_cols + ["cnv_leiden", "cnv_ref"] + ([depth_col] if keep_depth else [])
            e = sub.obs.loc[take, cols].copy()
            e["donor"] = d
            extra.append(e)
        del sub, q, ref
        gc.collect()

    out = acnv.obs[["donor"] + score_cols + ["cnv_leiden"]].copy()
    out["cnv_ref"] = acnv.obs["cnv_ref"].astype(str).to_numpy()
    if in_acnv_null:
        out.loc[in_acnv_null, "cnv_ref"] = "ref_null"         # held out inside acnv, not renamed
    if keep_depth:
        out[depth_col] = acnv.obs[depth_col].to_numpy()
    if extra:
        out = pd.concat([out, pd.concat(extra, axis=0)[out.columns]], axis=0)
    out.index.name = "obs_name"
    out.reset_index().to_parquet(cache)
    print(f"computed inferCNV; cached -> {cache}  ({len(out)} rows, "
          f"{int((out['cnv_ref'] == 'ref_null').sum())} null)")
    return out if (null_frac > 0 or keep_ref_scores) else None


# ------------------------------------------------- Step 6: malignancy callers
def diploid_mask(acnv):
    """Benign/diploid cells: within-donor non-clonal T, or any non-tumor non-dominant T."""
    return (acnv.obs["cnv_ref"].eq("nonclonal")
            | ((acnv.obs["cell_type"].astype(str) != "tumor_cell")
               & ~acnv.obs["tcr_is_dominant_clone"]))


def _best_f1_thr(score, y):                                   # F1-optimal cut over a 99-quantile grid
    ok = np.isfinite(score)
    if ok.sum() < 50 or not (0 < y[ok].sum() < ok.sum()):
        return np.nan, np.nan, int(ok.sum())
    s, yy = score[ok], y[ok].astype(bool)
    best_f1, best_t = 0.0, np.nan
    for t in np.unique(np.percentile(s, np.linspace(1, 99, 99))):
        yp = s >= t
        tp = int((yp & yy).sum()); fp = int((yp & ~yy).sum()); fn = int((~yp & yy).sum())
        pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn); f1 = 2 * pr * rc / max(1e-9, pr + rc)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1, int(ok.sum())


def call_per_donor(score_col, diploid, donors, obs, verbose=True, seed=0, thr_scale=1.0):
    """Per-donor 2-comp GMM crossover on log(score) + diploid-median floor.

    `thr_scale` < 1 lowers the final threshold (more cells malignant), > 1 raises it.
    Returns a bool ndarray aligned to `obs`. Falls back to ref-p90 / global-p90 when the
    GMM is degenerate. Prints per-donor diagnostics (incl. TCR-F1-optimal cut) if verbose.
    """
    from sklearn.mixture import GaussianMixture

    call   = pd.Series(False, index=obs.index)
    method = pd.Series("", index=obs.index, dtype=object)
    dip = diploid.to_numpy() if hasattr(diploid, "to_numpy") else np.asarray(diploid)
    for d in donors:
        m = (obs["donor"] == d).to_numpy()
        s = obs.loc[m, score_col].to_numpy(dtype=float)
        finite = np.isfinite(s) & (s > 0)
        ref_s = obs.loc[m & dip, score_col].to_numpy(dtype=float)
        ref_s = ref_s[np.isfinite(ref_s) & (ref_s > 0)]
        floor = np.nan
        if ref_s.size >= 20:
            ref_s = ref_s[ref_s <= np.percentile(ref_s, 90)]
            floor = float(np.median(ref_s))
        thr, meth = np.nan, "gmm_xover"
        if finite.sum() >= 50:
            x = np.log(s[finite]).reshape(-1, 1)
            gm = GaussianMixture(2, random_state=seed, n_init=3).fit(x)
            mu = gm.means_.ravel(); lo, hi = np.argsort(mu)
            n_hi = int((gm.predict(x) == hi).sum())
            if (np.exp(mu[hi] - mu[lo]) > 1.1) and n_hi >= 20:
                grid = np.linspace(mu[lo], mu[hi], 2001).reshape(-1, 1)
                post = gm.predict_proba(grid)[:, hi]
                thr = float(np.exp(grid[np.argmin(np.abs(post - 0.5)), 0]))
        if not np.isfinite(thr):
            thr = float(np.percentile(ref_s, 90)) if ref_s.size >= 20 else float(np.nanpercentile(s, 90))
            meth = "ref_p90_fallback" if ref_s.size >= 20 else "global_p90_fallback"
        if np.isfinite(floor):
            thr = max(thr, floor)
        thr = thr * thr_scale                                 # nudge the cut
        c = np.where(np.isfinite(s), s > thr, False)
        call.loc[obs.index[m]]   = c
        method.loc[obs.index[m]] = meth
        if verbose:
            tcr_m = m & obs["has_tcr"].to_numpy()
            yt = obs.loc[tcr_m, "tcr_is_malignant"].to_numpy().astype(bool)
            f1t, f1v, n_tcr = _best_f1_thr(obs.loc[tcr_m, score_col].to_numpy(dtype=float), yt)
            msg = f"  | TCR-F1opt thr={f1t:.4f} f1={f1v:.2f} n={n_tcr}" if np.isfinite(f1t) else ""
            print(f"[{d}] {meth:18s} thr={thr:.4f}  malignant={int(c.sum())}/{int(m.sum())}{msg}")
    return call.to_numpy(), method.to_numpy()


# ------------------------------- Step 6b: null-anchored caller (nb34, v3)
def _null_threshold_by_depth(null_score, null_depth, query_depth, q=0.99, n_bins=5, min_bin=50):
    """Depth-conditional `q` quantile of the diploid null; one threshold per query cell.

    |X_cnv| is a magnitude statistic that falls as detection rises, so a null drawn from cells
    deeper than the query sets the bar too low — and the external PBMC reference (10x 3') is not
    depth-matched to the query (5'/ECCITE), with per-donor median detection spanning 830-3,356
    genes. Query cells are binned on their own depth quantiles and each bin is judged against the
    null cells *in that bin*; bins with too few null cells widen to their neighbours and are
    reported as uncovered rather than silently extrapolated. Non-parametric, so it survives the
    non-monotone depth/score relationship seen within donors.

    Returns (thr per query cell, info dict) or (None, info) when the null is unusable.
    """
    s = np.asarray(null_score, dtype=float)
    nq = np.size(query_depth) if query_depth is not None else 0
    ok = np.isfinite(s) & (s > 0)
    info = {"n_null": int(ok.sum()), "n_bins": 0, "frac_uncovered": np.nan}
    if ok.sum() < min_bin:
        return None, info
    s = s[ok]
    if null_depth is None or query_depth is None:
        info.update(n_bins=1, frac_uncovered=0.0)
        return np.full(nq, float(np.quantile(s, q))), info

    dn = np.asarray(null_depth, dtype=float)[ok]
    dq = np.asarray(query_depth, dtype=float)
    if not np.isfinite(dn).any() or not np.isfinite(dq).any():
        info.update(n_bins=1, frac_uncovered=0.0)
        return np.full(nq, float(np.quantile(s, q))), info

    cuts = np.unique(np.quantile(dq[np.isfinite(dq)], np.linspace(0, 1, n_bins + 1)))[1:-1]
    qb, nb = np.digitize(dq, cuts), np.digitize(dn, cuts)
    n_b = len(cuts) + 1
    thr = np.full(nq, float(np.quantile(s, q)))
    uncovered = 0
    for b in range(n_b):
        mq = qb == b
        if not mq.any():
            continue
        pool = nb == b
        if pool.sum() < min_bin:                       # widen to nearest bins, and say so
            uncovered += int(mq.sum())
            for bb in np.argsort(np.abs(np.arange(n_b) - b), kind="stable"):
                pool = pool | (nb == bb)
                if pool.sum() >= min_bin:
                    break
        if pool.sum() >= min_bin:
            thr[mq] = float(np.quantile(s[pool], q))
    info.update(n_bins=n_b, frac_uncovered=uncovered / max(1, nq))
    return thr, info


def call_per_donor_null(per_cell, donors, obs, *, score_col="cnv_focal_score", q=0.99,
                        depth_col=None, min_null=50, cluster_col="cnv_leiden",
                        cluster_min_cells=20, cluster_min_frac=0.5, run_col=None,
                        verbose=True):
    """Threshold each donor's cells against its own held-out diploid reference.

    Replaces `call_per_donor` for nb34. That caller took `max(GMM crossover on the donor's own
    scores, median of the donor's own diploid cells)`, both internal to the donor, so a
    tumour-free donor is cut at its own median and ~half its cells come back malignant by
    construction — the observed 61 % on the held-out healthy control. Here the cut comes from
    cells that are known diploid and went through the identical per-donor run: the `ref_null`
    split written by `run_per_donor_infercnv(null_frac=...)`.

    `per_cell` is that function's return value (or its parquet), indexed by obs_name and carrying
    `donor`, `cnv_ref` and the score columns. `obs` holds the cells to call (index must be a
    subset of `per_cell`). `depth_col`, if given, must be a column of `per_cell` present for both
    null and query rows (see `prepare_blood_pooled_cnv_inputs`, which stores `n_genes_cnv`).

    `run_col` separates the *calling* unit from the *inferCNV* unit. inferCNV runs per sample, so
    the held-out null belongs to the sample; a hashtag-pooled sample (`B4__SZ5` -> `SZ5_HTO1..6`)
    holds several patients that must be called separately but share one null. Set `run_col` to the
    sample-id column (present in both `per_cell` and `obs`) and `obs["donor"]` to the patient: each
    patient is then judged against the null of the run its cells went through. Without it, a null
    row can only serve the one donor whose id it happens to carry.

    Returns (call, thr_per_cell, diag) — a bool array and a float array aligned to `obs`, plus a
    per-donor DataFrame. Donors whose null is too small get `thr = nan` and no calls; the
    cluster-level columns in `diag` feed the callability test in `arm_consensus_score`.
    """
    pc = per_cell if isinstance(per_cell, pd.DataFrame) else pd.read_parquet(per_cell)
    if pc.index.name != "obs_name":
        pc = pc.set_index("obs_name")
    is_null = pc["cnv_ref"].astype(str).eq("ref_null").to_numpy()

    call = np.zeros(len(obs), dtype=bool)
    thr_out = np.full(len(obs), np.nan)
    rows = []
    donor_of = obs["donor"].astype(str).to_numpy()
    score = obs[score_col].to_numpy(dtype=float) if score_col in obs else \
        pc[score_col].reindex(obs.index).to_numpy(dtype=float)
    depth = (pc[depth_col].reindex(obs.index).to_numpy(dtype=float)
             if depth_col else None)
    clusters = (obs[cluster_col].astype(str).to_numpy() if cluster_col in obs
                else pc[cluster_col].reindex(obs.index).astype(str).to_numpy())
    run_pc = pc[run_col or "donor"].astype(str).to_numpy()
    run_obs = (obs[run_col].astype(str).to_numpy() if (run_col and run_col in obs)
               else pc[run_col or "donor"].reindex(obs.index).astype(str).to_numpy())

    for d in donors:
        m = donor_of == str(d)
        # the null of every inferCNV run this donor's cells went through (pooled samples share one)
        nsel = is_null & np.isin(run_pc, np.unique(run_obs[m]) if m.any() else [])
        t, info = _null_threshold_by_depth(
            pc.loc[nsel, score_col].to_numpy(),
            pc.loc[nsel, depth_col].to_numpy() if depth_col else None,
            depth[m] if depth is not None else (np.zeros(int(m.sum())) if m.any() else None),
            q=q, min_bin=min_null)
        n_null = int(nsel.sum())
        if t is None or not m.any():
            rows.append({"donor": str(d), "n_null": n_null, "n_query": int(m.sum()),
                         "thr_median": np.nan, "frac_called": np.nan,
                         "frac_uncovered": np.nan, "n_clusters_enriched": 0})
            if verbose:
                print(f"[{d}] null too small ({n_null} < {min_null}) — no calls")
            continue
        c = np.isfinite(score[m]) & (score[m] > t)
        call[m], thr_out[m] = c, t
        # a cluster is 'enriched' when most of it clears the null, not just a tail of it
        cl = pd.Series(c, index=clusters[m])
        sizes = cl.groupby(level=0).size()
        fracs = cl.groupby(level=0).mean()
        enr = (sizes >= cluster_min_cells) & (fracs >= cluster_min_frac)
        rows.append({"donor": str(d), "n_null": n_null, "n_query": int(m.sum()),
                     "thr_median": float(np.median(t)), "frac_called": float(c.mean()),
                     "frac_uncovered": info["frac_uncovered"],
                     "n_clusters_enriched": int(enr.sum()),
                     "enriched_clusters": ",".join(sorted(enr.index[enr.to_numpy()]))})
        if verbose:
            unc = ("" if not info["frac_uncovered"]
                   else f"  [{info['frac_uncovered']:.0%} of cells outside the null's depth range]")
            print(f"[{d}] null n={n_null:>5} thr={np.median(t):.4f}  "
                  f"called={int(c.sum())}/{int(m.sum())} ({c.mean():.1%})  "
                  f"enriched clusters={int(enr.sum())}/{len(sizes)}{unc}")
    return call, thr_out, pd.DataFrame(rows)


def arm_consensus_score(arm, per_cell, donors, *, arm_cols=None, cluster_col="cnv_leiden",
                        min_cluster=30, min_cluster_frac=0.05, z_thresh=5.0, min_amp=0.01,
                        max_event_arms=15, global_arm_frac=0.7, q=0.99, depth_col=None,
                        min_null=50, run_col=None, verbose=True):
    """De novo signed arm-level consensus call: direction and coherence, not magnitude.

    `cnv_focal_score` is `mean|X_cnv|` over the top windows, so scattered dropout noise scores
    like a clean chr7q gain — which is why the healthy control's false positives carry arm
    amplitudes (mean |arm| 0.0029) indistinguishable from real Sézary cells (0.0035). A real
    event is *signed*, *contiguous* and *shared across the clone*, so here each donor's own CNV
    clusters are tested arm-by-arm against the held-out diploid null:

      1. per arm, the null's mean and sd over that donor's `ref_null` cells;
      2. a cluster carries an event on arm a when its mean is `z_thresh` standard errors from
         the null mean *and* shifted by at least `min_amp` in absolute terms (large clusters
         make trivial shifts significant);
      3. a candidate cluster must hold at least `max(min_cluster, min_cluster_frac * n_query)`
         cells — a tumour clone is not a handful of cells, and a signature fitted to ~40 cells
         is fitted to their noise (the anchors below 2 % of the donor all landed at AUROC <= 0.66);
      4. a cluster shifted on more than `global_arm_frac` of all arms is a global offset, not a
         karyotype, and is rejected; one shifted on more than `max_event_arms` keeps only its
         `max_event_arms` strongest arms by |z| — high-burden Sezary karyotypes really do carry
         13-15 arms, so rejecting them outright left the anchor to a tiny residual cluster;
      5. the donor's signature is the signed event vector of the largest surviving cluster;
      6. every cell scores as its projection onto that unit signature, called against the `q`
         quantile of the same projection over the null cells.

    No TCR input anywhere, so the call stays independent evidence. A tumour-free donor has no
    cluster clearing the null, hence no signature and no calls — it comes back `indeterminate`
    rather than silently negative.

    `arm` is the per-cell arm matrix (from `compute_arm_cnv_per_cell`, including its `ref_null`
    rows); `per_cell` supplies `donor`, `cnv_ref` and `cluster_col`. `run_col` names the inferCNV
    unit when it is coarser than the calling unit, so pooled samples share their null — see
    `call_per_donor_null`. Returns (proj, call, state, sig_table) aligned to `arm.index`.
    """
    pc = per_cell if isinstance(per_cell, pd.DataFrame) else pd.read_parquet(per_cell)
    if pc.index.name != "obs_name":
        pc = pc.set_index("obs_name")
    if arm_cols is None:
        arm_cols = [c for c in arm.columns if str(c).startswith("chr")]
    A = arm[arm_cols].to_numpy(dtype=float)
    A = np.nan_to_num(A, nan=0.0)
    meta = pc.reindex(arm.index)
    donor_of = meta["donor"].astype(str).to_numpy()
    is_null = meta["cnv_ref"].astype(str).eq("ref_null").to_numpy()
    clusters = meta[cluster_col].astype(str).to_numpy()
    depth = meta[depth_col].to_numpy(dtype=float) if depth_col else None
    run_of = meta[run_col].astype(str).to_numpy() if run_col else donor_of

    proj = np.full(len(arm), np.nan)
    call = np.zeros(len(arm), dtype=bool)
    state = np.array(["indeterminate"] * len(arm), dtype=object)
    rows = []
    for d in donors:
        # query cells are this donor's; the null is every run they went through (pooled samples
        # share one null, so `m` is not simply `donor_of == d`)
        qm = (donor_of == str(d)) & ~is_null
        nm = is_null & np.isin(run_of, np.unique(run_of[qm]) if qm.any() else [])
        m = qm | nm
        if nm.sum() < min_null or not qm.any():
            rows.append({"donor": str(d), "callable": False, "reason": "null too small",
                         "n_null": int(nm.sum()), "n_event_arms": 0, "signature": ""})
            continue
        mu0, sd0 = A[nm].mean(axis=0), A[nm].std(axis=0, ddof=1)
        sd0 = np.where(sd0 > 0, sd0, np.inf)
        size_floor = max(min_cluster, int(round(min_cluster_frac * int(qm.sum()))))
        best, n_candidates = None, 0
        for cl in pd.unique(clusters[qm]):
            sel = qm & (clusters == cl)
            n = int(sel.sum())
            if n < size_floor:
                continue
            n_candidates += 1
            delta = A[sel].mean(axis=0) - mu0
            z = delta / (sd0 / np.sqrt(n))
            ev = (np.abs(z) > z_thresh) & (np.abs(delta) > min_amp)
            if not ev.any() or ev.sum() > global_arm_frac * len(arm_cols):
                continue
            if ev.sum() > max_event_arms:          # keep the strongest arms, don't drop the clone
                keep_arms = np.argsort(-np.abs(np.where(ev, z, 0.0)))[:max_event_arms]
                ev = np.zeros_like(ev)
                ev[keep_arms] = True
            if best is None or n > best[0]:
                best = (n, cl, np.where(ev, delta, 0.0), ev)
        if best is None:
            why = ("no cluster clears null" if n_candidates
                   else f"no cluster >= {size_floor} cells")
            rows.append({"donor": str(d), "callable": False, "reason": why,
                         "n_null": int(nm.sum()), "n_event_arms": 0, "signature": ""})
            if verbose:
                print(f"[{d}] indeterminate — {why} "
                      f"({n_candidates} candidate clusters >= {size_floor} cells)")
            continue
        n_best, cl_best, sig, ev = best
        u = sig / np.linalg.norm(sig)
        p = (A[m] - mu0) @ u
        proj[m] = p
        nm_local, qm_local = is_null[m], ~is_null[m]
        # the projection of a diploid cell is mean-zero but noisier when the cell is shallow,
        # so the cut is conditioned on depth exactly as the score threshold is
        t, info = _null_threshold_by_depth(p[nm_local],
                                           depth[m][nm_local] if depth is not None else None,
                                           depth[m] if depth is not None else np.zeros(m.sum()),
                                           q=q, min_bin=min_null)
        if t is None:
            t = np.full(int(m.sum()), float(np.quantile(p[nm_local], q)))
            info = {"frac_uncovered": np.nan}
        thr = float(np.median(t))
        c = p > t
        call[m] = c
        state[m] = np.where(c, "malignant", "benign")
        named = ", ".join(f"{arm_cols[i]}{'+' if sig[i] > 0 else '-'}"
                          for i in np.argsort(-np.abs(sig))[:int(ev.sum())])
        rows.append({"donor": str(d), "callable": True, "reason": "",
                     "n_null": int(nm.sum()), "anchor_cluster": cl_best, "anchor_n": n_best,
                     "anchor_frac": float(n_best / int(qm.sum())),
                     "n_event_arms": int(ev.sum()), "thr": thr,
                     "frac_called": float(c[qm_local].mean()),
                     "null_fpr": float(c[nm_local].mean()),
                     "frac_uncovered": info["frac_uncovered"], "signature": named})
        if verbose:
            print(f"[{d}] anchor={cl_best} (n={n_best}) arms={int(ev.sum())} [{named}]  "
                  f"called={int(c[qm_local].sum())}/{int(qm.sum())} "
                  f"({c[qm_local].mean():.1%})  null FPR={c[nm_local].mean():.1%}")
    return proj, call, state, pd.DataFrame(rows)


# ----------------------------------------- Step 6: MRVI-Leiden smoothing
def compute_mrvi_leiden(acnv, res, seed=0, key="mrvi_leiden", rep="X_mrvi_u"):
    """High-res Leiden on the MRVI latent (computed once, shared by the smoothed methods).

    HEAVY (neighbors + Leiden on ~400k cells) -> GPU kernel.
    """
    assert rep in acnv.obsm, f"{rep} missing on acnv (expected carried from adata)"
    sc.pp.neighbors(acnv, use_rep=rep, random_state=seed)
    sc.tl.leiden(acnv, resolution=res, random_state=seed, key_added=key,
                 flavor="igraph", n_iterations=2, directed=False)
    print(f"{acnv.obs[key].nunique()} MRVI clusters @res={res}")


def vote_cluster(acnv, call_col, frac, leiden_key="mrvi_leiden"):
    """Majority-vote a per-cell boolean call across MRVI Leiden clusters.

    A cluster is malignant if >= `frac` of its cells are True in `call_col`.
    Returns a bool ndarray aligned to acnv.obs.
    """
    cluster_frac = acnv.obs.groupby(leiden_key, observed=True)[call_col].mean()
    return acnv.obs[leiden_key].map(cluster_frac >= frac).astype(bool).to_numpy()


# ----------------------------------------------- Step 7: strategy comparison
def strategy_agreement(adata, strat_cols, tcr_col, avail_mask):
    """precision/recall/F1/Jaccard of each strategy vs the TCR call on available cells."""
    ref = adata.obs[tcr_col].to_numpy().astype(bool)
    rows = []
    for name, col in strat_cols.items():
        yp = adata.obs[col].to_numpy().astype(bool)[avail_mask]; yt = ref[avail_mask]
        tp = int((yp & yt).sum()); fp = int((yp & ~yt).sum()); fn = int((~yp & yt).sum())
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec); jac = tp / max(1, tp + fp + fn)
        rows.append(dict(strategy=name, n_called=int(yp.sum()), precision=round(prec, 3),
                         recall=round(rec, 3), f1=round(f1, 3), jaccard=round(jac, 3)))
    return pd.DataFrame(rows).set_index("strategy")


def tcr_cnv_quality(adata, strat_cols, tcr_col, avail_mask):
    """Pooled TCR<->CNV agreement per strategy on TCR+ cells.
    sensitivity = of TCR-malignant, fraction CNV-malignant;
    specificity = of TCR-non-malignant, fraction CNV-non-malignant."""
    ref = adata.obs[tcr_col].to_numpy().astype(bool)[avail_mask]
    rows = []
    for name, col in strat_cols.items():
        yp = adata.obs[col].to_numpy().astype(bool)[avail_mask]
        tp = int((yp & ref).sum());  fn = int((~yp & ref).sum())
        tn = int((~yp & ~ref).sum()); fp = int((yp & ~ref).sum())
        rows.append(dict(strategy=name,
                         n_tcr_malig=tp + fn, n_tcr_benign=tn + fp, tp=tp, tn=tn,
                         sensitivity=round(tp / max(1, tp + fn), 3),
                         specificity=round(tn / max(1, tn + fp), 3)))
    return pd.DataFrame(rows).set_index("strategy")


# ------------------------------------------------- Step 9: chromosome heatmaps
def _chr_separators(chr_pos):
    items = sorted(chr_pos.items(), key=lambda kv: kv[1])     # {chromosome: start_col}
    bounds = [v for _, v in items]
    labels = [k.replace("chr", "") for k, _ in items]
    return bounds, labels


def _heatmap_payload(sub, order_by):
    """Minimal arrays needed to draw a CNV heatmap, extracted from an inferCNV'd sub object.

    Caching this (rather than the AnnData) lets a re-run skip the per-donor inferCNV recompute.
    """
    q = sub[sub.obs["cnv_ref"] == "query"]
    Xc = q.obsm["X_cnv"]
    Xc = Xc.toarray() if hasattr(Xc, "toarray") else np.asarray(Xc)
    payload = {"obs_names": np.asarray(q.obs_names, dtype=object),
               "X_cnv": np.asarray(Xc, dtype=np.float32),
               "chr_pos": dict(sub.uns["cnv"]["chr_pos"])}
    if order_by == "recompute":
        payload["hm_leiden"] = q.obs["hm_leiden"].astype(str).to_numpy()
    return payload


def _save_payload(path: Path, payload):
    cp = payload["chr_pos"]
    arrs = {"obs_names": payload["obs_names"], "X_cnv": payload["X_cnv"],
            "chr_labels": np.asarray(list(cp.keys()), dtype=object),
            "chr_pos_val": np.asarray(list(cp.values()), dtype=np.int64)}
    if "hm_leiden" in payload:
        arrs["hm_leiden"] = payload["hm_leiden"]
    np.savez_compressed(path, **arrs)


def _load_payload(path: Path):
    d = np.load(path, allow_pickle=True)
    payload = {"obs_names": d["obs_names"], "X_cnv": d["X_cnv"],
               "chr_pos": {str(k): int(v) for k, v in zip(d["chr_labels"], d["chr_pos_val"])}}
    if "hm_leiden" in d.files:
        payload["hm_leiden"] = d["hm_leiden"]
    return payload


def _plot_cnv_heatmap(payload, donor, vlim, acnv, *, call_col, ann_cols, order_by, cmap, fig_dir,
                      fig_prefix="skin_T_cnv_heatmap", fmt="png", dpi=150):
    from matplotlib.colors import LogNorm, ListedColormap
    import matplotlib.pyplot as plt

    if fmt == "svg":
        plt.rcParams["svg.fonttype"] = "none"   # editable text; data layers rasterized below

    obs_names = pd.Index(payload["obs_names"])
    ann = acnv.obs.loc[obs_names, ann_cols]                   # query-only cols dropped by concat
    if order_by == "recompute":
        key = np.asarray(payload["hm_leiden"]).astype(str)    # this run's CNV clusters
        grp_label = "inferCNV leiden (this run)"
    else:
        key = ann["cnv_leiden"].astype(str).to_numpy()        # cached cluster = strat-3 unit
        grp_label = "cached cnv_leiden (strategy-3 unit)"
    codes = pd.factorize(key)[0]
    order = np.argsort(codes, kind="stable")
    ann = ann.iloc[order]
    Xc = np.asarray(payload["X_cnv"])[order]
    n  = Xc.shape[0]
    clu_bounds = np.where(np.diff(codes[order]) != 0)[0] + 1
    a_call = ann[call_col].to_numpy().astype(bool).astype(float)[:, None]
    a_tcr  = ann["tcr_is_malignant"].to_numpy().astype(bool).astype(float)[:, None]
    a_sz   = ann["tcr_clone_size"].to_numpy(dtype=float)
    a_sz   = np.where(a_sz > 0, a_sz, np.nan)[:, None]

    bounds, labels = _chr_separators(payload["chr_pos"])
    centers = [(bounds[i] + (bounds[i + 1] if i + 1 < len(bounds) else Xc.shape[1])) / 2
               for i in range(len(bounds))]
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 4, width_ratios=[40, 1, 1, 1.4], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    im0 = ax.imshow(Xc, aspect="auto", cmap=cmap, vmin=-vlim, vmax=vlim, interpolation="none",
                    rasterized=True)
    for b in bounds[1:]:
        ax.axvline(b, color="k", lw=0.4, alpha=0.5)
    for r in clu_bounds:
        ax.axhline(r - 0.5, color="k", lw=0.5, alpha=0.6)
    ax.set_xticks(centers); ax.set_xticklabels(labels, fontsize=5, rotation=90)
    ax.set_yticks([]); ax.set_ylabel(f"{n} query cells (grouped by {grp_label})")
    ax.set_title(f"inferCNV — {donor}: CNV clusters + {call_col} + TCR clone")
    ax.set_ylim(n - 0.5, -0.5)
    fig.colorbar(im0, ax=ax, fraction=0.015, pad=0.01, label="inferCNV")

    def _strip(gi, arr, title, cmap, norm=None, vmin=None, vmax=None, cbar=False):
        a = fig.add_subplot(gs[gi], sharey=ax)
        cmap = cmap.copy() if hasattr(cmap, "copy") else cmap
        if hasattr(cmap, "set_bad"):
            cmap.set_bad("white")
        im = a.imshow(arr, aspect="auto", cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                      interpolation="none", rasterized=True)
        for r in clu_bounds:
            a.axhline(r - 0.5, color="k", lw=0.5, alpha=0.6)
        a.set_xticks([0]); a.set_xticklabels([title], fontsize=6, rotation=90); a.set_yticks([])
        if cbar:
            fig.colorbar(im, ax=a, fraction=0.6, pad=0.05)

    _strip(1, a_call, f"{call_col}", ListedColormap(["#e5e5e5", "#d62728"]), vmin=0, vmax=1)
    _strip(2, a_tcr, "TCR malig clone", ListedColormap(["#e5e5e5", "#111111"]), vmin=0, vmax=1)
    sz_norm = (LogNorm(vmin=max(1, np.nanmin(a_sz)), vmax=np.nanmax(a_sz))
               if np.isfinite(a_sz).any() else None)
    _strip(3, a_sz, "clone size", plt.cm.viridis, norm=sz_norm, cbar=True)

    fig.savefig(fig_dir / f"{fig_prefix}_{donor}.{fmt}", dpi=dpi, bbox_inches="tight", format=fmt)
    plt.show()


def run_cnv_heatmaps(acnv, shared_ref, *, call_col, fig_dir: Path, n_per_study=2,
                     window=150, vlim_scale=3.0, cmap="RdBu_r", order_by="cnv_leiden",
                     reference_cat=("nonclonal", "hc_atlas", "healthy"),
                     ref_mode="bounded", dynamic_threshold=None,
                     fig_prefix="skin_T_cnv_heatmap", n_jobs=8, chunk=2500,
                     fmt="png", dpi=150, cnv_cache_dir=None, force_cnv=False):
    """Re-run inferCNV (keeping X_cnv) for top-burden donors and draw annotated heatmaps.

    Picks the top `n_per_study` donors per study by TCR-malignant burden, then for each draws
    a chromosome heatmap with right-side strips: the chosen method's call (`call_col`),
    malignant-TCR-clone membership, and clone size. HEAVY (GPU kernel).

    `shared_ref` is concatenated to each per-donor query block (nb14). Pass `shared_ref=None`
    when the reference cells already live in `acnv` (nb15 same-sample CD8 ref). `reference_cat`
    lists the candidate `cnv_ref` categories to use as the inferCNV baseline (kept if >=20 cells).

    `ref_mode` / `dynamic_threshold` mirror `run_per_donor_infercnv` so the picture can be drawn
    with the *same* estimator that produced the call — pass `ref_mode="mean"` (pooled reference
    vector) instead of the default multi-category bounded dead-band. Defaults keep the legacy
    nb15/nb22 behaviour.

    If `cnv_cache_dir` is given, each donor's heatmap inputs (X_cnv + obs_names + chr_pos) are
    cached to ``<cnv_cache_dir>/<fig_prefix>_<donor>.npz`` and reloaded on later runs, skipping
    the per-donor inferCNV recompute. The cache is keyed only by donor/`fig_prefix`; pass
    `force_cnv=True` to invalidate it after changing `window`/`order_by`/the cohort.
    """
    ann_cols = ["cnv_leiden", call_col, "tcr_is_malignant", "tcr_clone_size"]
    burden   = acnv.obs.groupby("donor", observed=True)["tcr_is_malignant"].sum()
    study_of = acnv.obs.groupby("donor", observed=True)["study"].first().astype(str)
    sel = pd.DataFrame({"study": study_of, "burden": burden})
    hm_donors = (sel.sort_values("burden", ascending=False)
                 .groupby("study", observed=True).head(n_per_study).index.tolist())
    print(f"heatmap donors ({n_per_study}/study, by TCR burden):", hm_donors,
          "| call_col =", call_col, "| ORDER_BY =", order_by)
    if cnv_cache_dir is not None:
        cnv_cache_dir = Path(cnv_cache_dir)
        cnv_cache_dir.mkdir(parents=True, exist_ok=True)
    for d in hm_donors:
        cache_f = (cnv_cache_dir / f"{fig_prefix}_{d}.npz") if cnv_cache_dir is not None else None
        if cache_f is not None and cache_f.exists() and not force_cnv:
            payload = _load_payload(cache_f)
            print(f"[{d}] loaded cached heatmap inputs ({cache_f.name})  X_cnv={payload['X_cnv'].shape}")
        else:
            q = acnv[acnv.obs["donor"] == d]
            if shared_ref is not None:
                sub = ad.concat([q, shared_ref], join="inner", index_unique=None)
                sub.var = acnv.var.loc[sub.var_names].copy()  # restore genomic positions
            else:
                sub = q.copy()                                # reference already in acnv
            ref_cats = [c for c in reference_cat
                        if int((sub.obs["cnv_ref"] == c).sum()) >= 20]
            kw = (dict(reference=_pooled_reference_vector(sub, ref_cats)) if ref_mode == "mean"
                  else dict(reference_key="cnv_ref", reference_cat=ref_cats))
            cnv.tl.infercnv(sub, window_size=window, dynamic_threshold=dynamic_threshold,
                            n_jobs=n_jobs, chunksize=chunk, **kw)
            if order_by == "recompute":
                cnv.tl.pca(sub); cnv.pp.neighbors(sub); cnv.tl.leiden(sub, key_added="hm_leiden")
            print(f"[{d}] n={sub.n_obs}  ref={ref_cats}")
            payload = _heatmap_payload(sub, order_by)
            if cache_f is not None:
                _save_payload(cache_f, payload)
                print(f"[{d}] cached heatmap inputs -> {cache_f.name}")
            del sub, q
            gc.collect()
        vlim = (float(np.nanpercentile(np.abs(payload["X_cnv"]), 99)) or 0.05) * vlim_scale
        st = acnv.obs.loc[acnv.obs["donor"] == d, "study"].iloc[0]
        print(f"[{st}/{d}] color vlim=±{vlim:.4f}")
        _plot_cnv_heatmap(payload, d, vlim, acnv, call_col=call_col, ann_cols=ann_cols,
                          order_by=order_by, cmap=cmap, fig_dir=fig_dir, fig_prefix=fig_prefix,
                          fmt=fmt, dpi=dpi)
        del payload
        gc.collect()


# ------------------------------------------- Step 10: per-cell arm-level CNV
# hg38 centromere positions (bp), used only to split each chromosome into p / q arms.
HG38_CENTROMERE = {
    "chr1": 123_400_000, "chr2": 93_900_000, "chr3": 90_900_000, "chr4": 50_000_000,
    "chr5": 48_800_000, "chr6": 59_800_000, "chr7": 60_100_000, "chr8": 45_200_000,
    "chr9": 43_000_000, "chr10": 39_800_000, "chr11": 53_400_000, "chr12": 35_500_000,
    "chr13": 17_700_000, "chr14": 17_200_000, "chr15": 19_000_000, "chr16": 36_800_000,
    "chr17": 25_100_000, "chr18": 18_500_000, "chr19": 26_200_000, "chr20": 28_100_000,
    "chr21": 12_000_000, "chr22": 15_000_000, "chrX": 61_000_000,
}


def _arm_labels(var, centromeres=HG38_CENTROMERE):
    """Map each gene (var row) to its chromosome arm, e.g. 'chr8q'. '' if unplaceable."""
    chrom = var["chromosome"].astype(str).to_numpy()
    start = pd.to_numeric(var["start"], errors="coerce").to_numpy()
    out = np.full(len(var), "", dtype=object)
    for i, (c, s) in enumerate(zip(chrom, start)):
        cen = centromeres.get(c)
        if cen is None or not np.isfinite(s):
            continue
        out[i] = f"{c}{'p' if s < cen else 'q'}"
    return out


def compute_arm_cnv_per_cell(acnv, cnv_donors, cache: Path, *, shared_ref=None,
                             reference_cat=("cd8_ref",), window=250, step=10,
                             n_jobs=8, chunk=100, min_genes_arm=15, keep_cats=("query",),
                             ref_mode="bounded", dynamic_threshold=1.5, shuffle=False,
                             null_frac=0.0, null_cats=("cd8_ref",), seed=0, force=False):
    """Per-sample inferCNV keeping per-gene CNV; aggregate to chromosome arms.

    For each sample runs `cnv.tl.infercnv(... calculate_gene_values=True)` so the per-gene CNV
    layer (`gene_values_cnv`, aligned to var_names, NaN where a gene isn't in any window) is
    available, then averages genes within each chromosome arm -> a compact `cells x ~39 arm`
    matrix. Cells kept are those whose `cnv_ref` is in `keep_cats`.

    Reference: by default the same-sample CD8 cells already in `acnv` (nb15). Pass `shared_ref`
    (the strat_3 diploid reference) to concat it onto each per-donor query block instead, with
    `reference_cat` listing the candidate `cnv_ref` baseline categories (kept if >=20 cells).

    `ref_mode` / `dynamic_threshold` / `shuffle` / `null_frac` mirror `run_per_donor_infercnv` —
    pass the same values so the arm matrix and the per-cell scores come from the same estimator.
    With the same `seed` the held-out `ref_null` cells are the same cells in both passes, which is
    what lets `arm_consensus_score` use them as its per-arm null (add `"ref_null"` to `keep_cats`).

    Cached to `cache` (parquet); reload unless `force` or the cache is missing donors. HEAVY
    (GPU kernel, per-sample inferCNV). Returns the arm DataFrame (index = obs_name, cols = arms + 'donor').
    """
    if (not force) and cache.exists():
        arm = pd.read_parquet(cache)
        have = set(arm["donor"].astype(str).unique()) if "donor" in arm else set()
        if set(map(str, cnv_donors)).issubset(have):
            print(f"loaded cached arm-CNV ({cache.name}) for {len(cnv_donors)} samples", arm.shape)
            return arm.set_index("obs_name") if "obs_name" in arm else arm

    part_dir = cache.parent / f"{cache.stem}_parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    def _part_path(d):
        return part_dir / f"{hashlib.md5(str(d).encode()).hexdigest()[:8]}.parquet"

    arm_label_full = pd.Series(_arm_labels(acnv.var), index=acnv.var_names)
    keep_arms = sorted({a for a in arm_label_full.unique() if a})  # stable column order
    parts = []
    for d in cnv_donors:
        pp = _part_path(d)
        if (not force) and pp.exists():
            part = pd.read_parquet(pp).set_index("obs_name")
            parts.append(part)
            print(f"[{d}] loaded cached part  query={part.shape[0]:>6}")
            continue
        rng = np.random.default_rng(_donor_seed(seed, d))      # same split as the score pass
        q = acnv[acnv.obs["donor"] == d]
        if shared_ref is not None:
            ref = shared_ref
            if null_frac > 0:
                ref, q_hold = _split_null(rng, shared_ref, q, null_frac, null_cats, d)
                if q_hold.any():
                    q = q.copy()
                    q.obs.loc[q.obs_names[q_hold], "cnv_ref"] = "ref_null"
            sub = ad.concat([q, ref], join="inner", index_unique=None)
            sub.var = acnv.var.loc[sub.var_names].copy()       # restore genomic positions
        else:
            sub = q.copy()                                     # reference already in acnv
        if shuffle:
            sub = sub[rng.permutation(sub.n_obs)].copy()
        ref_cats = [c for c in reference_cat
                    if int((sub.obs["cnv_ref"] == c).sum()) >= 20]
        kw = (dict(reference=_pooled_reference_vector(sub, ref_cats)) if ref_mode == "mean"
              else dict(reference_key="cnv_ref", reference_cat=ref_cats))
        cnv.tl.infercnv(sub, window_size=window, step=step, n_jobs=n_jobs, chunksize=chunk,
                        dynamic_threshold=dynamic_threshold, calculate_gene_values=True, **kw)
        G = sub.layers["gene_values_cnv"]                      # cells x genes (NaN-filled)
        G = G.toarray() if hasattr(G, "toarray") else np.asarray(G)
        lab = arm_label_full.reindex(sub.var_names).to_numpy()
        qmask = sub.obs["cnv_ref"].astype(str).isin(list(keep_cats)).to_numpy()
        rows = {}
        for a in keep_arms:
            cols = np.where(lab == a)[0]
            if cols.size < min_genes_arm:
                continue
            with np.errstate(invalid="ignore"):
                rows[a] = np.nanmean(G[np.ix_(qmask, cols)], axis=1)
        df = pd.DataFrame(rows, index=sub.obs_names[qmask])
        df["donor"] = d
        df.index.name = "obs_name"
        df.reset_index().to_parquet(pp)                        # checkpoint before appending
        parts.append(df)
        print(f"[{d}] query={int(qmask.sum()):>6}  arms={df.shape[1] - 1}")
        del sub, G
        gc.collect()
    arm = pd.concat(parts, axis=0)
    arm.index.name = "obs_name"
    arm.reset_index().to_parquet(cache)
    print(f"computed arm-CNV; cached -> {cache}  {arm.shape}")
    return arm


# ============================ nb15: CD4 query / same-sample CD8 reference ============================
def prepare_cd4_cd8ref_inputs(adata, gtf: Path, *, kept_lineages, min_cd8_ref=20, std_chr=None,
                              min_positioned=8000):
    """Build the CD4 inferCNV query + same-sample CD8 reference (nb15).

    Query = CD4 T cells; reference = CD8 T cells from the SAME sample. Keeps samples whose
    `lineage` is in `kept_lineages` and that have >= `min_cd8_ref` CD8 cells and >= 1 CD4 cell.
    Log-normalises, sets `cnv_ref` (cd8_ref / query) and `donor = sample_id`, adds genomic
    positions from `gtf`, and restricts to `std_chr` (drops the unpositioned 'chrnan' block +
    chrM that would otherwise dominate the heatmap and dilute cnv_cell_score). Returns
    (acnv, cnv_donors).
    """
    if std_chr is None:
        std_chr = [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]]
    lcol = "cell_type_T2" if "cell_type_T2" in adata.obs.columns else "cell_type_T"
    lin5 = adata.obs[lcol].astype(str).str.split("_").str[0]   # CD4 / CD8 / NK (or UNK)
    adata.obs["lin5"] = lin5.values
    kept_sample = adata.obs["lineage"].astype(str).isin(kept_lineages)
    sel = kept_sample.to_numpy() & lin5.isin(["CD4", "CD8"]).to_numpy()
    print("kept samples (lineage CD4/NA/unresolved):", adata.obs.loc[kept_sample, "sample_id"].nunique(),
          "/", adata.obs["sample_id"].nunique(),
          "| excluded (CD8/gamma_delta):", adata.obs.loc[~kept_sample, "sample_id"].nunique())

    # ---- query (CD4) + reference (CD8), log-normalised ----
    acnv = adata[sel].copy()
    acnv.X = acnv.layers["raw_counts"].copy()
    sc.pp.normalize_total(acnv, target_sum=1e4)
    sc.pp.log1p(acnv)
    acnv.obs["cnv_ref"] = np.where(acnv.obs["lin5"].to_numpy() == "CD8", "cd8_ref", "query")
    acnv.obs["donor"]   = acnv.obs["sample_id"].astype(str).values     # call_per_donor groups on 'donor'

    # ---- keep only samples with enough CD8 reference + at least one CD4 query ----
    n_cd8 = acnv.obs[acnv.obs["cnv_ref"] == "cd8_ref"].groupby("donor", observed=True).size()
    n_cd4 = acnv.obs[acnv.obs["cnv_ref"] == "query"].groupby("donor", observed=True).size()
    all_d = acnv.obs["donor"].unique()
    n_cd8 = n_cd8.reindex(all_d, fill_value=0); n_cd4 = n_cd4.reindex(all_d, fill_value=0)
    cnv_donors = sorted(d for d in all_d if n_cd8[d] >= min_cd8_ref and n_cd4[d] >= 1)
    dropped = sorted(set(map(str, all_d)) - set(cnv_donors))
    print(f"usable samples (CD8 ref >= {min_cd8_ref}): {len(cnv_donors)} | dropped (too few CD8): {len(dropped)}")
    if dropped:
        print("  dropped:", dropped)
    acnv = acnv[acnv.obs["donor"].isin(cnv_donors)].copy()

    # ---- genomic positions; drop unpositioned 'chrnan' / chrM (a huge near-zero window block) ----
    cnv.io.genomic_position_from_gtf(gtf, adata=acnv, gtf_gene_id="gene_name")
    acnv = acnv[:, acnv.var["chromosome"].astype(str).isin(std_chr)].copy()
    print(f"query (CD4) cells: {int((acnv.obs['cnv_ref']=='query').sum())} | "
          f"reference (CD8) cells: {int((acnv.obs['cnv_ref']=='cd8_ref').sum())} | "
          f"genes on chr1-22/X/Y: {acnv.n_vars}")
    assert acnv.n_vars > min_positioned, (
        f"only {acnv.n_vars} positioned genes (< {min_positioned}) — for a full-gene object this means a "
        "GTF gene_name/symbol mismatch; for an HVG-subset object lower `min_positioned`")
    return acnv, cnv_donors


# ==================== nb34: CD4 query / pooled diploid reference (blood) ====================
def prepare_blood_pooled_cnv_inputs(adata, gtf: Path, healthy_ref, seed: int, *,
                                    kept_lineages, hc_ref_frac=2 / 3, min_cd8_ref=20,
                                    std_chr=None, min_positioned=5000, hc_disease="HC"):
    """Blood CD4 inferCNV query + a *pooled* diploid reference (nb34).

    The nb15 same-sample-CD8 reference is lineage-mismatched: CD4-vs-CD8 expression differences
    cluster along the genome (cytotoxic program, HLA 6p, TRB 7q, TRA 14q), so window smoothing
    leaves every CD4 cell — healthy included — with a nonzero |X_cnv| floor. Here CD8 is demoted
    to one of four baseline categories whose biases differ and therefore partly cancel:

      nonclonal     within-sample non-dominant CD4  right batch + lineage, tumour-contaminated
      cd8_ref       same-sample CD8                 right batch, wrong lineage
      hc_atlas      healthy blood CD4 (H__HC1)      right tissue + lineage, one donor
      healthy_pbmc  external healthy PBMC CD4       right lineage, many donors, wrong batch

    `nonclonal` and `cd8_ref` live inside `acnv` (per sample); `hc_atlas` and `healthy_pbmc` are
    returned in `shared_ref`, which `run_per_donor_infercnv` concatenates onto every sample block.

    The atlas holds a single healthy blood sample, so `hc_ref_frac` of its CD4 goes to the
    reference and the rest stays in the query as a held-out negative control — the run's sanity
    check, since a working caller must leave it near-zero malignant.

    `healthy_ref` is the AnnData (or h5ad path) from `build_healthy_pbmc_ref`.
    Returns (acnv, shared_ref, cnv_donors, hc_control_names).
    """
    if std_chr is None:
        std_chr = [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]]
    lcol = "cell_type_T2" if "cell_type_T2" in adata.obs.columns else "cell_type_T"
    lin5 = adata.obs[lcol].astype(str).str.split("_").str[0]          # CD4 / CD8 / NK (or UNK)
    adata.obs["lin5"] = lin5.values
    kept_sample = adata.obs["lineage"].astype(str).isin(kept_lineages)
    sel = kept_sample.to_numpy() & lin5.isin(["CD4", "CD8"]).to_numpy()

    # ---- query (CD4) + within-sample CD8 reference, log-normalised ----
    acnv = adata[sel].copy()
    acnv.X = acnv.layers["raw_counts"].copy()
    sc.pp.normalize_total(acnv, target_sum=1e4)
    sc.pp.log1p(acnv)
    acnv.obs["cnv_ref"] = np.where(acnv.obs["lin5"].to_numpy() == "CD8", "cd8_ref", "query")
    acnv.obs["donor"] = acnv.obs["sample_id"].astype(str).values      # inferCNV runs per sample

    # ---- within-sample non-dominant CD4 -> 'nonclonal' (as prepare_infercnv_inputs) ----
    is_q = (acnv.obs["cnv_ref"] == "query").to_numpy()
    nonclonal = (is_q
                 & acnv.obs["has_tcr"].to_numpy()
                 & ~acnv.obs["tcr_is_dominant_clone"].to_numpy()
                 & (acnv.obs["cell_type"].astype(str) != "tumor_cell").to_numpy())
    acnv.obs.loc[nonclonal, "cnv_ref"] = "nonclonal"

    # ---- healthy blood CD4: hc_ref_frac -> reference, remainder -> held-out control ----
    rng = np.random.default_rng(seed)
    hc_cd4 = np.where((acnv.obs["disease"].astype(str) == hc_disease).to_numpy()
                      & (acnv.obs["lin5"].to_numpy() == "CD4"))[0]
    assert hc_cd4.size, f"no {hc_disease} CD4 cells found for the hc_atlas reference"
    n_ref = int(round(hc_ref_frac * hc_cd4.size))
    ref_pos = np.sort(rng.choice(hc_cd4, n_ref, replace=False))
    hc_ref_names = acnv.obs_names[ref_pos]
    hc_control_names = acnv.obs_names[np.setdiff1d(hc_cd4, ref_pos)]
    hc_ref = _clean_ref(adata[hc_ref_names], "hc_atlas")
    acnv = acnv[~acnv.obs_names.isin(hc_ref_names)].copy()            # ref cells leave the query
    acnv.obs.loc[hc_control_names, "cnv_ref"] = "query"               # control is scored, not baseline
    print(f"healthy blood CD4: {hc_ref.n_obs} -> hc_atlas reference | "
          f"{len(hc_control_names)} -> held-out control (stays in query)")

    # ---- external healthy PBMC CD4 ----
    if not isinstance(healthy_ref, ad.AnnData):
        healthy_ref = sc.read_h5ad(Path(healthy_ref))
    healthy = healthy_ref.copy()
    healthy.var_names_make_unique()
    sc.pp.normalize_total(healthy, target_sum=1e4)
    sc.pp.log1p(healthy)
    healthy.obs = pd.DataFrame({"cnv_ref": "healthy_pbmc", "donor": "HEALTHY_PBMC_REF",
                                "cell_type": healthy.obs.get("cell_type", "CD4").astype(str).values},
                               index=healthy.obs_names)
    print(f"external healthy PBMC reference: {healthy.n_obs} cells")

    # ---- keep samples with enough CD8 + at least one CD4 query ----
    n_cd8 = acnv.obs[acnv.obs["cnv_ref"] == "cd8_ref"].groupby("donor", observed=True).size()
    n_cd4 = acnv.obs[acnv.obs["cnv_ref"] != "cd8_ref"].groupby("donor", observed=True).size()
    all_d = acnv.obs["donor"].unique()
    n_cd8 = n_cd8.reindex(all_d, fill_value=0); n_cd4 = n_cd4.reindex(all_d, fill_value=0)
    cnv_donors = sorted(d for d in all_d if n_cd8[d] >= min_cd8_ref and n_cd4[d] >= 1)
    dropped = sorted(set(map(str, all_d)) - set(cnv_donors))
    print(f"usable samples (CD8 >= {min_cd8_ref}): {len(cnv_donors)} | dropped: {dropped}")
    acnv = acnv[acnv.obs["donor"].isin(cnv_donors)].copy()

    # ---- common gene space + genomic positions; one shared reference ----
    refs = [hc_ref, healthy]
    common = acnv.var_names
    for r in refs:
        common = common.intersection(r.var_names)
    acnv = acnv[:, common].copy()
    cnv.io.genomic_position_from_gtf(gtf, adata=acnv, gtf_gene_id="gene_name")
    acnv = acnv[:, acnv.var["chromosome"].astype(str).isin(std_chr)].copy()
    shared_ref = ad.concat([r[:, acnv.var_names] for r in refs], join="inner", index_unique=None)
    shared_ref.var = acnv.var.copy()                                  # share genomic positions

    # detection depth on the SHARED gene space — query and reference must be counted over the
    # same genes for the depth-corrected null in `call_per_donor_null` to mean anything.
    for a in (acnv, shared_ref):
        a.obs["n_genes_cnv"] = np.asarray((a.X > 0).sum(axis=1)).ravel()

    hc_control_names = hc_control_names[hc_control_names.isin(acnv.obs_names)]
    print("cnv_ref:", dict(acnv.obs["cnv_ref"].value_counts()),
          "| shared ref:", shared_ref.n_obs, dict(shared_ref.obs["cnv_ref"].value_counts()),
          "| genes on chr1-22/X/Y:", acnv.n_vars)
    assert acnv.n_vars > min_positioned, (
        f"only {acnv.n_vars} positioned genes (< {min_positioned}) — check the GTF gene_name / "
        "symbol intersection, or lower `min_positioned` for an HVG-subset object")
    return acnv, shared_ref, cnv_donors, hc_control_names


def run_per_sample_cd8ref_infercnv(acnv, cnv_donors, cache: Path, *, window=250, leiden_res=2.0,
                                   n_jobs=8, chunk=2500, std_chr=None, force=False):
    """Per-sample inferCNV with same-sample CD8 reference; cluster + score the CD4 query (nb15).

    Fills acnv.obs: cnv_cell_score (per CD4 cell: genome-wide mean |X_cnv|), cnv_leiden
    (sample-local CNV Leiden cluster), cnv_score (per-cnv_leiden group-mean of cnv_cell_score
    == infercnvpy cnv_score). Cached to `cache` (parquet, cols donor/cnv_ref/cnv_cell_score/
    cnv_leiden); reloads unless `force` or the cache is missing donors. HEAVY (GPU kernel).
    """
    if std_chr is None:
        std_chr = [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]]
    acnv.obs["cnv_cell_score"] = np.nan   # per CD4 cell: genome-wide mean |X_cnv|
    acnv.obs["cnv_leiden"]     = ""       # per CD4 cell: sample-local cnv Leiden cluster

    use_cache = (not force) and cache.exists()
    if use_cache:
        cc = pd.read_parquet(cache).set_index("obs_name")
        use_cache = ({"cnv_cell_score", "cnv_leiden"}.issubset(cc.columns)
                     and set(cnv_donors).issubset(set(cc["donor"].astype(str).unique())))
    if use_cache:
        acnv.obs["cnv_cell_score"] = cc["cnv_cell_score"].reindex(acnv.obs_names).to_numpy()
        acnv.obs["cnv_leiden"]     = cc["cnv_leiden"].reindex(acnv.obs_names).fillna("").to_numpy()
        print(f"loaded cached inferCNV ({cache.name}) for {len(cnv_donors)} samples @res={leiden_res:g}")
    else:
        for d in cnv_donors:
            sub = acnv[acnv.obs["donor"] == d].copy()                          # CD4 query + same-sample CD8 ref
            sub = sub[:, sub.var["chromosome"].astype(str).isin(std_chr)].copy()   # drop unpositioned chrnan / chrM
            print(f"[{d}] CD4 query={int((sub.obs['cnv_ref']=='query').sum()):>6}  "
                  f"CD8 ref={int((sub.obs['cnv_ref']=='cd8_ref').sum()):>5}")
            cnv.tl.infercnv(sub, reference_key="cnv_ref", reference_cat=["cd8_ref"],
                            window_size=window, n_jobs=n_jobs, chunksize=chunk)
            subq = sub[sub.obs["cnv_ref"] == "query"].copy()                   # cluster + score the CD4 query only
            cnv.tl.pca(subq)
            cnv.pp.neighbors(subq)
            cnv.tl.leiden(subq, resolution=leiden_res)                         # -> subq.obs['cnv_leiden']
            Xc = subq.obsm["X_cnv"]
            Xc = np.abs(Xc.toarray() if hasattr(Xc, "toarray") else np.asarray(Xc))
            subq.obs["cnv_cell_score"] = Xc.mean(axis=1)
            del Xc
            acnv.obs.loc[subq.obs_names, "cnv_cell_score"] = subq.obs["cnv_cell_score"].to_numpy()
            acnv.obs.loc[subq.obs_names, "cnv_leiden"] = (
                d + "_" + subq.obs["cnv_leiden"].astype(str)).to_numpy()
            del sub, subq
            gc.collect()
        out = acnv.obs[["donor", "cnv_ref", "cnv_cell_score", "cnv_leiden"]].copy()
        out.index.name = "obs_name"
        out.reset_index().to_parquet(cache)
        print(f"computed inferCNV; cached -> {cache}")

    # per-cnv-cluster score = group-mean of per-cell cnv_cell_score (== infercnvpy cnv_score)
    _q = (acnv.obs["cnv_ref"] == "query").to_numpy()
    acnv.obs["cnv_score"] = np.nan
    acnv.obs.loc[_q, "cnv_score"] = (acnv.obs.loc[_q]
                                     .groupby("cnv_leiden", observed=True)["cnv_cell_score"].transform("mean"))
