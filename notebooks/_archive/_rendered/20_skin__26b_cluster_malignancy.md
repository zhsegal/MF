# 20_skin/26b_cluster_malignancy.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/20_skin/26b_cluster_malignancy.ipynb` and is never run._

# 26b — Cluster-level malignancy: your decisions and manual calls

Every sample's T cells (CD4 + Treg + CD8) are Leiden-clustered on their own (res 1.0), and each cluster
is called malignant or benign. The compute runs as two jobs; this notebook only reads their outputs.

| | where | what |
|---|---|---|
| clustering | `jobs/run_cluster_malig.sh` | per-sample HVG → PCA → Leiden 1.0 → UMAP; cluster pseudobulks; per-cell TRBV/TRAV call and ALICE family |
| Step 1 | `jobs/run_cluster_malig_model.sh` | cluster ALICE label (family share of TCR+ cells ≥ 0.8 / ≤ 0.1); four scores compared leave-one-study-out; cutoffs from held-out P |
| Step 2 | same job | HC → benign · ALICE samples keep ALICE · clear → auto · rest → review sheet + CSV row |
| **Decision 1** | here | which score |
| **Decision 2** | here | how strict (`MAX_ERR`) → how many clusters you review |
| **Manual calls** | `tables/cluster_malig_review_v1.csv` | `m` / `b` per open cluster, using `figures/nb26b_review/` |
| **Merge** | here | → `data/atlas_joint/skin_T_malignancy_cluster_v1.parquet` → `malignancy.load_malignancy(...)["mal_cluster"]` |

To change a decision: `bash jobs/run_cluster_malig_model.sh --program <name> --max-err <x>` (a few
minutes; filled `user_call` rows are kept), then re-run this notebook.

```python  # cell 1
from pathlib import Path


def _resolve_nb_dir() -> Path:
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data from {start}")


NB_DIR = _resolve_nb_dir(); print("NB_DIR =", NB_DIR)
N_SHOW = 3          # review sheets shown inline below (all are in figures/nb26b_review/)
```

```python  # cell 2
import sys, json, re, importlib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

sys.path.insert(0, str(NB_DIR / "helpers"))
import cluster_malig_helpers as CM
import malignancy as MAL
for _m in (CM, MAL):
    importlib.reload(_m)
```

## What the jobs produced

Sample roles:

| role | used for |
|---|---|
| `train` | clonal ALICE samples: program, model and held-out evaluation |
| `train_hc` | HC samples (any study): negatives |
| `tcr_nonclonal` | ALICE samples without a clonal family: keep their ALICE labels, not trained on |
| `apply` | everything else, scored in Step 2 |
| `excluded` | γδ / CD8-aggressive entities |

The purity ceiling is the share of TCR+ cells that a cluster-level call gets right if every cluster
took its majority ALICE label. It is the best res 1.0 can do.

```python  # cell 4
man = json.loads((CM.ATLAS_JOINT / "skin_T_clusters_v1_manifest.json").read_text())
res = json.loads(CM.MODEL_JSON.read_text())
cl = pd.read_csv(CM.CLUSTERS_CSV, index_col="cluster")
print(f"clustering: {man['n_samples']} samples / {man['n_clusters']:,} clusters / "
      f"{man['n_cells']:,} cells | res {man['params']['res']} | written {man['written']}")
print(f"  skipped (<= 30 T cells): {len(man['skipped'])} {sorted(man['skipped'])[:6]} | failed: {man['failed']}")
print(f"model run: {res['written']} | score = {res['program']} (asked: {res['program_arg']}) | "
      f"MAX_ERR {res['max_err']:.0%} | cutoffs {res['p_lo']:.3f} / {res['p_hi']:.3f}")
print(f"  TCR+ cells in mixed clusters {res['mixed_tcr_frac']:.1%} | purity ceiling {res['purity_ceiling']:.1%}"
      f" | HC cells the score alone would call malignant {res['hc_fp_cells']:.1%}"
      f" | ALICE clusters the score confidently contradicts {res['tcr_disagree_clusters']}")
print("\nroles:")
print(cl.groupby("role").agg(samples=("sample_id", "nunique"), clusters=("n_cells", "size"),
                             cells=("n_cells", "sum")).to_string())
print("\nclusters with a V-gene skew, by study (li2024 has no TR V reads in its matrix):")
print(cl.groupby("study")["v_z"].apply(lambda s: s.notna().mean()).round(2).to_string())
```

## Decision 1 — which score?

Each score is scored leave-one-study-out on the ALICE-labelled clusters. For the program, it is
relearned inside every fold.

- **program**: genes that move the same way in ≥ 75 % of donors, comparing malignant clusters with
  benign clusters that are ≥ 80 % CD4, within the same sample. Score = mean log1p CPM(up) − mean(down).
- **ctcl_apriori**: nb26's marker lists, scored the same way.
- **vskew**: how dominant the cluster's top TRBV/TRAV is, against a polyclonal null. Missing where a
  study has no V reads (li2024; gaydosik2019 is 3' and nearly empty).
- **program+vskew**: both together.
- **knn_frac**: nb26 `mal_knn` aggregated per cluster. For reference only; it is **not** held out.

`auto_cells_frac` is the share of labelled cells that land outside the two cutoffs, with ≤ `MAX_ERR`
of them wrong. The job picks the candidate with the highest value unless you pass `--program`.
`auc_within_median` is the median AUROC within samples: can the score rank a sample's own clusters?

```python  # cell 6
cv = pd.read_csv(CM.CV_CSV)
cols = ["coverage", "auc_cluster", "auc_cells", "auc_within_median", "auto_cells_frac",
        "err_mal", "err_ben", "p_lo", "p_hi"]
print(f"held-out, pooled (MAX_ERR {res['max_err']:.0%}):")
print(cv[cv["fold"].eq("pooled")].set_index("candidate")[cols].round(3).to_string())
print("\nauto_cells_frac within each held-out study, same cutoffs:")
print(cv[cv["fold"].ne("pooled")].pivot(index="fold", columns="candidate", values="auto_cells_frac")
      .round(2).to_string())
print("\nwrong cells among auto calls within each held-out study (max of the two sides):")
print(cv[cv["fold"].ne("pooled")].assign(err=lambda d: d[["err_mal", "err_ben"]].max(axis=1))
      .pivot(index="fold", columns="candidate", values="err").round(3).to_string())
display(Image(str(CM.FIG_DIR / "nb26b_candidates_auto_share.png"), width=500))
```

```python  # cell 7
prog = pd.read_csv(CM.PROGRAM_CSV)
n_folds = len(res["folds"])
print(f"learned program: {int(prog.direction.eq('up').sum())} up / {int(prog.direction.eq('down').sum())} down, "
      f"from {res['n_train_donors_program']} donors. n_folds_selected = in how many of the "
      f"{n_folds} leave-one-study-out refits the gene was picked (stability)")
for d in ("up", "down"):
    g = prog[prog["direction"].eq(d)]
    print(f"\n{d} ({int(g.n_folds_selected.ge(n_folds - 1).sum())} picked in >= {n_folds - 1} folds):")
    print(", ".join(f"{r.gene}({r.n_folds_selected})" for r in g.itertuples()))
```

## Decision 2 — how strict?

**`MAX_ERR`** is the largest share of auto-called cells allowed to be wrong on the held-out ALICE
clusters, on each side. Stricter means more clusters in your queue. The table shows, for every
score × `MAX_ERR`, the cutoffs, the held-out auto share and error, and the queue it would give you.
The queue counts clusters that are not HC, not ALICE-labelled, and fall between the cutoffs or have
no score. It does not include the clone-V additions below.

**Clone-V check** (on by default, `--no-clone-v` turns it off): an auto-benign cluster goes to your
queue when it carries the sample's clone V, i.e. the top V of its most skewed non-CD8 cluster, with
the V at ≥ 30 % of its V calls. On ALICE samples the held-out score almost never auto-calls a
malignant cluster benign, so the check cannot be validated there. It exists for the no-V(D)J cohorts,
where a clonal cluster can still score low.

```python  # cell 9
queue = pd.read_csv(CM.QUEUE_CSV)
print(queue.round(3).to_string(index=False))
print(f"\ncurrent run: {res['program']} at MAX_ERR {res['max_err']:.0%}, clone-V check "
      f"{'on' if res.get('clone_v_check', True) else 'off'} ({res['clone_v_flipped_to_review']} clusters added) "
      f"-> {res['review_open_clusters']} open clusters in {res['review_open_samples']} samples")
print(cl[cl["source"].eq("review")].groupby("reason")["n_cells"].agg(clusters="size", cells="sum").to_string())
print("to change:  bash jobs/run_cluster_malig_model.sh --program <score> --max-err <x> [--no-clone-v]"
      "   (then re-run this notebook)")
```

## Manual calls

All sheets are in one PDF, `figures/nb26b_review/_all_review_sheets.pdf`, or as one PNG per sample.
Orange = open. Each sheet has:
- UMAP of the calls, and UMAP of the score
- bars: P per cluster, the two cutoffs, n cells, top TRBV and its share, and ALICE family / TCR+
  where the sample has TCR
- dotplot: top program genes (up | down) + lineage/MF markers

The `reason` column says why a cluster is in the queue:
- `uncertain`: P falls between the cutoffs, or there is no score.
- `clone_v`: the score called it benign, but it carries the sample's clone V gene, i.e. the top V of
  the sample's most skewed non-CD8 cluster.

Fill `user_call` in `tables/cluster_malig_review_v1.csv` (the `cluster_idx` column matches the numbers
on the sheet):
- `malignant` or `m`
- `benign` or `b`
- blank or `unsure` stays undefined (`review_pending`)

Then run **Merge**. Re-running the model job keeps filled rows.

**Whole samples out:** a sample listed in `tables/cluster_malig_excluded_samples_v1.csv` (sample_id,
note) gets `excluded_user` for every cluster, so `mal_cluster` is undefined there. This beats any
other call.

**Optional audit:** `figures/nb26b_review/_audit_auto_samples.pdf` has the same sheet for every sample
that was called fully automatically. To override one of those clusters, add a row to the review CSV
with its cluster key (`<sample_id>__<cluster_idx>`) and a `user_call`; Merge applies any key it finds.

**2026-09-28 review:**
- 93 queued clusters were called in `cluster_malig_review_v1.numbers` and copied into the CSV.
- D1__P311_thick 5 and 7 were set benign from the note on cluster 8.
- 8 calls that contradicted ALICE were set to ALICE (see `note`): 7 → malignant, including
  D1__P311_thick 10, and D3__P105_Skin 7 → benign.
- 3 samples were excluded.

```python  # cell 11
rev = pd.read_csv(CM.REVIEW_CSV, index_col="cluster", dtype={"user_call": str, "note": str},
                  keep_default_na=False)
open_ = rev[rev["user_call"].str.strip().eq("")]
print(f"review queue: {len(rev)} rows | open {len(open_)} clusters / {open_['n_cells'].astype(int).sum():,} "
      f"cells in {open_['sample_id'].nunique()} samples | filled {len(rev) - len(open_)}")
print(open_.groupby("study").agg(samples=("sample_id", "nunique"), clusters=("n_cells", "size"),
                                 cells=("n_cells", "sum")).to_string())
print(f"\nPDF: {CM.REVIEW_DIR / '_all_review_sheets.pdf'}")
for sid in list(dict.fromkeys(open_["sample_id"]))[:N_SHOW]:
    display(Image(str(CM.REVIEW_DIR / (re.sub(r"[^A-Za-z0-9_.-]", "_", sid) + ".png")), width=1100))
```

## Merge

Applies every filled `user_call` on top of the model job's calls (`cluster_malig_clusters_v1.csv` is
the base and is never overwritten, so this cell can be re-run as often as you like). Writes the per-cell
`skin_T_malignancy_cluster_v1.parquet`, which `malignancy.load_malignancy` reads as `mal_cluster`.

```python  # cell 13
cells = pd.read_parquet(CM.CLUSTERS_PQ, columns=["cell_id", "sample_id", "cluster"])
out, final = CM.finalize_calls(cl[["sample_id", "study", "n_cells", "source", "call", "prob"]], cells,
                               CM.REVIEW_CSV)
out.to_parquet(CM.CALLS_PQ, index=False)
print(final.groupby("source")["n_cells"].agg(clusters="size", cells="sum").to_string())
print(f"\nwrote {CM.CALLS_PQ.name}: {len(out):,} cells | mal_cluster defined on "
      f"{out['mal_cluster'].notna().mean():.1%} | malignant where defined "
      f"{out['mal_cluster'].mean():.1%}")
```

```python  # cell 14
# ---- what the call says, and how it sits next to the other three ----
smp = (out.assign(study=out["sample_id"].map(final.drop_duplicates("sample_id")
                                             .set_index("sample_id")["study"]),
                  m=out["mal_cluster"].astype("Float64"))
       .groupby(["study", "sample_id"])["m"].mean().dropna().reset_index())
studies = smp.groupby("study")["m"].median().sort_values().index
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(5, 3))
for i, s in enumerate(studies):
    v = smp.loc[smp["study"].eq(s), "m"]
    ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=8)
ax.set_xticks(range(len(studies)), studies, rotation=60, ha="right", fontsize=7)
ax.set_ylabel("malignant share of T cells")
ax.set_title(f"median sample is {smp['m'].median():.0%} malignant T cells ({len(smp)} samples)",
             fontsize=9)
fig.tight_layout()
fig.savefig(CM.FIG_DIR / "nb26b_malignant_share_by_study.png", dpi=150)
plt.show()

mal = MAL.load_malignancy(out["cell_id"])
print(MAL.concordance(mal).query("a == 'mal_cluster' or b == 'mal_cluster'").to_string())
```
