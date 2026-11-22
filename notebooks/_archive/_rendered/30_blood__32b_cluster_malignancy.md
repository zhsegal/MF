# 30_blood/32b_cluster_malignancy.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/30_blood/32b_cluster_malignancy.ipynb` and is never run._

# 32b — Cluster-level malignancy in blood: your decisions and manual calls

The blood analogue of `26b_cluster_malignancy`, on the same two jobs with `--tissue blood`.
Everything that differs is a property of the compartment, not a different method.

| | where | what |
|---|---|---|
| clustering | `jobs/run_cluster_malig.sh --tissue blood` | per **donor** Leiden 1.0 → UMAP; cluster pseudobulks; TRBV/TRAV call; ALICE family |
| Step 1 | `jobs/run_cluster_malig_model.sh --tissue blood` | cluster ALICE label; scores compared leave-one-study-out; cutoffs from held-out P |
| Step 2 | same job | HC → benign · ALICE donors keep ALICE · clear → auto · rest → review |
| **Decision 1** | here | which score |
| **Decision 2** | here | how strict (`MAX_ERR`) |
| **Manual calls** | `tables/blood_cluster_malig_review_v1.csv` | `m` / `b`, using `figures/nb32b_review/` |
| **Merge** | here | → `blood_T_malignancy_cluster_v1.parquet` → `malignancy.load_malignancy(..., tissue="blood")["mal_cluster"]` |

### Four things that are different in blood, and why

1. **The unit is the donor, not `sample_id`.** The geskin `sample_id` is an HTO lane pooling 5–6
   donors; a cluster spanning one lane would span several patients and its ALICE family share
   would mean nothing. The column is still called `sample_id` so the shared code is unchanged.
2. **The benign anchor is scarce.** SS blood is ~92% malignant CD4, so a benign cluster that is
   ≥80% CD4 barely exists; `MIN_CD4` is relaxed to 0.5 and the HC donors plus CD8 carry the
   negative side. If `consensus_signature` raises "no sample has both", that is this problem.
3. **The markers are the Sézary panel**, not the skin one. `CCR7`, `SELL` and `TCF7` are
   deliberately **not** in `CTCL_DOWN`: Sézary cells are central-memory-like, so their positivity
   is not evidence that a cluster is benign — the single most common way to mis-call SS blood.
4. **There is no `knn_frac` arm.** `mal_knn` is a skin-only call from the retired nb26, so that
   reference column is NaN here.

**Prerequisite:** `32_malignancy_tcr_cnv` rerun on the v2 annotation — `32b` reads
`alice_malignancy_blood_v2.parquet` and `tables/blood_tcr_clonality_per_sample.csv` (Step 3b).
The August `_v1` artifacts are from the v1 cohort and their cell ids do not match this object.

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
N_SHOW = 3          # review sheets shown inline (all are in figures/nb32b_review/)
```

```
NB_DIR = /home/projects/nyosef/zvise/MF
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

# Point BOTH modules at blood before any path is read. Without this the notebook would load the
# skin parquets, reindex them onto blood cell ids and report an empty call as a real one.
CM.use_tissue("blood")
MAL.use_tissue("blood")
print("tissue:", CM.TISSUE, "| object:", CM.TCR_OBJ.name, "| counts layer:", CM.COUNTS)
print("group key:", CM.GROUP_KEY, "| cell types:", CM.CELL_TYPES)
print("ALICE:", CM.ALICE_PQ.name, f"({CM.ALICE_COL})")
print("markers UP :", CM.CTCL_UP)
print("markers DOWN:", CM.CTCL_DOWN, " <- note CCR7/SELL/TCF7 are deliberately absent")
```

```
tissue: blood | object: blood_T_annotated.h5ad | counts layer: counts
group key: donor | cell types: ['CD4', 'CD4_Treg', 'CD8', 'NK']
ALICE: alice_malignancy_blood_v2.parquet (tcr_malignant_alice)
markers UP : ['KIR3DL2', 'CADM1', 'TWIST1', 'PLS3', 'DNM3', 'TNFRSF8', 'TNFRSF1B', 'TOX', 'GATA3']
markers DOWN: ['CD7', 'DPP4', 'CD27', 'CD5', 'CD6']  <- note CCR7/SELL/TCF7 are deliberately absent
```

## What the jobs produced

Sample roles are as in skin: `train` (clonal ALICE donors), `train_hc` (healthy controls, the
negatives), `tcr_nonclonal` (ALICE donors with no founder family — their labels are kept but not
trained on), `apply` (everything else) and `excluded` (γδ / CD8-aggressive entities).

The purity ceiling is the best a cluster-level call can do at res 1.0: the share of TCR+ cells a
call gets right if every cluster took its majority ALICE label.

```python  # cell 4
man = json.loads(CM.MANIFEST_JSON.read_text())
res = json.loads(CM.MODEL_JSON.read_text())
cl = pd.read_csv(CM.CLUSTERS_CSV, index_col="cluster")

print(f"clustering: {man['n_samples']} {man.get('group_key', 'sample')}s / "
      f"{man['n_clusters']:,} clusters / {man['n_cells']:,} cells | res {man['params']['res']} "
      f"| written {man['written']}")
print(f"  skipped (<= 30 T cells): {len(man['skipped'])} | failed: {man['failed']}")
print(f"model run: {res['written']} | score = {res['program']} (asked: {res['program_arg']}) | "
      f"MAX_ERR {res['max_err']:.0%} | cutoffs {res['p_lo']:.3f} / {res['p_hi']:.3f}")
print(f"  TCR+ in mixed clusters {res['mixed_tcr_frac']:.1%} | purity ceiling "
      f"{res['purity_ceiling']:.1%} | HC cells the score alone calls malignant "
      f"{res['hc_fp_cells']:.1%}")

print("\nroles:")
print(cl.groupby("role").agg(samples=("sample_id", "nunique"), clusters=("n_cells", "size"),
                             cells=("n_cells", "sum")).to_string())

# The blood-specific risk, as a number: how much benign CD4 is there to anchor the program on?
if "frac_CD4" in cl:
    ben = cl[cl["tcr_label"].eq("benign")]
    print(f"\nbenign-labelled clusters: {len(ben)} | of those >= 50% CD4: "
          f"{int(ben['frac_CD4'].ge(0.5).sum())} | >= 80% CD4 (the skin gate): "
          f"{int(ben['frac_CD4'].ge(0.8).sum())}")
    print("If the >=80% count is near zero, MIN_CD4=0.5 is what made this run possible at all.")
```

```
clustering: 131 donors / 1,438 clusters / 667,584 cells | res 1.0 | written 2026-10-23 03:39
  skipped (<= 30 T cells): 0 | failed: {}
model run: 2026-10-23 03:44 | score = program (asked: auto) | MAX_ERR 2% | cutoffs 0.326 / 0.326
  TCR+ in mixed clusters 3.8% | purity ceiling 98.1% | HC cells the score alone calls malignant 0.0%

roles:

               samples  clusters   cells
role                                    
apply               35       373  102924
tcr_nonclonal        5        70   85804
train               84       922  434566
train_hc             7        73   44290

benign-labelled clusters: 438 | of those >= 50% CD4: 261 | >= 80% CD4 (the skin gate): 174
If the >=80% count is near zero, MIN_CD4=0.5 is what made this run possible at all.
```

## Decision 1 — which score?

Each candidate is scored leave-one-study-out on the ALICE-labelled clusters; the program is
relearned inside every fold.

- **program** — genes moving the same way in ≥75% of donors, malignant vs benign clusters within
  the same donor.
- **ctcl_apriori** — the Sézary marker lists above.
- **vskew** — how dominant the cluster's top TRBV/TRAV is against a polyclonal null.
- **program+vskew** — both.
- **knn_frac** — NaN in blood (no `mal_knn`); the column is kept so the tables keep their shape.

In blood, expect **vskew to carry more weight than in skin**: SS clones are large and the blood
studies are 5′ with real V reads, where several skin cohorts have none.

```python  # cell 6
cv = pd.read_csv(CM.CV_CSV)
cols = ["coverage", "auc_cluster", "auc_cells", "auc_within_median", "auto_cells_frac",
        "err_mal", "err_ben", "p_lo", "p_hi"]
print(f"held-out, pooled (MAX_ERR {res['max_err']:.0%}):")
print(cv[cv["fold"].eq("pooled")].set_index("candidate")[cols].round(3).to_string())
print("\nauto_cells_frac within each held-out study:")
print(cv[cv["fold"].ne("pooled")].pivot(index="fold", columns="candidate",
                                        values="auto_cells_frac").round(2).to_string())
_fig = CM.FIG_DIR / "nb32b_candidates_auto_share.png"
if _fig.exists():
    display(Image(str(_fig), width=500))
```

```
held-out, pooled (MAX_ERR 2%):
               coverage  auc_cluster  auc_cells  auc_within_median  auto_cells_frac  err_mal  err_ben   p_lo   p_hi
candidate                                                                                                          
program           1.000        1.000      1.000                1.0            1.000    0.005    0.018  0.326  0.326
ctcl_apriori      1.000        0.990      0.995                1.0            0.962    0.020    0.010  0.262  0.461
vskew             0.995        0.993      0.999                1.0            1.000    0.018    0.014  0.490  0.490
program+vskew     1.000        1.000      1.000                1.0            1.000    0.000    0.000  0.429  0.429
knn_frac          0.000          NaN        NaN                NaN            0.000    0.000    0.000 -0.000  1.000

auto_cells_frac within each held-out study:

candidate        ctcl_apriori  knn_frac  program  program+vskew  vskew
fold                                                                  
borcherding2023          0.99       0.0      1.0            1.0    1.0
buus2025                 0.95       0.0      1.0            1.0    1.0
dorando2026              1.00       0.0      1.0            1.0    1.0
gaydosik2022             0.97       0.0      1.0            1.0    1.0
geskin2026               0.96       0.0      1.0            1.0    1.0
harro2023                0.88       0.0      1.0            1.0    1.0
herrera2021              0.97       0.0      1.0            1.0    1.0
ren2023                  0.97       0.0      1.0            1.0    1.0
rindler2021_fi           1.00       0.0      1.0            1.0    1.0

<IPython.core.display.Image object>
```

```python  # cell 7
prog = pd.read_csv(CM.PROGRAM_CSV)
n_folds = len(res["folds"])
print(f"learned program: {int(prog.direction.eq('up').sum())} up / "
      f"{int(prog.direction.eq('down').sum())} down, from "
      f"{res['n_train_donors_program']} donors")
for d in ("up", "down"):
    g = prog[prog["direction"].eq(d)]
    print(f"\n{d}:")
    print(", ".join(f"{r.gene}({r.n_folds_selected})" for r in g.itertuples()))

# Sanity: the program must not be a memory/naive axis. In SS that is the classic confound --
# Sezary cells are central-memory-like, so a program that separates CM from EM separates the
# arms for the wrong reason.
MEMORY = {"CCR7", "SELL", "TCF7", "LEF1", "IL7R", "S1PR1", "KLF2", "GZMB", "GNLY", "NKG7",
          "CX3CR1", "KLRG1"}
hits = sorted(set(prog["gene"]) & MEMORY)
print(f"\nmemory/differentiation genes in the learned program: {hits or 'none'}")
if hits:
    print("*** Check these. A program resting on them is a CM-vs-EM axis, not a malignancy "
          "axis, and it will call every central-memory T cell in a healthy donor malignant. ***")
```

```
learned program: 50 up / 50 down, from 69 donors

up:
KIR3DL2(9), DNM3OS(9), HACD1(9), DNM3(9), PTHLH(9), NEDD4L(9), PTH2R(9), NPAS3(9), CA8(9), LINC00271(9), PAGE5(9), GNG4(8), HPGDS(8), IGFL2(8), HDAC9(9), TPO(9), LINC01480(9), TNFSF11(9), ANK1(9), TOX(9), MLF1(9), KSR2(9), TNFSF9(9), TWIST1(9), CEMIP(9), IKZF2(9), CXCL13(9), PDE7B(8), IGFBP4(9), CD70(9), KRT7(9), LINC01362(8), MANSC4(9), GTSF1(9), CDO1(8), ZNF503(9), C14orf132(9), GPR15(9), KCNQ2(6), ID3(8), EVC2(9), PLK2(8), AC246817.2(6), KCNK1(8), KLHL42(6), BCAS1(8), FOXP2(8), CDH1(8), ADGRE1(3), CPNE2(2)

down:
TMIGD2(9), DPP4(9), GZMK(9), C1orf162(9), THEMIS(9), HOPX(9), KLRB1(9), LGALS3BP(9), PECAM1(9), ITGA4(9), CCL5(9), APBA2(9), CD7(9), GSTP1(9), AIF1(9), ABI3(9), GSTM2(9), SERPINB6(9), STAT4(9), S100B(8), TRPS1(9), LAIR1(9), PLAC8(9), GPR25(9), CXCR6(9), IGFBP3(9), GZMA(9), LINC01934(8), CD300A(8), NSG1(8), PDE4B(9), AC005842.1(9), RAB34(9), CTSW(9), PTGER2(9), CDCA7L(9), PPP2R2B(9), CCR6(9), NKG7(8), SYTL2(9), MATK(9), RALGPS2(9), ANKRD55(8), RIPK2(9), ENG(7), NME4(8), CTSH(2), IL18R1(7), BAG2(2), ALOX5(8)

memory/differentiation genes in the learned program: ['NKG7']
*** Check these. A program resting on them is a CM-vs-EM axis, not a malignancy axis, and it will call every central-memory T cell in a healthy donor malignant. ***
```

## Decision 2 — how strict?

`MAX_ERR` is the largest share of auto-called cells allowed to be wrong on the held-out ALICE
clusters, per side. Stricter means a longer review queue.

The clone-V check is on by default: an auto-benign cluster goes to the queue when it carries the
donor's clone V gene at ≥30% of its V calls. In blood this matters more than in skin — the
no-V(D)J cohorts are where a clonal cluster can still score low, and in SS a missed malignant
cluster is a large fraction of the compartment.

```python  # cell 9
queue = pd.read_csv(CM.QUEUE_CSV)
print(queue.round(3).to_string(index=False))
print(f"\ncurrent run: {res['program']} at MAX_ERR {res['max_err']:.0%}, clone-V "
      f"{'on' if res.get('clone_v_check', True) else 'off'} "
      f"({res['clone_v_flipped_to_review']} added) -> {res['review_open_clusters']} open clusters "
      f"in {res['review_open_samples']} donors")
print(cl[cl["source"].eq("review")].groupby("reason")["n_cells"]
      .agg(clusters="size", cells="sum").to_string())
print("\nto change:  bash jobs/run_cluster_malig_model.sh --tissue blood --program <score> "
      "--max-err <x> [--no-clone-v]")
```

```
candidate  max_err  p_lo  p_hi  heldout_auto_cells_frac  heldout_err_mal  heldout_err_ben  review_clusters  review_cells  review_samples  auto_clusters
      program     0.01 0.288 0.292                    0.991            0.005            0.000                0             0               0            492
      program     0.02 0.326 0.326                    1.000            0.005            0.018                0             0               0            492
      program     0.05 0.372 0.372                    1.000            0.002            0.019                0             0               0            492
 ctcl_apriori     0.01 0.262 0.637                    0.925            0.009            0.010               90         22021              60            402
 ctcl_apriori     0.02 0.262 0.461                    0.962            0.020            0.010               42          8574              39            450
 ctcl_apriori     0.05 0.305 0.305                    1.000            0.034            0.034                0             0               0            492
        vskew     0.01 0.456 0.620                    0.982            0.009            0.010               40          4291              33            452
        vskew     0.02 0.490 0.490                    1.000            0.018            0.014               25          2452              20            467
        vskew     0.05 0.418 0.418                    1.000            0.024            0.006               25          2452              20            467
program+vskew     0.01 0.424 0.424                    1.000            0.000            0.000                0             0               0            492
program+vskew     0.02 0.429 0.429                    1.000            0.000            0.000                0             0               0            492
program+vskew     0.05 0.428 0.428                    1.000            0.000            0.000                0             0               0            492

current run: program at MAX_ERR 2%, clone-V on (27 added) -> 27 open clusters in 26 donors
         clusters  cells
reason                  
clone_v        27   3002

to change:  bash jobs/run_cluster_malig_model.sh --tissue blood --program <score> --max-err <x> [--no-clone-v]
```

## Manual calls

Sheets are in `figures/nb32b_review/`. Fill `user_call` in
`tables/blood_cluster_malig_review_v1.csv` — `malignant`/`m`, `benign`/`b`, blank or `unsure`
stays undefined (`review_pending`) — then run **Merge**. Re-running the model job keeps filled rows.

**Reading a blood sheet.** Two traps specific to this compartment:

- A cluster that is CD7-low / CD26-low **and** carries the donor's clone V is malignant even when
  the program score is middling: in SS those two losses are the diagnostic immunophenotype.
- A CCR7+ SELL+ TCF7+ cluster is **not** benign by that fact alone. Sézary cells are
  central-memory-like; this is the single most common way to mis-call SS blood.

Whole donors can be dropped via `tables/blood_cluster_malig_excluded_samples_v1.csv`
(`sample_id`, `note`), which beats every other call.

```python  # cell 11
rev = pd.read_csv(CM.REVIEW_CSV, index_col="cluster",
                  dtype={"user_call": str, "note": str}, keep_default_na=False)
open_ = rev[rev["user_call"].str.strip().eq("")]
print(f"review queue: {len(rev)} rows | open {len(open_)} clusters / "
      f"{open_['n_cells'].astype(int).sum():,} cells in {open_['sample_id'].nunique()} donors "
      f"| filled {len(rev) - len(open_)}")
if len(open_):
    print(open_.groupby("study").agg(donors=("sample_id", "nunique"),
                                     clusters=("n_cells", "size"),
                                     cells=("n_cells", "sum")).to_string())
print(f"\nPDF: {CM.REVIEW_DIR / '_all_review_sheets.pdf'}")
for sid in list(dict.fromkeys(open_["sample_id"]))[:N_SHOW]:
    png = CM.REVIEW_DIR / (re.sub(r"[^A-Za-z0-9_.-]", "_", sid) + ".png")
    if png.exists():
        display(Image(str(png), width=1100))
```

```
review queue: 27 rows | open 0 clusters / 0 cells in 0 donors | filled 27

PDF: /home/projects/nyosef/zvise/MF/figures/nb32b_review/_all_review_sheets.pdf
```

## Cell-type QC — review clusters that are not what `nb31` called them

The 27-cluster queue surfaced an annotation problem rather than a malignancy one. Every open
cluster was called **benign** (2026-10-24), and reading them against the cluster pseudobulk sorts
them into three kinds:

- **`CD8A`+ `CD8B`− cytotoxic clusters with almost no TCR.** NK cells carry CD8 as an αα
  homodimer and so are `CD8A`+ `CD8B`−; a CD8 T cell is `CD8B`+. `nb31` labelled on `CD8A`, so NK
  clusters arrived here as CD8. `CM.nk_vs_t_scores` separates them on two marker sets, and
  `n_tcr / n_cells` is the independent check — an NK cluster has no rearranged TRB, so a cluster
  that scores NK *and* recovers no TCR is NK twice over. **These are relabelled `NK`.**
- **Clusters already labelled `NK`** (`frac_NK ≈ 1`, `n_tcr = 0`). No change: being NK is exactly
  what takes them out of the CD4 malignancy question.
- **Genuine CD8 T** (`CD8B`+, TCR recovered) and the CD4/mixed clusters. Left alone.

Writes `data/atlas_joint/blood_T_celltype_corrections_v1.csv`, which `CM.read_obs` overlays on
`cell_type_T` for every downstream consumer (the target screen reads it through that function).
`blood_T_annotated.h5ad` is **not** rewritten: the object stays the record of what `nb31` called,
this file is the record of what the review found, and `cell_type_T_nb31` is kept beside the
corrected column so the difference is always visible.

```python  # cell 13
sc = CM.nk_vs_t_scores()
sc["nb31"] = np.select([sc["frac_NK"] > 0.5, sc["frac_CD8"] > 0.5, sc["frac_CD4"] > 0.5],
                       ["NK", "CD8", "CD4"], default="mixed")
cols = ["n_cells", "n_tcr", "frac_CD4", "frac_CD8", "frac_NK", "t_score", "nk_score", "delta",
        "tcr_rate", "nb31", "nk_like"]
print(f"rule: nk_score - t_score >= {CM.NK_DELTA} AND tcr_rate < {CM.NK_MAX_TCR}\n")
print(sc.loc[rev.index, cols].sort_values("delta", ascending=False).round(2).to_string())

fix = sc.loc[rev.index].query("nb31 == 'CD8' and nk_like")
cells_cl = pd.read_parquet(CM.CLUSTERS_PQ, columns=["cell_id", "cluster"])
corr = (cells_cl[cells_cl["cluster"].isin(fix.index)]
        .assign(cell_type_T="NK", cell_type_T_nb31="CD8",
                reason=f"nb32b review: nk-t delta >= {CM.NK_DELTA}, tcr_rate < {CM.NK_MAX_TCR}"))
corr.to_csv(CM.CELLTYPE_FIX_CSV, index=False)
CM.CELLTYPE_FIX_CSV.with_suffix(".json").write_text(json.dumps(
    {"written": pd.Timestamp.now().isoformat(timespec="seconds"), "source": "notebooks/30_blood/32b",
     "rule": {"t_genes": CM.T_SCORE_GENES, "nk_genes": CM.NK_SCORE_GENES,
              "nk_delta": CM.NK_DELTA, "nk_max_tcr": CM.NK_MAX_TCR},
     "scope": "the nb32b review queue only", "clusters": sorted(fix.index),
     "n_cells": int(len(corr)), "from": "CD8", "to": "NK"}, indent=1))
print(f"\nCD8 -> NK: {len(fix)} clusters, {len(corr):,} cells -> {CM.CELLTYPE_FIX_CSV.name}")

# Atlas-wide context. NOT applied: the rule is calibrated on clusters that were looked at, and
# blood `cell_type_T` has only 606 NK cells in 667,584, which is not a credible NK fraction for
# PBMC. This is a finding about nb31, to be fixed there, not silently here.
wide = sc.query("nb31 == 'CD8' and nk_like")
cd8 = sc.query("nb31 == 'CD8'")
print(f"\nNOT APPLIED — atlas-wide, {len(wide)} of {len(cd8)} CD8-labelled clusters carry the "
      f"same signature ({int(wide['n_cells'].sum()):,} of {int(cd8['n_cells'].sum()):,} cells, "
      f"{wide['n_cells'].sum() / cd8['n_cells'].sum():.0%}). Blood CD8-vs-NK needs its own pass "
      "in nb31; until then the CD8 arm of the target screen is a CD8+NK mixture and the NK arm "
      "is too small to trust.")
```

```
rule: nk_score - t_score >= 1.0 AND tcr_rate < 0.15

                   n_cells  n_tcr  frac_CD4  frac_CD8  frac_NK  t_score  nk_score  delta  tcr_rate nb31  nk_like
cluster                                                                                                         
H__SS1__5              245      3      0.00      1.00     0.00     1.00      3.29   2.29      0.01  CD8     True
H__MFIV1__5            243      2      0.00      1.00     0.00     1.10      3.31   2.21      0.01  CD8     True
B6__P9__13              64      6      0.00      1.00     0.00     1.20      2.97   1.77      0.09  CD8     True
B4__SZ5_HTO6__8        152      4      0.07      0.93     0.00     1.20      2.87   1.67      0.03  CD8     True
B4__SZ5_HTO4__9        137      6      0.06      0.94     0.00     1.22      2.86   1.64      0.04  CD8     True
B4__SZ5_HTO2__7        237      9      0.06      0.94     0.00     1.16      2.78   1.62      0.04  CD8     True
B4__SZ5_HTO3__8        162      7      0.05      0.95     0.00     1.25      2.82   1.57      0.04  CD8     True
B4__SZ5_HTO1__7        117      7      0.09      0.91     0.00     1.21      2.75   1.55      0.06  CD8     True
B4__SZ5_HTO5__9        151      7      0.03      0.97     0.00     1.11      2.64   1.53      0.05  CD8     True
B4__SZ39_HTO1__10       57      3      0.05      0.95     0.00     0.99      2.19   1.20      0.05  CD8     True
B4__SZ39_HTO2__8        85      5      0.06      0.94     0.00     1.00      2.19   1.19      0.06  CD8     True
B4__SZ39_HTO6__9        66      9      0.02      0.98     0.00     1.06      2.14   1.08      0.14  CD8     True
H__SS2__3              198     40      0.10      0.90     0.00     1.93      2.80   0.87      0.20  CD8    False
H__SS4__7               35     16      0.17      0.83     0.00     1.76      2.16   0.40      0.46  CD8    False
B4__PF48_HTO2__10       78      0      0.00      0.00     1.00     1.45      1.64   0.19      0.00   NK    False
B6__P6__9               70     39      0.16      0.84     0.00     2.03      2.02  -0.01      0.56  CD8    False
B4__PF47_HTO2__7       219      0      0.00      0.00     1.00     1.55      1.37  -0.18      0.00   NK    False
B4__SZ30_HTO7__7        77     37      0.52      0.48     0.00     1.33      1.13  -0.20      0.48  CD4    False
B4__SZ30_HTO9__7        57     24      0.47      0.53     0.00     1.29      1.08  -0.21      0.42  CD8    False
B4__SZ30_HTO6__7        85     33      0.49      0.51     0.00     1.33      1.10  -0.23      0.39  CD8    False
B4__SZ38_HTO1__10       42     24      0.69      0.31     0.00     1.70      1.31  -0.39      0.57  CD4    False
B4__PF47_HTO4__10       68      0      0.19      0.00     0.81     1.62      1.19  -0.43      0.00   NK    False
B4__SZ33_HTO6__5        77     33      0.78      0.22     0.00     2.25      1.31  -0.93      0.43  CD4    False
H__SS3__7               57     53      0.56      0.44     0.00     1.82      0.87  -0.95      0.93  CD4    False
B4__SZ39_HTO3__10       27     23      0.30      0.70     0.00     1.33      0.13  -1.20      0.85  CD8    False
H__SS3__6               70     68      0.80      0.20     0.00     2.25      0.48  -1.77      0.97  CD4    False
B6__P2__7              126     30      0.99      0.01     0.00     1.98      0.14  -1.83      0.24  CD4    False


CD8 -> NK: 12 clusters, 1,716 cells -> blood_T_celltype_corrections_v1.csv

NOT APPLIED — atlas-wide, 102 of 399 CD8-labelled clusters carry the same signature (23,058 of 93,547 cells, 25%). Blood CD8-vs-NK needs its own pass in nb31; until then the CD8 arm of the target screen is a CD8+NK mixture and the NK arm is too small to trust.
```

## Merge

Applies every filled `user_call` on top of the job's calls and writes the per-cell
`blood_T_malignancy_cluster_v1.parquet`, which `malignancy.load_malignancy(..., tissue="blood")`
reads as `mal_cluster`. The base table is never overwritten, so this cell is re-runnable.

```python  # cell 15
cells_pq = pd.read_parquet(CM.CLUSTERS_PQ, columns=["cell_id", "sample_id", "cluster"])
out, final = CM.finalize_calls(
    cl[["sample_id", "study", "n_cells", "source", "call", "prob"]], cells_pq, CM.REVIEW_CSV)
out.to_parquet(CM.CALLS_PQ, index=False)
print(final.groupby("source")["n_cells"].agg(clusters="size", cells="sum").to_string())
print(f"\nwrote {CM.CALLS_PQ.name}: {len(out):,} cells | mal_cluster defined on "
      f"{out['mal_cluster'].notna().mean():.1%} | malignant where defined "
      f"{out['mal_cluster'].mean():.1%}")
```

```
clusters   cells
source                  
auto         465  128856
hc            73   44290
tcr          873  491436
user          27    3002

wrote blood_T_malignancy_cluster_v1.parquet: 667,584 cells | mal_cluster defined on 100.0% | malignant where defined 52.2%
```

```python  # cell 16
# ---- what the call says, and how it sits next to the TCR and CNV arms ----
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
ax.set_title(f"median donor is {smp['m'].median():.0%} malignant T cells "
             f"({len(smp)} donors)", fontsize=9)
fig.tight_layout()
fig.savefig(CM.FIG_DIR / "nb32b_malignant_share_by_study.png", dpi=150)
plt.show()

mal = MAL.load_malignancy(out["cell_id"], strict=False)
print(MAL.coverage_report(mal).to_string())
print()
print(MAL.concordance(mal).query("a == 'mal_cluster' or b == 'mal_cluster'").to_string())
print("\nHC donors should be ~0% malignant here BY CONSTRUCTION (the job assigns HC benign). "
      "That is an input, not a validation -- never quote it as evidence of specificity.")
```

```
<Figure size 500x300 with 1 Axes>
/home/projects/nyosef/zvise/MF/helpers/malignancy.py:181: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["mal_tcr"] = (r.fillna(False).to_numpy().astype(bool) & defined)

/home/projects/nyosef/zvise/MF/helpers/malignancy.py:204: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  defined = (call.notna() & ok.fillna(False).astype(bool)).to_numpy()
/home/projects/nyosef/zvise/MF/helpers/malignancy.py:205: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["mal_cnv"] = call.fillna(False).to_numpy().astype(bool) & defined

             n_defined  frac_defined  n_malignant  frac_malignant_where_defined
mal_tcr       557772.0        0.8355     292553.0                        0.5245
mal_cnv       203868.0        0.3054     127665.0                        0.6262
mal_knn            0.0        0.0000          0.0                           NaN
mal_cluster   667584.0        1.0000     348693.0                        0.5223
             n_defined  frac_defined  n_malignant  frac_malignant_where_defined
mal_tcr       557772.0        0.8355     292553.0                        0.5245
mal_cnv       203868.0        0.3054     127665.0                        0.6262
mal_knn            0.0        0.0000          0.0                           NaN
mal_cluster   667584.0        1.0000     348693.0                        0.5223

  group        a            b  n_both   agree  jaccard   a_pos   b_pos
2   all  mal_tcr  mal_cluster  557772  0.9146   0.8598  0.5245  0.6078
4   all  mal_cnv  mal_cluster  203868  0.8779   0.8335  0.6262  0.7184
5   all  mal_knn  mal_cluster       0     NaN      NaN     NaN     NaN

HC donors should be ~0% malignant here BY CONSTRUCTION (the job assigns HC benign). That is an input, not a validation -- never quote it as evidence of specificity.
```

## Next

`notebooks/40_targets/43_target_screen_blood` consumes this call. Build its input with

```
jobs/run_target_screen.sh --tissue blood
```
