# helpers/

Flat modules, imported by notebooks after `sys.path.insert(0, str(NB_DIR / "helpers"))`.
Anything here that needs the project root uses `Path(__file__).resolve().parent.parent`.

## Atlas build & dedup — a 4-stage pipeline, run in this order

| module | role |
|---|---|
| `check_overlap.py` | CLI. Compares raw deposits barcode-by-barcode → `tables/atlas_dedup_v2.csv` |
| `make_dedup_decisions.py` | CLI. Turns that into the keep/drop ledger → `tables/atlas_dedup_v2_decisions.csv` |
| `atlas_join_helpers.py` | The build itself (`concat_joint`, …); consumes the ledger |
| `preflight_v2.py` | CLI. Validates the inputs before a `concat_joint` run |
| `atlas_v2_fixups.py` | Post-build metadata patches; driven by `jobs/apply_atlas_v2_fixups.py` |

The three CLIs have no importers — that is expected, they are entry points, not libraries.

## Analysis modules

| module | used by |
|---|---|
| `alice_helpers.py` | ALICE TCR-neighbourhood statistics — skin/blood malignancy + subclones |
| `skin_T_cnv_helpers.py` | inferCNV pipeline for T cells (both compartments, despite the name) |
| `subclone_helpers.py` | subclone calling and programs |
| `tcr_signaling_helpers.py` | TCR / co-stimulation signaling contrasts |
| `semantic_malig_helpers.py` | malignancy-plane fitting and plots |
| `semantic_clonality_helpers.py` | clonality transfer in a shared latent |
| `delta_axis_helpers.py` | the δ̂ malignancy axis; **the only module that reaches into the SemanticSCVI fork** (`SCVI_NB`, env `SCVI_NB_DIR`) |
| `degradome_data.py` / `degradome_helpers.py` | config / functions for the degradome notebook |
| `final_figures_helpers.py` | plotting for the retired `old/22_final_figures.ipynb` |

## The two CCC generations — deliberately not merged

`ccc_helpers.py` (v1) and `ccc_utils.py` (v2) share 19 function names: `ccc_utils.py` is a
rewrite that resolves its defaults through a config module (`ccc_data_sub.py`) instead of module
constants (`ccc_data.py`). Both are live:

- `27_ccc_subtype` uses `ccc_utils` + `ccc_data_sub`.
- `degradome_helpers.py` / `degradome_data.py` use `ccc_helpers` for eight functions that have
  **no** `ccc_utils` equivalent: `stream_lognorm_subset`, `_write_testset`, `load_ccc_obs`,
  `read_source_index`, `read_source_var`, `resolve_rows`, `sanitize_obs`, and a
  `build_ccc_celltype` with a different signature.

Retiring `ccc_helpers.py` + `ccc_data.py` therefore means porting those eight into `ccc_utils.py`
and re-validating the degradome outputs. Open follow-up; not a rename.

`ccc_helpers`/`ccc_data` are also what the archived `old/ccc_v1_*` notebooks import.
