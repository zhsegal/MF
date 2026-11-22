# Archived notebooks — not an execution surface

**Nothing here is ever run.** These notebooks are kept as the record of how the project's frozen
artifacts were made: which Leiden cluster became which label, which parameters were chosen, what
the printed numbers were.

- `_rendered/` holds a markdown rendering of every notebook — its markdown cells, its code, and
  its text output, with images dropped and long outputs truncated. **Grep here first**: git
  stores `.ipynb` files stripped of outputs (`.gitattributes`: `*.ipynb filter=nbstrip`), so the
  rendered copies are the only version-controlled record of what these notebooks printed.
- The `.ipynb` files themselves keep their outputs in the working tree.

`KNOWLEDGE/HISTORY.md` says what each notebook produced, which ones are already scripted in
`jobs/`, and which contain analysis that exists nowhere else and would need porting before it
could be re-run.

If you need one of these analyses again: port it to `jobs/` as a headless script with a bsub
wrapper and a `for_analysis/` bundle. Do not re-execute the notebook.
