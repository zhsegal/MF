"""The output contract: one directory you can paste into another model.

Every analysis in this project ends the same way -- `reports/<name>/for_analysis/` holding a
generated `INDEX.md`, the numerical results as plain CSV, and a prose `report.md`. Nothing else:
no figures, no `.npz`, no `.parquet`. The point is that the bundle has to be *self-orienting*.
Someone -- or some model -- opening it cold gets the question, the row counts, what the columns
mean, and the traps, without this repository and without asking.

The INDEX is generated rather than written by hand because a hand-written one drifts: it keeps
listing a file that got renamed, or quotes a row count from two runs ago. Here the file table is
read off the directory at write time, so it cannot be wrong.

Usage, as the last step of a job:

    import bundle
    bundle.write_bundle(
        OUT / "for_analysis",
        files=[OUT / "de_genes.csv", OUT / "report.md"],
        title="Malignant vs benign CD4, skin and blood",
        question="Which ligand-receptor genes does the malignant clone change ...",
        headline=["skin: 27 significant of 411 LR genes tested (7 up, 20 down)", ...],
        traps=["The benign arm is CD4_conv + CD4_naive in skin but CD4_conv alone in blood.", ...],
        columns={"de_genes.csv": "one row per gene; log2FC is malignant minus benign"},
    )
"""
from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path

BINARY_SUFFIXES = {".npz", ".npy", ".parquet", ".h5ad", ".h5", ".nc", ".pkl", ".pt",
                   ".png", ".pdf", ".svg", ".jpg", ".jpeg"}


def _rows(path):
    """Data-row count for a CSV (header excluded); None for anything else."""
    if path.suffix.lower() != ".csv":
        return None
    try:
        with path.open(newline="") as fh:
            return max(sum(1 for _ in csv.reader(fh)) - 1, 0)
    except Exception:
        return None


def write_bundle(out_dir, files, title, question=None, headline=(), traps=(),
                 columns=None, provenance=None, annotation="nonmal_T_labels_v1"):
    """Copy `files` into `out_dir` and generate INDEX.md beside them.

    `files`      paths to copy. Binary formats are refused -- the bundle is text.
    `title`      what this analysis is, in a line.
    `question`   the question it answers, in a sentence or two.
    `headline`   the numbers that matter, one string per line.
    `traps`      what would mislead a reader who does not know this project.
    `columns`    {filename: how to read its columns}.
    `provenance` {label: value} -- script, versions, run date, inputs.
    `annotation` which lineage call the result was built on. Recorded in every bundle because
                 the project has two and they give different arms (see KNOWLEDGE/CAVEATS.md).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = columns or {}
    provenance = dict(provenance or {})

    bad = [Path(f).name for f in files if Path(f).suffix.lower() in BINARY_SUFFIXES]
    if bad:
        raise ValueError(f"the bundle is text only; refusing binary files: {bad}")

    copied = []
    for f in files:
        f = Path(f)
        if not f.exists():
            raise FileNotFoundError(f"bundle input missing: {f}")
        dst = out_dir / f.name
        if f.resolve() != dst.resolve():
            shutil.copyfile(f, dst)
        copied.append(dst)

    L = []
    w = L.append
    w(f"# {title}")
    w("")
    if question:
        w(question)
        w("")
    w(f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Lineage annotation: **{annotation}**.")
    w("")
    if headline:
        w("## What it found")
        w("")
        for h in headline:
            w(f"- {h}")
        w("")
    w("## Files")
    w("")
    w("| file | rows | what it holds |")
    w("|---|---|---|")
    for p in sorted(copied, key=lambda x: (x.suffix != ".md", x.name)):
        n = _rows(p)
        w(f"| `{p.name}` | {n if n is not None else '-'} | {columns.get(p.name, '')} |")
    w("")
    if any(p.name in columns for p in copied):
        w("## How to read the columns")
        w("")
        for p in sorted(copied, key=lambda x: x.name):
            if p.name in columns:
                w(f"- **`{p.name}`** — {columns[p.name]}")
        w("")
    if traps:
        w("## Things that will mislead you if you do not know them")
        w("")
        for t in traps:
            w(f"- {t}")
        w("")
    if provenance:
        w("## Provenance")
        w("")
        for k, v in provenance.items():
            w(f"- **{k}**: {v}")
        w("")
    (out_dir / "INDEX.md").write_text("\n".join(L) + "\n")
    return out_dir
