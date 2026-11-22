#!/usr/bin/env python
"""Does the knowledge base still describe the repository?

`KNOWLEDGE/` is the project's source of truth, and the only thing keeping it true is that whoever
finishes an analysis also updates it (`CLAUDE.md`, "Closing the loop"). That is an instruction,
not an enforcement: nothing fails when the step is skipped, and the failure is silent -- the next
agent reads a knowledge base that is confidently out of date, which is worse than no knowledge
base at all.

This job is the enforcement. It is pure bookkeeping: it reads text, compares it to the directory
tree, and never opens a data object, so it runs in seconds on the login node. Every check answers
"is the map still the territory", never "is the science right".

    python jobs/run_knowledge_check.py            # report
    python jobs/run_knowledge_check.py --strict   # exit 1 on any ERROR (for a pre-commit hook)

Checks, and what each one catches:

  bundle_registered  a result exists that RESULTS.md never heard of -- the commonest drift, and
                     the one that makes a number unciteable because nobody knows its vintage
  bundle_shape       a bundle missing its INDEX.md, or carrying a binary, i.e. not pasteable
  job_registered     a jobs/*.py that ANALYSES.md and HISTORY.md both ignore -- an analysis a
                     later agent cannot find, so it gets rewritten instead of re-run
  job_wrapper        a run_*.py with no run_*.sh, so it has no documented way onto LSF
  helper_documented  a helpers/*.py missing from helpers/README.md
  path_ref           KNOWLEDGE cites a path that no longer exists
  line_ref           KNOWLEDGE cites file:line past the end of that file (a refactor moved it)
  results_annotation a RESULTS.md row with an empty annotation column -- an unciteable number
  memory_pointer     a project-type Claude memory that holds content instead of pointing at
                     KNOWLEDGE/. Machine-local, so this one is skipped where the store is absent.

A finding is an ERROR when it would mislead a reader, and a WARN when it is only untidy.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NB_DIR / "helpers"))

KB = NB_DIR / "KNOWLEDGE"
OUT = NB_DIR / "reports" / "knowledge_check"

# Claude's memory store: outside the repo, one machine, keyed to the working directory. Absent on
# a fresh checkout, which is exactly why a project fact must not live only there.
MEMORY_DIR = (NB_DIR.parent / ".claude-wis" / "projects"
              / ("-" + str(NB_DIR).strip("/").replace("/", "-")) / "memory")

# memories that are about HOW to work rather than about the project, so they legitimately hold
# content of their own rather than a pointer into KNOWLEDGE/
MEMORY_EXEMPT = {"wexac-queue-guide", "submit-and-watch-jobs", "mf-project-map"}

# jobs that are imported rather than run, or that are drivers for another job
JOB_SKIP_PREFIX = ("_",)

BINARY_SUFFIXES = {".npz", ".npy", ".parquet", ".h5ad", ".h5", ".nc", ".pkl", ".pt",
                   ".png", ".pdf", ".svg", ".jpg", ".jpeg", ".xlsx"}

# A path citation inside prose. Deliberately conservative: it must start with one of the
# project's own top-level directories, so ordinary words with slashes are not candidates.
ROOTS = ("data", "jobs", "helpers", "reports", "docs", "notebooks", "tables", "figures",
         "models", "KNOWLEDGE")
PATH_RE = re.compile(r"`((?:" + "|".join(ROOTS) + r")/[^`\s]+)`")
LINE_RE = re.compile(r"`([^`\s]+\.(?:py|md|sh|csv|json))[:#](\d+)[-\d]*`")

# a path citation that is a pattern, a placeholder or a shell brace expansion, not a real file
PLACEHOLDER = re.compile(r"[*?<>{}]|\.\.\.")

# These two files exist to describe what is GONE -- the retirement ledger and the notebook
# decoder -- so a path in them that no longer resolves is expected, not a defect. Still worth
# reporting, because a reader will try to follow it, but as a WARN.
HISTORICAL = {"HISTORY.md", "NOTEBOOK_MAP.md"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Findings:
    """Accumulator. `add` is the only way in, so every finding has the same shape."""

    def __init__(self):
        self.rows = []

    def add(self, check, severity, subject, detail, fix):
        assert severity in ("ERROR", "WARN")
        self.rows.append(dict(check=check, severity=severity, subject=subject,
                              detail=detail, suggested_fix=fix))

    def errors(self):
        return [r for r in self.rows if r["severity"] == "ERROR"]


def _kb_text():
    """Every KNOWLEDGE file as one blob, plus the per-file texts."""
    per_file = {p.name: p.read_text() for p in sorted(KB.glob("*.md"))}
    return "\n".join(per_file.values()), per_file


# ======================================================================================
# checks
# ======================================================================================
def check_bundles(f, results_md):
    """Every for_analysis/ bundle is registered in RESULTS.md, and is actually pasteable."""
    bundles = sorted(NB_DIR.glob("reports/*/for_analysis"))
    if not bundles:
        f.add("bundle_registered", "WARN", "reports/", "no for_analysis/ bundle exists at all",
              "an analysis that produced nothing to paste has not finished")
    for b in bundles:
        rel = b.parent.relative_to(NB_DIR)
        if str(rel) not in results_md:
            f.add("bundle_registered", "ERROR", str(rel),
                  "a result bundle that RESULTS.md does not mention",
                  f"add a row to KNOWLEDGE/RESULTS.md for {rel}, with its annotation vintage")

        if not (b / "INDEX.md").exists():
            f.add("bundle_shape", "ERROR", str(b.relative_to(NB_DIR)),
                  "no INDEX.md, so the bundle cannot orient its own reader",
                  "write it with helpers/bundle.py rather than by hand")
        bad = sorted(p.name for p in b.iterdir() if p.suffix.lower() in BINARY_SUFFIXES)
        if bad:
            f.add("bundle_shape", "ERROR", str(b.relative_to(NB_DIR)),
                  f"binary files in a text bundle: {', '.join(bad)}",
                  "keep binaries in the parent reports/<name>/ directory, out of for_analysis/")
        if not any(p.suffix.lower() == ".csv" for p in b.iterdir()):
            f.add("bundle_shape", "WARN", str(b.relative_to(NB_DIR)),
                  "no CSV -- the numbers are not machine-readable",
                  "export the result table alongside the prose")
        prose = [p for p in b.glob("*.md") if p.name != "INDEX.md"]
        if not prose:
            f.add("bundle_shape", "WARN", str(b.relative_to(NB_DIR)),
                  "no report.md beside INDEX.md -- nothing says what was done",
                  "write the prose report into the bundle")


def check_jobs(f, analyses_md, history_md):
    """Every job is findable from ANALYSES.md (current) or HISTORY.md (retired/one-off)."""
    for p in sorted(NB_DIR.glob("jobs/*.py")):
        if p.name.startswith(JOB_SKIP_PREFIX):
            continue
        if p.name not in analyses_md and p.name not in history_md:
            f.add("job_registered", "ERROR", f"jobs/{p.name}",
                  "named in neither ANALYSES.md nor HISTORY.md",
                  "add a row to ANALYSES.md if it is current, or to HISTORY.md if it is retired")

    for p in sorted(NB_DIR.glob("jobs/run_*.py")):
        if not p.with_suffix(".sh").exists():
            f.add("job_wrapper", "WARN", f"jobs/{p.name}",
                  "no run_*.sh wrapper, so it has no documented route onto LSF",
                  "add the bsub wrapper, or say in ANALYSES.md that it runs inline")


def check_helpers(f):
    readme = NB_DIR / "helpers" / "README.md"
    if not readme.exists():
        f.add("helper_documented", "WARN", "helpers/README.md", "missing", "write it")
        return
    txt = readme.read_text()
    for p in sorted(NB_DIR.glob("helpers/*.py")):
        if p.name == "__init__.py":
            continue
        if p.stem not in txt:
            f.add("helper_documented", "WARN", f"helpers/{p.name}",
                  "not described in helpers/README.md",
                  "add a line; an undocumented helper gets reimplemented")


def check_path_refs(f, per_file):
    """Paths cited in KNOWLEDGE must exist. Catches a reorganisation that moved a file."""
    for name, txt in per_file.items():
        for raw in sorted(set(PATH_RE.findall(txt))):
            cite = raw.rstrip(".,;:)")
            if PLACEHOLDER.search(cite):
                continue
            # "file.py:12" and "file.md#heading" are handled by the line-ref check
            cite = cite.split(":")[0].split("#")[0]
            if (NB_DIR / cite).exists():
                continue
            # prose names a module as `helpers/ccc_helpers`; that is a legitimate reference
            if (NB_DIR / (cite + ".py")).exists():
                continue
            sev = "WARN" if name in HISTORICAL else "ERROR"
            f.add("path_ref", sev, f"KNOWLEDGE/{name}",
                  f"cites `{cite}`, which does not exist",
                  "update the citation, or say in the text that the file is gone"
                  if sev == "WARN" else
                  "update the citation, or restore the file if it was moved by mistake")


def check_line_refs(f, per_file):
    """`file.py:123` must still have a line 123. A refactor silently invalidates these."""
    for name, txt in per_file.items():
        for cite, num in sorted(set(LINE_RE.findall(txt))):
            target = NB_DIR / cite
            if not target.exists() or target.is_dir():
                continue            # the path check already owns this case
            try:
                n_lines = sum(1 for _ in target.open(errors="replace"))
            except OSError:
                continue
            if int(num) > n_lines:
                f.add("line_ref", "ERROR", f"KNOWLEDGE/{name}",
                      f"cites `{cite}:{num}` but that file has {n_lines} lines",
                      "re-locate the reference; the code moved under it")


def check_results_annotation(f, results_md):
    """Every result row records which lineage call it used. A number without one is unciteable."""
    for line in results_md.splitlines():
        if not line.startswith("|") or line.startswith("|--") or "| Where |" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        if not cells[2]:
            f.add("results_annotation", "WARN", cells[0][:60],
                  "RESULTS.md row with an empty annotation column",
                  "state v1, v2, or why neither applies")


def check_memory(f, kb_all):
    """Project-type memories must point at KNOWLEDGE/, not duplicate it.

    Skipped where the store is absent, which is the normal case on a fresh checkout -- and is
    itself the argument for the rule: a fact that lives only here does not survive a clone.
    """
    if not MEMORY_DIR.exists():
        log(f"memory store not present at {MEMORY_DIR}; skipping memory_pointer")
        return
    for p in sorted(MEMORY_DIR.glob("*.md")):
        if p.name == "MEMORY.md":
            continue
        txt = p.read_text()
        if "type: project" not in txt:
            continue
        if p.stem in MEMORY_EXEMPT:
            continue
        if "KNOWLEDGE/" not in txt:
            f.add("memory_pointer", "WARN", f"memory/{p.name}",
                  "a project-type memory that never points into KNOWLEDGE/",
                  "move the fact into KNOWLEDGE/ and leave a one-line pointer here")
    if "KNOWLEDGE/README.md" not in (MEMORY_DIR / "MEMORY.md").read_text() and \
            "KNOWLEDGE/README.md" not in kb_all:
        f.add("memory_pointer", "WARN", "memory/MEMORY.md",
              "the memory index does not route to KNOWLEDGE/README.md",
              "add the pointer so a cold session starts in the right place")


# ======================================================================================
# output
# ======================================================================================
def write_outputs(f, strict):
    import csv as _csv

    import bundle

    OUT.mkdir(parents=True, exist_ok=True)
    rows = sorted(f.rows, key=lambda r: (r["severity"] != "ERROR", r["check"], r["subject"]))
    csv_path = OUT / "knowledge_findings.csv"
    with csv_path.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["check", "severity", "subject", "detail",
                                            "suggested_fix"])
        w.writeheader()
        w.writerows(rows)

    by_check = {}
    for r in rows:
        by_check.setdefault((r["check"], r["severity"]), 0)
        by_check[(r["check"], r["severity"])] += 1
    n_err = len(f.errors())

    md = [f"# Is the knowledge base still true? ({time.strftime('%Y-%m-%d %H:%M')})",
          "",
          f"`jobs/run_knowledge_check.py` compared `KNOWLEDGE/` against the repository.",
          "",
          f"**{n_err} ERROR, {len(rows) - n_err} WARN.**",
          ""]
    if not rows:
        md += ["Nothing to report: every bundle is registered, every job is findable, and every "
               "path and line reference in `KNOWLEDGE/` still resolves.", ""]
    else:
        md += ["| check | severity | n |", "|---|---|---|"]
        for (chk, sev), n in sorted(by_check.items(), key=lambda kv: (kv[0][1] != "ERROR", kv[0])):
            md.append(f"| {chk} | {sev} | {n} |")
        md += ["", "## Findings", "",
               "| severity | check | subject | detail | suggested fix |", "|---|---|---|---|---|"]
        for r in rows:
            md.append(f"| {r['severity']} | {r['check']} | `{r['subject']}` | {r['detail']} | "
                      f"{r['suggested_fix']} |")
        md.append("")
    md += ["## What this does not check", "",
           "Only that the map matches the territory. It cannot tell you whether a number in "
           "`RESULTS.md` is *correct*, whether a caveat is still true, or whether an analysis "
           "was any good. A registered bundle with a wrong headline passes every check here.",
           ""]
    report = OUT / "knowledge_check_report.md"
    report.write_text("\n".join(md))

    bundle.write_bundle(
        OUT / "for_analysis",
        files=[csv_path, report],
        title="Knowledge-base consistency check",
        question="Does KNOWLEDGE/ still describe this repository -- is every result registered, "
                 "every job findable, and every path and line reference still valid?",
        headline=[f"{n_err} ERROR and {len(rows) - n_err} WARN findings",
                  "ERROR = would mislead a reader; WARN = untidy only",
                  "bookkeeping only: it never checks whether a number is right"],
        traps=["A clean run does not mean the knowledge base is correct, only that it is "
               "internally consistent with the file tree.",
               "memory_pointer is skipped when Claude's memory store is absent, which is the "
               "normal case on a fresh checkout."],
        columns={"knowledge_findings.csv":
                 "one row per finding: check, severity (ERROR/WARN), the file or directory it "
                 "is about, what is wrong, and what to do"},
        provenance={"script": "jobs/run_knowledge_check.py",
                    "mode": "strict" if strict else "report"},
        annotation="n/a -- reads no cell data")
    log(f"{n_err} ERROR, {len(rows) - n_err} WARN -> {OUT/'for_analysis'}")
    return n_err


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any ERROR was found (for a pre-commit hook or CI)")
    args = ap.parse_args()

    kb_all, per_file = _kb_text()
    results_md = per_file.get("RESULTS.md", "")
    analyses_md = per_file.get("ANALYSES.md", "")
    history_md = per_file.get("HISTORY.md", "")
    for need in ("RESULTS.md", "ANALYSES.md", "HISTORY.md"):
        if need not in per_file:
            print(f"FATAL: KNOWLEDGE/{need} is missing", file=sys.stderr)
            return 2

    f = Findings()
    check_bundles(f, results_md)
    check_jobs(f, analyses_md, history_md)
    check_helpers(f)
    check_path_refs(f, per_file)
    check_line_refs(f, per_file)
    check_results_annotation(f, results_md)
    check_memory(f, kb_all)

    n_err = write_outputs(f, args.strict)
    for r in sorted(f.rows, key=lambda r: r["severity"] != "ERROR")[:40]:
        print(f"  {r['severity']:5s} {r['check']:20s} {r['subject']}: {r['detail']}")
    if len(f.rows) > 40:
        print(f"  ... {len(f.rows) - 40} more in {OUT/'for_analysis'/'knowledge_findings.csv'}")
    return 1 if (args.strict and n_err) else 0


if __name__ == "__main__":
    sys.exit(main())
