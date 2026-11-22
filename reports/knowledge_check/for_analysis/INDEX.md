# Knowledge-base consistency check

Does KNOWLEDGE/ still describe this repository -- is every result registered, every job findable, and every path and line reference still valid?

Generated 2026-11-22 18:14. Lineage annotation: **n/a -- reads no cell data**.

## What it found

- 0 ERROR and 5 WARN findings
- ERROR = would mislead a reader; WARN = untidy only
- bookkeeping only: it never checks whether a number is right

## Files

| file | rows | what it holds |
|---|---|---|
| `knowledge_check_report.md` | - |  |
| `knowledge_findings.csv` | 5 | one row per finding: check, severity (ERROR/WARN), the file or directory it is about, what is wrong, and what to do |

## How to read the columns

- **`knowledge_findings.csv`** — one row per finding: check, severity (ERROR/WARN), the file or directory it is about, what is wrong, and what to do

## Things that will mislead you if you do not know them

- A clean run does not mean the knowledge base is correct, only that it is internally consistent with the file tree.
- memory_pointer is skipped when Claude's memory store is absent, which is the normal case on a fresh checkout.

## Provenance

- **script**: jobs/run_knowledge_check.py
- **mode**: report

