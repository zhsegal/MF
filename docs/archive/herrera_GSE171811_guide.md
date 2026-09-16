# Downloading GSE171811 (Herrera et al. 2021, *Blood* — ECCITE-seq CTCL)

Goal: download the dataset. Nothing else.

> Accession `GSE171811` is verified. The NCBI FTP path is deterministic. Exact supplementary filenames are **not** assumed — list the directory first and download what's actually there.

## Setup

```bash
mkdir -p gse171811 && cd gse171811
```

## Download (processed data — recommended)

```bash
BASE="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE171nnn/GSE171811"

# 1. See what's in the record FIRST
curl -s "$BASE/suppl/" | grep -oE 'href="[^"]+"' | sed 's/href="//;s/"//'

# 2a. If a bundle exists:
wget -nc "$BASE/suppl/GSE171811_RAW.tar"
tar -xvf GSE171811_RAW.tar && rm GSE171811_RAW.tar

# 2b. Fallback — grab everything in suppl/ regardless of filenames:
wget -r -np -nd -nc -e robots=off \
  -A 'tar,gz,csv,mtx,tsv,h5,h5ad,rds' \
  "$BASE/suppl/"
```

Programmatic alternative (also pulls sample metadata):

```bash
pip install GEOparse
```
```python
import GEOparse
gse = GEOparse.get_GEO(geo="GSE171811", destdir="./")
for n, g in gse.gsms.items():
    g.download_supplementary_files(directory="./")
```

## Download (raw FASTQs — only if you need reads)

```bash
mamba install -y -c bioconda sra-tools pysradb

pysradb gse-to-srp GSE171811        # -> SRP study accession
pysradb srp-to-srr <SRPxxxxxxx>     # -> SRR run accessions (read them off; don't guess)

prefetch SRR_XXXXXXX
fasterq-dump --split-files SRR_XXXXXXX -O ./
```
