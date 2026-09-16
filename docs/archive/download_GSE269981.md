# Downloading GSE269981

scRNA-seq + scTCR-seq, blood + skin. Brunner/Vienna group (chronic idiopathic erythroderma study; the CTCL content is 8 erythrodermic-CTCL cases among 68 samples). **Processed data only** — raw FASTQs are withheld for patient privacy, so there is no SRA path.

## Setup

```bash
mkdir -p gse269981 && cd gse269981
```

## Download the processed data (the only version available)

One ~2.0 GB tarball of per-sample CSV / MTX / TSV files.

```bash
# Option 1 — GEO download endpoint
wget -nc -O GSE269981_RAW.tar "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE269981&format=file"

# Option 2 — NCBI FTP (same file, if Option 1 is slow/blocked)
# wget -nc "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE269nnn/GSE269981/suppl/GSE269981_RAW.tar"

# Unpack
tar -xvf GSE269981_RAW.tar && rm GSE269981_RAW.tar

# Some files may still be gzipped:
gunzip -f *.gz 2>/dev/null || true
```

After unpacking you'll have files prefixed by GSM ID, e.g. `GSM8331789_P105_Skin_*` (matrix.mtx / barcodes.tsv / features.tsv for scRNA) and CSV contig files for scTCR.

## Get the sample → patient / tissue / disease map

The GSM titles encode patient, tissue (Skin/Blood), assay (scRNA-seq/scTCR-seq) and group (HC/AD; CIE and eCTCL are the unlabeled patient IDs).

```bash
wget -nc "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE269nnn/GSE269981/matrix/GSE269981_series_matrix.txt.gz"
gunzip -k GSE269981_series_matrix.txt.gz
grep -E "Sample_title|Sample_geo_accession|characteristics" GSE269981_series_matrix.txt
```
