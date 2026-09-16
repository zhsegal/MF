# Claude Code task: download scTCR/VDJ data (non-Haniffa atlas papers)

Goal: retrieve all available T-cell receptor (VDJ) data for the five non-Haniffa datasets.
Four have TCR; one does not. Download both the **processed clonotype files** (fast path, from GEO
supplementary) and the **raw VDJ-library FASTQs** (fallback, from SRA/ENA). Report per dataset what
was found vs not.

## Targets

| key | GEO | TCR? | suppl FTP block | notes |
|---|---|---|---|---|
| chennareddy2025 | GSE266862 | yes | GSE266nnn | 5′ + scTCR (TCRβ + TCRγ); incl. γδ-MF & CD8 Berti |
| brunner2024 | GSE269981 | yes | GSE269nnn | 5′ + scTCR; only the **8 eCTCL** samples are CTCL |
| rindler2021_fi | GSE165623 | yes | GSE165nnn | 5′ + V(D)J (TCRαβ); skin/blood/LN; n=1 |
| herrera2021 | GSE171811 | yes | GSE171nnn | ECCITE-seq: TCRαβ **+ TCRγδ** + 49-ADT + HTO |
| gaydosik2019 | GSE128531 | **no** | GSE128nnn | 3′ only — **skip TCR**, no VDJ recoverable |

## Environment

```bash
conda create -n tcrdl -c bioconda -c conda-forge -y sra-tools pysradb ffq wget curl
conda activate tcrdl
mkdir -p geo_suppl meta ena fastq tcr_out
```

## Step 1 — GEO supplementary sweep (fast path: processed contigs/clonotypes)

For each TCR target (skip gaydosik2019), mirror the GEO supplementary dir and look for VDJ files
(`*filtered_contig_annotations*`, `*clonotypes*`, `*vdj*`, `*tcr*`):

```bash
for GSE in GSE266862 GSE269981 GSE165623 GSE171811; do
  BLOCK=$(echo "$GSE" | sed -E 's/[0-9]{3}$/nnn/')
  wget -q -r -np -nd -R "index.html*" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/${BLOCK}/${GSE}/suppl/" \
    -P "geo_suppl/${GSE}/"
  echo "== ${GSE} VDJ-looking files =="
  ls -la "geo_suppl/${GSE}/" | grep -iE 'contig|clonotype|vdj|tcr|airr' || echo "  none in suppl"
done
```

If `*_filtered_contig_annotations.csv.gz` / `*_clonotypes.csv.gz` are present, those are the ready-to-use
clonotype calls — no re-processing needed. Untar any `*_RAW.tar` and re-check.

## Step 2 — SRA/ENA FASTQ for VDJ libraries (fallback when Step 1 has no contigs)

Map GSE → study accession, pull run metadata, and identify the **VDJ/TCR** libraries (filter on
`library_name`/`experiment_title` containing VDJ / TCR / V(D)J / IR / immune):

```bash
for GSE in GSE266862 GSE269981 GSE165623 GSE171811; do
  SRP=$(pysradb gse-to-srp "$GSE" | awk 'NR==2{print $NF}')
  echo "${GSE} -> ${SRP}"
  pysradb metadata "$SRP" --detailed --saveto "meta/${GSE}.tsv"
  curl -s "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=${SRP}&result=read_run&fields=run_accession,experiment_title,library_name,library_strategy,fastq_ftp,fastq_md5&format=tsv" \
    -o "ena/${GSE}.tsv"
  echo "  VDJ runs:"
  grep -iE 'vdj|tcr|v\(d\)j|immune|[^a-z]ir[_ ]' "ena/${GSE}.tsv" | cut -f1-4 || echo "  (none flagged — inspect ena/${GSE}.tsv manually)"
done
```

Download the FASTQs for the flagged VDJ runs (ENA `fastq_ftp` is `;`-separated, scheme-less):

```bash
# example: pull VDJ-flagged runs for one GSE
GSE=GSE266862
grep -iE 'vdj|tcr|v\(d\)j|immune|[^a-z]ir[_ ]' "ena/${GSE}.tsv" | \
  awk -F'\t' '{print $5}' | tr ';' '\n' | sed 's#^#https://#' | \
  while read u; do wget -q -P "fastq/${GSE}/" "$u"; done
```

If `library_name`/`experiment_title` does not distinguish GEX vs VDJ (common for Vienna/Brunner
deposits), fall back to `library_strategy` and download all runs, then let `cellranger vdj` (Step 3)
sort VDJ from GEX — only the VDJ libraries will yield contigs.

## Step 3 — cellranger vdj (only if you downloaded raw FASTQ)

Reference: `refdata-cellranger-vdj-GRCh38-alts-ensembl-7.1.0` (10x VDJ human reference).

```bash
cellranger vdj \
  --id="${GSE}_${SAMPLE}" \
  --reference=/path/to/refdata-cellranger-vdj-GRCh38-alts-ensembl-7.1.0 \
  --fastqs=fastq/${GSE}/${SAMPLE} \
  --sample=${SAMPLE} \
  --chain=auto
# output: outs/filtered_contig_annotations.csv  -> copy to tcr_out/${GSE}/${SAMPLE}/
```

γδ note: **chennareddy2025** (γδ-MF arm) and **herrera2021** (TCRγδ modality) contain γδ chains.
`--chain=auto` handles standard αβ/γδ from a normal V(D)J library; if a sample used ECCITE-seq
γδ inner primers (Mimitou 2019), the γδ contigs may need that protocol's primer handling — only
relevant at this step, not for download.

## Per-dataset handling

- **chennareddy2025 (GSE266862):** download all 18 CTCL + 4 HC. Only the CD4⁺ MF subset is directly
  comparable; keep γδ-MF and CD8 Berti tagged separately.
- **brunner2024 (GSE269981):** CTCL = the **8 eCTCL** samples only (mixed with CIE/AD/Pso/HC in the
  series). The eCTCL and HC samples are likely **reused from other Vienna deposits**
  (GSE173205 / GSE222840 / GSE247047 / GSE266862) — record GSM IDs and de-duplicate before merging.
- **rindler2021_fi (GSE165623):** 3 libraries (skin, blood, LN) from one patient; expect small.
- **herrera2021 (GSE171811):** richest — also grab the ADT (49-marker) and HTO libraries alongside VDJ.
- **gaydosik2019 (GSE128531):** no VDJ. Do not attempt TRUST4 on the 3′ data (negligible yield).
  Cells stay TCR-null.

## Output layout

```
geo_suppl/<GSE>/        # mirrored GEO supplementary (processed contigs if present)
meta/<GSE>.tsv          # pysradb run metadata
ena/<GSE>.tsv           # ENA fastq ftp + library fields
fastq/<GSE>/<sample>/   # raw VDJ FASTQs
tcr_out/<GSE>/<sample>/ # filtered_contig_annotations.csv (final clonotypes)
```

## Report back (per GSE)

1. Processed contigs found in GEO suppl? (filenames)
2. Study accession (SRP/PRJNA) + count of VDJ runs identified.
3. FASTQs downloaded (run accessions) and/or contig files obtained.
4. Anything missing or access-blocked.
