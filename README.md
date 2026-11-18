# 🧬 NGS-Genius

**GenAI-enabled RNA-seq analysis and intelligent troubleshooting platform**

Python · R · DESeq2 · Linux · Streamlit · Gemini API

NGS-Genius takes a raw count matrix and sample metadata through validation, quality assessment,
differential expression (DESeq2) and interactive visualisation, then uses the Gemini API and
PubMed retrieval to explain QC reports, pipeline errors, DE results and gene function.

> ⚠️ The bundled demo dataset is **synthetic** (simulated negative-binomial counts with planted
> effects). It exists to demonstrate and test the platform; it must not be used for biological conclusions.

---

## Features

| Capability | Where |
|---|---|
| Input validation (sample-name mismatch, non-integer/negative counts, duplicate genes, replicates, confounded/rank-deficient designs) | `ngs_genius/validate.py` |
| Quality assessment: median-of-ratios normalisation, per-sample QC metrics + automatic flags, PCA, sample correlation | `ngs_genius/qc.py` |
| Differential expression with **R / DESeq2** (`Rscript`), automatic **PyDESeq2** fallback | `r/run_deseq2.R`, `ngs_genius/de.py` |
| Interactive plots: library size, PCA, correlation, volcano, MA, clustered heatmap | `ngs_genius/plots.py` |
| **Gemini explanations** of QC reports and DE results (grounded in computed numbers) | `ngs_genius/genai.py` |
| **Intelligent troubleshooting**: 16-rule pattern library (DESeq2, STAR, HISAT2, featureCounts, memory, FastQC) + Gemini for unmatched logs | `ngs_genius/troubleshoot.py` |
| **Literature-based retrieval (RAG)**: PubMed search → rank → Gemini interpretation with citations restricted to retrieved PMIDs | `ngs_genius/literature.py` |
| Streamlit dashboard with Markdown report export | `app.py`, `ngs_genius/report.py` |
| CLI for headless runs | `ngs_genius/cli.py` |

The app is fully usable **without** a Gemini key: every AI feature falls back to a deterministic
rule-based summary and says so in the UI.

## Architecture

```
counts.csv + metadata.csv
        │
   validate.py ──► issues (errors/warnings) ─────────────┐
        │                                                │
   filter → qc.py (size factors, QC table, PCA)          │
        │                                                ▼
   de.py ── R DESeq2 (Rscript)  ──or──  PyDESeq2     troubleshoot.py ◄── pasted logs / DE errors
        │                                                │
   plots.py ◄── results                                  │
        │                                                ▼
   genai.py (Gemini) ◄── facts from QC/DE ── literature.py (PubMed RAG)
        │
   app.py (Streamlit)  /  cli.py  →  report.md
```

## Quick start

```bash
cd NGS-Genius
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# optional: Gemini
export GEMINI_API_KEY=your_key           # get one at https://aistudio.google.com/apikey

# optional: real R DESeq2 engine (otherwise PyDESeq2 is used)
Rscript r/install_deps.R

streamlit run app.py
```

Then choose **Demo dataset** → **Run analysis**.

### Headless

```bash
python -m ngs_genius.cli --demo --covariates batch --out results/            # add --ai for Gemini text
python -m ngs_genius.cli --counts my_counts.csv --meta my_meta.csv \
       --condition group --reference WT --treatment KO --out results/
```

### Your own data

* **Counts**: CSV/TSV, genes in rows, samples in columns, first column = gene ID, **raw integer counts**
  (featureCounts, HTSeq, STAR `--quantMode GeneCounts`).
* **Metadata**: CSV/TSV, first column = sample ID matching the count-matrix columns; one column for
  the condition, optional covariates (batch, sex, …).
* Literature lookup needs **real gene symbols** (HGNC for human). Ensembl IDs won't match PubMed
  text; map them to symbols first.

### Regenerate demo data

```bash
python scripts/generate_demo_data.py
```

### Tests

```bash
pytest -q        # 28 offline tests: no R, no network, no API key required
```

## How the GenAI parts stay trustworthy

* **Grounding** – prompts contain the computed QC tables / DE statistics; the system prompt forbids
  invented numbers, genes or citations.
* **Citation guard** – gene interpretations may cite only PMIDs that were actually retrieved; any other
  `[PMID:…]` in the model output is stripped and replaced with *[unverified citation removed]* (unit-tested).
* **No fake enrichment** – the DE explanation prompt tells the model that no pathway enrichment was
  computed, so themes are labelled as inferences from gene names.
* **Offline mode** – deterministic fallbacks when no key is set or the API call fails.

## Project layout

```
NGS-Genius/
├── app.py                    Streamlit dashboard
├── ngs_genius/               validate, qc, de, plots, pipeline, genai, troubleshoot, literature, report, cli
├── r/run_deseq2.R            DESeq2 script (+ install_deps.R)
├── scripts/generate_demo_data.py
├── data/                     synthetic demo counts + metadata
└── tests/test_ngs_genius.py
```

## Status and known limitations

* Tested here: validation, QC, PyDESeq2 engine, plots, troubleshooting rules, offline GenAI paths,
  PubMed XML parsing/ranking, the Streamlit app (headless `AppTest`), and the CLI.
* **Not exercised in the build environment**: the R DESeq2 script (R was not installed there),
  live Gemini calls (no key), and live PubMed requests (no network allowlist). Run them once on your
  machine; the code paths are isolated so failures surface as readable messages, not crashes.
* Gemini model names change over time; set `GEMINI_MODEL` if the default is retired.
* No gene-set enrichment (GSEA/ORA) yet.

## Roadmap ideas

GSEA/ORA with gseapy · FastQC/MultiQC report ingestion · Ensembl→symbol mapping · batch correction
(ComBat-seq) · Dockerfile with R + Bioconductor · Nextflow (nf-core/rnaseq) upstream integration.

## Résumé description

> **NGS-Genius – GenAI-Enabled RNA-seq Analysis and Intelligent Troubleshooting Platform**
> *Personal Project | Python, R, DESeq2, Linux, Streamlit, Gemini API*
> * Developed an RNA-seq analysis platform integrating DESeq2 differential expression with interactive visualisation.
> * Implemented QC, PCA, volcano-plot and heatmap workflows for gene expression data.
> * Integrated the Gemini API to explain QC reports, pipeline errors and differential-expression results.
> * Built a Streamlit dashboard to explore results and export analysis summaries.
> * Added PubMed literature retrieval (RAG) with citation guarding for evidence-based gene-function interpretation.
