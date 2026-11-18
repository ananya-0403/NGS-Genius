"""Markdown analysis report generator."""
from __future__ import annotations

from datetime import date

from .literature import Paper
from .pipeline import AnalysisResult


def build_report(result: AnalysisResult, qc_text: str = "", de_text: str = "",
                 gene_notes: dict[str, tuple[str, list[Paper]]] | None = None, title: str = "NGS-Genius report") -> str:
    p, de, qc = result.params, result.de, result.qc
    L: list[str] = [f"# {title}", f"_Generated {date.today().isoformat()} by NGS-Genius_", ""]

    L += ["## 1. Analysis setup", "",
          f"- **Comparison:** {p.treatment} vs {p.reference} (metadata column `{p.condition}`)",
          f"- **Covariates in design:** {', '.join(p.covariates) or 'none'}",
          f"- **Samples:** {0 if result.meta is None else len(result.meta)}",
          f"- **Genes:** {result.n_genes_input:,} in input, {result.n_genes_kept:,} after filtering "
          f"(total count ≥ {p.min_total_count})",
          f"- **DE engine:** {result.engine_used or 'n/a'}",
          f"- **Thresholds:** padj < {p.padj_cutoff}, |log2FC| ≥ {p.lfc_cutoff}",
          f"- **Context:** {p.context or 'not provided'}", ""]

    if result.issues:
        L += ["### Input validation notes", ""] + [f"- {i}" for i in result.issues] + [""]

    if qc is not None:
        L += ["## 2. Quality assessment", "",
              qc[["total_reads", "detected_genes", "median_spearman", "flags"]].round(3).to_markdown(), ""]
        if qc_text:
            L += ["**Interpretation**", "", qc_text, ""]

    if de is not None:
        up, down = int((de.regulation == "Up").sum()), int((de.regulation == "Down").sum())
        L += ["## 3. Differential expression", "",
              f"**{up + down} significant genes** — {up} up, {down} down in {p.treatment}.", ""]
        top = de[de.regulation != "NS"].head(20)[["gene", "baseMean", "log2FoldChange", "padj", "regulation"]]
        if len(top):
            L += ["Top 20 by adjusted p-value:", "", top.round(4).to_markdown(index=False), ""]
        if de_text:
            L += ["**Interpretation**", "", de_text, ""]

    if gene_notes:
        L += ["## 4. Gene-level literature notes", ""]
        for gene, (text, papers) in gene_notes.items():
            L += [f"### {gene}", "", text, ""]
            if papers:
                L += ["References:", ""] + [f"- {x.title}. {x.cite()} — {x.url}" for x in papers] + [""]

    L += ["## Methods (auto-generated)", "",
          "Raw counts were filtered, normalised by median-of-ratios size factors, and log2-transformed for PCA "
          "and visualisation. Differential expression used DESeq2's negative-binomial model "
          f"({result.engine_used or 'engine not run'}) with Wald tests and Benjamini–Hochberg adjustment. "
          "AI-generated text is hypothesis-generating and must be reviewed by the analyst.", ""]
    return "\n".join(L)
