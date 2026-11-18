"""Gemini-powered explanations for QC reports, DE results, pipeline errors and gene function.

Design principles
-----------------
* **Grounded**: every prompt contains the computed numbers/evidence; the model is told not to invent
  values, genes or citations.
* **Cited**: gene interpretations may cite only PMIDs present in the retrieved evidence.
* **Graceful offline mode**: with no API key (or on API failure) each function returns a deterministic
  rule-based summary instead, so the app is fully usable without Gemini.
"""
from __future__ import annotations

import os
import re

import pandas as pd

from . import troubleshoot
from .literature import Paper, format_context

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM = (
    "You are a careful bioinformatics assistant embedded in an RNA-seq analysis tool. "
    "Use ONLY the facts and numbers provided in the prompt; never invent values, gene functions, or citations. "
    "If the evidence is insufficient, say so. Distinguish statistical association from biological causation, "
    "note sample-size limitations, and keep the answer concise, structured and in plain language for a "
    "biologist. Use Markdown."
)


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model = model or DEFAULT_MODEL
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        if not self.available:
            raise RuntimeError("No Gemini API key configured.")
        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        resp = self._client.models.generate_content(
            model=self.model, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM, temperature=temperature))
        return (resp.text or "").strip()


def _try(client: GeminiClient | None, prompt: str, fallback: str) -> tuple[str, str]:
    """Return (text, source) where source is 'gemini' or 'offline' (with reason)."""
    if client is None or not client.available:
        return fallback, "offline (no Gemini API key)"
    try:
        return client.generate(prompt), f"gemini ({client.model})"
    except Exception as e:  # network, quota, bad key, model name...
        return fallback + f"\n\n> Gemini call failed ({type(e).__name__}: {str(e)[:200]}); showing the rule-based summary.", \
            "offline (Gemini error)"


# --------------------------------------------------------------------------------------- QC
def _qc_facts(qc: pd.DataFrame, issues, meta: pd.DataFrame, condition: str) -> str:
    t = qc.copy()
    t[condition] = meta.loc[t.index, condition].astype(str).values
    cols = [condition, "total_reads", "detected_genes", "pct_zero_genes", "top_gene", "top_gene_pct",
            "size_factor", "median_spearman", "flags"]
    iss = "\n".join(str(i) for i in issues) or "none"
    return f"Per-sample QC table:\n{t[cols].round(3).to_string()}\n\nInput validation issues:\n{iss}"


def explain_qc(client: GeminiClient | None, qc: pd.DataFrame, issues, meta: pd.DataFrame, condition: str):
    flagged = qc[qc["flags"] != ""]
    lines = [f"- **{s}**: {r['flags'].replace(',', ', ')} (reads {int(r['total_reads']):,}, "
             f"genes {int(r['detected_genes']):,}, median ρ {r['median_spearman']:.3f})" for s, r in flagged.iterrows()]
    fallback = (f"**Rule-based QC summary** — {len(qc)} samples, median library size "
                f"{int(qc.total_reads.median()):,} reads.\n\n"
                + ("Flagged samples:\n" + "\n".join(lines) if lines else "No samples were flagged.")
                + "\n\nFlags: `low_depth` (<50% of median reads), `low_complexity` (<85% of median detected genes), "
                  "`dominant_gene` (one gene >10% of reads), `possible_outlier` (low correlation with other samples).")
    prompt = (f"{_qc_facts(qc, issues, meta, condition)}\n\n"
              "Task: Explain this RNA-seq QC report to a biologist. Cover (1) overall data quality, (2) each flagged "
              "sample and what likely causes it, (3) whether to exclude, keep, or re-sequence it and how that would "
              "affect differential expression, (4) concrete next steps. Do not exceed ~250 words.")
    return _try(client, prompt, fallback)


# ---------------------------------------------------------------------------- DE interpretation
def _de_facts(res: pd.DataFrame, params, top_n: int) -> str:
    up, down = int((res.regulation == "Up").sum()), int((res.regulation == "Down").sum())
    top = res[res.regulation != "NS"].head(top_n)[["gene", "log2FoldChange", "padj", "baseMean", "regulation"]]
    return (f"Comparison: {params.treatment} vs {params.reference} (column '{params.condition}'); "
            f"covariates: {params.covariates or 'none'}.\n"
            f"Thresholds: padj < {params.padj_cutoff}, |log2FC| >= {params.lfc_cutoff}.\n"
            f"Genes tested (non-NA padj): {int(res.padj.notna().sum())}; significant: {up + down} "
            f"({up} up, {down} down).\n"
            f"Biological context given by the user: {params.context or 'not provided'}.\n\n"
            f"Top significant genes (by padj):\n{top.round(4).to_string(index=False)}")


def explain_de(client: GeminiClient | None, res: pd.DataFrame, params, top_n: int = 25):
    up, down = int((res.regulation == "Up").sum()), int((res.regulation == "Down").sum())
    top = res[res.regulation != "NS"].head(10)
    fallback = (f"**Rule-based DE summary** — {params.treatment} vs {params.reference}: {up + down} significant genes "
                f"({up} up, {down} down) at padj < {params.padj_cutoff} and |log2FC| ≥ {params.lfc_cutoff}.\n\n"
                + ("Top genes: " + ", ".join(f"{r.gene} ({r.log2FoldChange:+.2f})" for r in top.itertuples())
                   if len(top) else "No genes passed the thresholds.")
                + "\n\nPositive log2FC = higher in " + params.treatment + ".")
    prompt = (f"{_de_facts(res, params, top_n)}\n\n"
              "Task: Summarise these differential-expression results for a biologist. Cover (1) the overall size and "
              "direction of the response, (2) recurring themes you can justify from the listed genes (state clearly when "
              "a theme is an inference from gene names rather than a tested enrichment), (3) caveats (sample size, "
              "thresholds, batch, low-depth samples), (4) 3 specific follow-up analyses (e.g. GSEA, validation by qPCR). "
              "Do not claim pathway enrichment significance — none was computed. ~300 words.")
    return _try(client, prompt, fallback)


# -------------------------------------------------------------------------------- gene function
def interpret_gene(client: GeminiClient | None, gene: str, row: pd.Series | None, context: str,
                   papers: list[Paper]):
    stats = ("not in the DE table" if row is None else
             f"log2FC={row['log2FoldChange']:+.2f}, padj={row['padj']:.2e}, baseMean={row['baseMean']:.0f}, "
             f"call={row['regulation']}")
    fallback = (f"**Evidence retrieved for {gene}** (offline summary; {stats}).\n\n"
                + ("\n".join(f"- {p.title} — {p.cite()}" for p in papers) if papers
                   else "No PubMed papers were retrieved."))
    if not papers:
        return fallback, "offline (no papers retrieved)"
    prompt = (f"Gene: {gene}\nObserved in this experiment: {stats}\nExperimental context: {context or 'not provided'}\n\n"
              f"Retrieved PubMed evidence (the ONLY sources you may cite):\n\n{format_context(papers)}\n\n"
              "Task: Explain what is known about this gene's function relevant to the context and what its observed "
              "direction of change could mean. Rules: every factual claim about the gene must be followed by a citation "
              "like [PMID:12345678] using only the PMIDs above; if the abstracts do not support a claim, do not make it; "
              "end with a one-line confidence statement and note that this is hypothesis-generating. ~200 words.")
    text, src = _try(client, prompt, fallback)
    # guard: strip any citation whose PMID was not retrieved
    allowed = {p.pmid for p in papers}
    def _fix(m: re.Match) -> str:
        return m.group(0) if m.group(1) in allowed else "[unverified citation removed]"
    return re.sub(r"\[PMID:\s*(\d+)\]", _fix, text), src


# ------------------------------------------------------------------------------ troubleshooting
def explain_error(client: GeminiClient | None, error_text: str, context: str = ""):
    diags = troubleshoot.diagnose(error_text)
    fallback = "**Built-in diagnosis**\n\n" + troubleshoot.format_diagnoses(diags)
    matched = "\n".join(f"- {d.rule.tool}/{d.rule.id}: {d.rule.cause}" for d in diags) or "none"
    prompt = (f"A user hit this error/log while running an RNA-seq pipeline:\n\n```\n{error_text[-4000:]}\n```\n\n"
              f"Pipeline context: {context or 'not provided'}\n"
              f"Matches from a rule-based library (may be incomplete or wrong):\n{matched}\n\n"
              "Task: (1) state the most likely root cause in one or two sentences, (2) give a numbered, minimal fix "
              "with exact commands/code, (3) list how to verify the fix, (4) mention one alternative cause if the log is "
              "ambiguous. If the text is not an error or lacks information, say what extra information you need.")
    return _try(client, prompt, fallback)
