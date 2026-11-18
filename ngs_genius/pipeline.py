"""End-to-end pipeline: validate -> filter -> QC -> PCA -> differential expression."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import de, qc
from .validate import Issue, has_errors, validate_inputs


@dataclass
class Params:
    condition: str = "condition"
    reference: str = "control"
    treatment: str = "treated"
    covariates: list[str] = field(default_factory=list)
    min_total_count: int = 10
    padj_cutoff: float = 0.05
    lfc_cutoff: float = 1.0
    engine: str = "auto"  # auto | r | pydeseq2
    n_top_var: int = 500
    context: str = ""  # free-text biological context, e.g. "p53 activation in cancer cells"


@dataclass
class AnalysisResult:
    params: Params
    issues: list[Issue]
    counts: pd.DataFrame | None = None
    meta: pd.DataFrame | None = None
    n_genes_input: int = 0
    n_genes_kept: int = 0
    size_factors: pd.Series | None = None
    log_norm: pd.DataFrame | None = None
    qc: pd.DataFrame | None = None
    pca_scores: pd.DataFrame | None = None
    pca_var: np.ndarray | None = None
    de: pd.DataFrame | None = None
    engine_used: str = ""
    de_error: str = ""  # raw error text if the DE engine failed (fed to the troubleshooter)


def run_pipeline(counts: pd.DataFrame, meta: pd.DataFrame, params: Params) -> AnalysisResult:
    p = params
    counts, meta, issues = validate_inputs(counts, meta, p.condition, p.reference, p.treatment, p.covariates)
    result = AnalysisResult(params=p, issues=issues, n_genes_input=counts.shape[0])
    if has_errors(issues):
        return result

    keep = counts.sum(axis=1) >= p.min_total_count
    counts = counts.loc[keep]
    result.counts, result.meta, result.n_genes_kept = counts, meta, int(keep.sum())
    if result.n_genes_kept < 100:
        result.issues.append(Issue("warning", "FEW_GENES",
                                   f"Only {result.n_genes_kept} genes pass the minimum-count filter."))
    if result.n_genes_kept == 0:
        result.issues.append(Issue("error", "NO_GENES", "No genes pass the minimum-count filter."))
        return result

    sf = qc.size_factors(counts)
    result.size_factors = sf
    result.log_norm = qc.log_normalise(counts, sf)
    result.qc = qc.qc_table(counts, result.log_norm, sf)
    result.pca_scores, result.pca_var = qc.run_pca(result.log_norm, p.n_top_var)

    try:
        out = de.run_de(counts, meta, p.condition, p.reference, p.treatment, p.covariates,
                        p.min_total_count, p.engine)
        result.de = de.annotate_results(out.results, p.padj_cutoff, p.lfc_cutoff)
        result.engine_used = out.engine
    except de.DEError as e:
        result.de_error = e.log
        result.issues.append(Issue("error", "DE_FAILED", str(e)))
    return result
