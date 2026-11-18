"""Differential expression.

Primary engine : R / DESeq2 (``r/run_deseq2.R`` via ``Rscript``).
Fallback engine: PyDESeq2, a Python re-implementation of the DESeq2 workflow, used
automatically when R or the DESeq2 package is not installed so the app still runs.
Both engines return the same table schema.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

R_SCRIPT = Path(__file__).resolve().parents[1] / "r" / "run_deseq2.R"
RESULT_COLS = ["gene", "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]


class DEError(RuntimeError):
    """Raised when a DE engine fails. ``log`` holds the raw stderr/traceback text."""

    def __init__(self, message: str, log: str = ""):
        super().__init__(message)
        self.log = log or message


@dataclass
class DEOutput:
    results: pd.DataFrame
    engine: str
    log: str = ""
    normalized_counts: pd.DataFrame | None = None


def r_available() -> bool:
    """True if Rscript exists and the DESeq2 package can be loaded."""
    if shutil.which("Rscript") is None:
        return False
    try:
        p = subprocess.run(
            ["Rscript", "-e", "quit(status = !requireNamespace('DESeq2', quietly = TRUE))"],
            capture_output=True, timeout=60)
        return p.returncode == 0
    except Exception:
        return False


def run_deseq2_r(counts: pd.DataFrame, meta: pd.DataFrame, condition: str, reference: str,
                 treatment: str, covariates: list[str] | None = None, min_total: int = 10) -> DEOutput:
    covariates = covariates or []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        counts.to_csv(tmp / "counts.csv")
        meta.to_csv(tmp / "meta.csv")
        cmd = ["Rscript", str(R_SCRIPT), str(tmp / "counts.csv"), str(tmp / "meta.csv"), str(tmp / "out"),
               condition, reference, treatment, ",".join(covariates), str(min_total)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        log = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            raise DEError("R DESeq2 failed", log)
        res = pd.read_csv(tmp / "out" / "results.csv")
        norm = pd.read_csv(tmp / "out" / "normalized_counts.csv", index_col=0)
    return DEOutput(res, "R DESeq2", log, norm)


def run_pydeseq2(counts: pd.DataFrame, meta: pd.DataFrame, condition: str, reference: str,
                 treatment: str, covariates: list[str] | None = None, min_total: int = 10) -> DEOutput:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    covariates = covariates or []
    try:
        keep = counts.sum(axis=1) >= min_total
        c = counts.loc[keep].T  # samples x genes
        m = meta[[*covariates, condition]].copy()
        for col in m.columns:
            if not pd.api.types.is_numeric_dtype(m[col]):
                m[col] = m[col].astype(str)
        design = "~ " + " + ".join([*covariates, condition])
        # The reference level is set through the explicit contrast below
        # (PyDESeq2 >= 0.5 deprecated ref_level).
        dds = DeseqDataSet(counts=c, metadata=m, design=design, quiet=True)
        dds.deseq2()
        stats = DeseqStats(dds, contrast=[condition, treatment, reference], quiet=True)
        stats.summary()
        res = stats.results_df.copy()
    except Exception as e:  # surface as DEError so the troubleshooter can diagnose it
        raise DEError(f"PyDESeq2 failed: {e}", f"{type(e).__name__}: {e}") from e

    res = res.reset_index().rename(columns={res.index.name or "index": "gene"})
    res = res[[col for col in RESULT_COLS if col in res.columns]]
    norm = pd.DataFrame(dds.layers["normed_counts"], index=c.index, columns=c.columns).T
    return DEOutput(res, "PyDESeq2", "", norm)


def run_de(counts: pd.DataFrame, meta: pd.DataFrame, condition: str, reference: str, treatment: str,
           covariates: list[str] | None = None, min_total: int = 10, engine: str = "auto") -> DEOutput:
    """engine: 'auto' | 'r' | 'pydeseq2'."""
    if engine == "r" or (engine == "auto" and r_available()):
        return run_deseq2_r(counts, meta, condition, reference, treatment, covariates, min_total)
    return run_pydeseq2(counts, meta, condition, reference, treatment, covariates, min_total)


def annotate_results(res: pd.DataFrame, padj_cutoff: float = 0.05, lfc_cutoff: float = 1.0) -> pd.DataFrame:
    """Add a 'regulation' column (Up / Down / NS) and -log10 adjusted p."""
    res = res.copy()
    padj = res["padj"].astype(float)
    lfc = res["log2FoldChange"].astype(float)
    sig = (padj < padj_cutoff) & (lfc.abs() >= lfc_cutoff)
    res["regulation"] = np.where(sig & (lfc > 0), "Up", np.where(sig & (lfc < 0), "Down", "NS"))
    res["neglog10_padj"] = -np.log10(padj.clip(lower=1e-300))
    return res.sort_values("padj", na_position="last").reset_index(drop=True)
