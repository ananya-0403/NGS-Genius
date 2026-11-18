"""Input validation for count matrices and sample metadata.

Catches the mistakes that most often make DESeq2 fail (mismatched sample names,
non-integer counts, confounded designs, too few replicates) *before* the model
is fitted, and returns structured issues that the UI and the AI troubleshooter
can both consume.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Issue:
    level: str  # "error" | "warning" | "info"
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level.upper()}] {self.code}: {self.message}"


def has_errors(issues: list[Issue]) -> bool:
    return any(i.level == "error" for i in issues)


def validate_inputs(
    counts: pd.DataFrame,
    meta: pd.DataFrame,
    condition: str,
    reference: str,
    treatment: str,
    covariates: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Issue]]:
    """Validate and align inputs.

    Returns (counts, meta, issues). Counts are returned as a genes x samples
    integer DataFrame with columns ordered like ``meta`` when alignment is possible.
    """
    covariates = covariates or []
    issues: list[Issue] = []

    # --- sample name agreement -------------------------------------------------
    c_samples, m_samples = set(counts.columns), set(meta.index)
    only_counts = sorted(c_samples - m_samples)
    only_meta = sorted(m_samples - c_samples)
    if only_counts or only_meta:
        if not (c_samples & m_samples):
            issues.append(Issue(
                "error", "SAMPLE_MISMATCH",
                "No sample names are shared between the count matrix columns and the "
                f"metadata index. Counts: {sorted(c_samples)[:5]}..., metadata: {sorted(m_samples)[:5]}..."))
            return counts, meta, issues
        if only_counts:
            issues.append(Issue("warning", "SAMPLE_IN_COUNTS_ONLY",
                                f"Samples without metadata were dropped: {only_counts}"))
        if only_meta:
            issues.append(Issue("warning", "SAMPLE_IN_META_ONLY",
                                f"Metadata rows without counts were dropped: {only_meta}"))
    shared = [s for s in meta.index if s in c_samples]
    if list(counts.columns) != shared:
        if set(counts.columns) == set(shared) and list(counts.columns) != shared:
            issues.append(Issue("info", "REORDERED",
                                "Count columns were reordered to match the metadata order."))
    counts = counts[shared]
    meta = meta.loc[shared]

    # --- gene identifiers ------------------------------------------------------
    if counts.index.has_duplicates:
        n = int(counts.index.duplicated().sum())
        issues.append(Issue("error", "DUPLICATE_GENES",
                            f"{n} duplicated gene identifiers. Collapse or de-duplicate them "
                            "(e.g. sum counts per gene) before running DESeq2."))

    # --- numeric / integer checks ---------------------------------------------
    non_numeric = [c for c in counts.columns if not pd.api.types.is_numeric_dtype(counts[c])]
    if non_numeric:
        issues.append(Issue("error", "NON_NUMERIC_COUNTS",
                            f"Non-numeric columns in the count matrix: {non_numeric}. "
                            "Is there an annotation column (gene name, length) mixed in?"))
        return counts, meta, issues
    if counts.isna().any().any():
        issues.append(Issue("error", "NA_COUNTS", "Count matrix contains missing values (NA/NaN)."))
        return counts, meta, issues
    if (counts < 0).any().any():
        issues.append(Issue("error", "NEGATIVE_COUNTS",
                            "Count matrix contains negative values; DESeq2 needs raw non-negative counts."))
        return counts, meta, issues
    vals = counts.to_numpy(dtype=float)
    if not np.allclose(vals, np.round(vals)):
        issues.append(Issue("warning", "NON_INTEGER_COUNTS",
                            "Counts are not integers (TPM/FPKM or RSEM/tximport estimates?). "
                            "DESeq2 expects raw counts; values were rounded. Prefer raw counts, or "
                            "use tximport + DESeqDataSetFromTximport."))
    counts = counts.round().astype(int)

    # --- metadata checks -------------------------------------------------------
    if condition not in meta.columns:
        issues.append(Issue("error", "MISSING_CONDITION",
                            f"Condition column '{condition}' not found. Available: {list(meta.columns)}"))
        return counts, meta, issues
    for col in [condition, *covariates]:
        if col not in meta.columns:
            issues.append(Issue("error", "MISSING_COVARIATE", f"Column '{col}' not found in metadata."))
            return counts, meta, issues
        if meta[col].isna().any():
            issues.append(Issue("error", "NA_METADATA", f"Column '{col}' has missing values."))
            return counts, meta, issues

    levels = set(meta[condition].astype(str))
    for lv, role in [(reference, "reference"), (treatment, "treatment")]:
        if lv not in levels:
            issues.append(Issue("error", "MISSING_LEVEL",
                                f"The {role} level '{lv}' is not present in '{condition}'. Levels: {sorted(levels)}"))
    if reference == treatment:
        issues.append(Issue("error", "SAME_LEVEL", "Reference and treatment levels must differ."))
    if has_errors(issues):
        return counts, meta, issues

    # --- replicates ------------------------------------------------------------
    for lv in (reference, treatment):
        n = int((meta[condition].astype(str) == lv).sum())
        if n < 2:
            issues.append(Issue("error", "NO_REPLICATES",
                                f"Group '{lv}' has only {n} sample(s). DESeq2 cannot estimate dispersion "
                                "without biological replicates."))
        elif n < 3:
            issues.append(Issue("warning", "FEW_REPLICATES",
                                f"Group '{lv}' has {n} replicates. Power will be low; 3+ is recommended."))

    # --- design rank / confounding --------------------------------------------
    sub = meta[meta[condition].astype(str).isin([reference, treatment])]
    design_df = pd.get_dummies(sub[[*covariates, condition]].astype(
        {c: str for c in [*covariates, condition] if not pd.api.types.is_numeric_dtype(sub[c])}),
        drop_first=True).astype(float)
    X = np.column_stack([np.ones(len(design_df)), design_df.to_numpy()])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        issues.append(Issue("error", "NOT_FULL_RANK",
                            "The design matrix is not full rank: a covariate is perfectly confounded "
                            "with the condition (e.g. every treated sample is in one batch). "
                            "Remove the covariate or add samples that break the confounding."))
    elif X.shape[0] - X.shape[1] < 2:
        issues.append(Issue("warning", "LOW_DF",
                            "Very few residual degrees of freedom; results will be unstable."))

    return counts, meta, issues
