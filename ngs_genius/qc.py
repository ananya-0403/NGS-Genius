"""Quality assessment: normalisation, per-sample QC metrics, PCA, correlation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def size_factors(counts: pd.DataFrame) -> pd.Series:
    """Median-of-ratios size factors (Anders & Huber 2010, as used by DESeq2).

    Falls back to a 'poscounts' geometric mean when fewer than 100 genes are
    non-zero in every sample (sparse data).
    """
    c = counts.to_numpy(dtype=float)
    with np.errstate(divide="ignore"):
        logc = np.log(c)
    logc[~np.isfinite(logc)] = np.nan
    with np.errstate(all="ignore"):
        log_geo = np.nanmean(logc, axis=1)
    all_pos = (c > 0).all(axis=1)
    use = all_pos if all_pos.sum() >= 100 else np.isfinite(log_geo)
    ratios = logc[use] - log_geo[use, None]
    sf = np.exp(np.nanmedian(ratios, axis=0))
    return pd.Series(sf, index=counts.columns, name="size_factor")


def log_normalise(counts: pd.DataFrame, sf: pd.Series | None = None) -> pd.DataFrame:
    """log2(normalised counts + 1)."""
    sf = size_factors(counts) if sf is None else sf
    return np.log2(counts.div(sf, axis=1) + 1.0)


def qc_table(counts: pd.DataFrame, log_norm: pd.DataFrame, sf: pd.Series) -> pd.DataFrame:
    """Per-sample QC metrics and automatic flags."""
    total = counts.sum(axis=0)
    detected = (counts > 0).sum(axis=0)
    detected10 = (counts >= 10).sum(axis=0)
    top_gene = counts.idxmax(axis=0)
    top_pct = counts.max(axis=0) / total * 100

    corr = log_norm.corr(method="spearman")
    n = corr.shape[0]
    med_corr = (corr.sum(axis=0) - 1.0) / max(n - 1, 1)

    df = pd.DataFrame({
        "total_reads": total,
        "detected_genes": detected,
        "genes_ge_10_reads": detected10,
        "pct_zero_genes": (1 - detected / counts.shape[0]) * 100,
        "top_gene": top_gene,
        "top_gene_pct": top_pct,
        "size_factor": sf,
        "median_spearman": med_corr,
    })

    flags: list[str] = []
    med_total = df["total_reads"].median()
    med_det = df["detected_genes"].median()
    med_corr_all = df["median_spearman"].median()
    for s, row in df.iterrows():
        f = []
        if row.total_reads < 0.5 * med_total:
            f.append("low_depth")
        if row.detected_genes < 0.85 * med_det:
            f.append("low_complexity")
        if row.top_gene_pct > 10:
            f.append("dominant_gene")
        if row.median_spearman < med_corr_all - 0.05:
            f.append("possible_outlier")
        flags.append(",".join(f))
    df["flags"] = flags
    return df


def run_pca(log_norm: pd.DataFrame, n_top: int = 500) -> tuple[pd.DataFrame, np.ndarray]:
    """PCA on the top-variance genes. Returns (scores, explained_variance_ratio)."""
    var = log_norm.var(axis=1)
    top = var.sort_values(ascending=False).head(min(n_top, len(var))).index
    x = log_norm.loc[top].T
    x = x - x.mean(axis=0)
    n_comp = min(5, x.shape[0], x.shape[1])
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(x)
    cols = [f"PC{i + 1}" for i in range(n_comp)]
    return pd.DataFrame(scores, index=log_norm.columns, columns=cols), pca.explained_variance_ratio_


def sample_correlation(log_norm: pd.DataFrame) -> pd.DataFrame:
    return log_norm.corr(method="spearman")
