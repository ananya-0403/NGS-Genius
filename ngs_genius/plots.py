"""Interactive Plotly figures: library sizes, PCA, correlation, volcano, MA, heatmap."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import leaves_list, linkage

# Colour-blind-safe (Okabe-Ito) palette
UP, DOWN, NS = "#D55E00", "#0072B2", "#B8B8B8"
REG_COLORS = {"Up": UP, "Down": DOWN, "NS": NS}
GROUP_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

_LAYOUT = dict(template="plotly_white", font=dict(size=13), margin=dict(l=50, r=20, t=50, b=50))


def library_size_bar(qc: pd.DataFrame, meta: pd.DataFrame, condition: str) -> go.Figure:
    d = qc.reset_index().rename(columns={"index": "sample"})
    d.columns = ["sample", *d.columns[1:]]
    d[condition] = meta.loc[d["sample"], condition].astype(str).values
    fig = px.bar(d, x="sample", y="total_reads", color=condition, color_discrete_sequence=GROUP_COLORS,
                 title="Library size (total assigned reads)")
    fig.add_hline(y=float(d["total_reads"].median()) * 0.5, line_dash="dot", line_color="#888",
                  annotation_text="50% of median", annotation_position="top left")
    fig.update_layout(**_LAYOUT, yaxis_title="Total reads")
    return fig


def detected_genes_bar(qc: pd.DataFrame, meta: pd.DataFrame, condition: str) -> go.Figure:
    d = qc.reset_index().rename(columns={qc.index.name or "index": "sample"})
    d[condition] = meta.loc[d["sample"], condition].astype(str).values
    fig = px.bar(d, x="sample", y="detected_genes", color=condition, color_discrete_sequence=GROUP_COLORS,
                 title="Genes detected (count > 0)")
    fig.update_layout(**_LAYOUT, yaxis_title="Genes")
    return fig


def pca_plot(scores: pd.DataFrame, var: np.ndarray, meta: pd.DataFrame, color: str,
             symbol: str | None = None, x: str = "PC1", y: str = "PC2") -> go.Figure:
    d = scores.copy()
    d["sample"] = d.index
    d[color] = meta.loc[d.index, color].astype(str).values
    if symbol and symbol != color:
        d[symbol] = meta.loc[d.index, symbol].astype(str).values
    xi, yi = int(x[2:]) - 1, int(y[2:]) - 1
    fig = px.scatter(d, x=x, y=y, color=color, symbol=symbol if symbol and symbol != color else None,
                     text="sample", color_discrete_sequence=GROUP_COLORS, title="PCA of top variable genes")
    fig.update_traces(marker=dict(size=13, line=dict(width=1, color="white")), textposition="top center")
    fig.update_layout(**_LAYOUT, xaxis_title=f"{x} ({var[xi] * 100:.1f}%)", yaxis_title=f"{y} ({var[yi] * 100:.1f}%)")
    return fig


def scree_plot(var: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Bar(x=[f"PC{i + 1}" for i in range(len(var))], y=var * 100, marker_color="#0072B2"))
    fig.update_layout(**_LAYOUT, title="Variance explained", yaxis_title="% variance")
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale="Viridis",
                               colorbar=dict(title="Spearman"), zmin=float(corr.values.min()), zmax=1.0))
    fig.update_layout(**_LAYOUT, title="Sample-to-sample correlation", yaxis_autorange="reversed")
    return fig


def volcano_plot(res: pd.DataFrame, padj_cutoff: float, lfc_cutoff: float, n_label: int = 12) -> go.Figure:
    d = res.dropna(subset=["padj", "log2FoldChange"]).copy()
    fig = go.Figure()
    for reg in ("NS", "Down", "Up"):
        s = d[d["regulation"] == reg]
        fig.add_trace(go.Scattergl(
            x=s["log2FoldChange"], y=s["neglog10_padj"], mode="markers", name=f"{reg} ({len(s)})",
            marker=dict(color=REG_COLORS[reg], size=5 if reg == "NS" else 7, opacity=0.5 if reg == "NS" else 0.85),
            text=s["gene"], customdata=np.c_[s["padj"], s["baseMean"]],
            hovertemplate="<b>%{text}</b><br>log2FC %{x:.2f}<br>padj %{customdata[0]:.2e}"
                          "<br>baseMean %{customdata[1]:.0f}<extra></extra>"))
    top = d[d["regulation"] != "NS"].nsmallest(n_label, "padj")
    fig.add_trace(go.Scatter(x=top["log2FoldChange"], y=top["neglog10_padj"], mode="text", text=top["gene"],
                             textposition="top center", showlegend=False, textfont=dict(size=11)))
    fig.add_hline(y=-np.log10(padj_cutoff), line_dash="dot", line_color="#888")
    for x in (-lfc_cutoff, lfc_cutoff):
        fig.add_vline(x=x, line_dash="dot", line_color="#888")
    fig.update_layout(**_LAYOUT, title="Volcano plot", xaxis_title="log2 fold change",
                      yaxis_title="-log10 adjusted p-value")
    return fig


def ma_plot(res: pd.DataFrame) -> go.Figure:
    d = res.dropna(subset=["padj", "log2FoldChange"]).copy()
    d["log10_baseMean"] = np.log10(d["baseMean"] + 1)
    fig = px.scatter(d, x="log10_baseMean", y="log2FoldChange", color="regulation",
                     color_discrete_map=REG_COLORS, hover_name="gene", opacity=0.6,
                     title="MA plot", render_mode="webgl")
    fig.add_hline(y=0, line_color="#444")
    fig.update_layout(**_LAYOUT, xaxis_title="log10 mean normalised count", yaxis_title="log2 fold change")
    return fig


def heatmap(log_norm: pd.DataFrame, res: pd.DataFrame, meta: pd.DataFrame, condition: str,
            n_genes: int = 40) -> go.Figure | None:
    """Row-z-scored heatmap of the top DE genes (by adjusted p), rows clustered."""
    sig = res[res["regulation"] != "NS"].dropna(subset=["padj"]).nsmallest(n_genes, "padj")
    genes = [g for g in sig["gene"] if g in log_norm.index]
    if len(genes) < 2:
        return None
    mat = log_norm.loc[genes]
    z = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1).replace(0, 1), axis=0)
    order = leaves_list(linkage(z.values, method="average", metric="euclidean"))
    z = z.iloc[order]
    cols = sorted(z.columns, key=lambda s: (str(meta.loc[s, condition]), s))
    z = z[cols]
    labels = [f"{s} ({meta.loc[s, condition]})" for s in cols]
    lim = float(np.nanmax(np.abs(z.values))) or 1.0
    fig = go.Figure(go.Heatmap(z=z.values, x=labels, y=z.index, colorscale="RdBu_r", zmid=0, zmin=-lim, zmax=lim,
                               colorbar=dict(title="row z-score")))
    fig.update_layout(**_LAYOUT, title=f"Top {len(genes)} differentially expressed genes",
                      height=max(400, 18 * len(genes) + 120), yaxis_autorange="reversed")
    return fig
