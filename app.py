"""NGS-Genius Streamlit dashboard.  Run with:  streamlit run app.py"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from ngs_genius import genai, literature, plots, report, troubleshoot
from ngs_genius.de import r_available
from ngs_genius.pipeline import Params, run_pipeline
from ngs_genius.validate import has_errors

ROOT = Path(__file__).parent
st.set_page_config(page_title="NGS-Genius", page_icon="🧬", layout="wide")


# ------------------------------------------------------------------------------------ helpers
def read_table(file, index_col=0) -> pd.DataFrame:
    return pd.read_csv(file, sep=None, engine="python", index_col=index_col)


@st.cache_data(show_spinner=False)
def load_demo() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.read_csv(ROOT / "data" / "demo_counts.csv", index_col=0),
            pd.read_csv(ROOT / "data" / "demo_metadata.csv", index_col=0))


@st.cache_data(show_spinner=False)
def r_ok() -> bool:
    return r_available()


def secret(name: str) -> str:
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


def client_from_sidebar() -> genai.GeminiClient | None:
    key = st.session_state.get("gemini_key") or os.environ.get("GEMINI_API_KEY") or secret("GEMINI_API_KEY")
    return genai.GeminiClient(key, st.session_state.get("gemini_model") or None) if key else None


def ai_box(text: str, source: str) -> None:
    st.markdown(text)
    st.caption(f"Source: {source}. AI-generated text is hypothesis-generating — review before use.")


# ------------------------------------------------------------------------------------ sidebar
st.sidebar.title("🧬 NGS-Genius")
st.sidebar.caption("RNA-seq analysis with GenAI-assisted interpretation and troubleshooting")

src = st.sidebar.radio("Data", ["Demo dataset (synthetic)", "Upload my data"])
counts = meta = None
if src.startswith("Demo"):
    counts, meta = load_demo()
    st.sidebar.info("Synthetic p53-activation-like data (4 vs 4). Not real measurements.")
else:
    cf = st.sidebar.file_uploader("Raw count matrix (genes × samples, CSV/TSV)", type=["csv", "tsv", "txt"])
    mf = st.sidebar.file_uploader("Sample metadata (CSV/TSV, first column = sample id)", type=["csv", "tsv", "txt"])
    if cf and mf:
        try:
            counts, meta = read_table(cf), read_table(mf)
        except Exception as e:
            st.sidebar.error(f"Could not read files: {e}")

params = Params()
if meta is not None:
    st.sidebar.subheader("Design")
    cols = list(meta.columns)
    params.condition = st.sidebar.selectbox("Condition column", cols, index=cols.index("condition") if "condition" in cols else 0)
    levels = sorted(meta[params.condition].astype(str).unique())
    params.reference = st.sidebar.selectbox("Reference level", levels, index=levels.index("control") if "control" in levels else 0)
    others = [l for l in levels if l != params.reference] or levels
    params.treatment = st.sidebar.selectbox("Treatment level", others)
    params.covariates = st.sidebar.multiselect("Covariates (e.g. batch)", [c for c in cols if c != params.condition],
                                               default=["batch"] if "batch" in cols and src.startswith("Demo") else [])
    st.sidebar.subheader("Thresholds")
    params.min_total_count = st.sidebar.number_input("Min total count per gene", 0, 1000, 10)
    params.padj_cutoff = st.sidebar.select_slider("Adjusted p-value", [0.001, 0.01, 0.05, 0.1], value=0.05)
    params.lfc_cutoff = st.sidebar.slider("|log2 fold change| ≥", 0.0, 3.0, 1.0, 0.1)
    avail = r_ok()
    eng = st.sidebar.selectbox("DE engine", ["Auto", "R DESeq2", "PyDESeq2 (Python)"],
                               help="Auto uses R DESeq2 when available, otherwise PyDESeq2.")
    params.engine = {"Auto": "auto", "R DESeq2": "r", "PyDESeq2 (Python)": "pydeseq2"}[eng]
    st.sidebar.caption(f"R + DESeq2 detected: {'✅' if avail else '❌ (PyDESeq2 fallback will be used)'}")
    params.context = st.sidebar.text_input("Biological context (optional)",
                                           "p53 activation in cancer cells" if src.startswith("Demo") else "",
                                           help="Used to focus AI explanations and PubMed searches.")

st.sidebar.subheader("Gemini (optional)")
st.sidebar.text_input("API key", type="password", key="gemini_key",
                      help="Or set GEMINI_API_KEY. Without a key the app shows rule-based summaries.")
st.sidebar.text_input("Model", value=genai.DEFAULT_MODEL, key="gemini_model")
run = st.sidebar.button("▶ Run analysis", type="primary", disabled=meta is None, width="stretch")

if run:
    with st.spinner("Validating, running QC and differential expression…"):
        st.session_state["result"] = run_pipeline(counts, meta, params)
    for k in ("qc_ai", "de_ai", "gene_notes"):
        st.session_state.pop(k, None)

res = st.session_state.get("result")

# ------------------------------------------------------------------------------------ main
st.title("NGS-Genius")
tabs = st.tabs(["Overview", "Quality control", "Differential expression", "Heatmap",
                "AI insights", "Gene literature", "Troubleshooter", "Report"])

if res is None:
    with tabs[0]:
        st.markdown("""
**GenAI-enabled RNA-seq analysis and intelligent troubleshooting.**

1. Pick the **demo dataset** or upload a raw count matrix and metadata.
2. Choose the comparison and thresholds in the sidebar, then press **Run analysis**.
3. Explore QC, PCA, volcano/MA plots and heatmaps; ask Gemini to explain results; look up gene function
   in PubMed; or paste a pipeline error into the **Troubleshooter**.
        """)
    with tabs[6]:
        st.info("The troubleshooter works without running an analysis — paste any error or log below.")
        txt = st.text_area("Paste an error message or log", height=200, key="ts_idle")
        if st.button("Diagnose", key="ts_btn_idle") and txt.strip():
            text, source = genai.explain_error(client_from_sidebar(), txt)
            ai_box(text, source)
    st.stop()

# ---- Overview
with tabs[0]:
    for i in res.issues:
        {"error": st.error, "warning": st.warning, "info": st.info}[i.level](f"**{i.code}** — {i.message}")
    if res.counts is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Samples", len(res.meta))
        c2.metric("Genes kept", f"{res.n_genes_kept:,}", f"of {res.n_genes_input:,}", delta_color="off")
        if res.de is not None:
            up, down = int((res.de.regulation == "Up").sum()), int((res.de.regulation == "Down").sum())
            c3.metric("Significant genes", up + down)
            c4.metric("Up / Down", f"{up} / {down}")
            st.caption(f"Engine: {res.engine_used} · {res.params.treatment} vs {res.params.reference} · "
                       f"padj < {res.params.padj_cutoff}, |log2FC| ≥ {res.params.lfc_cutoff}")
        st.dataframe(res.meta, width="stretch")
    if res.de_error:
        st.error("The DE engine failed. Open the **Troubleshooter** tab — the log is pre-filled.")
        st.code(res.de_error[-2000:])

if res.counts is None:
    with tabs[6]:
        txt = st.text_area("Paste an error message or log", height=200, key="ts_fail")
        if st.button("Diagnose", key="ts_btn_fail") and txt.strip():
            ai_box(*genai.explain_error(client_from_sidebar(), txt))
    st.stop()

# ---- QC
with tabs[1]:
    cond = res.params.condition
    flagged = res.qc[res.qc["flags"] != ""]
    if len(flagged):
        st.warning("Flagged samples: " + ", ".join(f"**{s}** ({r['flags']})" for s, r in flagged.iterrows()))
    a, b = st.columns(2)
    a.plotly_chart(plots.library_size_bar(res.qc, res.meta, cond), width="stretch")
    b.plotly_chart(plots.detected_genes_bar(res.qc, res.meta, cond), width="stretch")
    pc_opts = list(res.pca_scores.columns)
    x_pc, y_pc = st.columns(2)
    xs = x_pc.selectbox("X axis", pc_opts, 0)
    ys = y_pc.selectbox("Y axis", pc_opts, 1 if len(pc_opts) > 1 else 0)
    color_by = st.selectbox("Colour by", list(res.meta.columns), list(res.meta.columns).index(cond))
    shape_by = st.selectbox("Shape by", [None, *res.meta.columns], 0)
    a, b = st.columns([2, 1])
    a.plotly_chart(plots.pca_plot(res.pca_scores, res.pca_var, res.meta, color_by, shape_by, xs, ys), width="stretch")
    b.plotly_chart(plots.scree_plot(res.pca_var), width="stretch")
    st.plotly_chart(plots.correlation_heatmap(__import__("ngs_genius.qc", fromlist=["x"]).sample_correlation(res.log_norm)),
                    width="stretch")
    st.dataframe(res.qc.round(3), width="stretch")

# ---- DE
with tabs[2]:
    if res.de is None:
        st.error("Differential expression did not run; see the Overview/Troubleshooter tabs.")
    else:
        a, b = st.columns(2)
        a.plotly_chart(plots.volcano_plot(res.de, res.params.padj_cutoff, res.params.lfc_cutoff), width="stretch")
        b.plotly_chart(plots.ma_plot(res.de), width="stretch")
        show = st.radio("Show", ["Significant", "All"], horizontal=True)
        tbl = res.de if show == "All" else res.de[res.de.regulation != "NS"]
        q = st.text_input("Filter by gene symbol (substring)")
        if q:
            tbl = tbl[tbl.gene.str.contains(q, case=False, na=False)]
        st.dataframe(tbl.drop(columns=["neglog10_padj"]), width="stretch", height=380)
        st.download_button("Download results (CSV)", res.de.to_csv(index=False), "deseq2_results.csv", "text/csv")

# ---- Heatmap
with tabs[3]:
    if res.de is None:
        st.info("Run a successful DE analysis first.")
    else:
        n = st.slider("Number of genes", 10, 100, 40, 5)
        fig = plots.heatmap(res.log_norm, res.de, res.meta, res.params.condition, n)
        st.plotly_chart(fig, width="stretch") if fig else st.info("Too few significant genes for a heatmap.")

# ---- AI insights
with tabs[4]:
    client = client_from_sidebar()
    st.caption("Gemini connected ✅" if client else "No Gemini key — showing deterministic rule-based summaries.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("QC report explained")
        if st.button("Explain QC", key="qc_btn"):
            with st.spinner("Thinking…"):
                st.session_state["qc_ai"] = genai.explain_qc(client, res.qc, res.issues, res.meta, res.params.condition)
        if "qc_ai" in st.session_state:
            ai_box(*st.session_state["qc_ai"])
    with c2:
        st.subheader("DE results explained")
        if st.button("Explain DE results", key="de_btn", disabled=res.de is None):
            with st.spinner("Thinking…"):
                st.session_state["de_ai"] = genai.explain_de(client, res.de, res.params)
        if "de_ai" in st.session_state:
            ai_box(*st.session_state["de_ai"])

# ---- Literature
with tabs[5]:
    if res.de is None:
        st.info("Run a successful DE analysis first.")
    else:
        sig = res.de[res.de.regulation != "NS"]
        gene = st.selectbox("Gene (significant genes first)", list(sig.gene) + [g for g in res.de.gene if g not in set(sig.gene)][:200])
        ctx = st.text_input("Search context", res.params.context, key="lit_ctx")
        st.caption("Searches PubMed live. Gene-symbol lookups are only meaningful for real gene symbols "
                   "(synthetic filler genes like GENE00042 will return nothing).")
        if st.button("Retrieve literature & interpret"):
            try:
                with st.spinner("Searching PubMed…"):
                    papers = literature.retrieve_for_gene(gene, ctx, top_k=5)
                row = res.de[res.de.gene == gene].iloc[0]
                with st.spinner("Interpreting…"):
                    text, source = genai.interpret_gene(client_from_sidebar(), gene, row, ctx, papers)
                notes = st.session_state.setdefault("gene_notes", {})
                notes[gene] = (text, papers)
                ai_box(text, source)
                for p in papers:
                    with st.expander(f"{p.title} — {p.cite()}"):
                        st.write(p.abstract or "_No abstract available._")
                        st.markdown(f"[Open in PubMed]({p.url})")
            except literature.LiteratureError as e:
                st.error(f"Literature retrieval failed: {e}")
        if st.session_state.get("gene_notes"):
            st.caption("Genes added to the report: " + ", ".join(st.session_state["gene_notes"]))

# ---- Troubleshooter
with tabs[6]:
    default = res.de_error or ""
    txt = st.text_area("Paste an error message or log (STAR, HISAT2, featureCounts, DESeq2, R, Python…)",
                       value=default, height=220, key="ts_main")
    ctx = st.text_input("Optional context (tool versions, command, organism)", key="ts_ctx")
    if st.button("Diagnose", key="ts_btn_main") and txt.strip():
        with st.spinner("Diagnosing…"):
            ai_box(*genai.explain_error(client_from_sidebar(), txt, ctx))
    with st.expander("What the built-in rule library covers"):
        st.dataframe(pd.DataFrame([{"tool": r.tool, "issue": r.id, "cause": r.cause} for r in troubleshoot.RULES]),
                     width="stretch", hide_index=True)

# ---- Report
with tabs[7]:
    qc_text = st.session_state.get("qc_ai", ("", ""))[0]
    de_text = st.session_state.get("de_ai", ("", ""))[0]
    md = report.build_report(res, qc_text, de_text, st.session_state.get("gene_notes"))
    st.caption("Includes any AI explanations and gene notes you've generated in this session.")
    st.download_button("Download report (Markdown)", md, "ngs_genius_report.md", "text/markdown")
    st.markdown(md)
