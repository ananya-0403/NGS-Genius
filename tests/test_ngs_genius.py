"""Offline test-suite (no R, no network, no Gemini key needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_demo_data import DOWN, UP, make_dataset  # noqa: E402
from ngs_genius import genai, literature, plots, qc, report, troubleshoot  # noqa: E402
from ngs_genius.pipeline import Params, run_pipeline  # noqa: E402
from ngs_genius.validate import has_errors, validate_inputs  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return make_dataset(n_genes=3000, seed=7)


@pytest.fixture(scope="module")
def result(data):
    c, m = data
    return run_pipeline(c, m, Params(covariates=["batch"], engine="pydeseq2", context="p53"))


# ------------------------------------------------------------------ validation
def test_valid_inputs_have_no_errors(data):
    c, m = data
    _, _, issues = validate_inputs(c, m, "condition", "control", "treated", ["batch"])
    assert not has_errors(issues)


def test_sample_mismatch_detected(data):
    c, m = data
    _, _, issues = validate_inputs(c.rename(columns=lambda s: s + "_x"), m, "condition", "control", "treated")
    assert any(i.code == "SAMPLE_MISMATCH" for i in issues)


def test_confounded_batch_detected(data):
    c, m = data
    m2 = m.copy()
    m2["batch"] = m2["condition"]  # perfectly confounded
    _, _, issues = validate_inputs(c, m2, "condition", "control", "treated", ["batch"])
    assert any(i.code == "NOT_FULL_RANK" for i in issues)


def test_non_integer_and_negative(data):
    c, m = data
    _, _, issues = validate_inputs(c + 0.5, m, "condition", "control", "treated")
    assert any(i.code == "NON_INTEGER_COUNTS" for i in issues)
    _, _, issues = validate_inputs(c - 1000, m, "condition", "control", "treated")
    assert any(i.code == "NEGATIVE_COUNTS" for i in issues)


def test_no_replicates(data):
    c, m = data
    keep = ["control_1", "treated_1", "treated_2"]
    _, _, issues = validate_inputs(c[keep], m.loc[keep], "condition", "control", "treated")
    assert any(i.code == "NO_REPLICATES" for i in issues)


# ------------------------------------------------------------------ QC
def test_size_factors_geometric_mean_one(data):
    sf = qc.size_factors(data[0])
    assert (sf > 0).all()
    assert abs(np.exp(np.log(sf).mean()) - 1) < 0.35  # roughly centred on 1


def test_planted_shallow_sample_flagged(result):
    assert "low_depth" in result.qc.loc["treated_4", "flags"]
    assert result.qc.loc["control_1", "flags"] == ""


def test_pca_separates_conditions(result):
    pc1 = result.pca_scores["PC1"]
    ctrl = pc1[[s for s in pc1.index if s.startswith("control")]]
    trt = pc1[[s for s in pc1.index if s.startswith("treated")]]
    assert ctrl.max() < trt.min() or trt.max() < ctrl.min()


# ------------------------------------------------------------------ differential expression
def test_de_recovers_planted_genes(result):
    de = result.de
    assert result.engine_used in ("PyDESeq2", "R DESeq2")
    called = de.set_index("gene")["regulation"]
    up_hit = sum(called.get(g) == "Up" for g in UP)
    down_hit = sum(called.get(g) == "Down" for g in DOWN)
    assert up_hit >= 0.8 * len(UP)
    assert down_hit >= 0.7 * len(DOWN)
    # no planted-up gene should be called Down and vice-versa
    assert not any(called.get(g) == "Down" for g in UP)
    assert not any(called.get(g) == "Up" for g in DOWN)


def test_false_positive_rate_reasonable(result):
    de = result.de[~result.de.gene.isin(UP + DOWN)]
    assert (de.padj < 0.05).mean() < 0.12  # includes ~4% genuinely perturbed background genes


def test_plots_build(result):
    assert plots.volcano_plot(result.de, 0.05, 1.0) is not None
    assert plots.ma_plot(result.de) is not None
    assert plots.heatmap(result.log_norm, result.de, result.meta, "condition", 30) is not None
    assert plots.pca_plot(result.pca_scores, result.pca_var, result.meta, "condition", "batch") is not None
    assert plots.library_size_bar(result.qc, result.meta, "condition") is not None


def test_pipeline_failure_is_captured(data):
    c, m = data
    bad = Params(condition="condition", reference="control", treatment="nonexistent")
    res = run_pipeline(c, m, bad)
    assert has_errors(res.issues) and res.de is None


# ------------------------------------------------------------------ troubleshooting
@pytest.mark.parametrize("msg,rule_id", [
    ("Error in estimateSizeFactorsForMatrix(counts(object)): every gene contains at least one zero, cannot compute log geometric means", "deseq_sizefactors"),
    ("Error: the model matrix is not full rank, so the model cannot be fit as specified.", "deseq_fullrank"),
    ("some values in assay are not integers", "deseq_nonint"),
    ("Error: rownames(colData) must equal colnames(countData)", "deseq_rowname_mismatch"),
    ("Error: cannot allocate vector of size 8.2 Gb", "memory"),
    ("EXITING because of FATAL ERROR: not enough memory for BAM sorting", "star_ram"),
    ("ERROR: No paired-end reads were detected in paired-end read library", "featurecounts_paired"),
    ("Failed to open the annotation file genes.gtf", "featurecounts_gtf"),
])
def test_troubleshooter_rules(msg, rule_id):
    assert rule_id in {d.rule.id for d in troubleshoot.diagnose(msg)}


def test_troubleshooter_no_match():
    assert troubleshoot.diagnose("all good here") == []


def test_explain_error_offline_fallback():
    text, src = genai.explain_error(None, "every gene contains at least one zero")
    assert "poscounts" in text and src.startswith("offline")


# ------------------------------------------------------------------ GenAI offline + report
def test_offline_summaries(result):
    t, s = genai.explain_qc(None, result.qc, result.issues, result.meta, "condition")
    assert "treated_4" in t and s.startswith("offline")
    t, s = genai.explain_de(None, result.de, result.params)
    assert "significant genes" in t


def test_report_builds(result):
    md = report.build_report(result, "QC text", "DE text")
    assert "Differential expression" in md and "Methods" in md


# ------------------------------------------------------------------ literature (offline parsing + guards)
PUBMED_XML = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>12345</PMID><Article>
<Journal><JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue><ISOAbbreviation>Nat Test</ISOAbbreviation></Journal>
<ArticleTitle>CDKN1A mediates p53-dependent arrest</ArticleTitle>
<Abstract><AbstractText Label="BACKGROUND">CDKN1A is a p53 target.</AbstractText><AbstractText>It halts the cell cycle.</AbstractText></Abstract>
<AuthorList><Author><LastName>Smith</LastName><Initials>J</Initials></Author></AuthorList>
</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""


def test_parse_pubmed_xml():
    (p,) = literature.parse_pubmed_xml(PUBMED_XML)
    assert p.pmid == "12345" and p.year == "2021" and "halts the cell cycle" in p.abstract
    assert "PMID 12345" in p.cite()


def test_rank_prefers_gene_in_title():
    a = literature.Paper("1", "CDKN1A in arrest", "p53 cell cycle", year="2020")
    b = literature.Paper("2", "Unrelated study", "mentions CDKN1A once", year="2020")
    ranked = literature.rank_papers([b, a], "CDKN1A", "p53")
    assert ranked[0].pmid == "1"


def test_gene_symbol_guard():
    with pytest.raises(literature.LiteratureError):
        literature.retrieve_for_gene("TP53; DROP TABLE", "x")


def test_hallucinated_citations_stripped(monkeypatch):
    papers = [literature.Paper("111", "T", "abs", year="2020")]

    class Fake(genai.GeminiClient):
        available = True
        def generate(self, prompt, temperature=0.2):
            return "Real claim [PMID:111]. Made-up claim [PMID:999]."

    text, _ = genai.interpret_gene(Fake("k"), "G", None, "ctx", papers)
    assert "[PMID:111]" in text and "999" not in text and "unverified" in text
