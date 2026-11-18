"""Literature retrieval for gene-function interpretation (PubMed via NCBI E-utilities).

Retrieval-augmented generation (RAG) flow:
  1. search PubMed for ``gene`` AND ``biological context``;
  2. fetch titles/abstracts and re-rank them against the query terms;
  3. hand the top abstracts to Gemini, which must cite PMIDs from this set only.

Results are cached on disk so repeated lookups are fast and polite to NCBI
(max ~3 requests/second without an API key).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CACHE_DIR = Path(os.environ.get("NGS_GENIUS_CACHE", Path.home() / ".cache" / "ngs_genius"))
_GENE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-_.@]{0,29}$")


@dataclass
class Paper:
    pmid: str
    title: str
    abstract: str
    journal: str = ""
    year: str = ""
    authors: str = ""
    score: float = 0.0

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    def cite(self) -> str:
        first = self.authors.split(",")[0] if self.authors else "Unknown"
        return f"{first} et al., {self.journal} {self.year} (PMID {self.pmid})"


class LiteratureError(RuntimeError):
    pass


def _clean_term(term: str) -> str:
    return re.sub(r"[^\w\s\-]", " ", term).strip()


def _cache_get(key: str):
    f = CACHE_DIR / f"{hashlib.sha1(key.encode()).hexdigest()}.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            return None
    return None


def _cache_put(key: str, value) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / f"{hashlib.sha1(key.encode()).hexdigest()}.json").write_text(json.dumps(value))
    except OSError:
        pass


def _get(endpoint: str, params: dict, api_key: str | None, retries: int = 3) -> requests.Response:
    params = {**params, "tool": "ngs-genius", "email": os.environ.get("NCBI_EMAIL", "ngs-genius@example.org")}
    key = api_key or os.environ.get("NCBI_API_KEY")
    if key:
        params["api_key"] = key
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(f"{EUTILS}/{endpoint}", params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(1.0 + attempt)
                continue
            r.raise_for_status()
            time.sleep(0.12 if key else 0.35)
            return r
        except requests.RequestException as e:
            last = e
            time.sleep(0.6 * (attempt + 1))
    raise LiteratureError(f"PubMed request failed: {last}")


def search_pubmed(query: str, retmax: int = 20, api_key: str | None = None) -> list[str]:
    r = _get("esearch.fcgi", {"db": "pubmed", "term": query, "retmax": retmax, "retmode": "json",
                              "sort": "relevance"}, api_key)
    return r.json().get("esearchresult", {}).get("idlist", [])


def parse_pubmed_xml(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []
    for art in root.findall(".//PubmedArticle"):
        pmid = (art.findtext(".//MedlineCitation/PMID") or "").strip()
        title = "".join(art.find(".//ArticleTitle").itertext()).strip() if art.find(".//ArticleTitle") is not None else ""
        abs_parts = ["".join(a.itertext()).strip() for a in art.findall(".//Abstract/AbstractText")]
        abstract = " ".join(p for p in abs_parts if p)
        journal = (art.findtext(".//Journal/ISOAbbreviation") or art.findtext(".//Journal/Title") or "").strip()
        year = (art.findtext(".//JournalIssue/PubDate/Year")
                or (art.findtext(".//JournalIssue/PubDate/MedlineDate") or "")[:4]).strip()
        names = []
        for au in art.findall(".//AuthorList/Author")[:3]:
            ln, ini = au.findtext("LastName"), au.findtext("Initials")
            if ln:
                names.append(f"{ln} {ini or ''}".strip())
        if pmid and title:
            papers.append(Paper(pmid, title, abstract, journal, year, ", ".join(names)))
    return papers


def fetch_papers(pmids: list[str], api_key: str | None = None) -> list[Paper]:
    if not pmids:
        return []
    r = _get("efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}, api_key)
    return parse_pubmed_xml(r.text)


def rank_papers(papers: list[Paper], gene: str, context: str = "") -> list[Paper]:
    """Cheap lexical re-ranking: gene in title >> gene in abstract, plus context-term overlap, plus recency."""
    ctx_terms = {t.lower() for t in re.findall(r"[A-Za-z0-9\-]{3,}", context)}
    g = gene.lower()
    for p in papers:
        title, abstract = p.title.lower(), p.abstract.lower()
        s = 0.0
        s += 3.0 if re.search(rf"\b{re.escape(g)}\b", title) else 0.0
        s += min(abstract.count(g), 5) * 0.5
        s += sum(1.0 for t in ctx_terms if t in title) + sum(0.3 for t in ctx_terms if t in abstract)
        s += 0.5 if p.abstract else -2.0
        try:
            s += max(0, int(p.year) - 2010) * 0.05
        except ValueError:
            pass
        p.score = round(s, 2)
    return sorted(papers, key=lambda p: p.score, reverse=True)


def retrieve_for_gene(gene: str, context: str = "", top_k: int = 5, api_key: str | None = None) -> list[Paper]:
    """Retrieve and rank the most relevant PubMed papers for a gene in a biological context."""
    gene = gene.strip()
    if not _GENE_RE.match(gene):
        raise LiteratureError(f"'{gene}' does not look like a gene symbol.")
    ctx = _clean_term(context)
    cache_key = f"v1|{gene}|{ctx}|{top_k}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return [Paper(**p) for p in cached]

    base = f"{gene}[Title/Abstract]"
    pmids = search_pubmed(f"{base} AND ({ctx})", retmax=20, api_key=api_key) if ctx else []
    if len(pmids) < top_k:  # widen to gene-only if the context was too narrow
        pmids = list(dict.fromkeys(pmids + search_pubmed(f"{base} AND humans[MeSH Terms]", retmax=20, api_key=api_key)))
    papers = rank_papers(fetch_papers(pmids, api_key), gene, ctx)[:top_k]
    _cache_put(cache_key, [asdict(p) for p in papers])
    return papers


def format_context(papers: list[Paper], max_chars: int = 1500) -> str:
    """Render papers as numbered evidence blocks for the LLM prompt."""
    blocks = []
    for p in papers:
        abs_ = p.abstract if len(p.abstract) <= max_chars else p.abstract[:max_chars].rsplit(" ", 1)[0] + " ..."
        blocks.append(f"[PMID:{p.pmid}] {p.title} ({p.journal}, {p.year})\n{abs_}")
    return "\n\n".join(blocks)
