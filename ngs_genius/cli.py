"""Command-line interface:  python -m ngs_genius.cli --demo --out results/"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import genai, report
from .pipeline import Params, run_pipeline
from .validate import has_errors

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="NGS-Genius RNA-seq analysis")
    ap.add_argument("--counts"), ap.add_argument("--meta")
    ap.add_argument("--demo", action="store_true", help="use the bundled synthetic dataset")
    ap.add_argument("--condition", default="condition")
    ap.add_argument("--reference", default="control")
    ap.add_argument("--treatment", default="treated")
    ap.add_argument("--covariates", default="", help="comma-separated, e.g. batch")
    ap.add_argument("--padj", type=float, default=0.05)
    ap.add_argument("--lfc", type=float, default=1.0)
    ap.add_argument("--engine", default="auto", choices=["auto", "r", "pydeseq2"])
    ap.add_argument("--context", default="")
    ap.add_argument("--ai", action="store_true", help="add Gemini explanations (needs GEMINI_API_KEY)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)

    if a.demo:
        counts = pd.read_csv(ROOT / "data" / "demo_counts.csv", index_col=0)
        meta = pd.read_csv(ROOT / "data" / "demo_metadata.csv", index_col=0)
    elif a.counts and a.meta:
        counts = pd.read_csv(a.counts, sep=None, engine="python", index_col=0)
        meta = pd.read_csv(a.meta, sep=None, engine="python", index_col=0)
    else:
        ap.error("provide --demo or both --counts and --meta")

    p = Params(a.condition, a.reference, a.treatment, [c for c in a.covariates.split(",") if c],
               padj_cutoff=a.padj, lfc_cutoff=a.lfc, engine=a.engine, context=a.context)
    res = run_pipeline(counts, meta, p)
    for i in res.issues:
        print(i)
    if res.de is None:
        if res.de_error:
            from .troubleshoot import diagnose, format_diagnoses
            print("\nDE failed. Diagnosis:\n" + format_diagnoses(diagnose(res.de_error)))
        return 1

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res.de.to_csv(out / "de_results.csv", index=False)
    res.qc.to_csv(out / "qc_table.csv")
    qc_text = de_text = ""
    if a.ai:
        client = genai.GeminiClient()
        qc_text, _ = genai.explain_qc(client, res.qc, res.issues, res.meta, a.condition)
        de_text, _ = genai.explain_de(client, res.de, p)
    (out / "report.md").write_text(report.build_report(res, qc_text, de_text))
    up, down = int((res.de.regulation == "Up").sum()), int((res.de.regulation == "Down").sum())
    print(f"{res.engine_used}: {up} up, {down} down. Outputs in {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
