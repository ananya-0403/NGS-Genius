"""Generate a small, fully SYNTHETIC RNA-seq dataset for demos and tests.

Scenario: a p53-activation-like experiment (4 control vs 4 treated samples, 2 batches).
Counts are simulated from a negative binomial model with a handful of planted
differentially expressed genes. The gene symbols are real so that literature
retrieval has something meaningful to look up, but THE DATA ARE NOT REAL
MEASUREMENTS and must not be used for biological conclusions.

Also plants one deliberately shallow sample (treated_4) so the QC module has
something to flag.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

UP = ["CDKN1A", "MDM2", "GADD45A", "BAX", "BBC3", "PMAIP1", "SESN1", "TP53INP1", "FAS", "ZMAT3",
      "XPC", "DDB2", "TNFRSF10B", "PLK3", "BTG2", "RRM2B", "AEN", "TRIM22", "FDXR", "SULF2"]
DOWN = ["E2F1", "CCNB1", "CDK1", "PLK1", "AURKA", "AURKB", "MKI67", "TOP2A", "CCNA2", "BUB1",
        "CDC20", "MYBL2", "FOXM1", "CENPF", "KIF11"]
BACKGROUND_REAL = ["TP53", "MYC", "GAPDH", "ACTB", "RPL13A", "B2M", "EEF1A1", "TUBB", "HSP90AB1", "PCNA"]


def make_dataset(n_genes: int = 6000, n_per_group: int = 4, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    named = UP + DOWN + BACKGROUND_REAL
    filler = [f"GENE{i:05d}" for i in range(1, n_genes - len(named) + 1)]
    genes = named + filler

    base_mean = np.exp(rng.normal(4.5, 2.0, size=n_genes)).clip(1, 20000)
    base_mean[[genes.index(g) for g in UP + DOWN]] = rng.uniform(80, 1500, size=len(UP + DOWN))
    dispersion = (0.05 + 0.6 / np.sqrt(base_mean)) * rng.uniform(0.8, 1.4, size=n_genes)

    lfc = np.zeros(n_genes)
    for g in UP:
        lfc[genes.index(g)] = rng.uniform(1.5, 3.2)
    for g in DOWN:
        lfc[genes.index(g)] = -rng.uniform(1.2, 2.5)
    # moderate effects on ~4% of the remaining genes
    pool = np.array([i for i in range(n_genes) if lfc[i] == 0])
    moderate = rng.choice(pool, size=int(0.04 * n_genes), replace=False)
    lfc[moderate] = rng.choice([-1, 1], size=len(moderate)) * rng.uniform(0.6, 1.5, size=len(moderate))

    samples = [f"control_{i + 1}" for i in range(n_per_group)] + [f"treated_{i + 1}" for i in range(n_per_group)]
    condition = ["control"] * n_per_group + ["treated"] * n_per_group
    batch = ["A", "B"] * n_per_group  # balanced across conditions
    depth = rng.uniform(0.85, 1.15, size=len(samples))
    depth[samples.index("treated_4")] = 0.40  # planted shallow sample

    counts = np.zeros((n_genes, len(samples)), dtype=int)
    for j, cond in enumerate(condition):
        mu = base_mean * depth[j] * (2.0 ** (lfc if cond == "treated" else 0.0))
        mu = mu * (1.08 if batch[j] == "B" else 1.0)  # mild batch effect
        r = 1.0 / dispersion
        p = r / (r + mu)
        counts[:, j] = rng.negative_binomial(r, p)

    counts_df = pd.DataFrame(counts, index=pd.Index(genes, name="gene"), columns=samples)
    meta_df = pd.DataFrame({"condition": condition, "batch": batch}, index=pd.Index(samples, name="sample"))
    return counts_df, meta_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--genes", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    counts, meta = make_dataset(a.genes, seed=a.seed)
    counts.to_csv(out / "demo_counts.csv")
    meta.to_csv(out / "demo_metadata.csv")
    print(f"Wrote {counts.shape[0]} genes x {counts.shape[1]} samples to {out}")


if __name__ == "__main__":
    main()
