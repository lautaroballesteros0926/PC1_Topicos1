"""Metricas de recuperacion sobre rankings de passages (qrels binarios) y bootstrap pareado.

Adaptadas de Semana7/Cuaderno7-CC-0F4.ipynb. Los rankings de chunks se expanden a rankings de passages
SIN duplicados antes de medir, para que el overlap no cuente dos veces el mismo passage.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from common import BOOTSTRAP_ALPHA, BOOTSTRAP_RESAMPLES, SEED


def expand_to_passage_ranking(ranked_chunks: list[dict[str, Any]]) -> list[str]:
    ordered, seen = [], set()
    for chunk in ranked_chunks:
        for pid in chunk["passage_ids"]:
            if pid not in seen:
                ordered.append(pid)
                seen.add(pid)
    return ordered


def recall_at_k(ranking: list[str], relevant: list[str], k: int) -> float:
    return len(set(relevant) & set(ranking[:k])) / len(set(relevant))


def reciprocal_rank(ranking: list[str], relevant: list[str]) -> float:
    rel = set(relevant)
    for rank, pid in enumerate(ranking, start=1):
        if pid in rel:
            return 1.0 / rank
    return 0.0


def dcg_at_k(ranking: list[str], relevant: list[str], k: int) -> float:
    rel = set(relevant)
    return sum((2.0 ** (1.0 if pid in rel else 0.0) - 1.0) / math.log2(i + 1)
               for i, pid in enumerate(ranking[:k], start=1))


def ndcg_at_k(ranking: list[str], relevant: list[str], k: int) -> float:
    ideal = dcg_at_k(list(relevant), relevant, k)
    return dcg_at_k(ranking, relevant, k) / ideal if ideal > 0 else 0.0


def classify_retrieval_failure(ranked_chunks: list[dict[str, Any]], relevant: list[str], depth: int, top_k: int = 3) -> str:
    ranking = expand_to_passage_ranking(ranked_chunks)
    if recall_at_k(ranking, relevant, min(depth, len(ranking))) == 0:
        return "RETRIEVAL_MISS"
    r_top = recall_at_k(ranking, relevant, top_k)
    if r_top == 0:
        return "RANKING_ERROR"
    return "PARTIAL_RECALL" if r_top < 1 else "RETRIEVAL_OK"


def paired_bootstrap_delta(frame: pd.DataFrame, metric: str, cond_a: str, cond_b: str,
                           n_resamples: int = BOOTSTRAP_RESAMPLES, alpha: float = BOOTSTRAP_ALPHA,
                           seed: int = SEED) -> dict[str, Any]:
    a = frame[frame["condition"] == cond_a].set_index("query_id")[metric].sort_index()
    b = frame[frame["condition"] == cond_b].set_index("query_id")[metric].sort_index()
    assert list(a.index) == list(b.index), "comparacion pareada: mismas query_id"
    deltas = b.to_numpy(float) - a.to_numpy(float)
    rng = np.random.default_rng(seed)
    means = np.array([deltas[rng.integers(0, len(deltas), len(deltas))].mean() for _ in range(n_resamples)])
    lo, hi = float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
    return {"metric": metric, "comparison": f"{cond_b} - {cond_a}", "n_queries": len(deltas),
            "delta_mean": float(deltas.mean()), "ci95_low": lo, "ci95_high": hi,
            "ci_crosses_zero": bool(lo <= 0.0 <= hi), "queries_improved": int((deltas > 0).sum()),
            "queries_worse": int((deltas < 0).sum()), "n_resamples": n_resamples}
