"""Diagnosticos de apoyo para el analisis (no cambian las metricas declaradas del Experimento 1)."""
from __future__ import annotations

import json
import statistics
from typing import Any

import pandas as pd

from common import RESULTS, TOP_K_VALUES

EXP1 = RESULTS / "exp1"


def load_rankings() -> list[dict[str, Any]]:
    return [json.loads(l) for l in (EXP1 / "rankings.jsonl").read_text(encoding="utf-8").splitlines()]


def chunk_coverage_table(rankings: list[dict[str, Any]], conditions: list[str]) -> pd.DataFrame:
    """Recall@k con la definicion de la PC2/Semana 4: passages relevantes CUBIERTOS por los k primeros chunks.
    Elimina el efecto del orden dentro del chunk, que el ranking de passages (Cuaderno 7) introduce."""
    rows = []
    for cond in conditions:
        rs = [r for r in rankings if r["condition"] == cond]
        row: dict[str, Any] = {"condition": cond}
        for k in TOP_K_VALUES:
            cov = []
            for r in rs:
                top = {p for c in r["top_chunks"][:k] for p in c["passage_ids"]}
                cov.append(len(top & set(r["relevant"])) / len(set(r["relevant"])))
            row[f"chunk_recall@{k}"] = round(statistics.mean(cov), 3)
        row["hit@3 (algun relevante en top-3 chunks)"] = round(statistics.mean(
            float(bool({p for c in r["top_chunks"][:3] for p in c["passage_ids"]} & set(r["relevant"]))) for r in rs), 3)
        rows.append(row)
    return pd.DataFrame(rows).set_index("condition")


def recall_ceilings(qrels: dict[str, list[str]]) -> dict[str, float]:
    """Maximo alcanzable de Recall@k con qrels de tamano variable (p. ej. Recall@1 <= 1/|relevantes|)."""
    return {f"recall@{k}": round(statistics.mean(min(k, len(v)) / len(v) for v in qrels.values()), 3) for k in TOP_K_VALUES}


def chunk_rank_of_relevant(rankings: list[dict[str, Any]], qid: str) -> pd.DataFrame:
    """Para una consulta: rank del primer chunk con evidencia relevante y chunks relevantes dentro del top-10, por condicion."""
    rows = []
    for r in rankings:
        if r["query_id"] != qid:
            continue
        rel = set(r["relevant"])
        hits = [(i + 1, c["chunk_id"], sorted(rel & set(c["passage_ids"]))) for i, c in enumerate(r["top_chunks"]) if rel & set(c["passage_ids"])]
        rows.append({"condition": r["condition"], "first_relevant_chunk_rank": hits[0][0] if hits else None,
                     "relevant_chunks_in_top10": "; ".join(f"#{i} {cid} {ps}" for i, cid, ps in hits) or "(ninguno)",
                     "top3_chunks": ", ".join(c["chunk_id"] for c in r["top_chunks"][:3])})
    return pd.DataFrame(rows).set_index("condition")


def wins_losses(per_query: pd.DataFrame, metric: str, a: str, b: str) -> pd.DataFrame:
    piv = per_query.pivot(index="query_id", columns="condition", values=metric)
    out = piv[[a, b]].copy()
    out["delta"] = out[b] - out[a]
    return out[out["delta"] != 0].sort_values("delta")
