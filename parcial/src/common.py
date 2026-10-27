"""Rutas, constantes fijas del experimento y carga de datos del benchmark parcial-v1."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PROMPTS = ROOT / "prompts"

SEED = 42

# Variables fijas del Experimento 1 (no se tocan despues de ver resultados)
TARGET_WORDS = 180          # misma segmentacion de referencia que Semanas 4, 5 y 7
OVERLAP_PASSAGES = 1
CANDIDATE_DEPTH = 10
RRF_CONSTANT = 60
TOP_K_VALUES = (1, 3, 5)
BM25_K1, BM25_B = 1.5, 0.75
BOOTSTRAP_RESAMPLES, BOOTSTRAP_ALPHA = 5000, 0.05

DENSE_MODEL_ID = "intfloat/multilingual-e5-small"
RERANKER_MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def load_benchmark() -> dict[str, Any]:
    """Corpus, consultas, qrels y referencias del benchmark congelado."""
    documents = read_jsonl(DATA / "corpus_parcial.jsonl")
    queries = read_jsonl(DATA / "queries_parcial.jsonl")
    qrels = json.loads((DATA / "qrels_parcial.json").read_text(encoding="utf-8"))
    passages = {p["passage_id"]: p["text"] for d in documents for p in d["passages"]}
    assert set(qrels) == {q["query_id"] for q in queries}
    assert all(set(v) <= set(passages) for v in qrels.values())
    return {"documents": documents, "queries": queries, "qrels": qrels, "passages": passages}


def benchmark_hashes() -> dict[str, str]:
    files = ["corpus_parcial.jsonl", "queries_parcial.jsonl", "qrels_parcial.json",
             "reference_answers_parcial.jsonl", "queries_sin_evidencia.jsonl"]
    return {f: sha256_12(DATA / f) for f in files}
