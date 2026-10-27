"""Segmentacion y recuperadores del Experimento 1: densa (e5 + FAISS), BM25, fusion RRF y Cross-Encoder.

Adaptado de Semana5/Cuaderno5 y Semana7/Cuaderno7: misma segmentacion por passages con overlap, mismo BM25 Okapi
(k1=1.5, b=0.75, IDF con piso de epsilon), mismo RRF (constante 60). El tokenizador y las stopwords se pasan al ingles
porque el corpus y las consultas del parcial estan en ingles.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from common import (BM25_B, BM25_K1, CANDIDATE_DEPTH, DENSE_MODEL_ID, OVERLAP_PASSAGES, RERANKER_MODEL_ID,
                    RRF_CONSTANT, TARGET_WORDS)

# --------------------------------------------------------------------------- segmentacion

def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def build_chunks(documents: list[dict], target_words: int = TARGET_WORDS,
                 overlap_passages: int = OVERLAP_PASSAGES) -> list[dict]:
    """Agrupa passages consecutivos de un documento hasta ~target_words, con overlap de passages."""
    chunks = []
    for doc in documents:
        passages, start, number = doc["passages"], 0, 1
        while start < len(passages):
            selected, words, i = [], 0, start
            while i < len(passages) and (words < target_words or not selected):
                selected.append(passages[i])
                words += count_words(passages[i]["text"])
                i += 1
            chunks.append({"chunk_id": f'{doc["doc_id"]}-C{number:02d}', "doc_id": doc["doc_id"], "title": doc["title"],
                           "passage_ids": [p["passage_id"] for p in selected],
                           "text": "\n".join(p["text"] for p in selected)})
            if i >= len(passages):
                break
            start = max(start + 1, i - overlap_passages)
            number += 1
    return chunks


# --------------------------------------------------------------------------- BM25

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*")
# Lista minima fija; se conservan negaciones y modales (not, no, can, will, should...) por su carga semantica.
ENGLISH_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "had", "has", "have", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "with", "which", "what", "how", "why", "when",
    "did", "do", "does", "after", "than", "into", "also", "about", "who", "whom", "there", "these", "those",
})
STOPWORDS_VERSION = "en-min-v1"


def tokenize_bm25(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in ENGLISH_STOPWORDS]


class BM25Okapi:
    """BM25 Okapi con la politica de IDF de rank_bm25 (IDF negativo -> epsilon * IDF medio)."""

    def __init__(self, texts: list[str], k1: float = BM25_K1, b: float = BM25_B, epsilon: float = 0.25):
        self.k1, self.b = k1, b
        self.docs = [tokenize_bm25(t) for t in texts]
        self.lengths = np.array([len(d) for d in self.docs], dtype=float)
        self.avgdl = float(self.lengths.mean())
        self.tf = [Counter(d) for d in self.docs]
        df = Counter(tok for d in self.docs for tok in set(d))
        n = len(self.docs)
        raw = {t: math.log(n - f + 0.5) - math.log(f + 0.5) for t, f in df.items()}
        avg = sum(raw.values()) / len(raw)
        self.idf = {t: (v if v >= 0 else epsilon * avg) for t, v in raw.items()}

    def score(self, query: str) -> np.ndarray:
        q, scores = tokenize_bm25(query), np.zeros(len(self.docs))
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
            for tok in q:
                f = tf.get(tok, 0)
                if f:
                    scores[i] += self.idf.get(tok, 0.0) * f * (self.k1 + 1) / (f + norm)
        return scores


# --------------------------------------------------------------------------- recuperadores

def stable_top(scores: np.ndarray, depth: int) -> np.ndarray:
    """Orden descendente por score con desempate determinista por indice."""
    return np.lexsort((np.arange(len(scores)), -scores))[:depth]


class Retrievers:
    """Indices construidos una vez; cada metodo devuelve chunks rankeados con su score."""

    def __init__(self, chunks: list[dict]):
        from sentence_transformers import CrossEncoder, SentenceTransformer
        import faiss

        self.chunks = chunks
        self.by_id = {c["chunk_id"]: c for c in chunks}
        t0 = time.perf_counter()
        self.encoder = SentenceTransformer(DENSE_MODEL_ID)
        self.max_seq_length = self.encoder.max_seq_length
        vecs = self.encoder.encode([f'passage: {c["text"]}' for c in chunks], normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False, batch_size=16).astype("float32")
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self.dense_index_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.bm25 = BM25Okapi([c["text"] for c in chunks])
        self.bm25_index_s = time.perf_counter() - t0
        self.cross = CrossEncoder(RERANKER_MODEL_ID)

    def dense(self, query: str, depth: int = CANDIDATE_DEPTH) -> list[dict]:
        q = self.encoder.encode([f"query: {query}"], normalize_embeddings=True, convert_to_numpy=True,
                                show_progress_bar=False).astype("float32")
        scores, idx = self.index.search(q, len(self.chunks))   # ranking completo, desempate abajo
        order = stable_top(scores[0], depth)
        return [{**self.chunks[int(idx[0][i])], "score": float(scores[0][i])} for i in order]

    def lexical(self, query: str, depth: int = CANDIDATE_DEPTH) -> list[dict]:
        s = self.bm25.score(query)
        return [{**self.chunks[int(i)], "score": float(s[i])} for i in stable_top(s, depth)]

    def rrf(self, rankings: list[list[dict]], depth: int = CANDIDATE_DEPTH, constant: int = RRF_CONSTANT) -> list[dict]:
        fused: dict[str, float] = defaultdict(float)
        for ranking in rankings:
            for rank, item in enumerate(ranking, start=1):
                fused[item["chunk_id"]] += 1.0 / (constant + rank)
        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:depth]
        return [{**self.by_id[cid], "score": float(sc)} for cid, sc in ordered]

    def rerank(self, query: str, candidates: list[dict], depth: int = CANDIDATE_DEPTH) -> list[dict]:
        scores = self.cross.predict([[query, c["text"]] for c in candidates], show_progress_bar=False)
        rows = sorted(({**c, "score": float(s)} for c, s in zip(candidates, scores)),
                      key=lambda r: (-r["score"], r["chunk_id"]))
        return rows[:depth]


def ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def run_condition(r: Retrievers, condition: str, query: str) -> tuple[list[dict], dict[str, float]]:
    """A = densa; B = A + BM25 + RRF; C = B + Cross-Encoder; X = solo BM25 (diagnostico, no es A/B/C)."""
    t = {"dense_ms": 0.0, "bm25_ms": 0.0, "rrf_ms": 0.0, "reranker_ms": 0.0}
    total = time.perf_counter()
    if condition == "X_bm25_only":
        t0 = time.perf_counter(); ranking = r.lexical(query); t["bm25_ms"] = ms(t0)
    else:
        t0 = time.perf_counter(); dense = r.dense(query); t["dense_ms"] = ms(t0)
        ranking = dense
        if condition in ("B_hybrid_rrf", "C_hybrid_rrf_reranker"):
            t0 = time.perf_counter(); lex = r.lexical(query); t["bm25_ms"] = ms(t0)
            t0 = time.perf_counter(); ranking = r.rrf([dense, lex]); t["rrf_ms"] = ms(t0)
            if condition == "C_hybrid_rrf_reranker":
                t0 = time.perf_counter(); ranking = r.rerank(query, ranking); t["reranker_ms"] = ms(t0)
    t["total_ms"] = ms(total)
    return ranking, t
