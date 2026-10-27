"""Pruebas de las metricas y del software de recuperacion (no usan modelos, corren en segundos)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from common import load_benchmark  # noqa: E402
from metrics import (expand_to_passage_ranking, ndcg_at_k, paired_bootstrap_delta, recall_at_k,  # noqa: E402
                     reciprocal_rank)
from retrieval import BM25Okapi, build_chunks, tokenize_bm25  # noqa: E402


def test_metrics_distinguish_order_from_coverage():
    a, b, rel = ["R", "X", "Y", "Z"], ["X", "Y", "R", "Z"], ["R"]
    assert recall_at_k(a, rel, 4) == recall_at_k(b, rel, 4) == 1.0
    assert reciprocal_rank(a, rel) == 1.0 and reciprocal_rank(b, rel) == 1 / 3
    assert ndcg_at_k(a, rel, 4) > ndcg_at_k(b, rel, 4)
    assert recall_at_k(["X", "R"], ["R", "S"], 2) == 0.5
    assert reciprocal_rank(["X"], ["R"]) == 0.0


def test_expand_removes_overlap_duplicates():
    chunks = [{"passage_ids": ["P1", "P2"]}, {"passage_ids": ["P2", "P3"]}]
    assert expand_to_passage_ranking(chunks) == ["P1", "P2", "P3"]


def test_bm25_prefers_lexical_match():
    docs = ["the vix closed at a record high", "central banks cut interest rates", "bank run on uninsured deposits"]
    bm = BM25Okapi(docs)
    assert bm.score("vix record close").argmax() == 0
    assert bm.score("uninsured deposits").argmax() == 2
    assert "the" not in tokenize_bm25("the VIX") and "vix" in tokenize_bm25("the VIX")


def test_chunks_cover_every_passage_with_overlap():
    bench = load_benchmark()
    chunks = build_chunks(bench["documents"])
    covered = {pid for c in chunks for pid in c["passage_ids"]}
    assert covered == set(bench["passages"])
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def test_benchmark_is_consistent():
    bench = load_benchmark()
    assert len(bench["documents"]) >= 15 and len(bench["passages"]) >= 100 and len(bench["queries"]) >= 20


def test_paired_bootstrap_detects_no_difference():
    frame = pd.DataFrame({"condition": ["A"] * 4 + ["B"] * 4, "query_id": list("abcd") * 2, "m": [1, 0, 1, 0] * 2})
    out = paired_bootstrap_delta(frame, "m", "A", "B", n_resamples=200)
    assert out["delta_mean"] == 0.0 and out["ci_crosses_zero"]
