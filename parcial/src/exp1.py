"""Experimento 1: recuperacion y reranking (A densa, B = A + BM25 + RRF, C = B + Cross-Encoder).

Mismo benchmark (parcial-v1), misma segmentacion, misma consulta original (Q0) y mismo top-k en las tres condiciones.
Escribe en results/exp1/: per_query.csv, summary.csv, latency_by_stage.csv, bootstrap_pairwise_ci.csv,
failure_types.csv, rankings.jsonl, chunks.csv y run_manifest.json.
"""
from __future__ import annotations

import json
import platform
import time
from importlib import metadata

import numpy as np
import pandas as pd

from common import (BM25_B, BM25_K1, BOOTSTRAP_RESAMPLES, CANDIDATE_DEPTH, DENSE_MODEL_ID, OVERLAP_PASSAGES, RERANKER_MODEL_ID,
                    RESULTS, RRF_CONSTANT, SEED, TARGET_WORDS, TOP_K_VALUES, benchmark_hashes, load_benchmark, set_seed)
from metrics import (classify_retrieval_failure, expand_to_passage_ranking, ndcg_at_k, paired_bootstrap_delta,
                     recall_at_k, reciprocal_rank)
from retrieval import STOPWORDS_VERSION, Retrievers, build_chunks, count_words, run_condition

CONDITIONS = ["A_dense", "B_hybrid_rrf", "C_hybrid_rrf_reranker"]
DIAGNOSTIC = ["X_bm25_only"]          # no es A/B/C: ayuda a atribuir el cambio de A a B
STAGES = ["dense_ms", "bm25_ms", "rrf_ms", "reranker_ms", "total_ms"]
LATENCY_REPEATS = 3
OUT = RESULTS / "exp1"


def p95(x) -> float:
    return float(np.percentile(np.asarray(x, float), 95))


def pkg(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "n/a"


def run_exp1() -> dict:
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    bench = load_benchmark()
    queries, qrels = bench["queries"], bench["qrels"]
    chunks = build_chunks(bench["documents"])
    r = Retrievers(chunks)

    # Control: ningun chunk debe exceder la ventana del encoder (para medir recuperacion y no truncacion)
    tok = r.encoder.tokenizer
    n_tokens = [len(tok(f'passage: {c["text"]}')["input_ids"]) for c in chunks]
    pd.DataFrame([{"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "passage_ids": " ".join(c["passage_ids"]),
                   "words": count_words(c["text"]), "e5_tokens": n} for c, n in zip(chunks, n_tokens)]
                 ).to_csv(OUT / "chunks.csv", index=False)

    # Calentamiento (descartado) y corridas medidas
    for cond in CONDITIONS + DIAGNOSTIC:
        for q in queries:
            run_condition(r, cond, q["query"])
    rows, rankings, failures = [], [], []
    for cond in CONDITIONS + DIAGNOSTIC:
        for q in queries:
            qid, rel = q["query_id"], qrels[q["query_id"]]
            passes = [run_condition(r, cond, q["query"]) for _ in range(LATENCY_REPEATS)]
            ranking = passes[0][0]
            assert all([c["chunk_id"] for c in p[0]] == [c["chunk_id"] for c in ranking] for p in passes), "no determinista"
            timing = {s: float(np.median([p[1][s] for p in passes])) for s in STAGES}
            pr = expand_to_passage_ranking(ranking)
            row = {"condition": cond, "query_id": qid, "difficulty": q["difficulty"], "category": q["category"], **timing,
                   "mrr": reciprocal_rank(pr, rel), "ndcg@5": ndcg_at_k(pr, rel, 5)}
            row.update({f"recall@{k}": recall_at_k(pr, rel, k) for k in TOP_K_VALUES})
            rows.append(row)
            failures.append({"condition": cond, "query_id": qid,
                             "failure_type": classify_retrieval_failure(ranking, rel, CANDIDATE_DEPTH)})
            rankings.append({"condition": cond, "query_id": qid, "relevant": rel,
                             "top_chunks": [{"chunk_id": c["chunk_id"], "score": round(c["score"], 5),
                                             "passage_ids": c["passage_ids"]} for c in ranking],
                             "passage_ranking": pr})
    per_query = pd.DataFrame(rows)
    per_query.to_csv(OUT / "per_query.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "failure_types.csv", index=False)
    (OUT / "rankings.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rankings), encoding="utf-8")

    metric_cols = [f"recall@{k}" for k in TOP_K_VALUES] + ["mrr", "ndcg@5"]
    summary = per_query.groupby("condition")[metric_cols].mean().loc[CONDITIONS + DIAGNOSTIC]
    summary.to_csv(OUT / "summary.csv")
    lat = []
    for cond in CONDITIONS + DIAGNOSTIC:
        sub = per_query[per_query["condition"] == cond]
        lat.append({"condition": cond, **{f"{s}_mean": sub[s].mean() for s in STAGES}, "total_p95_ms": p95(sub["total_ms"])})
    pd.DataFrame(lat).to_csv(OUT / "latency_by_stage.csv", index=False)

    boot = []
    for m in ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5"]:
        boot.append(paired_bootstrap_delta(per_query, m, "A_dense", "B_hybrid_rrf"))
        boot.append(paired_bootstrap_delta(per_query, m, "B_hybrid_rrf", "C_hybrid_rrf_reranker"))
    pd.DataFrame(boot).to_csv(OUT / "bootstrap_pairwise_ci.csv", index=False)

    manifest = {
        "benchmark_hashes": benchmark_hashes(), "seed": SEED, "conditions": CONDITIONS, "diagnostic": DIAGNOSTIC,
        "target_words": TARGET_WORDS, "overlap_passages": OVERLAP_PASSAGES, "candidate_depth": CANDIDATE_DEPTH,
        "rrf_constant": RRF_CONSTANT, "top_k": list(TOP_K_VALUES), "bm25": {"k1": BM25_K1, "b": BM25_B, "epsilon": 0.25,
        "stopwords": STOPWORDS_VERSION, "stemming": False}, "dense_model": DENSE_MODEL_ID, "reranker_model": RERANKER_MODEL_ID,
        "n_chunks": len(chunks), "chunk_words_max": int(max(count_words(c["text"]) for c in chunks)),
        "e5_tokens_max": int(max(n_tokens)), "e5_max_seq_length": int(r.max_seq_length),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "latency_repeats": LATENCY_REPEATS,
        "dense_index_build_s": round(r.dense_index_s, 2), "bm25_index_build_s": round(r.bm25_index_s, 4),
        "python": platform.python_version(), "torch": pkg("torch"), "sentence_transformers": pkg("sentence-transformers"),
        "faiss": pkg("faiss-cpu"), "transformers": pkg("transformers"), "numpy": pkg("numpy"), "pandas": pkg("pandas"),
        "cpu": platform.processor() or platform.machine(), "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"summary": summary, "latency": pd.DataFrame(lat), "bootstrap": pd.DataFrame(boot), "per_query": per_query,
            "failures": pd.DataFrame(failures), "manifest": manifest, "chunks": chunks}


if __name__ == "__main__":
    out = run_exp1()
    print(out["summary"].round(3).to_string())
    print(out["latency"].round(1).to_string())
    print(out["manifest"]["n_chunks"], "chunks; e5 tokens max", out["manifest"]["e5_tokens_max"])
