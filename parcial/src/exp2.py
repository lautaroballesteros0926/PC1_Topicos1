"""Experimento 2: mismo LLM (Qwen2.5-0.5B) sin recuperacion (closed-book) y con evidencia recuperada (RAG).

Fija modelo, preguntas, decoding greedy y presupuesto de contexto; solo cambia la presencia de evidencia. La configuracion de
recuperacion se declara en `results/exp2/declared_config.json` ANTES de generar. Produce respuestas, comprobaciones mecanicas de
citas y las plantillas para la AUDITORIA MANUAL (correccion, citas, grounding). Tambien mide la consulta estructurada (Semana 3).
"""
from __future__ import annotations

import json
import platform
import sys
import time
from importlib import metadata

import pandas as pd
from jsonschema import Draft202012Validator

from common import DATA, RESULTS, SEED, benchmark_hashes, load_benchmark, read_jsonl, set_seed
from llm import DECODING, LLM_ID, LLM_REVISION, LocalLLM
from rag import (build_evidence, citation_metrics, is_abstention, load_prompt, numeric_fact_proxy, parse_json_object, split_claims,
                 token_f1)
from retrieval import Retrievers, build_chunks, run_condition

OUT = RESULTS / "exp2" / "qwen"
RAG_CONDITION = "C_hybrid_rrf_reranker"
RAG_TOP_K_CHUNKS = 3


def declare_config() -> dict:
    cfg = {"declared_before_generation_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "retrieval_condition": RAG_CONDITION, "top_k_chunks": RAG_TOP_K_CHUNKS,
           "selection_rule": ("Se elige C (B + Cross-Encoder) por tener el mayor Recall@3, Recall@5 y nDCG@5 del Experimento 1, "
                              "aunque los intervalos de bootstrap casi siempre cruzan cero; B queda como alternativa mas barata."),
           "llm": LLM_ID, "llm_revision": LLM_REVISION, "decoding": DECODING, "seed": SEED,
           "same_questions_and_decoding_in_both_conditions": True}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT.parent / "declared_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def pkg(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "n/a"


def write_audit_templates(gens: list[dict], refs: dict, bench: dict) -> None:
    """Plantillas de auditoria manual. Respuestas: closed-book y RAG (correccion; y citas en RAG). Afirmaciones: solo RAG, porque
    el grounding se mide contra la evidencia entregada; una respuesta closed-book no tiene evidencia contra la cual anclarse."""
    ans_rows, claim_rows = [], []
    for g in gens:
        ref = refs.get(g["query_id"])
        ans_rows.append({"query_id": g["query_id"], "condition": g["condition"], "question": g["question"], "answer": g["answer"],
                         "expected_abstain": g["expected_abstain"], "reference_answer": ref["reference_answer"] if ref else "(sin evidencia: debe abstenerse)",
                         "required_facts": " | ".join(ref["required_facts"]) if ref else "", "supporting_passages": " ".join(ref["supporting_passage_ids"]) if ref else "",
                         "answer_correct_0_1": "", "abstention_appropriate_0_1": "", "citations_correct_0_1": "", "notes": ""})
        if g["condition"] != "rag":
            continue
        for i, claim in enumerate(split_claims(g["answer"]), 1):
            cited = g["cited"].split() if g["cited"] else []
            cites = [p for p in cited if p in g["context_passage_ids"]]
            claim_rows.append({"query_id": g["query_id"], "condition": g["condition"], "claim_id": f'{g["query_id"]}-{g["condition"]}-{i}',
                               "claim": claim, "cited_passages": " ".join(cited),
                               "cited_text": " || ".join(f"[{p}] {bench['passages'][p][:350]}" for p in cites),
                               "supported_yes_partial_no": "", "notes": ""})
    pd.DataFrame(ans_rows).to_csv(OUT / "audit_answers_template.csv", index=False)
    pd.DataFrame(claim_rows).to_csv(OUT / "audit_claims_template.csv", index=False)


def run_exp2() -> dict:
    set_seed(SEED)
    cfg = declare_config()
    bench = load_benchmark()
    qrels, passages = bench["qrels"], bench["passages"]
    refs = {r["query_id"]: r for r in read_jsonl(DATA / "reference_answers_parcial.jsonl")}
    noev = read_jsonl(DATA / "queries_sin_evidencia.jsonl")
    items = [{"query_id": q["query_id"], "question": q["query"], "expected_abstain": False} for q in bench["queries"] if q["exp2_rag"]]
    items += [{"query_id": n["query_id"], "question": n["query"], "expected_abstain": True} for n in noev]

    # 1) Recuperacion con la configuracion declarada (mismo codigo que el Experimento 1)
    r = Retrievers(build_chunks(bench["documents"]))
    for it in items:
        ranking, _ = run_condition(r, RAG_CONDITION, it["question"])
        evidence, ids = build_evidence(ranking, RAG_TOP_K_CHUNKS, passages)
        it.update({"top_chunks": [c["chunk_id"] for c in ranking[:RAG_TOP_K_CHUNKS]], "context_passage_ids": ids, "evidence": evidence,
                   "relevant": qrels.get(it["query_id"], [])})
    del r

    # 2) Generacion: closed-book y RAG con el mismo LLM y decoding
    llm = LocalLLM()
    sys_cb, usr_cb = load_prompt("closed_book_system.txt"), load_prompt("closed_book_user.txt")
    sys_rag, usr_rag = load_prompt("rag_system.txt"), load_prompt("rag_user.txt")
    rows, gens = [], []
    for n, it in enumerate(items, 1):
        for cond in ("closed_book", "rag"):
            user = usr_cb.format(question=it["question"]) if cond == "closed_book" else usr_rag.format(evidence=it["evidence"], question=it["question"])
            g = llm.chat(sys_cb if cond == "closed_book" else sys_rag, user)
            ref = refs.get(it["query_id"])
            cm = citation_metrics(g["text"], it["context_passage_ids"] if cond == "rag" else [], it["relevant"])
            row = {"query_id": it["query_id"], "condition": cond, "expected_abstain": it["expected_abstain"], "answer": g["text"],
                   "abstained": is_abstention(g["text"]), "n_cited": cm["n_cited"], "cited": " ".join(cm["cited"]),
                   "cited_in_context_rate": cm["cited_in_context_rate"], "cited_relevant_rate": cm["cited_relevant_rate"],
                   "relevant_in_context_recall": cm["relevant_in_context_recall"] if cond == "rag" else None,
                   "token_f1_proxy": round(token_f1(g["text"], ref["reference_answer"]), 3) if ref else None,
                   "numeric_fact_proxy": numeric_fact_proxy(g["text"], ref["required_facts"]) if ref else None,
                   "n_prompt_tokens": g["n_prompt_tokens"], "n_new_tokens": g["n_new_tokens"], "hit_max_new_tokens": g["hit_max_new_tokens"],
                   "latency_s": g["latency_s"]}
            rows.append(row)
            gens.append({**row, "question": it["question"], "context_chunks": it["top_chunks"] if cond == "rag" else [],
                         "context_passage_ids": it["context_passage_ids"] if cond == "rag" else [], "rendered_prompt": g["prompt"]})
            print(f"[{n}/{len(items)}] {it['query_id']} {cond}: {g['n_new_tokens']} tok, {g['latency_s']}s -> {g['text'][:90]!r}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "generations_summary.csv", index=False)
    (OUT / "generations.jsonl").write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in gens), encoding="utf-8")

    # 3) Plantillas de auditoria manual (las completa la persona, no el sistema)
    write_audit_templates(gens, refs, bench)

    # 4) Consulta estructurada (Semana 3): parseable / schema-valid / copia literal
    schema = json.loads((DATA / "schema_consulta.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    sq_sys, sq_usr = load_prompt("structured_query_system.txt"), load_prompt("structured_query_user.txt")
    sq_rows = []
    all_queries = [(q["query_id"], q["query"]) for q in bench["queries"]] + [(n["query_id"], n["query"]) for n in noev]
    for qid, text in all_queries:
        g = llm.chat(sq_sys, sq_usr.format(need=text), max_new_tokens=96)
        obj, mode = parse_json_object(g["text"])
        errors = sorted(e.message for e in validator.iter_errors(obj)) if obj is not None else ["unparseable"]
        valid = obj is not None and not errors
        sq_rows.append({"query_id": qid, "need": text, "raw_output": g["text"], "parse_mode": mode, "parseable": obj is not None,
                        "schema_valid": valid, "errors": "; ".join(errors)[:200], "query_q1": obj.get("query") if valid else "",
                        "intent": obj.get("intent") if valid else "", "abstain": obj.get("abstain") if valid else None,
                        "q1_equals_q0": bool(valid and obj["query"].strip().rstrip("?").lower() == text.strip().rstrip("?").lower()),
                        "latency_s": g["latency_s"]})
        print(f"[SQ] {qid}: parse={mode} valid={valid}", flush=True)
    sq = pd.DataFrame(sq_rows)
    sq.to_csv(OUT / "structured_queries.csv", index=False)

    manifest = {"config": cfg, "benchmark_hashes": benchmark_hashes(), "n_items": len(items), "n_generations": len(rows),
                "llm_context_window": llm.context_window, "prompt_tokens_max": int(df["n_prompt_tokens"].max()),
                "structured_query": {"n": len(sq), "parseable": int(sq["parseable"].sum()), "schema_valid": int(sq["schema_valid"].sum()),
                                     "exact_json": int((sq["parse_mode"] == "exact").sum()), "q1_equals_q0": int(sq["q1_equals_q0"].sum())},
                "python": platform.python_version(), "torch": pkg("torch"), "transformers": pkg("transformers"),
                "sentence_transformers": pkg("sentence-transformers"), "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"generations": df, "structured": sq, "manifest": manifest}


if __name__ == "__main__":
    sys.exit(0 if run_exp2() else 1)
