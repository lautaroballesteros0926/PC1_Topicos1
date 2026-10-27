"""ANEXO del Experimento 2: la misma corrida closed-book vs RAG con DeepSeek (API de pago, compatible con OpenAI).

No forma parte del analisis principal (que usa Qwen2.5-0.5B con auditoria manual). Reutiliza EXACTAMENTE las mismas preguntas, el
mismo contexto recuperado (guardado por exp2.py) y los mismos prompts; solo cambia el modelo. Solo metricas automaticas.

Requiere DEEPSEEK_API_KEY (variable de entorno o parcial/.env, que esta en .gitignore). Sin clave no hace nada.
Las respuestas crudas, el modelo que respondio, el uso de tokens y la fecha se guardan en results/exp2/deepseek_anexo/; si el archivo ya
existe no se vuelve a llamar a la API (la API no es 100 % determinista ni versionada: lo guardado es la evidencia).
"""
from __future__ import annotations

import json
import os
import time

import httpx
import pandas as pd
from dotenv import load_dotenv

from common import DATA, RESULTS, ROOT, load_benchmark, read_jsonl
from rag import citation_metrics, is_abstention, load_prompt, numeric_fact_proxy, token_f1

OUT = RESULTS / "exp2" / "deepseek_anexo"
BASE_URL = "https://api.deepseek.com"
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
# El modelo razona antes de responder y esos tokens cuentan contra max_tokens (con 300 hubo respuestas vacias: ver run0).
DECODING = {"temperature": 0, "max_tokens": 2000}


def chat(client: httpx.Client, system: str, user: str) -> dict:
    r = client.post(f"{BASE_URL}/chat/completions", json={"model": MODEL, "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": user}], "stream": False, **DECODING}, timeout=120)
    r.raise_for_status()
    return r.json()


def run_deepseek() -> pd.DataFrame | None:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    target = OUT / "generations.jsonl"
    if target.exists():
        print("ya existen respuestas guardadas:", target)
    elif not key:
        print("DEEPSEEK_API_KEY no definida: se omite el anexo (crear parcial/.env a partir de .env.example).")
        return None
    bench = load_benchmark()
    passages, qrels = bench["passages"], bench["qrels"]
    refs = {r["query_id"]: r for r in read_jsonl(DATA / "reference_answers_parcial.jsonl")}
    base = [g for g in read_jsonl(RESULTS / "exp2" / "qwen" / "generations.jsonl")]
    OUT.mkdir(parents=True, exist_ok=True)
    sys_cb, usr_cb = load_prompt("closed_book_system.txt"), load_prompt("closed_book_user.txt")
    sys_rag, usr_rag = load_prompt("rag_system.txt"), load_prompt("rag_user.txt")
    if not target.exists():
        out = []
        with httpx.Client(headers={"Authorization": f"Bearer {key}"}) as client:
            for g in base:
                if g["condition"] == "closed_book":
                    system, user = sys_cb, usr_cb.format(question=g["question"])
                else:
                    ev = "\n\n".join(f"[{p}] {passages[p]}" for p in g["context_passage_ids"])
                    system, user = sys_rag, usr_rag.format(evidence=ev, question=g["question"])
                t0 = time.perf_counter()
                resp = chat(client, system, user)
                out.append({"query_id": g["query_id"], "condition": g["condition"], "question": g["question"],
                            "context_passage_ids": g["context_passage_ids"], "expected_abstain": g["expected_abstain"],
                            "answer": resp["choices"][0]["message"]["content"].strip(), "finish_reason": resp["choices"][0]["finish_reason"],
                            "model_reported": resp.get("model"), "usage": resp.get("usage"), "latency_s": round(time.perf_counter() - t0, 2),
                            "called_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "decoding": DECODING})
                print(g["query_id"], g["condition"], "->", out[-1]["answer"][:80].replace("\n", " "), flush=True)
        target.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out), encoding="utf-8")
    rows = []
    for g in read_jsonl(target):
        ref, rel = refs.get(g["query_id"]), qrels.get(g["query_id"], [])
        cm = citation_metrics(g["answer"], g["context_passage_ids"] if g["condition"] == "rag" else [], rel)
        rows.append({"query_id": g["query_id"], "condition": g["condition"], "expected_abstain": g["expected_abstain"], "answer": g["answer"],
                     "abstained": is_abstention(g["answer"]), "n_cited": cm["n_cited"], "cited_in_context_rate": cm["cited_in_context_rate"],
                     "cited_relevant_rate": cm["cited_relevant_rate"],
                     "token_f1_proxy": round(token_f1(g["answer"], ref["reference_answer"]), 3) if ref else None,
                     "numeric_fact_proxy": numeric_fact_proxy(g["answer"], ref["required_facts"]) if ref else None,
                     "n_output_tokens": (g.get("usage") or {}).get("completion_tokens"), "finish_reason": g.get("finish_reason"),
                     "reasoning_tokens": ((g.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"), "model_reported": g.get("model_reported")})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "generations_summary.csv", index=False)
    return df


if __name__ == "__main__":
    d = run_deepseek()
    if d is not None:
        print(d.groupby("condition")[["abstained", "n_cited", "cited_relevant_rate", "numeric_fact_proxy", "token_f1_proxy"]].mean().round(3))
