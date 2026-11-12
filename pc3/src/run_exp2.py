"""Experimento 2: el mismo agente W1 (Qwen) con cuatro configuraciones de memoria sobre 3 secuencias encadenadas (9 interacciones).

    M0 sin memoria | M1 working memory (sesion) | M2 memoria episodica persistente | M3 memoria semantica persistente (e5 + FAISS)

Fijos: modelo, decoding, prompt, herramientas, controlador, limites (incluido max_input_tokens y el sub-presupuesto de memoria)
y el orden de las interacciones. Cada condicion empieza con almacenes vacios; las sesiones se simulan reiniciando la working
memory y releyendo del disco los almacenes persistentes (lo que no se persistio, no sobrevive).

    python src/run_exp2.py                   # agente Qwen, ~30-60 min en CPU (EXPERIMENTO OFICIAL)
    python src/run_exp2.py --policy reader   # diagnostico sin LLM (MemoryReaderWorkflow), segundos -> results/exp2/diagnostic_reader/
Salidas: results/exp2/{per_interaction,summary}.csv, results/exp2/memory_store/<cond>/ (JSONL + auditoria), trazas en
results/traces/exp2/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone

import pandas as pd

from agent import QwenAgentPolicy
from config import DATA, RESULTS, TRACES, read_jsonl, sha256_12
from controller import run_task
from evaluation import memory_use, summarize_exp2
from memory import AuditLog, EpisodicStore, SemanticStore, WorkingMemory
from run_exp1 import declare_config
from tools import build_toolbox
from workflow import MemoryReaderWorkflow

OUT = RESULTS / "exp2"
CONDITIONS = ["M0_none", "M1_working", "M2_episodic", "M3_semantic"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["agent", "reader"], default="agent")
    policy = ap.parse_args().policy
    out_dir = OUT if policy == "agent" else OUT / "diagnostic_reader"
    trace_dir = TRACES / ("exp2" if policy == "agent" else "exp2_reader")
    cfg = declare_config()                      # verifica que benchmark y limites no cambiaron
    scenarios = read_jsonl(DATA / "memory_scenarios.jsonl")
    toolbox, retr = build_toolbox()
    agent = QwenAgentPolicy() if policy == "agent" else MemoryReaderWorkflow()
    encode = lambda texts: retr.encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    prices_version = sha256_12(DATA.parent.parent / "parcial" / "data" / "prices" / "SPY.csv")
    rows = []
    t_all = time.perf_counter()
    for cond in CONDITIONS:
        store_dir = out_dir / "memory_store" / cond
        if store_dir.exists():
            shutil.rmtree(store_dir)
        store_dir.mkdir(parents=True)
        audit = AuditLog(store_dir / "memory_ops.jsonl")
        working = WorkingMemory()
        episodic = EpisodicStore(store_dir / "episodic.jsonl", audit)
        semantic = SemanticStore(store_dir / "semantic.jsonl", encode, audit, data_version=f"prices:SPY.csv@{prices_version}")
        session = None
        for turn, scen in enumerate(scenarios, start=1):
            if scen["session"] != session:                       # nueva sesion
                session = scen["session"]
                working.start_session(session)
                episodic.reload()
                semantic.reload()
            if cond == "M1_working":
                shown = working.retrieve(scen, turn)
            elif cond == "M2_episodic":
                shown = episodic.retrieve(scen, turn)
            elif cond == "M3_semantic":
                shown = semantic.retrieve(scen, turn)
            else:
                shown = []
            res = run_task(scen, agent, toolbox, memory=shown,
                           extra_trace={"condition": cond, "session": session, "sequence": scen["sequence"]})
            res["policy"] = f"{agent.name}__{cond}"
            # escritura de memoria (controlador, solo campos validados)
            written = []
            if cond == "M1_working":
                written = [working.write(scen, res, turn)["memory_id"]]
            elif cond == "M2_episodic":
                written = [episodic.write(scen, res, session, turn)["memory_id"]]
            elif cond == "M3_semantic":
                written = [m["memory_id"] for m in semantic.consolidate(scen, res, session, turn)]
            res["memory_written"] = written
            path = trace_dir / f"{cond}__{scen['task_id']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

            sources = {m["memory_id"]: m["provenance"]["interaction_id"] for m in shown}
            mu = memory_use(res, scen, sources)
            ans = res["answer"] or {}
            rows.append({"condition": cond, "task_id": scen["task_id"], "sequence": scen["sequence"], "session": session,
                         **scen["flags"], "stop_reason": res["stop_reason"], "stop_detail": res["stop_detail"],
                         "task_success": mu["task_success"], "termination_correct": mu["termination_correct"],
                         "memory_recall_ok": mu["memory_recall_ok"], "stale_error": mu["stale_error"],
                         "wrong_memory_use": mu["wrong_memory_use"], "gold_shown": mu["gold_shown"],
                         "gold_top1": bool(shown) and shown[0]["provenance"]["interaction_id"] in scen["gold_memory_sources"],
                         "n_memory_shown": len(shown), "memory_shown": ";".join(
                             f"{m['memory_id']}<-{m['provenance']['interaction_id']}({m['status']},v{m.get('version', 1)})" for m in shown),
                         "n_memory_cited": len(mu["memory_cited"]), "memory_cited": ";".join(
                             f"{m}<-{s}" for m, s in zip(mu["memory_cited"], mu["memory_cited_sources"])),
                         "memory_written": ";".join(written),
                         "memory_tokens": max([t.get("memory_tokens", 0) for t in res["trace"]] or [0]),
                         "max_context_tokens": res["max_context_tokens"], "prompt_tokens": res["prompt_tokens"],
                         "output_tokens": res["output_tokens"], "latency_ms": res["latency_ms"], "steps": res["steps"],
                         "tool_calls": res["tool_calls"], "answer_decision": ans.get("decision"),
                         "answer_ticker": ans.get("ticker"), "answer_as_of": ans.get("as_of"),
                         "answer_severity": ans.get("severity")})
            r = rows[-1]
            print(f"{cond:12s} {scen['task_id']} ses={session} stop={r['stop_reason']:15s} ok={r['task_success']!s:5s} "
                  f"mem_shown={r['memory_shown'] or '-'} cited={r['memory_cited'] or '-'} {res['latency_ms'] / 1000:.0f}s")
    per = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    per.to_csv(out_dir / "per_interaction.csv", index=False)
    summ = summarize_exp2(per)
    summ.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "run_manifest.json").write_text(json.dumps({
        "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "declared_at_utc": cfg["declared_at_utc"],
        "policy": agent.name, "conditions": CONDITIONS, "interactions": len(scenarios), "wall_time_s": round(time.perf_counter() - t_all, 1)},
        indent=1), encoding="utf-8")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
