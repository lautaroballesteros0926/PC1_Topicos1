"""Experimento 1: W0 (workflow fijo) frente a W1 (agente Qwen) sobre las 15 tareas, mas las pruebas de robustez guionizadas.

    python src/run_exp1.py              # W0 + W1 + pruebas (Qwen en CPU: ~10-20 min)
    python src/run_exp1.py --only W0    # solo el workflow y las pruebas (sin LLM)

Salidas: results/declared_config.json (antes de ejecutar), results/exp1/{per_task,summary,robustness}.csv,
results/exp1/run_manifest.json y una traza JSON por tarea y politica en results/traces/exp1/.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone

import pandas as pd

from config import (DATA, LIMITS, RESULTS, RETRIEVAL_CONDITION, STOP_REASONS, TRACES, data_hashes, read_jsonl,
                    write_jsonl)
from controller import run_task
from evaluation import exp1_row, summarize_exp1
from tools import build_toolbox
from workflow import FixedPlanWorkflow, ScriptedPolicy

OUT = RESULTS / "exp1"


def declare_config() -> dict:
    """Se escribe ANTES de ejecutar: limites, modelo, recuperador y hashes del benchmark congelado."""
    from llm import DECODING, LLM_ID, LLM_REVISION
    path = RESULTS / "declared_config.json"
    if path.exists():
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if cfg["data_hashes"] != data_hashes() or cfg["limits"] != LIMITS.as_dict():
            raise SystemExit("El benchmark o los limites cambiaron despues de declararlos: revisar antes de ejecutar.")
        return cfg
    cfg = {"declared_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "limits": LIMITS.as_dict(),
           "stop_reasons": STOP_REASONS, "retrieval": RETRIEVAL_CONDITION,
           "llm": {"id": LLM_ID, "revision": LLM_REVISION, "decoding": {**DECODING, "max_new_tokens": LIMITS.max_output_tokens}},
           "policies": {"W0": "FixedPlanWorkflow (sin LLM)", "W1": "QwenAgentPolicy"}, "data_hashes": data_hashes()}
    path.write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
    return cfg


def save_trace(folder: str, res: dict) -> None:
    path = TRACES / folder / f"{res['policy']}__{res['task_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


PROBES = [  # pruebas guionizadas del controlador: NO son ejecuciones de un LLM
    ("R1_repeated_action", None, [{"action": "search_evidence", "arguments": {"query": "Silicon Valley Bank failure", "top_k": 2}},
                                  {"action": "search_evidence", "arguments": {"query": "Silicon Valley Bank failure", "top_k": 2}}],
     "REPEATED_ACTION"),
    ("R2_budget_exhausted", None, [{"action": "search_evidence", "arguments": {"query": q, "top_k": 1}} for q in
                                   ["COVID-19 stock market crash March 2020", "Silicon Valley Bank deposits run",
                                    "carry trade unwind August 2024 yen", "April 2025 tariffs stock market crash",
                                    "2022 bear market inflation", "VIX volatility index calculation"]],
     "MAX_STEPS"),
    ("R3_invalid_output", None, ["Sure! The alert should be high.", "I think SPY is risky"], "INVALID_OUTPUT"),
    ("R4_unsupported_finish", None, [{"action": "finish", "arguments": {"decision": "alert", "ticker": "SPY", "as_of": "2020-03-16",
                                                                         "severity": "high", "evidence_ids": ["D99-P99"]}},
                                     {"action": "finish", "arguments": {"decision": "alert", "ticker": "SPY", "as_of": "2020-03-16",
                                                                         "severity": "high", "evidence_ids": []}}],
     "INVALID_OUTPUT"),
    ("R5_transient_errors", {"kind": "transient", "fail_times": 50, "note": "INYECTADO"},
     [{"action": "compute_signal", "arguments": {"ticker": "SPY", "as_of": "2020-03-16"}},
      {"action": "compute_signal", "arguments": {"ticker": "SPY", "as_of": "2020-03-13"}}], "TOOL_ERROR"),
    ("R6_no_progress", None, [{"action": "search_evidence", "arguments": {"query": "Silicon Valley Bank failure", "top_k": 3}},
                              {"action": "search_evidence", "arguments": {"query": "Silicon Valley Bank failure", "top_k": 2}},
                              {"action": "search_evidence", "arguments": {"query": "Silicon Valley Bank failure", "top_k": 1}}],
     "NO_PROGRESS"),
    ("R7_tool_timeout", {"kind": "slow", "delay_s": 1.0, "note": "INYECTADO: cada lectura tarda 1 s (> 0.5 s por intento)"},
     [{"action": "compute_signal", "arguments": {"ticker": "SPY", "as_of": "2020-03-16"}},
      {"action": "compute_signal", "arguments": {"ticker": "SPY", "as_of": "2020-03-13"}}], "TIMEOUT"),
]


def run_probes(toolbox) -> pd.DataFrame:
    rows = []
    for pid, fault, script, expected_stop in PROBES:
        task = {"task_id": pid, "input": {"question": "controller robustness probe", "signal_requests": []},
                "injected_fault": fault}
        res = run_task(task, ScriptedPolicy(script, name="scripted_probe"), toolbox)
        save_trace("exp1", res)
        emitted = bool(res["answer"]) and res["answer"]["decision"] != "abstain"
        rows.append({"probe": pid, "expected_stop": expected_stop, "stop_reason": res["stop_reason"],
                     "stop_detail": res["stop_detail"], "ok": res["stop_reason"] == expected_stop and not emitted,
                     "answer_emitted": emitted, "steps": res["steps"], "tool_calls": res["tool_calls"],
                     "tool_errors": res["tool_errors"], "trace_steps": len(res["trace"]), "latency_ms": res["latency_ms"]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["W0", "W1"], default=None)
    args = ap.parse_args()
    cfg = declare_config()
    tasks = read_jsonl(DATA / "tasks.jsonl")
    expected = {e["task_id"]: e for e in read_jsonl(DATA / "expected_results.jsonl")}
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    toolbox, _ = build_toolbox()
    print(f"recuperador C listo en {time.perf_counter() - t0:.1f} s")

    policies = []
    if args.only in (None, "W0"):
        policies.append(FixedPlanWorkflow())
    if args.only in (None, "W1"):
        from agent import QwenAgentPolicy
        policies.append(QwenAgentPolicy())

    rows, results = [], []
    for pol in policies:
        for task in tasks:
            res = run_task(task, pol, toolbox)
            save_trace("exp1", res)
            results.append({k: v for k, v in res.items() if k != "trace"})
            row = exp1_row(task, expected[task["task_id"]], res)
            rows.append(row)
            print(f"{pol.name:14s} {task['task_id']} stop={res['stop_reason']:15s} ok={row['task_success']!s:5s} "
                  f"steps={res['steps']} calls={res['tool_calls']} {res['latency_ms'] / 1000:.1f}s")
    per_task = pd.DataFrame(rows)
    prev = OUT / "per_task.csv"
    if args.only and prev.exists():            # conservar la otra politica si ya existia
        old = pd.read_csv(prev)
        per_task = pd.concat([old[~old.policy.isin(per_task.policy.unique())], per_task], ignore_index=True)
    per_task.to_csv(prev, index=False)
    summarize_exp1(per_task).to_csv(OUT / "summary.csv", index=False)
    write_jsonl(OUT / f"results_{args.only or 'all'}.jsonl", results)

    probes = run_probes(toolbox)
    probes.to_csv(OUT / "robustness.csv", index=False)
    print(probes[["probe", "expected_stop", "stop_reason", "ok"]].to_string(index=False))

    manifest = {"finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "python": platform.python_version(), "declared_config": "results/declared_config.json",
                "declared_at_utc": cfg["declared_at_utc"], "policies": [p.name for p in policies],
                "wall_time_s": round(time.perf_counter() - t0, 1)}
    try:
        import torch, transformers, sentence_transformers, faiss  # noqa: E401
        manifest["packages"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                                "sentence_transformers": sentence_transformers.__version__, "faiss": faiss.__version__}
    except ImportError:
        pass
    (OUT / f"run_manifest_{args.only or 'all'}.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(summarize_exp1(per_task).to_string(index=False))


if __name__ == "__main__":
    main()
