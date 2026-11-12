"""Criterios de exito y metricas de la PC3 (definidos antes de ejecutar; denominadores explicitos).

Experimento 1 (por tarea)
  task_success          resultado final correcto segun expected_results.jsonl (ver `score_task`)
  termination_correct   stop_reason dentro de allowed_stops de la tarea
  steps, tool_calls, tool_errors, repeated_actions, latency_ms, prompt/output tokens: de la traza
  safe_failure          (solo tareas de fallo o abstencion) no se emitio alert/no_alert

Experimento 2 (por interaccion; denominadores)
  task_success              / todas las interacciones (9)
  memory_recall_accuracy    / interacciones con needs_memory (4): exito Y cita un recuerdo cuya fuente esta en gold_memory_sources
  cross_session_retention   / interacciones con cross_session (3): mismo criterio
  stale_memory_error_rate   / interacciones con stale_risk (2): responde con la fecha o severidad del recuerdo obsoleto o cita un
                              recuerdo cuya fuente esta en stale_sources
  wrong_memory_use_rate     / interacciones en que el agente cito >= 1 recuerdo: cita alguno cuya fuente NO esta en gold
  gold_retrieved_at3 / top1 / interacciones con needs_memory: el recuerdo correcto estuvo entre los mostrados / fue el primero
                            (mide la MEMORIA, no al agente; se anadio despues de ver la corrida con Qwen)
  context_tokens, latency_ms: medias por interaccion
Una respuesta correcta sin citar el recuerdo correcto cuenta en task_success, pero NO como recuperacion de memoria.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def score_answer(answer: dict[str, Any] | None, stop_reason: str, expected: dict[str, Any]) -> dict[str, Any]:
    term_ok = stop_reason in expected["allowed_stops"]
    if expected["decision"] == "abstain":
        emitted = bool(answer) and answer["decision"] != "abstain"
        return {"task_success": term_ok and not emitted, "termination_correct": term_ok, "decision_ok": not emitted,
                "fields_ok": None, "evidence_ok": None}
    if not answer or stop_reason != "SUCCESS":
        return {"task_success": False, "termination_correct": term_ok, "decision_ok": False, "fields_ok": False,
                "evidence_ok": False if expected["evidence_any_of"] else None}
    decision_ok = answer["decision"] == expected["decision"]
    fields_ok = all(answer.get(k) == expected[k] for k in ("ticker", "as_of", "severity"))
    if expected["evidence_any_of"]:
        evidence_ok = bool(set(answer["evidence_ids"]) & set(expected["evidence_any_of"]))
    else:
        evidence_ok = None
    ok = term_ok and decision_ok and fields_ok and (evidence_ok is not False)
    return {"task_success": ok, "termination_correct": term_ok, "decision_ok": decision_ok, "fields_ok": fields_ok,
            "evidence_ok": evidence_ok}


def exp1_row(task: dict[str, Any], expected: dict[str, Any], res: dict[str, Any]) -> dict[str, Any]:
    s = score_answer(res["answer"], res["stop_reason"], expected)
    ans = res["answer"] or {}
    return {"task_id": task["task_id"], "category": task["category"], "policy": res["policy"], **s,
            "safe_failure": (not ans or ans.get("decision") == "abstain") if task["category"] != "ordinary" else None,
            "stop_reason": res["stop_reason"], "stop_detail": res["stop_detail"], "steps": res["steps"],
            "tool_calls": res["tool_calls"], "tool_errors": res["tool_errors"], "invalid_outputs": res["invalid_outputs"],
            "repeated_actions": res["repeated_actions"], "prompt_tokens": res["prompt_tokens"],
            "output_tokens": res["output_tokens"], "tokens": res["prompt_tokens"] + res["output_tokens"],
            "max_context_tokens": res["max_context_tokens"], "latency_ms": res["latency_ms"],
            "answer_decision": ans.get("decision"), "answer_ticker": ans.get("ticker"), "answer_as_of": ans.get("as_of"),
            "answer_severity": ans.get("severity"), "answer_evidence": ",".join(ans.get("evidence_ids", []))}


def summarize_exp1(per_task: pd.DataFrame) -> pd.DataFrame:
    g = per_task.groupby("policy")
    out = pd.DataFrame({
        "tasks": g.size(),
        "task_success": g["task_success"].mean(),
        "termination_correct": g["termination_correct"].mean(),
        "steps_mean": g["steps"].mean(), "tool_calls_mean": g["tool_calls"].mean(),
        "tool_errors_total": g["tool_errors"].sum(), "invalid_outputs_total": g["invalid_outputs"].sum(),
        "repeated_actions_total": g["repeated_actions"].sum(),
        "latency_ms_mean": g["latency_ms"].mean(), "latency_ms_median": g["latency_ms"].median(),
        "prompt_tokens_total": g["prompt_tokens"].sum(), "output_tokens_total": g["output_tokens"].sum(),
        "tokens_mean_per_task": g["tokens"].mean(), "max_context_tokens": g["max_context_tokens"].max()})
    return out.reset_index()


def memory_use(res: dict[str, Any], scen: dict[str, Any], shown_sources: dict[str, str]) -> dict[str, Any]:
    """shown_sources: memory_id -> interaction_id de origen de cada recuerdo mostrado en esta interaccion."""
    ans = res["answer"] or {}
    cited = ans.get("memory_ids", []) or []
    cited_sources = [shown_sources.get(m) for m in cited]
    gold, stale = set(scen["gold_memory_sources"]), set(scen["stale_sources"])
    s = score_answer(res["answer"], res["stop_reason"], scen["expected"])
    used_gold = any(src in gold for src in cited_sources)
    exp = scen["expected"]
    stale_answer = False
    if scen["flags"]["stale_risk"] and ans and ans.get("decision") != "abstain":
        stale_answer = ans.get("as_of") != exp["as_of"] and ans.get("ticker") == exp["ticker"] or \
            (ans.get("as_of") == exp["as_of"] and ans.get("severity") != exp["severity"])
    return {**s, "memory_cited": cited, "memory_cited_sources": cited_sources,
            "gold_shown": any(src in gold for src in shown_sources.values()),
            "memory_recall_ok": bool(s["task_success"] and used_gold),
            "stale_error": bool(stale_answer or any(src in stale for src in cited_sources)),
            "wrong_memory_use": bool(cited) and any(src not in gold for src in cited_sources)}


def summarize_exp2(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cond, g in rows.groupby("condition"):
        need, cross, stale = g[g.needs_memory], g[g.cross_session], g[g.stale_risk]
        cited = g[g.n_memory_cited > 0]
        out.append({"condition": cond, "interactions": len(g),
                    "task_success": g.task_success.mean(),
                    "memory_recall_accuracy": need.memory_recall_ok.mean(), "n_needs_memory": len(need),
                    "gold_retrieved_at3": need.gold_shown.mean(), "gold_retrieved_top1": need.gold_top1.mean(),
                    "cross_session_retention": cross.memory_recall_ok.mean(), "n_cross_session": len(cross),
                    "stale_memory_error_rate": stale.stale_error.mean(), "n_stale_risk": len(stale),
                    "wrong_memory_use_rate": cited.wrong_memory_use.mean() if len(cited) else np.nan,
                    "n_cited_memory": len(cited),
                    "context_tokens_mean": g.max_context_tokens.mean(), "memory_tokens_mean": g.memory_tokens.mean(),
                    "latency_ms_mean": g.latency_ms.mean(), "steps_mean": g.steps.mean(),
                    "tool_calls_mean": g.tool_calls.mean()})
    return pd.DataFrame(out)
