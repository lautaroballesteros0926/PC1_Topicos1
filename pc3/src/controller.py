"""Controlador determinista comun a W0 (workflow) y W1 (agente LLM): Perceive -> Decide -> Act -> Observe.

Lo que decide la POLITICA (reglas fijas en W0, el LLM en W1): la siguiente accion, sus argumentos, cuando terminar y la
respuesta final. Lo que decide el CONTROLADOR (siempre codigo): si la accion esta en la allowlist y cumple su contrato,
si quedan pasos, llamadas, errores y tiempo, si la accion es repetida o no aporta informacion, que observacion validada
entra al estado y si la respuesta final esta sustentada. El controlador termina siempre con un `stop_reason` de
config.STOP_REASONS y conserva la traza completa (acciones rechazadas incluidas).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from config import LIMITS, Limits

ACTIONS = ("search_evidence", "compute_signal", "read_passage", "finish")
FINISH_KEYS = {"decision", "ticker", "as_of", "severity", "evidence_ids", "memory_ids", "rationale"}


class Policy(Protocol):
    name: str

    def decide(self, view: dict[str, Any]) -> dict[str, Any]:
        """Devuelve {"raw": texto o None, "decision": dict o None, "usage": {...}, "latency_ms": float}."""


@dataclass
class RunState:
    """Estado de ejecucion de UNA tarea (no es memoria: se descarta al terminar)."""
    task_id: str
    step: int = 0
    tool_calls: int = 0
    invalid_outputs: int = 0
    consecutive_tool_errors: int = 0
    tool_errors: int = 0
    no_progress: int = 0
    repeated_actions: int = 0
    signatures: list[str] = field(default_factory=list)
    seen_passages: list[str] = field(default_factory=list)
    signals: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)        # observaciones validadas y rechazos, en orden
    trace: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    max_context_tokens: int = 0
    stop_reason: str | None = None
    stop_detail: str = ""
    answer: dict[str, Any] | None = None


def signature(action: str, args: Any) -> str:
    return json.dumps({"a": action, "args": args}, sort_keys=True, ensure_ascii=False, default=str)


def validate_decision(decision: Any) -> tuple[str, dict[str, Any], str]:
    """Contrato de la decision del modelo: {action, arguments, reason?}. Devuelve (action, args, reason) o lanza ValueError."""
    if not isinstance(decision, dict):
        raise ValueError("not_json_object")
    extra = set(decision) - {"action", "arguments", "reason"}
    if extra:
        raise ValueError("unexpected_keys")
    action, args = decision.get("action"), decision.get("arguments")
    if action not in ACTIONS:
        raise ValueError("action_not_allowed")
    if not isinstance(args, dict):
        raise ValueError("arguments_not_object")
    reason = decision.get("reason", "")
    return action, args, reason if isinstance(reason, str) else ""


def check_finish(args: dict[str, Any], state: RunState, memory: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    """Valida la respuesta final contra lo OBSERVADO. Devuelve (respuesta normalizada, "") o (None, codigo de rechazo)."""
    if set(args) - FINISH_KEYS:
        return None, "finish_unexpected_keys"
    decision = args.get("decision")
    if decision not in ("alert", "no_alert", "abstain"):
        return None, "finish_bad_decision"
    ev = args.get("evidence_ids") or []
    mem_ids = args.get("memory_ids") or []
    if not isinstance(ev, list) or not isinstance(mem_ids, list) or not all(isinstance(x, str) for x in ev + mem_ids):
        return None, "finish_bad_lists"
    shown = {m["memory_id"]: m for m in memory}
    if any(m not in shown for m in mem_ids):
        return None, "finish_unknown_memory_id"
    mem_evidence = {p for m in mem_ids for p in shown[m]["content"].get("evidence_ids", [])}
    if any(p not in state.seen_passages and p not in mem_evidence for p in ev):
        return None, "finish_unobserved_evidence"
    answer = {"decision": decision, "ticker": args.get("ticker"), "as_of": args.get("as_of"), "severity": args.get("severity"),
              "evidence_ids": list(dict.fromkeys(ev)), "memory_ids": list(dict.fromkeys(mem_ids)),
              "rationale": str(args.get("rationale", ""))[:400]}
    if decision == "abstain":
        return answer, ""
    t, d, s = answer["ticker"], answer["as_of"], answer["severity"]
    if s not in ("low", "medium", "high"):
        return None, "finish_bad_severity"
    observed = any(x["ticker"] == t and x["as_of"] == d and x["severity"] == s for x in state.signals)
    remembered = any(shown[m]["content"].get("ticker") == t and shown[m]["content"].get("as_of") == d
                     and shown[m]["content"].get("severity") == s for m in mem_ids)
    if not (observed or remembered):
        return None, "finish_severity_not_supported"
    if decision == "alert" and s == "low":
        return None, "finish_alert_on_low_signal"
    if decision == "no_alert" and s != "low":
        return None, "finish_no_alert_on_risk_signal"
    if decision == "alert" and not ev and not mem_evidence:   # una alerta necesita evidencia observada o la de un recuerdo citado
        return None, "finish_alert_without_evidence"
    return answer, ""


def run_task(task: dict[str, Any], policy: Policy, toolbox, limits: Limits = LIMITS,
             memory: list[dict[str, Any]] | None = None, extra_trace: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ejecuta una tarea con presupuestos fijos. Nunca lanza: siempre devuelve resultado + traza + stop_reason."""
    memory = memory or []
    toolbox.reset(task.get("injected_fault"))
    state = RunState(task_id=task["task_id"])
    t_begin = time.perf_counter()

    def elapsed_s() -> float:
        return time.perf_counter() - t_begin

    def stop(reason: str, detail: str = "") -> dict[str, Any]:
        state.stop_reason, state.stop_detail = reason, detail
        if state.trace:
            state.trace[-1]["stop_reason"] = reason
        else:
            state.trace.append({"task_id": state.task_id, "step": 0, "selected_tool": None, "arguments": {},
                                "observation_status": None, "state_updated": False, "tokens": 0, "latency_ms": 0,
                                "stop_reason": reason})
        return {"task_id": state.task_id, "policy": policy.name, "stop_reason": reason, "stop_detail": detail,
                "answer": state.answer, "steps": state.step, "tool_calls": state.tool_calls, "tool_errors": state.tool_errors,
                "invalid_outputs": state.invalid_outputs, "repeated_actions": state.repeated_actions,
                "prompt_tokens": state.prompt_tokens, "output_tokens": state.output_tokens,
                "max_context_tokens": state.max_context_tokens, "latency_ms": round(elapsed_s() * 1000, 1),
                "seen_passages": state.seen_passages, "signals": state.signals,
                "memory_shown": [{k: m[k] for k in ("memory_id", "type", "status", "version", "provenance", "score")
                                  if k in m} for m in memory],
                "injected_fault": task.get("injected_fault"), **(extra_trace or {}), "trace": state.trace}

    while True:
        # ---- Perceive: presupuestos antes de decidir
        if elapsed_s() >= limits.total_deadline_s:
            return stop("TIMEOUT", "total_deadline")
        if state.step >= limits.max_steps:
            return stop("MAX_STEPS", "max_steps")
        state.step += 1
        view = {"task": task, "step": state.step, "max_steps": limits.max_steps, "events": list(state.events),
                "seen_passages": list(state.seen_passages), "signals": list(state.signals), "memory": memory,
                "limits": limits}
        # ---- Decide (politica)
        out = policy.decide(view)
        usage = out.get("usage") or {}
        p_tok, o_tok = int(usage.get("prompt_tokens", 0)), int(usage.get("output_tokens", 0))
        state.prompt_tokens += p_tok
        state.output_tokens += o_tok
        state.max_context_tokens = max(state.max_context_tokens, p_tok)
        rec = {"task_id": state.task_id, "step": state.step, "selected_tool": None, "arguments": {}, "reason": "",
               "decision_status": "accepted", "observation_status": None, "state_updated": False,
               "tokens": {"prompt": p_tok, "output": o_tok}, "context_tokens": p_tok,
               "memory_tokens": int(usage.get("memory_tokens", 0)), "latency_ms": round(float(out.get("latency_ms", 0.0)), 1),
               "raw_output": (out.get("raw") or "")[:600] if out.get("raw") is not None else None, "stop_reason": None}
        state.trace.append(rec)
        try:
            if p_tok > limits.max_input_tokens:
                raise ValueError("context_budget_exceeded")
            if out.get("decision") is None:
                raise ValueError(out.get("parse_error") or "unparseable_output")
            action, args, reason = validate_decision(out["decision"])
        except ValueError as exc:
            state.invalid_outputs += 1
            rec.update(decision_status="rejected", observation_status="rejected", error=str(exc))
            state.events.append({"step": state.step, "kind": "rejected", "code": str(exc)})
            if state.invalid_outputs >= limits.max_invalid_outputs:
                return stop("INVALID_OUTPUT", str(exc))
            continue
        rec.update(selected_tool=action, arguments=args, reason=reason[:300])
        if elapsed_s() >= limits.total_deadline_s:
            return stop("TIMEOUT", "total_deadline")

        # ---- finish: validacion de la respuesta estructurada
        if action == "finish":
            answer, code = check_finish(args, state, memory)
            if answer is None:
                state.invalid_outputs += 1
                rec.update(decision_status="rejected", observation_status="rejected", error=code)
                state.events.append({"step": state.step, "kind": "rejected", "code": code})
                if state.invalid_outputs >= limits.max_invalid_outputs:
                    return stop("INVALID_OUTPUT", code)
                continue
            state.answer = answer
            rec.update(observation_status="final", state_updated=True)
            return stop("ABSTAIN" if answer["decision"] == "abstain" else "SUCCESS")

        # ---- Act: presupuestos y repeticion
        if state.tool_calls >= limits.max_tool_calls:
            rec.update(decision_status="denied_budget")
            return stop("MAX_STEPS", "max_tool_calls")
        sig = signature(action, args)
        if sig in state.signatures:
            state.repeated_actions += 1
            rec.update(decision_status="repeated")
            return stop("REPEATED_ACTION", action)
        state.signatures.append(sig)
        state.tool_calls += 1
        res = toolbox.execute(action, args, set(state.seen_passages))
        obs = res["observation"]
        rec["latency_ms"] = round(rec["latency_ms"] + res["latency_ms"], 1)
        rec["tool_latency_ms"] = res["latency_ms"]
        rec["observation_status"] = obs["status"]

        # ---- Observe: actualizar el estado solo con observaciones validadas
        if obs["status"] == "error":
            state.tool_errors += 1
            state.consecutive_tool_errors += 1
            rec["observation"] = obs
            state.events.append({"step": state.step, "kind": "observation", "action": action, "arguments": args, "obs": obs})
            if state.consecutive_tool_errors >= limits.max_tool_errors:
                return stop("TIMEOUT" if obs["kind"] == "timeout" else "TOOL_ERROR", f'{obs["kind"]}:{obs["code"]}')
            continue
        state.consecutive_tool_errors = 0
        new_info = False
        if action == "search_evidence":
            for p in obs["passages"]:
                if p["passage_id"] not in state.seen_passages:
                    state.seen_passages.append(p["passage_id"])
                    new_info = True
            rec["observation"] = {"passage_ids": [p["passage_id"] for p in obs["passages"]]}
        elif action == "compute_signal":
            sig_obs = {k: obs[k] for k in ("ticker", "as_of", "severity", "score", "ret5_pct", "vol20_pct", "dd20_pct")}
            if sig_obs not in state.signals:
                state.signals.append(sig_obs)
                new_info = True
            rec["observation"] = {**sig_obs, "attempts": obs["attempts"]}
        else:  # read_passage: aporta texto completo (cuenta como progreso la primera vez)
            new_info = True
            rec["observation"] = {"passage_id": obs["passage"]["passage_id"]}
        rec["state_updated"] = new_info
        state.events.append({"step": state.step, "kind": "observation", "action": action, "arguments": args, "obs": obs})
        state.no_progress = 0 if new_info else state.no_progress + 1
        if state.no_progress >= limits.no_progress_patience:
            return stop("NO_PROGRESS", action)


def strip_trace(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k != "trace"}


def first_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    """Primer objeto JSON balanceado del texto (acepta ```json ...```). Sin reparaciones heuristicas del contenido."""
    if not text:
        return None, "empty_output"
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        return (obj, "") if isinstance(obj, dict) else (None, "not_json_object")
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None, "unparseable_output" if re.search(r"\{", text) else "no_json_object"
