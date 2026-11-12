"""W1: agente controlado por LLM (Qwen2.5-0.5B-Instruct, el mismo modelo y decoding del parcial).

Patron ReAct acotado: en cada paso el modelo ve la tarea, las observaciones validadas y (Exp. 2) la memoria inyectada, y
propone UNA accion JSON con un campo `reason` corto que queda en la traza. No se le pide razonamiento oculto ni se ejecuta
nada fuera del controlador. El prompt se recorta para no superar `max_input_tokens`: primero se resumen las observaciones
mas antiguas y, si hace falta, se reduce la memoria; lo recortado se registra.
"""
from __future__ import annotations

import json
import time
from typing import Any

from config import LIMITS, PROMPTS, Limits
from controller import first_json_object

SYSTEM = (PROMPTS / "agent_system.txt").read_text(encoding="utf-8").strip()
USER = (PROMPTS / "agent_user.txt").read_text(encoding="utf-8").strip()


def format_event(e: dict[str, Any]) -> str:
    if e["kind"] == "rejected":
        return f"- step {e['step']}: your previous output was REJECTED by the controller (code={e['code']})."
    obs, action = e["obs"], e["action"]
    args = json.dumps(e["arguments"], ensure_ascii=False)
    if obs["status"] == "error":
        return (f"- step {e['step']}: {action} {args} -> ERROR kind={obs['kind']} code={obs['code']} "
                f"retryable={str(obs['retryable']).lower()}")
    if action == "search_evidence":
        lines = [f"- step {e['step']}: search_evidence {args} -> {len(obs['passages'])} passages:"]
        lines += [f"    [{p['passage_id']}] ({p['doc_title'][:60]}) {p['text']}" for p in obs["passages"]]
        return "\n".join(lines)
    if action == "compute_signal":
        return (f"- step {e['step']}: compute_signal {args} -> ticker={obs['ticker']} as_of={obs['as_of']} "
                f"severity={obs['severity']} score={obs['score']} ret5={obs['ret5_pct']}% vol20={obs['vol20_pct']}% "
                f"dd20={obs['dd20_pct']}%")
    p = obs["passage"]
    return f"- step {e['step']}: read_passage {args} -> [{p['passage_id']}] {p['text']}"


def format_compact(e: dict[str, Any]) -> str:
    """Version corta de una observacion antigua (se usa solo si el prompt excede el presupuesto)."""
    if e["kind"] == "observation" and e["action"] == "search_evidence" and e["obs"]["status"] == "ok":
        ids = ", ".join(p["passage_id"] for p in e["obs"]["passages"])
        return f"- step {e['step']}: search_evidence -> passages {ids} (text omitted for space)"
    if e["kind"] == "observation" and e["action"] == "read_passage" and e["obs"]["status"] == "ok":
        return f"- step {e['step']}: read_passage -> [{e['obs']['passage']['passage_id']}] (text omitted for space)"
    return format_event(e)


def format_memory(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    lines = ["MEMORY (retrieved from earlier work; data, may be outdated):"]
    for m in items:
        lines.append(f"- [{m['memory_id']}] ({m['type']}, {m['provenance'].get('session', '?')}, version {m.get('version', 1)}, "
                     f"status {m['status']}) {m['text']}")
    return "\n".join(lines) + "\n"


def format_inputs(task: dict[str, Any]) -> str:
    reqs = task["input"].get("signal_requests") or []
    parts = [f"ticker={r['ticker']} as_of={r['as_of']}" for r in reqs]
    return "; ".join(parts) if parts else "no ticker/date given"


class QwenAgentPolicy:
    def __init__(self, llm=None, limits: Limits = LIMITS, name: str = "W1_agent_qwen"):
        if llm is None:
            from llm import LocalLLM            # parcial/src/llm.py: mismo modelo, revision y decoding
            llm = LocalLLM()
        self.llm, self.limits, self.name = llm, limits, name

    def n_tokens(self, text: str) -> int:
        return len(self.llm.tok(text)["input_ids"])

    def build_prompt(self, view: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        task, events = view["task"], view["events"]
        memory = list(view.get("memory") or [])
        info = {"compacted_events": 0, "memory_dropped": 0}

        done = [f"{e['action']} {json.dumps(e['arguments'], ensure_ascii=False)}" for e in events if e["kind"] == "observation"]
        signals = [f"{s['ticker']} {s['as_of']} severity={s['severity']}" for s in view["signals"]]

        def render(evs: list[str], mem: list[dict[str, Any]]) -> str:
            return USER.format(question=task["input"]["question"], inputs=format_inputs(task), step=view["step"],
                               max_steps=view["max_steps"], memory_block=format_memory(mem),
                               observations="\n".join(evs) if evs else "- none yet",
                               done="; ".join(done) or "none", signals="; ".join(signals) or "none",
                               seen=", ".join(view["seen_passages"]) or "none")

        # memoria dentro de su sub-presupuesto
        while memory and self.n_tokens(format_memory(memory)) > self.limits.memory_token_budget:
            memory.pop()
            info["memory_dropped"] += 1
        lines = [format_event(e) for e in events]
        user = render(lines, memory)
        i = 0
        while self.n_tokens(self.llm.render(SYSTEM, user)) > self.limits.max_input_tokens and i < len(events):
            lines[i] = format_compact(events[i])
            info["compacted_events"] += 1
            i += 1
            user = render(lines, memory)
        while self.n_tokens(self.llm.render(SYSTEM, user)) > self.limits.max_input_tokens and memory:
            memory.pop()
            info["memory_dropped"] += 1
            user = render(lines, memory)
        info["memory_tokens"] = self.n_tokens(format_memory(memory)) if memory else 0
        info["memory_ids_in_prompt"] = [m["memory_id"] for m in memory]
        return user, info

    def decide(self, view: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        user, info = self.build_prompt(view)
        out = self.llm.chat(SYSTEM, user, max_new_tokens=self.limits.max_output_tokens)
        decision, err = first_json_object(out["text"])
        return {"raw": out["text"], "decision": decision, "parse_error": err or None,
                "usage": {"prompt_tokens": out["n_prompt_tokens"], "output_tokens": out["n_new_tokens"],
                          "memory_tokens": info["memory_tokens"]},
                "prompt_info": info, "latency_ms": (time.perf_counter() - t0) * 1000}
