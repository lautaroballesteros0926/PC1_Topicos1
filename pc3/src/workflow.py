"""W0: workflow determinista (plan fijo, sin LLM) ejecutado por el MISMO controlador que el agente.

    entrada validada -> plan fijo -> search_evidence(pregunta, top_k=3) -> compute_signal(cada par pedido)
    -> evaluacion por reglas -> respuesta estructurada -> validacion final (controlador)

Reglas de evaluacion (fijadas antes de los experimentos):
  * si alguna llamada a compute_signal devolvio error -> abstain (no se repara ni se reintenta: eso lo hace la herramienta);
  * si no hay senal -> abstain;
  * se toma la senal de mayor score (empate: la primera pedida);
  * severity medium/high y al menos un passage recuperado -> alert; low -> no_alert;
  * evidence_ids = los 3 primeros passages recuperados (el workflow no juzga su pertinencia).
No usa tokens: su costo es solo el de las herramientas.
"""
from __future__ import annotations

import time
from typing import Any


class FixedPlanWorkflow:
    name = "W0_workflow"

    def decide(self, view: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        task = view["task"]
        done = [e for e in view["events"] if e["kind"] == "observation"]
        searched = any(e["action"] == "search_evidence" for e in done)
        requests = task["input"]["signal_requests"]
        attempted = {(e["arguments"].get("ticker"), e["arguments"].get("as_of")) for e in done if e["action"] == "compute_signal"}
        errors = [e for e in done if e["action"] == "compute_signal" and e["obs"]["status"] == "error"]

        if not searched:
            decision = {"action": "search_evidence", "arguments": {"query": task["input"]["question"], "top_k": 3},
                        "reason": "plan step 1: retrieve evidence for the question"}
        elif any((r["ticker"], r["as_of"]) not in attempted for r in requests):
            r = next(r for r in requests if (r["ticker"], r["as_of"]) not in attempted)
            decision = {"action": "compute_signal", "arguments": {"ticker": r["ticker"], "as_of": r["as_of"]},
                        "reason": "plan step 2: compute the requested signal"}
        elif errors or not view["signals"]:
            code = errors[0]["obs"]["code"] if errors else "no_signal"
            decision = {"action": "finish", "arguments": {"decision": "abstain", "evidence_ids": [],
                                                          "rationale": f"signal tool failed ({code}); no supported decision"},
                        "reason": "rule: tool error -> abstain"}
        else:
            best = max(view["signals"], key=lambda s: s["score"])     # max() conserva el primero en caso de empate
            evidence = view["seen_passages"][:3]
            dec = "alert" if best["severity"] in ("medium", "high") and evidence else "no_alert"
            decision = {"action": "finish", "arguments": {
                "decision": dec, "ticker": best["ticker"], "as_of": best["as_of"], "severity": best["severity"],
                "evidence_ids": evidence if dec == "alert" else [],
                "rationale": f"rule: {best['ticker']} {best['as_of']} severity {best['severity']} (score {best['score']})"},
                "reason": "rule: severity >= medium -> alert, low -> no_alert"}
        return {"raw": None, "decision": decision, "usage": {"prompt_tokens": 0, "output_tokens": 0},
                "latency_ms": (time.perf_counter() - t0) * 1000}


class MemoryReaderWorkflow(FixedPlanWorkflow):
    """DIAGNOSTICO del Exp. 2 (anadido tras ver que Qwen no usa la memoria; no reemplaza a M0..M3 con el agente).

    Sin LLM: si la tarea trae ticker y fecha, sigue el plan de W0; si no, confia en el PRIMER recuerdo mostrado (ranking
    de la memoria) cuyo activo aparece en la pregunta y que tenga severidad, y termina citandolo; sin recuerdo -> abstain.
    Asi se mide la calidad de la memoria (que recupera y en que orden) aislada de la capacidad del LLM.
    """
    name = "W0_memory_reader"

    def decide(self, view: dict[str, Any]) -> dict[str, Any]:
        task = view["task"]
        if task["input"]["signal_requests"]:
            return super().decide(view)
        t0 = time.perf_counter()
        q = task["input"]["question"]
        cand = [m for m in view["memory"] if m["content"].get("ticker") and m["content"]["ticker"] in q
                and m["content"].get("severity")]
        if cand:
            c = cand[0]["content"]
            dec = "alert" if c["severity"] != "low" else "no_alert"
            args = {"decision": dec, "ticker": c["ticker"], "as_of": c["as_of"], "severity": c["severity"],
                    "evidence_ids": [], "memory_ids": [cand[0]["memory_id"]], "rationale": "rule: top-ranked matching memory"}
        else:
            args = {"decision": "abstain", "evidence_ids": [], "rationale": "rule: no matching memory"}
        return {"raw": None, "decision": {"action": "finish", "arguments": args, "reason": "rule: answer from memory"},
                "usage": {"prompt_tokens": 0, "output_tokens": 0}, "latency_ms": (time.perf_counter() - t0) * 1000}


class ScriptedPolicy:
    """Secuencia guionizada para pruebas de robustez del controlador. NO es un LLM ni un agente."""

    def __init__(self, actions: list[Any], name: str = "scripted_probe"):
        self.actions, self.cursor, self.name = list(actions), 0, name

    def decide(self, view: dict[str, Any]) -> dict[str, Any]:
        item = self.actions[min(self.cursor, len(self.actions) - 1)]
        self.cursor += 1
        if isinstance(item, str):                 # salida cruda (p. ej. texto no JSON)
            return {"raw": item, "decision": None, "parse_error": "unparseable_output", "usage": {}, "latency_ms": 0.0}
        return {"raw": None, "decision": item, "usage": {}, "latency_ms": 0.0}
