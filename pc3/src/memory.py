"""Memoria del agente (Experimento 2): working, episodica y semantica, implementadas con Python, JSONL y FAISS.

Distincion operacional
  * Estado de ejecucion (controller.RunState): contadores y observaciones de UNA tarea; se descarta al terminar. Todas las
    condiciones lo tienen; no es memoria.
  * M1 working memory: resultados de las tareas anteriores de la MISMA sesion; vive en RAM y se borra al cambiar de sesion.
  * M2 episodic memory: un episodio por tarea (tarea, acciones, observaciones validadas, resultado, errores) en un JSONL
    persistente. Se recupera por relevancia + recencia + importancia (Park et al., 2023). Es historia: no se invalida.
  * M3 semantic memory: hechos consolidados (una senal de riesgo por activo y fecha) con procedencia, version y estado,
    indexados con el encoder e5 y FAISS del parcial (Semana 4). Un hecho mas reciente del mismo activo (<= 120 dias)
    invalida (status=superseded) al anterior; los invalidados se conservan para auditoria pero no se sirven.

Politica de escritura (igual para todas): escribe el CONTROLADOR al terminar cada tarea, solo con campos validados
(observaciones de herramientas y la respuesta aceptada). No se guardan prompts, salidas crudas ni el `rationale` libre del
modelo. Toda operacion (write, update, supersede, delete) queda en un registro de auditoria sin contenido.
"""
from __future__ import annotations

import json
import math
import re
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np

from config import read_jsonl, write_jsonl

SUPERSEDE_WINDOW_DAYS = 120
RECENCY_DECAY = 0.9          # por interaccion transcurrida
IMPORTANCE = {"high": 1.0, "medium": 0.7, "low": 0.4, None: 0.3}
WORD_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
STOP = frozenset("the a an of and to in on at is was for with that as or be by this from what which did you we it its "
                 "during after before report again give tell your our same".split())


def words(text: str) -> set[str]:
    return {w for w in WORD_RE.findall(text.lower()) if w not in STOP}


def task_query(task: dict[str, Any]) -> str:
    reqs = " ".join(f"{r['ticker']} {r['as_of']}" for r in task["input"].get("signal_requests") or [])
    return f"{task['input']['question']} {reqs}".strip()


def outcome(result: dict[str, Any]) -> dict[str, Any]:
    """Contenido validado de una tarea: la respuesta aceptada o, si no hubo, la ultima senal observada."""
    ans = result.get("answer")
    if ans and ans["decision"] != "abstain":
        sig = next((s for s in result["signals"] if s["ticker"] == ans["ticker"] and s["as_of"] == ans["as_of"]), None)
        return {"decision": ans["decision"], "ticker": ans["ticker"], "as_of": ans["as_of"], "severity": ans["severity"],
                "evidence_ids": ans["evidence_ids"], "signal": sig}
    last = result["signals"][-1] if result["signals"] else None
    return {"decision": ans["decision"] if ans else None, "ticker": last["ticker"] if last else None,
            "as_of": last["as_of"] if last else None, "severity": last["severity"] if last else None,
            "evidence_ids": [], "signal": last}


def describe(content: dict[str, Any], stop_reason: str) -> str:
    if content["ticker"] is None:
        return f"no signal obtained (stop {stop_reason})"
    dec = content["decision"] or "none"
    ev = ", ".join(content["evidence_ids"]) or "none"
    return (f"{content['ticker']} as of {content['as_of']}: severity {content['severity']}, decision {dec}; "
            f"evidence {ev}; stop {stop_reason}")


class AuditLog:
    def __init__(self, path: Path | None):
        self.path = path

    def log(self, op: str, memory_id: str, **meta: Any) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"op": op, "memory_id": memory_id, **meta}, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- M1

class WorkingMemory:
    def __init__(self, k: int = 3):
        self.items: list[dict[str, Any]] = []
        self.session: str | None = None
        self.k = k

    def start_session(self, session: str) -> None:
        if session != self.session:
            self.items, self.session = [], session

    def write(self, task: dict[str, Any], result: dict[str, Any], turn: int) -> dict[str, Any]:
        content = outcome(result)
        item = {"memory_id": f"W-{task['task_id']}", "type": "working", "status": "active", "version": 1,
                "provenance": {"session": self.session, "interaction_id": task["task_id"], "turn": turn},
                "content": content, "text": f"earlier in this session, task '{task['input']['question'][:90]}': "
                                            f"{describe(content, result['stop_reason'])}"}
        self.items.append(item)
        return item

    def retrieve(self, task: dict[str, Any], turn: int) -> list[dict[str, Any]]:
        return [{**m, "score": 1.0} for m in self.items[-self.k:][::-1]]


# --------------------------------------------------------------------------- M2

class EpisodicStore:
    def __init__(self, path: Path, audit: AuditLog | None = None, k: int = 3):
        self.path, self.audit, self.k = path, audit or AuditLog(None), k
        self.items: list[dict[str, Any]] = read_jsonl(path) if path.exists() else []

    def reload(self) -> None:
        """Al iniciar una sesion se relee del disco: lo que sobrevive es solo lo persistido."""
        self.items = read_jsonl(self.path) if self.path.exists() else []

    def _save(self) -> None:
        write_jsonl(self.path, self.items)

    def write(self, task: dict[str, Any], result: dict[str, Any], session: str, turn: int) -> dict[str, Any]:
        content = outcome(result)
        actions = [f"{t['selected_tool']}({json.dumps(t['arguments'], ensure_ascii=False)})"
                   for t in result["trace"] if t.get("selected_tool") and t["selected_tool"] != "finish"
                   and t.get("decision_status") == "accepted"]
        errors = [f"{t['observation']['kind']}:{t['observation']['code']}" for t in result["trace"]
                  if isinstance(t.get("observation"), dict) and t["observation"].get("status") == "error"]
        mid = f"E{len(self.items) + 1:03d}"
        item = {"memory_id": mid, "type": "episodic", "status": "active", "version": 1,
                "provenance": {"session": session, "interaction_id": task["task_id"], "turn": turn,
                               "tool": "risk.compute_signal" if content["signal"] else None},
                "importance": IMPORTANCE.get(content["severity"], 0.3) if content["decision"] != "abstain" else 0.5,
                "content": {**content, "task": task["input"]["question"][:160], "actions": actions[:6], "errors": errors[:4],
                            "stop_reason": result["stop_reason"]},
                "text": f"episode of task '{task['input']['question'][:90]}': {describe(content, result['stop_reason'])}"}
        self.items.append(item)
        self._save()
        self.audit.log("write", mid, type="episodic", session=session, interaction_id=task["task_id"])
        return item

    def delete(self, memory_id: str) -> bool:
        before = len(self.items)
        self.items = [m for m in self.items if m["memory_id"] != memory_id]
        if len(self.items) < before:
            self._save()
            self.audit.log("delete", memory_id, type="episodic")
            return True
        return False

    def retrieve(self, task: dict[str, Any], turn: int) -> list[dict[str, Any]]:
        if not self.items:
            return []
        q = words(task_query(task))
        rows = []
        for m in self.items:
            w = words(m["content"]["task"] + " " + m["text"])
            relevance = len(q & w) / math.sqrt(max(len(q), 1) * max(len(w), 1))
            recency = RECENCY_DECAY ** max(turn - m["provenance"]["turn"], 0)
            rows.append((m, relevance, recency, m["importance"]))
        # normalizacion min-max por componente (Park et al., 2023) y suma con pesos iguales
        cols = np.array([[r[1], r[2], r[3]] for r in rows], dtype=float)
        span = cols.max(0) - cols.min(0)
        norm = np.where(span > 0, (cols - cols.min(0)) / np.where(span > 0, span, 1), 1.0)
        scores = norm.sum(1)
        order = sorted(range(len(rows)), key=lambda i: (-scores[i], rows[i][0]["memory_id"]))[: self.k]
        return [{**rows[i][0], "score": round(float(scores[i]), 3),
                 "score_parts": {"relevance": round(rows[i][1], 3), "recency": round(rows[i][2], 3),
                                 "importance": rows[i][3]}} for i in order]


# --------------------------------------------------------------------------- M3

class SemanticStore:
    def __init__(self, path: Path, encode: Callable[[list[str]], np.ndarray], audit: AuditLog | None = None, k: int = 3,
                 data_version: str = ""):
        self.path, self.encode, self.audit, self.k, self.data_version = path, encode, audit or AuditLog(None), k, data_version
        self.items: list[dict[str, Any]] = read_jsonl(path) if path.exists() else []
        self._index()

    def reload(self) -> None:
        self.items = read_jsonl(self.path) if self.path.exists() else []
        self._index()

    def _index(self) -> None:
        import faiss
        self.active = [m for m in self.items if m["status"] == "active"]
        self.index = None
        if self.active:
            vecs = self.encode([f"passage: {m['text']}" for m in self.active]).astype("float32")
            self.index = faiss.IndexFlatIP(vecs.shape[1])
            self.index.add(vecs)

    def _save(self) -> None:
        write_jsonl(self.path, self.items)
        self._index()

    def consolidate(self, task: dict[str, Any], result: dict[str, Any], session: str, turn: int) -> list[dict[str, Any]]:
        """Solo tras SUCCESS: un hecho por senal observada. Actualiza version si el hecho cambia e invalida hechos viejos."""
        if result["stop_reason"] != "SUCCESS":
            return []
        ans = result["answer"]
        written = []
        for s in result["signals"]:
            ev = ans["evidence_ids"] if (s["ticker"], s["as_of"]) == (ans["ticker"], ans["as_of"]) else []
            context = task["input"]["question"][:110]
            text = (f"{s['ticker']} risk signal as of {s['as_of']}: severity {s['severity']} (score {s['score']}; 5-day return "
                    f"{s['ret5_pct']}%, 20-day volatility {s['vol20_pct']}%, drawdown {s['dd20_pct']}%). "
                    f"Context: {context} Evidence: {', '.join(ev) or 'none'}.")
            content = {"ticker": s["ticker"], "as_of": s["as_of"], "severity": s["severity"],
                       "decision": "alert" if s["severity"] != "low" else "no_alert", "evidence_ids": ev, "signal": s}
            same = next((m for m in self.items if m["status"] == "active" and m["content"]["ticker"] == s["ticker"]
                         and m["content"]["as_of"] == s["as_of"]), None)
            if same and same["content"]["signal"] == s and same["content"]["evidence_ids"] == ev:
                continue                                  # hecho ya conocido: no se duplica
            if same:                                       # mismo hecho con contenido nuevo -> nueva version
                same["status"], same["superseded_by"] = "superseded", None
            mid = f"F{len(self.items) + 1:03d}"
            item = {"memory_id": mid, "type": "semantic", "status": "active", "version": (same["version"] + 1) if same else 1,
                    "valid_from": s["as_of"],
                    "provenance": {"session": session, "interaction_id": task["task_id"], "turn": turn,
                                   "tool": "risk.compute_signal", "data_version": self.data_version, "passage_ids": ev},
                    "content": content, "text": text}
            if same:
                same["superseded_by"] = mid
                self.audit.log("update", same["memory_id"], new_version=mid)
            for old in self.items:                         # invalidacion temporal: senal mas reciente del mismo activo
                if (old["status"] == "active" and old["content"]["ticker"] == s["ticker"]
                        and old["content"]["as_of"] < s["as_of"]
                        and (date.fromisoformat(s["as_of"]) - date.fromisoformat(old["content"]["as_of"])).days
                        <= SUPERSEDE_WINDOW_DAYS):
                    old["status"], old["superseded_by"], old["valid_to"] = "superseded", mid, s["as_of"]
                    self.audit.log("supersede", old["memory_id"], by=mid)
            self.items.append(item)
            self.audit.log("write", mid, type="semantic", session=session, interaction_id=task["task_id"])
            written.append(item)
        if written:
            self._save()
        return written

    def delete(self, memory_id: str) -> bool:
        before = len(self.items)
        self.items = [m for m in self.items if m["memory_id"] != memory_id]
        if len(self.items) < before:
            self._save()
            self.audit.log("delete", memory_id, type="semantic")
            return True
        return False

    def retrieve(self, task: dict[str, Any], turn: int) -> list[dict[str, Any]]:
        if self.index is None:
            return []
        q = self.encode([f"query: {task_query(task)}"]).astype("float32")
        scores, idx = self.index.search(q, min(self.k, len(self.active)))
        return [{**self.active[int(i)], "score": round(float(s), 3)} for s, i in zip(scores[0], idx[0])]

    def superseded(self) -> list[dict[str, Any]]:
        return [m for m in self.items if m["status"] == "superseded"]
