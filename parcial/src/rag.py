"""Experimento 2: construccion de contexto, parseo de citas y metricas AUTOMATICAS de apoyo.

Las metricas de este modulo NO sustituyen la auditoria manual (correccion, citas, grounding): son proxies de depuracion
y comprobaciones mecanicas (por ejemplo, si un id citado existe en el contexto o en los qrels).
"""
from __future__ import annotations

import json
import re
from typing import Any

from common import PROMPTS

CITE_RE = re.compile(r"\bD\d{2}-P\d{2}\b")
ABSTAIN_PATTERNS = ("insufficient evidence", "i don't know", "i do not know")
STOP = frozenset({"the", "a", "an", "of", "and", "to", "in", "on", "at", "is", "was", "were", "it", "its", "by", "for", "with",
                  "that", "as", "or", "be", "has", "had", "have", "this", "from"})


def load_prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8").strip()


def build_evidence(ranked_chunks: list[dict], top_k: int, passages: dict[str, str]) -> tuple[str, list[str]]:
    """Passages (sin duplicados, en orden de aparicion) de los top_k chunks, etiquetados con su passage_id."""
    ids: list[str] = []
    for chunk in ranked_chunks[:top_k]:
        for pid in chunk["passage_ids"]:
            if pid not in ids:
                ids.append(pid)
    return "\n\n".join(f"[{pid}] {passages[pid]}" for pid in ids), ids


def parse_citations(answer: str) -> list[str]:
    seen: list[str] = []
    for pid in CITE_RE.findall(answer):
        if pid not in seen:
            seen.append(pid)
    return seen


def is_abstention(answer: str) -> bool:
    norm = answer.lower().replace("’", "'")
    return any(p in norm for p in ABSTAIN_PATTERNS)


def citation_metrics(answer: str, context_ids: list[str], relevant: list[str]) -> dict[str, Any]:
    """Comprobaciones mecanicas de citas (no miden si la afirmacion esta respaldada)."""
    cited = parse_citations(answer)
    ctx, rel = set(context_ids), set(relevant)
    return {"n_cited": len(cited), "cited": cited,
            "cited_in_context_rate": (sum(c in ctx for c in cited) / len(cited)) if cited else None,
            "cited_relevant_rate": (sum(c in rel for c in cited) / len(cited)) if cited else None,
            "relevant_cited_recall": (len(set(cited) & rel) / len(rel)) if rel else None,
            "relevant_in_context_recall": (len(ctx & rel) / len(rel)) if rel else None}


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", text.replace(",", "")))


def numeric_fact_proxy(answer: str, required_facts: list[str]) -> float | None:
    """Fraccion de las cifras de los hechos requeridos que aparecen en la respuesta (proxy lexico, NO correccion)."""
    want = set().union(*(numbers(f) for f in required_facts)) if required_facts else set()
    return (len(want & numbers(answer)) / len(want)) if want else None


def token_f1(pred: str, ref: str) -> float:
    tok = lambda s: [t for t in re.findall(r"[a-z0-9.]+", re.sub(r"\[[^\]]*\]", " ", s.lower())) if t not in STOP]
    p, r = tok(pred), tok(ref)
    if not p or not r:
        return 0.0
    common = sum(min(p.count(t), r.count(t)) for t in set(p))
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def split_claims(answer: str) -> list[str]:
    """Afirmaciones candidatas para la auditoria manual: cada segmento con etiqueta de cita ([D01-P05], [Passage D01-P05]) o,
    si no hay etiquetas, cada frase. Protege la abreviatura 'U.S.' para no partir las frases en ese punto."""
    text = re.sub(r"U\.S\.", "U․S․", answer.strip())
    segments = [x for x in re.split(r"(?=\[(?:Passage\s+)?[A-Z]{1,2}\d{2}-P\d{2}\])", text) if x.strip()]
    if not (len(segments) > 1 or text.startswith("[")):
        segments = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [x.replace("․", ".").strip() for x in segments if len(x.strip()) > 3]


def parse_json_object(text: str) -> tuple[dict | None, str]:
    """Parseo estricto, o recuperando el primer {...} (misma politica que la PC1 y la PC2)."""
    try:
        obj = json.loads(text)
        return (obj, "exact") if isinstance(obj, dict) else (None, "not_object")
    except json.JSONDecodeError:
        m = re.search(r"\{.*?\}", text, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                return (obj, "recovered") if isinstance(obj, dict) else (None, "not_object")
            except json.JSONDecodeError:
                pass
    return None, "unparseable"
