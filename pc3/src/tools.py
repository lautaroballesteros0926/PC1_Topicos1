"""Herramientas de la PC3 y validacion de observaciones.

Tres operaciones de solo lectura, todas detras del controlador (el modelo nunca ejecuta codigo ni comandos):

    search_evidence(query, top_k)   recuperador C del parcial (e5 + FAISS, BM25, RRF, Cross-Encoder)     [Semanas 4, 5 y 7]
    compute_signal(ticker, as_of)   herramienta risk.compute_signal del parcial, con sus contratos         [Semana 6]
    read_passage(passage_id)        operacion auxiliar nueva: texto completo de un passage YA observado

Cada salida cruda pasa por un modelo Pydantic (`validate_observation`) antes de entrar al contexto del LLM: los textos se
truncan y se marcan como datos, los ids se comprueban contra el corpus y los errores se reducen a campos estructurados
(tipo, codigo, reintentable). El mensaje libre del error NO se pasa al modelo: un fallo no se convierte en instruccion.
"""
from __future__ import annotations

import asyncio
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

import config  # noqa: F401  (anade parcial/src al path)
from config import LIMITS, RETRIEVAL_CONDITION, Limits

PID_RE = re.compile(r"^D\d{2}-P\d{2}$")
DATE_RE = r"^\d{4}-\d{2}-\d{2}$"
TOOL_NAMES = ("search_evidence", "compute_signal", "read_passage")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


# --------------------------------------------------------------------------- contratos de argumentos

class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=3, max_length=300)
    top_k: int = Field(default=3, ge=1, le=5)


class SignalArgs(BaseModel):
    """Forma de los argumentos; el dominio (ticker soportado, dia habil) lo valida la herramienta del parcial."""
    model_config = ConfigDict(extra="forbid")
    ticker: str = Field(pattern=r"^[A-Z]{1,5}$")
    as_of: str = Field(pattern=DATE_RE)


class ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    passage_id: str = Field(pattern=PID_RE.pattern)


ARG_MODELS: dict[str, type[BaseModel]] = {"search_evidence": SearchArgs, "compute_signal": SignalArgs, "read_passage": ReadArgs}


# --------------------------------------------------------------------------- contratos de observacion

def clean_text(text: str, limit: int) -> str:
    text = CONTROL_CHARS.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class PassageObs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    passage_id: str = Field(pattern=PID_RE.pattern)
    doc_title: str
    text: str = Field(max_length=700)


class SearchObs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["ok"] = "ok"
    tool: Literal["search_evidence"] = "search_evidence"
    query: str
    passages: list[PassageObs] = Field(max_length=6)


class SignalObs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["ok"] = "ok"
    tool: Literal["compute_signal"] = "compute_signal"
    ticker: str
    as_of: str
    severity: Literal["low", "medium", "high"]
    score: int = Field(ge=0, le=6)
    ret5_pct: float
    vol20_pct: float
    dd20_pct: float
    attempts: int


class ReadObs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["ok"] = "ok"
    tool: Literal["read_passage"] = "read_passage"
    passage: PassageObs


class ErrorObs(BaseModel):
    """Error como dato: sin mensaje libre ni traza interna."""
    model_config = ConfigDict(extra="forbid")
    status: Literal["error"] = "error"
    tool: str
    kind: Literal["protocol", "validation", "domain", "transient", "timeout", "execution", "output_contract", "invalid_observation"]
    code: str = Field(max_length=60)
    retryable: bool
    attempts: int = 0

    @field_validator("code")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not re.fullmatch(r"[a-z0-9_]+", v):
            raise ValueError("code debe ser un identificador")
        return v


# --------------------------------------------------------------------------- fuentes de precios (con fallos inyectados)

def make_price_source(fault: dict[str, Any] | None):
    """Fuente real de la PC1 o, si la tarea lo declara, la misma fuente con un fallo INYECTADO del parcial."""
    from tool_risk import CorruptSource, FilePriceSource, FlakySource, SlowSource
    base = FilePriceSource()
    if not fault:
        return base
    kind = fault["kind"]
    if kind == "transient":
        return FlakySource(base, fail_times=int(fault["fail_times"]))
    if kind == "corrupt":
        return CorruptSource(base)
    if kind == "slow":
        return SlowSource(base, delay_s=float(fault["delay_s"]))
    raise ValueError(f"fallo inyectado desconocido: {kind}")


# --------------------------------------------------------------------------- caja de herramientas

class ToolBox:
    """Ejecuta herramientas con contrato, timeout y validacion de la observacion. Nunca lanza excepciones al controlador."""

    def __init__(self, search_fn: Callable[[str], list[dict]], passages: dict[str, str], titles: dict[str, str],
                 price_source_factory: Callable[[], Any] | None = None, limits: Limits = LIMITS):
        self.search_fn = search_fn                    # query -> chunks rankeados (dicts con passage_ids)
        self.passages, self.titles = passages, titles
        self.price_source_factory = price_source_factory or (lambda: make_price_source(None))
        self.price_source = self.price_source_factory()
        self.limits = limits
        self.pool = ThreadPoolExecutor(max_workers=1)

    def reset(self, fault: dict[str, Any] | None = None) -> None:
        """Nueva tarea: fuente de precios nueva (los contadores de fallos inyectados empiezan de cero)."""
        self.price_source = make_price_source(fault) if fault else self.price_source_factory()

    # ---- implementaciones crudas
    def _search(self, args: SearchArgs) -> dict[str, Any]:
        ranked = self.search_fn(args.query)
        ids: list[str] = []
        for chunk in ranked[: args.top_k]:
            for pid in chunk["passage_ids"]:
                if pid not in ids:
                    ids.append(pid)
        ids = ids[: self.limits.max_passages_per_search]
        return {"query": args.query, "passages": [
            {"passage_id": p, "doc_title": self.titles[p[:3]], "text": clean_text(self.passages[p], self.limits.excerpt_chars)}
            for p in ids]}

    def _signal(self, args: SignalArgs) -> dict[str, Any]:
        from tool_risk import call_tool
        res = asyncio.run(call_tool("risk.compute_signal", {"ticker": args.ticker, "as_of": args.as_of}, self.price_source))
        if not res.ok:
            return {"__error__": {"tool": "compute_signal", "kind": res.kind, "code": res.code[:60],
                                  "retryable": res.retryable, "attempts": res.attempts}}
        d = res.data
        return {"ticker": d.ticker, "as_of": str(d.as_of), "severity": d.label, "score": d.score,
                "ret5_pct": round(d.ret5_pct, 2), "vol20_pct": round(d.vol20_pct, 2), "dd20_pct": round(d.dd20_pct, 2),
                "attempts": res.attempts}

    def _read(self, args: ReadArgs, seen: set[str]) -> dict[str, Any]:
        if args.passage_id not in seen:
            return {"__error__": {"tool": "read_passage", "kind": "domain", "code": "passage_not_observed",
                                  "retryable": False, "attempts": 1}}
        p = args.passage_id
        return {"passage": {"passage_id": p, "doc_title": self.titles[p[:3]], "text": clean_text(self.passages[p], 650)}}

    # ---- entrada unica
    def execute(self, name: str, arguments: Any, seen_passages: set[str]) -> dict[str, Any]:
        """Devuelve una observacion validada (dict de SearchObs/SignalObs/ReadObs/ErrorObs) y metadatos de ejecucion."""
        t0 = time.perf_counter()
        if name not in TOOL_NAMES:
            return self._wrap(ErrorObs(tool=str(name)[:40], kind="protocol", code="unknown_tool", retryable=False), t0)
        try:
            args = ARG_MODELS[name].model_validate(arguments)
        except ValidationError:
            return self._wrap(ErrorObs(tool=name, kind="validation", code="input_contract", retryable=False), t0)
        job = {"search_evidence": lambda: self._search(args), "compute_signal": lambda: self._signal(args),
               "read_passage": lambda: self._read(args, seen_passages)}[name]
        future = self.pool.submit(job)
        try:
            raw = future.result(timeout=self.limits.timeout_per_tool_s)
        except FutureTimeout:
            # cooperativo: el hilo puede seguir corriendo, pero su resultado se descarta
            self.pool = ThreadPoolExecutor(max_workers=1)
            return self._wrap(ErrorObs(tool=name, kind="timeout", code="tool_timeout", retryable=True), t0)
        except Exception as exc:  # error interno -> dato estable, sin stack
            return self._wrap(ErrorObs(tool=name, kind="execution", code=re.sub(r"[^a-z0-9_]", "_", type(exc).__name__.lower())[:60],
                                       retryable=False), t0)
        if "__error__" in raw:
            return self._wrap(ErrorObs(**raw["__error__"]), t0)
        model = {"search_evidence": SearchObs, "compute_signal": SignalObs, "read_passage": ReadObs}[name]
        try:
            obs = model.model_validate(raw)
            if name == "search_evidence":
                assert all(p.passage_id in self.passages for p in obs.passages)
        except (ValidationError, AssertionError):
            return self._wrap(ErrorObs(tool=name, kind="invalid_observation", code="observation_contract", retryable=False), t0)
        return self._wrap(obs, t0)

    @staticmethod
    def _wrap(obs: BaseModel, t0: float) -> dict[str, Any]:
        return {"observation": obs.model_dump(), "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


def build_toolbox(limits: Limits = LIMITS) -> tuple[ToolBox, Any]:
    """Construye el recuperador C del parcial (mismos chunks, modelos y profundidad) y la caja de herramientas."""
    from common import load_benchmark
    from retrieval import Retrievers, build_chunks, run_condition

    bench = load_benchmark()
    chunks = build_chunks(bench["documents"])
    retr = Retrievers(chunks)
    titles = {d["doc_id"]: d["title"] for d in bench["documents"]}

    def search_fn(query: str) -> list[dict]:
        ranking, _ = run_condition(retr, RETRIEVAL_CONDITION, query)
        return ranking

    return ToolBox(search_fn, bench["passages"], titles, limits=limits), retr
