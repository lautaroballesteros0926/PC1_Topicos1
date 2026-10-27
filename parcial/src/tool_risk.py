"""Experimento 3: herramienta local `risk.compute_signal` con contratos verificables (Semana 6).

Frontera de confianza entre una peticion probabilistica (p. ej. de un LLM) y una funcion determinista:

    ToolCall {name, arguments}
      -> allowlist (protocol) -> contrato de entrada (validation) -> validacion de dominio (domain)
      -> politica de ejecucion: retry selectivo + backoff + timeout por intento + deadline total
      -> implementacion -> contrato de salida (output_contract) -> ToolResult | ToolError

La herramienta calcula, con la regla de la PC1, los indicadores (ret5, vol20, dd20) y la severidad
(low/medium/high) de un activo en una fecha. No hay agentes ni cadenas de herramientas.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from common import DATA
from risk_rules import compute_features, label_from_score, score_parts

TOOL_NAME = "risk.compute_signal"
ALLOWLIST = frozenset({TOOL_NAME})
SUPPORTED_TICKERS = ("GLD", "IWM", "KO", "SPY")
PRICES_DIR = DATA / "prices"

TOOL_SPEC = {
    "name": TOOL_NAME,
    "description": ("Calcula el retorno a 5 dias, la volatilidad anualizada de 20 dias y la caida desde el maximo de 20 dias de un "
                    "activo en una fecha, y devuelve la severidad de riesgo (low, medium, high) segun la politica de la PC1."),
    "parameters": {"type": "object", "additionalProperties": False, "required": ["ticker", "as_of"],
                   "properties": {"ticker": {"type": "string", "enum": list(SUPPORTED_TICKERS)},
                                  "as_of": {"type": "string", "format": "date", "description": "Fecha de cierre YYYY-MM-DD"}}},
}

ErrorKind = Literal["protocol", "validation", "domain", "transient", "timeout", "execution", "output_contract"]
RETRYABLE_KINDS = {"transient", "timeout"}     # el resto no se reintenta: repetir no cambia el resultado


# --------------------------------------------------------------------------- contratos

class ComputeSignalInput(BaseModel):
    """Contrato de entrada: forma y tipos. NO sabe si el ticker o la fecha existen (eso es dominio)."""
    model_config = ConfigDict(extra="forbid")
    ticker: str = Field(pattern=r"^[A-Z]{1,5}$")
    as_of: date


class Points(BaseModel):
    vol20: int = Field(ge=0, le=2)
    dd20: int = Field(ge=0, le=2)
    ret5: int = Field(ge=0, le=2)


class RiskSignal(BaseModel):
    """Contrato de salida: lo que la herramienta promete devolver."""
    model_config = ConfigDict(extra="forbid")
    ticker: str
    as_of: date
    close: float = Field(gt=0, allow_inf_nan=False)
    ret5_pct: float = Field(allow_inf_nan=False)
    vol20_pct: float = Field(ge=0, allow_inf_nan=False)
    dd20_pct: float = Field(le=0, allow_inf_nan=False)
    points: Points
    score: int = Field(ge=0, le=6)
    label: Literal["low", "medium", "high"]

    @model_validator(mode="after")
    def _consistent(self) -> "RiskSignal":
        total = self.points.vol20 + self.points.dd20 + self.points.ret5
        if self.score != total or self.label != label_from_score(self.score):
            raise ValueError("score o label inconsistentes con los puntos por indicador")
        return self


class ToolResult(BaseModel):
    ok: Literal[True] = True
    tool: str
    data: RiskSignal
    attempts: int
    elapsed_ms: float


class ToolError(BaseModel):
    ok: Literal[False] = False
    tool: str
    kind: ErrorKind
    code: str
    message: str
    retryable: bool            # si ESTE tipo de error admite reintento (no si se reintento o se agoto)
    attempts: int
    elapsed_ms: float
    details: dict[str, Any] = Field(default_factory=dict)


class DomainError(Exception):
    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.code, self.details = code, details


class TransientToolError(Exception):
    """Fallo temporal de la fuente de datos (reintentable)."""


# --------------------------------------------------------------------------- fuentes de datos

class FilePriceSource:
    """Fuente real: CSV de precios de cierre ajustados (Yahoo Finance, descargados en la PC1)."""

    def __init__(self, prices_dir: Path = PRICES_DIR):
        self.prices_dir, self._cache = prices_dir, {}
        self.calls = 0

    def _read(self, ticker: str) -> pd.Series:
        df = pd.read_csv(self.prices_dir / f"{ticker}.csv", parse_dates=["Date"], index_col="Date")
        return df["Close"].astype(float).sort_index()

    async def load_close(self, ticker: str) -> pd.Series:
        self.calls += 1
        if ticker not in self._cache:
            self._cache[ticker] = await asyncio.to_thread(self._read, ticker)
        return self._cache[ticker]


class FlakySource:
    """Inyeccion de fallos: lanza TransientToolError en las primeras `fail_times` llamadas."""

    def __init__(self, inner, fail_times: int):
        self.inner, self.fail_times, self.calls = inner, fail_times, 0

    async def load_close(self, ticker: str) -> pd.Series:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TransientToolError(f"fuente de precios no disponible (llamada {self.calls})")
        return await self.inner.load_close(ticker)


class SlowSource:
    """Inyeccion de fallos: tarda `delay_s` (cancelable) antes de responder."""

    def __init__(self, inner, delay_s: float):
        self.inner, self.delay_s, self.calls = inner, delay_s, 0

    async def load_close(self, ticker: str) -> pd.Series:
        self.calls += 1
        await asyncio.sleep(self.delay_s)
        return await self.inner.load_close(ticker)


class CorruptSource:
    """Inyeccion de fallos: datos corruptos (cierres negativos) que violan el contrato de salida."""

    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    async def load_close(self, ticker: str) -> pd.Series:
        self.calls += 1
        return -(await self.inner.load_close(ticker))


# --------------------------------------------------------------------------- implementacion

async def compute_signal(inp: ComputeSignalInput, source) -> dict[str, Any]:
    """Implementacion de la herramienta. Contiene la validacion de dominio que depende de los datos."""
    if inp.ticker not in SUPPORTED_TICKERS:
        raise DomainError("ticker_not_supported", f"{inp.ticker} no esta en la lista soportada", supported=list(SUPPORTED_TICKERS))
    close = await source.load_close(inp.ticker)
    ts = pd.Timestamp(inp.as_of)
    first, last = close.index[0], close.index[-1]
    if ts < first or ts > last:
        raise DomainError("date_out_of_range", f"{inp.as_of} fuera del rango de datos",
                          first=str(first.date()), last=str(last.date()))
    if ts not in close.index:
        raise DomainError("date_not_trading_day", f"{inp.as_of} no es un dia de negociacion con datos")
    row = compute_features(close).loc[ts]
    if row[["ret5", "vol20", "dd20"]].isna().any():
        raise DomainError("insufficient_history", f"{inp.as_of} no tiene historia suficiente (se necesitan 20 retornos previos)")
    parts = score_parts(float(row["vol20"]), float(row["dd20"]), float(row["ret5"]))
    score = sum(parts.values())
    return {"ticker": inp.ticker, "as_of": inp.as_of, "close": float(row["close"]), "ret5_pct": float(row["ret5"]),
            "vol20_pct": float(row["vol20"]), "dd20_pct": float(row["dd20"]), "points": parts, "score": score,
            "label": label_from_score(score)}


# --------------------------------------------------------------------------- politica de ejecucion

@dataclass(frozen=True)
class Policy:
    max_attempts: int = 3
    backoff_base_s: float = 0.05
    backoff_factor: float = 2.0
    per_attempt_timeout_s: float = 0.5
    overall_deadline_s: float = 2.0


def _error(kind: ErrorKind, code: str, message: str, attempts: int, t0: float, **details: Any) -> ToolError:
    return ToolError(tool=TOOL_NAME, kind=kind, code=code, message=message, retryable=kind in RETRYABLE_KINDS,
                     attempts=attempts, elapsed_ms=round((time.monotonic() - t0) * 1000, 2), details=details)


async def call_tool(name: str, arguments: Any, source, policy: Policy = Policy()) -> ToolResult | ToolError:
    """Despacha una llamada. Nunca lanza: devuelve ToolResult o un ToolError estructurado."""
    t0 = time.monotonic()
    if name not in ALLOWLIST:
        return _error("protocol", "unknown_tool", f"herramienta '{name}' no permitida", 0, t0, allowed=sorted(ALLOWLIST))
    if not isinstance(arguments, dict):
        return _error("protocol", "arguments_not_object", "los argumentos deben ser un objeto JSON", 0, t0)
    try:
        inp = ComputeSignalInput.model_validate(arguments)
    except ValidationError as exc:
        return _error("validation", "input_contract", "argumentos no cumplen el contrato de entrada", 0, t0,
                      errors=[{"loc": list(e["loc"]), "type": e["type"]} for e in exc.errors()])

    attempts = 0
    while True:
        attempts += 1
        budget = policy.overall_deadline_s - (time.monotonic() - t0)
        try:
            raw = await asyncio.wait_for(compute_signal(inp, source), timeout=min(policy.per_attempt_timeout_s, budget))
            data = RiskSignal.model_validate(raw)
            return ToolResult(tool=TOOL_NAME, data=data, attempts=attempts, elapsed_ms=round((time.monotonic() - t0) * 1000, 2))
        except DomainError as exc:
            return _error("domain", exc.code, str(exc), attempts, t0, **exc.details)
        except ValidationError as exc:
            return _error("output_contract", "output_contract", "la salida no cumple el contrato prometido", attempts, t0,
                          errors=[{"loc": list(e["loc"]), "type": e["type"]} for e in exc.errors()])
        except TransientToolError as exc:
            kind, code, message = "transient", "source_unavailable", str(exc)
        except (asyncio.TimeoutError, TimeoutError):
            kind, code, message = "timeout", "attempt_timeout", f"intento {attempts} excedio {policy.per_attempt_timeout_s}s"
        except Exception as exc:  # excepcion interna -> error estable, sin filtrar el stack al llamador
            return _error("execution", "internal_error", type(exc).__name__, attempts, t0)

        # Solo transient y timeout llegan aqui: reintentar solo si queda cupo de intentos y presupuesto de tiempo.
        delay = policy.backoff_base_s * policy.backoff_factor ** (attempts - 1)
        remaining = policy.overall_deadline_s - (time.monotonic() - t0)
        if attempts >= policy.max_attempts:
            return _error(kind, f"{code}_retries_exhausted", message, attempts, t0)
        if remaining < delay + policy.per_attempt_timeout_s:
            return _error(kind, f"{code}_deadline_exhausted", message, attempts, t0, remaining_s=round(remaining, 3))
        await asyncio.sleep(delay)
