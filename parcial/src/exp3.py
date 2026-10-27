"""Experimento 3: escenarios de la herramienta `risk.compute_signal` y reproduccion de los 8 casos de la PC1.

Compara la llamada con contratos (`call_tool`) frente a un despacho ingenuo (sin validacion, sin errores estructurados,
sin reintentos ni timeouts) sobre los MISMOS escenarios y los MISMOS fallos inyectados.
Escribe en results/exp3/: scenarios.jsonl, scenarios_summary.csv, pc1_reproduction.csv, pytest_output.txt.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from common import DATA, RESULTS, ROOT
from tool_risk import (TOOL_NAME, ComputeSignalInput, CorruptSource, FilePriceSource, FlakySource, Policy, SlowSource,
                       ToolError, ToolResult, call_tool, compute_signal)

OUT = RESULTS / "exp3"
DEFAULT = Policy()
TIGHT = Policy(max_attempts=5, per_attempt_timeout_s=0.2, backoff_base_s=0.05, overall_deadline_s=0.7)


@dataclass
class Scenario:
    sid: str
    group: str
    description: str
    arguments: Any
    make_source: Callable[[], Any]
    policy: Policy = DEFAULT
    tool: str = TOOL_NAME
    expect: dict[str, Any] = field(default_factory=dict)   # kind/code/attempts/label esperados


def real():
    return FilePriceSource()


SCENARIOS = [
    Scenario("S01", "valida", "COVID-19, SPY 2020-03-16 (caso H1 de la PC1)", {"ticker": "SPY", "as_of": "2020-03-16"}, real,
             expect={"kind": "ok", "label": "high", "attempts": 1}),
    Scenario("S02", "valida", "Mercado en calma, SPY 2017-06-30 (caso L1)", {"ticker": "SPY", "as_of": "2017-06-30"}, real,
             expect={"kind": "ok", "label": "low", "attempts": 1}),
    Scenario("S03", "valida", "Carry trade, SPY 2024-08-05 (caso M1)", {"ticker": "SPY", "as_of": "2024-08-05"}, real,
             expect={"kind": "ok", "label": "medium", "attempts": 1}),
    Scenario("S04", "entrada_invalida", "ticker en minuscula", {"ticker": "spy", "as_of": "2020-03-16"}, real,
             expect={"kind": "validation", "attempts": 0}),
    Scenario("S05", "entrada_invalida", "fecha con formato dd/mm/aaaa", {"ticker": "SPY", "as_of": "16/03/2020"}, real,
             expect={"kind": "validation", "attempts": 0}),
    Scenario("S06", "entrada_invalida", "falta el campo as_of", {"ticker": "SPY"}, real,
             expect={"kind": "validation", "attempts": 0}),
    Scenario("S07", "entrada_invalida", "campo extra no permitido (window)", {"ticker": "SPY", "as_of": "2020-03-16", "window": 20}, real,
             expect={"kind": "validation", "attempts": 0}),
    Scenario("S08", "protocolo", "herramienta fuera de la lista permitida", {"ticker": "SPY", "as_of": "2020-03-16"}, real,
             tool="risk.delete_all", expect={"kind": "protocol", "attempts": 0}),
    Scenario("S09", "dominio", "ticker valido por tipo pero no soportado (TSLA)", {"ticker": "TSLA", "as_of": "2020-03-16"}, real,
             expect={"kind": "domain", "code": "ticker_not_supported", "attempts": 1}),
    Scenario("S10", "dominio", "fecha de sabado (no hay cierre)", {"ticker": "SPY", "as_of": "2020-03-14"}, real,
             expect={"kind": "domain", "code": "date_not_trading_day", "attempts": 1}),
    Scenario("S11", "dominio", "fecha sin 20 retornos previos", {"ticker": "SPY", "as_of": "2016-06-10"}, real,
             expect={"kind": "domain", "code": "insufficient_history", "attempts": 1}),
    Scenario("S12", "dominio", "fecha fuera del rango de datos", {"ticker": "SPY", "as_of": "2031-01-02"}, real,
             expect={"kind": "domain", "code": "date_out_of_range", "attempts": 1}),
    Scenario("S13", "transitorio", "la fuente falla 2 veces y luego responde (reintento con backoff)",
             {"ticker": "SPY", "as_of": "2020-03-16"}, lambda: FlakySource(FilePriceSource(), fail_times=2),
             expect={"kind": "ok", "label": "high", "attempts": 3}),
    Scenario("S14", "transitorio", "la fuente siempre falla (reintentos agotados)",
             {"ticker": "SPY", "as_of": "2020-03-16"}, lambda: FlakySource(FilePriceSource(), fail_times=99),
             expect={"kind": "transient", "code": "source_unavailable_retries_exhausted", "attempts": 3}),
    Scenario("S15", "timeout", "la fuente tarda 1 s; intento 0.2 s, deadline 0.7 s",
             {"ticker": "SPY", "as_of": "2020-03-16"}, lambda: SlowSource(FilePriceSource(), delay_s=1.0), policy=TIGHT,
             expect={"kind": "timeout", "code": "attempt_timeout_deadline_exhausted", "attempts": 2}),
    Scenario("S16", "contrato_salida", "datos corruptos: cierre negativo viola el contrato de salida",
             {"ticker": "SPY", "as_of": "2020-03-16"}, lambda: CorruptSource(FilePriceSource()),
             expect={"kind": "output_contract", "attempts": 1}),
]


def summarize(res: ToolResult | ToolError) -> dict[str, Any]:
    if isinstance(res, ToolResult):
        return {"kind": "ok", "code": "", "attempts": res.attempts, "label": res.data.label, "score": res.data.score,
                "elapsed_ms": res.elapsed_ms}
    return {"kind": res.kind, "code": res.code, "attempts": res.attempts, "label": "", "score": None, "elapsed_ms": res.elapsed_ms,
            "retryable": res.retryable}


def matches(obs: dict[str, Any], expect: dict[str, Any]) -> bool:
    return all(obs.get(k) == v for k, v in expect.items())


async def naive_dispatch(sc: Scenario) -> dict[str, Any]:
    """Despacho ingenuo: sin allowlist, sin contrato, sin errores estructurados, sin reintento ni timeout."""
    t0 = time.monotonic()
    try:
        inp = ComputeSignalInput.model_construct(**sc.arguments)      # no valida
        data = await compute_signal(inp, sc.make_source())
        return {"outcome": "ok", "label": data["label"], "uncaught": "", "elapsed_ms": round((time.monotonic() - t0) * 1000, 1)}
    except Exception as exc:                                          # el llamador recibiria esta excepcion
        return {"outcome": "uncaught_exception", "label": "", "uncaught": type(exc).__name__,
                "elapsed_ms": round((time.monotonic() - t0) * 1000, 1)}


async def run_scenarios() -> pd.DataFrame:
    rows, trace = [], []
    for sc in SCENARIOS:
        src = sc.make_source()
        res = await call_tool(sc.tool, sc.arguments, src, sc.policy)
        obs = summarize(res)
        naive = await naive_dispatch(sc) if sc.tool == TOOL_NAME else {"outcome": "executed_unknown_tool", "label": "", "uncaught": "",
                                                                         "elapsed_ms": 0.0}
        row = {"scenario": sc.sid, "group": sc.group, "description": sc.description, "expected": json.dumps(sc.expect),
               "observed_kind": obs["kind"], "observed_code": obs["code"], "attempts": obs["attempts"], "label": obs["label"],
               "elapsed_ms": obs["elapsed_ms"], "source_calls": getattr(src, "calls", None),
               "correct_outcome": matches(obs, sc.expect), "naive_outcome": naive["outcome"], "naive_exception": naive["uncaught"],
               "naive_elapsed_ms": naive["elapsed_ms"]}
        rows.append(row)
        trace.append({"scenario": sc.sid, "tool": sc.tool, "arguments": sc.arguments, "policy": sc.policy.__dict__,
                      "result": json.loads(res.model_dump_json()), "naive": naive})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scenarios.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in trace), encoding="utf-8")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "scenarios_summary.csv", index=False)
    return df


async def reproduce_pc1() -> pd.DataFrame:
    rows = []
    with open(DATA / "pc1_casos_gold.csv", encoding="utf-8") as f:
        cases = list(csv.DictReader(f))
    for c in cases:
        res = await call_tool(TOOL_NAME, {"ticker": c["ticker"], "as_of": c["date"]}, FilePriceSource())
        assert isinstance(res, ToolResult)
        d = res.data
        rows.append({"case": c["id"], "ticker": c["ticker"], "date": c["date"], "gold_label": c["gold_label"], "tool_label": d.label,
                     "gold_score": int(c["score"]), "tool_score": d.score,
                     "ret5_gold": float(c["ret5"]), "ret5_tool": round(d.ret5_pct, 2),
                     "vol20_gold": float(c["vol20"]), "vol20_tool": round(d.vol20_pct, 2),
                     "dd20_gold": float(c["dd20"]), "dd20_tool": round(d.dd20_pct, 2),
                     "match": d.label == c["gold_label"] and d.score == int(c["score"]) and round(d.ret5_pct, 2) == float(c["ret5"])
                     and round(d.vol20_pct, 2) == float(c["vol20"]) and round(d.dd20_pct, 2) == float(c["dd20"])})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "pc1_reproduction.csv", index=False)
    return df


def run_pytest() -> tuple[int, str]:
    """Ejecuta la suite de pruebas y guarda la salida original."""
    p = subprocess.run([sys.executable, "-m", "pytest", "tests", "-v", "-p", "no:cacheprovider", "--color=no"], cwd=ROOT,
                       capture_output=True, text=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pytest_output.txt").write_text(p.stdout + p.stderr, encoding="utf-8")
    return p.returncode, p.stdout


def _run(coro):
    """asyncio.run, tambien cuando ya hay un bucle en marcha (p. ej. dentro de Jupyter): entonces corre en un hilo aparte."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


def run_exp3() -> dict[str, Any]:
    scenarios = _run(run_scenarios())
    pc1 = _run(reproduce_pc1())
    code, out = run_pytest()
    return {"scenarios": scenarios, "pc1": pc1, "pytest_returncode": code, "pytest_output": out}


if __name__ == "__main__":
    r = run_exp3()
    print(r["scenarios"][["scenario", "group", "observed_kind", "observed_code", "attempts", "correct_outcome", "naive_outcome",
                          "naive_exception"]].to_string())
    print(r["pc1"][["case", "gold_label", "tool_label", "match"]].to_string())
    print("pytest returncode:", r["pytest_returncode"], "| ", r["pytest_output"].strip().splitlines()[-1])
