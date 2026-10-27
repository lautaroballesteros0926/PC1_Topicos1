"""Experimento 3: pruebas de contratos de `risk.compute_signal` (valida, entrada invalida, dominio, transitorio, timeout)."""
import asyncio
import csv
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import DATA  # noqa: E402
from tool_risk import (CorruptSource, FilePriceSource, FlakySource, Policy, SlowSource, ToolError, ToolResult,  # noqa: E402
                       call_tool)

TOOL = "risk.compute_signal"


def run(coro):
    return asyncio.run(coro)


def gold_cases():
    with open(DATA / "pc1_casos_gold.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---- ejecucion valida y reproduccion de la PC1
def test_valid_call_returns_tool_result():
    res = run(call_tool(TOOL, {"ticker": "SPY", "as_of": "2020-03-16"}, FilePriceSource()))
    assert isinstance(res, ToolResult) and res.attempts == 1
    assert res.data.label == "high" and res.data.score == 6
    assert (round(res.data.ret5_pct, 2), round(res.data.vol20_pct, 2), round(res.data.dd20_pct, 2)) == (-12.54, 76.95, -29.11)


@pytest.mark.parametrize("case", gold_cases(), ids=lambda c: c["id"])
def test_reproduces_pc1_gold_cases(case):
    res = run(call_tool(TOOL, {"ticker": case["ticker"], "as_of": case["date"]}, FilePriceSource()))
    assert isinstance(res, ToolResult)
    assert res.data.label == case["gold_label"] and res.data.score == int(case["score"])
    assert round(res.data.ret5_pct, 2) == float(case["ret5"])
    assert round(res.data.vol20_pct, 2) == float(case["vol20"])
    assert round(res.data.dd20_pct, 2) == float(case["dd20"])


# ---- entrada invalida: se rechaza ANTES de ejecutar, sin reintento
@pytest.mark.parametrize("args", [{"ticker": "spy", "as_of": "2020-03-16"}, {"ticker": "SPY", "as_of": "16/03/2020"},
                                  {"ticker": "SPY"}, {"ticker": "SPY", "as_of": "2020-03-16", "window": 20}])
def test_invalid_input_is_validation_error_without_execution(args):
    src = FilePriceSource()
    err = run(call_tool(TOOL, args, src))
    assert isinstance(err, ToolError) and err.kind == "validation" and not err.retryable
    assert err.attempts == 0 and src.calls == 0


def test_protocol_errors():
    assert run(call_tool("risk.delete_all", {}, FilePriceSource())).kind == "protocol"
    assert run(call_tool(TOOL, "SPY 2020-03-16", FilePriceSource())).kind == "protocol"


# ---- errores de dominio: tipo-validos pero no ejecutables; no se reintentan
@pytest.mark.parametrize("args,code", [({"ticker": "TSLA", "as_of": "2020-03-16"}, "ticker_not_supported"),
                                       ({"ticker": "SPY", "as_of": "2020-03-14"}, "date_not_trading_day"),
                                       ({"ticker": "SPY", "as_of": "2016-06-10"}, "insufficient_history"),
                                       ({"ticker": "SPY", "as_of": "2031-01-02"}, "date_out_of_range")])
def test_domain_errors_are_not_retried(args, code):
    src = FilePriceSource()
    err = run(call_tool(TOOL, args, src))
    assert isinstance(err, ToolError) and err.kind == "domain" and err.code == code
    assert err.attempts == 1 and not err.retryable


# ---- fallo transitorio: se reintenta con backoff
def test_transient_failure_recovers_with_retry():
    src = FlakySource(FilePriceSource(), fail_times=2)
    res = run(call_tool(TOOL, {"ticker": "SPY", "as_of": "2020-03-16"}, src, Policy(max_attempts=3)))
    assert isinstance(res, ToolResult) and res.attempts == 3 and src.calls == 3


def test_transient_failure_exhausts_retries():
    src = FlakySource(FilePriceSource(), fail_times=99)
    err = run(call_tool(TOOL, {"ticker": "SPY", "as_of": "2020-03-16"}, src, Policy(max_attempts=3)))
    assert isinstance(err, ToolError) and err.kind == "transient" and err.code.endswith("retries_exhausted")
    assert err.attempts == 3 and err.retryable


# ---- timeout controlado: por intento y deadline total
def test_timeout_is_bounded_by_deadline():
    policy = Policy(max_attempts=5, per_attempt_timeout_s=0.2, backoff_base_s=0.05, overall_deadline_s=0.7)
    src = SlowSource(FilePriceSource(), delay_s=5.0)
    t0 = time.monotonic()
    err = run(call_tool(TOOL, {"ticker": "SPY", "as_of": "2020-03-16"}, src, policy))
    assert isinstance(err, ToolError) and err.kind == "timeout" and err.code.endswith("deadline_exhausted")
    assert 1 <= err.attempts < 5 and time.monotonic() - t0 < policy.overall_deadline_s + 0.2


# ---- contrato de salida: la herramienta corrio pero la salida es invalida
def test_output_contract_violation_is_detected():
    src = CorruptSource(FilePriceSource())
    err = run(call_tool(TOOL, {"ticker": "SPY", "as_of": "2020-03-16"}, src))
    assert isinstance(err, ToolError) and err.kind == "output_contract" and not err.retryable and err.attempts == 1
