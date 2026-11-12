"""Pruebas del controlador con politicas guionizadas (no hay LLM): cada stop_reason tiene una condicion verificable."""
from config import Limits
from controller import first_json_object, run_task
from workflow import FixedPlanWorkflow, ScriptedPolicy

S = lambda q, k=2: {"action": "search_evidence", "arguments": {"query": q, "top_k": k}}
C = lambda t, d: {"action": "compute_signal", "arguments": {"ticker": t, "as_of": d}}


def task(requests=(), fault=None, q="probe"):
    return {"task_id": "P", "input": {"question": q, "signal_requests": [{"ticker": t, "as_of": d} for t, d in requests]},
            "injected_fault": fault}


def fin(**kw):
    base = {"decision": "alert", "ticker": "SPY", "as_of": "2020-03-16", "severity": "high", "evidence_ids": ["D01-P12"]}
    return {"action": "finish", "arguments": {**base, **kw}}


def test_success_requires_observed_signal_and_evidence(toolbox):
    res = run_task(task(), ScriptedPolicy([S("covid crash"), C("SPY", "2020-03-16"), fin()]), toolbox)
    assert res["stop_reason"] == "SUCCESS" and res["answer"]["severity"] == "high"
    assert [t["selected_tool"] for t in res["trace"]] == ["search_evidence", "compute_signal", "finish"]


def test_finish_with_unobserved_evidence_is_rejected(toolbox):
    res = run_task(task(), ScriptedPolicy([C("SPY", "2020-03-16"), fin(evidence_ids=["D99-P99"]), fin(evidence_ids=["D05-P01"])]), toolbox)
    assert res["stop_reason"] == "INVALID_OUTPUT" and res["answer"] is None
    assert [t.get("error") for t in res["trace"] if t["decision_status"] == "rejected"] == ["finish_unobserved_evidence"] * 2


def test_guessed_severity_is_rejected(toolbox):
    res = run_task(task(), ScriptedPolicy([S("covid"), fin(severity="high"), fin(severity="medium")]), toolbox)
    assert res["stop_reason"] == "INVALID_OUTPUT"
    assert res["trace"][1]["error"] == "finish_severity_not_supported"


def test_alert_on_low_signal_is_rejected(toolbox):
    res = run_task(task(), ScriptedPolicy([S("covid"), C("KO", "2021-04-30"),
                                           fin(ticker="KO", as_of="2021-04-30", severity="low"),
                                           fin(ticker="KO", as_of="2021-04-30", severity="low", decision="no_alert")]), toolbox)
    assert res["trace"][2]["error"] == "finish_alert_on_low_signal"
    assert res["stop_reason"] == "SUCCESS" and res["answer"]["decision"] == "no_alert"


def test_repeated_action(toolbox):
    res = run_task(task(), ScriptedPolicy([S("svb"), S("svb")]), toolbox)
    assert res["stop_reason"] == "REPEATED_ACTION" and res["tool_calls"] == 1 and res["repeated_actions"] == 1


def test_no_progress(toolbox):
    res = run_task(task(), ScriptedPolicy([S("svb", 2), S("svb", 1), S("silicon", 1)]), toolbox)
    assert res["stop_reason"] == "NO_PROGRESS"


def test_max_steps_and_tool_budget(toolbox):
    res = run_task(task(), ScriptedPolicy([S("a b c" + str(i)) for i in range(10)]), toolbox,
                   limits=Limits(max_steps=6, max_tool_calls=5, no_progress_patience=99))
    assert res["stop_reason"] == "MAX_STEPS" and res["tool_calls"] == 5
    res = run_task(task(), ScriptedPolicy(["no json"] + [S(f"covid {i}") for i in range(10)]), toolbox,
                   limits=Limits(max_steps=3, max_tool_calls=9, no_progress_patience=99, max_invalid_outputs=5))
    assert res["stop_reason"] == "MAX_STEPS" and res["stop_detail"] == "max_steps"


def test_invalid_output_and_errors_are_data(toolbox):
    res = run_task(task(), ScriptedPolicy(["texto libre", '{"action": "rm -rf", "arguments": {}}']), toolbox)
    assert res["stop_reason"] == "INVALID_OUTPUT" and res["tool_calls"] == 0


def test_tool_error_stop_and_no_message_leak(toolbox):
    res = run_task(task(fault={"kind": "transient", "fail_times": 50}),
                   ScriptedPolicy([C("SPY", "2020-03-16"), C("SPY", "2020-03-13")]), toolbox)
    assert res["stop_reason"] == "TOOL_ERROR" and res["answer"] is None
    obs = res["trace"][0]["observation"]
    assert obs["kind"] == "transient" and "message" not in obs


def test_transient_recovered_inside_tool(toolbox):
    res = run_task(task(fault={"kind": "transient", "fail_times": 1}),
                   ScriptedPolicy([C("GLD", "2020-03-16")]), toolbox, limits=Limits(max_steps=1))
    assert res["trace"][0]["observation"]["attempts"] == 2 and res["signals"][0]["severity"] == "high"


def test_corrupt_output_is_not_turned_into_signal(toolbox):
    res = run_task(task(fault={"kind": "corrupt"}), ScriptedPolicy([C("SPY", "2025-04-08"), C("SPY", "2025-04-07")]), toolbox)
    assert res["stop_reason"] == "TOOL_ERROR" and res["signals"] == []
    assert res["trace"][0]["observation"]["kind"] == "output_contract"


def test_tool_timeout(toolbox):
    res = run_task(task(fault={"kind": "slow", "delay_s": 1.0}),
                   ScriptedPolicy([C("SPY", "2020-03-16"), C("SPY", "2020-03-13")]), toolbox)
    assert res["stop_reason"] == "TIMEOUT"


def test_workflow_plan_and_abstain_on_domain_error(toolbox):
    ok = run_task(task([("SPY", "2020-03-16")], q="covid crash"), FixedPlanWorkflow(), toolbox)
    assert ok["stop_reason"] == "SUCCESS" and ok["answer"]["evidence_ids"] == ["D01-P12", "D01-P11", "D18-P08"]
    bad = run_task(task([("SPY", "2020-03-15")], q="covid crash"), FixedPlanWorkflow(), toolbox)
    assert bad["stop_reason"] == "ABSTAIN" and bad["trace"][1]["observation"]["code"] == "date_not_trading_day"


def test_read_passage_only_seen(toolbox):
    res = run_task(task(), ScriptedPolicy([{"action": "read_passage", "arguments": {"passage_id": "D01-P12"}},
                                           {"action": "read_passage", "arguments": {"passage_id": "D01-P11"}}]), toolbox)
    assert res["trace"][0]["observation"]["code"] == "passage_not_observed" and res["stop_reason"] == "TOOL_ERROR"


def test_first_json_object():
    assert first_json_object('ok ```json\n{"action": "finish", "arguments": {"a": "}"}}\n```')[0]["arguments"]["a"] == "}"
    assert first_json_object("nada")[0] is None
