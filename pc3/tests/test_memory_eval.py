"""Pruebas de memoria (escritura, lectura, persistencia, invalidacion, borrado) y de las metricas, sin LLM."""
import numpy as np

from controller import run_task
from evaluation import memory_use, score_answer
from memory import EpisodicStore, SemanticStore, WorkingMemory
from workflow import ScriptedPolicy


def fake_encode(texts):
    """Encoder determinista de juguete (bolsa de palabras hasheada) para no cargar e5 en las pruebas."""
    out = np.zeros((len(texts), 64), dtype="float32")
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, hash(w) % 64] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def scen(iid, q, reqs):
    return {"task_id": iid, "input": {"question": q, "signal_requests": [{"ticker": t, "as_of": d} for t, d in reqs]},
            "injected_fault": None}


def run_alert(toolbox, iid, ticker, as_of, sev, q):
    dec = "alert" if sev != "low" else "no_alert"
    ev = ([] if dec == "no_alert" else ["D04-P07"] if "silicon" in q else ["D01-P12"])   # primer id del buscador falso
    acts = [{"action": "search_evidence", "arguments": {"query": q, "top_k": 2}},
            {"action": "compute_signal", "arguments": {"ticker": ticker, "as_of": as_of}},
            {"action": "finish", "arguments": {"decision": dec, "ticker": ticker, "as_of": as_of, "severity": sev,
                                               "evidence_ids": ev}}]
    s = scen(iid, q, [(ticker, as_of)])
    res = run_task(s, ScriptedPolicy(acts), toolbox)
    assert res["stop_reason"] == "SUCCESS", res["trace"]
    return s, res


def test_working_memory_resets_between_sessions(toolbox):
    wm = WorkingMemory()
    wm.start_session("A")
    s, res = run_alert(toolbox, "I1", "SPY", "2023-03-13", "medium", "silicon valley bank")
    wm.write(s, res, 1)
    assert wm.retrieve(s, 2)[0]["content"]["severity"] == "medium"
    wm.start_session("B")
    assert wm.retrieve(s, 3) == []


def test_episodic_persists_and_deletes(toolbox, tmp_path):
    st = EpisodicStore(tmp_path / "ep.jsonl")
    s, res = run_alert(toolbox, "I1", "SPY", "2023-03-13", "medium", "silicon valley bank")
    m = st.write(s, res, "A", 1)
    other = EpisodicStore(tmp_path / "ep.jsonl")          # otra "sesion": se lee del disco
    got = other.retrieve(scen("Q", "SPY alert during the Silicon Valley Bank failure", []), 2)
    assert got[0]["memory_id"] == m["memory_id"] and "score_parts" in got[0]
    assert "rationale" not in got[0]["content"]          # no se guarda texto libre del modelo
    assert other.delete(m["memory_id"]) and EpisodicStore(tmp_path / "ep.jsonl").items == []


def test_semantic_supersedes_older_signal(toolbox, tmp_path):
    st = SemanticStore(tmp_path / "sem.jsonl", fake_encode)
    s1, r1 = run_alert(toolbox, "I1", "SPY", "2024-08-05", "medium", "covid carry trade unwind")
    s2, r2 = run_alert(toolbox, "I2", "SPY", "2024-08-30", "low", "covid carry trade unwind update")
    f1 = st.consolidate(s1, r1, "C", 1)[0]
    f2 = st.consolidate(s2, r2, "D", 2)[0]
    st2 = SemanticStore(tmp_path / "sem.jsonl", fake_encode)
    old = next(m for m in st2.items if m["memory_id"] == f1["memory_id"])
    assert old["status"] == "superseded" and old["superseded_by"] == f2["memory_id"]
    got = st2.retrieve(scen("Q", "current status SPY carry trade alert", []), 3)
    assert [m["memory_id"] for m in got] == [f2["memory_id"]]          # el obsoleto no se sirve
    assert got[0]["provenance"]["tool"] == "risk.compute_signal"


def test_finish_from_memory_and_metrics(toolbox):
    s, res = run_alert(toolbox, "S1-1", "SPY", "2023-03-13", "medium", "silicon valley bank")
    wm = WorkingMemory()
    wm.start_session("A")
    item = wm.write(s, res, 1)
    q = scen("S1-2", "report again", [])
    ans = run_task(q, ScriptedPolicy([{"action": "finish", "arguments": {
        "decision": "alert", "ticker": "SPY", "as_of": "2023-03-13", "severity": "medium", "evidence_ids": [],
        "memory_ids": [item["memory_id"]]}}]), toolbox, memory=[item])
    assert ans["stop_reason"] == "SUCCESS"
    sc = {"expected": {"decision": "alert", "ticker": "SPY", "as_of": "2023-03-13", "severity": "medium",
                       "evidence_any_of": [], "allowed_stops": ["SUCCESS"]},
          "gold_memory_sources": ["S1-1"], "stale_sources": [], "flags": {"stale_risk": False}}
    mu = memory_use(ans, sc, {item["memory_id"]: "S1-1"})
    assert mu["memory_recall_ok"] and not mu["wrong_memory_use"]
    mu_wrong = memory_use(ans, {**sc, "gold_memory_sources": ["S9"]}, {item["memory_id"]: "S1-1"})
    assert mu_wrong["task_success"] and not mu_wrong["memory_recall_ok"] and mu_wrong["wrong_memory_use"]


def test_score_answer_abstain_and_evidence():
    e = {"decision": "abstain", "allowed_stops": ["ABSTAIN", "TOOL_ERROR"], "evidence_any_of": []}
    assert score_answer(None, "TOOL_ERROR", e)["task_success"]
    assert not score_answer(None, "REPEATED_ACTION", e)["task_success"]
    e2 = {"decision": "alert", "ticker": "SPY", "as_of": "2020-03-16", "severity": "high", "evidence_any_of": ["D01-P12"],
          "allowed_stops": ["SUCCESS"]}
    a = {"decision": "alert", "ticker": "SPY", "as_of": "2020-03-16", "severity": "high", "evidence_ids": ["D05-P01"]}
    assert not score_answer(a, "SUCCESS", e2)["task_success"]
    assert score_answer({**a, "evidence_ids": ["D01-P12"]}, "SUCCESS", e2)["task_success"]
