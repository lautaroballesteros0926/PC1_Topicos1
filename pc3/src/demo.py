"""Demostracion de 2 minutos para la defensa (en vivo; los experimentos completos ya estan en results/).

    python src/demo.py                 # W0 y W1 sobre T09, recuerdo semantico y recuerdo obsoleto (Qwen: ~1-2 min)
    python src/demo.py --no-llm        # solo W0 + memoria (sin cargar Qwen)

Muestra: una tarea multi-paso, el workflow, el agente sobre la misma tarea, una decision y una observacion registradas,
una recuperacion de memoria, un error controlado / recuerdo obsoleto, una metrica y la condicion de terminacion.
"""
from __future__ import annotations

import argparse
import json

from config import DATA, RESULTS, read_jsonl
from controller import run_task
from evaluation import score_answer
from tools import build_toolbox
from workflow import FixedPlanWorkflow


def show(res: dict, expected: dict) -> None:
    print(f"  politica={res['policy']}  stop_reason={res['stop_reason']} ({res['stop_detail'] or '-'})")
    for t in res["trace"]:
        obs = t.get("observation")
        print(f"   paso {t['step']}: {t['decision_status']:9s} {t['selected_tool'] or '-':16s} {json.dumps(t['arguments'])[:70]}"
              f"\n            reason: {t.get('reason', '')[:90]!r}\n            obs: {json.dumps(obs)[:110] if obs else t.get('error', '-')}")
    sc = score_answer(res["answer"], res["stop_reason"], expected)
    print(f"  respuesta={res['answer']}\n  task_success={sc['task_success']}  termination_correct={sc['termination_correct']}  "
          f"tool_calls={res['tool_calls']}  tokens={res['prompt_tokens'] + res['output_tokens']}  latency={res['latency_ms'] / 1000:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T09")
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    tasks = {t["task_id"]: t for t in read_jsonl(DATA / "tasks.jsonl")}
    exp = {e["task_id"]: e for e in read_jsonl(DATA / "expected_results.jsonl")}
    task = tasks[a.task]
    print(f"TAREA {a.task}: {task['input']['question']}\n  esperado: {exp[a.task]['decision']} {exp[a.task]['ticker']} "
          f"{exp[a.task]['as_of']} {exp[a.task]['severity']} | stops permitidos {exp[a.task]['allowed_stops']}")
    toolbox, _ = build_toolbox()
    print("\n[1] W0 workflow fijo")
    show(run_task(task, FixedPlanWorkflow(), toolbox), exp[a.task])
    if not a.no_llm:
        from agent import QwenAgentPolicy
        print("\n[2] W1 agente Qwen2.5-0.5B (misma tarea, mismo controlador y limites)")
        show(run_task(task, QwenAgentPolicy(), toolbox), exp[a.task])
    print("\n[3] Memoria semantica M3 persistida (diagnostico con lector): hechos activos y obsoletos")
    store = RESULTS / "exp2" / "diagnostic_reader" / "memory_store" / "M3_semantic" / "semantic.jsonl"
    if store.exists():
        for m in read_jsonl(store):
            print(f"   {m['memory_id']} v{m['version']} {m['status']:10s} <- {m['provenance']['interaction_id']} "
                  f"(sesion {m['provenance']['session']}) {m['content']['ticker']} {m['content']['as_of']} {m['content']['severity']}"
                  + (f"  superseded_by={m.get('superseded_by')}" if m["status"] == "superseded" else ""))
    print("\n[4] Recuerdo obsoleto en M2 (episodica) para S2-3 (traza guardada del lector)")
    tr = RESULTS / "traces" / "exp2_reader" / "M2_episodic__S2-3.json"
    if tr.exists():
        r = json.loads(tr.read_text(encoding="utf-8"))
        for m in r["memory_shown"]:
            print(f"   mostrado: {m['memory_id']} score={m.get('score')} {m['provenance']['interaction_id']}")
        print(f"   respuesta: {r['answer']}  stop={r['stop_reason']}")


if __name__ == "__main__":
    main()
