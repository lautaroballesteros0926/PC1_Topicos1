"""Corrida de DESARROLLO (no evaluada): prueba el prompt del agente sobre data/dev_tasks.jsonl, que no estan en el benchmark.

Cada ajuste del prompt hecho con estas tareas queda registrado en results/dev/ (una carpeta por version del prompt).
    python src/run_dev.py --tag v1
"""
from __future__ import annotations

import argparse
import json
import shutil

from agent import QwenAgentPolicy
from config import DATA, PROMPTS, RESULTS, read_jsonl
from controller import run_task
from tools import build_toolbox
from workflow import FixedPlanWorkflow


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    tag = ap.parse_args().tag
    out = RESULTS / "dev" / tag
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(PROMPTS / "agent_system.txt", out / "agent_system.txt")
    toolbox, _ = build_toolbox()
    agent = QwenAgentPolicy()
    for task in read_jsonl(DATA / "dev_tasks.jsonl"):
        for pol in (FixedPlanWorkflow(), agent):
            res = run_task(task, pol, toolbox)
            (out / f"{pol.name}__{task['task_id']}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
            print(pol.name, task["task_id"], res["stop_reason"], res["answer"])
            for t in res["trace"]:
                print("   ", t["step"], t["decision_status"], t["selected_tool"], json.dumps(t["arguments"])[:120],
                      "|", (t.get("raw_output") or "")[:160].replace("\n", " "), "|", t.get("error", ""))


if __name__ == "__main__":
    main()
