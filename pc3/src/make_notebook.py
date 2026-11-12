"""Genera notebook/PC3_workflows_agentes_memoria.ipynb (cuaderno unico de la PC3).

    python src/make_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebook/PC3_workflows_agentes_memoria.ipynb   (en Docker)

El cuaderno NO vuelve a correr los experimentos (Qwen en CPU tarda ~1 h en total): carga los resultados y las trazas
originales de results/, recalcula las metricas desde los CSV, ejecuta las pruebas y una demostracion corta en vivo.
Los textos de analisis estan en src/notebook_text.py.
"""
from __future__ import annotations

import nbformat as nbf

from config import PC3
from notebook_text import TEXT

cells: list = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(TEXT["portada"])
code("""
import os; os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"; os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import json, sys, subprocess, platform, warnings; warnings.filterwarnings("ignore")
from transformers.utils import logging as hf_logging; hf_logging.disable_progress_bar(); hf_logging.set_verbosity_error()
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
pd.set_option("display.max_colwidth", 120); pd.set_option("display.width", 200)
from IPython.display import display, Image, Markdown
from config import *                      # rutas, LIMITS, STOP_REASONS (y parcial/src en el path)
cfg = json.loads((RESULTS / "declared_config.json").read_text(encoding="utf-8"))
print("python", platform.python_version(), "| declarado:", cfg["declared_at_utc"], "| LLM:", cfg["llm"]["id"], cfg["llm"]["decoding"])
""")

md(TEXT["s1"])
code("""
display(pd.DataFrame([{"limite": k, "valor": v} for k, v in cfg["limits"].items()]))
display(pd.DataFrame([{"stop_reason": k, "definicion verificable": v} for k, v in cfg["stop_reasons"].items()]))
same = cfg["data_hashes"] == data_hashes()
print("benchmark, prompts, corpus y precios sin cambios desde la declaracion:", same)
""")

md(TEXT["s2"])
code("""
tasks = read_jsonl(DATA / "tasks.jsonl"); expected = {e["task_id"]: e for e in read_jsonl(DATA / "expected_results.jsonl")}
display(pd.DataFrame([{"task_id": t["task_id"], "tipo": t["category"], "objetivo": t["objective"],
    "entrada": "; ".join(f'{r["ticker"]} {r["as_of"]}' for r in t["input"]["signal_requests"]),
    "operaciones": " -> ".join(t["operations"]), "fallo inyectado": (t["injected_fault"] or {}).get("note", ""),
    "esperado": f'{expected[t["task_id"]]["decision"]} {expected[t["task_id"]]["ticker"] or ""} {expected[t["task_id"]]["as_of"] or ""} {expected[t["task_id"]]["severity"] or ""}',
    "evidencia": expected[t["task_id"]]["evidence_set"] or "-", "stops permitidos": "/".join(expected[t["task_id"]]["allowed_stops"])}
    for t in tasks]))
""")
code("""
mem = read_jsonl(DATA / "memory_scenarios.jsonl")
display(pd.DataFrame([{"id": m["task_id"], "sesion": m["session"], "pregunta": m["input"]["question"][:95],
    "esperado": f'{m["expected"]["decision"]} {m["expected"]["ticker"]} {m["expected"]["as_of"]} {m["expected"]["severity"]}',
    "flags": ",".join(k for k, v in m["flags"].items() if v) or "-", "recuerdo correcto": ",".join(m["gold_memory_sources"]) or "-",
    "obsoleto / interferente": ",".join(m["stale_sources"] + m["interfering_sources"]) or "-"} for m in mem]))
""")

md(TEXT["s3"])
code("""
print((PROMPTS / "agent_system.txt").read_text(encoding="utf-8"))
print("---- plantilla del turno ----")
print((PROMPTS / "agent_user.txt").read_text(encoding="utf-8"))
""")
md(TEXT["s3b"])
code("""
r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "--color=no"], cwd=ROOT, capture_output=True, text=True)
print(r.stdout[-1500:])
""")

md(TEXT["s4"])
code("""
per1 = pd.read_csv(RESULTS / "exp1" / "per_task.csv"); sum1 = pd.read_csv(RESULTS / "exp1" / "summary.csv")
display(sum1.round(3))
piv = per1.pivot(index="task_id", columns="policy", values=["task_success", "stop_reason", "steps", "tool_calls", "tokens"])
display(piv)
display(per1.groupby(["policy", "category"]).task_success.agg(["mean", "sum", "count"]).round(2))
""")
code("""
Image(filename=str(ROOT / "slides" / "fig" / "exp1_matriz.png"))
""")
md(TEXT["s4b"])
code("""
def show_trace(folder, name):
    r = json.loads((TRACES / folder / f"{name}.json").read_text(encoding="utf-8"))
    print(f'{name}: stop={r["stop_reason"]} ({r["stop_detail"]}) answer={r["answer"]}')
    for t in r["trace"]:
        obs = t.get("observation")
        print(f'  step {t["step"]} [{t["decision_status"]}] {t["selected_tool"]} {json.dumps(t["arguments"])[:110]}')
        if t.get("reason"): print(f'      reason: {t["reason"][:120]}')
        print(f'      obs: {json.dumps(obs)[:150] if obs else t.get("error")} | tokens={t["tokens"]} latency_ms={t["latency_ms"]}')
    print()
""")
code(TEXT["code_exp1_traces"])
md(TEXT["s4c"])
code("""
rob = pd.read_csv(RESULTS / "exp1" / "robustness.csv"); display(rob)
""")
md(TEXT["s4d"])

md(TEXT["s5"])
code("""
per2 = pd.read_csv(RESULTS / "exp2" / "per_interaction.csv"); sum2 = pd.read_csv(RESULTS / "exp2" / "summary.csv")
display(sum2.round(3))
display(per2[["condition", "task_id", "session", "stop_reason", "task_success", "memory_shown", "memory_cited",
              "memory_recall_ok", "stale_error", "wrong_memory_use", "answer_as_of", "answer_severity", "max_context_tokens"]])
""")
code("""
Image(filename=str(ROOT / "slides" / "fig" / "exp2_metricas_qwen.png"))
""")
md(TEXT["s5b"])
code("""
for cond in ["M2_episodic", "M3_semantic"]:
    rows = read_jsonl(RESULTS / "exp2" / "memory_store" / cond / ("episodic.jsonl" if cond == "M2_episodic" else "semantic.jsonl"))
    display(Markdown(f"**{cond}** ({len(rows)} recuerdos persistidos)"))
    display(pd.DataFrame([{"id": m["memory_id"], "status": m["status"], "version": m["version"], "origen": m["provenance"]["interaction_id"],
        "sesion": m["provenance"]["session"], "superseded_by": m.get("superseded_by", ""), "texto": m["text"][:120]} for m in rows]))
ops = read_jsonl(RESULTS / "exp2" / "memory_store" / "M3_semantic" / "memory_ops.jsonl")
display(pd.Series([o["op"] for o in ops]).value_counts().rename("operaciones M3"))
""")
code(TEXT["code_exp2_traces"])
md(TEXT["s5c"])
code("""
rd = RESULTS / "exp2" / "diagnostic_reader"
sumr = pd.read_csv(rd / "summary.csv"); perr = pd.read_csv(rd / "per_interaction.csv")
display(sumr.drop(columns=["context_tokens_mean", "memory_tokens_mean"]).round(3))
display(perr[perr.needs_memory | perr.stale_risk][["condition", "task_id", "task_success", "memory_shown", "memory_cited",
        "memory_recall_ok", "stale_error", "answer_as_of", "answer_severity"]])
display(Image(filename=str(ROOT / "slides" / "fig" / "exp2_metricas_lector.png")))
sem = read_jsonl(rd / "memory_store" / "M3_semantic" / "semantic.jsonl")
display(pd.DataFrame([{"id": m["memory_id"], "status": m["status"], "superseded_by": m.get("superseded_by", ""),
        "origen": m["provenance"]["interaction_id"], "hecho": m["text"][:110]} for m in sem]))
""")

md(TEXT["s6"])
code("""
# Demostracion en vivo sin LLM (rapida): W0 sobre T09 y borrado de un recuerdo en una copia del almacen M2
import shutil, tempfile
from tools import build_toolbox
from controller import run_task
from workflow import FixedPlanWorkflow
from memory import EpisodicStore
toolbox, retr = build_toolbox()
res = run_task(next(t for t in tasks if t["task_id"] == "T09"), FixedPlanWorkflow(), toolbox)
print("W0 en T09 ->", res["stop_reason"], [ (t["selected_tool"], (t.get("observation") or {}).get("code")) for t in res["trace"]])
tmp = Path(tempfile.mkdtemp()) / "ep.jsonl"; shutil.copy(RESULTS / "exp2" / "memory_store" / "M2_episodic" / "episodic.jsonl", tmp)
st = EpisodicStore(tmp); first = st.items[0]["memory_id"]
print("borrar", first, "->", st.delete(first), "| quedan", len(EpisodicStore(tmp).items), "recuerdos en disco")
""")

md(TEXT["s7"])
md(TEXT["s8"])

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = PC3 / "notebook" / "PC3_workflows_agentes_memoria.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("escrito", out)
