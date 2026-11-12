"""Rutas, limites fijos del controlador y enlace con los modulos del examen parcial.

La PC3 NO copia el codigo del parcial: lo importa desde `../parcial/src` (recuperador C, herramienta risk.compute_signal,
regla de la PC1, corpus y qrels). Asi el baseline del parcial queda intacto y cualquier correccion se ve en el diff.
Los limites de este archivo se fijaron antes de ejecutar los experimentos y se copian a results/declared_config.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PC3 = Path(__file__).resolve().parents[1]
REPO = PC3.parent
PARCIAL = REPO / "parcial"
PARCIAL_SRC = PARCIAL / "src"
DATA = PC3 / "data"
RESULTS = PC3 / "results"
PROMPTS = PC3 / "prompts"
TRACES = RESULTS / "traces"

if str(PARCIAL_SRC) not in sys.path:          # reutilizacion explicita del parcial (no se modifica)
    sys.path.append(str(PARCIAL_SRC))

SEED = 42
RETRIEVAL_CONDITION = "C_hybrid_rrf_reranker"   # arquitectura seleccionada en el parcial (Exp. 1 y Exp. 2)


@dataclass(frozen=True)
class Limits:
    """Presupuestos del controlador: iguales para W0, W1 y M0..M3."""
    max_steps: int = 6                 # iteraciones del controlador (decision + accion)
    max_tool_calls: int = 5            # llamadas reales a herramientas
    max_invalid_outputs: int = 2       # decisiones invalidas (JSON, accion, argumentos, finish no verificado)
    max_tool_errors: int = 2           # errores de herramienta consecutivos antes de TOOL_ERROR
    no_progress_patience: int = 2      # acciones ejecutadas seguidas sin informacion nueva antes de NO_PROGRESS
    timeout_per_tool_s: float = 20.0   # por llamada (la herramienta de riesgo ademas tiene su deadline interno de 2 s)
    total_deadline_s: float = 300.0    # por tarea, cooperativo: se comprueba entre pasos
    max_input_tokens: int = 2048       # presupuesto de contexto por decision del LLM (prompt completo)
    max_output_tokens: int = 200       # tokens nuevos por decision (= max_new_tokens del parcial)
    max_passages_per_search: int = 6
    excerpt_chars: int = 220
    memory_items: int = 3              # recuerdos maximos inyectados (Exp. 2)
    memory_token_budget: int = 450     # parte del contexto reservada a memoria (Exp. 2)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


LIMITS = Limits()

STOP_REASONS = {
    "SUCCESS": "finish aceptado con decision alert/no_alert: ids de evidencia observados y severidad igual a una senal observada o a un recuerdo citado",
    "ABSTAIN": "finish aceptado con decision abstain (el sistema declara que no puede sustentar una conclusion)",
    "MAX_STEPS": "se alcanzo max_steps o max_tool_calls sin un finish aceptado",
    "TIMEOUT": "se excedio total_deadline_s o una herramienta excedio su tiempo (timeout_per_tool_s o el deadline interno)",
    "TOOL_ERROR": "max_tool_errors errores de herramienta consecutivos sin un finish",
    "INVALID_OUTPUT": "max_invalid_outputs decisiones invalidas (JSON no parseable, accion fuera de la allowlist, argumentos o finish que no pasan el contrato)",
    "NO_PROGRESS": "no_progress_patience acciones ejecutadas seguidas que no anaden passages, senales ni recuerdos nuevos",
    "REPEATED_ACTION": "la politica propone exactamente una accion (nombre + argumentos) ya ejecutada en la tarea",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows), encoding="utf-8")


def sha256_12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def data_hashes() -> dict[str, str]:
    files = [DATA / "tasks.jsonl", DATA / "expected_results.jsonl", DATA / "memory_scenarios.jsonl",
             DATA / "dev_tasks.jsonl", PROMPTS / "agent_system.txt", PROMPTS / "agent_user.txt",
             PARCIAL / "data" / "corpus_parcial.jsonl", PARCIAL / "data" / "qrels_parcial.json"]
    files += sorted((PARCIAL / "data" / "prices").glob("*.csv"))
    return {str(p.relative_to(REPO)).replace("\\", "/"): sha256_12(p) for p in files if p.exists()}
