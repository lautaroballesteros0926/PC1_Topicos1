"""Figuras del cuaderno y las diapositivas, generadas SOLO desde results/ (no hay cifras escritas a mano).

    python src/make_figures.py      ->  slides/fig/*.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PC3, RESULTS

FIG = PC3 / "slides" / "fig"
INK, INK3, GRID = "#1f1d1a", "#77736c", "#e7e3db"
W0C, W1C = "#8a8579", "#b4532a"
OK, BAD, NEU = "#3d7a4f", "#b4532a", "#d9d4c9"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK3, "axes.labelcolor": INK,
                     "xtick.color": INK3, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})

STOP_COLORS = {"SUCCESS": OK, "ABSTAIN": "#5b7fa3", "TOOL_ERROR": "#c9a227", "INVALID_OUTPUT": BAD,
               "REPEATED_ACTION": "#8e5a9b", "NO_PROGRESS": "#a67c52", "MAX_STEPS": "#555", "TIMEOUT": "#222"}


def exp1_matrix() -> None:
    per = pd.read_csv(RESULTS / "exp1" / "per_task.csv")
    tasks = sorted(per.task_id.unique())
    pols = ["W0_workflow", "W1_agent_qwen"]
    fig, ax = plt.subplots(figsize=(11, 2.6), dpi=200)
    for i, p in enumerate(pols):
        for j, t in enumerate(tasks):
            r = per[(per.policy == p) & (per.task_id == t)].iloc[0]
            ax.add_patch(plt.Rectangle((j, i), .94, .86, color=STOP_COLORS.get(r.stop_reason, NEU),
                                       alpha=1 if r.task_success else .35, lw=0))
            ax.text(j + .47, i + .43, ("✓" if r.task_success else "✗"), ha="center", va="center", color="white",
                    fontsize=13, fontweight="bold")
    ax.set_xlim(0, len(tasks)); ax.set_ylim(len(pols), 0)
    ax.set_xticks(np.arange(len(tasks)) + .47, tasks); ax.set_yticks([.43, 1.43], ["W0 workflow", "W1 agente Qwen"])
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    used = [s for s in STOP_COLORS if s in set(per.stop_reason)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=STOP_COLORS[s]) for s in used]
    ax.legend(handles, used, ncol=len(used), loc="upper center", bbox_to_anchor=(.5, -.18), frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "exp1_matriz.png", bbox_inches="tight"); plt.close(fig)


def exp2_bars(folder, out_name: str) -> None:
    s = pd.read_csv(folder / "summary.csv")
    metrics = [("task_success", "task_success"), ("gold_retrieved_at3", "recuerdo correcto\nrecuperado (@3)"),
               ("memory_recall_accuracy", "memory_recall"), ("cross_session_retention", "cross_session"),
               ("stale_memory_error_rate", "stale_error"), ("wrong_memory_use_rate", "wrong_memory_use")]
    conds = list(s.condition)
    colors = ["#bdb7aa", "#8a8579", "#b4532a", "#5b7fa3"]
    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=200)
    w = .2
    for k, c in enumerate(conds):
        vals = [s.loc[s.condition == c, m].iloc[0] for m, _ in metrics]
        xs = np.arange(len(metrics)) + (k - 1.5) * w
        ax.bar(xs, np.nan_to_num(vals), w * .92, color=colors[k], label=c)
        for x, v in zip(xs, vals):
            ax.text(x, (0 if np.isnan(v) else v) + .02, "n/a" if np.isnan(v) else f"{v:.2f}".replace(".", ","),
                    ha="center", fontsize=7.5, color=INK3)
    ax.set_xticks(np.arange(len(metrics)), [l for _, l in metrics]); ax.set_ylim(0, 1.12)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.16), fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / out_name, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    exp1_matrix()
    if (RESULTS / "exp2" / "summary.csv").exists():
        exp2_bars(RESULTS / "exp2", "exp2_metricas_qwen.png")
    if (RESULTS / "exp2" / "diagnostic_reader" / "summary.csv").exists():
        exp2_bars(RESULTS / "exp2" / "diagnostic_reader", "exp2_metricas_lector.png")
    print("figuras en", FIG)


if __name__ == "__main__":
    main()
