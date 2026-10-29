"""Figuras de las diapositivas, generadas desde results/ y data/ (nada se escribe a mano).

Uso:  python src/make_figures.py      -> slides/fig/*.png (200 dpi)
Paleta categórica validada (orden fijo): A azul, B naranja, C aqua; BM25 solo en gris (diagnóstico).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, ROOT, read_jsonl  # noqa: E402

OUT = ROOT / "slides" / "fig"
OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, INK3, GRID = "#1B2027", "#4B5563", "#8A91A0", "#E3E6EC"
A, B, C, X = "#2a78d6", "#eb6834", "#1baf7a", "#9AA1AE"
GOOD, WARN, BAD = "#1baf7a", "#eda100", "#e34948"
COND = {"A_dense": ("A · densa", A), "B_hybrid_rrf": ("B · + BM25 + RRF", B), "C_hybrid_rrf_reranker": ("C · + reranker", C),
        "X_bm25_only": ("solo BM25", X)}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": INK3, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200})


def coma(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


from matplotlib.ticker import FuncFormatter  # noqa: E402


def comma_axis(ax, which="y", d=1):
    f = FuncFormatter(lambda v, _: f"{v:.{d}f}".replace(".", ",").replace("-", "\u2212"))
    (ax.yaxis if which == "y" else ax.xaxis).set_major_formatter(f)


def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ corpus
docs = read_jsonl(DATA / "corpus_parcial.jsonl")
rows = [{"episodio": d["episode"].split(" (")[0] if not d["episode"].startswith("Indicador") else "Indicadores",
         "fuente": "Wikipedia" if d["kind"] == "wiki" else "Fed y BIS (oficial)", "n": len(d["passages"])} for d in docs]
cp = pd.DataFrame(rows).pivot_table(index="episodio", columns="fuente", values="n", aggfunc="sum", fill_value=0)
order = ["COVID-19", "Alzas de tasas", "SVB", "Carry trade", "Aranceles", "Indicadores"]
cp = cp.reindex(order)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
left = np.zeros(len(cp))
for col, colr in (("Fed y BIS (oficial)", A), ("Wikipedia", B)):
    v = cp[col].to_numpy()
    ax.barh(cp.index, v, left=left, color=colr, height=0.62, edgecolor="white", linewidth=2, label=col)
    for i, (l, w) in enumerate(zip(left, v)):
        if w >= 4:
            ax.text(l + w / 2, i, str(int(w)), ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    left += v
for i, t in enumerate(left):
    ax.text(t + 0.6, i, f"{int(t)}", va="center", color=INK, fontsize=12)
ax.invert_yaxis(); ax.set_xlabel("passages"); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.legend(frameon=False, loc="lower right", fontsize=11)
save(fig, "corpus.png")

# ------------------------------------------------------------------ exp1: métricas
s = pd.read_csv(RESULTS / "exp1/summary.csv", index_col=0)
metrics = ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5"]
labels = ["Recall@1", "Recall@3", "Recall@5", "MRR", "nDCG@5"]
fig, ax = plt.subplots(figsize=(8.4, 3.9))
x = np.arange(len(metrics)); w = 0.26
for k, cond in enumerate(["A_dense", "B_hybrid_rrf", "C_hybrid_rrf_reranker"]):
    name, colr = COND[cond]
    vals = s.loc[cond, metrics].to_numpy()
    bars = ax.bar(x + (k - 1) * w, vals, width=w - 0.03, color=colr, label=name)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.012, coma(v), ha="center", va="bottom", fontsize=9.5, color=INK2)
ax.set_xticks(x, labels); ax.set_ylim(0, 0.75); ax.set_ylabel("media en 22 consultas"); ygrid(ax); comma_axis(ax, "y", 1)
ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=11)
save(fig, "exp1_metricas.png")

# ------------------------------------------------------------------ exp1: bootstrap
bs = pd.read_csv(RESULTS / "exp1/bootstrap_pairwise_ci.csv")
nice = dict(zip(metrics, labels))
bs["m"] = bs["metric"].map(nice)
bs["cmp"] = bs["comparison"].map({"B_hybrid_rrf - A_dense": "B − A (léxico + fusión)", "C_hybrid_rrf_reranker - B_hybrid_rrf": "C − B (reranker)"})
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ypos, yl = [], []
for i, m in enumerate(labels):
    for j, (cmp_, colr) in enumerate((("B − A (léxico + fusión)", B), ("C − B (reranker)", C))):
        r = bs[(bs.m == m) & (bs.cmp == cmp_)].iloc[0]
        y = i * 3 + j
        ax.plot([r.ci95_low, r.ci95_high], [y, y], color=colr, lw=2.5, solid_capstyle="round")
        excl = not r.ci_crosses_zero
        ax.scatter([r.delta_mean], [y], s=70, color=colr if excl else "white", edgecolor=colr, linewidth=2, zorder=3)
        if excl:
            ax.text(r.ci95_high + 0.01, y, "IC sin cero", va="center", fontsize=10.5, color=INK)
        ypos.append(y); yl.append(f"{m}  ·  {cmp_.split(' (')[0]}")
ax.axvline(0, color=INK2, lw=1.2); ax.set_xlim(-0.22, 0.42); comma_axis(ax, "x", 1)
ax.set_yticks(ypos, yl, fontsize=11); ax.invert_yaxis(); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.set_xlabel("diferencia de medias con IC 95 % (bootstrap pareado, 5 000 remuestreos)")
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
from matplotlib.lines import Line2D  # noqa: E402
ax.legend(handles=[Line2D([0], [0], color=B, lw=2.5, label="B − A (léxico + fusión)"), Line2D([0], [0], color=C, lw=2.5, label="C − B (reranker)")],
          frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, fontsize=10.5)
save(fig, "exp1_bootstrap.png")

# ------------------------------------------------------------------ exp1: latencia
lat = pd.read_csv(RESULTS / "exp1/latency_by_stage.csv").set_index("condition")
fig, ax = plt.subplots(figsize=(6.6, 2.6))
conds = ["A_dense", "B_hybrid_rrf", "C_hybrid_rrf_reranker"]
vals = [lat.loc[c, "total_ms_mean"] for c in conds]
ax.barh([COND[c][0] for c in conds], vals, color=[COND[c][1] for c in conds], height=0.55)
for i, v in enumerate(vals):
    ax.text(v * 1.12, i, f"{v:,.0f} ms".replace(",", " "), va="center", fontsize=12, color=INK)
ax.set_xscale("log"); ax.set_xlim(10, 10000); ax.invert_yaxis()
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
ax.set_xlabel("latencia media por consulta (escala logarítmica)")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
save(fig, "exp1_latencia.png")

# ------------------------------------------------------------------ exp1: Recall@3 por consulta
pq = pd.read_csv(RESULTS / "exp1/per_query.csv")
qs = {q["query_id"]: q for q in read_jsonl(DATA / "queries_parcial.jsonl")}
piv = pq.pivot(index="query_id", columns="condition", values="recall@3")[["A_dense", "B_hybrid_rrf", "C_hybrid_rrf_reranker", "X_bm25_only"]]
fig, ax = plt.subplots(figsize=(5.0, 7.4))
im = ax.imshow(piv.to_numpy(), cmap="Blues", vmin=0, vmax=1.15, aspect="auto")
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.iat[i, j]
        ax.text(j, i, coma(v, 2) if v not in (0, 1) else str(int(v)), ha="center", va="center", fontsize=9.5,
                color="white" if v > 0.6 else INK)
ax.set_xticks(range(4), ["A", "B", "C", "BM25"], fontsize=11); ax.xaxis.tick_top()
ax.set_yticks(range(len(piv)), [f"{q} · {qs[q]['difficulty']}" for q in piv.index], fontsize=9.5)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0)
for i, q in enumerate(piv.index):
    if q == "Q01":
        ax.add_patch(plt.Rectangle((-0.5, i - 0.5), 4, 1, fill=False, edgecolor=BAD, lw=2))
save(fig, "exp1_por_consulta.png")

# ------------------------------------------------------------------ exp2: cobertura y auditoría
gen = pd.read_csv(RESULTS / "exp2/qwen/generations_summary.csv")
rag = gen[(gen.condition == "rag") & (~gen.expected_abstain)].copy()
fig, ax = plt.subplots(figsize=(6.6, 3.0))
ax.bar(rag.query_id, rag.relevant_in_context_recall, color=A, width=0.62)
for i, v in enumerate(rag.relevant_in_context_recall):
    ax.text(i, v + 0.03, coma(v, 2) if v not in (0, 1) else str(int(v)), ha="center", fontsize=10.5, color=INK2)
ax.set_ylim(0, 1.18); ax.set_ylabel("evidencia relevante\nque llegó al contexto"); ygrid(ax); comma_axis(ax, "y", 1)
save(fig, "exp2_cobertura.png")

aud = pd.read_csv(RESULTS / "exp2/qwen/audit_answers_template.csv")
cl = pd.read_csv(RESULTS / "exp2/qwen/audit_claims_template.csv")


def frac(df, col, cond, expected):
    d = df[(df.condition == cond) & (df.expected_abstain == expected)][col].dropna()
    return d.sum(), len(d)


groups = [("Respuesta correcta\n(10 con respuesta)", "answer_correct_0_1", False),
          ("Se abstiene bien\n(3 sin evidencia)", "abstention_appropriate_0_1", True)]
fig, ax = plt.subplots(figsize=(8.8, 3.8))
xs = np.arange(4); wd = 0.36
for k, (cond, colr, name) in enumerate((("closed_book", X, "sin evidencia (closed-book)"), ("rag", A, "con evidencia (RAG)"))):
    for i, (lab, col, exp_) in enumerate(groups):
        n, d = frac(aud, col, cond, exp_)
        ax.bar(xs[i] + (k - 0.5) * wd, n / d, width=wd - 0.04, color=colr, label=name if i == 0 else None)
        ax.text(xs[i] + (k - 0.5) * wd, n / d + 0.03, f"{int(n)}/{d}", ha="center", fontsize=11.5, color=INK)
cit = aud[aud.condition == "rag"]["citations_correct_0_1"].dropna()
ax.bar(xs[2] + 0.5 * wd, cit.mean(), width=wd - 0.04, color=A)
ax.text(xs[2] + 0.5 * wd, cit.mean() + 0.03, f"{int(cit.sum())}/{len(cit)}", ha="center", fontsize=11.5, color=INK)
score = cl.supported_yes_partial_no.map({"yes": 1, "partial": 0.5, "no": 0})
ax.bar(xs[3] + 0.5 * wd, score.mean(), width=wd - 0.04, color=A)
ax.text(xs[3] + 0.5 * wd, score.mean() + 0.03, coma(score.mean(), 2), ha="center", fontsize=11.5, color=INK)
for i in (2, 3):
    ax.text(xs[i] - 0.5 * wd, 0.03, "no aplica", ha="center", fontsize=9.5, color=INK3, rotation=90, va="bottom")
ax.set_xticks(xs, [g[0] for g in groups] + ["Citas correctas\n(13 respuestas RAG)", "Afirmaciones\nrespaldadas (19)"], fontsize=10.5)
ax.set_ylim(0, 1.1); ax.set_yticks([0, .25, .5, .75, 1], ["0 %", "25 %", "50 %", "75 %", "100 %"]); ygrid(ax)
ax.legend(frameon=False, loc="upper right", fontsize=10.5)
save(fig, "exp2_auditoria.png")

# ------------------------------------------------------------------ exp3: contratos vs ingenuo
sc = pd.read_csv(RESULTS / "exp3/scenarios_summary.csv")


def naive_cat(r):
    if r.naive_outcome == "uncaught_exception":
        return "excepción sin capturar"
    if r.group == "valida":
        return "resultado esperado"
    return "acepta en silencio / sin control"


sc["naive_cat"] = sc.apply(naive_cat, axis=1)
cats = [("resultado esperado", GOOD), ("acepta en silencio / sin control", WARN), ("excepción sin capturar", BAD)]
counts = {"Con contratos": {"resultado esperado": int(sc.correct_outcome.sum())},
          "Despacho ingenuo": sc.naive_cat.value_counts().to_dict()}
fig, ax = plt.subplots(figsize=(7.4, 2.4))
for i, (row, cnt) in enumerate(counts.items()):
    left = 0
    for cat, colr in cats:
        v = cnt.get(cat, 0)
        if v:
            ax.barh(i, v, left=left, color=colr, height=0.6, edgecolor="white", linewidth=2)
            ax.text(left + v / 2, i, str(v), ha="center", va="center", color="white" if colr != WARN else INK, fontsize=13, fontweight="bold")
            left += v
ax.set_yticks([0, 1], list(counts)); ax.invert_yaxis(); ax.set_xlim(0, 16); ax.set_xlabel("16 escenarios (mismos fallos inyectados)")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color=c, label=l) for l, c in cats], frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.32), fontsize=10.5)
save(fig, "exp3_escenarios.png")

print("figuras en", OUT, sorted(p.name for p in OUT.glob("*.png")))
