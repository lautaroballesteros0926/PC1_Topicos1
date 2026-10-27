"""Genera notebook/Parcial_RAG_evaluable.ipynb (cuaderno unico del examen parcial).

Uso:  python src/make_notebook.py        # escribe el cuaderno sin ejecutar
Ejecutar y guardar salidas (en Docker):
      jupyter nbconvert --to notebook --execute --inplace notebook/Parcial_RAG_evaluable.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# =============================================================================================== 0. portada
md("""
# CC0F4 · Examen parcial — Del prototipo LLM y la recuperación densa a un sistema RAG evaluable
**Pedro Lautaro Quispe Ballesteros** · tema (PC1 → PC2 → parcial): *Sistema multiagente con RAG híbrido para la detección temprana de
riesgo de mercado con alertas sustentadas en evidencia*. Aquí no hay agentes: solo evidencia recuperada, una respuesta RAG evaluada y
una herramienta con contratos.

```text
problema del proyecto (¿qué evidencia sustenta una alerta de riesgo?)
-> consulta (original Q0; la estructurada Q1 se mide aparte)
-> recuperación A densa / B + BM25 + RRF / C + Cross-Encoder          [Experimento 1]
-> evidencia top-k -> respuesta RAG con citas -> auditoría             [Experimento 2]

herramienta local independiente: risk.compute_signal
-> contrato de entrada -> validación de dominio -> ejecución o ToolError [Experimento 3]
```
**Continuidad.** PC1: un LLM de 0.5B no aplicó la regla de riesgo (dijo `low` en todo) → el cálculo de indicadores pasa a una herramienta
determinista (Exp. 3). PC2: la recuperación densa dejó a un solo documento acaparando el ranking (Q01) y se propuso diversificar o reordenar
→ Exp. 1 prueba BM25, fusión y reranker sobre un benchmark mayor y con documentos reales.

**Cómo ejecutar:** `docker compose up` en `parcial/` y abrir <http://localhost:8890/lab>. Estado de cada sección: Exp. 1 y Exp. 3 se
recalculan al ejecutar; Exp. 2 carga las respuestas ya generadas (regenerarlas tarda ~25 min en CPU con `RERUN_EXP2 = True`).
""")
code("""
import json, sys, platform
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
pd.set_option("display.max_colwidth", 140); pd.set_option("display.width", 180)
from common import *            # constantes fijas, rutas, carga del benchmark
from common import read_jsonl
from IPython.display import display
print("python", platform.python_version(), "| pandas", pd.__version__, "| ROOT", ROOT)
""")

# =============================================================================================== 1. benchmark
md("""
## 1. Benchmark `parcial-v1` (datos reales, qrels anotados antes de comparar sistemas)
El examen descarta los conjuntos diminutos (6 a 12 documentos, 4 a 6 consultas): coincidencias fáciles y métricas inestables. Se reemplaza el
corpus de resúmenes propios de la PC2 por **extractos textuales** de documentos públicos en inglés (Reserva Federal, BIS, Wikipedia con
revisión fijada). Procedencia, licencias y hashes: [`data/PROCEDENCIA.md`](../data/PROCEDENCIA.md); construcción y límites:
[`data/README.md`](../data/README.md). Los qrels se redactaron leyendo el corpus y **sin ejecutar ningún sistema**.
""")
code("""
bench = load_benchmark()
docs = bench["documents"]
print("documentos:", len(docs), "| passages:", len(bench["passages"]), "| consultas:", len(bench["queries"]),
      "| pares relevantes:", sum(len(v) for v in bench["qrels"].values()))
print("hashes (sha256, 12):", benchmark_hashes())
pd.DataFrame([{"doc": d["doc_id"], "titulo": d["title"][:70], "editor": d["publisher"], "episodio": d["episode"],
               "passages": len(d["passages"]), "palabras": sum(p["words"] for p in d["passages"])} for d in docs])
""")
code("""
qdf = pd.DataFrame(bench["queries"])
qdf["n_relevantes"] = qdf["query_id"].map(lambda q: len(bench["qrels"][q]))
print(qdf["difficulty"].value_counts().to_dict(), "| consultas del Exp. 2 (RAG):", int(qdf["exp2_rag"].sum()))
qdf[["query_id", "query", "difficulty", "category", "n_relevantes", "exp2_rag"]]
""")

# =============================================================================================== 2. modelo y decoding
md("""
## 2. Modelo y decoding (evidencia de las Semanas 1 y 2)
Un único LLM genera en el Exp. 2: **Qwen2.5-0.5B-Instruct** (el de la PC1 y la PC2), un *decoder-only Transformer* con atención causal.
Las celdas siguientes muestran, con el modelo real, lo que la defensa pide explicar: tokens, Q/K/V, máscara causal, logits → softmax →
greedy, ventana de contexto y KV cache. El codificador de la recuperación es `multilingual-e5-small` (384 dimensiones, 512 tokens).
""")
code("""
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm import LLM_ID, LLM_REVISION, DECODING
tok = AutoTokenizer.from_pretrained(LLM_ID, revision=LLM_REVISION)
qwen = AutoModelForCausalLM.from_pretrained(LLM_ID, revision=LLM_REVISION, dtype=torch.float32).eval()
cfg = qwen.config
print(f"{LLM_ID} @ {LLM_REVISION[:7]} | capas={cfg.num_hidden_layers} | d_model={cfg.hidden_size} | cabezas Q={cfg.num_attention_heads} "
      f"| cabezas K/V={cfg.num_key_value_heads} | vocab={cfg.vocab_size} | ventana={cfg.max_position_embeddings} tokens")
attn = qwen.model.layers[0].self_attn
print("capa 0 -> q_proj", tuple(attn.q_proj.weight.shape), "| k_proj", tuple(attn.k_proj.weight.shape), "| v_proj", tuple(attn.v_proj.weight.shape),
      "(menos cabezas K/V que Q: atención de consultas agrupadas)")
sent = bench["passages"]["D05-P13"][:60]
print("\\ntexto:", repr(sent)); print("tokens Qwen:", tok.convert_ids_to_tokens(tok(sent)["input_ids"])[:14])
""")
code("""
# Logits -> softmax -> greedy: el siguiente token que elige greedy es el argmax de la distribución
sysm = (ROOT / "prompts/closed_book_system.txt").read_text().strip(); usr = (ROOT / "prompts/closed_book_user.txt").read_text().strip()
msgs = [{"role": "system", "content": sysm}, {"role": "user", "content": usr.format(question=bench["queries"][18]["query"])}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
ids = tok(prompt, return_tensors="pt")
with torch.no_grad():
    logits = qwen(**ids).logits[0, -1]
probs = torch.softmax(logits, dim=-1); top = torch.topk(probs, 5)
print("prompt:", ids["input_ids"].shape[1], "tokens")
print(pd.DataFrame({"token": tok.convert_ids_to_tokens(top.indices.tolist()), "logit": logits[top.indices].tolist(), "prob": top.values.tolist()}).round(4))
g = qwen.generate(**ids, max_new_tokens=1, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
print("primer token con greedy:", repr(tok.decode(g[0, -1])), "| argmax:", repr(tok.decode(int(logits.argmax()))), "| iguales:", int(g[0, -1]) == int(logits.argmax()))
""")
code("""
# Máscara causal: cambiar el ÚLTIMO token no altera los logits de las posiciones anteriores
a = ids["input_ids"].clone(); b = a.clone(); b[0, -1] = (b[0, -1] + 1) % cfg.vocab_size
with torch.no_grad():
    la, lb = qwen(input_ids=a).logits, qwen(input_ids=b).logits
print("diferencia máxima en posiciones previas:", float((la[0, :-1] - lb[0, :-1]).abs().max()), "| en la última:", float((la[0, -1] - lb[0, -1]).abs().max()))
# KV cache: mismo texto, mismos 40 tokens, con y sin caché (greedy, así que la salida es idéntica)
short = tok(prompt, return_tensors="pt"); res = {}
for use_cache in (True, False):
    t0 = time.perf_counter()
    out = qwen.generate(**short, max_new_tokens=40, min_new_tokens=40, do_sample=False, use_cache=use_cache, temperature=None, top_p=None, top_k=None,
                        pad_token_id=tok.eos_token_id)
    res[use_cache] = (time.perf_counter() - t0, out[0].tolist())
print(f"con KV cache: {res[True][0]:.1f} s | sin KV cache: {res[False][0]:.1f} s | x{res[False][0] / res[True][0]:.1f} más lento | misma salida: {res[True][1] == res[False][1]}")
""")
md("""
**Decoding elegido: greedy** (`do_sample=False`, `max_new_tokens=200`, `repetition_penalty=1.05`, semilla 42). Justificación: el Exp. 2 compara *con y sin
evidencia* y necesita que la misma pregunta produzca siempre la misma respuesta; con sampling parte de la diferencia sería azar (la PC1 midió
estabilidad de etiqueta de solo 0.75 con T=0.8 y p=0.9). Greedy no mejora la calidad, la hace reproducible. **Ventana de contexto:** el prompt RAG más
largo se mide en el Exp. 2 y se compara con los 32 768 tokens de Qwen. El codificador e5, en cambio, solo admite 512 tokens: el control está en el Exp. 1.
""")
code("del qwen; import gc; gc.collect()")

# =============================================================================================== 3. experimento 1
md("""
## 3. Experimento 1 — Recuperación y reranking (A, B, C)
**Pregunta.** Sobre un mismo benchmark, ¿cómo cambian la calidad del ranking y el costo al añadir recuperación léxica con fusión (B) y un
Cross-Encoder (C) a la recuperación densa (A)?

| | Sistema | Qué cambia respecto del anterior |
|---|---|---|
| **A** (línea base) | `multilingual-e5-small` + `faiss.IndexFlatIP` (vectores L2-normalizados, prefijos `query:`/`passage:`) | – |
| **B** | A + BM25 Okapi + fusión RRF (constante 60) | **dos cambios a la vez**: señal léxica **y** fusión; no se puede atribuir a una sola |
| **C** | B + Cross-Encoder `mmarco-mMiniLMv2-L12-H384-v1` sobre los 10 candidatos | un cambio: el efecto incremental del reranker se aísla mejor |
| X (diagnóstico) | solo BM25 | no es A/B/C; ayuda a ver cuánto aporta la parte léxica |

**Variables fijas.** Corpus, consultas originales (Q0), qrels, segmentación (180 palabras objetivo, overlap de 1 passage), modelo denso, profundidad de
candidatos = 10, top-k ∈ {1, 3, 5}, semilla 42. **Métricas** (Semana 7): Recall@1/3/5, MRR y nDCG@5, calculadas sobre el ranking de *passages* sin
duplicados (los qrels son por `passage_id`), más la latencia por etapa y el bootstrap pareado (5 000 remuestreos) sobre las diferencias por consulta.
""")
code("""
RERUN_EXP1 = True    # False: solo lee results/exp1/ (mismos números, sin recalcular)
from exp1 import run_exp1, CONDITIONS, DIAGNOSTIC
if RERUN_EXP1:
    e1 = run_exp1()
else:
    e1 = {"summary": pd.read_csv(RESULTS / "exp1/summary.csv", index_col=0), "latency": pd.read_csv(RESULTS / "exp1/latency_by_stage.csv"),
          "bootstrap": pd.read_csv(RESULTS / "exp1/bootstrap_pairwise_ci.csv"), "per_query": pd.read_csv(RESULTS / "exp1/per_query.csv"),
          "failures": pd.read_csv(RESULTS / "exp1/failure_types.csv"), "manifest": json.loads((RESULTS / "exp1/run_manifest.json").read_text())}
m = e1["manifest"]
print(f"chunks: {m['n_chunks']} | palabras máx.: {m['chunk_words_max']} | tokens e5 máx.: {m['e5_tokens_max']} (ventana {m['e5_max_seq_length']})")
chunks_df = pd.read_csv(RESULTS / "exp1/chunks.csv")
over = chunks_df[chunks_df["e5_tokens"] > m["e5_max_seq_length"]]
print("chunks que exceden la ventana del encoder (se truncan por la cola):")
over
""")
md("""
**Control de truncación.** 4 de los 102 chunks superan 512 tokens (514, 519, 527 y 608): el encoder (y el Cross-Encoder) descartan la cola de esos chunks.
En la PC2 el control exigía cero excedidos; aquí se **conserva la segmentación de referencia del curso** (180 palabras, igual en Semanas 4, 5 y 7) en vez de
reajustarla tras ver resultados, y se declara como limitación. La pérdida es de entre 2 y 96 tokens de la última passage de cada uno de esos 4 chunks.
""")
code("""
from analysis import recall_ceilings, chunk_coverage_table, load_rankings, chunk_rank_of_relevant, wins_losses
labels = {"A_dense": "A  e5 + FAISS", "B_hybrid_rrf": "B  A + BM25 + RRF", "C_hybrid_rrf_reranker": "C  B + Cross-Encoder", "X_bm25_only": "X  solo BM25 (diagnóstico)"}
summary = e1["summary"].rename(index=labels).round(3)
print("Techo alcanzable con estos qrels:", recall_ceilings(bench["qrels"]), "(Recall@1 máx. = media de 1/|relevantes|)")
summary
""")
code("""
# Estabilidad: ¿las diferencias son más grandes que el ruido entre 22 consultas?  Bootstrap pareado (IC 95 %) de B-A y C-B
bs = e1["bootstrap"].copy()
bs["IC 95 %"] = bs.apply(lambda r: f"[{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]", axis=1)
bs[["metric", "comparison", "delta_mean", "IC 95 %", "ci_crosses_zero", "queries_improved", "queries_worse"]].round(3)
""")
md("""
**Lectura de la tabla (sin declarar superioridad general).**
- **A → B** (léxico + fusión): sube Recall@3 (0.310 → 0.382), MRR (0.541 → 0.619) y nDCG@5 (0.420 → 0.446), baja Recall@5 (0.492 → 0.480). Ningún intervalo de
  bootstrap excluye el cero y hay consultas que mejoran y consultas que empeoran: con 22 consultas **no se distingue de azar**.
- **B → C** (Cross-Encoder): sube Recall@3 (0.382 → 0.442) y Recall@5 (0.480 → 0.611), pero **baja MRR** (0.619 → 0.587) y Recall@1 (0.180 → 0.153). La única diferencia
  cuyo IC excluye el cero es Recall@5 (+0.130, IC [0.002, 0.280]) y apenas: es evidencia débil, no una prueba.
- **BM25 solo (X)** logra Recall@3 = 0.352, por encima de la densa sola (0.310): buena parte de lo que aporta B es la señal léxica, no la fusión.
""")
code("""
# Diagnóstico: Recall@k con la definición de la PC2 (passages cubiertos por los k primeros CHUNKS) — quita el efecto del orden dentro del chunk
rk = load_rankings()
chunk_coverage_table(rk, CONDITIONS + DIAGNOSTIC).rename(index=labels)
""")
md("""
**Por qué se añade este diagnóstico (decisión posterior a ver los números, declarada como tal).** El ranking de passages hereda el orden *dentro* de cada chunk
(orden del documento), así que un chunk correcto en el rank 1 puede dejar la evidencia en la 3.ª o 4.ª passage. Con la definición de la PC2/Semana 4
(passages cubiertos por los k primeros chunks) las cifras suben y el orden entre sistemas cambia: **C queda primero en los tres k (0.352 / 0.611 / 0.728), pero B queda por debajo de A
en k = 3 (0.548 vs 0.575)**, lo contrario de lo que sugería Recall@3 por passage. Moraleja: la ventaja de B sobre A **depende de la métrica**; no la declaro.
""")
code("""
lat = e1["latency"].set_index("condition").rename(index=labels).round(1)
lat["factor vs A"] = (lat["total_ms_mean"] / lat.loc["A  e5 + FAISS", "total_ms_mean"]).round(1)
print(f"construcción del índice denso: {m['dense_index_build_s']} s (una vez) | índice BM25: {m['bm25_index_build_s']} s | repeticiones por consulta: {m['latency_repeats']} (mediana)")
lat
""")
md("""
**Latencia por etapa.** La parte léxica y la fusión cuestan menos de 1 ms; la etapa densa (codificar la consulta) ~40 ms; el **reranker ~1.8 s para 10 candidatos, ~46 veces
el costo de A** en CPU. Es una medición docente (una máquina, sin aislar procesos), no un microbenchmark. El costo se justifica solo si el reranker mejora lo que importa para la
respuesta; en estas 22 consultas su mejora más clara es Recall@5, con pérdida de MRR.
""")
code("""
pq = e1["per_query"]; piv = pq.pivot(index="query_id", columns="condition", values="recall@3")[CONDITIONS + DIAGNOSTIC]
piv.insert(0, "n_rel", [len(bench["qrels"][q]) for q in piv.index])
print("Recall@3 por consulta (passages):")
piv.round(2)
""")
code("""
ft = e1["failures"]
print("Taxonomía de fallos de recuperación (Semana 7) sobre top-3 y candidatos (profundidad 10):")
pd.crosstab(ft["condition"].map(labels), ft["failure_type"])
""")
md("""
### Análisis de errores (≥ 2 consultas con diferencias relevantes)
**Q01 — fallo común y heredado de la PC2** *("¿Cuánto cerró el VIX en el peor día del crash de marzo de 2020?")*. Recall@3 = 0 en A, B, C y X. La evidencia (`D01-P12`: «The CBOE Volatility Index closed at 82.69 on 16
March…») está en los chunks `D01-C11` y `D01-C12`, y **ninguno aparece en el top-10 de ningún sistema**. Dos causas, verificadas en el corpus: (1) *brecha léxica*: el passage dice «CBOE Volatility Index» y
**nunca «VIX»**, así que BM25 no puede conectarlo con la consulta; (2) *distractores fuertes*: los documentos que sí dicen «VIX» (`D06` sobre el índice, `D05` con 45,31, `D20`) ocupan los primeros puestos. El segundo relevante
(`D18-P08`, récord de volatilidad esperada en marzo) entra en el top-10 pero no en las 3 primeras passages. Es el mismo patrón que la PC2 («un documento acapara el ranking») y **ni la fusión ni el reranker lo arreglan**
porque el reranker solo reordena los candidatos y aquí la evidencia no está entre ellos (límite de la generación de candidatos).
""")
code("""
chunk_rank_of_relevant(rk, "Q01").rename(index=labels)
""")
md("""
**Q20 — C acierta donde A y B fallan** *("¿Qué insumos y horizonte usa el cálculo del VIX?")*. Los tres sistemas traen el chunk correcto (`D06-C02`) entre los 3 primeros (rank 2 en A, 3 en B), pero Recall@3 por passage es 0 en A y B porque
los chunks anteriores ocupan las primeras posiciones del ranking de passages; el reranker lo sube al **rank 1** y Recall@3 pasa a 1.0. **Q11 — el reranker corrige el orden pero no la cobertura** *("¿Por qué fallaron los supervisores de la Fed con SVB?")*: el
chunk `D17-C07` está en el rank 6 en A, 4 en B y 1 en C (MRR 0.08 → 0.12 → 1.00), pero Recall@3 solo llega a 0.20 porque las otras 4 passages relevantes están en otros chunks que no suben al top.
**Q13 y Q12 — la señal léxica ayuda** (`uninsured deposits`, `EGRRCPA`): BM25 solo logra Recall@3 de 0.67 en Q13 (A: 0.00) y 1.00 en Q12 (A: 0.00); B recupera parte (0.33 y 1.00). **Q05 — la léxica falla donde la densa acierta** *("cuánto cayó un activo desde su máximo reciente")*:
BM25 solo obtiene Recall@3 = 0 porque el texto dice *drawdown*, mientras A, B y C obtienen 1.0: la fusión conserva lo que la densa sabía. Esta complementariedad es lo que justifica B, aun sin ventaja promedio demostrada.
""")
code("""
for qid in ["Q20", "Q11", "Q13", "Q12", "Q05"]:
    print(qid, bench["queries"][int(qid[1:]) - 1]["query"])
    display(chunk_rank_of_relevant(rk, qid).rename(index=labels)[["first_relevant_chunk_rank", "top3_chunks"]])
""")
md("""
### Registro experimental — Experimento 1
| Pregunta | Línea base | Variables fijas | Modificación | Métricas | Resultado | Error analizado | Limitaciones | Conclusión |
|---|---|---|---|---|---|---|---|---|
| ¿Qué aportan BM25+RRF y el Cross-Encoder a la recuperación densa? | A (e5 + FAISS) | corpus, Q0, qrels, chunks 180/overlap 1, profundidad 10, top-k | A→B: BM25+RRF (2 cambios); B→C: Cross-Encoder | Recall@1/3/5, MRR, nDCG@5, latencia, bootstrap | Recall@3: A 0.310, B 0.382, C 0.442; latencia ≈ 40 / 38 / 1 862 ms | Q01 (nadie lo recupera), Q20, Q11, Q13, Q05 | 22 consultas; 4 chunks truncados; qrels binarios de un anotador; métrica por passage depende del orden en el chunk | Mejoras pequeñas y no distinguibles de azar salvo Recall@5 de C sobre B; C cuesta ~46× más; BM25 y densa son complementarias |

**Limitaciones adicionales.** (1) Los qrels son binarios y pudieron omitir passages útiles (duplicados entre Wikipedia y la Fed); eso castiga a un recuperador que acierta con una passage no anotada. (2) Recall@1 está
acotado por 0.49 (hay entre 1 y 5 relevantes por consulta). (3) Un solo encoder, un solo reranker, una sola segmentación. (4) La latencia mide una laptop con Docker, no producción. (5) El diagnóstico por chunks se añadió
después de ver los resultados principales.
""")

# =============================================================================================== 4. experimento 3
md("""
## 4. Experimento 3 — Herramienta con contratos: `risk.compute_signal` (Semana 6)
**Pregunta.** ¿Se puede convertir una petición probabilística (la de un LLM) en una ejecución controlada, validable y acotada en el tiempo? **Herramienta.** Dada una acción/ETF y una fecha, calcula con la regla de la PC1
el retorno a 5 días, la volatilidad anualizada de 20 días y la caída desde el máximo de 20 días, y devuelve la severidad `low/medium/high`. Es la parte determinista que la PC1 pedía separar del LLM; **no hay agente ni cadena de herramientas**.

```text
ToolCall -> allowlist (protocol) -> contrato de entrada Pydantic (validation) -> dominio (domain) -> política: retry+backoff+timeout+deadline
         -> implementación -> contrato de salida (output_contract) -> ToolResult | ToolError
```
Validar por tipo ≠ por esquema ≠ por dominio: `"TSLA"` es un `str` válido y cumple el patrón, pero no está soportado; `2020-03-14` es una fecha válida, pero fue sábado. **Cuándo reintentar:** solo `transient` (la fuente se recupera
sola) y `timeout` (si queda presupuesto); `validation`, `domain`, `protocol`, `output_contract` y `execution` **no** se reintentan, porque repetir la misma llamada produce el mismo error. Reintentar ≠ reparar: reparar cambiaría los argumentos y eso queda fuera (llevaría a un flujo con agente).
""")
code("""
import inspect, tool_risk
print(json.dumps(tool_risk.TOOL_SPEC, indent=2, ensure_ascii=False))
print(inspect.getsource(tool_risk.ComputeSignalInput)); print(inspect.getsource(tool_risk.RiskSignal))
""")
code("""
from exp3 import run_exp3
e3 = run_exp3()
sc = e3["scenarios"]
cols = ["scenario", "group", "description", "observed_kind", "observed_code", "attempts", "source_calls", "elapsed_ms", "correct_outcome"]
sc[cols]
""")
code("""
# Contraste con un despacho ingenuo (sin allowlist, contrato, errores estructurados, reintentos ni timeout) ante los MISMOS escenarios y fallos
sc2 = sc.assign(naive_ok_but_wrong=(sc.naive_outcome == "ok") & (~sc.group.isin(["valida", "transitorio"])))
print("Con contratos: resultado correcto en", int(sc.correct_outcome.sum()), "de", len(sc), "escenarios; excepciones sin capturar: 0")
print("Ingenuo: excepciones sin capturar en", int((sc.naive_outcome == "uncaught_exception").sum()), "escenarios; acepta en silencio entradas/datos inválidos (S05, S07, S16) (S08 no se ejecuta en el ingenuo: sin allowlist nada impediría despachar `risk.delete_all`).")
sc[["scenario", "group", "naive_outcome", "naive_exception", "naive_elapsed_ms"]]
""")
code("""
pc1 = e3["pc1"]
print("Reproducción de los 8 casos gold de la PC1:", int(pc1["match"].sum()), "de", len(pc1), "coinciden (etiqueta, puntos e indicadores a 2 decimales)")
pc1[["case", "ticker", "date", "gold_label", "tool_label", "ret5_gold", "ret5_tool", "vol20_gold", "vol20_tool", "dd20_gold", "dd20_tool", "match"]]
""")
code("""
print(e3["pytest_output"][-1800:])
""")
md("""
**Lectura.** Con contratos los 16 escenarios terminan en el resultado correcto (incluyendo los que *deben* terminar en error) y ninguno lanza una excepción al llamador. Observaciones concretas: (a) las entradas inválidas (S04–S07) se rechazan **antes de ejecutar**
(0 intentos, 0 lecturas de datos); (b) los errores de dominio (S09–S12) usan 1 solo intento: reintentar no los arregla; (c) el fallo transitorio S13 se recupera en el 3.er intento y S14 agota 3 intentos con `retryable=True` (el error *admite* reintento aunque ya se agotó);
(d) en el timeout S15 el deadline de 0.7 s detiene los reintentos tras 2 intentos aunque `max_attempts=5` (más reintentos ≠ mejor sistema); (e) el despacho ingenuo acepta en silencio `16/03/2020` (S05, la fecha se interpreta por su cuenta), ignora el campo extra (S07) y devuelve datos corruptos como si fueran válidos (S16).
La herramienta reproduce los 8 casos de la PC1 (3 `low`, 2 `medium`, 3 `high`) donde el LLM de 0.5B decía `low` en todos.

**Limitaciones.** Escenarios y fallos están simulados (determinista, sin red); la fuente real son 4 CSV de Yahoo Finance de la PC1; solo 4 tickers; el timeout cancela la corrutina pero no un hilo bloqueado en lectura de disco; no se probó concurrencia.
""")
# =============================================================================================== 5. experimento 2
md("""
## 5. Experimento 2 — RAG y evidencia (closed-book vs con evidencia recuperada)
**Pregunta.** Con el **mismo LLM** (Qwen2.5-0.5B-Instruct, greedy) y las mismas preguntas, ¿qué cambia cuando se le entrega evidencia recuperada con citas a `passage_id`? **Línea base:** closed-book (sin
recuperación). **Fijo:** modelo, 13 preguntas (10 con evidencia + 3 **sin** evidencia en el corpus, donde corresponde abstenerse), decoding, presupuesto de contexto (top-3 chunks). **Modificación:** presencia de evidencia.
La configuración de recuperación se **declaró antes de generar** (`results/exp2/declared_config.json`): C (B + Cross-Encoder), por tener el mayor Recall@3, Recall@5 y nDCG@5 en el Exp. 1, aunque casi todos los IC cruzan cero. Prompts en `prompts/`.

Se evalúan **por separado** (1) corrección de la respuesta, (2) corrección de las citas y (3) faithfulness/grounding, con **auditoría manual** de respuestas, citas y afirmaciones. Las tablas de esta sección que calcula el código son **comprobaciones
mecánicas** (¿el id citado existe en el contexto? ¿está en los qrels? ¿aparecen las cifras?) y proxies léxicos: ni fluidez, ni similitud, ni un LLM como juez se aceptan como prueba de calidad.
""")
code("""
RERUN_EXP2 = False   # True: vuelve a recuperar y generar con Qwen (~25 min en CPU) y SOBRESCRIBE results/exp2/qwen
print(json.dumps(json.loads((RESULTS / "exp2/declared_config.json").read_text()), indent=1)[:900])
if RERUN_EXP2:
    from exp2 import run_exp2
    e2 = run_exp2(); gen, sq = e2["generations"], e2["structured"]
else:
    gen = pd.read_csv(RESULTS / "exp2/qwen/generations_summary.csv"); sq = pd.read_csv(RESULTS / "exp2/qwen/structured_queries.csv")
m2 = json.loads((RESULTS / "exp2/qwen/run_manifest.json").read_text())
print(f"\\nrespuestas: {len(gen)} | prompt RAG más largo: {m2['prompt_tokens_max']} tokens (ventana de Qwen: {m2['llm_context_window']}) -> el contexto cabe holgadamente")
""")
md("""
**Cobertura de la evidencia en el contexto.** Antes de mirar lo que hace el LLM, cuánta de la evidencia relevante *llegó* a su contexto. Es el techo de lo que cualquier generador puede hacer (misma recuperación C del Exp. 1).
""")
code("""
qtext = {**{q["query_id"]: q["query"] for q in bench["queries"]}, **{n["query_id"]: n["query"] for n in read_jsonl(DATA / "queries_sin_evidencia.jsonl")}}
ctx = gen[gen.condition == "rag"][["query_id", "expected_abstain", "relevant_in_context_recall", "n_prompt_tokens"]].copy()
ctx["pregunta"] = ctx["query_id"].map(qtext).str[:95]
print("Passages relevantes presentes en el contexto (0 = el generador no tiene la respuesta):")
ctx.round(2)
""")
code("""
ans = gen.pivot(index="query_id", columns="condition", values="answer")[["closed_book", "rag"]]
ans["closed_book"] = ans["closed_book"].str.replace("\\n", " ").str[:230]; ans["rag"] = ans["rag"].str.replace("\\n", " ").str[:230]
ans
""")
code("""
def auto_summary(df):
    rows = []
    for cond in ["closed_book", "rag"]:
        a = df[(df.condition == cond) & (~df.expected_abstain)]; n = df[(df.condition == cond) & (df.expected_abstain)]
        c = a[a.n_cited > 0]
        rows.append({"condición": cond, "se abstiene (respondibles, n=10)": int(a.abstained.sum()), "se abstiene (sin evidencia, n=3)": int(n.abstained.sum()),
                     "respuestas con cita (de 10)": int((a.n_cited > 0).sum()),
                     "ids citados que existen en el contexto": round(c.cited_in_context_rate.mean(), 2) if len(c) else None,
                     "ids citados que están en los qrels": round(c.cited_relevant_rate.mean(), 2) if len(c) else None,
                     "cifras de la referencia presentes (proxy)": round(a.numeric_fact_proxy.mean(), 2), "F1 léxico vs referencia (proxy)": round(a.token_f1_proxy.mean(), 3)})
    return pd.DataFrame(rows).set_index("condición")
print("Qwen2.5-0.5B — comprobaciones MECÁNICAS (no sustituyen la auditoría manual):")
auto_summary(gen).T
""")
md("""
### Auditoría manual (la completa el autor)
Plantillas: `results/exp2/qwen/audit_answers_template.csv` (26 respuestas: corrección; citas en RAG; abstención) y `audit_claims_template.csv` (19 afirmaciones de las respuestas RAG: respaldada `yes` / `partial` / `no`
por el passage citado o por el contexto). Criterios en el docstring de `src/exp2_audit.py`. La celda siguiente resume lo que se haya completado.
""")
code("""
from exp2_audit import summarize_audit
audit = summarize_audit()
print("Estado de la auditoría manual:", audit["status"])
if audit["answers"] is not None:
    display(audit["answers"]); display(audit["claims"]); print("pendiente:", audit["pending"])
    display(audit["unsupported"])
""")
md("""
### Casos observados (lectura de las respuestas; la auditoría los formaliza)
- **Alucinación con evidencia equivocada (Q07).** El contexto **no contiene** la passage correcta (cobertura 0) y Qwen+RAG responde «Nathan Sheets dissented at the March 2020 emergency meeting…» citando `D01-P02`, que solo dice que Sheets comentó sobre la curva de rendimientos:
  afirmación **sin respaldo** y falsa (quien disintió fue Mester, `D10-P05`). Con el mismo contexto, DeepSeek responde `INSUFFICIENT EVIDENCE` (anexo): el fallo no es solo de recuperación, también de generación.
- **Cita presente ≠ cita correcta (Q19).** Qwen+RAG responde bien («S&P 500 rose 9.52%… biggest one-day gain since 2008»), con la evidencia en su contexto (cobertura 1.0), pero la cita es `[D05-P05]` y la cifra está en `D05-P13`. El contenido está respaldado; la cita, no.
- **Detalle inventado (Q10).** Identifica el *Bank Term Funding Program* pero añade que «raised the interest rate on new BTFP loans in March 2023» (la passage habla de enero de 2024) y cita ids que no corresponden a la passage del BTFP (`D04-P08`).
- **No se abstiene (N01–N03).** Con RAG, Qwen se abstiene 0/3: en N01 **reescribe** la passage del VIX cambiando 2020 por 2008, y en N02 y N03 devuelve solo una cita vacía (`[D20-P03]`) o un fragmento (`[30-year government bond yield]`). En closed-book afirma sin respaldo en N01 («around 150 on March 16, 2008») y N03 (un recorte del Banco de Inglaterra); solo N02 es una abstención correcta.
- **Closed-book abstiene mucho y afirma el resto sin respaldo.** En las 10 preguntas respondibles dice «I don't know» 5 veces (honesto pero inútil) y en otras afirma cosas no sustentadas (Q16: «Nikkei fell 1.7% to 26,891»; Q10: un «Emergency Liquidity Facility» que no es el programa de la Fed).
- **La recuperación fija el techo.** El contexto cubre toda la evidencia solo en Q16 y Q19; en Q01, Q07 y Q09 la cobertura es 0, igual que el Exp. 1 anticipaba (Q01) y sin que reranker o fusión lo arreglaran.
""")
md("""
### Consulta estructurada (Semana 3, interfaz heredada de la PC1/PC2)
El flujo del examen empieza con una consulta estructurada y validada. Para mantener limpia la comparación A/B/C del Exp. 1, **A/B/C usan la consulta original Q0**; aquí se mide aparte qué hace Qwen al convertir cada necesidad en JSON
(`data/schema_consulta.json`, JSON Schema 2020-12; prompt en `prompts/structured_query_*.txt`) sobre las 22 + 3 necesidades. Tres niveles, como en las prácticas anteriores: parseable → schema-valid → adecuada.
""")
code("""
tot = len(sq)
print(f"necesidades: {tot} | parseables: {int(sq.parseable.sum())} | JSON exacto (sin recuperar): {int((sq.parse_mode == 'exact').sum())} | schema-valid: {int(sq.schema_valid.sum())} "
      f"-> schema_valid_rate = {sq.schema_valid.mean():.2f}")
print(f"Q1 idéntica a Q0 (ignorando '?' y mayúsculas): {int(sq.q1_equals_q0.sum())} de {tot} | abstain=true: {int((sq.abstain == True).sum())} | intents: {sq.intent.value_counts().to_dict()}")
sq[~sq.q1_equals_q0][["query_id", "need", "query_q1", "intent"]]
""")
md("""
**Lectura.** Con el prompt en inglés, 25/25 salidas fueron JSON exacto y schema-valid (la PC2, con otro prompt y en español, tuvo 5/6 y todas con cercas de Markdown: **no son comparables**, cambian prompt, idioma y corpus). Pero «valid» no es «adecuada»: en
20 de 25 casos Q1 es **copia literal** de Q0 y en las otras 5 solo recorta palabras; `intent` fue `risk_event` en 23 de 25, incluso para necesidades de indicador (Q05, Q20) y de política monetaria (Q08), y solo Q04 y N03 salieron `policy_response`.
La transformación conserva la necesidad porque casi no la transforma, como en la PC2. Un contrato válido no demuestra que el LLM aporte; no se midió su efecto sobre la recuperación.
""")
md("""
### Registro experimental — Experimento 2
| Pregunta | Línea base | Fijo | Modificación | Métricas | Resultado (mecánico) | Error | Limitaciones | Conclusión |
|---|---|---|---|---|---|---|---|---|
| ¿Cambia la evidencia recuperada las respuestas del mismo LLM? | closed-book | Qwen2.5-0.5B, 13 preguntas, greedy, top-3 chunks de C | + evidencia con citas | corrección, citas y grounding (manual); proxies mecánicos | Qwen RAG: 0/10 abstenciones en respondibles (closed-book: 5/10), 0/3 en sin evidencia; 7/10 con cita, 29 % de ids citados en los qrels | Q07 (alucinación), Q19 (cita mal asignada), Q10 (detalle inventado), N01–N03 | 13 preguntas, un modelo de 0.5B, un auditor, contexto fijo en top-3 chunks | Con este modelo la evidencia elimina el «no sé» pero no garantiza respuestas ni citas fieles; el techo lo marca la recuperación |
""")

# =============================================================================================== 6. anexo deepseek
md("""
## 6. Anexo — la misma corrida con DeepSeek (`deepseek-flash`)
**No forma parte de las conclusiones principales.** Mismas preguntas, mismo contexto recuperado y mismos prompts que Qwen; solo cambia el modelo (API de pago, `temperature=0`). Sin auditoría manual: únicamente comprobaciones mecánicas. Las respuestas crudas, el modelo reportado por
la API, el uso de tokens y la fecha están en `results/exp2/deepseek_anexo/`. Una primera corrida con `max_tokens=300` se descartó porque el modelo razona antes de responder y cortó 8 de 26 respuestas (`run0_max300_truncado/LEEME.md`); se repitió con 2 000.
""")
code("""
ds_path = RESULTS / "exp2/deepseek_anexo/generations_summary.csv"
if ds_path.exists():
    ds = pd.read_csv(ds_path)
    print("modelo reportado por la API:", ds.model_reported.unique().tolist(), "| finish_reason:", ds.finish_reason.unique().tolist(), "| tokens de razonamiento medios:", round(ds.reasoning_tokens.mean(), 1))
    display(auto_summary(ds).T)
else:
    print("Anexo no ejecutado (falta results/exp2/deepseek_anexo/): requiere DEEPSEEK_API_KEY en parcial/.env y `python src/exp2_deepseek.py`.")
""")
md("""
**Lectura (solo mecánica).** Con la misma evidencia, DeepSeek se abstiene 3/3 en las preguntas sin evidencia (Qwen 0/3), cita siempre ids del contexto y el 83 % apunta a passages relevantes (Qwen 29 %). Se abstiene también en Q07 y Q09, donde el contexto no tenía la respuesta: conducta coherente con la evidencia.
En Q01 responde con una cifra real pero del día equivocado (75,47 el 12-mar, de `D06-P07`; el cierre del peor día, 82,69, no llegó al contexto). Cerrado (sin recuperación), DeepSeek no se abstiene en ninguna respondible y responde también N01 y N03 con su conocimiento propio, sin respaldo del corpus (no verifiqué esas respuestas).
Esto sugiere que el cuello de botella de Qwen es el generador y el de ambos, la recuperación; **no** autoriza a concluir que DeepSeek sea «mejor»: 13 preguntas, sin auditoría y un modelo de pago que cambia de versión.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
if __name__ == "__main__":
    out = ROOT / "notebook" / "Parcial_RAG_evaluable.ipynb"
    nbf.write(nb, out)
    print("escrito", out, "| celdas:", len(cells))
