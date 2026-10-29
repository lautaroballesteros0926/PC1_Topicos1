# CC0F4 · Examen parcial — Del prototipo LLM y la recuperación densa a un sistema RAG evaluable

**Pedro Lautaro Quispe Ballesteros** · CC0F4 Tópicos de Ciencia de la Computación I (2026-2) · Semanas 1 a 7 · modalidad individual.
Commit evaluado: `8aa4ca0` (`8aa4ca0cd854cc05d6185f0579f72d2ac41e0ff1`, repositorio `lautaroballesteros0926/PC1_Topicos1`, rama `main`).

Tema de investigación (el mismo de la PC1 y la PC2): *Sistema multiagente con RAG híbrido para la detección temprana de riesgo de mercado con alertas sustentadas en evidencia*.
Este examen **no implementa agentes**: construye y evalúa la base que los agentes necesitarán (evidencia recuperada, respuesta con citas y una herramienta con contratos) y dice con datos dónde falla.

## 1. Problema y continuidad con PC1 y PC2

```text
problema del proyecto: ¿qué evidencia sustenta una alerta de riesgo de mercado?
-> consulta (original Q0; la estructurada Q1 se mide aparte)
-> recuperación  A densa / B + BM25 + RRF / C + Cross-Encoder          [Experimento 1]
-> evidencia top-k -> respuesta RAG con citas -> auditoría              [Experimento 2]

herramienta local independiente  risk.compute_signal
-> contrato de entrada -> validación de dominio -> ejecución o ToolError [Experimento 3]
```

| Práctica | Qué mostró | Qué se hereda en el parcial |
|---|---|---|
| **PC1** (`../README.md`) | Un LLM de 0.5B (Qwen2.5-0.5B-Instruct) devolvió JSON válido pero dijo `low` en los 8 casos: no aplica la regla de riesgo. Conclusión: separar el cálculo de indicadores del LLM. | La regla y los 4 CSV de precios → herramienta del Exp. 3 (reproduce los 8 casos). El mismo modelo → Exp. 2. |
| **PC2** (`../pc2/README.md`) | 8 documentos (resúmenes propios), 6 consultas, e5-small + FAISS. Un documento acaparaba el ranking (Q01) y se propuso diversificar o reordenar. Q1 copiaba a Q0. | Misma recuperación densa (sistema **A**), mismo `IndexFlatIP`, mismo chunking de 180 palabras y overlap 1, misma interfaz JSON. Las 6 consultas, traducidas. |

`pc1/` y `pc2/` no se modificaron: se conserva la evidencia previa. La PC2 queda como antecedente, **no como línea base numérica**: el corpus y las consultas cambian por exigencia del examen.

## 2. Cómo ejecutar

Requisitos: Docker Desktop abierto. Desde esta carpeta (`parcial/`):

```bash
docker compose up
```
Abrir <http://localhost:8890/lab> y entrar a `notebook/Parcial_RAG_evaluable.ipynb` (ya trae las salidas guardadas). Detener: `Ctrl+C` y `docker compose down`.

| Qué | Comando (dentro del contenedor) | Tiempo aprox. (CPU) |
|---|---|---|
| Pruebas (28) | `docker compose run --rm parcial python -m pytest tests -q` | 3 s |
| Experimento 1 | `docker compose run --rm parcial python src/exp1.py` | 4 min |
| Experimento 3 | `docker compose run --rm parcial python src/exp3.py` | 15 s |
| Experimento 2 (Qwen) | `docker compose run --rm parcial python src/exp2.py` | 25 min |
| Anexo DeepSeek | `docker compose run --rm parcial python src/exp2_deepseek.py` (necesita `.env`, ver abajo) | 3 min |
| Cuaderno completo | `docker compose run --rm parcial jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebook/Parcial_RAG_evaluable.ipynb` | 10 min |
| Regenerar datos | `python src/build_corpus.py --download` (requiere `pdftotext`) y `python src/build_benchmark.py` | 1 min |

Anexo DeepSeek: copiar `.env.example` a `.env` y poner `DEEPSEEK_API_KEY=...` (`.env` está en `.gitignore`). Si ya existe `results/exp2/deepseek_anexo/generations.jsonl` no se vuelve a llamar a la API.
Los modelos (`Qwen2.5-0.5B-Instruct`, `multilingual-e5-small`, `mmarco-mMiniLMv2-L12-H384-v1`) se descargan la primera vez al volumen `pc1_hf` (compartido con PC1 y PC2). Semilla 42. Entorno de la corrida: `results/exp1/run_manifest.json` y `results/exp2/qwen/run_manifest.json`.

## 3. Corpus, consultas y qrels

**21 documentos reales y 150 passages textuales** (27 a 242 palabras, media 105; sin traducir ni resumir), todos en inglés: 8 artículos de Wikipedia (CC BY-SA 4.0, revisión fijada con `oldid`), 8 comunicados de la Reserva Federal, 3 informes de la Fed (revisión de supervisión de SVB, Financial Stability Report 2020 y 2025) y 2 publicaciones del BIS (Bulletin 90 y Quarterly Review de marzo de 2020). Cubren los episodios de la PC1: COVID-19 (2020), alzas de tasas (2022), SVB (2023), carry trade (2024) y aranceles (2025), más los indicadores que usa su regla (VIX, drawdown).
Procedencia, URL, versión, hash y licencia de cada documento: [`data/PROCEDENCIA.md`](data/PROCEDENCIA.md). Construcción, protocolo de anotación y límites: [`data/README.md`](data/README.md).

- **22 consultas** en inglés con evidencia (4 `easy`, 10 `medium`, 8 `hard`), 57 pares relevantes (1 a 5 por consulta), **qrels sobre `passage_id`** (no sobre chunks). Seis vienen de la PC2.
- **Consultas difíciles y distractores:** pasajes casi idénticos entre comunicados FOMC (D12 a D15), hechos de SVB repetidos en 4 documentos, otros máximos del VIX que compiten con el cierre de 82,69, y una evidencia que dice «CBOE Volatility Index» y nunca «VIX» (Q01).
- **3 consultas sin evidencia** en el corpus (máximo del VIX en 2008, mercados peruanos, decisión del Banco de Inglaterra) para el Exp. 2.
- **Anotación y congelamiento.** Los qrels y las respuestas de referencia se redactaron leyendo el corpus y **antes de ejecutar ningún sistema**; no se generaron con las salidas del sistema evaluado ni se hizo *pooling*. El autor los revisó pregunta por pregunta contra los passages marcados. Hashes en `data/benchmark_manifest.json`; no se modificaron tras ver resultados.

Estos mínimos permiten una evaluación docente más significativa, pero **no prueban generalización estadística**.

## 4. Arquitectura y estructura del repositorio

```text
parcial/
├── README.md
├── Dockerfile, docker-compose.yml, requirements.txt, .env.example
├── notebook/Parcial_RAG_evaluable.ipynb     # cuaderno único, con salidas
├── src/  common.py (constantes fijas)  retrieval.py (chunks, e5+FAISS, BM25, RRF, Cross-Encoder)  metrics.py  analysis.py
│        exp1.py  exp2.py  exp2_audit.py  exp2_deepseek.py  exp3.py  tool_risk.py  risk_rules.py  llm.py  rag.py
│        build_corpus.py  build_benchmark.py  make_notebook.py
├── data/    corpus_parcial.jsonl  queries_parcial.jsonl  qrels_parcial.json  reference_answers_parcial.jsonl
│            queries_sin_evidencia.jsonl  schema_consulta.json  prices/ (CSV de la PC1)  pc1_casos_gold.csv  PROCEDENCIA.md  README.md
├── prompts/ rag_*.txt  closed_book_*.txt  structured_query_*.txt
├── tests/   test_retrieval.py  test_tool_risk.py                 # 28 pruebas
├── results/ exp1/  exp2/qwen/  exp2/deepseek_anexo/  exp3/  declared_config.json
└── slides/Examen_Parcial_CC0F4.pdf
```

Archivos con **salidas originales** para comprobar los cálculos: `results/exp1/` (`per_query.csv`, `summary.csv`, `latency_by_stage.csv`, `bootstrap_pairwise_ci.csv`, `failure_types.csv`, `rankings.jsonl` con el top-10 de cada consulta y condición, `chunks.csv`, `run_manifest.json`), `results/exp2/qwen/` (`generations.jsonl` con el prompt exacto de cada respuesta, `generations_summary.csv`, `structured_queries.csv`, plantillas y auditoría), `results/exp3/` (`scenarios.jsonl`, `scenarios_summary.csv`, `pc1_reproduction.csv`, `pytest_output.txt`).

**Variables fijas.** Segmentación: 180 palabras objetivo, overlap de 1 passage (102 chunks). Densa: `intfloat/multilingual-e5-small` (384 dimensiones), prefijos `query:`/`passage:`, vectores L2-normalizados, `faiss.IndexFlatIP`. BM25 Okapi (`k1=1.5`, `b=0.75`, IDF con piso `epsilon=0.25`, sin stemming, stopwords mínimas en inglés). RRF con constante 60. Reranker `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` sobre 10 candidatos. Top-k ∈ {1, 3, 5}. LLM `Qwen/Qwen2.5-0.5B-Instruct` (revisión `7ae5576…`), greedy, `max_new_tokens=200`, `repetition_penalty=1.05`. Bootstrap pareado de 5 000 remuestreos. Semilla 42.

## 5. Matriz de trazabilidad: las siete semanas

| Sem. | Recurso revisado | Concepto | Dónde se aplica o analiza | Evidencia verificable | Limitación o conclusión propia |
|---|---|---|---|---|---|
| **1** | `Semana1/Cuaderno1-CC-0F4.ipynb`, `Semana1/Laboratorio1-CC-0F4.ipynb`, `Semana1/Lectura1-CC-0F4.md` | Tokens, embeddings, Q/K/V, atención y máscara causal; arquitectura del modelo elegido | Cuaderno §2 «Modelo y decoding» (3 celdas): config de Qwen, formas de `q_proj`/`k_proj`/`v_proj`, tokenización, prueba de máscara causal | Qwen2.5-0.5B: 24 capas, d_model 896, **14 cabezas de consulta y 2 de clave/valor** (`q_proj` 896×896, `k_proj`/`v_proj` 128×896: atención de consultas agrupadas); máscara causal: al cambiar el último token la diferencia máxima en los logits de las posiciones previas es **0,0** (19,88 en la última) | Solo se inspecciona Qwen2.5-0.5B (pesos locales). De DeepSeek (anexo, API) no se ven pesos ni logits; su arquitectura queda en la guía de estudio. |
| **2** | `Semana2/Cuaderno2-CC-0F4.ipynb`, `Semana2/Laboratorio2-CC-0F4.ipynb`, `Semana2/Lectura3-CC-0F4.md` | Logits, softmax, greedy vs sampling, ventana de contexto, KV cache; justificar el decoding | Cuaderno §2: logits→softmax→argmax; KV cache con y sin caché; Exp. 2 (decoding fijo, tokens de prompt vs ventana) | Primer token con greedy = argmax (`I`, prob. 0,62); **KV cache: 11,2 s con caché vs 42,3 s sin caché (×3,8)**, misma salida; prompt RAG más largo 1 571 tokens vs ventana de 32 768 | Greedy se elige por reproducibilidad, no por calidad; no se comparó con sampling en este examen (la PC1 sí: estabilidad 0.75 con T=0.8). |
| **3** | `Semana3/Cuaderno3-CC-0F4.ipynb`, `Semana3/Exposiciones1_CC0F4.md` | Instrucción/contexto separados, JSON Schema, validación estructural y semántica, parseable vs schema-valid vs adecuada | `prompts/structured_query_*.txt`, `data/schema_consulta.json`; Cuaderno §5 «Consulta estructurada»; contrato Pydantic del Exp. 3 | 25/25 salidas JSON exacto y schema-valid; **20/25 con Q1 = Q0** (copia); intent `risk_event` en 23/25 | Válido ≠ adecuado: el contrato no obliga a transformar la consulta. No se midió su efecto en la recuperación; A/B/C usan Q0. |
| **4** | `Semana4/Cuaderno4-CC-0F4.ipynb`, `Semana4/Laboratorio4-CC-0F4.ipynb` | Chunking con overlap, embeddings, similitud, FAISS, top-k, qrels; línea base de la PC2 | `src/retrieval.py::build_chunks` y `Retrievers.dense`; Cuaderno §3 (sistema A); `results/exp1/chunks.csv` | 102 chunks; A: Recall@3 0.310, MRR 0.541; control de truncación: **4 chunks > 512 tokens** | La segmentación del curso deja 4 chunks truncados en la cola (2 a 96 tokens); se declara, no se reajustó tras ver resultados. |
| **5** | `Semana5/Cuaderno5-CC-0F4.ipynb`, `Semana5/Lectura5-CC-0F4.md` | BM25, recuperación híbrida, RRF, Cross-Encoder, RAG; comparar arquitecturas | `src/retrieval.py` (BM25Okapi, `rrf`, `rerank`); Cuaderno §3 (B y C); Cuaderno §5 (RAG) | Tabla A/B/C + bootstrap + BM25 solo (diagnóstico); latencia ≈ 40 / 38 / 1 862 ms | B mezcla léxico y fusión (no atribuible); solo B→C aísla el reranker. Casi todos los IC cruzan 0. |
| **6** | `Semana6/Cuaderno6-CC-0F4.ipynb`, `Semana6/Lectura6-CC-0F4.md` | Function calling, contratos, validación de dominio, errores, retries, timeouts | `src/tool_risk.py`, `src/exp3.py`, `tests/test_tool_risk.py`; Cuaderno §4 | 16 escenarios correctos de 16; 8/8 casos de la PC1 reproducidos; 28 pruebas pasan; ingenuo: 8 excepciones sin capturar | Fallos simulados y 4 tickers; el timeout cancela corrutinas, no hilos bloqueados. No hay LLM llamando la herramienta (fuera de alcance). |
| **7** | `Semana7/Cuaderno7-CC-0F4.ipynb`, `Semana7/Lectura7-CC-0F4.md` | Recall@k, MRR, nDCG, grounding, corrección de citas, latencia, análisis de errores | `src/metrics.py`, `src/exp2_audit.py`, `src/rag.py`; Cuaderno §3 (métricas, taxonomía de fallos) y §5 (auditoría) | `results/exp1/*.csv`; plantillas de auditoría y su resumen; casos Q01, Q07, Q19 | Qrels binarios y de un anotador; las métricas por passage dependen del orden dentro del chunk (se añadió un diagnóstico por chunk, declarado como posterior). |

## 6. Experimento 1 — Recuperación y reranking

**Pregunta.** Con el mismo benchmark, ¿cómo cambian la calidad del ranking y el costo al añadir BM25+RRF y un Cross-Encoder a la recuperación densa? **Línea base:** A. **Fijo:** corpus, consultas originales Q0, qrels, segmentación, modelo denso, profundidad 10, top-k, semilla. **Modificación:** A→B añade BM25 **y** fusión RRF (dos cambios a la vez: no se puede atribuir a una sola); B→C añade el Cross-Encoder (el efecto incremental del reranker se aísla mejor). Se añadió **BM25 solo** como diagnóstico (no es A/B/C).

| | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 | Latencia total (media) |
|---|---|---|---|---|---|---|
| **A** e5 + FAISS | 0.153 | 0.310 | 0.492 | 0.541 | 0.420 | 40 ms |
| **B** A + BM25 + RRF | 0.180 | 0.382 | 0.480 | 0.619 | 0.446 | 38 ms |
| **C** B + Cross-Encoder | 0.153 | 0.442 | 0.611 | 0.587 | 0.500 | 1 862 ms (reranker 1 823 ms) |
| X solo BM25 (diagnóstico) | 0.170 | 0.352 | 0.488 | 0.579 | 0.428 | 0.3 ms |

Techo con estos qrels: Recall@1 ≤ 0.49, Recall@3 ≤ 0.93. Latencia por etapa: densa ~37 ms, BM25 0.6 ms, RRF < 0.1 ms, reranker ~1.8 s (mediana de 3 repeticiones por consulta; p95 total de C: 2.0 s). La latencia varía entre corridas (una corrida previa dio ~31 ms para A y ~1,7 s para C).

**Incertidumbre (bootstrap pareado, 22 consultas, IC 95 %).** Solo una diferencia excluye el cero: **Recall@5, C − B = +0.130 [0.002, 0.280]**, y apenas. A→B: Recall@3 +0.072 [−0.015, +0.186], MRR +0.079 [−0.005, +0.186]. B→C: MRR −0.032 [−0.186, +0.127].
**Diagnóstico por chunk** (definición de la PC2: passages cubiertos por los k primeros chunks; añadido después de ver los resultados): C 0.352 / 0.611 / 0.728 en k = 1 / 3 / 5; A 0.263 / 0.575 / 0.692; B 0.312 / 0.548 / 0.686. B queda **por debajo de A en k = 3** con esta definición, lo contrario de Recall@3 por passage: la ventaja de B depende de la métrica.

**Análisis de errores (≥ 2 consultas).**
- **Q01** *(¿cuánto cerró el VIX en el peor día de marzo de 2020?)*: Recall@3 = 0 en A, B, C y X. La evidencia (`D01-P12`) está en `D01-C11`/`D01-C12`, que no entran en el top-10 de ningún sistema. Causa verificada: el passage dice «CBOE Volatility Index» y **no contiene «VIX»** (brecha léxica para BM25) y los documentos que sí dicen «VIX» (D06, D05, D20) lo desplazan. Es el patrón de la PC2: ni la fusión ni el reranker lo arreglan, porque el reranker solo reordena candidatos y aquí la evidencia no es candidato.
- **Q20** (C acierta, A y B no): el chunk correcto está en el rank 2 (A) y 3 (B); el reranker lo sube al 1 y Recall@3 pasa de 0 a 1.
- **Q11** (el reranker arregla el orden, no la cobertura): MRR 0.08 → 0.12 → 1.00, pero Recall@3 solo 0.20.
- **Q13 y Q12** (la léxica ayuda): BM25 solo logra 0.67 y 1.00 donde A logra 0.00. **Q05** (la léxica falla): BM25 solo 0.00 donde A, B, C logran 1.00 («drawdown» vs «cuánto cayó desde su máximo»).

**Limitaciones.** 22 consultas; 4 de 102 chunks truncados por el encoder; qrels binarios de un anotador; Recall@1 acotado; latencia de una sola máquina con Docker.
**Conclusión (acotada al benchmark).** Ninguna configuración domina. B y C mejoran Recall@3 sobre A (0.310 → 0.382 → 0.442), pero los intervalos incluyen el cero salvo Recall@5 de C sobre B; C baja el MRR y cuesta ~46 veces más. La señal léxica y la densa son complementarias (Q05 vs Q12/Q13); la parte léxica explica buena parte del salto de A a B. No se declara superioridad general a partir de promedios.

## 7. Experimento 2 — RAG y evidencia

**Pregunta.** Con el mismo LLM, ¿qué cambia cuando se le entrega evidencia recuperada con citas? **Línea base:** closed-book. **Fijo:** Qwen2.5-0.5B-Instruct, 13 preguntas fijadas de antemano (10 con evidencia y 3 **sin** evidencia, donde corresponde abstenerse), decoding greedy, top-3 chunks. **Modificación:** presencia de evidencia. La configuración de recuperación (C, por tener el mayor Recall@3, Recall@5 y nDCG@5 del Exp. 1) se **declaró antes de generar**: `results/exp2/declared_config.json`. Prompts en `prompts/`.

**Comprobaciones mecánicas** (no sustituyen la auditoría):

| | closed-book | RAG |
|---|---|---|
| Se abstiene en las 10 respondibles | 5 | 0 |
| Se abstiene en las 3 sin evidencia | 1 | **0** |
| Respuestas con cita (de 10) | 0 | 7 |
| Ids citados que existen en el contexto | – | 0.95 |
| Ids citados que están en los qrels | – | **0.29** |
| Cobertura de la evidencia en el contexto | – | total solo en Q16 y Q19; **cero** en Q01, Q07, Q09 |

**Auditoría manual (corrección, citas y grounding por separado).** Se revisaron las 26 respuestas (`results/exp2/qwen/audit_answers_template.csv`) y las 19 afirmaciones de las respuestas RAG (`audit_claims_template.csv`) contra la respuesta de referencia, los hechos requeridos y el texto de los passages citados; criterios en `src/exp2_audit.py`, notas por fila en la columna `notes`. Resultado (`python src/exp2_audit.py`):

| | closed-book | RAG |
|---|---|---|
| Respuesta correcta (10 respondibles) | 0 / 10 | **2 / 10** (Q16 y Q19) |
| Abstención adecuada (3 sin evidencia) | 1 / 3 | **0 / 3** |
| Citas correctas (13 respuestas RAG) | – | **1 / 13** (solo Q16; 1 de 10 en las respondibles) |
| Afirmaciones respaldadas (19 del RAG) | – | 12 sí · 2 parcial · 5 no (faithfulness 0,68) |

Las tres medidas **no coinciden**: el faithfulness (0,68) es mucho más alto que la corrección (0,2) porque muchas afirmaciones son copias literales del contexto que no responden la pregunta (Q01, Q02, Q09). Eso es fiel pero no correcto, y las citas fallan incluso cuando el contenido es correcto (Q19).

**Errores y respuestas no sustentadas.** *Q07* (alucinación): el contexto no trae la passage correcta y Qwen+RAG responde «Nathan Sheets dissented…» citando `D01-P02`, que solo dice que Sheets comentó sobre la curva de rendimientos; el disenso fue de Mester (`D10-P05`). *Q19* (cita presente ≠ correcta): responde bien 9,52 % pero cita `D05-P05`; la cifra está en `D05-P13`. *Q10*: identifica el BTFP pero inventa que la Fed «subió la tasa… en marzo de 2023» (fue enero de 2024). *N01–N03*: con RAG no se abstiene nunca; en N01 reescribe una passage del VIX cambiando 2020 por 2008, y en N02/N03 devuelve solo una cita vacía.
**Consulta estructurada (Semana 3).** 25/25 JSON exacto y schema-valid, pero 20/25 son copia literal de la consulta original: válido no es adecuado. No comparable con la PC2 (cambian prompt, idioma y corpus).
**Anexo DeepSeek** (`deepseek-flash`, mismas preguntas, contexto y prompts; solo métricas mecánicas, sin auditoría y fuera de las conclusiones): se abstiene 3/3 en las sin evidencia y 83 % de sus ids citados están en los qrels. Una primera corrida con `max_tokens=300` se descartó por truncar el razonamiento del modelo (`results/exp2/deepseek_anexo/run0_max300_truncado/LEEME.md`).
**Limitaciones.** 13 preguntas; un modelo de 0.5B; un auditor; contexto fijo en 3 chunks; el generador y la recuperación fallan por separado y no se aislaron con un *oracle context*.
**Conclusión (acotada).** Con este modelo la evidencia elimina el «no sé» pero no garantiza respuestas ni citas fieles: Qwen+RAG copia passages, inventa ids y no se abstiene. El techo lo marca la recuperación (cobertura 0 en 3 de 10 preguntas) y la fidelidad depende del generador.

## 8. Experimento 3 — Herramienta con contratos: `risk.compute_signal`

**Qué es.** Dada una acción/ETF y una fecha, calcula con la regla de la PC1 el retorno a 5 días, la volatilidad anualizada de 20 días y la caída desde el máximo de 20 días, y devuelve `low`/`medium`/`high`. Es la parte determinista que la PC1 pedía separar del LLM. **Independiente de RAG; sin agentes ni cadenas.**

| Nivel | Implementación |
|---|---|
| Especificación | `TOOL_SPEC` (nombre, descripción, parámetros) en `src/tool_risk.py` |
| Contrato de entrada | `ComputeSignalInput` (Pydantic): `ticker` con patrón `^[A-Z]{1,5}$`, `as_of` fecha, sin campos extra |
| Validación de dominio | ticker ∈ {GLD, IWM, KO, SPY}; fecha dentro del rango, día de negociación y con 20 retornos previos |
| Contrato de salida | `RiskSignal`: cierre > 0, valores finitos, `score` = suma de puntos y `label` coherente |
| Resultado / error | `ToolResult` o `ToolError` con `kind` ∈ protocol, validation, domain, transient, timeout, execution, output_contract |

**Cuándo reintentar.** Solo `transient` (la fuente se recupera sola) y `timeout` (si queda presupuesto: `max_attempts=3`, backoff 50 ms × 2, timeout por intento 0.5 s, deadline total 2 s). `validation`, `domain`, `protocol`, `output_contract` y `execution` **no**: repetir la misma llamada da el mismo error. Reintentar ≠ reparar (reparar cambiaría los argumentos y llevaría a un flujo con agente).

**Resultados** (`results/exp3/`): 16 escenarios (válidas, entrada inválida, protocolo, dominio, transitorio, timeout, contrato de salida) → **16/16 con el resultado esperado**; **8/8 casos de la PC1 reproducidos** (etiqueta, puntos y los tres indicadores a 2 decimales); 28 pruebas pasan. Un despacho ingenuo (sin contratos, errores estructurados, reintentos ni timeout) sobre los mismos escenarios produce **8 excepciones sin capturar**, acepta en silencio la fecha `16/03/2020`, un campo extra y datos corruptos, y espera 1 s sin límite en el timeout.
**Limitaciones.** Fallos simulados (no red real); solo 4 tickers; el timeout cancela la corrutina pero no un hilo bloqueado; sin concurrencia. **Conclusión.** Los contratos convierten fallos en datos que el sistema puede decidir (reintentar o no, informar), y la herramienta recupera lo que el LLM de la PC1 no lograba.

## 9. Resultados: tabla de registro

| Experimento | Variable modificada | Variables fijas | Métricas | Resultado | Error analizado | Conclusión |
|---|---|---|---|---|---|---|
| 1 Recuperación | A → B (BM25+RRF) → C (+Cross-Encoder) | corpus, Q0, qrels, chunks 180/1, profundidad 10, top-k | Recall@1/3/5, MRR, nDCG@5, latencia, bootstrap | Recall@3: 0.310 / 0.382 / 0.442; latencia ≈ 40 / 38 / 1 862 ms | Q01, Q20, Q11, Q13, Q12, Q05 | Ninguna domina; mejoras no distinguibles de azar salvo Recall@5 de C; C ~46× más lento |
| 2 RAG | sin evidencia → con evidencia (top-3 de C) | Qwen2.5-0.5B, 13 preguntas, greedy | corrección, citas, grounding (auditoría); proxies mecánicos | Correcta 2/10 (closed-book 0/10); abstención correcta 0/3; citas correctas 1/13; faithfulness 0,68 | Q07, Q19, Q10, N01–N03 | La evidencia no garantiza fidelidad; el techo lo marca la recuperación |
| 3 Herramienta | ingenuo → contratos + reintento + timeout | escenarios y fallos inyectados | resultado correcto, excepciones sin capturar | 16/16 correctos; 0 vs 8 excepciones; 8/8 casos PC1 | S05, S07, S16 aceptados en silencio por el ingenuo | Los contratos hacen los fallos observables y acotados |

## 10. Limitaciones generales
Escala docente (150 passages, 22 consultas, 13 preguntas RAG): las diferencias se interpretan con bootstrap y no se generalizan. Qrels binarios, de un anotador y sin *pooling*; pueden faltar passages útiles (duplicados entre Wikipedia y la Fed). Wikipedia es fuente secundaria y cambia (revisión fijada). Corpus en un idioma y un dominio. 4 chunks truncados por el encoder. Un solo modelo por etapa. Latencias de una laptop con Docker. Modelo de 0.5B: sus fallos de seguimiento de instrucciones limitan lo que el Exp. 2 puede mostrar.

## 11. Conclusiones
1. **Recuperación:** BM25, densa y reranker son complementarios; con 22 consultas no hay una ganadora clara y el reranker multiplica la latencia por ~46. Lo que ninguna etapa arregla es la evidencia que no es candidata (Q01).
2. **RAG:** dar evidencia a un LLM de 0.5B elimina el «no sé» pero lo hace copiar passages, citar mal y no abstenerse; la cobertura de la evidencia (cero en 3 de 10 preguntas) fija el techo.
3. **Herramienta:** el cálculo de indicadores pertenece a código con contratos, no al LLM; los 8 casos de la PC1 se reproducen exactamente.
4. **Metodología:** la validez estructural, la métrica agregada y una cita presente no demuestran calidad; hay que mirar errores concretos.

**Continuidad hacia la Semana 9:** el repositorio deja un *baseline* reproducible, corpus y qrels trazables, recuperación comparada, una demostración RAG evaluada y una herramienta con contratos, listos para ampliarse con *workflows* y agentes, que **no** se implementan aquí.

## 12. Fuentes

**Técnicas (método y modelos)**
1. Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP. arXiv:2004.04906 — recuperación densa con producto interno y evaluación top-k (sistema A).
2. Cormack, G. V., Clarke, C. L. A., Büttcher, S. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods.* SIGIR — RRF y su constante 60 (sistema B).
3. Robertson, S., Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in IR 3(4) — BM25 Okapi (`k1`, `b`).
4. Nogueira, R., Cho, K. (2019). *Passage Re-ranking with BERT.* arXiv:1901.04085 — reranking con Cross-Encoder (sistema C). Bonifacio, L., et al. (2021). *mMARCO.* arXiv:2108.13897 — origen del Cross-Encoder `mmarco-mMiniLMv2`.
5. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS. arXiv:2005.11401 — diseño del Exp. 2.
6. Wang, L., et al. (2024). *Multilingual E5 Text Embeddings: A Technical Report.* arXiv:2402.05672 — encoder y prefijos `query:`/`passage:`.
7. Qwen Team (2024). *Qwen2.5 Technical Report.* arXiv:2412.15115 — LLM del Exp. 2. Vaswani, A., et al. (2017). *Attention Is All You Need.* arXiv:1706.03762 — Q/K/V y máscara causal. Holtzman, A., et al. (2020). *The Curious Case of Neural Text Degeneration.* arXiv:1904.09751 — decoding.
8. Manning, Raghavan, Schütze. *Introduction to Information Retrieval*, cap. 8, y Es, S., et al. (2023). *RAGAS.* arXiv:2309.15217 — evaluación de recuperación y RAG (referencia conceptual; no se usa RAGAS ni un LLM juez).
9. JSON Schema Draft 2020-12 y documentación de Pydantic v2 — contratos de entrada y salida.

**Dominio (finanzas)**
10. Barr, M. S. (2023). *Review of the Federal Reserve's Supervision and Regulation of Silicon Valley Bank.* Board of Governors of the Federal Reserve System (D17) — consultas Q11, Q12 y contexto de SVB.
11. Aquilina, M., Lombardi, M. J., Schrimpf, A., Sushko, V. (2024). *The market turbulence and carry trade unwind of August 2024.* BIS Bulletin No 90 (D20) — consultas Q03, Q15, Q16.
12. Board of Governors of the Federal Reserve System. *Financial Stability Report*, mayo de 2020 y abril de 2025 (D18, D19), y comunicados del FOMC de 2020 a 2025 (D09 a D16) — consultas Q01, Q04, Q07 a Q10, Q17, Q18.

**Material del curso (referenciado en la matriz):** `Semana1` a `Semana7` (Cuadernos 1 a 7; Laboratorios 1, 2 y 4; Lecturas 1, 2, 3, 5, 6 y 7; `Semana3/Exposiciones1_CC0F4.md`), repositorio `kapumota/CC-0F4`. Las fuentes del corpus y su licencia están en `data/PROCEDENCIA.md`.

**Herramientas.** Python 3.12, Docker, FAISS, sentence-transformers, transformers, Pydantic y pytest. La escritura del código y de los textos contó con apoyo de un asistente de IA (Claude Code). Los resultados salen de las corridas guardadas en `results/`.
