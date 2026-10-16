# CC0F4 · PC2 — De una consulta estructurada a recuperación densa verificable

**Práctica calificada 2 · continuación de la PC1 (vigilante de riesgo de mercado).**
Pedro Lautaro Quispe Ballesteros · Curso CC0F4 (2026-2), Semanas 3 y 4.

Tema de investigación (el mismo de la PC1): *Sistema multiagente con RAG híbrido para la detección temprana de riesgo de mercado con alertas sustentadas en evidencia*. La PC1 respondió "¿puede un LLM pequeño clasificar una señal?". Esta práctica trabaja la pregunta auxiliar del enunciado: **¿qué evidencia textual necesitaría recuperar el sistema antes de emitir una alerta?** Solo se construye `consulta estructurada → recuperación densa → Recall@k`. No hay RAG, BM25, reranking ni agentes, y lo recuperado no se usa todavía para generar una respuesta.

```text
necesidad -> LLM -> JSON {query, intent, abstain} -> JSON Schema -> "query: ..." -> e5-small
          -> FAISS IndexFlatIP -> top-k -> Recall@k -> análisis de errores
```

## Cómo ejecutar
Todo está en `notebook/PC2_recuperacion_densa.ipynb` (ya trae las salidas guardadas).

```bash
# desde esta carpeta (pc2/), con Docker Desktop abierto
docker compose up
```
Abre <http://localhost:8889/lab> y entra a `notebook/PC2_recuperacion_densa.ipynb`. Para detenerlo: `Ctrl+C` y luego `docker compose down`.

Ejecutar todo sin abrir Jupyter (sobrescribe `results/`; tarda unos 5 minutos en CPU):
```bash
docker compose run --rm pc2 jupyter nbconvert --to notebook --execute --inplace notebook/PC2_recuperacion_densa.ipynb
```
Notas:
- Usa el puerto **8889** para poder tener abierta también la PC1 (8888).
- El volumen `pc1_hf` se comparte con la PC1: Qwen ya está descargado y aquí se guarda además `multilingual-e5-small` (~470 MB, solo la primera vez). Docker puede avisar "volume already exists"; es solo una advertencia.
- Regenerar los datos del benchmark: `python src/build_dataset.py`.
- Sin Docker: `pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt`.

```text
pc2/
├── README.md
├── Dockerfile, docker-compose.yml, requirements.txt
├── src/build_dataset.py        # genera corpus, consultas, qrels y schema
├── data/                       # corpus_pc2.jsonl, queries_pc2.jsonl, qrels_pc2.json, schema_consulta.json, PROCEDENCIA.md
├── prompts/consulta_estructurada.txt
├── notebook/PC2_recuperacion_densa.ipynb
├── results/                    # CSV, config_corrida.json, prompts_renderizados.jsonl, fig_resumen.png
└── slides/PC2.pdf (+ PC2.html)
```

## 1. Problema de recuperación
Un vigilante de riesgo que emite una alerta ("riesgo alto en SPY") debe poder **sustentarla con texto**: ¿qué pasó en episodios parecidos?, ¿qué decidió la Reserva Federal?, ¿cómo se define el indicador? El problema de esta práctica es el paso previo a RAG: dada una necesidad de información en lenguaje natural, ¿el sistema coloca la evidencia relevante entre los primeros resultados?

Pregunta del Exp. 1: ¿convertir la necesidad en una consulta estructurada con un LLM conserva, mejora o degrada la recuperación?
Pregunta del Exp. 2: ¿cómo cambia Recall@k al modificar solo el tamaño del fragmento?

## 2. Corpus y consultas
**Corpus:** 8 documentos públicos (50 passages, 2.101 palabras) sobre los mismos episodios de la PC1 y dos conceptos de riesgo. Cada documento es un resumen propio en español de la fuente indicada (las cifras vienen de la fuente; la redacción es mía). Procedencia, URL y licencia de cada uno en [`data/PROCEDENCIA.md`](data/PROCEDENCIA.md).

| Doc | Contenido | Fuente | Caso PC1 |
|---|---|---|---|
| D01 | Crisis bursátil por COVID-19 (feb-abr 2020) | Wikipedia | H1 |
| D02 | Quiebra de Silicon Valley Bank (mar 2023) | Wikipedia | M2 |
| D03 | Venta global de ago-2024 y carry trade del yen | AP vía CSMonitor + Wikipedia | M1 |
| D04 | Caída bursátil por aranceles (abr 2025) | Wikipedia | H3 |
| D05 | Índice de volatilidad VIX | Wikipedia | indicador |
| D06 | Drawdown y drawdown máximo | Wikipedia | indicador `dd20` |
| D07 | Comunicado del FOMC, 15-jun-2022 | Federal Reserve | H2 |
| D08 | Comunicado del FOMC, 15-mar-2020 | Federal Reserve | H1 |

**Consultas (Q0) y qrels mínimos** (relevancia sobre `passage_id`, no sobre chunks, para que no cambie con la segmentación):

| Id | Necesidad de información (Q0, como la formularía yo) | Dificultad | Passages relevantes |
|---|---|---|---|
| Q01 | ¿Qué pasó con la volatilidad esperada del mercado en el peor día de la crisis del COVID en marzo de 2020? | easy | D01-P05, D05-P05 |
| Q02 | ¿Por qué el banco de Silicon Valley no pudo resistir si tenía una cartera de bonos tan grande? | medium | D02-P02, D02-P03 |
| Q03 | Quiero entender qué significa pedir dinero prestado en una moneda barata para invertir y por qué eso puede provocar ventas masivas de golpe. | hard | D03-P02, D03-P03 |
| Q04 | ¿Qué hizo la Reserva Federal con las tasas de interés para responder al pánico por el coronavirus? | ambiguous | D08-P01, D01-P06 |
| Q05 | ¿Cómo se mide cuánto ha caído un activo desde su máximo reciente? | easy | D06-P01, D06-P02 |
| Q06 | ¿Cómo reaccionaron los bonos del Tesoro cuando se anunciaron los aranceles de abril de 2025? | medium | D04-P05 |

Corpus, consultas y qrels se redactaron **antes** de ejecutar y no se modificaron después de ver resultados (misma regla que la Semana 4). Q04 y Q01 están hechas para tener "vecinos" confusos (D01 y D08 hablan del mismo mes; D02 y D07 hablan de alzas de tasas).

## 3. Consulta estructurada y JSON Schema
El LLM recibe la necesidad y debe devolver un objeto pequeño (`data/schema_consulta.json`, JSON Schema 2020-12):

```json
{
  "type": "object",
  "properties": {
    "query":   {"type": "string", "minLength": 8, "maxLength": 200},
    "intent":  {"type": "string", "enum": ["risk_event", "risk_indicator", "policy_response", "mechanism", "other"]},
    "abstain": {"type": "boolean"}
  },
  "required": ["query", "intent", "abstain"],
  "additionalProperties": false
}
```
Se distinguen tres niveles, igual que en la PC1: **parseable** (`json.loads`, exacto o recuperando el primer `{...}`), **schema-valid** (`Draft202012Validator`) y **adecuada** (conserva la necesidad; se juzga con la recuperación y leyendo los casos). El prompt exacto está en `prompts/consulta_estructurada.txt` (secciones TAREA / NECESIDAD / CONTRATO); solo cambia `{need}`. Decoding: **greedy**, `max_new_tokens=96`, `repetition_penalty=1.05`, seed 42. Modelo: `Qwen/Qwen2.5-0.5B-Instruct` (mismo de la PC1, revisión `7ae5576…`).

## 4. Modelo de embeddings
`intfloat/multilingual-e5-small`: multilingüe, 384 dimensiones, ventana de 512 tokens, entrenado para recuperación. Exige prefijos: **`query: `** para las consultas y **`passage: `** para los fragmentos; se conservan tal cual. Antes de indexar, el cuaderno comprueba que **ningún chunk excede 512 tokens** (máximo observado: 444 en `large`), para que el Exp. 2 mida granularidad y no truncación.

## 5. Segmentación
Cada documento viene en passages atómicos. El chunker agrupa passages consecutivos hasta aproximar `target_words`, con **overlap fijo de 1 passage**, y conserva `chunk_id`, `doc_id`, `passage_ids`, `n_words`, `text`. Es la misma lógica del Laboratorio 4.

| condición | target_words | n_chunks | palabras (media / máx) | passages por chunk |
|---|---|---|---|---|
| small | 100 | 40 | 76.8 / 99 | 1.85 |
| baseline | 180 | 20 | 131.2 / 173 | 3.10 |
| large | 320 | 9 | 239.8 / 312 | 5.67 |

## 6. Índice y similitud
Vectores L2-normalizados; `faiss.IndexFlatIP` (búsqueda exhaustiva por producto interno = **exacta**). Con vectores unitarios `cos(x,y) = xᵀy / (‖x‖‖y‖) = xᵀy`, por eso ordenar por producto interno equivale a ordenar por coseno; el cuaderno lo comprueba con vectores reales (coseno = 0.863245 = producto interno normalizado) y verifica que FAISS y una multiplicación NumPy dan el mismo top-5. Distinción de capas: *documento* (unidad de fuente) ≠ *fragmento/chunk* (unidad indexada) ≠ *índice* (organiza y busca vectores) ≠ *recuperador* (política: top-k, qué se entrega).

**Métrica.** `Recall@k(q) = |G_q ∩ C_k(q)| / |G_q|`, donde `G_q` son los passages relevantes y `C_k(q)` los passages contenidos en los k primeros chunks. Complemento propio, no pedido por el enunciado: `evidence_precision@3` = fracción de passages cubiertos por el top-3 que son relevantes (mide dilución).

## 7. Experimento 1 — Consulta (Q0 vs Q1)
**Variable modificada:** la consulta. **Fijo:** corpus, segmentación (`baseline` = 180 palabras), encoder, índice, top-k. Una Q1 no utilizable (no parseable, no schema-valid o `abstain=true`) no se envía al encoder y cuenta como Recall = 0, para que la media refleje el pipeline completo.

| | parseable | JSON exacto | schema_valid_rate | abstain | Recall@1 | **Recall@3** | Recall@5 |
|---|---|---|---|---|---|---|---|
| Q0 | – | – | – | – | 0.667 | **0.833** | 0.917 |
| Q1 | 6/6 | 0/6 | **0.833** (5/6) | 0 | 0.667 | **0.667** | 0.750 |

Sobre las 5 Q1 utilizables, mean Recall@3 es 0.800 para Q0 y para Q1 (sin cambio en ninguna consulta). Hallazgos:
- **Contrato roto con Markdown:** las 6 salidas venían dentro de una cerca ```` ```json ````, aunque el prompt decía "no uses Markdown"; el parser las recupera.
- **Q03 (hard) no es schema-valid:** el modelo inventó `policy_response` y `other` y omitió `abstain`. Sin Q1, Recall@3 pasa de 1.0 a 0.0. Es todo el descenso de 0.833 a 0.667.
- **Q1 ≈ Q0:** 4 de las 5 válidas son copia literal de Q0 y Q05 solo recorta "¿Cómo se mide". Por eso los top-3 coinciden 3/3. La transformación conservó la necesidad porque casi no la transformó.
- **`intent` = `risk_event` en las 5**, incluso para Q04 (política monetaria) y Q05 (indicador). Schema-valid no implica semánticamente adecuado.

## 8. Experimento 2 — Chunking
**Variable modificada:** `target_words` ∈ {100, 180, 320}. **Fijo:** corpus, consultas (Q0), qrels, overlap = 1 passage, encoder, normalización L2, producto interno, `IndexFlatIP`, top-k ∈ {1, 3, 5}. Métrica principal: mean Recall@3.

| condición | n_chunks | Recall@1 | **Recall@3** | Recall@5 | evidence_precision@3 |
|---|---|---|---|---|---|
| small (100) | 40 | 0.583 | **0.833** | 0.833 | 0.336 |
| baseline (180) | 20 | 0.667 | **0.833** | 0.917 | 0.194 |
| large (320) | 9 | 0.667 | **0.917** | 1.000 | 0.097 |

Recall@3 por consulta (`results/exp2_chunking_por_consulta.csv`):

| | Q01 | Q02 | Q03 | Q04 | Q05 | Q06 |
|---|---|---|---|---|---|---|
| small | 0.5 | 1 | 0.5 | 1 | 1 | 1 |
| baseline | 0.5 | 1 | 1 | 0.5 | 1 | 1 |
| large | 0.5 | 1 | 1 | 1 | 1 | 1 |

- **Ayudó (Q04, no monótono):** en `baseline`, dos chunks de D08 que se solapan en D08-P05 más un chunk de D02 desplazan a D01-P06 (0.5); con `small` y con `large` entra (1.0).
- **Cambió el reparto (Q03):** `small` pone D03-P02 en rank 1 pero D03-P03 queda en otro chunk fuera del top-3 (0.5); `baseline` y `large` los agrupan en un chunk (1.0), aunque en rank 3 y 2. Más contexto ayudó a cobertura, no a orden.
- **No cambió:** Q02, Q05, Q06 (1.0 siempre) y Q01 (0.5 siempre).
- **Dilución:** `evidence_precision@3` cae de 0.336 a 0.097. Con `large` hay 9 chunks que cubren 5.7 passages en promedio: top-5 ya es más de la mitad del índice y Recall@5 = 1.0 es casi trivial.

## 9. Resultados
Registro de experimentos (detalle en `results/resumen_experimentos.csv`, `results/config_corrida.json` y `results/fig_resumen.png`):

| Experimento | Variable modificada | Variable fija | Métrica | Resultado | Conclusión |
|---|---|---|---|---|---|
| Consulta | Q0 original → Q1 estructurada (Qwen2.5-0.5B, greedy) | corpus, segmentación 180, e5-small, IndexFlatIP, top-k | schema_valid_rate + Recall@k | schema_valid 0.833; Recall@3 0.833 → 0.667 (0.800 → 0.800 en las utilizables) | la transformación casi copió Q0; el descenso viene solo del caso con JSON inválido; sin evidencia de mejora |
| Chunking | target_words 100 / 180 / 320 | corpus, queries, qrels, overlap 1, e5-small, IndexFlatIP, top-k | Recall@1/3/5 (principal: Recall@3) | Recall@3 0.833 / 0.833 / 0.917 | la diferencia es medio caso de una consulta; mayor cobertura con menor precisión de evidencia |

## 10. Un error de recuperación: Q01
La necesidad pide la volatilidad esperada en el peor día de la crisis del COVID. Hay **dos** passages relevantes en **dos** documentos: D01-P05 (crónica del 16-mar-2020, que menciona el VIX) y D05-P05 (documento del VIX, mismo hecho). Recall@3 = 0.5 en las tres condiciones: se recupera D01-P05 y nunca D05-P05.

El diagnóstico del cuaderno muestra por qué:

| condición | rank del chunk con D05-P05 | score de ese chunk | score del top-1 | documentos del top-5 |
|---|---|---|---|---|
| small (40 chunks) | 10 | 0.8544 | 0.8723 | D01, D01, D01, D01, D01 |
| baseline (20 chunks) | 9 | 0.8479 | 0.8825 | D01, D01, D01, D08, D03 |
| large (9 chunks) | 4 | 0.8536 | 0.8908 | D01, D08, D04, D05, D01 |

La consulta comparte vocabulario con la crónica del COVID, así que **un solo documento acapara el ranking** y el segundo documento relevante queda desplazado. Los scores de e5 viven en una banda estrecha (0.85-0.89): una diferencia de 0.002 decide el orden. Cambiar el tamaño del chunk no resuelve esto; apunta a diversificar resultados o reordenar (Semana 5 en adelante), no a tocar `target_words`.

## 11. Limitación
- **Escala:** 6 consultas, 50 passages, una corrida por condición. Cada consulta pesa 0.167 de la media; diferencias de un caso están dentro del ruido y no hay intervalos de confianza.
- **LLM pequeño:** un 0.5B no respetó "sin Markdown", copió la consulta y asignó el mismo `intent` a todo. El Exp. 1 no mide si un LLM *bueno* mejora la consulta, sino lo que hace este.
- **Corpus resumido por mí:** los passages son mis resúmenes en español; comparten léxico con mis propias consultas, lo que facilita la recuperación. Las cifras de D03 sobre S&P 500 y VIX se cotejaron con prensa secundaria (ver `PROCEDENCIA.md`).
- **Recall por passages cubiertos:** un chunk grande cubre más passages, lo que puede inflar Recall@k; por eso se reporta `evidence_precision@3`.
- **Qrels mínimos y binarios:** puede haber evidencia útil no marcada, y el overlap fijo repite passages entre chunks consecutivos.
- **Un solo encoder y un solo índice exacto:** nada se dice de otros modelos ni de índices aproximados.

## 12. Conclusión
Con estos 6 casos: (1) el LLM produjo siempre texto parseable pero nunca JSON exacto, y rompió el schema en 1 de 6; cuando fue válido, copió la necesidad, así que la consulta estructurada no mejoró la recuperación y su único efecto medible fue negativo (el caso inválido); (2) el tamaño del fragmento movió Recall@3 de 0.833 a 0.917 solo por una consulta, a costa de bajar la precisión de la evidencia de 0.336 a 0.097; (3) el error de Q01 no depende del chunking: un documento desplaza al otro. Lo que **no** puede afirmarse es que la consulta estructurada ayude, ni que chunks grandes sean mejores en general. Para el sistema completo: un modelo mayor para la consulta (o gramática/JSON mode para forzar el contrato), más consultas con qrels graduados, y diversificación o reranking antes de pasar evidencia al generador.

## 13. Fuentes
**Primarias (método)**
1. Wang, L., et al. (2024). *Multilingual E5 Text Embeddings: A Technical Report.* arXiv:2402.05672. — `multilingual-e5-small`, prefijos `query:`/`passage:`.
2. Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP. arXiv:2004.04906. — recuperación densa con producto interno y evaluación con top-k.
3. Johnson, J., Douze, M., Jégou, H. (2019). *Billion-scale similarity search with GPUs.* IEEE Trans. Big Data. arXiv:1702.08734. — FAISS e `IndexFlatIP`.

**Adicionales**
4. Wang, L., et al. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training.* arXiv:2212.03533. — E5 original.
5. JSON Schema, *Draft 2020-12*. https://json-schema.org/draft/2020-12 — contrato de la consulta estructurada; validación con `jsonschema` (`Draft202012Validator`).
6. Qwen Team (2024). *Qwen2.5 Technical Report.* arXiv:2412.15115. — modelo usado para la consulta estructurada.
7. Material del curso CC0F4: `Semana3/Cuaderno3-CC-0F4.ipynb`, `Semana4/Cuaderno4-CC-0F4.ipynb`, `Semana4/Laboratorio4-CC-0F4.ipynb`, repositorio `kapumota/CC-0F4`. — chunking, Recall@k y protocolo experimental.

**Fuentes del corpus** (dominio de aplicación; detalle en `data/PROCEDENCIA.md`)
8. Wikipedia: *2020 stock market crash*, *Collapse of Silicon Valley Bank*, *2025 stock market crash*, *VIX*, *Drawdown (economics)*, *Carry trade* (CC BY-SA 4.0; consultadas el 2026-10-13).
9. Federal Reserve Board, comunicados del FOMC del 15-mar-2020 y del 15-jun-2022 (dominio público).
10. Kurtenbach, E. (AP), *Carry trades partly drove market jitters. What are they?*, Christian Science Monitor, 6-ago-2024.
