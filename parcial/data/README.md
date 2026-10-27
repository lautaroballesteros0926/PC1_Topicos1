# Datos del examen parcial (benchmark `parcial-v1`)

Benchmark nuevo para la evaluacion del parcial. Reemplaza al de la PC2 (8 documentos, 50 passages resumidos por mi y 6
consultas), que solo sirve como antecedente: el examen exige documentos reales, mas consultas y evidencia gold trazable.

| Archivo | Contenido |
|---|---|
| `corpus_parcial.jsonl` | 21 documentos, 150 passages **textuales** (sin traducir ni resumir), con procedencia por documento |
| `PROCEDENCIA.md` | fuente, URL, version (`oldid` en Wikipedia), hash y licencia de cada documento |
| `queries_parcial.jsonl` | 22 necesidades de informacion con evidencia en el corpus (Exp. 1 y Exp. 2); `exp2_rag=true` marca las 10 del Exp. 2 |
| `qrels_parcial.json` | `query_id -> [passage_id]` relevantes (binarios, 57 pares; 1 a 5 por consulta) |
| `reference_answers_parcial.jsonl` | respuesta de referencia y `required_facts` de las 10 consultas del Exp. 2 |
| `queries_sin_evidencia.jsonl` | 3 consultas sin evidencia en el corpus (esperado: abstencion); solo Exp. 2 |
| `benchmark_manifest.json` | conteos y hashes (sha256, 12 caracteres) de los archivos anteriores |

Se regeneran con `python src/build_corpus.py` y `python src/build_benchmark.py` (el primero necesita los originales; ver el
docstring de `build_corpus.py`).

## Corpus
- **Tema:** los mismos episodios de la PC1 (COVID-19 2020, alzas de tasas 2022, SVB 2023, carry trade 2024, aranceles 2025) y
  los indicadores que usa su regla (VIX como volatilidad esperada, drawdown), todo en ingles.
- **Fuentes:** 8 articulos de Wikipedia (CC BY-SA 4.0, revision fijada), 8 comunicados de la Reserva Federal, 3 informes de la Fed
  (revision de supervision de SVB, Financial Stability Report 2020 y 2025) y 2 publicaciones del BIS (Bulletin 90 y Quarterly Review
  de marzo de 2020).
- **Passages:** parrafos completos del original, o una subcadena contigua cuando el parrafo extraido mezclaba pies de figura o
  notas (`start_at`/`end_at` en `src/build_corpus.py`). Entre 27 y 242 palabras (media 105). Id estable `Dxx-Pyy`.
- **Distractores deliberados** (no se crearon despues de ver resultados; son propios de las fuentes):
  - los parrafos finales de los comunicados FOMC D12 a D15 son casi identicos ("In assessing the appropriate stance...");
  - los hechos de SVB aparecen repetidos en D03, D04, D16 y D17 (p. ej. el 89 % de depositos no asegurados);
  - D06 da otros maximos del VIX (62,12 y 75,47) que compiten con el cierre de 82,69 de D01;
  - hay comunicados con la misma estructura y distinta decision (recorte en 2020 y 2024, alza en 2022 y 2023, pausa en 2025).

## Consultas y qrels
- 22 consultas en ingles, redactadas como necesidades de un analista de riesgo. Dificultad: 4 `easy`, 10 `medium`, 8 `hard`
  (columna `difficulty`); la columna `category` indica el tipo (busqueda numerica, definicion, causal multi-documento,
  brecha lexica, distractores de boilerplate, etc.). Seis proceden de la PC2 (`pc2_origin`), traducidas al ingles.
- **Relevancia sobre `passage_id`**, no sobre chunks, para que no dependa de la segmentacion.
- **Protocolo:** lei el corpus completo, redacte consultas y marque como relevantes los passages que contienen la informacion
  necesaria para responder. **No se uso ningun sistema de recuperacion ni generacion** para proponer, filtrar o corregir
  qrels. No se hizo *pooling*.
- **Congelamiento:** corpus, consultas, qrels y referencias no se modifican despues de observar resultados. Una correccion crea
  `parcial-v2` y obliga a repetir los experimentos.

## Abstencion (Exp. 2)
Las 3 consultas de `queries_sin_evidencia.jsonl` estan cerca del tema pero sin respuesta en el corpus (un maximo historico del VIX
que el corpus solo menciona sin cifra, un mercado ausente, una decision de otro banco central).

## Limitaciones
- **Escala docente:** 150 passages y 22 consultas; las diferencias entre sistemas se interpretan con bootstrap, sin generalizar.
- **Qrels binarios e incompletos:** puede haber passages utiles no marcados (sobre todo por duplicados entre Wikipedia y las fuentes
  primarias); eso penaliza a un recuperador que acierta con un passage no anotado.
- **Anotador unico:** sin acuerdo entre anotadores.
- **Wikipedia es fuente secundaria** y cambia; se fija la revision. Varios passages heredan marcas de nota al pie del PDF.
- **Un solo idioma y un solo dominio:** ingles y riesgo de mercado en EE. UU.
