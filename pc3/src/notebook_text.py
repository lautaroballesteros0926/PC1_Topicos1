"""Textos del cuaderno de la PC3 (separados del generador para poder revisarlos como prosa).

Toda cifra citada aqui se puede comprobar en la celda de codigo que la acompana (lee results/)."""

TEXT = {}

TEXT["portada"] = r"""
# CC0F4 · PC3 — De un sistema RAG evaluable a workflows, agentes controlados y memoria verificable
**Pedro Lautaro Quispe Ballesteros** · Semanas 9 y 10 (con reutilización de las semanas 4, 6 y 7) · modalidad individual.
Tema (PC1 → PC2 → parcial → PC3): *Sistema multiagente con RAG híbrido para la detección temprana de riesgo de mercado con alertas
sustentadas en evidencia*.

**Tarea multi-paso.** Dada una pregunta sobre un activo y una fecha: (1) recuperar evidencia del episodio con el recuperador **C** del
parcial, (2) calcular la señal con la herramienta `risk.compute_signal` del parcial, (3) contrastar señal y evidencia y (4) decidir
`alert`, `no_alert` o `abstain` con una respuesta estructurada y verificable.

```text
Exp. 1   mismas 15 tareas, mismo controlador, mismas herramientas y límites
         W0 = workflow fijo (reglas, sin LLM)      vs      W1 = agente Qwen2.5-0.5B que elige la acción y cuándo parar
Exp. 2   mismo agente W1, 3 secuencias encadenadas (9 interacciones, 7 sesiones)
         M0 sin memoria | M1 working | M2 episódica (JSONL) | M3 semántica (e5 + FAISS, versión e invalidación)
         + diagnóstico sin LLM (lector determinista) para separar la calidad de la memoria de la del modelo
```

**Continuidad.** El parcial dejó corpus (21 documentos, 150 passages), qrels, el recuperador C (e5 + FAISS + BM25 + RRF +
Cross-Encoder) y la herramienta con contratos que reproduce los 8 casos de la PC1. Aquí se **importan** desde `../parcial/src` sin
modificarlos (`src/config.py`). La PC1 ya mostró que un LLM de 0.5B no aplica la regla de riesgo; la PC3 pregunta si ese mismo
modelo puede, al menos, **coordinar** las herramientas que hacen el cálculo.

**Cómo ejecutar:** `docker compose up` en `pc3/` → <http://localhost:8891/lab>. Este cuaderno **no** vuelve a correr los
experimentos (Qwen en CPU tarda ~1 h): carga las trazas y los CSV originales de `results/`, ejecuta las 20 pruebas y una demostración
corta. Comandos para regenerar todo: `README.md` §14.
"""

TEXT["s1"] = r"""
## 1. Límites y condiciones de terminación (fijados antes de ejecutar)

El controlador es **código**, no el LLM. Comprueba los presupuestos antes de cada decisión, valida la acción propuesta contra una
*allowlist* y su contrato, ejecuta la herramienta con timeout, valida la observación antes de que entre al contexto, detecta
repeticiones y falta de progreso, y valida la respuesta final contra lo observado. Los límites y las definiciones se escribieron en
`results/declared_config.json` antes de la primera corrida evaluada, junto con los hashes del benchmark, los prompts, el corpus y los
precios; la celda comprueba que no cambiaron.
"""

TEXT["s2"] = r"""
## 2. Benchmark de tareas (15) y escenarios de memoria (9 interacciones)

Las respuestas esperadas **no salen de W0 ni de W1**: la severidad se calcula con la regla de la PC1 (la misma que reproduce los 8 casos
gold) y la evidencia aceptable es el conjunto de passages del corpus que describen el episodio, anotado leyendo su texto
(`src/build_tasks.py`, `data/evidence_sets.json`; T08 reutiliza los qrels de Q01 del parcial). Tipos: **8 ordinarias** (≥ 2
operaciones; T07 compara dos activos), **4 con fallo de herramienta** (T09 es un error real de dominio: el 15-mar-2020 fue domingo;
T10–T12 son fallos **inyectados** y así se rotulan) y **3 de abstención o rechazo** (activo sin datos, fecha fuera de rango y una
afirmación falsa de un colega). Cada tarea trae objetivo, entrada, operaciones, fuentes permitidas, resultado esperado, criterio de
éxito y terminaciones permitidas.
"""

TEXT["s3"] = r"""
## 3. Arquitectura: qué decide el LLM y qué decide el controlador

| Decide | W0 workflow | W1 agente | Controlador (ambos) |
|---|---|---|---|
| Siguiente acción y argumentos | plan fijo: buscar con la pregunta → señal de cada par pedido | **el LLM** (JSON con `action`, `arguments`, `reason`) | rechaza acciones fuera de la allowlist o que no cumplen el contrato |
| Cuándo terminar | después del plan | **el LLM** (`finish`) | corta por `max_steps`, `max_tool_calls`, deadline, errores, repetición o falta de progreso |
| Respuesta | regla: severidad ≥ medium → `alert`; low → `no_alert`; error → `abstain` | **el LLM** | acepta solo si la severidad coincide con una señal observada (o un recuerdo citado) y los ids de evidencia fueron vistos |
| Ejecución de herramientas | — | — | siempre el controlador, con timeout; el modelo nunca ejecuta código |

**Herramientas** (`src/tools.py`): `search_evidence(query, top_k≤5)` = recuperador C del parcial (Semanas 4, 5 y 7);
`compute_signal(ticker, as_of)` = `risk.compute_signal` del parcial con sus contratos, reintentos y timeout (Semana 6);
`read_passage(passage_id)` = operación auxiliar nueva, solo para passages ya vistos. Cada observación pasa por un modelo Pydantic:
el texto se trunca y se limpia, los ids se comprueban contra el corpus y los errores llegan al modelo **solo como campos**
(`kind`, `code`, `retryable`), nunca como mensaje libre. Así un fallo no se convierte en una instrucción.

**ReAct acotado.** El campo `reason` (≤ 20 palabras) es el «pensamiento» que queda en la traza; se intercala con la acción y la
observación. No hay planificador separado: el plan de W1 es implícito, paso a paso; el *replanning* posible es reformular una
búsqueda o corregir una fecha tras un error (T09). El prompt se ajustó en 3 versiones **solo** con dos tareas de desarrollo que no
están en el benchmark (`data/dev_tasks.jsonl`, `results/dev/v1..v3`); la v3 quedó congelada antes de las corridas evaluadas.
"""

TEXT["s3b"] = r"""
**Pruebas** (`tests/`, sin LLM): cada `stop_reason` tiene una prueba con una política guionizada; también se prueban la validación de la
respuesta final, que los errores no filtren el mensaje, la memoria (escritura, persistencia, invalidación y borrado) y las métricas.
"""

TEXT["s4"] = r"""
## 4. Experimento 1 — Workflow determinista (W0) frente a agente controlado (W1)

**Pregunta.** ¿Permitir que un LLM seleccione herramientas y decida el siguiente paso aumenta el éxito en tareas multi-paso respecto de un
workflow fijo? **Hipótesis (declarada antes):** con un modelo de 0.5B, W1 no supera a W0 en éxito y cuesta más; podría ayudar solo donde
el plan fijo no se adapta (T08, reformular la búsqueda; T09, corregir la fecha). **Fijo:** benchmark, corpus, recuperador C,
herramientas y contratos, controlador y límites, modelo y decoding (greedy), sin memoria. **Variable:** la política que elige la
secuencia de acciones.
"""

TEXT["s4b"] = r"""
**Resultados.** W0 resuelve **12 de 15** (0,80) con 0 tokens y ~1,8 s por tarea. W1 resuelve **1 de 15** (0,07, solo T02) con
~3 020 tokens y ~38 s por tarea (contexto máximo de 1 545 tokens, por debajo del presupuesto de 2 048). En **14 tareas** el controlador
detuvo a W1 con `REPEATED_ACTION`: el modelo volvió a proponer exactamente una acción ya ejecutada (una búsqueda, una señal o la
lectura del mismo passage). W1 no emitió **ninguna** alerta falsa: en las 4 tareas en las que correspondía abstenerse (T10, T11, T13
y T14) no respondió, aunque terminó por repetición y no por `ABSTAIN`, así que `termination_correct` es 0 en esas tareas.

**El agente avanza pero no cierra.** W1 observó la señal correcta en 8 de las 11 tareas que la requerían (W0 en 10) y vio evidencia
del episodio en 7 de 9 (W0 en 8). Lo que falla es decidir que ya terminó: en DEV1 (desarrollo) incluso propuso `alert` con una señal
`low` y el controlador lo rechazó (`finish_alert_on_low_signal`).

Trazas originales de los casos que pide el enunciado:
* **Donde W1 mejora un paso que W0 hace mal: T04.** W0 busca con la pregunta completa y el ranking pone primero a SVB (D17, D03);
  como el workflow no juzga la pertinencia, cita `D17-P03, D17-P04, D03-P04` y falla la evidencia. W1 reformula
  («small caps IWM 2022 interest rate increases market decline») y recupera `D02-P01…P04`, que sí describen el 2022. Pero después lee
  `D02-P01` dos veces y se detiene: **mejor recuperación, mismo resultado final (fallo)**.
* **Donde W1 no mejora y cuesta más: T09.** El enunciado pide usar el día hábil anterior si la fecha no lo es. W0 recibe el error real
  `date_not_trading_day` y se abstiene (falla, pero de forma segura). W1 ni siquiera llega a pedir la señal: repite la misma búsqueda y
  se detiene a los 2 pasos (25,7 s, 1 984 tokens). Una regla de una línea en el workflow («si `date_not_trading_day`, usar el día hábil
  previo») resolvería T09 sin LLM.
* **Único éxito de W1: T02** — buscar, calcular y terminar con evidencia `D04-P01…P06`, igual que W0 pero ~28 veces más lento (48,4 s frente a 1,7 s).
* **T08** muestra el riesgo de un plan fijo: W0 busca y cita `D06-P03…P05` (definición del VIX), no el récord de 82,69; W1 inventa el
  id `P01-01`, que el contrato rechaza (`input_contract`), y lo repite.
"""

TEXT["code_exp1_traces"] = r"""
for name in ["W0_workflow__T04", "W1_agent_qwen__T04", "W0_workflow__T09", "W1_agent_qwen__T09", "W1_agent_qwen__T02"]:
    show_trace("exp1", name)
# progreso parcial: ¿la politica llego a observar la senal esperada y evidencia del episodio?
rows = []
for pol in ["W0_workflow", "W1_agent_qwen"]:
    for t, e in expected.items():
        r = json.loads((TRACES / "exp1" / f"{pol}__{t}.json").read_text(encoding="utf-8"))
        rows.append({"policy": pol, "task_id": t,
                     "senal_esperada_observada": None if not e["severity"] else any(s["ticker"] == e["ticker"] and s["as_of"] == e["as_of"] for s in r["signals"]),
                     "evidencia_vista": None if not e["evidence_any_of"] else bool(set(r["seen_passages"]) & set(e["evidence_any_of"]))})
prog = pd.DataFrame(rows); display(prog.groupby("policy")[["senal_esperada_observada", "evidencia_vista"]].agg(lambda x: f"{int(x.dropna().sum())}/{x.notna().sum()}"))
"""

TEXT["s4c"] = r"""
### Pruebas de robustez y terminación

Además de los fallos dentro del benchmark (T09 error real de dominio, T10 fuente caída, T11 salida inválida, T12 fallo transitorio que se
recupera con el reintento interno de la herramienta: `attempts = 2` en la traza), se ejecutaron 7 pruebas **guionizadas** sobre las
herramientas reales. No son ejecuciones de un LLM: comprueban que el controlador termina con el `stop_reason` correcto, conserva la
traza y no deja salir ninguna respuesta.
"""

TEXT["s4d"] = r"""
Las 7 terminan como se esperaba y ninguna emite respuesta. Junto con T10/T11 (W0 se abstiene; W1 no responde) muestran que un
error de herramienta **no** se transforma en una alerta aparentemente válida: la salida corrupta de T11 se detiene en el contrato de
salida (`output_contract`) y nunca llega a ser una «señal».

**Conclusión del Exp. 1 (acotada a 15 tareas y a un modelo de 0.5B).** Para esta tarea un workflow fijo es suficiente y mucho más
barato: 0,80 frente a 0,07 de éxito, ~21 veces menos latencia y cero tokens. La autonomía solo aportó en un paso intermedio (reformular
la consulta en T04) y no se tradujo en éxito. La ganancia potencial de un agente (reparar T09, buscar mejor en T04 y T08) se puede
obtener añadiendo ramas al workflow (*routing* ante códigos de error, una segunda búsqueda si la evidencia no menciona el evento). No
se puede afirmar nada sobre modelos mayores: no se probaron.
"""

TEXT["s5"] = r"""
## 5. Experimento 2 — Estado y memoria del agente

**Pregunta.** ¿Qué mejora aporta cada mecanismo de memoria a tareas encadenadas y qué errores introduce? **Hipótesis:** M1 ayuda
solo dentro de una sesión; M2 y M3 permiten recuperar entre sesiones; M2, al guardar historia sin invalidarla, puede devolver un
recuerdo obsoleto (S2-3) o uno parecido pero equivocado (S3-3); M3 evita el obsoleto porque lo invalida.
**Fijo:** agente W1 (modelo, prompt, decoding), herramientas, controlador, límites, presupuesto de contexto (2 048 tokens, de los cuales
como máximo 450 para memoria y 3 recuerdos) y orden de las 9 interacciones. **Variable:** el mecanismo de memoria.

| | Qué guarda | Dónde | Cómo se recupera | Persistencia | Actualización / invalidación |
|---|---|---|---|---|---|
| Estado de ejecución (todas) | contadores, observaciones de **una** tarea | `RunState` en RAM | — | se descarta al terminar | — |
| **M1** working | resultado de cada tarea anterior **de la sesión** | lista en RAM | los 3 más recientes | se borra al cambiar de sesión | — |
| **M2** episódica | un episodio por tarea: tarea, acciones, observaciones validadas, resultado, errores, `stop_reason` | `episodic.jsonl` | relevancia léxica + recencia + importancia, min-max y suma (Park et al.) | sí, se relee del disco en cada sesión | no se invalida: es historia |
| **M3** semántica | un hecho por señal observada en una tarea con `SUCCESS`, con procedencia (herramienta, hash de precios, passages), versión y estado | `semantic.jsonl` + FAISS (e5) | similitud coseno sobre hechos **activos** | sí | una señal más reciente del mismo activo (≤ 120 días) marca la anterior `superseded`; deja de servirse pero se conserva para auditoría |

Escribe **el controlador** al terminar cada tarea, solo con campos validados: no se guardan prompts, salidas crudas ni el `rationale`
libre del modelo (privacidad y trazabilidad). Toda operación queda en `memory_ops.jsonl` sin contenido. El borrado (`delete`) está
implementado y probado. **Escenarios:** S1 recuperación en la misma sesión (S1-2) y desde otra sesión (S1-3); S2 actualización
controlada del escenario: la fecha avanza al 30-ago-2024 con precios reales, la alerta *medium* del 5-ago queda obsoleta (S2-2) y se
pregunta por su estado actual (S2-3); S3 interferencia entre dos alertas *high* de SPY (COVID y aranceles) (S3-3).
**Denominadores:** `memory_recall_accuracy` / 4 interacciones que necesitan memoria; `cross_session_retention` / 3;
`stale_memory_error_rate` / 2 con riesgo de obsoleto; `wrong_memory_use_rate` / interacciones en que el agente citó algún recuerdo.
Una respuesta correcta que no cita el recuerdo correcto **no** cuenta como recuperación de memoria.
"""

TEXT["s5b"] = r"""
**Resultados con el agente Qwen (experimento oficial).** El éxito es 0,22 (M0), 0,11 (M1), 0,11 (M2) y 0,11 (M3), y
`memory_recall_accuracy` y `cross_session_retention` valen **0 en las cuatro**: el modelo **nunca** listó un recuerdo en
`memory_ids`. Que M0 «gane» no significa que la memoria estorbe. Su éxito extra es **S1-2 acertada por casualidad**: sin memoria, busca
«Silicon Valley Bank failure» y pide la señal del 2023-03-13, la misma fecha del ejemplo del prompt. La prueba está en **S3-3**:
también sin memoria, ante una pregunta sobre el COVID, vuelve a usar 2023-03-13 y responde mal. Por eso esos aciertos no cuentan como
recuperación.

La memoria **sí funcionó como recuperación**: el recuerdo correcto apareció entre los mostrados en 4/4 interacciones con M2, 2/4
con M3 y 1/4 con M1 (`gold_retrieved_at3`). M3 quedó casi vacía porque solo consolida tareas con `SUCCESS`, y el agente casi no las
tuvo: su única entrada (F001, SVB) se sirvió en todas las preguntas, pertinente o no. Es el costo de una política de escritura
conservadora con un agente débil. El consumo de contexto fue similar (≈ 1 100–1 190 tokens de prompt por interacción; memoria: 0, 23,
205 y 137 tokens de media).
"""

TEXT["code_exp2_traces"] = r"""
for name in ["M0_none__S1-2", "M0_none__S3-3", "M2_episodic__S1-2", "M2_episodic__S2-3", "M2_episodic__S3-3"]:
    show_trace("exp2", name)
    r = json.loads((TRACES / "exp2" / f"{name}.json").read_text(encoding="utf-8"))
    print("   memoria mostrada:", [(m["memory_id"], m["provenance"]["interaction_id"], m.get("score")) for m in r["memory_shown"]], "\n")
"""

TEXT["s5c"] = r"""
**Los tres casos que pide el enunciado, en las trazas del agente:**
* **La memoria ayuda (a un paso, no al resultado): S1-3 y S2-3.** Sin la fecha en la pregunta, con M2 el agente toma la fecha del
  recuerdo mejor rankeado y recalcula la señal correcta: 2023-03-13 en S1-3 (E001) y 2024-08-30, la vigente, en S2-3 (E005, primero
  por recencia). Sin memoria (M0), en esas mismas interacciones no tiene la fecha: repite una búsqueda (S1-3) o la lectura de un passage (S2-3) y se detiene. Luego vuelve a repetir la acción y no
  termina.
* **No mejora: S1-2 con M1 y M2.** El agente tiene el recuerdo y la fecha correcta, pero pone el id del recuerdo (`E001`) en
  `evidence_ids` en vez de en `memory_ids`; el controlador rechaza dos veces (`finish_unobserved_evidence`) y termina en
  `INVALID_OUTPUT`. El contrato protege la respuesta, pero el modelo no aprende el formato.
* **Recupera información equivocada: S3-3 con M2 (interferencia).** Para «la alerta de SPY durante el COVID», la memoria episódica pone
  primero E008 (aranceles 2025, más reciente) y la correcta E007 tercera. El agente usa la fecha equivocada: calcula 2025-04-08.

### Diagnóstico sin LLM: la misma memoria con un lector determinista (añadido después de ver la corrida con Qwen)

Como el modelo no usa la memoria, la corrida oficial no permite decir **qué memoria es mejor**. Se repitieron las mismas 9 interacciones,
con el mismo controlador, herramientas y almacenes, cambiando el agente por `MemoryReaderWorkflow`: sigue el plan de W0 cuando hay ticker
y fecha y, si no, responde con el **primer** recuerdo mostrado que menciona el activo. Mide la memoria aislada del modelo y no
reemplaza el experimento oficial.
"""

TEXT["s6"] = r"""
Con el lector determinista (`results/exp2/diagnostic_reader/`) el orden es claro: éxito **0,56 → 0,67 → 0,78 → 1,00** (M0 → M3).
* **M1** responde S1-2 (misma sesión) y nada entre sesiones (`cross_session_retention` 0).
* **M2** recupera entre sesiones (S1-3), pero en **S2-3 responde con el recuerdo obsoleto**: E004 (5-ago, *medium*) quedó primero (puntaje 2,40) sobre
  E005 (30-ago, *low*, 2,25): la importancia premia la alerta *medium* más que la recencia a la señal nueva. `stale_memory_error_rate` 0,5 y `wrong_memory_use_rate` 0,33. Guardar historia sin
  invalidarla es un riesgo real, no teórico.
* **M3** acierta las 4 que necesitan memoria (`memory_recall_accuracy` 1,0) porque el hecho del 5-ago quedó `superseded` por el del
  30-ago y ya no se sirve. En S3-3 distingue COVID de aranceles por similitud semántica.

La interferencia de S3-3 aparece con el agente y no con el lector: el ranking de M2 depende de los episodios escritos antes, que
cambian con la política. Es una fragilidad de la memoria episódica, no un resultado estable.

**Demostración rápida en vivo** (sin LLM): W0 sobre T09 y el borrado de un recuerdo persistido, en una copia del almacén.
"""

TEXT["s7"] = r"""
## 6. Matriz de trazabilidad

| Sem. | Recurso estudiado | Concepto | Archivo / función | Evidencia de ejecución | Limitación propia |
|---|---|---|---|---|---|
| **9** | `Semana9/Lectura9-CC-0F4.md` (§2 pipeline/workflow/agente, §3 patrones, §5 Perceive-Decide-Act-Observe e invariantes, §6 ReAct, §8 límites y parada, §10 diseño experimental), `Semana9/Cuaderno9-CC-0F4.ipynb` (§13 controlador, §16 acciones permitidas, §18 ciclos y presupuestos), `Semana9/agent_loop.py`; Yao et al. (ReAct); Anthropic, *Building Effective Agents* | Workflow vs agente; ReAct; controlador con allowlist, presupuestos, detección de repetición y validación del `finish`; comparar contra baselines simples | `src/controller.py::run_task`, `check_finish`; `src/workflow.py::FixedPlanWorkflow`; `src/agent.py::QwenAgentPolicy`; `src/run_exp1.py` | W0 12/15 vs W1 1/15; 14 `REPEATED_ACTION`; 7/7 pruebas de robustez; 20 pruebas unitarias | Un solo modelo (0.5B); el deadline es cooperativo; la parada inmediata ante la primera repetición es estricta y explica parte del 1/15 |
| **10** | Enunciado PC3 §2 (Semana 10: working, episódica, semántica, procedimental, persistencia, invalidación); Park et al. (*Generative Agents*: relevancia + recencia + importancia); Packer et al. (*MemGPT*: contexto acotado frente a almacenamiento externo) | Estado de ejecución ≠ memoria; working vs episódica vs semántica; persistencia; versión, invalidación y borrado; procedencia | `src/memory.py` (`WorkingMemory`, `EpisodicStore`, `SemanticStore`); `src/run_exp2.py`; `tests/test_memory_eval.py` | Agente: recall 0 en M0–M3, recuerdo correcto mostrado 4/4 en M2; lector: 0,56 / 0,67 / 0,78 / 1,00, stale 0,5 en M2 y 0 en M3 | La memoria procedimental es solo el prompt fijo; 9 interacciones; el diagnóstico con lector se añadió tras ver resultados |
| 4 | `Semana4/Cuaderno4` (FAISS, e5) | Recuperación densa e índice | `SemanticStore` reutiliza el encoder e5 y `IndexFlatIP` | 5 hechos en M3 con el lector (1 invalidado) | Sin umbral de similitud: siempre devuelve algo (F001 servido sin ser pertinente) |
| 6 | `Semana6/Cuaderno6`, parcial Exp. 3 | Contratos, errores como datos, reintento selectivo y timeout | `src/tools.py` (`ARG_MODELS`, `ErrorObs`), `tool_risk.call_tool` del parcial | T11 `output_contract`, T12 `attempts=2`, R7 `TIMEOUT` | El timeout de la búsqueda es cooperativo (hilo) |
| 7 | `Semana7/Cuaderno7`, parcial Exp. 1 | Métricas con denominador explícito, análisis de errores | `src/evaluation.py` | `per_task.csv`, `per_interaction.csv`, trazas por tarea | 15 tareas y 9 interacciones: sin pruebas estadísticas |
"""

TEXT["s8"] = r"""
## 7. Conclusiones y limitaciones

1. **¿Cuándo basta un workflow fijo?** Aquí, siempre: la tarea tiene pocas ramas previsibles. W0 resuelve 12/15 con 0 tokens. Sus 3
   fallos (evidencia no pertinente en T04 y T08, fecha no hábil en T09) se arreglan con reglas, no con autonomía.
2. **¿Cuándo aporta valor que el LLM controle las acciones?** Solo se vio en un paso intermedio: reformular la búsqueda (T04). Con un
   modelo de 0.5B ese valor no llega al resultado: 1/15 de éxito, ~21 veces más lento y ~3 000 tokens por tarea.
3. **¿Qué memoria mejora el desempeño?** Con el agente Qwen, ninguna: no usa los recuerdos (recall 0). Aislando la memoria, la
   semántica con versión e invalidación es la única sin errores (1,0). La episódica recupera entre sesiones, pero devuelve el recuerdo
   obsoleto (S2-3) y se confunde entre alertas parecidas (S3-3 con el agente).
4. **¿Cuándo la autonomía o la memoria aumentan el costo o el riesgo?** La autonomía multiplicó latencia y tokens sin ganar éxito. La
   memoria episódica sin invalidación introduce errores de obsolescencia. La semántica depende de su política de escritura: con un
   agente que casi no termina, quedó vacía y sirvió un hecho irrelevante.

**Lo que sostiene la seguridad es el controlador, no el modelo:** en 36 + 15 corridas del agente no salió ninguna alerta con severidad
inventada o evidencia no observada. Su límite: M0 S3-3 respondió con la alerta de SVB a una pregunta sobre el COVID. El controlador
verifica que lo dicho esté sustentado, no que responda a la pregunta.

**Limitaciones.** Un solo LLM (0.5B, CPU, greedy). 15 tareas y 9 interacciones: resultados descriptivos, sin significancia. Benchmark y
evidencia anotados por el autor. El prompt se ajustó con 2 tareas de desarrollo. El diagnóstico con lector determinista y las métricas
`gold_retrieved_*` se añadieron **después** de ver la corrida con Qwen (declarado). Parar ante la primera repetición es una decisión
estricta del controlador: con una política más tolerante W1 quizá habría terminado más tareas, pero no se probó para no cambiar los
límites tras ver los resultados. Las fallas inyectadas son controladas, no de red real. La memoria procedimental no se evaluó.
"""
