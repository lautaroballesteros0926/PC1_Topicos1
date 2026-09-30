# CC0F4 · PC1 — Vigilante de riesgo de mercado con alertas tempranas

**Práctica calificada 1 · clasificador de riesgo de mercado con un LLM pequeño.**
Pedro Lautaro Quispe Ballesteros · Curso CC0F4 (2026-2), Semanas 1-3.

Tema de investigación general: *Sistema multiagente con RAG híbrido para la detección temprana de riesgo de mercado con alertas sustentadas en evidencia*. Esta práctica implementa **solo** la unidad mínima: un clasificador de una señal de mercado con salida JSON validada. No incluye RAG, recuperación híbrida, agentes ni multimodalidad.

## 1. Problema
Un vigilante de riesgo de mercado debe convertir señales (precios, indicadores técnicos) en una alerta con severidad y una justificación revisable. Antes de construir el sistema multiagente hay que responder una pregunta más pequeña: **¿puede un LLM local pequeño aplicar una política de riesgo a indicadores numéricos y devolver una alerta estructurada que se pueda validar automáticamente?**

## 2. Tarea mínima
```text
riesgo de mercado -> clasificar una señal como low / medium / high
```
**Entrada:** activo + tres indicadores (retorno a 5 días, volatilidad de 20 días anualizada, caída desde el máximo de 20 días).
**Salida:** `{"label": "low|medium|high", "confidence": 0..1, "reason": "..."}`.

## 3. Relación con las Semanas 1, 2 y 3
| Semana | Concepto | Dónde aparece (cuaderno, secciones 9 y 5) |
|---|---|---|
| 1 | Transformer, tokens, atención causal | El prompt P1 del caso H1 son 478 tokens (70 de la entrada); los números se fragmentan en dígitos. Prueba empírica de causalidad: cambiar el último token deja los logits de las posiciones previas **idénticos (diferencia 0.0)**. |
| 2 | Logits, softmax, greedy vs sampling, temperature, top-p, contexto, KV cache | Logits del primer token de `label` (H1: `low` 25.95, `medium` 24.65, `high` 23.90 → 0.71 / 0.19 / 0.09). Efecto de T (0.3, 0.8, 1, 2) y top-p (núcleo de 2 tokens). Exp. 3 compara greedy y sampling. KV cache: mismo texto, ≈10× más lento sin caché. |
| 3 | Separación instrucción/entrada/contexto, salida estructurada, JSON Schema, parseable vs schema-valid vs correcto | Prompt P1 con secciones TAREA / ENTRADA / CONTEXTO / CONTRATO; `data/schema.json` validado con `Draft202012Validator`; métricas que separan los tres niveles. |

## 4. Modelo
`Qwen/Qwen2.5-0.5B-Instruct` (revisión `7ae557604adf67be50417f59c2c2f167def9a775`), 494M parámetros, ejecutado localmente en CPU con `float32` y el chat template oficial. Se eligió por reproducibilidad (corre sin GPU) y porque es el modelo de la Semana 3.
Entorno de la corrida: `torch 2.14.0+cpu`, `transformers 5.17.0`.

## 5. Datos
8 señales **reales** de mercado (`data/casos_pc1.csv`). Precios diarios ajustados por dividendos de **Yahoo Finance** vía `yfinance`, descargados una sola vez y cacheados en `data/raw/`. **No se inyectó ningún shock sintético.**

| Caso | Activo | Fecha | Evento | ret5 % | vol20 % | dd20 % | Puntos | Gold |
|---|---|---|---|---|---|---|---|---|
| L1 | SPY | 2017-06-30 | Mercado en calma | -0.55 | 7.17 | -1.17 | 0 | low |
| L2 | GLD | 2019-04-30 | Oro estable | 0.90 | 7.45 | -1.89 | 0 | low |
| L3 | KO | 2021-04-30 | Acción defensiva estable | -0.90 | 9.54 | -1.15 | 0 | low |
| M1 | SPY | 2024-08-05 | Unwind del carry trade | -5.03 | 19.19 | -8.41 | 3 | medium |
| M2 | SPY | 2023-03-13 | Crisis bancaria SVB | -4.72 | 15.79 | -6.91 | 3 | medium |
| H1 | SPY | 2020-03-16 | Crash COVID-19 | -12.54 | 76.95 | -29.11 | 6 | high |
| H2 | IWM | 2022-06-13 | Venta masiva de small caps | -9.17 | 34.14 | -10.61 | 6 | high |
| H3 | SPY | 2025-04-08 | Shock arancelario | -11.50 | 31.44 | -13.72 | 6 | high |

**Etiqueta gold.** Sale de una regla determinista de puntos (`src/market_features.py`), no de un LLM ni de mi criterio caso por caso: 0-2 puntos por indicador (volatilidad: <15 / 15-25 / ≥25; caída: >-5 / -5 a -10 / ≤-10; retorno 5d: >-3 / -3 a -6 / ≤-6); total 0-1 = low, 2-3 = medium, 4-6 = high. La misma regla se entrega al modelo como contexto C0. La columna `event` es solo documentación y **no entra al prompt**.
**Por qué son adecuados:** cubren las tres clases con eventos reconocibles y valores muy distintos; son pocos para revisar cada salida a mano. **Limitación:** 8 casos, 4 activos, 5 de 8 son SPY; ningún resultado se generaliza a otros mercados.

## 6. Salida estructurada
`data/schema.json` (JSON Schema 2020-12):
```json
{
  "type": "object",
  "properties": {
    "label": {"type": "string", "enum": ["low", "medium", "high"]},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "reason": {"type": "string", "minLength": 10, "maxLength": 500}
  },
  "required": ["label", "confidence", "reason"],
  "additionalProperties": false
}
```
Tres niveles de validez, medidos por separado: **parseable** (`json.loads`, exacto o recuperando el primer `{...}`), **schema-valid** (`Draft202012Validator`) y **semánticamente correcto** (`label == gold`; además `reason_grounded`: la explicación cita algún valor real de la entrada). Una salida inválida cuenta como error en accuracy y macro-F1; no se descarta.

## 7. Cómo ejecutar
Todo está en el cuaderno `notebook/PC1_riesgo_mercado.ipynb`. Los precios ya vienen en `data/raw/`, así que no hace falta internet, salvo la primera vez para descargar el modelo (~1 GB).

### Con Docker Compose (recomendado)
Requisito: tener Docker Desktop abierto.

1. Abre una terminal dentro de esta carpeta (la que tiene `docker-compose.yml`).
2. Levanta Jupyter:
   ```bash
   docker compose up
   ```
   La primera vez construye la imagen (unos minutos). Las siguientes veces arranca en segundos.
3. Abre en el navegador <http://localhost:8888/lab> (no pide contraseña).
4. En el panel izquierdo abre `notebook/PC1_riesgo_mercado.ipynb`. Ya trae las salidas guardadas, así que puedes leerlo sin ejecutar nada.
5. Para detenerlo, `Ctrl+C` en la terminal y luego:
   ```bash
   docker compose down
   ```

Notas:
- **Volver a ejecutar todo el cuaderno** (`Run > Run All Cells`) tarda **cerca de 1 hora** en CPU y sobrescribe los resultados guardados. El modelo corre a unos 5 tokens por segundo.
- **Aviso "volume already exists"**: si Docker muestra que el volumen `pc1_hf` ya existe pero no lo creó Compose, es solo una advertencia; el volumen guarda el modelo descargado y se reutiliza.
- **Puerto 8888 ocupado**: cambia `"8888:8888"` por `"8889:8888"` en `docker-compose.yml` y abre `localhost:8889`.
- Si cambias `Dockerfile` o `requirements.txt`, usa `docker compose up --build`.
- Sin token: pensado para uso local. No lo expongas a la red.

### Ejecutar el cuaderno completo sin abrir Jupyter
```bash
docker compose run --rm pc1 jupyter nbconvert --to notebook --execute --inplace notebook/PC1_riesgo_mercado.ipynb
```

### Sin Docker
Python 3.12:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
jupyter lab
```

### Regenerar los datos (opcional)
```bash
python src/download_prices.py   # descarga precios de Yahoo Finance a data/raw/
python src/build_dataset.py     # reconstruye data/casos_pc1.csv y data/schema.json
```

### Reproducibilidad
Seed 42. `results/config_corrida.json` guarda la revisión del modelo, las versiones de las librerías, el hash de los casos y de cada prompt, el schema y la política de decoding. `results/prompts_renderizados.jsonl` contiene el prompt exacto enviado en cada caso y condición. Greedy es determinista en una misma máquina y con las mismas versiones; en otro hardware las cifras pueden variar un poco.

## 8. Prompts utilizados
Guardados tal cual en `prompts/` (el cuaderno los lee de esos archivos):
- `baseline.txt` — **P0**: system "Eres un asistente de finanzas." y user "Clasifica el riesgo de mercado (low, medium o high) de la siguiente señal y responde en JSON." + entrada + contexto.
- `experimento_prompt.txt` — **P1**: system de clasificador que obedece un contrato; user con secciones TAREA (con procedimiento: puntuar cada indicador, sumar, convertir a label) / ENTRADA / CONTEXTO / CONTRATO (claves exactas, tipos, rango, sin Markdown ni texto extra).
- `experimento_contexto.txt` — **idéntico a P1**; el Exp. 2 solo cambia lo que se inserta en `{context}` (C0 o C1).

## 9. Experimento 1 — Prompt (P0 vs P1)
**Variable modificada:** prompt. **Fijo:** modelo, 8 casos, contexto C0, greedy, `max_new_tokens=192`, schema. P1 cambia varias cosas juntas (system prompt, secciones, procedimiento, contrato de formato), así que el efecto se atribuye al paquete, no a cada parte.

| | schema_valid_rate | accuracy | macro-F1 | error ordinal medio | reason anclada |
|---|---|---|---|---|---|
| P0 | **0.00** (0/8) | 0.00 | 0.00 | – | – |
| P1 | **1.00** (8/8) | 0.375 (3/8) | 0.182 | 1.00 | 7/8 |

P0 inventó claves en los 8 casos (`"Riesgo de Mercado"`, `"Severidad Segun el Total de Puntos"`): JSON parseable pero no schema-valid. P1 corrige eso. Sin embargo, con P1 el modelo respondió `low` en los 8 casos; la accuracy de 0.375 es exactamente la de un clasificador constante (3 de 8 son `low`). **La mejora que se sostiene es de formato, no de criterio.**
*Caso analizado, H1 (COVID, gold=high):* P1 devuelve `label: "low"` con la razón "una situación de riesgo de mercado muy alta": schema-valid, correctamente anclada y aun así incorrecta e inconsistente consigo misma.

## 10. Experimento 2 — Contexto (C0 vs C1 conflictivo)
**Variable modificada:** contexto. C1 = C0 + una "nota de mesa" que contradice los indicadores (p. ej. "mercado tranquilo, riesgo bajo" en H1; "riesgo alto" en L1). Las notas se redactaron antes de ver resultados (`src/build_dataset.py`). **Fijo:** modelo, prompt P1, casos, greedy, schema.

| | schema_valid_rate | accuracy | macro-F1 | etiquetas que cambiaron | reason anclada |
|---|---|---|---|---|---|
| C0 | 1.00 | 0.375 | 0.182 | – | 7/8 |
| C1 | 1.00 | 0.375 | 0.182 | 0 de 8 | 6/8 |

La etiqueta no cambió en ningún caso, pero esto **no demuestra robustez**: con C0 el modelo ya dice `low` en todo, así que las notas hacia "bajo" no tenían margen, y en los 3 casos `low` la nota hacia "alto" tampoco lo movió (sesgo hacia `low`). Lo que sí cambió fue la explicación: en H1 el modelo copia el lenguaje de la nota ("el mercado se muestra tranquilo y estable, con una volatilidad de 76.95%"), en M2 repite "la situación es normal y no requiere seguimiento" y la confianza de M1 sube de 0.7 a 0.9. El contexto conflictivo contaminó las explicaciones sin alterar la etiqueta.

## 11. Experimento 3 — Decoding (greedy vs sampling)
**Variable modificada:** política de decoding. **D0:** greedy. **D1:** `temperature=0.8`, `top_p=0.9`, `top_k=0` (desactivado), seeds 42-46, sobre los 8 casos (40 muestras). **Fijo:** modelo, P1, C0, casos, `max_new_tokens=192`, `repetition_penalty=1.05`.

| | schema_valid_rate | accuracy | macro-F1 | estabilidad de etiqueta vs greedy | reason anclada |
|---|---|---|---|---|---|
| D0 | 1.000 (8/8) | 0.375 | 0.182 | – | 7/8 |
| D1 | 0.925 (37/40) | 0.325 | 0.241 | 0.75 (30/40) | 0.946 |

Sampling dio 3 salidas inválidas (2 bucles repetitivos que no cerraron el JSON, 1 `reason` fuera de schema). Hay 4-5 explicaciones distintas por caso y la confianza va de 0.6 a 0.9. Hallazgo importante: **la seed 46 dio `medium` en 7 de 8 casos** y las demás casi siempre `low`; con la misma seed y una distribución de primer token casi igual entre casos (`low`≈0.7, `medium`≈0.2, `high`≈0.1), la seed elige lo mismo en todos. Las 5 seeds no son muestras independientes entre casos. El macro-F1 más alto de D1 se debe a que aparece la clase `medium` con aciertos casuales, no a un mejor criterio. Con `top_p=0.9` el núcleo del primer token tiene 2 tokens, así que `high` (≈0.09) casi nunca puede elegirse.

## 12. Resultados
Tabla de registro (detalle en `results/resumen_experimentos.csv` y `results/fig_resumen.png`):

| Experimento | Variable modificada | Variable fija | Métricas | Resultado | Conclusión |
|---|---|---|---|---|---|
| Prompt | prompt (P0→P1) | modelo, casos, contexto, decoding | schema_valid_rate, accuracy, macro-F1 | schema-valid 0.00→1.00; accuracy 0.00→0.375 (= siempre `low`) | P1 arregla el formato; no hay evidencia de mejor criterio |
| Contexto | contexto (C0→C1 conflictivo) | modelo, prompt, casos, decoding | schema_valid_rate, accuracy, macro-F1, cambios de etiqueta, reason anclada | 1.00→1.00; 0.375→0.375; 0/8 cambios; anclada 7/8→6/8 | la etiqueta no se movió (piso por sesgo a `low`); las explicaciones sí se contaminaron |
| Decoding | decoding (greedy→sampling T=0.8, p=0.9, 5 seeds) | modelo, prompt, contexto, casos, max_new_tokens | schema_valid_rate, accuracy, estabilidad | 1.000→0.925; 0.375→0.325; estabilidad 0.75 | sampling añade variabilidad y algunos fallos estructurales; no mejora ni empeora demostrablemente la accuracy |

Con 8 casos, una corrida por condición y un modelo de 0.5B, estas cifras describen el piloto y no permiten afirmar que un prompt o un decoding "sea mejor en general".

## 13. Limitación
- **Escala:** 8 casos, una corrida por condición; sin intervalos de confianza. Diferencias de 1 caso (0.125) están dentro del ruido.
- **Modelo:** un 0.5B responde `low` en todo y no aplica la regla de puntos aunque está escrita; el piso de accuracy limita lo que Exp. 2 y 3 pueden mostrar. No se probó un modelo mayor.
- **Contaminación posible:** el prompt incluye ticker y valores; el modelo podría conocer eventos famosos. No se incluyeron fechas ni nombres de eventos.
- **Datos:** precios ajustados por dividendos; 5 de 8 casos son SPY; la etiqueta depende de umbrales que definí yo (documentados, fijados antes de correr el modelo).
- **Seeds compartidas:** en Exp. 3 las mismas 5 seeds se usan en los 8 casos, lo que correlaciona las muestras.
- **Exp. 1 agrupado:** P1 cambia varios elementos a la vez; no se aísla cuál causó la mejora de formato.
- **Hardware:** la corrida es en CPU y los tiempos (≈5 tokens/s) son de una laptop con Docker Desktop.

## 14. Conclusión
Con estos 8 casos reales: (1) un contrato de salida explícito en el prompt fue la diferencia entre 0/8 y 8/8 salidas schema-valid; (2) la validez estructural no implicó corrección semántica: el modelo dijo `low` en todos los casos con greedy, incluido el crash de COVID, y a veces con una explicación que se contradice; (3) una nota conflictiva no cambió ninguna etiqueta pero sí contaminó las justificaciones; (4) sampling introdujo variabilidad (75 % de estabilidad) y un 7.5 % de salidas inválidas, con la advertencia de que las seeds compartidas hacen que la variabilidad entre casos esté correlacionada. Lo que sigue sin poder afirmarse es que el modelo pueda clasificar riesgo con esta política. Próximos pasos para el sistema completo: probar un modelo mayor, ampliar los casos, medir con intervalos de confianza y separar el cálculo de indicadores (código determinista) de la lectura del modelo.

## 15. Fuentes
**Primarias (método, modelo, problema técnico)**
1. Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS. arXiv:1706.03762. — arquitectura Transformer y atención causal (Semana 1).
2. Qwen Team (2024). *Qwen2.5 Technical Report.* arXiv:2412.15115. — modelo `Qwen2.5-0.5B-Instruct`.
3. Holtzman, A., et al. (2020). *The Curious Case of Neural Text Degeneration.* ICLR. arXiv:1904.09751. — top-p (nucleus sampling) y degeneración por decoding.

**Adicionales**
4. Fan, A., Lewis, M., Dauphin, Y. (2018). *Hierarchical Neural Story Generation.* ACL. arXiv:1805.04833. — sampling top-k.
5. JSON Schema, *Draft 2020-12*. https://json-schema.org/draft/2020-12 — especificación del contrato; validación con la librería `jsonschema` (`Draft202012Validator`).
6. Yahoo Finance, precios históricos obtenidos con `yfinance` (https://github.com/ranaroussi/yfinance). — procedencia de los datos.
7. Material del curso CC0F4 (Semanas 1-3): `Cuaderno2-CC-0F4.ipynb` y `Cuaderno3-CC-0F4.ipynb`, repositorio `kapumota/CC-0F4`. — convenciones de schema, prompt en secciones y evaluación (accuracy, macro-F1, auditoría de errores).

*Nota: las referencias se listan tal como las recuerdo; conviene verificar título y numeración arXiv en las herramientas de literatura del curso (Elicit, Consensus, Connected Papers) antes de exponer.*
