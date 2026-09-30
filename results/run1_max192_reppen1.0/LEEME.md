# Corrida 1 (archivada como evidencia)

`max_new_tokens=192`, `repetition_penalty=1.0`, `reason` con `maxLength=240`.
Resultado: `schema_valid_rate` P0=0.00, P1=0.125, C1=0.25, D1=0.175 y `accuracy` ~0 en todas las condiciones.
Causas de falla en las 64 salidas distintas: `reason` mayor a 240 caracteres (36), JSON sin cerrar por bucles
repetitivos (10), claves inventadas en P0 (8), válidas (10). El modelo dijo `low` en 8/8 casos con P1 y C1.

Ajuste para la corrida 2 (solo infraestructura, no variables experimentales):
`repetition_penalty=1.05` (default de Qwen y del cuaderno de la Semana 3) y `reason` con `maxLength=500`.
