# Corrida 0 (descartada como resultado, conservada como evidencia)

Primera ejecución completa con `max_new_tokens=96`. Todas las generaciones con P1 llegaron a exactamente
96 tokens: el modelo abre con un bloque ```json y escribe un `reason` largo, así que el JSON quedó
**truncado antes de cerrarse** y `schema_valid_rate` fue 0.0 en P1/C0/D0 (0.125 en C1, 0.05 en D1).
Con ambos lados de cada comparación en ~0, los experimentos 2 y 3 no eran informativos.

Causa: presupuesto de tokens insuficiente (decisión de configuración mía), no un efecto del prompt, el contexto o el decoding.
Corrección: `max_new_tokens=192` para TODAS las condiciones. Los prompts, casos y contextos no se modificaron.
