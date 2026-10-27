# run0 — max_tokens=300 (descartada)

Primera corrida del anexo DeepSeek. `deepseek-flash` razona antes de responder y los tokens de razonamiento cuentan contra `max_tokens`.
Con 300, 8 de 26 respuestas terminaron con `finish_reason=length`; 3 quedaron vacias (Q02 rag, Q07 closed_book, Q13 closed_book) y otras cortadas.
Es una falla de configuracion mia, no un resultado del experimento: se conserva por transparencia y se repitio con `max_tokens=2000`
(`../generations.jsonl`). No se usa en ninguna conclusion.
