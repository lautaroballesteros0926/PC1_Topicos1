"""Resume la AUDITORIA MANUAL del Experimento 2 a partir de las plantillas que completa la persona.

Plantillas (results/exp2/qwen/):
  audit_answers_template.csv  -> por respuesta: answer_correct_0_1, abstention_appropriate_0_1, citations_correct_0_1, notes
  audit_claims_template.csv   -> por afirmacion: supported_yes_partial_no (yes / partial / no), notes

Criterios (se aplican a mano, contra la respuesta de referencia, los hechos requeridos y el texto de los passages citados):
  - correccion de la respuesta: contiene los hechos requeridos y no contradice la referencia (1), o no (0). Abstenerse ante una pregunta
    respondible NO es correcto (0); abstenerse ante una sin evidencia es la conducta esperada (abstention_appropriate = 1).
  - correccion de las citas: cada id citado existe en el contexto Y el passage citado contiene el hecho que se afirma (1), o no (0).
  - faithfulness / grounding: cada afirmacion esta respaldada por el passage citado o por el contexto (yes), solo en parte (partial) o no (no).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from common import RESULTS

BASE = RESULTS / "exp2" / "qwen"


def _filled(series: pd.Series) -> pd.Series:
    return series.notna() & (series.astype(str).str.strip() != "")


def summarize_audit(base=BASE) -> dict[str, Any]:
    ans = pd.read_csv(base / "audit_answers_template.csv")
    claims = pd.read_csv(base / "audit_claims_template.csv")
    n_ans = int(_filled(ans["answer_correct_0_1"]).sum() + _filled(ans["abstention_appropriate_0_1"]).sum())
    n_claims = int(_filled(claims["supported_yes_partial_no"]).sum())
    if n_ans == 0 and n_claims == 0:
        return {"status": "pendiente", "answers": None, "claims": None}

    ans = ans.copy()
    for c in ("answer_correct_0_1", "abstention_appropriate_0_1", "citations_correct_0_1"):
        ans[c] = pd.to_numeric(ans[c], errors="coerce")
    answerable = ans[~ans["expected_abstain"]]
    no_evidence = ans[ans["expected_abstain"]]
    table = pd.DataFrame({
        "respondibles: respuesta correcta": answerable.groupby("condition")["answer_correct_0_1"].mean(),
        "sin evidencia: abstencion adecuada": no_evidence.groupby("condition")["abstention_appropriate_0_1"].mean(),
        "citas correctas (solo RAG)": ans[ans["condition"] == "rag"].groupby("condition")["citations_correct_0_1"].mean(),
    }).round(3)

    cl = claims[_filled(claims["supported_yes_partial_no"])].copy()
    cl["supported"] = cl["supported_yes_partial_no"].str.strip().str.lower()
    score = {"yes": 1.0, "partial": 0.5, "no": 0.0}
    cl["score"] = cl["supported"].map(score)
    faith = cl.groupby("condition").agg(afirmaciones=("claim_id", "count"), yes=("supported", lambda s: int((s == "yes").sum())),
                                        partial=("supported", lambda s: int((s == "partial").sum())),
                                        no=("supported", lambda s: int((s == "no").sum())), faithfulness=("score", "mean")).round(3)
    pending = {"respuestas_sin_auditar": int(len(ans) - n_ans), "afirmaciones_sin_auditar": int(len(claims) - n_claims)}
    return {"status": "completa" if not any(pending.values()) else "parcial", "answers": table, "claims": faith, "pending": pending,
            "unsupported": cl[cl["supported"] == "no"][["claim_id", "claim", "notes"]]}


if __name__ == "__main__":
    res = summarize_audit()
    print(res["status"])
    if res["answers"] is not None:
        print(res["answers"].to_string()); print(res["claims"].to_string()); print(res["pending"])
