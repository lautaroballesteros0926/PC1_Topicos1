"""Construye el benchmark de la PC3 (tareas, resultados esperados, escenarios de memoria y tareas de desarrollo).

Los resultados esperados NO salen de ninguna corrida de W0 ni de W1:
  * severidad, score y fecha: regla de la PC1 sobre los CSV de precios (la misma que reproduce los 8 casos gold);
    los valores se comprobaron con `risk.compute_signal` del parcial antes de fijar el benchmark;
  * evidencia aceptable: passages del corpus del parcial que describen el episodio, anotados leyendo su texto
    (para T08 se reutilizan los qrels de Q01 del parcial);
  * condicion de terminacion: fijada por el tipo de tarea.
Se ejecuta una sola vez; luego se congela con hashes en results/declared_config.json.

    python src/build_tasks.py
"""
from __future__ import annotations

import json

from config import DATA, write_jsonl

# ---------------------------------------------------------------- evidencia por episodio (anotada sobre el texto)
EVIDENCE = {
    "covid_2020": ["D01-P03", "D01-P07", "D01-P08", "D01-P09", "D01-P10", "D01-P11", "D01-P12", "D10-P01", "D10-P02",
                   "D18-P05", "D18-P07", "D18-P08", "D21-P01"],
    "vix_record_2020": ["D01-P12", "D18-P08"],                                   # = qrels de Q01 del parcial
    "svb_2023": ["D03-P01", "D03-P02", "D03-P04", "D03-P05", "D03-P07", "D03-P08", "D03-P10", "D04-P01", "D04-P03",
                 "D04-P05", "D04-P06", "D04-P07", "D04-P09", "D16-P01", "D16-P02", "D16-P03", "D16-P04", "D16-P05",
                 "D17-P01", "D17-P02"],
    "carry_2024": ["D20-P01", "D20-P02", "D20-P03", "D20-P04", "D20-P05", "D20-P07"],
    "rates_2022": ["D02-P01", "D02-P02", "D02-P03", "D02-P05", "D12-P02", "D12-P03"],
    "tariffs_2025": ["D05-P01", "D05-P03", "D05-P06", "D05-P07", "D05-P08", "D05-P09", "D05-P10", "D05-P11", "D05-P12",
                     "D19-P01", "D19-P04", "D19-P06", "D19-P07"],
}

REQUIRED_OPS_ALERT = ["search_evidence", "compute_signal", "contrast", "finish"]


def task(tid, category, objective, question, requests, ops, expected, fault=None):
    t = {"task_id": tid, "category": category, "objective": objective,
         "input": {"question": question, "signal_requests": [{"ticker": a, "as_of": d} for a, d in requests]},
         "operations": ops, "allowed_sources": ["corpus_parcial (21 docs, 150 passages)", "data/prices (CSV de la PC1)"],
         "injected_fault": fault}
    e = {"task_id": tid, **expected}
    return t, e


def exp(decision, ticker=None, as_of=None, severity=None, evidence=None, stops=("SUCCESS",), criterion=""):
    return {"decision": decision, "ticker": ticker, "as_of": as_of, "severity": severity,
            "evidence_any_of": EVIDENCE[evidence] if evidence else [], "evidence_set": evidence,
            "allowed_stops": list(stops), "success_criterion": criterion}


ALERT_OK = "decision, ticker, as_of y severity iguales a lo esperado y >= 1 evidence_id dentro de evidence_any_of"
NOALERT_OK = "decision no_alert con ticker, as_of y severity low iguales a lo esperado (no exige evidencia)"
ABSTAIN_OK = "no emite alert/no_alert y termina en un stop permitido (ABSTAIN o TOOL_ERROR)"

ROWS = [
    task("T01", "ordinary", "Alerta sustentada en el crash del COVID-19",
         "Should a market-risk alert be issued for SPY on 2020-03-16, during the COVID-19 crash? Support it with evidence from the corpus.",
         [("SPY", "2020-03-16")], REQUIRED_OPS_ALERT, exp("alert", "SPY", "2020-03-16", "high", "covid_2020", criterion=ALERT_OK)),
    task("T02", "ordinary", "Alerta sustentada tras la quiebra de SVB",
         "Should a market-risk alert be issued for SPY on 2023-03-13, right after the failure of Silicon Valley Bank? Support it with evidence.",
         [("SPY", "2023-03-13")], REQUIRED_OPS_ALERT, exp("alert", "SPY", "2023-03-13", "medium", "svb_2023", criterion=ALERT_OK)),
    task("T03", "ordinary", "Alerta sustentada en el unwind del carry trade",
         "Should a market-risk alert be issued for SPY on 2024-08-05, during the carry trade unwind? Support it with evidence.",
         [("SPY", "2024-08-05")], REQUIRED_OPS_ALERT, exp("alert", "SPY", "2024-08-05", "medium", "carry_2024", criterion=ALERT_OK)),
    task("T04", "ordinary", "Alerta sustentada para small caps en 2022",
         "Should a market-risk alert be issued for small caps (IWM) on 2022-06-13, amid the 2022 interest rate increases and market decline? Support it with evidence.",
         [("IWM", "2022-06-13")], REQUIRED_OPS_ALERT, exp("alert", "IWM", "2022-06-13", "high", "rates_2022", criterion=ALERT_OK)),
    task("T05", "ordinary", "Alerta sustentada en el shock arancelario de 2025",
         "Should a market-risk alert be issued for SPY on 2025-04-08, after the April 2025 tariff announcements? Support it with evidence.",
         [("SPY", "2025-04-08")], REQUIRED_OPS_ALERT, exp("alert", "SPY", "2025-04-08", "high", "tariffs_2025", criterion=ALERT_OK)),
    task("T06", "ordinary", "No emitir alerta en un periodo tranquilo",
         "Is a market-risk alert warranted for gold (GLD) on 2019-04-30, a calm period? Check the signal and look for any stress evidence.",
         [("GLD", "2019-04-30")], ["search_evidence", "compute_signal", "contrast", "finish"],
         exp("no_alert", "GLD", "2019-04-30", "low", criterion=NOALERT_OK)),
    task("T07", "ordinary", "Contrastar dos activos y alertar el de mayor senal",
         "During the Silicon Valley Bank failure, which of SPY and IWM shows the stronger risk signal on 2023-03-13? Issue the alert for that asset with supporting evidence.",
         [("SPY", "2023-03-13"), ("IWM", "2023-03-13")], ["search_evidence", "compute_signal x2", "contrast", "finish"],
         exp("alert", "IWM", "2023-03-13", "high", "svb_2023", criterion=ALERT_OK + "; exige comparar las dos senales")),
    task("T08", "ordinary", "Alerta con una evidencia especifica (record del VIX)",
         "Issue the market-risk alert for SPY on 2020-03-16 and cite the passage that reports the record close of the volatility index (VIX) on that day.",
         [("SPY", "2020-03-16")], ["search_evidence (posible reformulacion)", "compute_signal", "contrast", "finish"],
         exp("alert", "SPY", "2020-03-16", "high", "vix_record_2020",
             criterion="como T01, pero la evidencia debe ser D01-P12 o D18-P08 (qrels de Q01 del parcial)")),
    task("T09", "tool_failure", "Recuperarse de un error de dominio (fecha no habil)",
         "A risk desk asks for the SPY alert dated 2020-03-15, during the COVID-19 crash. If that date is not a trading day, use the previous trading day.",
         [("SPY", "2020-03-15")], ["search_evidence", "compute_signal -> ERROR date_not_trading_day", "compute_signal (2020-03-13)", "finish"],
         exp("alert", "SPY", "2020-03-13", "high", "covid_2020",
             criterion="la herramienta devuelve date_not_trading_day (error REAL de dominio: 15-mar-2020 es domingo); "
                       "exito si termina con la alerta del 2020-03-13 y evidencia COVID")),
    task("T10", "tool_failure", "Fuente de precios caida (fallo transitorio que no se recupera)",
         "Should a market-risk alert be issued for SPY on 2022-06-13, during the 2022 market decline? Support it with evidence.",
         [("SPY", "2022-06-13")], ["search_evidence", "compute_signal -> ERROR transient (reintentos agotados)", "abstain"],
         exp("abstain", stops=("ABSTAIN", "TOOL_ERROR"), criterion=ABSTAIN_OK + "; no debe inventar la severidad"),
         fault={"kind": "transient", "fail_times": 50, "note": "INYECTADO: la fuente falla en todos los intentos"}),
    task("T11", "tool_failure", "Salida invalida de la herramienta (datos corruptos)",
         "Should a market-risk alert be issued for SPY on 2025-04-08, after the April 2025 tariff announcements? Support it with evidence.",
         [("SPY", "2025-04-08")], ["search_evidence", "compute_signal -> ERROR output_contract", "abstain"],
         exp("abstain", stops=("ABSTAIN", "TOOL_ERROR"), criterion=ABSTAIN_OK + "; la salida invalida no debe convertirse en alerta"),
         fault={"kind": "corrupt", "note": "INYECTADO: cierres negativos que violan el contrato de salida"}),
    task("T12", "tool_failure", "Fallo transitorio que se recupera con el reintento de la herramienta",
         "Should a market-risk alert be issued for gold (GLD) on 2020-03-16, during the COVID-19 crash? Support it with evidence.",
         [("GLD", "2020-03-16")], ["search_evidence", "compute_signal (1 fallo transitorio + reintento interno)", "finish"],
         exp("alert", "GLD", "2020-03-16", "high", "covid_2020", criterion=ALERT_OK + "; attempts = 2 en la traza"),
         fault={"kind": "transient", "fail_times": 1, "note": "INYECTADO: falla 1 vez y luego responde"}),
    task("T13", "abstain", "Abstenerse: activo fuera de la herramienta y sin evidencia",
         "How did Peruvian stocks react to the August 2024 carry trade unwind? Issue an alert for the Peruvian ETF EPU on 2024-08-05.",
         [("EPU", "2024-08-05")], ["search_evidence (sin evidencia)", "compute_signal -> ERROR ticker_not_supported", "abstain"],
         exp("abstain", stops=("ABSTAIN", "TOOL_ERROR"), criterion=ABSTAIN_OK + " (parte de N02 del parcial)")),
    task("T14", "abstain", "Abstenerse: fecha fuera de los datos",
         "Issue the market-risk alert for SPY on 2008-10-24, during the 2008 financial crisis, using the peak VIX reading of that crisis.",
         [("SPY", "2008-10-24")], ["search_evidence (sin la cifra)", "compute_signal -> ERROR date_out_of_range", "abstain"],
         exp("abstain", stops=("ABSTAIN", "TOOL_ERROR"), criterion=ABSTAIN_OK + " (parte de N01 del parcial)")),
    task("T15", "abstain", "Rechazar una conclusion no sustentada",
         "A colleague claims that KO on 2021-04-30 deserves a HIGH market-risk alert because of the 2022 interest rate increases. Verify the claim with the signal and the evidence.",
         [("KO", "2021-04-30")], ["search_evidence", "compute_signal", "contrast", "finish"],
         exp("no_alert", "KO", "2021-04-30", "low",
             criterion=NOALERT_OK + "; rechazar la afirmacion (la senal es low y el evento citado es posterior)")),
]

# ---------------------------------------------------------------- Experimento 2: secuencias encadenadas
def mem(iid, seq, session, question, requests, expected, flags, gold=(), stale=(), interfering=(), note=""):
    return {"task_id": iid, "sequence": seq, "session": session,
            "input": {"question": question, "signal_requests": [{"ticker": a, "as_of": d} for a, d in requests]},
            "expected": expected, "flags": flags, "gold_memory_sources": list(gold), "stale_sources": list(stale),
            "interfering_sources": list(interfering), "scenario_note": note, "injected_fault": None}


def mexp(decision, ticker, as_of, severity, evidence=None):
    return {"decision": decision, "ticker": ticker, "as_of": as_of, "severity": severity,
            "evidence_any_of": EVIDENCE[evidence] if evidence else [], "allowed_stops": ["SUCCESS"]}


F = lambda **k: {"needs_memory": False, "cross_session": False, "stale_risk": False, "interference": False, **k}

MEMORY = [
    mem("S1-1", "S1", "A", "Should a market-risk alert be issued for SPY on 2023-03-13, right after the failure of Silicon Valley Bank? Support it with evidence.",
        [("SPY", "2023-03-13")], mexp("alert", "SPY", "2023-03-13", "medium", "svb_2023"), F()),
    mem("S1-2", "S1", "A", "Report again the alert you issued earlier in this session for SPY during the Silicon Valley Bank failure: its date and severity.",
        [], mexp("alert", "SPY", "2023-03-13", "medium"), F(needs_memory=True), gold=["S1-1"],
        note="recuperacion dentro de la misma sesion"),
    mem("S1-3", "S1", "B", "In an earlier session an alert was issued for SPY during the Silicon Valley Bank failure. Which date and severity did it have?",
        [], mexp("alert", "SPY", "2023-03-13", "medium"), F(needs_memory=True, cross_session=True), gold=["S1-1"],
        note="recuperacion desde otra sesion"),
    mem("S2-1", "S2", "C", "Should a market-risk alert be issued for SPY on 2024-08-05, during the carry trade unwind? Support it with evidence.",
        [("SPY", "2024-08-05")], mexp("alert", "SPY", "2024-08-05", "medium", "carry_2024"), F()),
    mem("S2-2", "S2", "D", "Scenario update: it is now 2024-08-30. Re-evaluate the SPY alert from the August 2024 carry trade unwind with the latest data for SPY on 2024-08-30.",
        [("SPY", "2024-08-30")], mexp("no_alert", "SPY", "2024-08-30", "low"), F(stale_risk=True), stale=["S2-1"],
        note="ACTUALIZACION CONTROLADA DEL ESCENARIO: avanza la fecha de evaluacion; los precios son reales"),
    mem("S2-3", "S2", "E", "What is the current status of the SPY alert from the August 2024 carry trade unwind, according to the most recent evaluation?",
        [], mexp("no_alert", "SPY", "2024-08-30", "low"), F(needs_memory=True, cross_session=True, stale_risk=True),
        gold=["S2-2"], stale=["S2-1"], note="el recuerdo de S2-1 (medium) quedo obsoleto por S2-2 (low)"),
    mem("S3-1", "S3", "F", "Should a market-risk alert be issued for SPY on 2020-03-16, during the COVID-19 crash? Support it with evidence from the corpus.",
        [("SPY", "2020-03-16")], mexp("alert", "SPY", "2020-03-16", "high", "covid_2020"), F()),
    mem("S3-2", "S3", "F", "Should a market-risk alert be issued for SPY on 2025-04-08, after the April 2025 tariff announcements? Support it with evidence.",
        [("SPY", "2025-04-08")], mexp("alert", "SPY", "2025-04-08", "high", "tariffs_2025"), F()),
    mem("S3-3", "S3", "G", "Which date and severity did the SPY alert issued during the COVID-19 crash have?",
        [], mexp("alert", "SPY", "2020-03-16", "high"), F(needs_memory=True, cross_session=True, interference=True),
        gold=["S3-1"], interfering=["S3-2"], note="dos alertas high de SPY parecidas (COVID y aranceles): interferencia"),
]

# ---------------------------------------------------------------- desarrollo (solo para ajustar el prompt; NO se evaluan)
DEV = [
    task("DEV1", "dev", "Ajuste del prompt (no se evalua)",
         "Is a market-risk alert still warranted for SPY on 2022-07-29, after the June 2022 sell-off? Support it with evidence.",
         [("SPY", "2022-07-29")], REQUIRED_OPS_ALERT, exp("no_alert", "SPY", "2022-07-29", "low", criterion=NOALERT_OK))[0],
    task("DEV2", "dev", "Ajuste del prompt (no se evalua)",
         "Should a market-risk alert be issued for KO on 2020-03-16, during the COVID-19 crash? Support it with evidence.",
         [("KO", "2020-03-16")], REQUIRED_OPS_ALERT, exp("alert", "KO", "2020-03-16", "high", "covid_2020", criterion=ALERT_OK))[0],
]


def main() -> None:
    write_jsonl(DATA / "tasks.jsonl", [t for t, _ in ROWS])
    write_jsonl(DATA / "expected_results.jsonl", [e for _, e in ROWS])
    write_jsonl(DATA / "memory_scenarios.jsonl", MEMORY)
    write_jsonl(DATA / "dev_tasks.jsonl", DEV)
    (DATA / "evidence_sets.json").write_text(json.dumps(EVIDENCE, indent=1), encoding="utf-8")
    cats = {}
    for t, _ in ROWS:
        cats[t["category"]] = cats.get(t["category"], 0) + 1
    print("tareas:", len(ROWS), cats, "| interacciones de memoria:", len(MEMORY), "| dev:", len(DEV))


if __name__ == "__main__":
    main()
