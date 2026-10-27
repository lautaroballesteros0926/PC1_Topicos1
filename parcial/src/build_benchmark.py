"""Genera consultas, qrels y respuestas de referencia del examen parcial y valida su consistencia con el corpus.

Uso:  python src/build_benchmark.py

Requiere data/corpus_parcial.jsonl (python src/build_corpus.py). Genera en data/:
  queries_parcial.jsonl            22 necesidades de informacion con evidencia en el corpus (Exp. 1 y 2)
  qrels_parcial.json               query_id -> [passage_id relevantes]  (binarios, sobre passages, no chunks)
  reference_answers_parcial.jsonl  respuesta de referencia + hechos requeridos para las 10 consultas del Exp. 2
  queries_sin_evidencia.jsonl      3 consultas SIN evidencia en el corpus (esperado: abstencion; solo Exp. 2)
  benchmark_manifest.json          hashes del corpus, consultas, qrels y referencias

Regla de congelamiento (igual que Semana 7): corpus, consultas, qrels y referencias se redactaron leyendo el corpus y
ANTES de ejecutar cualquier sistema de recuperacion o generacion; no se modifican despues de ver resultados. Si se corrige
una anotacion, se registra como nueva version del benchmark.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
VERSION = "parcial-v1"

# (id, consulta, dificultad, categoria, passages relevantes, origen en PC2)
QUERIES = [
    ("Q01", "How high did the VIX close on the worst day of the March 2020 COVID-19 market crash?",
     "easy", "numeric_lookup", ["D01-P12", "D18-P08"], "PC2-Q01"),
    ("Q02", "Why couldn't Silicon Valley Bank withstand withdrawals even though it held such a large portfolio of bonds?",
     "medium", "causal_multi_doc", ["D03-P01", "D03-P04", "D17-P02", "D04-P01"], "PC2-Q02"),
    ("Q03", "What does it mean to borrow in a cheap currency to invest elsewhere, and why can that trigger sudden mass selling?",
     "hard", "conceptual_multi_doc", ["D08-P01", "D08-P03", "D20-P01", "D20-P02", "D20-P05"], "PC2-Q03"),
    ("Q04", "What did the Federal Reserve do with interest rates in response to the coronavirus panic?",
     "hard", "ambiguous_similar_docs", ["D09-P01", "D10-P02", "D01-P02"], "PC2-Q04"),
    ("Q05", "How do you measure how far an asset has fallen from its most recent peak?",
     "easy", "definition", ["D07-P01", "D07-P02"], "PC2-Q05"),
    ("Q06", "How did U.S. Treasury bonds react when the April 2025 tariffs were announced?",
     "medium", "causal_multi_doc", ["D05-P03", "D05-P09", "D05-P10", "D19-P05"], "PC2-Q06"),
    ("Q07", "Which FOMC member dissented at the March 2020 emergency meeting, and what rate did they prefer?",
     "medium", "boilerplate_distractors", ["D10-P05"], None),
    ("Q08", "What did the FOMC decide on the federal funds rate in June 2022 and what did it signal about further increases?",
     "hard", "boilerplate_distractors", ["D12-P03"], None),
    ("Q09", "What did the Fed say about the health of the U.S. banking system right after the Silicon Valley Bank failure?",
     "medium", "multi_doc", ["D13-P02", "D16-P06"], None),
    ("Q10", "Which emergency lending program did the Fed create in March 2023 for banks, and how were pledged securities valued?",
     "medium", "multi_doc", ["D04-P08", "D16-P05"], None),
    ("Q11", "Why did Federal Reserve supervisors fail to stop Silicon Valley Bank's problems before it collapsed?",
     "hard", "causal_multi_passage", ["D17-P01", "D17-P04", "D17-P05", "D17-P06", "D17-P07"], None),
    ("Q12", "Which 2018 law raised the asset threshold for applying enhanced prudential standards to large banks?",
     "hard", "lexical_gap", ["D17-P08"], None),
    ("Q13", "What share of Silicon Valley Bank's deposits was not covered by deposit insurance?",
     "medium", "paraphrase_duplicates", ["D03-P02", "D03-P07", "D04-P06"], None),
    ("Q14", "What move in the S&P 500 triggers a 15-minute trading halt on U.S. stock markets?",
     "easy", "definition", ["D01-P09", "D01-P10"], None),
    ("Q15", "Why do volatility spikes force leveraged investors to sell assets even in unrelated markets?",
     "hard", "conceptual_multi_doc", ["D20-P01", "D20-P07", "D20-P09"], None),
    ("Q16", "What happened to Japanese stocks and to the VIX on August 5, 2024?",
     "medium", "numeric_lookup", ["D20-P02", "D20-P04"], None),
    ("Q17", "How did option-implied stock volatility in April 2025 compare with earlier stress episodes?",
     "hard", "comparison_multi_doc", ["D19-P06", "D05-P07"], None),
    ("Q18", "Why did dollar funding markets seize up for foreign banks in March 2020, and how did central banks respond?",
     "hard", "causal_multi_doc", ["D18-P02", "D18-P03", "D18-P04", "D10-P06"], None),
    ("Q19", "How much did the S&P 500 gain on the day the 2025 tariff pause was announced?",
     "easy", "numeric_lookup", ["D05-P13"], None),
    ("Q20", "What inputs and time horizon are used to calculate the VIX?",
     "medium", "definition", ["D06-P04", "D06-P05"], None),
    ("Q21", "How far did the S&P 500 fall in 2022 and what drove the decline?",
     "medium", "multi_passage", ["D02-P01", "D02-P02", "D02-P05"], None),
    ("Q22", "When did the U.S. Treasury yield curve invert before the COVID crash and when did it return to normal?",
     "medium", "multi_doc", ["D01-P01", "D01-P02", "D21-P03"], None),
]

# Experimento 2 (RAG): 10 consultas con respuesta de referencia; supporting_passage_ids subconjunto de los qrels.
REFERENCES = {
    "Q01": ("On 16 March 2020 the CBOE Volatility Index closed at 82.69, the highest closing level ever recorded for the index.",
            ["D01-P12"], ["82.69", "16 March 2020", "highest ever closing"]),
    "Q02": ("SVB had bought large amounts of long-term securities, mostly held to maturity. As the Fed raised interest rates their "
            "market value fell, creating large unrealized losses (over $15 billion on the held-to-maturity book). Because most "
            "deposits were uninsured, withdrawals forced it to sell securities and realize losses, which triggered a bank run.",
            ["D03-P01", "D03-P04", "D17-P02"],
            ["long-term / held-to-maturity securities", "rising rates cut market value (unrealized losses)",
             "mostly uninsured deposits and bank run"]),
    "Q04": ("The FOMC cut the target range for the federal funds rate by 1/2 percentage point to 1 to 1-1/4 percent on March 3, 2020, "
            "and on March 15, 2020 lowered it to 0 to 1/4 percent.",
            ["D09-P01", "D10-P02"], ["1/2 point cut to 1 to 1-1/4 percent on March 3, 2020", "cut to 0 to 1/4 percent on March 15, 2020"]),
    "Q06": ("Yields first fell as investors moved into bonds (10-year at 3.86% on April 4), then reversed sharply: the 10-year rose to about "
            "4.5% by April 9 and the 30-year jumped 54 basis points to 4.92%. The Fed's April 2025 report noted that Treasury prices fell "
            "alongside equities, unlike previous flight-to-safety episodes.",
            ["D05-P09", "D05-P10", "D19-P05"],
            ["yields initially fell", "reversal with yields surging (10-year about 4.5%)", "Treasuries fell alongside equities"]),
    "Q07": ("Loretta J. Mester dissented; she preferred to reduce the target range for the federal funds rate to 1/2 to 3/4 percent.",
            ["D10-P05"], ["Loretta J. Mester", "1/2 to 3/4 percent"]),
    "Q09": ("The FOMC said on March 22, 2023 that the U.S. banking system is sound and resilient. The March 12, 2023 joint statement of the "
            "Treasury, the Fed and the FDIC said it remains resilient on a solid foundation thanks to reforms made after the financial crisis.",
            ["D13-P02", "D16-P06"], ["banking system is sound and resilient", "post-crisis reforms"]),
    "Q10": ("On March 12, 2023 the Fed created the Bank Term Funding Program, which offered loans of up to one year; the pledged securities "
            "(Treasuries, agency debt, mortgage-backed securities) were valued at par instead of market value.",
            ["D04-P08"], ["Bank Term Funding Program", "loans of up to one year", "collateral valued at par"]),
    "Q13": ("About 89 percent of SVB's roughly $172 billion in deposit liabilities exceeded the FDIC insurance limit, that is, were uninsured.",
            ["D03-P02", "D03-P07", "D04-P06"], ["89 percent", "uninsured / above the FDIC limit"]),
    "Q16": ("On 5 August 2024 Japan's TOPIX lost 12% in a single day and the VIX spiked above 60 in off-hours trading, levels not seen since "
            "Covid-19. Markets then stabilised quickly: by 9 August the S&P 500 had recovered all its losses.",
            ["D20-P02", "D20-P04"], ["TOPIX lost 12%", "VIX spiked above 60 / levels not seen since Covid-19", "recovery by 9 August"]),
    "Q19": ("The S&P 500 rose 9.52%, its biggest one-day gain since 2008, after the April 9, 2025 tariff pause.",
            ["D05-P13"], ["9.52%", "biggest one-day gain since 2008"]),
}

# Consultas SIN evidencia en el corpus: el comportamiento correcto es abstenerse.
NO_EVIDENCE = [
    ("N01", "What was the peak value of the VIX during the 2008 financial crisis, and on what date did it occur?",
     "El corpus solo dice que hubo picos intradia mas altos en 2008, sin cifra ni fecha."),
    ("N02", "How did Peruvian stocks and the sol react to the August 2024 carry trade unwind?",
     "Ningun documento trata mercados peruanos."),
    ("N03", "What interest rate decision did the Bank of England announce at its September 2024 meeting?",
     "El corpus solo tiene decisiones de la Reserva Federal para esa fecha."),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main() -> None:
    corpus = [json.loads(l) for l in (DATA / "corpus_parcial.jsonl").read_text(encoding="utf-8").splitlines()]
    pids = {p["passage_id"] for d in corpus for p in d["passages"]}

    qids = [q[0] for q in QUERIES]
    assert len(qids) == len(set(qids)) == 22
    qrels = {}
    for qid, text, diff, cat, rel, _ in QUERIES:
        assert rel and set(rel) <= pids, (qid, set(rel) - pids)
        assert len(set(rel)) == len(rel)
        qrels[qid] = rel
    for qid, (ans, support, facts) in REFERENCES.items():
        assert qid in qrels and set(support) <= set(qrels[qid]), (qid, "soporte fuera de qrels")
        assert len(REFERENCES) == 10 and facts

    write_jsonl(DATA / "queries_parcial.jsonl", [
        {"query_id": q, "query": t, "difficulty": d, "category": c, **({"pc2_origin": o} if o else {}),
         "exp2_rag": q in REFERENCES} for q, t, d, c, _, o in QUERIES])
    (DATA / "qrels_parcial.json").write_text(json.dumps(qrels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_jsonl(DATA / "reference_answers_parcial.jsonl", [
        {"query_id": q, "reference_answer": a, "supporting_passage_ids": s, "required_facts": f}
        for q, (a, s, f) in REFERENCES.items()])
    write_jsonl(DATA / "queries_sin_evidencia.jsonl", [
        {"query_id": q, "query": t, "expected": "abstain", "why_no_evidence": w} for q, t, w in NO_EVIDENCE])

    manifest = {"version": VERSION, "n_documents": len(corpus), "n_passages": len(pids), "n_queries": len(QUERIES),
                "n_queries_no_evidence": len(NO_EVIDENCE), "n_rag_queries": len(REFERENCES),
                "n_relevant_pairs": sum(len(v) for v in qrels.values()),
                "sha256_12": {f: sha(DATA / f) for f in ["corpus_parcial.jsonl", "queries_parcial.jsonl", "qrels_parcial.json",
                                                         "reference_answers_parcial.jsonl", "queries_sin_evidencia.jsonl"]}}
    (DATA / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rel_counts = [len(v) for v in qrels.values()]
    print(json.dumps(manifest, indent=2))
    print("relevantes por consulta: min", min(rel_counts), "media", round(sum(rel_counts) / len(rel_counts), 2), "max", max(rel_counts))


if __name__ == "__main__":
    main()
