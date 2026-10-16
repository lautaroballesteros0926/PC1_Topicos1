"""Construye el benchmark de la PC2 (corpus, consultas, qrels y schema de la consulta estructurada).

Uso:  python src/build_dataset.py

Genera en data/:
  corpus_pc2.jsonl         8 documentos, cada uno dividido en passages atomicos con passage_id
  queries_pc2.jsonl        6 necesidades de informacion (Q0 = consulta original del estudiante)
  qrels_pc2.json           query_id -> [passage_id relevantes]  (relevancia sobre passages, no chunks)
  schema_consulta.json     JSON Schema de la consulta estructurada que produce el LLM
  PROCEDENCIA.md           fuente, URL, licencia y fecha de consulta de cada documento

Regla (igual que Semana 4): corpus, queries y qrels se redactaron ANTES de ejecutar cualquier
experimento y no se modifican despues de ver resultados. Si cambian, es una nueva version del benchmark.

Los passages son resumenes propios en espanol de hechos publicados en las fuentes indicadas:
las cifras (fechas, puntos, porcentajes) se tomaron de la fuente; la redaccion es mia.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RETRIEVED = "2026-10-13"

WIKI = "CC BY-SA 4.0 (Wikipedia); passages = resumen/traduccion propia"
FED = "Dominio publico (gobierno de EE. UU.); passages = resumen/traduccion propia"
AP = "Articulo con copyright; solo se resumen hechos con redaccion propia"

DOCS = [
    {
        "doc_id": "D01",
        "title": "Crisis bursátil por COVID-19 (febrero-abril de 2020)",
        "source": "Wikipedia, 2020 stock market crash",
        "url": "https://en.wikipedia.org/wiki/2020_stock_market_crash",
        "license": WIKI,
        "pc1_case": "H1 (SPY, 2020-03-16)",
        "passages": [
            "A fines de febrero de 2020 las bolsas mundiales cayeron con fuerza por la incertidumbre de la pandemia de COVID-19. Entre el 24 y el 28 de febrero los principales índices de Wall Street retrocedieron al menos 10 % desde su máximo histórico, la corrección más rápida registrada, en solo seis días.",
            "El 27 de febrero el Dow Jones perdió 1.191 puntos, su mayor caída diaria en puntos desde la crisis financiera de 2008. Tras el fracaso de las negociaciones de la OPEP el 6 de marzo, Arabia Saudita y Rusia anunciaron aumentos de producción y el petróleo cayó cerca de 25 %; el 9 de marzo bajó otro 22 %.",
            "El lunes 9 de marzo el Dow Jones cayó 2.013,76 puntos (−7,79 %) y cerró en 23.851,02, mientras el S&P 500 perdió 7,6 %. Se activaron las pausas automáticas de negociación (circuit breakers) al inicio de la sesión.",
            "El 11 de marzo el Dow Jones entró en mercado bajista y puso fin a once años de mercado alcista. El jueves 12 de marzo cayó 2.352,60 puntos (−9,99 %) hasta 21.200,62 y el S&P 500 retrocedió 9,5 %.",
            "El lunes 16 de marzo el Dow Jones perdió 2.997,10 puntos (−12,93 %) y cerró en 20.188,52; el S&P 500 y el Nasdaq cayeron entre 12 % y 13 %. Ese día el índice de volatilidad VIX cerró en 82,69, el nivel de cierre más alto registrado.",
            "La Reserva Federal recortó la tasa 50 puntos básicos el 3 de marzo y a mediados de mes la llevó a un rango de 0 a 0,25 %, redujo los encajes a cero y lanzó un programa de compras de activos de 700.000 millones de dólares. El 12 de marzo anunció operaciones de recompra por 1,5 billones.",
            "El 13 de marzo los tres índices principales subieron más de 9 %, su mayor repunte desde 2008. Los mercados globales volvieron a territorio alcista en abril, pero los índices estadounidenses no recuperaron sus niveles de enero de 2020 hasta noviembre de ese año.",
        ],
    },
    {
        "doc_id": "D02",
        "title": "Quiebra de Silicon Valley Bank (marzo de 2023)",
        "source": "Wikipedia, Collapse of Silicon Valley Bank",
        "url": "https://en.wikipedia.org/wiki/Collapse_of_Silicon_Valley_Bank",
        "license": WIKI,
        "pc1_case": "M2 (SPY, 2023-03-13)",
        "passages": [
            "Silicon Valley Bank (SVB) quebró el 10 de marzo de 2023 tras una corrida bancaria. Fue la tercera mayor quiebra bancaria de la historia de Estados Unidos y la mayor desde la crisis financiera de 2008.",
            "A fines de 2022 SVB mantenía una cartera de bonos de 117.000 millones de dólares: 91.300 millones mantenidos a vencimiento y 26.000 millones disponibles para la venta. Las alzas de tasas de la Reserva Federal para frenar la inflación redujeron el valor de mercado de esos bonos.",
            "Los valores mantenidos a vencimiento acumulaban pérdidas no realizadas superiores a 15.000 millones de dólares. A comienzos de 2023 el banco vendió todos sus valores disponibles para la venta y realizó una pérdida de 1.800 millones.",
            "El 8 de marzo SVB anunció que había vendido más de 21.000 millones de dólares en inversiones y que haría una venta de acciones de emergencia por 2.250 millones. Cerca del 89 % de sus depósitos superaba el límite asegurado por la FDIC, de modo que la mayoría de los clientes no tenía garantía.",
            "Al cierre del 9 de marzo los clientes habían retirado 42.000 millones de dólares, dejando al banco con un saldo de caja negativo de unos 958 millones. Se esperaban retiros por otros 100.000 millones el viernes 10 de marzo.",
            "El 10 de marzo las autoridades de California intervinieron SVB y la pusieron bajo la FDIC. El 12 de marzo se cerró Signature Bank y el Tesoro autorizó a la FDIC a garantizar los depósitos no asegurados de ambos bancos.",
            "La Reserva Federal creó el Bank Term Funding Program para dar liquidez a otros bancos. La capitalización bursátil de los bancos estadounidenses perdió 100.000 millones de dólares en dos días y la de los bancos europeos 50.000 millones; las acciones de First Republic, Western Alliance y PacWest se desplomaron el 13 de marzo.",
        ],
    },
    {
        "doc_id": "D03",
        "title": "Venta global de agosto de 2024 y carry trade del yen",
        "source": "Associated Press vía Christian Science Monitor (6-ago-2024) + Wikipedia, Carry trade",
        "url": "https://csmonitor.com/World/2024/0806/stock-market-recession-carry-trade ; https://en.wikipedia.org/wiki/Carry_trade",
        "license": AP + " / " + WIKI,
        "pc1_case": "M1 (SPY, 2024-08-05)",
        "passages": [
            "El 5 de agosto de 2024 las bolsas mundiales sufrieron una venta intensa. El Nikkei 225 de Japón cayó 12,4 % en una sola jornada, su mayor caída porcentual desde 1987, y los mercados de Europa y Estados Unidos también registraron pérdidas fuertes.",
            "Un carry trade consiste en endeudarse en una moneda de baja tasa para invertir en activos de mayor rendimiento. Durante años se pidió prestado en yenes para comprar acciones y bonos del Tesoro de Estados Unidos. La operación solo es rentable mientras nada cambie en contra del diferencial de tasas.",
            "A fines de julio el Banco de Japón subió su tasa principal a 0,25 %, frente a un rango de 5 % a 5,25 % de la Reserva Federal. El yen se apreció con fuerza, lo que encareció la deuda en yenes de los inversionistas apalancados y los obligó a cerrar posiciones y vender activos.",
            "El 2 de agosto un informe de empleo de Estados Unidos más débil de lo esperado, publicado poco después de que la Reserva Federal decidiera no recortar tasas, avivó el temor a una recesión. Ese temor se sumó a la presión de cierre de posiciones de carry trade.",
            "El 5 de agosto el S&P 500 cayó cerca de 3 % y el índice de volatilidad VIX se disparó a niveles que no se veían desde los primeros días de la pandemia. La baja liquidez típica del verano amplificó el movimiento.",
            "Los mercados recuperaron gran parte de las pérdidas el 6 de agosto; el Nikkei subió 10,2 % ese día. El episodio se interpretó como un ajuste técnico de posiciones apalancadas y no como un deterioro inmediato de los fundamentos económicos.",
        ],
    },
    {
        "doc_id": "D04",
        "title": "Caída bursátil por aranceles (abril de 2025)",
        "source": "Wikipedia, 2025 stock market crash",
        "url": "https://en.wikipedia.org/wiki/2025_stock_market_crash",
        "license": WIKI,
        "pc1_case": "H3 (SPY, 2025-04-08)",
        "passages": [
            "El 2 de abril de 2025, tras el anuncio de aranceles del llamado «Día de la Liberación» por parte del presidente de Estados Unidos, los futuros del S&P 500 cayeron 3,9 %, los del Nasdaq-100 4,7 % y los del Dow Jones 2,7 %.",
            "El 3 de abril el S&P 500 cerró con una baja de 4,84 % en 5.074,08 puntos y el Dow Jones perdió 1.679,39 puntos (3,98 %). El índice Russell 2000 de pequeñas empresas cayó 6,59 % y entró en mercado bajista.",
            "El 4 de abril el Dow Jones cayó 2.231,07 puntos (5,50 %), el S&P 500 5,97 % y el Nasdaq Composite 5,82 %, con lo que entró en territorio bajista. En los dos días el S&P 500 perdió cerca de 10 % y se estimó una pérdida de unos 6,6 billones de dólares, la mayor en dos jornadas.",
            "El VIX subió 15 puntos en esos dos días y cerró en 45,31, su cierre más alto desde la crisis de 2020. El 7 y el 8 de abril los índices siguieron oscilando: el S&P 500 bajó 0,23 % y 1,57 % respectivamente.",
            "El rendimiento del bono del Tesoro a 10 años cayó a 3,86 % el 4 de abril, su nivel más bajo desde octubre de 2024, por la compra de refugio. Luego hubo ventas de bonos y la tasa subió a cerca de 4,5 % en la mañana del 9 de abril; la de 30 años subió 54 puntos básicos a 4,92 %, su mayor alza en tres días desde 1982.",
            "El 9 de abril el gobierno anunció una pausa de 90 días con un arancel reducido de 10 % para todos los países salvo China. Ese día el S&P 500 subió 9,52 %, su mayor ganancia diaria desde 2008, y el Nasdaq 12,16 %. El 10 de abril los mercados volvieron a caer por la incertidumbre.",
        ],
    },
    {
        "doc_id": "D05",
        "title": "El índice de volatilidad VIX",
        "source": "Wikipedia, VIX",
        "url": "https://en.wikipedia.org/wiki/VIX",
        "license": WIKI,
        "pc1_case": "indicador de volatilidad (vol20 de la PC1 es volatilidad realizada)",
        "passages": [
            "El VIX es el índice de volatilidad del Chicago Board Options Exchange (CBOE) y refleja la volatilidad esperada del S&P 500 a corto plazo a partir de los precios de las opciones sobre el índice. Se le conoce como el «índice del miedo».",
            "Se calcula como un promedio ponderado de opciones call y put fuera del dinero del S&P 500 con unos 30 días de vencimiento. El resultado es la raíz cuadrada de la varianza esperada a 30 días y se expresa como una desviación estándar anualizada.",
            "Menachem Brenner y Dan Galai propusieron los índices de volatilidad en 1989. El CBOE empezó a publicar en tiempo real su índice original, basado en el S&P 100, el 19 de enero de 1993, y en 2003 lo cambió al S&P 500.",
            "Los futuros sobre el VIX comenzaron a negociarse el 26 de marzo de 2004 y las opciones el 24 de febrero de 2006. El índice en sí no puede comprarse directamente.",
            "El máximo intradía histórico fue 89,53 el 24 de octubre de 2008. El cierre más alto de 2008 fue 80,74 el 21 de noviembre, y el 16 de marzo de 2020 el VIX cerró en 82,69, el valor de cierre más alto desde su creación.",
            "Algunos describen el VIX como una medida de los precios actuales de las opciones más que como un pronóstico, y hay críticos que sostienen que su poder predictivo es similar al de la volatilidad pasada simple. Muchos ETF vinculados al VIX se correlacionan mal con el índice, sobre todo cuando este se mueve rápido.",
        ],
    },
    {
        "doc_id": "D06",
        "title": "Drawdown y drawdown máximo",
        "source": "Wikipedia, Drawdown (economics)",
        "url": "https://en.wikipedia.org/wiki/Drawdown_(economics)",
        "license": WIKI,
        "pc1_case": "indicador dd20 (caída desde el máximo de 20 días)",
        "passages": [
            "El drawdown mide la caída desde un máximo histórico de alguna variable, normalmente la ganancia acumulada o el valor total de una estrategia. Es la distancia entre el máximo acumulado hasta el momento y el valor actual.",
            "El drawdown máximo (MDD) es la mayor caída observada en el historial hasta un momento T. Se calcula siguiendo el valor patrimonial más alto alcanzado y midiendo en cada período 100 × (máximo − valor) / máximo; el MDD es el mayor de esos valores.",
            "El término tiene dos sentidos: la magnitud, que es la pérdida de máximo a mínimo, y la duración, que es el tiempo entre máximos consecutivos. La mayor duración de un drawdown puede no coincidir con la mayor pérdida.",
            "Las razones de Calmar, Sterling y Burke usan el drawdown máximo como medida de riesgo: modifican la razón de Sharpe reemplazando la desviación estándar por una función del drawdown.",
            "Para un movimiento browniano con deriva, el drawdown máximo esperado crece de forma logarítmica si la deriva es positiva, con la raíz del tiempo si es cero y linealmente si es negativa. Las duraciones largas se asocian con la recuperación de grandes pérdidas.",
            "Minimizar el drawdown es difícil porque el problema no es convexo. Chekhlov, Uryasev y Zabarankin lo reformularon como un programa lineal mediante el drawdown condicional en riesgo, cuyos casos límite son el drawdown promedio y el máximo.",
        ],
    },
    {
        "doc_id": "D07",
        "title": "Comunicado del FOMC del 15 de junio de 2022",
        "source": "Federal Reserve Board, FOMC statement, 15-jun-2022",
        "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20220615a.htm",
        "license": FED,
        "pc1_case": "H2 (IWM, 2022-06-13)",
        "passages": [
            "El 15 de junio de 2022 el Comité Federal de Mercado Abierto (FOMC) elevó el rango objetivo de la tasa de fondos federales a 1,5 %–1,75 %, un aumento de 0,75 puntos porcentuales respecto del rango anterior.",
            "El Comité indicó que prevé que los aumentos continuos del rango objetivo serán apropiados. Afirmó estar fuertemente comprometido a devolver la inflación a su objetivo de 2 % y altamente atento a los riesgos inflacionarios.",
            "La inflación sigue elevada, reflejando desequilibrios de oferta y demanda relacionados con la pandemia, precios de la energía más altos y presiones de precios más amplias.",
            "La actividad económica se recuperó tras una caída en el primer trimestre, las ganancias de empleo fueron sólidas y el desempleo se mantuvo bajo.",
            "El Comité continuará reduciendo sus tenencias de valores del Tesoro, deuda de agencias y valores respaldados por hipotecas de agencias, según el plan anunciado en mayo.",
            "La invasión de Rusia a Ucrania y las cuarentenas por COVID-19 en China generan presión alcista sobre la inflación, afectan el crecimiento global y probablemente agravarán las disrupciones de las cadenas de suministro. Esther L. George votó en contra porque prefería un aumento de 0,5 puntos.",
        ],
    },
    {
        "doc_id": "D08",
        "title": "Comunicado del FOMC del 15 de marzo de 2020",
        "source": "Federal Reserve Board, FOMC statement, 15-mar-2020",
        "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20200315a.htm",
        "license": FED,
        "pc1_case": "H1 (SPY, 2020-03-16): respuesta de politica monetaria",
        "passages": [
            "El 15 de marzo de 2020 el FOMC redujo el rango objetivo de la tasa de fondos federales a 0 %–0,25 %. Indicó que mantendrá ese rango hasta estar seguro de que la economía ha resistido los acontecimientos recientes.",
            "El Comité afirmó que los efectos del coronavirus pesarán sobre la actividad económica en el corto plazo y que plantean riesgos para las perspectivas.",
            "Para sostener el flujo de crédito, el Comité comprará al menos 500.000 millones de dólares en valores del Tesoro y al menos 200.000 millones en valores respaldados por hipotecas de agencias en los próximos meses.",
            "Reinvertirá todos los pagos de principal de la deuda de agencias y de los valores hipotecarios en valores hipotecarios de agencias. La mesa de mercado abierto amplió recientemente sus operaciones de recompra a un día y a plazo.",
            "La Reserva Federal anunció además medidas sobre la ventanilla de descuento, el crédito intradía, los colchones de capital y liquidez de los bancos, los encajes y las líneas de intercambio de dólares.",
            "Loretta J. Mester votó en contra de la decisión porque prefería un rango objetivo de 0,5 % a 0,75 %.",
        ],
    },
]

# Necesidades de informacion. "need" es Q0: la consulta original, tal como la formularia el estudiante.
# difficulty se fijo antes de ejecutar.
QUERIES = [
    ("Q01", "¿Qué pasó con la volatilidad esperada del mercado en el peor día de la crisis del COVID en marzo de 2020?", "easy"),
    ("Q02", "¿Por qué el banco de Silicon Valley no pudo resistir si tenía una cartera de bonos tan grande?", "medium"),
    ("Q03", "Quiero entender qué significa pedir dinero prestado en una moneda barata para invertir y por qué eso puede provocar ventas masivas de golpe.", "hard"),
    ("Q04", "¿Qué hizo la Reserva Federal con las tasas de interés para responder al pánico por el coronavirus?", "ambiguous"),
    ("Q05", "¿Cómo se mide cuánto ha caído un activo desde su máximo reciente?", "easy"),
    ("Q06", "¿Cómo reaccionaron los bonos del Tesoro cuando se anunciaron los aranceles de abril de 2025?", "medium"),
]

# Relevancia sobre passage_id (no sobre chunk_id): estable aunque cambie la segmentacion.
QRELS = {
    "Q01": ["D01-P05", "D05-P05"],
    "Q02": ["D02-P02", "D02-P03"],
    "Q03": ["D03-P02", "D03-P03"],
    "Q04": ["D08-P01", "D01-P06"],
    "Q05": ["D06-P01", "D06-P02"],
    "Q06": ["D04-P05"],
}

# Schema de la consulta estructurada (salida del LLM).
INTENTS = ["risk_event", "risk_indicator", "policy_response", "mechanism", "other"]
SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Consulta estructurada de recuperacion",
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 8, "maxLength": 200},
        "intent": {"type": "string", "enum": INTENTS},
        "abstain": {"type": "boolean"},
    },
    "required": ["query", "intent", "abstain"],
    "additionalProperties": False,
}


def main() -> None:
    DATA.mkdir(exist_ok=True)

    with (DATA / "corpus_pc2.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for doc in DOCS:
            row = {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "source": doc["source"],
                "url": doc["url"],
                "license": doc["license"],
                "retrieved": RETRIEVED,
                "passages": [
                    {"passage_id": f"{doc['doc_id']}-P{i:02d}", "text": text}
                    for i, text in enumerate(doc["passages"], start=1)
                ],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (DATA / "queries_pc2.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for query_id, need, difficulty in QUERIES:
            handle.write(json.dumps(
                {"query_id": query_id, "query": need, "difficulty": difficulty},
                ensure_ascii=False) + "\n")

    (DATA / "qrels_pc2.json").write_text(
        json.dumps(QRELS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (DATA / "schema_consulta.json").write_text(
        json.dumps(SCHEMA, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Procedencia del corpus de la PC2",
        "",
        f"Fecha de consulta de todas las fuentes: {RETRIEVED}. Los passages son resumenes propios en espanol; ",
        "las cifras provienen de la fuente. Las consultas y los qrels los redacte yo antes de ejecutar los experimentos.",
        "",
        "| Doc | Titulo | Fuente | URL | Licencia / uso | Caso PC1 relacionado |",
        "|---|---|---|---|---|---|",
    ]
    for doc in DOCS:
        lines.append(
            f"| {doc['doc_id']} | {doc['title']} | {doc['source']} | {doc['url']} | {doc['license']} | {doc['pc1_case']} |")
    lines += [
        "",
        "Nota D03: las cifras del Nikkei (-12,4 % el 5-ago-2024; +10,2 % el 6-ago) vienen del articulo de AP. "
        "El movimiento aproximado del S&P 500 (~3 %) y el salto del VIX se cotejaron con resumenes de prensa secundaria, "
        "no con datos primarios de S&P/CBOE; por eso el passage los describe de forma aproximada.",
    ]
    (DATA / "PROCEDENCIA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    n_passages = sum(len(d["passages"]) for d in DOCS)
    n_words = sum(len(p.split()) for d in DOCS for p in d["passages"])
    print(f"documentos={len(DOCS)} passages={n_passages} palabras={n_words} consultas={len(QUERIES)}")
    for d in DOCS:
        words = [len(p.split()) for p in d["passages"]]
        print(d["doc_id"], "passages", len(words), "palabras", sum(words), "min/max", min(words), max(words))


if __name__ == "__main__":
    main()
