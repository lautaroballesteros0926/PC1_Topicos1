"""Construye el corpus del examen parcial (passages TEXTUALES de documentos oficiales/publicos en ingles).

Uso:  python src/build_corpus.py [--raw-dir data/_raw] [--download]

Genera en data/:
  corpus_parcial.jsonl     doc_id, metadatos de procedencia y passages (passage_id, text, locator)
  PROCEDENCIA.md           fuente, URL, licencia, version y hash de cada documento

Regla: los passages son extractos literales (sin traducir ni resumir) de la fuente. Para PDFs largos y
paginas con figuras se recorta con start_at/end_at a una subcadena contigua del parrafo, nunca se reescribe.
Seleccion y recortes quedan en SELECTION; el corpus se fija ANTES de definir consultas y qrels.

--download baja las fuentes a --raw-dir (Wikipedia: texto actual del articulo; el revid usado queda en el
corpus y la URL permanente en PROCEDENCIA.md). Reconstruir con --download exige poppler (pdftotext) y
puede diferir si las paginas cambiaron; el artefacto evaluable es data/corpus_parcial.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RETRIEVED = "2026-10-26"
UA_BOT = {"User-Agent": "cc0f4-parcial/1.0 (student project; GitHub lautaroballesteros0926)"}
UA_BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}

WIKI_LIC = "CC BY-SA 4.0 (Wikipedia)"
FED_LIC = ("Contenido del sitio de la Reserva Federal; extractos literales breves con atribucion "
           "(terminos: https://www.federalreserve.gov/disclaimer.htm)")
BIS_LIC = ("(c) Bank for International Settlements; extractos literales breves con atribucion para uso academico "
           "(terminos: https://www.bis.org/terms_conditions.htm)")

FED_PR = "https://www.federalreserve.gov/newsevents/pressreleases/"
BIS_PUB = "https://www.bis.org/publications/"

# kind: fed_html | pdf | wiki.  sel: indice de parrafo | (indice, start_at, end_at)
DOCS = [
    # --- Wikipedia (CC BY-SA 4.0), episodios y conceptos de la PC1
    dict(doc_id="D01", cache="wiki_covid.json", kind="wiki", title="2020 stock market crash", publisher="Wikipedia",
         url="https://en.wikipedia.org/wiki/2020_stock_market_crash", license=WIKI_LIC, episode="COVID-19 (2020)",
         sel=[1, 2, 5, 8, 10, 18, 21, 30, 33, 35, 40, 41, 45, 52]),
    dict(doc_id="D02", cache="wiki_2022.json", kind="wiki", title="2022 stock market decline", publisher="Wikipedia",
         url="https://en.wikipedia.org/wiki/2022_stock_market_decline", license=WIKI_LIC, episode="Alzas de tasas (2022)",
         sel=[0, 1, 2, 3, 5]),
    dict(doc_id="D03", cache="wiki_svb.json", kind="wiki", title="Collapse of Silicon Valley Bank", publisher="Wikipedia",
         url="https://en.wikipedia.org/wiki/Collapse_of_Silicon_Valley_Bank", license=WIKI_LIC, episode="SVB (2023)",
         sel=[1, 2, 6, 7, 8, 10, 15, 16, 23, 32, 38, 47]),
    dict(doc_id="D04", cache="wiki_bank23.json", kind="wiki", title="2023 United States banking crisis", publisher="Wikipedia",
         url="https://en.wikipedia.org/wiki/2023_United_States_banking_crisis", license=WIKI_LIC, episode="SVB (2023)",
         sel=[0, 8, 10, 16, 20, 21, 31, 35, 41, 48]),
    dict(doc_id="D05", cache="wiki_2025.json", kind="wiki", title="2025 stock market crash", publisher="Wikipedia",
         url="https://en.wikipedia.org/wiki/2025_stock_market_crash", license=WIKI_LIC, episode="Aranceles (2025)",
         sel=[0, 1, 2, 3, 5, 8, 10, 12, 20, 21, 22, 23, 29, 36]),
    dict(doc_id="D06", cache="wiki_vix.json", kind="wiki", title="VIX", publisher="Wikipedia",
         url="https://en.wikipedia.org/wiki/VIX", license=WIKI_LIC, episode="Indicador (volatilidad)",
         sel=[0, 1, 3, 6, 8, 18, 19, 21, 23, 27]),
    dict(doc_id="D07", cache="wiki_drawdown.json", kind="wiki", title="Drawdown (economics)", publisher="Wikipedia",
         url="https://en.wikipedia.org/wiki/Drawdown_(economics)", license=WIKI_LIC, episode="Indicador (drawdown)",
         sel=[0, 2, 3, 4, 7]),
    dict(doc_id="D08", cache="wiki_carry.json", kind="wiki", title="Carry (investment)", publisher="Wikipedia",
         url="https://en.wikipedia.org/wiki/Carry_(investment)", license=WIKI_LIC, episode="Carry trade (2024)",
         sel=[1, 2, 3, 4, 5, 9]),
    # --- Comunicados de la Reserva Federal
    dict(doc_id="D09", cache="fomc20200303.bin", kind="fed_html", title="FOMC statement, March 3, 2020",
         publisher="Federal Reserve Board", url=FED_PR + "monetary20200303a.htm", license=FED_LIC, episode="COVID-19 (2020)",
         sel=[0]),
    dict(doc_id="D10", cache="fomc20200315.bin", kind="fed_html", title="FOMC statement, March 15, 2020",
         publisher="Federal Reserve Board", url=FED_PR + "monetary20200315a.htm", license=FED_LIC, episode="COVID-19 (2020)",
         sel=[0, 1, 2, 3, 4, 5]),
    dict(doc_id="D11", cache="fomc20200323.bin", kind="fed_html", title="FOMC statement, March 23, 2020",
         publisher="Federal Reserve Board", url=FED_PR + "monetary20200323a.htm", license=FED_LIC, episode="COVID-19 (2020)",
         sel=[0, 1, 5, 6]),
    dict(doc_id="D12", cache="fomc20220615.bin", kind="fed_html", title="FOMC statement, June 15, 2022",
         publisher="Federal Reserve Board", url=FED_PR + "monetary20220615a.htm", license=FED_LIC, episode="Alzas de tasas (2022)",
         sel=[0, 1, 2, 3, 4]),
    dict(doc_id="D13", cache="fomc20230322.bin", kind="fed_html", title="FOMC statement, March 22, 2023",
         publisher="Federal Reserve Board", url=FED_PR + "monetary20230322a.htm", license=FED_LIC, episode="SVB (2023)",
         sel=[0, 1, 2, 3, 4]),
    dict(doc_id="D14", cache="fomc20240918.bin", kind="fed_html", title="FOMC statement, September 18, 2024",
         publisher="Federal Reserve Board", url=FED_PR + "monetary20240918a.htm", license=FED_LIC, episode="Carry trade (2024)",
         sel=[0, 1, 2, 3, 4]),
    dict(doc_id="D15", cache="fomc20250507.bin", kind="fed_html", title="FOMC statement, May 7, 2025",
         publisher="Federal Reserve Board", url=FED_PR + "monetary20250507a.htm", license=FED_LIC, episode="Aranceles (2025)",
         sel=[0, 1, 2, 3, 4]),
    dict(doc_id="D16", cache="joint20230312.bin", kind="fed_html",
         title="Joint Statement by Treasury, Federal Reserve, and FDIC (March 12, 2023)",
         publisher="Federal Reserve Board", url=FED_PR + "monetary20230312b.htm", license=FED_LIC, episode="SVB (2023)",
         sel=[1, 2, 3, 4, 5, 6]),
    # --- Informes (PDF)
    dict(doc_id="D17", cache="barr_svb.pdf", kind="pdf",
         title="Review of the Federal Reserve's Supervision and Regulation of Silicon Valley Bank (Barr, April 28, 2023)",
         publisher="Federal Reserve Board", url="https://www.federalreserve.gov/publications/files/svb-review-20230428.pdf",
         license=FED_LIC, episode="SVB (2023)",
         sel=[12, 14, 20, (21, None, "in August 2022."), 22, 23, 25, 32, (40, None, "based in the United States.29"), 55]),
    dict(doc_id="D18", cache="fsr202005.pdf", kind="pdf", title="Financial Stability Report, May 2020",
         publisher="Federal Reserve Board",
         url="https://www.federalreserve.gov/publications/files/financial-stability-report-20200515.pdf",
         license=FED_LIC, episode="COVID-19 (2020)",
         sel=[12, 20, (21, None, "harming U.S. households and businesses."), (22, None, "would begin daily auctions."),
              (28, None, "average bid-ask spreads widened dramatically."),
              (33, None, "rose well above its historical median (figure 1-7).9"),
              (38, None, "fallen in response to the economic deterioration."), 40]),
    dict(doc_id="D19", cache="fsr202504.pdf", kind="pdf", title="Financial Stability Report, April 2025",
         publisher="Federal Reserve Board",
         url="https://www.federalreserve.gov/publications/files/financial-stability-report-20250425.pdf",
         license=FED_LIC, episode="Aranceles (2025)",
         sel=[9, 10, 11, 12, (17, "A model-based estimate", "(figure 1.3)."),
              (18, None, "update more slowly than market prices."), 22]),
    dict(doc_id="D20", cache="bisbull90.pdf", kind="pdf",
         title="The market turbulence and carry trade unwind of August 2024 (BIS Bulletin No 90)",
         publisher="Bank for International Settlements",
         url=BIS_PUB + "bulletin-90-market-turbulence-and-carry-trade-unwind-august-2024.pdf", license=BIS_LIC,
         episode="Carry trade (2024)",
         sel=[(1, "Key takeaways", "deserve continued attention by policymakers."),
              (1, "In the first days of August 2024", "were the hardest hit.1"),
              (1, "While the most intense market moves", "let alone a looming recession."),
              (3, "Thus, the news acted as a catalyst", None),
              (5, "Large FX carry trade positions", "unwound during the event (Graph 3.A)."),
              (7, "BIS Global Liquidity Indicators further show", "commenced in Japan (Graph 4.A)."),
              (10, "Overall, there were many signs of spillovers", "Clancy (2024))."),
              (11, "Underlying vulnerabilities linger", "to restore calm."),
              (11, "That said, the factors behind", "could be kindled.")]),
    dict(doc_id="D21", cache="qr-202003.pdf", kind="pdf", title="BIS Quarterly Review, March 2020",
         publisher="Bank for International Settlements", url=BIS_PUB + "qr-202003.pdf", license=BIS_LIC,
         episode="COVID-19 (2020)",
         sel=[(17, "First in mid-January, and then with even greater intensity",
               "cautious mood by late February the end of the review period."),
              21,
              (24, "In fixed income markets", "negative territory.")]),
]
WIKI_TITLES = {"wiki_covid.json": "2020 stock market crash", "wiki_2022.json": "2022 stock market decline",
               "wiki_svb.json": "Collapse of Silicon Valley Bank", "wiki_bank23.json": "2023 United States banking crisis",
               "wiki_2025.json": "2025 stock market crash", "wiki_vix.json": "VIX",
               "wiki_drawdown.json": "Drawdown (economics)", "wiki_carry.json": "Carry (investment)"}


def wc(t: str) -> int:
    return len(t.split())


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def download(doc: dict, raw: Path) -> None:
    dest = raw / doc["cache"]
    if dest.exists():
        return
    if doc["kind"] == "wiki":
        r = requests.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "format": "json", "redirects": 1, "titles": WIKI_TITLES[doc["cache"]],
            "prop": "extracts|revisions", "explaintext": 1, "rvprop": "ids|timestamp", "rvslots": "main"},
            headers=UA_BOT, timeout=60).json()
        dest.write_text(json.dumps(list(r["query"]["pages"].values())[0], ensure_ascii=False), encoding="utf-8")
        time.sleep(1.5)
    else:
        r = requests.get(doc["url"], headers=UA_BROWSER, timeout=90)
        r.raise_for_status()
        dest.write_bytes(r.content)


def pdf_paragraphs(path: Path) -> list[dict]:
    txt = subprocess.run(["pdftotext", str(path), "-"], capture_output=True, check=True).stdout.decode("utf-8", "ignore")
    out = []
    for pg, page in enumerate(txt.split("\f"), 1):
        for block in re.split(r"\n\s*\n", page):
            t = re.sub(r"\s+", " ", " ".join(l.strip() for l in block.split("\n") if l.strip()))
            if wc(t) >= 35 and not re.match(r"^[\d\W]+$", t):
                out.append({"text": t, "locator": f"p. {pg}"})
    return out


def fed_paragraphs(path: Path) -> list[dict]:
    soup = BeautifulSoup(path.read_bytes().decode("utf-8", "ignore"), "lxml")
    art = soup.select_one("#article")
    ps = [re.sub(r"\s+", " ", p.get_text(" ", strip=True)) for p in art.find_all("p")]
    return [{"text": t, "locator": "texto del comunicado"} for t in ps if wc(t) >= 12]


def wiki_paragraphs(page: dict) -> list[dict]:
    sec, out = "Lead", []
    for line in page["extract"].split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(=+)\s*(.+?)\s*=+$", line)
        if m:
            sec = m.group(2)
            continue
        if wc(line) >= 25:
            out.append({"text": line, "locator": f"seccion: {sec}"})
    return out


def cut(text: str, start_at: str | None, end_at: str | None) -> str:
    a = 0
    if start_at is not None:
        a = text.index(start_at)
    b = len(text)
    if end_at is not None:
        b = text.index(end_at, a) + len(end_at)
    return text[a:b].strip()


def build(raw: Path, out: Path, do_download: bool) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    docs_out = []
    for d in DOCS:
        if do_download:
            download(d, raw)
        path = raw / d["cache"]
        meta = {}
        if d["kind"] == "wiki":
            page = json.loads(path.read_text(encoding="utf-8"))
            paras = wiki_paragraphs(page)
            rev = page["revisions"][0]
            meta = {"revid": rev["revid"], "revision_timestamp": rev["timestamp"],
                    "permalink": f"https://en.wikipedia.org/w/index.php?title={page['title'].replace(' ', '_')}&oldid={rev['revid']}",
                    "source_sha256": sha256_bytes(page["extract"].encode("utf-8"))}
        elif d["kind"] == "fed_html":
            paras = fed_paragraphs(path)
            meta = {"source_sha256": sha256_bytes(path.read_bytes())}
        else:
            paras = pdf_paragraphs(path)
            meta = {"source_sha256": sha256_bytes(path.read_bytes())}
        passages = []
        for n, s in enumerate(d["sel"], 1):
            idx, a, b = (s, None, None) if isinstance(s, int) else s
            para = paras[idx]
            text = cut(para["text"], a, b)
            assert text and text in para["text"], (d["doc_id"], idx)
            passages.append({"passage_id": f"{d['doc_id']}-P{n:02d}", "text": text, "locator": para["locator"],
                             "words": wc(text)})
        docs_out.append({"doc_id": d["doc_id"], "title": d["title"], "publisher": d["publisher"], "url": d["url"],
                         "license": d["license"], "episode": d["episode"], "retrieved": RETRIEVED, "kind": d["kind"],
                         **meta, "passages": passages})
    with open(out / "corpus_parcial.jsonl", "w", encoding="utf-8") as f:
        for doc in docs_out:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    write_provenance(docs_out, out)
    n = sum(len(d["passages"]) for d in docs_out)
    words = [p["words"] for d in docs_out for p in d["passages"]]
    print(f"{len(docs_out)} documentos, {n} passages, {sum(words)} palabras "
          f"(min {min(words)}, media {sum(words)/len(words):.0f}, max {max(words)})")


def write_provenance(docs: list[dict], out: Path) -> None:
    lines = [
        "# Procedencia del corpus del examen parcial",
        "",
        f"Fecha de consulta de todas las fuentes: {RETRIEVED}. Los passages son **extractos literales** (sin traducir ni resumir) "
        "de las fuentes; algunos se recortan a una subcadena contigua del parrafo para excluir pies de figura o notas (ver "
        "`src/build_corpus.py`, campo `sel`). Las consultas y los qrels se redactaron DESPUES de fijar este corpus y ANTES de "
        "ejecutar cualquier sistema.",
        "",
        "| Doc | Titulo | Editor | Episodio | Passages | URL | Version / hash | Licencia / uso |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in docs:
        version = d.get("permalink") or d["url"]
        if "revid" in d:
            version = f"[revid {d['revid']}]({d['permalink']}) ({d['revision_timestamp']})"
        h = d["source_sha256"][:12]
        lines.append(f"| {d['doc_id']} | {d['title']} | {d['publisher']} | {d['episode']} | {len(d['passages'])} | "
                     f"{d['url']} | {version}; sha256 `{h}` | {d['license']} |")
    lines += [
        "",
        "Notas:",
        "- El hash de Wikipedia es del texto plano del articulo en la revision indicada; el de los demas documentos es del archivo "
        "descargado (HTML o PDF). Los archivos originales no se versionan (los PDFs pesan varios MB); se pueden volver a bajar con "
        "`python src/build_corpus.py --download`.",
        "- Wikipedia cambia: la URL permanente (`oldid`) fija la revision citada.",
        "- Limitaciones: Wikipedia es fuente secundaria; los PDFs se extraen con `pdftotext` y pueden conservar marcas de nota al pie "
        "(p. ej. `2022.11`) dentro del texto; no se corrigieron para no alterar el extracto.",
    ]
    (out / "PROCEDENCIA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "_raw"))
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    build(Path(a.raw_dir), ROOT / "data", a.download)
