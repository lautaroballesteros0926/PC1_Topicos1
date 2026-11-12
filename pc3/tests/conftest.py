import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config  # noqa: E402  (anade parcial/src)
from tools import ToolBox  # noqa: E402


@pytest.fixture(scope="session")
def corpus():
    from common import load_benchmark
    bench = load_benchmark()
    return bench["passages"], {d["doc_id"]: d["title"] for d in bench["documents"]}


@pytest.fixture()
def toolbox(corpus):
    """Caja de herramientas real (precios de la PC1 y corpus del parcial) con un buscador falso y rapido, sin modelos."""
    passages, titles = corpus

    def fake_search(query: str):
        q = query.lower()
        if "silicon" in q or "svb" in q:
            ids = [["D04-P07", "D04-P08"], ["D03-P02"]]
        elif "covid" in q or "vix" in q:
            ids = [["D01-P12"], ["D01-P11", "D18-P08"]]
        else:
            ids = [["D07-P01"]]
        return [{"chunk_id": f"X{i}", "passage_ids": p} for i, p in enumerate(ids)]

    return ToolBox(fake_search, passages, titles)
