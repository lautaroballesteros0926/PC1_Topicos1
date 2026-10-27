"""Indicadores de mercado y regla de severidad de la PC1 (copiados de `src/market_features.py` de la PC1).

La regla es determinista: no depende de un LLM. En el examen sirve de logica de dominio de la herramienta
`risk.compute_signal` (Experimento 3). Cualquier cambio aqui rompe la reproduccion de los 8 casos gold de la PC1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VOL_BANDS = (15.0, 25.0)      # volatilidad anualizada 20d (%)
DD_BANDS = (-5.0, -10.0)      # caida desde el maximo de 20d (%)
RET5_BANDS = (-3.0, -6.0)     # retorno a 5 dias (%)


def compute_features(close: pd.Series) -> pd.DataFrame:
    ret = close.pct_change()
    return pd.DataFrame({
        "close": close,
        "ret5": close.pct_change(5) * 100,
        "vol20": ret.rolling(20).std() * np.sqrt(252) * 100,
        "dd20": (close / close.rolling(20).max() - 1) * 100,
    })


def points_high_is_bad(value: float, bands: tuple[float, float]) -> int:
    return 0 if value < bands[0] else 1 if value < bands[1] else 2


def points_low_is_bad(value: float, bands: tuple[float, float]) -> int:
    return 0 if value > bands[0] else 1 if value > bands[1] else 2


def score_parts(vol20: float, dd20: float, ret5: float) -> dict[str, int]:
    return {"vol20": points_high_is_bad(vol20, VOL_BANDS), "dd20": points_low_is_bad(dd20, DD_BANDS),
            "ret5": points_low_is_bad(ret5, RET5_BANDS)}


def label_from_score(score: int) -> str:
    return "low" if score <= 1 else "medium" if score <= 3 else "high"
