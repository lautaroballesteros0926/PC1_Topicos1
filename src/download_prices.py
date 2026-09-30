"""Descarga UNA vez precios diarios ajustados de Yahoo Finance y los cachea en data/raw/."""
from pathlib import Path
import yfinance as yf

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
TICKERS = ["SPY", "GLD", "KO", "IWM"]

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for t in TICKERS:
        d = yf.download(t, start="2016-06-01", end="2025-09-01", auto_adjust=True, progress=False)
        d.columns = d.columns.get_level_values(0)
        d[["Close"]].to_csv(RAW / f"{t}.csv")
        print(t, len(d), d.index[0].date(), d.index[-1].date())
