"""Fetch, freeze, and validate adjusted daily prices for the six-stock study."""

from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from .validate_data import validate_rows


TECH = ("NVDA", "AMD", "AVGO")
INFRA = ("VRT", "ETN", "PWR")
SYMBOLS = TECH + INFRA
START = "2020-02-10"


def download_snapshot(data_dir: Path, end: str | None = None) -> dict:
    """Save both the untouched provider close matrix and its complete-case sample."""
    data_dir.mkdir(parents=True, exist_ok=True)
    end = end or (date.today() + timedelta(days=1)).isoformat()
    downloaded = yf.download(
        list(SYMBOLS), start=START, end=end, interval="1d",
        auto_adjust=True, group_by="column", progress=False,
        threads=False, timeout=20,
    )
    if downloaded.empty or "Close" not in downloaded.columns.get_level_values(0):
        raise RuntimeError("The price provider returned no adjusted close data.")
    raw = downloaded["Close"].reindex(columns=SYMBOLS).sort_index()
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    raw.index.name = "date"
    if raw.index.has_duplicates:
        raise ValueError("Provider returned duplicate session dates.")
    raw_path = data_dir / "provider_adjusted_close.csv"
    raw.to_csv(raw_path, float_format="%.10f")

    numeric = raw.apply(pd.to_numeric, errors="coerce")
    valid = numeric.notna().all(axis=1) & np.isfinite(numeric).all(axis=1) & (numeric > 0).all(axis=1)
    clean = numeric.loc[valid]
    if len(clean) < 700:
        raise ValueError(f"Only {len(clean)} complete sessions; need at least 700.")
    long_rows = [
        {"date": day.strftime("%Y-%m-%d"), "symbol": symbol, "adjusted_close": value}
        for day, values in clean.iterrows() for symbol, value in values.items()
    ]
    quality = validate_rows(long_rows, clean.index.strftime("%Y-%m-%d"), SYMBOLS)
    if any(quality.values()):
        raise ValueError(f"Clean sample failed validation: {quality}")
    clean_path = data_dir / "prices_adjusted.csv"
    clean.to_csv(clean_path, float_format="%.10f")
    manifest = {
        "source": "Yahoo Finance via yfinance",
        "source_url": "https://github.com/ranaroussi/yfinance",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "yfinance_version": version("yfinance"),
        "start_inclusive": START,
        "end_exclusive": end,
        "adjustment": "yfinance auto_adjust=True; adjusted daily close",
        "symbols": list(SYMBOLS),
        "raw_sessions": len(raw),
        "complete_sessions": len(clean),
        "dropped_sessions": raw.index[~valid].strftime("%Y-%m-%d").tolist(),
        "first_complete_session": clean.index[0].date().isoformat(),
        "last_complete_session": clean.index[-1].date().isoformat(),
        "raw_sha256": sha256(raw_path.read_bytes()).hexdigest(),
        "clean_sha256": sha256(clean_path.read_bytes()).hexdigest(),
        "selection_note": "These are present-day chosen stocks, not historical sector indexes; selection bias remains.",
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_snapshot(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "prices_adjusted.csv"
    if not path.exists():
        raise FileNotFoundError(f"No frozen data at {path}; run the fetch command first.")
    frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
    if list(frame.columns) != list(SYMBOLS) or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Snapshot has wrong symbols, duplicate dates, or unsorted dates.")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Snapshot contains missing, nonfinite, or nonpositive prices.")
    if len(frame) < 700:
        raise ValueError("Snapshot is too short for calibration and evaluation.")
    return frame
