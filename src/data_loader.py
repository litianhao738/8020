from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATE_COLUMNS = ("date", "Date", "trading_date", "TradingDate")
TIME_COLUMNS = ("time", "Time", "datetime_time", "UpdateTime")
DATETIME_COLUMNS = ("datetime", "Datetime", "date_time", "timestamp", "Timestamp")
PRICE_COLUMNS = ("close", "Close", "price", "Price", "last", "LastPrice", "hi1_close")
OPEN_COLUMNS = ("open", "Open", "hi1_open")
HIGH_COLUMNS = ("high", "High", "hi1_high")
LOW_COLUMNS = ("low", "Low", "hi1_low")
VOLUME_COLUMNS = ("volume", "Volume", "vol", "VolumeTraded", "hi1_volume")

MORNING_START = 9 * 60 + 15
MORNING_END = 11 * 60 + 59
AFTERNOON_START = 13 * 60
AFTERNOON_END = 16 * 60 + 30
NIGHT_START = 17 * 60 + 15
NIGHT_END = 23 * 60 + 59
NIGHT_EARLY_END = 3 * 60


def _first_existing(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    return None


def _parse_datetime(df: pd.DataFrame) -> pd.Series:
    datetime_col = _first_existing(df.columns, DATETIME_COLUMNS)
    if datetime_col:
        return pd.to_datetime(df[datetime_col], errors="coerce")

    date_col = _first_existing(df.columns, DATE_COLUMNS)
    time_col = _first_existing(df.columns, TIME_COLUMNS)
    if not date_col or not time_col:
        raise ValueError(
            "Cannot find datetime columns. Expected one datetime column or date/time columns."
        )

    dates = df[date_col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    times = df[time_col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    return pd.to_datetime(dates + times, format="%Y%m%d%H%M%S", errors="coerce")


def load_intraday_data(path: str | Path) -> pd.DataFrame:
    """Load minute or bar data into a normalized OHLCV dataframe."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]
    df["datetime"] = _parse_datetime(df)
    df = df.dropna(subset=["datetime"]).sort_values("datetime").set_index("datetime")

    close_col = _first_existing(df.columns, PRICE_COLUMNS)
    if not close_col:
        raise ValueError("Cannot find a close/price column.")

    normalized = pd.DataFrame(index=df.index)
    normalized["close"] = pd.to_numeric(df[close_col], errors="coerce")

    for target, candidates in {
        "open": OPEN_COLUMNS,
        "high": HIGH_COLUMNS,
        "low": LOW_COLUMNS,
        "volume": VOLUME_COLUMNS,
    }.items():
        source = _first_existing(df.columns, candidates)
        if source:
            normalized[target] = pd.to_numeric(df[source], errors="coerce")

    normalized["open"] = normalized.get("open", normalized["close"])
    normalized["high"] = normalized.get("high", normalized["close"])
    normalized["low"] = normalized.get("low", normalized["close"])
    normalized["volume"] = normalized.get("volume", 0.0)

    normalized = normalized.dropna(subset=["close"])
    return normalized[["open", "high", "low", "close", "volume"]]


def add_trading_session(data: pd.DataFrame) -> pd.DataFrame:
    """Annotate HI1 continuous trading sessions and exchange trading dates."""
    annotated = data.copy()
    idx = pd.Series(annotated.index, index=annotated.index)
    minutes = idx.dt.hour * 60 + idx.dt.minute
    calendar_date = idx.dt.normalize()

    session = pd.Series(pd.NA, index=annotated.index, dtype="object")
    trading_date = calendar_date.copy()

    morning = minutes.between(MORNING_START, MORNING_END)
    afternoon = minutes.between(AFTERNOON_START, AFTERNOON_END)
    night_evening = minutes.between(NIGHT_START, NIGHT_END)
    night_early = minutes.between(0, NIGHT_EARLY_END)

    session[morning] = "morning"
    session[afternoon] = "afternoon"
    session[night_evening | night_early] = "night"
    trading_date[night_early] = trading_date[night_early] - pd.Timedelta(days=1)

    valid_trading_day = trading_date.dt.weekday < 5
    valid_session = session.notna() & valid_trading_day
    annotated = annotated.loc[valid_session].copy()
    session = session.loc[valid_session]
    trading_date = trading_date.loc[valid_session]

    annotated["trading_date"] = trading_date.dt.date
    annotated["session"] = session
    annotated["session_id"] = (
        trading_date.dt.strftime("%Y-%m-%d") + "_" + session.astype(str)
    )
    return annotated


def clean_minute_data(data: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate minutes, invalid rows, and non-HI1 session timestamps."""
    cleaned = data.copy()
    cleaned = cleaned[cleaned.index.notna()]
    cleaned = cleaned[cleaned["close"] > 0]
    cleaned = cleaned.groupby(cleaned.index).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    cleaned = add_trading_session(cleaned)
    return cleaned.sort_index()


def resample_ohlcv(data: pd.DataFrame, frequency: str = "5min") -> pd.DataFrame:
    """Convert cleaned minute data to OHLCV bars."""
    aggregations = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "session_id" not in data.columns:
        bars = data.resample(frequency).agg(aggregations)
        return bars.dropna(subset=["open", "high", "low", "close"])

    pieces = []
    for session_id, session_data in data.groupby("session_id", sort=True):
        bars = session_data.resample(frequency).agg(aggregations)
        bars = bars.dropna(subset=["open", "high", "low", "close"])
        if bars.empty:
            continue
        bars["trading_date"] = session_data["trading_date"].iloc[0]
        bars["session"] = session_data["session"].iloc[0]
        bars["session_id"] = session_id
        pieces.append(bars)

    if not pieces:
        return pd.DataFrame(columns=[*aggregations, "trading_date", "session", "session_id"])
    return pd.concat(pieces).sort_index()


def add_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Add simple returns and log returns."""
    enriched = data.copy()
    enriched["return"] = enriched["close"].pct_change()
    enriched["log_return"] = np.log(enriched["close"] / enriched["close"].shift(1))
    return enriched.dropna(subset=["return", "log_return"])


def split_sample(
    data: pd.DataFrame,
    in_sample_end: str,
    out_sample_start: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into in-sample and out-of-sample windows."""
    in_sample = data.loc[:in_sample_end].copy()
    start = out_sample_start or in_sample_end
    out_sample = data.loc[start:].copy()
    return in_sample, out_sample
