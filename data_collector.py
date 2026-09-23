"""Data layer: SQLite storage (stocks.db), demo/real history, and simulated live ticks."""
import os
import random
import sqlite3
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "stocks.db"
API_KEY = os.getenv("ALPHA_VANTAGE_KEY")  # optional: real daily data for US symbols

# symbol, company, market, approx. price, avg daily volume
STOCKS = [
    ("AAPL", "Apple Inc.", "US", 228.0, 55_000_000),
    ("MSFT", "Microsoft Corp.", "US", 432.0, 20_000_000),
    ("GOOG", "Alphabet Inc.", "US", 168.0, 22_000_000),
    ("NVDA", "NVIDIA Corp.", "US", 118.0, 280_000_000),
    ("AMZN", "Amazon.com Inc.", "US", 185.0, 40_000_000),
    ("META", "Meta Platforms", "US", 560.0, 14_000_000),
    ("TSLA", "Tesla Inc.", "US", 245.0, 90_000_000),
    ("NFLX", "Netflix Inc.", "US", 690.0, 3_500_000),
    ("RELIANCE", "Reliance Industries", "IN", 2950.0, 6_000_000),
    ("TCS", "Tata Consultancy", "IN", 4100.0, 2_000_000),
    ("INFY", "Infosys Ltd.", "IN", 1880.0, 7_000_000),
    ("HDFCBANK", "HDFC Bank", "IN", 1720.0, 9_000_000),
    ("ICICIBANK", "ICICI Bank", "IN", 1240.0, 12_000_000),
    ("SBIN", "State Bank of India", "IN", 820.0, 15_000_000),
    ("ITC", "ITC Ltd.", "IN", 480.0, 14_000_000),
    ("BHARTIARTL", "Bharti Airtel", "IN", 1650.0, 5_000_000),
]
_REGISTRY = {s[0]: s for s in STOCKS}
CURRENCY = {"US": "$", "IN": "₹"}

_lock = threading.Lock()
_hist_cache: dict[str, pd.DataFrame] = {}
_live: dict[str, dict] = {}


def _conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT NOT NULL, date TEXT NOT NULL,
                open REAL, high REAL, low REAL, close REAL, volume INTEGER,
                PRIMARY KEY (symbol, date))"""
        )
        c.execute("CREATE TABLE IF NOT EXISTS meta (symbol TEXT PRIMARY KEY, source TEXT, updated TEXT)")


def stock_info(symbol: str) -> dict:
    symbol = (symbol or "AAPL").upper().strip()
    row = _REGISTRY.get(symbol)
    if row is None:  # unknown ticker: still works, with generated demo data
        rng = random.Random(symbol)
        row = (symbol, symbol, "US", round(rng.uniform(20, 400), 2), rng.randint(1, 30) * 1_000_000)
    return {"symbol": row[0], "name": row[1], "market": row[2], "currency": CURRENCY[row[2]],
            "base": row[3], "avg_volume": row[4]}


def _weekdays_back(n: int) -> list[date]:
    out, d = [], date.today()
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out[::-1]


def _generate(symbol: str, n: int = 260) -> pd.DataFrame:
    info = stock_info(symbol)
    rng = random.Random(symbol)
    sigma = rng.uniform(0.011, 0.022)
    drift = rng.uniform(-0.0002, 0.0009)
    closes, p = [], 1.0
    for _ in range(n):
        p *= 1 + rng.gauss(drift, sigma)
        closes.append(p)
    scale = info["base"] * rng.uniform(0.96, 1.04) / closes[-1]
    rows, prev = [], closes[0] * scale
    for d, c in zip(_weekdays_back(n), closes):
        c *= scale
        o = prev * (1 + rng.gauss(0, sigma / 3))
        hi = max(o, c) * (1 + abs(rng.gauss(0, sigma / 2)))
        lo = min(o, c) * (1 - abs(rng.gauss(0, sigma / 2)))
        vol = int(info["avg_volume"] * rng.uniform(0.6, 1.6))
        rows.append((symbol, d.isoformat(), round(o, 2), round(hi, 2), round(lo, 2), round(c, 2), vol))
        prev = c
    return pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low", "close", "volume"])


def _try_alpha_vantage(symbol: str) -> pd.DataFrame | None:
    """Real daily candles for US tickers when ALPHA_VANTAGE_KEY is set."""
    if not API_KEY or stock_info(symbol)["market"] != "US":
        return None
    try:
        from alpha_vantage.timeseries import TimeSeries

        data, _ = TimeSeries(key=API_KEY, output_format="pandas").get_daily(symbol=symbol, outputsize="compact")
        if data is None or data.empty:
            return None
        df = data.rename(columns={"1. open": "open", "2. high": "high", "3. low": "low",
                                  "4. close": "close", "5. volume": "volume"}).reset_index()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["symbol"] = symbol
        return df[["symbol", "date", "open", "high", "low", "close", "volume"]].sort_values("date")
    except Exception:
        return None


def get_history(symbol: str) -> pd.DataFrame:
    """Ascending daily OHLCV. Loaded from stocks.db; created on first request."""
    symbol = stock_info(symbol)["symbol"]
    with _lock:
        if symbol in _hist_cache:
            return _hist_cache[symbol]
    with _conn() as c:
        df = pd.read_sql_query("SELECT * FROM candles WHERE symbol=? ORDER BY date", c, params=(symbol,))
        fresh = c.execute("SELECT updated FROM meta WHERE symbol=?", (symbol,)).fetchone()
        stale = df.empty or (API_KEY and (not fresh or fresh[0] != date.today().isoformat()))
        if stale:
            new = _try_alpha_vantage(symbol)
            source = "alphavantage" if new is not None else "demo"
            if new is None and df.empty:
                new = _generate(symbol)
            if new is not None:
                c.execute("DELETE FROM candles WHERE symbol=?", (symbol,))
                c.executemany("INSERT INTO candles VALUES (?,?,?,?,?,?,?)", new.itertuples(index=False, name=None))
                df = new.reset_index(drop=True)
            c.execute("INSERT OR REPLACE INTO meta VALUES (?,?,?)", (symbol, source, date.today().isoformat()))
    with _lock:
        _hist_cache[symbol] = df
    return df


def live_quote(symbol: str) -> dict:
    """Latest session quote. Each call nudges the price a little to mimic a live tape."""
    info = stock_info(symbol)
    df = get_history(info["symbol"])
    with _lock:
        st = _live.get(info["symbol"])
        if st is None:
            last = df.iloc[-1]
            st = {"date": last["date"], "prev_close": float(df.iloc[-2]["close"]) if len(df) > 1 else float(last["close"]),
                  "open": float(last["open"]), "high": float(last["high"]), "low": float(last["low"]),
                  "price": float(last["close"]), "anchor": float(last["close"]), "volume": int(last["volume"])}
            _live[info["symbol"]] = st
        pull = 0.03 * (st["anchor"] - st["price"]) / st["anchor"]
        st["price"] = round(max(0.01, st["price"] * (1 + random.gauss(pull, 0.0008))), 2)
        st["high"] = max(st["high"], st["price"])
        st["low"] = min(st["low"], st["price"])
        st["volume"] += random.randint(200, 4000)
        change = st["price"] - st["prev_close"]
        return {**info, **st, "change": round(change, 2), "pct": round(change / st["prev_close"] * 100, 2),
                "updated": datetime.now().strftime("%H:%M:%S")}


def market_snapshot(with_spark: bool = False) -> list[dict]:
    out = []
    for sym, *_ in STOCKS:
        q = live_quote(sym)
        if with_spark:
            q["spark"] = [round(float(x), 2) for x in get_history(sym)["close"].tail(30)]
        out.append(q)
    return out


def fetch_stock_data(symbol: str) -> pd.DataFrame:
    """Backwards-compatible helper (ascending order, ready for the forecasting models)."""
    return get_history(symbol)
