import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from data_collector import get_history, init_db, live_quote, market_snapshot, stock_info
from models import predict_arima, predict_lstm

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="Market Pulse", lifespan=lifespan)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

_forecast_cache: dict[tuple, tuple] = {}
FORECAST_TTL = 300  # seconds; LSTM training is slow, so results are cached


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"stocks": market_snapshot(with_spark=True)})


@app.get("/stock/{symbol}", response_class=HTMLResponse)
def stock_page(request: Request, symbol: str, model: str = "arima"):
    info = stock_info(symbol)
    model = model.lower() if model.lower() in {"arima", "lstm"} else "arima"
    return templates.TemplateResponse(
        request, "stock.html",
        {"info": info, "model": model, "quote": live_quote(info["symbol"]), "others": market_snapshot()},
    )


@app.get("/api/quotes")
def api_quotes():
    return market_snapshot()


@app.get("/api/stock/{symbol}/candles")
def api_candles(symbol: str):
    df = get_history(symbol)
    return {
        "candles": [{"time": r.date, "open": r.open, "high": r.high, "low": r.low, "close": r.close}
                    for r in df.itertuples()],
        "volume": [{"time": r.date, "value": int(r.volume), "up": bool(r.close >= r.open)} for r in df.itertuples()],
    }


@app.get("/api/stock/{symbol}/live")
def api_live(symbol: str):
    return live_quote(symbol)


@app.get("/api/stock/{symbol}/forecast")
def api_forecast(symbol: str, model: str = "arima"):
    symbol = stock_info(symbol)["symbol"]
    model = model.lower() if model.lower() in {"arima", "lstm"} else "arima"
    key = (symbol, model)
    hit = _forecast_cache.get(key)
    if hit and time.time() - hit[0] < FORECAST_TTL:
        return hit[1]
    closes = get_history(symbol)["close"]
    prediction = predict_arima(closes) if model == "arima" else predict_lstm(closes)
    last = float(closes.iloc[-1])
    result = {"model": model, "prediction": round(prediction, 2), "last": last,
              "change": round(prediction - last, 2), "pct": round((prediction - last) / last * 100, 2)}
    _forecast_cache[key] = (time.time(), result)
    return result
