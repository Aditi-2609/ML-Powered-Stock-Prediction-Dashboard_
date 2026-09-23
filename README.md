# Market Pulse

FastAPI stock dashboard: live-style market board, candlestick charts, ARIMA / LSTM next-close forecasts.

## Run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
./start.sh            # or: uvicorn main:app --reload
```
Open http://localhost:8000

## Data
- History is stored in `stocks.db` (SQLite) and created automatically on first run.
- Without an API key the app uses generated demo data with simulated live ticks.
- For real US daily candles: `export ALPHA_VANTAGE_KEY=your_key` (Alpha Vantage free tier, 25 requests/day).
- LSTM needs `tensorflow` (uncomment it in requirements.txt); without it a simple trend fallback is used.
