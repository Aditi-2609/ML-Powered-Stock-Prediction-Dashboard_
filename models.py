"""Forecasting models. Both return a one-step-ahead close price."""
import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:  # pragma: no cover
    ARIMA = None

try:
    from tensorflow.keras.layers import LSTM, Dense, Input
    from tensorflow.keras.models import Sequential
except ImportError:  # pragma: no cover
    Sequential = LSTM = Dense = Input = None

WINDOW = 5


def _clean(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def predict_arima(series: pd.Series) -> float:
    values = _clean(series)
    if values.empty:
        return 0.0
    if ARIMA is None or len(values) < 20:
        return float(values.iloc[-1])
    try:
        fit = ARIMA(values.to_numpy(), order=(5, 1, 0)).fit()
        return float(fit.forecast(steps=1)[0])
    except Exception:
        return float(values.iloc[-1])


def predict_lstm(series: pd.Series) -> float:
    values = _clean(series)
    if values.empty:
        return 0.0
    last = float(values.iloc[-1])
    if len(values) < 30:
        return last

    arr = values.to_numpy(dtype="float32")
    lo, hi = arr.min(), arr.max()
    scaled = (arr - lo) / (hi - lo + 1e-9)

    if Sequential is None:  # no TensorFlow: damped recent-trend estimate
        return float(last + 0.5 * np.mean(np.diff(arr[-WINDOW:])))

    X = np.array([scaled[i:i + WINDOW] for i in range(len(scaled) - WINDOW)]).reshape(-1, WINDOW, 1)
    y = scaled[WINDOW:]
    try:
        model = Sequential([Input(shape=(WINDOW, 1)), LSTM(32), Dense(1)])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=15, batch_size=16, verbose=0)
        nxt = model.predict(scaled[-WINDOW:].reshape(1, WINDOW, 1), verbose=0)[0][0]
        return float(nxt * (hi - lo) + lo)
    except Exception:
        return last
