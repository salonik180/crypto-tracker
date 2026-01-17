# main.py
import asyncio
import datetime as dt
import os
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine, select, func
from sqlalchemy.orm import declarative_base, sessionmaker

# -------------------------------
# Configuration
# -------------------------------
SCRAPE_INTERVAL_SECONDS = int(os.getenv("SCRAPE_INTERVAL_SECONDS", "60"))  # match 1m candles
TRAIN_WINDOW_POINTS = int(os.getenv("TRAIN_WINDOW_POINTS", "120"))        # 2 hours
PREDICT_HORIZON_MINUTES = float(os.getenv("PREDICT_HORIZON_MINUTES", "5"))
DB_URL = os.getenv("DB_URL", "sqlite:///./prices.db")
BINANCE_BASE = "https://api.binance.com"

DEFAULT_SOURCES = {
    "BTCUSDT": {"provider": "binance", "interval": "1m"},
    "ETHUSDT": {"provider": "binance", "interval": "1m"},
}

# -------------------------------
# Database setup
# -------------------------------
Base = declarative_base()

class Price(Base):
    __tablename__ = "prices"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, index=True, nullable=False)
    price = Column(Float, nullable=False)
    ts = Column(DateTime, index=True, nullable=False)

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base.metadata.create_all(bind=engine)

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="Crypto Price Tracker (Binance Candlesticks)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Schemas
# -------------------------------
class SymbolConfig(BaseModel):
    symbol: str = Field(..., example="BTCUSDT")
    provider: str = Field("binance", example="binance")
    interval: str = Field("1m", example="1m")  # e.g., 1m, 5m, 15m

class PriceOut(BaseModel):
    symbol: str
    price: float
    ts: dt.datetime

class HistoryOut(BaseModel):
    symbol: str
    points: int
    start: dt.datetime
    end: dt.datetime
    data: List[PriceOut]

class PredictionOut(BaseModel):
    symbol: str
    horizon_minutes: float
    predicted_price: float
    trained_points: int
    last_price: float
    last_timestamp: dt.datetime

# -------------------------------
# Cache and shared state
# -------------------------------
class ModelCache:
    def __init__(self):
        self.models: Dict[str, LinearRegression] = {}
        self.buffers: Dict[str, deque] = {}

    def buffer(self, symbol: str) -> deque:
        if symbol not in self.buffers:
            self.buffers[symbol] = deque(maxlen=TRAIN_WINDOW_POINTS)
        return self.buffers[symbol]

SOURCES: Dict[str, Dict] = {k: v.copy() for k, v in DEFAULT_SOURCES.items()}
CACHE = ModelCache()
SCRAPER_TASK_STARTED = False  # guard against multiple loops under --reload

# -------------------------------
# Utilities
# -------------------------------
def now() -> dt.datetime:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

async def fetch_binance_candle(symbol: str, interval: str = "1m") -> Optional[dict]:
    """
    Fetch the latest candlestick (OHLCV) for the given symbol/interval.
    Returns dict with time and close price, or None on failure.
    """
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": 1}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return None
        c = data[-1]
        # c format: [openTime, open, high, low, close, volume, closeTime, ...]
        try:
            return {
                "time": dt.datetime.fromtimestamp(c[0] / 1000, tz=dt.timezone.utc),
                "close": float(c[4]),
            }
        except Exception:
            return None

def store_price(symbol: str, price: float, ts: Optional[dt.datetime] = None):
    ts = ts or now()
    with SessionLocal() as s:
        s.add(Price(symbol=symbol, price=price, ts=ts))
        s.commit()

def load_recent(symbol: str, limit: int) -> List[Price]:
    with SessionLocal() as s:
        stmt = (
            select(Price)
            .where(Price.symbol == symbol)
            .order_by(Price.ts.desc())
            .limit(limit)
        )
        rows = list(reversed(s.execute(stmt).scalars().all()))
        return rows

def train_model(symbol: str) -> Optional[LinearRegression]:
    data = load_recent(symbol, TRAIN_WINDOW_POINTS)
    if len(data) < 10:
        return None
    t0 = data[0].ts
    X = np.array([[(p.ts - t0).total_seconds() / 60.0] for p in data], dtype=float)
    y = np.array([p.price for p in data], dtype=float)
    model = LinearRegression().fit(X, y)
    CACHE.models[symbol] = model
    buf = CACHE.buffer(symbol)
    buf.clear()
    for x, p in zip(X, y):
        buf.append((float(x[0]), float(p)))
    return model

def predict(symbol: str, horizon_minutes: float) -> Optional[PredictionOut]:
    model = CACHE.models.get(symbol) or train_model(symbol)
    recent = load_recent(symbol, 1)
    if not model or not recent:
        return None
    last = recent[-1]
    buf = CACHE.buffer(symbol)
    if buf:
        t_last = buf[-1][0]
    else:
        window = load_recent(symbol, TRAIN_WINDOW_POINTS)
        if not window:
            return None
        t0 = window[0].ts
        t_last = (last.ts - t0).total_seconds() / 60.0
    x_future = np.array([[t_last + horizon_minutes]], dtype=float)
    y_hat = float(model.predict(x_future)[0])
    return PredictionOut(
        symbol=symbol,
        horizon_minutes=horizon_minutes,
        predicted_price=round(y_hat, 2),
        trained_points=len(buf) if buf else min(TRAIN_WINDOW_POINTS, len(load_recent(symbol, TRAIN_WINDOW_POINTS))),
        last_price=last.price,
        last_timestamp=last.ts,
    )

# -------------------------------
# Scraper
# -------------------------------
async def scrape_once(symbol: str, cfg: Dict):
    try:
        if cfg.get("provider") != "binance":
            raise ValueError("Unsupported provider")
        interval = cfg.get("interval", "1m")
        candle = await fetch_binance_candle(symbol, interval=interval)
        if candle is None:
            raise ValueError("No candle fetched")
        store_price(symbol, candle["close"], candle["time"])
        train_model(symbol)
    except Exception as e:
        print(f"[scrape_once] {symbol} error: {e}")

async def scraper_loop():
    while True:
        tasks = [scrape_once(sym, cfg) for sym, cfg in SOURCES.items()]
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            print(f"[scraper_loop] error: {e}")
        await asyncio.sleep(SCRAPE_INTERVAL_SECONDS)

# -------------------------------
# Routes
# -------------------------------
@app.on_event("startup")
async def startup():
    global SCRAPER_TASK_STARTED
    if not SCRAPER_TASK_STARTED:
        SCRAPER_TASK_STARTED = True
        asyncio.create_task(scraper_loop())

@app.get("/health")
async def health():
    with SessionLocal() as s:
        count = s.execute(select(func.count(Price.id))).scalar()
    return {
        "status": "ok",
        "tracked_symbols": list(SOURCES.keys()),
        "rows": count,
        "scrape_interval_seconds": SCRAPE_INTERVAL_SECONDS,
    }

@app.get("/symbols/list")
async def list_symbols():
    return {"symbols": [{"symbol": sym, **cfg} for sym, cfg in SOURCES.items()]}

@app.post("/symbols/add")
async def add_symbol(cfg: SymbolConfig, background_tasks: BackgroundTasks):
    sym = cfg.symbol.upper()
    if cfg.provider != "binance":
        raise HTTPException(400, "Only 'binance' provider is supported.")
    SOURCES[sym] = {"provider": "binance", "interval": cfg.interval}
    background_tasks.add_task(scrape_once, sym, SOURCES[sym])  # prime data quickly
    return {"added": sym, "provider": "binance", "interval": cfg.interval}

@app.post("/symbols/remove")
async def remove_symbol(symbol: str):
    sym = symbol.upper()
    if sym in SOURCES:
        del SOURCES[sym]
        CACHE.models.pop(sym, None)
        CACHE.buffers.pop(sym, None)
        return {"removed": sym}
    raise HTTPException(404, f"{sym} not tracked.")

@app.get("/price/{symbol}", response_model=PriceOut)
async def get_price(symbol: str):
    rows = load_recent(symbol, 1)
    if not rows:
        raise HTTPException(404, f"No price found for {symbol}.")
    r = rows[-1]
    return PriceOut(symbol=r.symbol, price=r.price, ts=r.ts)

@app.get("/history/{symbol}", response_model=HistoryOut)
async def get_history(symbol: str, points: int = 200):
    rows = load_recent(symbol, points)
    if not rows:
        raise HTTPException(404, f"No history for {symbol}.")
    return HistoryOut(
        symbol=symbol,
        points=len(rows),
        start=rows[0].ts,
        end=rows[-1].ts,
        data=[PriceOut(symbol=r.symbol, price=r.price, ts=r.ts) for r in rows],
    )

@app.get("/predict/{symbol}", response_model=PredictionOut)
async def get_prediction(symbol: str, horizon_minutes: float = PREDICT_HORIZON_MINUTES):
    pred = predict(symbol, horizon_minutes)
    if pred is None:
        raise HTTPException(400, "Model not trained yet or insufficient data.")
    return pred
