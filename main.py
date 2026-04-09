# ═══════════════════════════════════════════════════════════════════
#  CryptoPulse — Blockchain Intelligence API
#  main.py  |  FastAPI + SQLAlchemy + scikit-learn
# ═══════════════════════════════════════════════════════════════════
#
#  Quick start:
#    pip install fastapi uvicorn httpx sqlalchemy scikit-learn numpy pydantic
#    uvicorn main:app --reload
#
#  Environment variables (all optional):
#    DB_URL                  sqlite:///./prices.db
#    SCRAPE_INTERVAL_SECONDS 60
#    TRAIN_WINDOW_POINTS     120
#    PREDICT_HORIZON_MINUTES 5
#    BINANCE_REGION          com   (set to "us" if hosted in the USA)
#    BINANCE_CONCURRENCY     2
#    SCRAPE_STAGGER_SECONDS  1.0
#    HTTP_TIMEOUT            8     (reduced from 20 — avoids long hangs on blocked APIs)
#    HTTP_MAX_RETRIES        2     (reduced from 3 — faster failure on blocked APIs)
#    COINGECKO_API_KEY       (optional — improves rate limits)
#    ETHERSCAN_API_KEY       (optional — used for gas tracker)
# ═══════════════════════════════════════════════════════════════════

import asyncio
import datetime as dt
import os
import random
from collections import deque
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import httpx
import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text,
    create_engine, func, select,
)
from sqlalchemy.orm import declarative_base, sessionmaker


# ───────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ───────────────────────────────────────────────────────────────────

SCRAPE_INTERVAL_SECONDS = int(os.getenv("SCRAPE_INTERVAL_SECONDS", "60"))
TRAIN_WINDOW_POINTS     = int(os.getenv("TRAIN_WINDOW_POINTS", "120"))
PREDICT_HORIZON_MINUTES = float(os.getenv("PREDICT_HORIZON_MINUTES", "5"))
DB_URL                  = os.getenv("DB_URL", "sqlite:///./prices.db")

_REGION      = os.getenv("BINANCE_REGION", "com").lower()
BINANCE_BASE = f"https://api.binance.{'us' if _REGION == 'us' else 'com'}"

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
COINGECKO_KEY  = os.getenv("COINGECKO_API_KEY", "")

BLOCKSTREAM_BASE   = "https://blockstream.info/api"
ETHERSCAN_KEY      = os.getenv("ETHERSCAN_API_KEY", "")
ETHERSCAN_BASE     = "https://api.etherscan.io/api"
BLOCKNATIVE        = "https://api.blocknative.com/gasprices/blockprices"
CRYPTOCOMPARE_NEWS = "https://min-api.cryptocompare.com/data/v2/news/"

# FIX: reduced defaults so blocked/unreachable APIs fail fast instead of
# hanging for 20s × 3 retries = 60s per endpoint.
HTTP_TIMEOUT     = float(os.getenv("HTTP_TIMEOUT", "8"))
HTTP_MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "2"))

# Limits simultaneous Binance connections — prevents DNS/pool exhaustion.
BINANCE_CONCURRENCY    = int(os.getenv("BINANCE_CONCURRENCY", "2"))
SCRAPE_STAGGER_SECONDS = float(os.getenv("SCRAPE_STAGGER_SECONDS", "1.0"))

# Default tracked symbols
DEFAULT_SOURCES: Dict[str, Dict] = {
    "BTCUSDT": {"provider": "binance", "interval": "1m"},
    "ETHUSDT": {"provider": "binance", "interval": "1m"},
    "BNBUSDT": {"provider": "binance", "interval": "1m"},
    "SOLUSDT": {"provider": "binance", "interval": "1m"},
    "ADAUSDT": {"provider": "binance", "interval": "1m"},
    "XRPUSDT": {"provider": "binance", "interval": "1m"},
}

# binance.us uses slightly different symbol names for some pairs
BINANCE_US_SYMBOL_MAP: Dict[str, str] = {
    "BNBUSDT": "BNBUSD",
}

# Maps our trading-pair symbols → CoinGecko coin IDs
COINGECKO_IDS: Dict[str, str] = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "BNBUSDT": "binancecoin",
    "SOLUSDT": "solana",
    "ADAUSDT": "cardano",
    "XRPUSDT": "ripple",
}


# ───────────────────────────────────────────────────────────────────
# 2. DATABASE
# ───────────────────────────────────────────────────────────────────

Base = declarative_base()


class Price(Base):
    """One scraped price sample for a symbol."""
    __tablename__ = "prices"

    id     = Column(Integer,                 primary_key=True, autoincrement=True)
    symbol = Column(String,                  index=True, nullable=False)
    price  = Column(Float,                   nullable=False)
    # timezone=True → SQLAlchemy stores/reads UTC-aware datetimes.
    # Without it, SQLite strips tzinfo on reads, causing TypeError in
    # train_model() and missing 'Z' in ISO-8601 API responses.
    ts     = Column(DateTime(timezone=True), index=True, nullable=False)


class NewsItem(Base):
    """Cached news headline (not currently served via API but ready to use)."""
    __tablename__ = "news"

    id        = Column(Integer,                  primary_key=True, autoincrement=True)
    title     = Column(Text,                     nullable=False)
    url       = Column(Text,                     nullable=False)
    source    = Column(String,                   nullable=True)
    published = Column(DateTime(timezone=True),  nullable=True)
    fetched   = Column(DateTime(timezone=True),  nullable=False)


engine       = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base.metadata.create_all(bind=engine)


# ───────────────────────────────────────────────────────────────────
# 3. PYDANTIC SCHEMAS
# ───────────────────────────────────────────────────────────────────

class SymbolConfig(BaseModel):
    symbol:   str = Field(...,       example="BTCUSDT")
    provider: str = Field("binance", example="binance")
    interval: str = Field("1m",      example="1m")


class PriceOut(BaseModel):
    symbol: str
    price:  float
    ts:     dt.datetime


class HistoryOut(BaseModel):
    symbol: str
    points: int
    start:  dt.datetime
    end:    dt.datetime
    data:   List[PriceOut]


class PredictionOut(BaseModel):
    symbol:          str
    horizon_minutes: float
    predicted_price: float
    trained_points:  int
    last_price:      float
    last_timestamp:  dt.datetime


class CandleOut(BaseModel):
    ts:     dt.datetime
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float


class MarketOut(BaseModel):
    symbol:               str
    name:                 str
    current_price:        float
    market_cap:           Optional[float] = None
    market_cap_rank:      Optional[int]   = None
    total_volume:         Optional[float] = None
    high_24h:             Optional[float] = None
    low_24h:              Optional[float] = None
    price_change_24h:     Optional[float] = None
    price_change_pct_24h: Optional[float] = None
    circulating_supply:   Optional[float] = None
    ath:                  Optional[float] = None
    ath_change_pct:       Optional[float] = None
    last_updated:         Optional[str]   = None


class GlobalOut(BaseModel):
    total_market_cap_usd:      float
    total_volume_usd:          float
    bitcoin_dominance_pct:     float
    active_cryptocurrencies:   int
    markets:                   int
    market_cap_change_pct_24h: float


class FearGreedOut(BaseModel):
    value:          int
    classification: str
    timestamp:      str


class BtcNetworkOut(BaseModel):
    height:           Optional[int]   = None
    difficulty:       Optional[float] = None
    hash_rate_th_s:   Optional[float] = None
    mempool_tx_count: Optional[int]   = None
    mempool_bytes:    Optional[int]   = None
    fee_fastest_sat:  Optional[int]   = None
    fee_halfhour_sat: Optional[int]   = None
    fee_hour_sat:     Optional[int]   = None


class EthGasOut(BaseModel):
    safe_gas_gwei:    Optional[float] = None
    propose_gas_gwei: Optional[float] = None
    fast_gas_gwei:    Optional[float] = None
    base_fee_gwei:    Optional[float] = None
    eth_price_usd:    Optional[float] = None


class ExchangeOut(BaseModel):
    id:                   str
    name:                 str
    trust_score:          Optional[int]   = None
    trade_volume_24h_btc: Optional[float] = None
    url:                  Optional[str]   = None
    country:              Optional[str]   = None


class TrendingOut(BaseModel):
    id:              str
    name:            str
    symbol:          str
    market_cap_rank: Optional[int] = None
    thumb:           Optional[str] = None


class NewsOut(BaseModel):
    title:     str
    url:       str
    source:    Optional[str]         = None
    published: Optional[dt.datetime] = None


class HoldingIn(BaseModel):
    symbol:   str   = Field(..., example="BTCUSDT")
    quantity: float = Field(..., gt=0, example=0.5)


class PortfolioIn(BaseModel):
    holdings: List[HoldingIn]


# ───────────────────────────────────────────────────────────────────
# 4. FALLBACK / STUB DATA
# ───────────────────────────────────────────────────────────────────
# FIX: When external APIs (CoinGecko, alternative.me, etc.) are unreachable
# (common on networks that block foreign API domains), return sensible stubs
# instead of HTTP 502 errors.  Stubs make the frontend still renderable
# and clearly signal that data is unavailable without crashing anything.

_UNAVAILABLE = "Unavailable"

FALLBACK_GLOBAL = GlobalOut(
    total_market_cap_usd=0,
    total_volume_usd=0,
    bitcoin_dominance_pct=0,
    active_cryptocurrencies=0,
    markets=0,
    market_cap_change_pct_24h=0,
)

FALLBACK_FEAR_GREED = FearGreedOut(
    value=50,
    classification=_UNAVAILABLE,
    timestamp="0",
)

FALLBACK_BTC_NETWORK = BtcNetworkOut()   # all fields None
FALLBACK_ETH_GAS     = EthGasOut()       # all fields None


# ───────────────────────────────────────────────────────────────────
# 5. IN-MEMORY MODEL CACHE
# ───────────────────────────────────────────────────────────────────

class ModelCache:
    """Stores trained LinearRegression models and recent price buffers."""

    def __init__(self) -> None:
        self.models:  Dict[str, LinearRegression] = {}
        self.buffers: Dict[str, deque]            = {}

    def buffer(self, symbol: str) -> deque:
        if symbol not in self.buffers:
            self.buffers[symbol] = deque(maxlen=TRAIN_WINDOW_POINTS)
        return self.buffers[symbol]


# Module-level singletons
SOURCES: Dict[str, Dict] = {k: v.copy() for k, v in DEFAULT_SOURCES.items()}
CACHE = ModelCache()


# ───────────────────────────────────────────────────────────────────
# 5b. RESPONSE CACHE  (prevents CoinGecko 429 rate-limit errors)
# ───────────────────────────────────────────────────────────────────
# CoinGecko free tier: ~30 req/min total.  Without caching, every browser
# tab calling /market/overview, /market/global, /exchanges simultaneously
# burns the quota in seconds.  This simple TTL cache ensures each expensive
# endpoint hits the upstream API at most once per CACHE_TTL seconds,
# regardless of how many clients are calling your server.

CACHE_TTL: Dict[str, int] = {
    "market_overview": 60,   # seconds — CoinGecko /coins/markets
    "market_global":   60,   # seconds — CoinGecko /global
    "market_trending": 120,  # seconds — CoinGecko /search/trending
    "exchanges":       120,  # seconds — CoinGecko /exchanges
    "fear_greed":      300,  # seconds — alternative.me
    "news":            120,  # seconds — CryptoCompare
    "btc_network":     30,   # seconds — mempool.space
    "eth_gas":         30,   # seconds — blocknative / etherscan
}

# Stores (payload, expiry_unix_timestamp) per cache key
_response_cache: Dict[str, tuple] = {}


def _cache_get(key: str):
    """Return cached payload if TTL has not expired, else None."""
    entry = _response_cache.get(key)
    if entry and entry[1] > dt.datetime.now(dt.timezone.utc).timestamp():
        return entry[0]
    return None


def _cache_set(key: str, value) -> None:
    """Store value; expires after CACHE_TTL[key] seconds from now."""
    # Use the longest matching prefix key for TTL lookup
    ttl = 60
    for prefix, t in CACHE_TTL.items():
        if key == prefix or key.startswith(prefix + ":"):
            ttl = t
            break
    _response_cache[key] = (value, dt.datetime.now(dt.timezone.utc).timestamp() + ttl)


# ───────────────────────────────────────────────────────────────────
# 6. HTTP HELPER
# ───────────────────────────────────────────────────────────────────

def _coingecko_headers() -> dict:
    """
    CoinGecko v3 requires the Accept header and rejects requests without it
    with a 401.  An optional demo/pro API key can be supplied via env var.
    """
    headers = {"accept": "application/json"}
    if COINGECKO_KEY:
        headers["x-cg-demo-api-key"] = COINGECKO_KEY
    return headers


async def _get(
    url: str,
    params:  Optional[dict] = None,
    headers: Optional[dict] = None,
    retries: int = HTTP_MAX_RETRIES,
) -> Optional[dict | list]:
    """
    Async GET with retry logic.

    Handles:
      - 429 rate-limit  → waits Retry-After seconds before retrying
      - 418 IP ban      → waits Retry-After seconds before retrying
      - 451 geo-block   → logs and returns None immediately (no point retrying)
      - Timeout / conn  → exponential back-off between attempts
    """
    for attempt in range(1, retries + 1):
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                response = await client.get(url, params=params, headers=headers)

                if response.status_code == 451:
                    print(
                        f"[HTTP] {url} → 451 geo-block. "
                        "Set BINANCE_REGION=us if running on a US server."
                    )
                    return None

                if response.status_code == 418:
                    wait = int(response.headers.get("Retry-After", 60))
                    print(f"[HTTP] {url} → 418 IP ban. Backing off {wait}s.")
                    await asyncio.sleep(wait)
                    continue

                if response.status_code == 429:
                    wait = int(response.headers.get("Retry-After", 10))
                    print(
                        f"[HTTP] {url} → 429 rate-limited. "
                        f"Waiting {wait}s (attempt {attempt}/{retries})."
                    )
                    await asyncio.sleep(wait)
                    continue

                response.raise_for_status()
                return response.json()

        except httpx.TimeoutException:
            print(
                f"[HTTP] Timeout on {url} "
                f"(attempt {attempt}/{retries}, timeout={HTTP_TIMEOUT}s)"
            )
        except httpx.ConnectError as exc:
            print(f"[HTTP] Connection error on {url}: {exc} (attempt {attempt}/{retries})")
        except httpx.HTTPStatusError as exc:
            print(
                f"[HTTP] HTTP {exc.response.status_code} on {url}: "
                f"{exc.response.text[:200]}"
            )
            return None  # Non-retryable HTTP error
        except Exception as exc:
            print(f"[HTTP] Unexpected error on {url}: {type(exc).__name__}: {exc}")

        if attempt < retries:
            await asyncio.sleep(2 ** attempt)  # Exponential back-off

    print(f"[HTTP] All {retries} attempts failed for {url}")
    return None


# ───────────────────────────────────────────────────────────────────
# 7. UTILITY FUNCTIONS
# ───────────────────────────────────────────────────────────────────

def utcnow() -> dt.datetime:
    """Return current UTC time as a timezone-aware datetime."""
    return dt.datetime.now(dt.timezone.utc)


def _binance_symbol(symbol: str) -> str:
    """Translate a symbol to its binance.us equivalent when needed."""
    if _REGION == "us":
        return BINANCE_US_SYMBOL_MAP.get(symbol.upper(), symbol.upper())
    return symbol.upper()


# ───────────────────────────────────────────────────────────────────
# 8. BINANCE FETCH HELPERS
# ───────────────────────────────────────────────────────────────────

async def fetch_binance_price(symbol: str) -> Optional[dict]:
    """
    Fetch the latest spot price from Binance ticker API.

    Uses a semaphore to cap concurrent requests and avoid pool exhaustion.
    Returns {"time": datetime, "close": float} or None on failure.
    """
    bs = _binance_symbol(symbol)

    async with _BINANCE_SEM:
        data = await _get(
            f"{BINANCE_BASE}/api/v3/ticker/price",
            params={"symbol": bs},
        )

    if not data or "price" not in data:
        print(f"[Binance] Unexpected ticker response for {bs}: {data}")
        return None

    try:
        return {"time": utcnow(), "close": float(data["price"])}
    except (KeyError, ValueError, TypeError) as exc:
        print(f"[Binance] Parse error for {bs}: {exc}  raw={data}")
        return None


async def fetch_binance_candle(symbol: str, interval: str = "1m") -> Optional[dict]:
    """
    Fetch the most recently *closed* 1-minute candle from Binance klines.

    Returns {"time": datetime, "close": float} or None on failure.
    """
    bs   = _binance_symbol(symbol)
    data = await _get(
        f"{BINANCE_BASE}/api/v3/klines",
        params={"symbol": bs, "interval": interval, "limit": 2},
    )

    if not isinstance(data, list) or not data:
        print(f"[Binance] Unexpected klines response for {bs}: {data}")
        return None

    # Use the second-to-last candle — the last one is still forming.
    row = data[-2] if len(data) >= 2 else data[-1]

    if not isinstance(row, list) or len(row) < 5:
        print(f"[Binance] Malformed candle row for {bs}: {row}")
        return None

    try:
        return {
            "time":  dt.datetime.fromtimestamp(row[0] / 1000, tz=dt.timezone.utc),
            "close": float(row[4]),
        }
    except (TypeError, ValueError) as exc:
        print(f"[Binance] Candle parse error for {bs}: {exc}  row={row}")
        return None


# ───────────────────────────────────────────────────────────────────
# 9. DATABASE HELPERS
# ───────────────────────────────────────────────────────────────────

def store_price(symbol: str, price: float, ts: Optional[dt.datetime] = None) -> None:
    """Persist one price sample to the database."""
    with SessionLocal() as session:
        session.add(Price(symbol=symbol, price=price, ts=ts or utcnow()))
        session.commit()


def load_recent(symbol: str, limit: int) -> List[Price]:
    """
    Return the most recent `limit` price rows for `symbol`,
    ordered oldest-first (chronological).
    """
    with SessionLocal() as session:
        stmt = (
            select(Price)
            .where(Price.symbol == symbol)
            .order_by(Price.ts.desc())
            .limit(limit)
        )
        rows = session.execute(stmt).scalars().all()
    return list(reversed(rows))


# ───────────────────────────────────────────────────────────────────
# 10. LINEAR REGRESSION MODEL
# ───────────────────────────────────────────────────────────────────

def train_model(symbol: str) -> Optional[LinearRegression]:
    """
    Fit a LinearRegression on the most recent TRAIN_WINDOW_POINTS prices.

    X = elapsed minutes since the oldest sample.
    Y = price.

    Returns the fitted model (also stored in CACHE) or None if insufficient data.
    """
    data = load_recent(symbol, TRAIN_WINDOW_POINTS)
    if len(data) < 10:
        return None

    t0 = data[0].ts
    X  = np.array([[(p.ts - t0).total_seconds() / 60.0] for p in data], dtype=float)
    y  = np.array([p.price for p in data], dtype=float)

    model = LinearRegression().fit(X, y)
    CACHE.models[symbol] = model

    buf = CACHE.buffer(symbol)
    buf.clear()
    for x_val, price in zip(X, y):
        buf.append((float(x_val[0]), float(price)))

    return model


def predict(symbol: str, horizon_minutes: float) -> Optional[PredictionOut]:
    """
    Return a price prediction `horizon_minutes` into the future.

    Reuses a cached model if available, otherwise trains one on demand.
    Returns None if there is insufficient data.
    """
    model  = CACHE.models.get(symbol) or train_model(symbol)
    recent = load_recent(symbol, 1)

    if not model or not recent:
        return None

    last = recent[-1]
    buf  = CACHE.buffer(symbol)

    if buf:
        t_last = buf[-1][0]
    else:
        window = load_recent(symbol, TRAIN_WINDOW_POINTS)
        if not window:
            return None
        t_last = (last.ts - window[0].ts).total_seconds() / 60.0

    predicted = float(model.predict(np.array([[t_last + horizon_minutes]]))[0])

    return PredictionOut(
        symbol=symbol,
        horizon_minutes=horizon_minutes,
        predicted_price=round(predicted, 2),
        trained_points=len(buf) if buf else TRAIN_WINDOW_POINTS,
        last_price=last.price,
        last_timestamp=last.ts,
    )


# ───────────────────────────────────────────────────────────────────
# 11. SCRAPER
# ───────────────────────────────────────────────────────────────────

async def scrape_once(symbol: str, cfg: Dict) -> None:
    """
    Fetch the latest price for one symbol and persist it.

    The try/except is intentionally split so that a model-training
    failure never hides a successful price write.
    """
    # --- Step 1: fetch price ---
    try:
        candle = await fetch_binance_price(symbol)
    except Exception as exc:
        print(f"[scrape] {symbol}: fetch error: {type(exc).__name__}: {exc}")
        return

    if not candle:
        print(f"[scrape] {symbol}: no data returned — see [HTTP] logs above.")
        return

    # --- Step 2: persist price ---
    try:
        store_price(symbol, candle["close"], candle["time"])
        print(f"[scrape] {symbol} = {candle['close']} @ {candle['time'].isoformat()}")
    except Exception as exc:
        print(f"[scrape] {symbol}: DB write error: {type(exc).__name__}: {exc}")
        return

    # --- Step 3: retrain model (non-fatal) ---
    try:
        train_model(symbol)
    except Exception as exc:
        print(f"[scrape] {symbol}: model training error (non-fatal): {type(exc).__name__}: {exc}")


async def scraper_loop() -> None:
    """
    Background task: scrape all tracked symbols on a fixed interval.

    Iterates sequentially with a small randomised stagger instead of
    asyncio.gather() to avoid bursting all requests simultaneously,
    which causes DNS failures and timeouts under load.
    """
    print(
        f"[scraper] starting — interval={SCRAPE_INTERVAL_SECONDS}s  "
        f"concurrency={BINANCE_CONCURRENCY}  endpoint={BINANCE_BASE}"
    )
    while True:
        for symbol, cfg in list(SOURCES.items()):
            await scrape_once(symbol, cfg)
            jitter = random.uniform(0, SCRAPE_STAGGER_SECONDS * 0.5)
            await asyncio.sleep(SCRAPE_STAGGER_SECONDS + jitter)
        await asyncio.sleep(SCRAPE_INTERVAL_SECONDS)


# ───────────────────────────────────────────────────────────────────
# 12. CACHE WARMING
# ───────────────────────────────────────────────────────────────────

async def _warm_coingecko_cache() -> None:
    """
    Pre-fill all CoinGecko / external caches on startup, then refresh
    them periodically.  Each warm helper is fully independent — a failure
    in one never cancels the others (return_exceptions=True).
    """
    await asyncio.sleep(2)  # let the event loop and scraper fully start

    # ── individual warm helpers ──────────────────────────────────────

    async def _warm_global() -> None:
        data = await _get(f"{COINGECKO_BASE}/global", headers=_coingecko_headers())
        if not data:
            return
        # FIX: removed duplicate variable assignment that was overwriting itself
        d   = data.get("data", {})
        mc  = d.get("total_market_cap", {})
        vol = d.get("total_volume", {})
        _cache_set("market_global", GlobalOut(
            total_market_cap_usd=mc.get("usd", 0),
            total_volume_usd=vol.get("usd", 0),
            bitcoin_dominance_pct=d.get("market_cap_percentage", {}).get("btc", 0),
            active_cryptocurrencies=d.get("active_cryptocurrencies", 0),
            markets=d.get("markets", 0),
            market_cap_change_pct_24h=d.get("market_cap_change_percentage_24h_usd", 0),
        ))

    async def _warm_overview() -> None:
        ids_str = ",".join(COINGECKO_IDS.values())
        data = await _get(
            f"{COINGECKO_BASE}/coins/markets",
            params={
                "vs_currency":             "usd",
                "ids":                     ids_str,
                "order":                   "market_cap_desc",
                "per_page":                20,
                "page":                    1,
                "sparkline":               False,
                "price_change_percentage": "24h",
            },
            headers=_coingecko_headers(),
        )
        if not data:
            return
        reverse_map = {v: k for k, v in COINGECKO_IDS.items()}
        result = [
            MarketOut(
                symbol=reverse_map.get(c["id"], c["symbol"].upper() + "USDT"),
                name=c.get("name", ""),
                current_price=c.get("current_price", 0),
                market_cap=c.get("market_cap"),
                market_cap_rank=c.get("market_cap_rank"),
                total_volume=c.get("total_volume"),
                high_24h=c.get("high_24h"),
                low_24h=c.get("low_24h"),
                price_change_24h=c.get("price_change_24h"),
                price_change_pct_24h=c.get("price_change_percentage_24h"),
                circulating_supply=c.get("circulating_supply"),
                ath=c.get("ath"),
                ath_change_pct=c.get("ath_change_percentage"),
                last_updated=c.get("last_updated"),
            )
            for c in data
        ]
        # FIX: removed dead-code `default_key` variable; use one correct cache key
        cache_key = "market_overview:" + ",".join(sorted(COINGECKO_IDS.keys()))
        _cache_set(cache_key, result)

    async def _warm_trending() -> None:
        data = await _get(f"{COINGECKO_BASE}/search/trending", headers=_coingecko_headers())
        if not data:
            return
        _cache_set("market_trending", [
            TrendingOut(
                id=i["item"].get("id", ""),
                name=i["item"].get("name", ""),
                symbol=i["item"].get("symbol", ""),
                market_cap_rank=i["item"].get("market_cap_rank"),
                thumb=i["item"].get("thumb"),
            )
            for i in data.get("coins", [])
        ])

    async def _warm_exchanges() -> None:
        data = await _get(
            f"{COINGECKO_BASE}/exchanges",
            params={"per_page": 10, "page": 1},
            headers=_coingecko_headers(),
        )
        if not data:
            return
        _cache_set("exchanges:10", [
            ExchangeOut(
                id=e.get("id", ""),
                name=e.get("name", ""),
                trust_score=e.get("trust_score"),
                trade_volume_24h_btc=e.get("trade_volume_24h_btc"),
                url=e.get("url"),
                country=e.get("country"),
            )
            for e in data
        ])

    async def _warm_fear_greed() -> None:
        data = await _get("https://api.alternative.me/fng/?limit=1")
        if not data or not data.get("data"):
            return
        e = data["data"][0]
        _cache_set("fear_greed", FearGreedOut(
            value=int(e["value"]),
            classification=e["value_classification"],
            timestamp=e["timestamp"],
        ))

    async def _warm_news() -> None:
        data = await _get(CRYPTOCOMPARE_NEWS, params={"lang": "EN", "sortOrder": "latest"})
        if not data or "Data" not in data:
            return
        raw   = data["Data"]
        items = list(raw.values()) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
        _cache_set("news", [
            NewsOut(
                title=item.get("title", ""),
                url=item.get("url", ""),
                source=item.get("source"),
                published=(
                    dt.datetime.fromtimestamp(item["published_on"], tz=dt.timezone.utc)
                    if item.get("published_on") else None
                ),
            )
            for item in items[:50]
        ])

    async def _warm_btc_network() -> None:
        # FIX: hashrate fetch added to the gather() call so all 4 requests
        # run concurrently instead of hashrate running sequentially after.
        tip, fees, mempool, hr_data = await asyncio.gather(
            _get(f"{BLOCKSTREAM_BASE}/blocks/tip/height"),
            _get("https://mempool.space/api/v1/fees/recommended"),
            _get("https://mempool.space/api/mempool"),
            _get("https://mempool.space/api/v1/mining/hashrate/3d"),
        )
        height: Optional[int] = None
        if isinstance(tip, int):
            height = tip
        elif isinstance(tip, str) and tip.strip().isdigit():
            height = int(tip.strip())

        hash_rate: Optional[float] = None
        if hr_data and "currentHashrate" in hr_data:
            hash_rate = hr_data["currentHashrate"] / 1e12

        _cache_set("btc_network", BtcNetworkOut(
            height=height,
            difficulty=None,
            hash_rate_th_s=hash_rate,
            mempool_tx_count=mempool.get("count") if mempool else None,
            mempool_bytes=mempool.get("vsize")    if mempool else None,
            fee_fastest_sat=fees.get("fastestFee")   if fees else None,
            fee_halfhour_sat=fees.get("halfHourFee") if fees else None,
            fee_hour_sat=fees.get("hourFee")         if fees else None,
        ))

    async def _warm_eth_gas() -> None:
        data = await _get(BLOCKNATIVE)
        if not data:
            return
        block  = data.get("blockPrices", [{}])[0]
        prices = block.get("estimatedPrices", [{}])
        _cache_set("eth_gas", EthGasOut(
            safe_gas_gwei=prices[2].get("maxFeePerGas")    if len(prices) > 2 else None,
            propose_gas_gwei=prices[1].get("maxFeePerGas") if len(prices) > 1 else None,
            fast_gas_gwei=prices[0].get("maxFeePerGas")    if prices else None,
            base_fee_gwei=block.get("baseFeePerGas"),
            eth_price_usd=None,
        ))

    async def _fetch_and_cache_all() -> None:
        print("[cache] warming external caches …")
        await asyncio.gather(
            _warm_global(),
            _warm_overview(),
            _warm_trending(),
            _warm_exchanges(),
            _warm_fear_greed(),
            _warm_news(),
            _warm_btc_network(),
            _warm_eth_gas(),
            return_exceptions=True,
        )
        print("[cache] external caches warmed.")

    # ── initial warm, then periodic refresh loop ─────────────────────
    await _fetch_and_cache_all()

    while True:
        await asyncio.sleep(55)
        await asyncio.gather(_warm_global(), _warm_overview(), return_exceptions=True)
        await asyncio.sleep(5)
        await asyncio.gather(_warm_fear_greed(), _warm_btc_network(), _warm_eth_gas(), return_exceptions=True)
        await asyncio.sleep(55)
        await asyncio.gather(_warm_trending(), _warm_exchanges(), _warm_news(), return_exceptions=True)
        await asyncio.sleep(5)


# ───────────────────────────────────────────────────────────────────
# 13. APPLICATION LIFESPAN
# ───────────────────────────────────────────────────────────────────

_BINANCE_SEM: asyncio.Semaphore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background scraper, create the Binance semaphore, warm caches."""
    global _BINANCE_SEM
    _BINANCE_SEM = asyncio.Semaphore(BINANCE_CONCURRENCY)
    asyncio.create_task(scraper_loop())
    asyncio.create_task(_warm_coingecko_cache())
    yield


# ───────────────────────────────────────────────────────────────────
# 14. APP INIT
# ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CryptoPulse — Blockchain Intelligence API",
    version="1.0.0",
    description=(
        "Real-time crypto prices, market data, on-chain metrics, "
        "news, and linear-regression price predictions."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────────────────────────────────────────────────────────────────
# 15. ROUTES — SYSTEM
# ───────────────────────────────────────────────────────────────────

@app.get("/health", summary="API health check and DB row counts")
async def health():
    with SessionLocal() as session:
        total = session.execute(select(func.count(Price.id))).scalar()
        per_symbol = dict(
            session.execute(
                select(Price.symbol, func.count(Price.id)).group_by(Price.symbol)
            ).all()
        )
    return {
        "status":                  "ok",
        "tracked_symbols":         list(SOURCES.keys()),
        "rows_total":              total,
        "rows_per_symbol":         per_symbol,
        "scrape_interval_seconds": SCRAPE_INTERVAL_SECONDS,
        "binance_endpoint":        BINANCE_BASE,
        "binance_region":          _REGION,
        "coingecko_key_set":       bool(COINGECKO_KEY),
    }


# ───────────────────────────────────────────────────────────────────
# 16. ROUTES — SYMBOL MANAGEMENT
# ───────────────────────────────────────────────────────────────────

@app.get("/symbols/list", summary="List all currently tracked symbols")
async def list_symbols():
    return {"symbols": [{"symbol": s, **cfg} for s, cfg in SOURCES.items()]}


@app.post("/symbols/add", summary="Start tracking a new symbol")
async def add_symbol(cfg: SymbolConfig, background_tasks: BackgroundTasks):
    sym = cfg.symbol.upper()
    if cfg.provider != "binance":
        raise HTTPException(status_code=400, detail="Only 'binance' provider is supported.")
    SOURCES[sym] = {"provider": "binance", "interval": cfg.interval}
    background_tasks.add_task(scrape_once, sym, SOURCES[sym])
    return {"added": sym, "provider": "binance", "interval": cfg.interval}


@app.post("/symbols/remove", summary="Stop tracking a symbol")
async def remove_symbol(symbol: str):
    sym = symbol.upper()
    if sym not in SOURCES:
        raise HTTPException(status_code=404, detail=f"{sym} is not currently tracked.")
    del SOURCES[sym]
    CACHE.models.pop(sym, None)
    CACHE.buffers.pop(sym, None)
    return {"removed": sym}


# ───────────────────────────────────────────────────────────────────
# 17. ROUTES — PRICE DATA
# ───────────────────────────────────────────────────────────────────

@app.get(
    "/price/{symbol}",
    response_model=PriceOut,
    summary="Latest stored price for a symbol",
)
async def get_price(symbol: str):
    rows = load_recent(symbol.upper(), 1)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No price data for {symbol}. "
                "Check /health → rows_per_symbol and your terminal logs."
            ),
        )
    row = rows[-1]
    return PriceOut(symbol=row.symbol, price=row.price, ts=row.ts)


@app.get(
    "/history/{symbol}",
    response_model=HistoryOut,
    summary="Recent price history for a symbol",
)
async def get_history(
    symbol: str,
    points: int = Query(200, ge=1, le=2000, description="Number of data points to return"),
):
    rows = load_recent(symbol.upper(), points)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No history for {symbol}.")
    return HistoryOut(
        symbol=symbol,
        points=len(rows),
        start=rows[0].ts,
        end=rows[-1].ts,
        data=[PriceOut(symbol=r.symbol, price=r.price, ts=r.ts) for r in rows],
    )


@app.get(
    "/predict/{symbol}",
    response_model=PredictionOut,
    summary="Linear-regression price prediction",
)
async def get_prediction(
    symbol: str,
    horizon_minutes: float = Query(
        PREDICT_HORIZON_MINUTES,
        gt=0,
        description="How many minutes ahead to predict",
    ),
):
    result = predict(symbol.upper(), horizon_minutes)
    if not result:
        raise HTTPException(
            status_code=400,
            detail="Insufficient data. At least 10 stored prices are required.",
        )
    return result


@app.get(
    "/candles/{symbol}",
    response_model=List[CandleOut],
    summary="OHLCV candlestick data (TradingView-compatible)",
)
async def get_candles(
    symbol:   str,
    interval: str = Query("1m",  description="Candle interval: 1m 5m 15m 1h 4h 1d"),
    limit:    int  = Query(100,  ge=1, le=1000, description="Number of candles"),
):
    bs   = _binance_symbol(symbol.upper())
    data = await _get(
        f"{BINANCE_BASE}/api/v3/klines",
        params={"symbol": bs, "interval": interval, "limit": limit},
    )
    if not isinstance(data, list) or not data:
        raise HTTPException(
            status_code=502,
            detail=f"Could not fetch candles from Binance for {bs}.",
        )

    candles: List[CandleOut] = []
    for row in data:
        if not isinstance(row, list) or len(row) < 6:
            continue
        try:
            candles.append(CandleOut(
                ts=dt.datetime.fromtimestamp(row[0] / 1000, tz=dt.timezone.utc),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            ))
        except Exception as exc:
            print(f"[candles] parse error on row {row}: {exc}")

    if not candles:
        raise HTTPException(
            status_code=502,
            detail="Candle data was returned but could not be parsed.",
        )
    return candles


@app.get(
    "/compare",
    summary="Side-by-side price and prediction comparison (up to 6 symbols)",
)
async def compare_symbols(
    symbols: str = Query(
        "BTCUSDT,ETHUSDT",
        description="Comma-separated list of symbols (max 6)",
    ),
):
    sym_list = [s.strip().upper() for s in symbols.split(",")][:6]
    results  = {}
    for sym in sym_list:
        rows = load_recent(sym, 1)
        pred = predict(sym, PREDICT_HORIZON_MINUTES)
        results[sym] = {
            "last_price":      rows[-1].price if rows else None,
            "last_ts":         rows[-1].ts.isoformat() if rows else None,
            "predicted_price": pred.predicted_price if pred else None,
            "pred_change_pct": (
                round(
                    (pred.predicted_price - pred.last_price) / pred.last_price * 100,
                    3,
                )
                if pred else None
            ),
        }
    return results


# ───────────────────────────────────────────────────────────────────
# 18. ROUTES — MARKET DATA  (CoinGecko)
# ───────────────────────────────────────────────────────────────────

@app.get(
    "/market/overview",
    response_model=List[MarketOut],
    summary="Full market data: price, market cap, volume, ATH, supply",
)
async def market_overview(
    symbols: str = Query(
        "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT",
        description="Comma-separated trading-pair symbols",
    ),
):
    cache_key = "market_overview:" + ",".join(sorted(symbols.upper().split(",")))
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    requested = [s.strip().upper() for s in symbols.split(",")]
    ids_map   = {k: v for k, v in COINGECKO_IDS.items() if k in requested}
    if not ids_map:
        raise HTTPException(status_code=400, detail="No recognised symbols in request.")

    data = await _get(
        f"{COINGECKO_BASE}/coins/markets",
        params={
            "vs_currency":             "usd",
            "ids":                     ",".join(ids_map.values()),
            "order":                   "market_cap_desc",
            "per_page":                20,
            "page":                    1,
            "sparkline":               False,
            "price_change_percentage": "24h",
        },
        headers=_coingecko_headers(),
    )
    if not data:
        # FIX: return fallback stubs instead of 502 when CoinGecko is unreachable
        print("[market/overview] CoinGecko unreachable — returning empty list.")
        return []

    reverse_map = {v: k for k, v in ids_map.items()}
    result = [
        MarketOut(
            symbol=reverse_map.get(coin["id"], coin["symbol"].upper() + "USDT"),
            name=coin.get("name", ""),
            current_price=coin.get("current_price", 0),
            market_cap=coin.get("market_cap"),
            market_cap_rank=coin.get("market_cap_rank"),
            total_volume=coin.get("total_volume"),
            high_24h=coin.get("high_24h"),
            low_24h=coin.get("low_24h"),
            price_change_24h=coin.get("price_change_24h"),
            price_change_pct_24h=coin.get("price_change_percentage_24h"),
            circulating_supply=coin.get("circulating_supply"),
            ath=coin.get("ath"),
            ath_change_pct=coin.get("ath_change_percentage"),
            last_updated=coin.get("last_updated"),
        )
        for coin in data
    ]
    _cache_set(cache_key, result)
    return result


@app.get(
    "/market/global",
    response_model=GlobalOut,
    summary="Global crypto market stats: total cap, BTC dominance, active coins",
)
async def market_global():
    cached = _cache_get("market_global")
    if cached is not None:
        return cached

    data = await _get(f"{COINGECKO_BASE}/global", headers=_coingecko_headers())
    if not data:
        # FIX: return zeroed-out fallback instead of 502
        print("[market/global] CoinGecko unreachable — returning fallback.")
        return FALLBACK_GLOBAL

    d   = data.get("data", {})
    mc  = d.get("total_market_cap", {})
    vol = d.get("total_volume", {})
    result = GlobalOut(
        total_market_cap_usd=mc.get("usd", 0),
        total_volume_usd=vol.get("usd", 0),
        bitcoin_dominance_pct=d.get("market_cap_percentage", {}).get("btc", 0),
        active_cryptocurrencies=d.get("active_cryptocurrencies", 0),
        markets=d.get("markets", 0),
        market_cap_change_pct_24h=d.get("market_cap_change_percentage_24h_usd", 0),
    )
    _cache_set("market_global", result)
    return result


@app.get(
    "/market/fear-greed",
    response_model=FearGreedOut,
    summary="Crypto Fear & Greed Index (alternative.me)",
)
async def fear_greed():
    cached = _cache_get("fear_greed")
    if cached is not None:
        return cached

    data = await _get("https://api.alternative.me/fng/?limit=1")
    if not data or not data.get("data"):
        # FIX: return neutral fallback instead of 502
        print("[market/fear-greed] alternative.me unreachable — returning fallback.")
        return FALLBACK_FEAR_GREED

    entry = data["data"][0]
    result = FearGreedOut(
        value=int(entry["value"]),
        classification=entry["value_classification"],
        timestamp=entry["timestamp"],
    )
    _cache_set("fear_greed", result)
    return result


@app.get(
    "/market/trending",
    response_model=List[TrendingOut],
    summary="Top 7 trending coins on CoinGecko",
)
async def trending_coins():
    cached = _cache_get("market_trending")
    if cached is not None:
        return cached

    data = await _get(f"{COINGECKO_BASE}/search/trending", headers=_coingecko_headers())
    if not data:
        # FIX: return empty list instead of 502
        print("[market/trending] CoinGecko unreachable — returning empty list.")
        return []

    result = [
        TrendingOut(
            id=item["item"].get("id", ""),
            name=item["item"].get("name", ""),
            symbol=item["item"].get("symbol", ""),
            market_cap_rank=item["item"].get("market_cap_rank"),
            thumb=item["item"].get("thumb"),
        )
        for item in data.get("coins", [])
    ]
    _cache_set("market_trending", result)
    return result


@app.get(
    "/exchanges",
    response_model=List[ExchangeOut],
    summary="Top crypto exchanges ranked by 24h BTC trading volume",
)
async def top_exchanges(
    limit: int = Query(10, ge=1, le=50, description="Number of exchanges to return"),
):
    cache_key = f"exchanges:{limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    data = await _get(
        f"{COINGECKO_BASE}/exchanges",
        params={"per_page": limit, "page": 1},
        headers=_coingecko_headers(),
    )
    if not data:
        # FIX: return empty list instead of 502
        print("[exchanges] CoinGecko unreachable — returning empty list.")
        return []

    result = [
        ExchangeOut(
            id=e.get("id", ""),
            name=e.get("name", ""),
            trust_score=e.get("trust_score"),
            trade_volume_24h_btc=e.get("trade_volume_24h_btc"),
            url=e.get("url"),
            country=e.get("country"),
        )
        for e in data
    ]
    _cache_set(cache_key, result)
    return result


@app.get(
    "/coin/{coin_id}",
    summary="Deep detail for a single coin (description, links, social, dev stats)",
)
async def coin_detail(coin_id: str):
    data = await _get(
        f"{COINGECKO_BASE}/coins/{coin_id}",
        params={
            "localization":   False,
            "tickers":        False,
            "market_data":    True,
            "community_data": True,
            "developer_data": True,
            "sparkline":      False,
        },
        headers=_coingecko_headers(),
    )
    if not data:
        raise HTTPException(status_code=404, detail=f"Coin '{coin_id}' not found or CoinGecko unreachable.")

    md  = data.get("market_data",    {})
    cd  = data.get("community_data", {})
    dev = data.get("developer_data", {})

    return {
        "id":                     data.get("id"),
        "name":                   data.get("name"),
        "symbol":                 data.get("symbol", "").upper(),
        "description":            data.get("description", {}).get("en", "")[:500],
        "homepage":               (data.get("links", {}).get("homepage") or [""])[0],
        "github":                 (data.get("links", {}).get("repos_url", {}).get("github") or [None])[0],
        "twitter":                data.get("links", {}).get("twitter_screen_name"),
        "reddit":                 data.get("links", {}).get("subreddit_url"),
        "price_usd":              md.get("current_price", {}).get("usd"),
        "market_cap_usd":         md.get("market_cap",    {}).get("usd"),
        "ath_usd":                md.get("ath",            {}).get("usd"),
        "price_change_pct_7d":    md.get("price_change_percentage_7d"),
        "price_change_pct_30d":   md.get("price_change_percentage_30d"),
        "reddit_subscribers":     cd.get("reddit_subscribers"),
        "twitter_followers":      cd.get("twitter_followers"),
        "github_stars":           dev.get("stars"),
        "github_commits_4w":      dev.get("commit_count_4_weeks"),
        "sentiment_votes_up_pct": data.get("sentiment_votes_up_percentage"),
    }


# ───────────────────────────────────────────────────────────────────
# 19. ROUTES — BLOCKCHAIN
# ───────────────────────────────────────────────────────────────────

@app.get(
    "/blockchain/bitcoin/network",
    response_model=BtcNetworkOut,
    summary="Bitcoin on-chain stats: block height, hashrate, mempool, fee estimates",
)
async def bitcoin_network():
    cached = _cache_get("btc_network")
    if cached is not None:
        return cached

    # FIX: hashrate is now fetched concurrently with the other 3 requests
    tip_data, fees_data, mempool_data, hr_data = await asyncio.gather(
        _get(f"{BLOCKSTREAM_BASE}/blocks/tip/height"),
        _get("https://mempool.space/api/v1/fees/recommended"),
        _get("https://mempool.space/api/mempool"),
        _get("https://mempool.space/api/v1/mining/hashrate/3d"),
    )

    height: Optional[int] = None
    if isinstance(tip_data, int):
        height = tip_data
    elif isinstance(tip_data, str) and tip_data.strip().isdigit():
        height = int(tip_data.strip())

    hash_rate: Optional[float] = None
    if hr_data and "currentHashrate" in hr_data:
        hash_rate = hr_data["currentHashrate"] / 1e12

    result = BtcNetworkOut(
        height=height,
        difficulty=None,
        hash_rate_th_s=hash_rate,
        mempool_tx_count=mempool_data.get("count")    if mempool_data else None,
        mempool_bytes=mempool_data.get("vsize")       if mempool_data else None,
        fee_fastest_sat=fees_data.get("fastestFee")   if fees_data else None,
        fee_halfhour_sat=fees_data.get("halfHourFee") if fees_data else None,
        fee_hour_sat=fees_data.get("hourFee")         if fees_data else None,
    )
    _cache_set("btc_network", result)
    return result


@app.get(
    "/blockchain/ethereum/gas",
    response_model=EthGasOut,
    summary="Ethereum gas price tracker (Etherscan or Blocknative fallback)",
)
async def ethereum_gas():
    cached = _cache_get("eth_gas")
    if cached is not None:
        return cached

    # Try Etherscan first if an API key is configured
    if ETHERSCAN_KEY:
        data = await _get(
            ETHERSCAN_BASE,
            params={
                "module": "gastracker",
                "action": "gasoracle",
                "apikey": ETHERSCAN_KEY,
            },
        )
        if data and data.get("status") == "1":
            r = data["result"]
            result = EthGasOut(
                safe_gas_gwei=float(r.get("SafeGasPrice",    0)),
                propose_gas_gwei=float(r.get("ProposeGasPrice", 0)),
                fast_gas_gwei=float(r.get("FastGasPrice",    0)),
                base_fee_gwei=float(r.get("suggestBaseFee",  0)),
                eth_price_usd=float(r.get("UsdPrice",        0)),
            )
            _cache_set("eth_gas", result)
            return result

    # Blocknative fallback
    data = await _get(BLOCKNATIVE)
    if data:
        block  = data.get("blockPrices", [{}])[0]
        prices = block.get("estimatedPrices", [{}])
        result = EthGasOut(
            safe_gas_gwei=prices[2].get("maxFeePerGas")    if len(prices) > 2 else None,
            propose_gas_gwei=prices[1].get("maxFeePerGas") if len(prices) > 1 else None,
            fast_gas_gwei=prices[0].get("maxFeePerGas")    if prices else None,
            base_fee_gwei=block.get("baseFeePerGas"),
            eth_price_usd=None,
        )
        _cache_set("eth_gas", result)
        return result

    # FIX: return all-None fallback instead of 502 when gas APIs are unreachable
    print("[blockchain/ethereum/gas] All gas APIs unreachable — returning fallback.")
    return FALLBACK_ETH_GAS


# ───────────────────────────────────────────────────────────────────
# 20. ROUTES — NEWS
# ───────────────────────────────────────────────────────────────────

@app.get(
    "/news",
    response_model=List[NewsOut],
    summary="Latest cryptocurrency news headlines (CryptoCompare)",
)
async def get_news(
    limit: int = Query(10, ge=1, le=50, description="Number of headlines to return"),
):
    cached = _cache_get("news")
    if cached is not None:
        return cached[:limit]

    data = await _get(
        CRYPTOCOMPARE_NEWS,
        params={"lang": "EN", "sortOrder": "latest"},
    )
    if not data or "Data" not in data:
        # FIX: return empty list instead of 502 when CryptoCompare is unreachable
        print("[news] CryptoCompare unreachable — returning empty list.")
        return []

    # CryptoCompare occasionally returns Data as a dict keyed by article ID
    # instead of a plain list.  Normalise to list so slicing always works.
    raw = data["Data"]
    if isinstance(raw, dict):
        items = list(raw.values())
    elif isinstance(raw, list):
        items = raw
    else:
        return []

    result = [
        NewsOut(
            title=item.get("title", ""),
            url=item.get("url", ""),
            source=item.get("source"),
            published=(
                dt.datetime.fromtimestamp(item["published_on"], tz=dt.timezone.utc)
                if item.get("published_on") else None
            ),
        )
        for item in items[:50]
    ]
    _cache_set("news", result)
    return result[:limit]


# ───────────────────────────────────────────────────────────────────
# 21. ROUTES — PORTFOLIO
# ───────────────────────────────────────────────────────────────────

@app.post(
    "/portfolio/value",
    summary="Calculate current USD value of a crypto portfolio",
)
async def portfolio_value(portfolio: PortfolioIn):
    """
    Accepts a list of {symbol, quantity} holdings and returns the
    total portfolio value plus a per-asset breakdown using the
    most recently scraped prices.
    """
    total     = 0.0
    breakdown = []

    for holding in portfolio.holdings:
        sym   = holding.symbol.upper()
        rows  = load_recent(sym, 1)
        price = rows[-1].price if rows else None
        value = round(price * holding.quantity, 2) if price is not None else None

        if value is not None:
            total += value

        breakdown.append({
            "symbol":    sym,
            "quantity":  holding.quantity,
            "price_usd": price,
            "value_usd": value,
        })

    return {"total_usd": round(total, 2), "breakdown": breakdown}