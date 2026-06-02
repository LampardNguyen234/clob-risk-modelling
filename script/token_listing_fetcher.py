"""
Token listing data fetcher.

Fetches and caches the four API responses required per token for the
RISEx token listing framework. All CMC calls use the undocumented
data-api endpoints (no auth required). CoinGecko uses the public free tier.

Cache lives at notes/token_listing/cache/{SYMBOL}_{source}.json.
Pass refresh=True to bypass the cache and re-fetch all endpoints.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

import requests
import yaml

CMC_DETAIL_URL        = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/detail"
CMC_HISTORICAL_URL    = "https://api.coinmarketcap.com/data-api/v3.1/cryptocurrency/historical"
CMC_MARKET_PAIRS_URL  = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/market-pairs/latest"
COINGECKO_TICKERS_URL = "https://api.coingecko.com/api/v3/coins/{coingecko_id}/tickers"
HYPERLIQUID_INFO_URL  = "https://api.hyperliquid.xyz/info"

# Resolved relative to cwd at call time; notebooks run from their own directory.
CACHE_DIR = Path("cache")

REQUIRED_FIELDS = {"symbol", "cmc_id", "cmc_slug", "coingecko_id", "safety_score"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict = {
    "fdmc_min":               200_000_000,
    "exchange_score_min":     6,
    "safety_pass_min":        75,
    "safety_conditional_min": 60,
    "metric_refs": {
        "spot_depth":    5_000_000,
        "es95":          0.05,
        "book_spread":   0.0005,
        "spot_volume":   500_000_000,
        "concentration": 0.80,
        "parkinson_vol": 0.7642,  # 4% x sqrt(365); IMR_base(Prime, 25x)
    },
    "metric_vmax": {
        "es95":          0.20,  # 20% = empirical ceiling for single-day tail loss
        "parkinson_vol": 3.00,  # 300% ann.; empirical ceiling (LUNA/DOGE 2021 90d)
    },
    "metric_weights": {
        "spot_depth":    30,
        "es95":          25,
        "book_spread":   15,
        "spot_volume":   10,
        "concentration": 10,
        "parkinson_vol": 10,
    },
    "tier_scores": {
        "Prime":     90,
        "Major":     80,
        "Mid-Cap":   55,
        "Small-Cap": 40,
        "Micro-Cap": 15,
    },
    "listing_age_min_days": 180,  # tokens with fewer OHLCV days receive Micro-Cap ceiling
}


DEFAULT_EXCHANGE_TIERS: dict = {
    "tier1": {"binance", "coinbase-exchange", "okx", "bybit", "kraken", "hyperliquid"},
    "tier2": {"gate", "bitget", "kucoin", "bitstamp", "htx", "mexc", "bingx"},
}


def load_exchange_tiers(config_path: str | Path) -> tuple[set[str], set[str]]:
    """
    Load exchange tier membership from config. Falls back to DEFAULT_EXCHANGE_TIERS
    for any tier absent in the config.
    Returns (tier1_slugs, tier2_slugs).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    raw = cfg.get("exchange_tiers", {})
    tier1 = set(raw["tier1"]) if "tier1" in raw else DEFAULT_EXCHANGE_TIERS["tier1"]
    tier2 = set(raw["tier2"]) if "tier2" in raw else DEFAULT_EXCHANGE_TIERS["tier2"]
    return tier1, tier2


def load_config(config_path: str | Path) -> list[dict]:
    """Load token list from YAML. Raises ValueError for missing required fields."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    tokens = cfg.get("tokens", [])
    for token in tokens:
        missing = REQUIRED_FIELDS - token.keys()
        if missing:
            raise ValueError(f"Token {token.get('symbol', '?')} missing fields: {missing}")
    return tokens


def load_thresholds(config_path: str | Path) -> dict:
    """
    Load thresholds from YAML config. Any key absent in the config falls back
    to DEFAULT_THRESHOLDS, so existing notebooks keep working unchanged.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    raw = cfg.get("thresholds", {})

    # Keys whose values are dicts — merge field-by-field rather than replacing wholesale.
    _dict_keys = {"metric_refs", "metric_vmax", "metric_weights", "tier_scores"}

    merged: dict = dict(DEFAULT_THRESHOLDS)
    for key, default in DEFAULT_THRESHOLDS.items():
        if key in _dict_keys:
            merged[key] = {**default, **raw.get(key, {})}
        elif key in raw:
            merged[key] = raw[key]
    return merged


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cache_path(symbol: str, source: str) -> Path:
    return CACHE_DIR / f"{symbol}_{source}.json"


def _load_cache(symbol: str, source: str) -> dict | None:
    p = _cache_path(symbol, source)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _save_cache(data: dict, symbol: str, source: str) -> None:
    p = _cache_path(symbol, source)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)


def _get(session: requests.Session, url: str, params: dict) -> dict:
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Per-endpoint fetchers
# ---------------------------------------------------------------------------

def fetch_cmc_detail(cmc_id: int, symbol: str, session: requests.Session,
                     refresh: bool = False) -> dict:
    """
    GET CMC detail?id={cmc_id}
    Fields used: statistics.fullyDilutedMarketCap, statistics.volume30d,
                 holders.holderList[:10].share
    """
    if not refresh:
        cached = _load_cache(symbol, "cmc_detail")
        if cached is not None:
            return cached
    data = _get(session, CMC_DETAIL_URL, {"id": cmc_id})
    _save_cache(data, symbol, "cmc_detail")
    return data


def fetch_cmc_historical(cmc_id: int, symbol: str, session: requests.Session,
                         refresh: bool = False) -> dict:
    """
    GET CMC historical?id={cmc_id}&interval=24h
    Fields used: data.quotes[].quote.{open,high,low,close}, timeOpen, timeClose
    Returns all available daily candles; caller filters to complete candles
    (timeClose < today 00:00 UTC).
    """
    if not refresh:
        cached = _load_cache(symbol, "cmc_historical")
        if cached is not None:
            return cached
    data = _get(session, CMC_HISTORICAL_URL, {"id": cmc_id, "interval": "24h"})
    _save_cache(data, symbol, "cmc_historical")
    return data


def fetch_cmc_market_pairs(cmc_slug: str, symbol: str, session: requests.Session,
                           refresh: bool = False) -> dict:
    """
    GET CMC market-pairs?slug={cmc_slug}&start=1&limit=50&category=spot
    Fields used: marketPairs[].{exchangeSlug,depthUsdNegativeTwo,depthUsdPositiveTwo,
                 volumeUsd,outlierDetected,volumeExcluded,marketReputation,quoteSymbol}
    """
    if not refresh:
        cached = _load_cache(symbol, "cmc_market_pairs")
        if cached is not None:
            return cached
    data = _get(session, CMC_MARKET_PAIRS_URL, {
        "slug": cmc_slug, "start": 1, "limit": 100, "category": "spot",
        "sort": "cmc_rank_advanced",
    })
    _save_cache(data, symbol, "cmc_market_pairs")
    return data


def fetch_hyperliquid_depth(symbol: str, session: requests.Session,
                            refresh: bool = False) -> dict:
    """
    Fetch ±2% spot depth from Hyperliquid's own L2 book API.

    Used as the authoritative depth source for the "hyperliquid" exchange slug:
    both CMC and CoinGecko misreport Hyperliquid on-chain book depth by 100x or more.

    Hyperliquid L2Book: levels[0] = bids (descending), levels[1] = asks (ascending).
    Mid is derived from best bid/ask; 2% depth is summed within [mid*0.98, mid*1.02].

    Returns {"depth_bid": float, "depth_ask": float} or {} if the coin has no
    Hyperliquid spot market or the request fails.
    """
    if not refresh:
        cached = _load_cache(symbol, "hl_depth")
        if cached is not None:
            return cached
    try:
        resp = session.post(
            HYPERLIQUID_INFO_URL,
            # nSigFigs=3: bucket ≈ 0.1% of price; 20 levels × 0.1% = 2% per side,
            # enough to cover the full ±2% depth range at any asset price.
            json={"type": "l2Book", "coin": symbol, "nSigFigs": 3},
            timeout=10,
        )
        resp.raise_for_status()
        levels = resp.json().get("levels", [])
        if len(levels) < 2 or not levels[0] or not levels[1]:
            _save_cache({}, symbol, "hl_depth")
            return {}
        bid_levels, ask_levels = levels[0], levels[1]
        best_bid = float(bid_levels[0]["px"])
        best_ask = float(ask_levels[0]["px"])
        mid = (best_bid + best_ask) / 2
        lo, hi = mid * 0.98, mid * 1.02
        bid_usd = sum(float(l["px"]) * float(l["sz"]) for l in bid_levels if float(l["px"]) >= lo)
        ask_usd = sum(float(l["px"]) * float(l["sz"]) for l in ask_levels if float(l["px"]) <= hi)
        result = {"depth_bid": bid_usd, "depth_ask": ask_usd}
        _save_cache(result, symbol, "hl_depth")
        return result
    except Exception as exc:
        warnings.warn(f"{symbol}: Hyperliquid depth fetch failed ({exc}); HL depth skipped")
        _save_cache({}, symbol, "hl_depth")
        return {}


def fetch_coingecko_tickers(coingecko_id: str, symbol: str, session: requests.Session,
                             refresh: bool = False) -> dict:
    """
    GET CoinGecko tickers?depth=true
    Field used: tickers[].bid_ask_spread_percentage, market.identifier, target
    Returns {} on any request failure (score falls back to 0, soft flag emitted).
    Retries up to 5 times on 429 with exponential backoff (8s, 16s, 32s, 64s, 128s).
    """
    if not refresh:
        cached = _load_cache(symbol, "coingecko_tickers")
        if cached is not None:
            return cached
    url = COINGECKO_TICKERS_URL.format(coingecko_id=coingecko_id) + "?depth=true"
    for attempt in range(5):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 429:
                wait = 60 * (2 ** attempt)
                warnings.warn(f"{symbol}: CoinGecko 429, retrying in {wait}s (attempt {attempt+1}/5)")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            _save_cache(data, symbol, "coingecko_tickers")
            return data
        except Exception as exc:
            if attempt == 4:
                warnings.warn(f"{symbol}: CoinGecko fetch failed ({exc}); spread will score 0")
                return {}
            time.sleep(60 * (2 ** attempt))
    return {}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def fetch_all(token: dict, refresh: bool = False) -> dict[str, Any]:
    """
    Fetch all four API responses for one token.

    Sleeps 4s before each live CoinGecko call to stay within the free-tier
    rate limit (~15 req/min conservative). 429s trigger exponential backoff.

    Returns:
        {
            "cmc_detail":       ...,
            "cmc_historical":   ...,
            "cmc_market_pairs": ...,
            "coingecko_tickers": ...,
        }
    """
    sym           = token["symbol"]
    cmc_id        = token["cmc_id"]
    cmc_slug      = token["cmc_slug"]
    coingecko_id  = token["coingecko_id"]

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    detail       = fetch_cmc_detail(cmc_id, sym, session, refresh)
    historical   = fetch_cmc_historical(cmc_id, sym, session, refresh)
    market_pairs = fetch_cmc_market_pairs(cmc_slug, sym, session, refresh)

    # Sleep before live CG requests to respect ~15 req/min free-tier limit.
    cg_cached = not refresh and _cache_path(sym, "coingecko_tickers").exists()
    if not cg_cached:
        time.sleep(4)
    tickers  = fetch_coingecko_tickers(coingecko_id, sym, session, refresh)
    hl_depth = fetch_hyperliquid_depth(sym, session, refresh)

    return {
        "cmc_detail":        detail,
        "cmc_historical":    historical,
        "cmc_market_pairs":  market_pairs,
        "coingecko_tickers": tickers,
        "hl_depth":          hl_depth,
    }
