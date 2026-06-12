"""
Listing radar data fetcher.

Fetches and caches all data sources required for the RISEx listing radar:
  - Binance USD-M Futures (fapi): ticker, OI history, klines, premiumIndex
  - Binance bapi: marketing/symbol/list (spot universe + tags)
  - Binance apex: AI sentiment and technical scores
  - CMC: unified-trending listing (unofficial, no auth)
  - CoinGecko: search/trending
  - DefiLlama: protocols
  - RISEx: active markets (hard gate)

Cache layout under CACHE_DIR (default Path("cache/radar"), relative to cwd):
  *.json              — global sources (ticker, spot list, sentiment, etc.)
  oi-hist/*.json      — per-symbol OI history (binance_oi_hist_{SYM}.json)
  klines/futures/     — per-symbol Binance futures daily klines
  klines/spot/        — per-symbol Binance spot daily klines
  premium/*.json      — per-symbol premiumIndex snapshots
Notebooks run from notes/token_listing/, so cache lands there.

Rate limits handled:
  Binance fapi  — weight-based (2400/min IP limit). X-MBX-USED-WEIGHT-1M header
                  monitored; sleeps 30 s if used weight > 1800. 429/418 trigger
                  exponential backoff. 418 (IP ban) raises immediately.
  Binance bapi/apex — unofficial, single bulk calls per TTL, no per-symbol calls.
                      5xx handled gracefully (returns empty, logs warning).
  CoinGecko     — ~30 req/min free tier. fetch_cg_search sleeps 4 s between calls.
                  429 triggers exponential backoff (same as existing token_listing_fetcher).
  CMC trending  — single call; retries once after 5 s on 500.
  DefiLlama     — no documented limit; single large call cached 30 min.
"""

from __future__ import annotations

import json
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Endpoint URLs
# ---------------------------------------------------------------------------

RISEX_MARKETS_URL       = "https://api.rise.trade/v1/markets"

_FAPI                   = "https://fapi.binance.com"
BINANCE_FAPI_TICKER     = f"{_FAPI}/fapi/v1/ticker/24hr"
BINANCE_FAPI_OI_HIST    = f"{_FAPI}/futures/data/openInterestHist"
BINANCE_FAPI_KLINES     = f"{_FAPI}/fapi/v1/klines"
BINANCE_FAPI_PREMIUM    = f"{_FAPI}/fapi/v1/premiumIndex"
BINANCE_FAPI_FUNDING    = f"{_FAPI}/fapi/v1/fundingRate"

_SPOT                   = "https://api.binance.com"
BINANCE_SPOT_KLINES_URL = f"{_SPOT}/api/v3/klines"

_BAPI                   = "https://www.binance.com"
BINANCE_SPOT_LIST_URL   = f"{_BAPI}/bapi/composite/v1/public/marketing/symbol/list"
BINANCE_APEX_URL        = f"{_BAPI}/bapi/apex/v1/friendly/apex/web/opportunity/assets"
BINANCE_APEX_DETAIL_URL = f"{_BAPI}/bapi/apex/v1/friendly/apex/web/opportunity/asset-details"
BINANCE_ALPHA_TICKER_URL  = f"{_BAPI}/bapi/defi/v1/public/alpha-trade/aggTicker24"
BINANCE_ALPHA_KLINES_URL  = f"{_BAPI}/bapi/defi/v1/public/alpha-trade/agg-klines"

CMC_TRENDING_URL        = "https://api.coinmarketcap.com/data-api/v3/unified-trending/listing"
CG_TRENDING_URL         = "https://api.coingecko.com/api/v3/search/trending"
CG_MARKETS_URL          = "https://api.coingecko.com/api/v3/coins/markets"
CG_SEARCH_URL           = "https://api.coingecko.com/api/v3/search"
DEFILLAMA_PROTOCOLS_URL = "https://api.llama.fi/protocols"

# ---------------------------------------------------------------------------
# Cache TTLs (seconds) — keyed by cache file stem prefix
# ---------------------------------------------------------------------------

_TTL: dict[str, int] = {
    "risex_markets":          60,
    "binance_futures_ticker": 60,
    "binance_spot_list":      600,
    "binance_sentiment":      300,
    "binance_technical":      300,
    "binance_technical_1h":   300,
    "binance_technical_1d":   300,
    "binance_alpha_ticker":   300,
    "binance_alpha_klines":   3600,
    "binance_alpha_sentiment": 300,
    "cmc_trending":           600,
    "cg_trending":            600,
    "cg_markets":             600,
    "defillama_protocols":    1800,
    # per-symbol keys
    "binance_oi_hist":        300,
    "binance_klines":         3600,
    "binance_spot_klines":    3600,
    "binance_premium":        60,
    "binance_funding_hist":   300,
}

# Resolved relative to cwd; override this before calling if needed.
CACHE_DIR: Path = Path("cache") / "radar"

# ---------------------------------------------------------------------------
# Binance fapi weight monitor
# ---------------------------------------------------------------------------

class _WeightMonitor:
    """Thread-safe tracker for Binance fapi X-MBX-USED-WEIGHT-1M response header."""

    _WARN_THRESHOLD = 1800   # pause if used weight exceeds this
    _SLEEP_SECONDS  = 30

    def __init__(self) -> None:
        self._lock        = threading.Lock()
        self._used_weight = 0

    @property
    def used_weight(self) -> int:
        with self._lock:
            return self._used_weight

    def update(self, response: requests.Response) -> None:
        raw = response.headers.get("X-MBX-USED-WEIGHT-1M")
        if raw is None:
            return
        try:
            weight = int(raw)
        except ValueError:
            return
        with self._lock:
            self._used_weight = weight
        if weight > self._WARN_THRESHOLD:
            warnings.warn(
                f"Binance fapi weight {weight}/2400 exceeds {self._WARN_THRESHOLD}; "
                f"sleeping {self._SLEEP_SECONDS}s"
            )
            time.sleep(self._SLEEP_SECONDS)


_weight_monitor = _WeightMonitor()

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    if key.startswith("binance_oi_hist_"):
        return CACHE_DIR / "oi-hist" / f"{key}.json"
    if key.startswith("binance_spot_klines_"):
        return CACHE_DIR / "klines" / "spot" / f"{key}.json"
    if key.startswith("binance_klines_"):
        return CACHE_DIR / "klines" / "futures" / f"{key}.json"
    if key.startswith("binance_premium_"):
        return CACHE_DIR / "premium" / f"{key}.json"
    if key.startswith("binance_funding_hist_"):
        return CACHE_DIR / "funding-hist" / f"{key}.json"
    return CACHE_DIR / f"{key}.json"


def _load_cache(key: str, ttl: int) -> Any | None:
    """Return parsed JSON if cache file exists and is younger than ttl seconds."""
    p = _cache_path(key)
    if not p.exists():
        return None
    if time.time() - p.stat().st_mtime > ttl:
        return None
    with open(p) as f:
        return json.load(f)


def _save_cache(data: Any, key: str) -> None:
    p = _cache_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f)


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s


def _make_cmc_session() -> requests.Session:
    """Session with browser headers required by CMC's unofficial trending endpoint."""
    s = requests.Session()
    s.headers.update({
        "User-Agent":      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/148.0.0.0 Safari/537.36",
        "accept":          "application/json, text/plain, */*",
        "accept-language": "en-US,en;q=0.9",
        "cache-control":   "no-cache",
        "origin":          "https://coinmarketcap.com",
        "platform":        "web",
        "referer":         "https://coinmarketcap.com/",
        "sec-ch-ua":       '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
        "sec-ch-ua-mobile":   "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest":  "empty",
        "sec-fetch-mode":  "cors",
        "sec-fetch-site":  "same-site",
    })
    return s


def _get_with_retry(
    session: requests.Session,
    url: str,
    params: dict | None = None,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    is_fapi: bool = False,
) -> requests.Response:
    """
    GET with exponential backoff on network errors and 429/5xx.

    418 (Binance IP ban): raises RuntimeError immediately — caller must stop fapi calls.
    429: waits backoff_base**attempt seconds before retry.
    5xx: same backoff schedule.
    is_fapi=True: updates _weight_monitor from response header on success.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = session.get(url, params=params or {}, timeout=30)
            if resp.status_code == 418:
                raise RuntimeError(
                    f"Binance IP ban (418) on {url}. "
                    "Stop fapi calls and wait ≥60 s before retrying."
                )
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt >= max_retries:
                    last_exc = requests.HTTPError(response=resp)
                    break
                wait = backoff_base ** attempt
                warnings.warn(
                    f"HTTP {resp.status_code} from {url}; "
                    f"retry {attempt + 1}/{max_retries} in {wait:.0f}s"
                )
                time.sleep(wait)
                continue
            # Other 4xx (e.g. 404, 403, 400) are non-transient — raise immediately.
            resp.raise_for_status()
            if is_fapi:
                _weight_monitor.update(resp)
            return resp
        except RuntimeError:
            raise
        except requests.HTTPError as exc:
            # Non-retryable 4xx (404, 403, 400, etc.) — raise immediately.
            if exc.response is not None and 400 <= exc.response.status_code < 500:
                raise
            if attempt >= max_retries:
                last_exc = exc
                break
            wait = backoff_base ** attempt
            warnings.warn(f"{url}: {exc}; retry {attempt + 1}/{max_retries} in {wait:.0f}s")
            time.sleep(wait)
        except requests.RequestException as exc:
            if attempt >= max_retries:
                last_exc = exc
                break
            wait = backoff_base ** attempt
            warnings.warn(f"{url}: {exc}; retry {attempt + 1}/{max_retries} in {wait:.0f}s")
            time.sleep(wait)
    raise last_exc or RuntimeError(f"Failed to GET {url} after {max_retries} retries")


def _ttl_for(key: str) -> int:
    """Return TTL for a cache key, matching on prefix for per-symbol keys."""
    for prefix, ttl in _TTL.items():
        if key == prefix or key.startswith(f"{prefix}_"):
            return ttl
    return 300  # safe default


# ---------------------------------------------------------------------------
# Global fetch functions (called once per TTL, not per symbol)
# ---------------------------------------------------------------------------

def fetch_risex_markets(session: requests.Session | None = None,
                        refresh: bool = False) -> set[str]:
    """
    Return the set of base symbols currently listed on RISEx.
    E.g. {"BTC", "ETH", "SOL"}.
    Used as a hard gate: any symbol in this set is excluded from radar output.
    """
    key = "risex_markets"
    if not refresh:
        cached = _load_cache(key, _TTL[key])
        if cached is not None:
            return set(cached)

    session = session or _make_session()
    try:
        resp = _get_with_retry(session, RISEX_MARKETS_URL)
        markets = resp.json().get("data", {}).get("markets", [])
        # base_asset_symbol is "BTC/USDC" — take everything before "/"
        symbols = {m["base_asset_symbol"].split("/")[0] for m in markets
                   if "base_asset_symbol" in m}
    except Exception as exc:
        warnings.warn(f"fetch_risex_markets failed ({exc}); returning empty set")
        symbols = set()

    _save_cache(sorted(symbols), key)
    return symbols


def fetch_binance_futures_ticker(session: requests.Session | None = None,
                                 refresh: bool = False) -> list[dict]:
    """
    Return all Binance USD-M Futures 24hr ticker records.
    Each dict: {symbol, quoteVolume, priceChangePercent, count, lastPrice, ...}
    Used for: universe top-movers sort, s_vol percentile base, timing badge.
    Weight: 40.
    """
    key = "binance_futures_ticker"
    if not refresh:
        cached = _load_cache(key, _TTL[key])
        if cached is not None:
            return cached

    session = session or _make_session()
    try:
        resp = _get_with_retry(session, BINANCE_FAPI_TICKER, is_fapi=True)
        data = resp.json()
    except Exception as exc:
        warnings.warn(f"fetch_binance_futures_ticker failed ({exc}); returning []")
        data = []

    _save_cache(data, key)
    return data


def fetch_binance_spot_list(session: requests.Session | None = None,
                            refresh: bool = False) -> list[dict]:
    """
    Return all tokens from Binance marketing/symbol/list (~430 tokens).
    Each dict: {baseAsset, symbol, dayChange, volume, rank, tags, legalMoney, listingTime, ...}
    Used for: universe top-movers, spot volume (P/S ratio), category tags, Monitoring flag.
    """
    key = "binance_spot_list"
    if not refresh:
        cached = _load_cache(key, _TTL[key])
        if cached is not None:
            return cached

    session = session or _make_session()
    try:
        resp = _get_with_retry(session, BINANCE_SPOT_LIST_URL)
        data = resp.json().get("data", [])
        if not isinstance(data, list):
            data = []
    except Exception as exc:
        warnings.warn(f"fetch_binance_spot_list failed ({exc}); returning []")
        data = []

    _save_cache(data, key)
    return data


def fetch_binance_sentiment(session: requests.Session | None = None,
                             refresh: bool = False) -> dict[str, dict]:
    """
    Return Binance apex AI sentiment scores keyed by base asset symbol.
    Parsed from data.items[].metrics into flat dicts:
      {asset: {sentiment_score, sentiment_score_kol, sentiment_24h_social_volume,
               sentiment_score_social, sentiment_score_news}}
    All values are floats. Missing keys absent (not zero) — caller handles nulls.
    Used for: s_sent, KOL contextual signal, social volume contextual signal.
    """
    key = "binance_sentiment"
    if not refresh:
        cached = _load_cache(key, _TTL[key])
        if cached is not None:
            return cached

    session = session or _make_session()
    result: dict[str, dict] = {}
    try:
        resp = _get_with_retry(session, BINANCE_APEX_URL, params={"type": "sentiment"})
        items = resp.json().get("data", {}).get("items", [])
        _SENTIMENT_KEYS = {
            "sentiment_score",
            "sentiment_score_kol",
            "sentiment_24h_social_volume",
            "sentiment_score_social",
            "sentiment_score_news",
        }
        for item in items:
            asset   = item.get("asset") or item.get("baseAsset")
            metrics = item.get("metrics", {})
            if not asset:
                continue
            parsed: dict[str, float] = {}
            for k in _SENTIMENT_KEYS:
                if k in metrics:
                    try:
                        parsed[k] = float(metrics[k]["value"])
                    except (KeyError, TypeError, ValueError):
                        pass
            result[asset] = parsed
    except Exception as exc:
        warnings.warn(
            f"fetch_binance_sentiment failed ({exc}); sentiment scores unavailable — "
            "attention bucket will renormalize to CG + technical only"
        )

    _save_cache(result, key)
    return result


def fetch_binance_technical(session: requests.Session | None = None,
                             refresh: bool = False,
                             interval: str = "1h") -> dict[str, dict]:
    """
    Return Binance apex AI technical scores keyed by base asset symbol.
    Interval-aware: field names carry the interval suffix (_1h or _1d).
      interval="1h" → keys: technical_score_1h, technical_score_momentum_1h, ...
      interval="1d" → keys: technical_score_1d, technical_score_momentum_1d, ...
    All values are floats.
    Used for: s_tech blended from technical_score_1h (30%) + technical_score_1d (70%).
    """
    key = f"binance_technical_{interval}"
    if not refresh:
        cached = _load_cache(key, _TTL.get(key, 300))
        if cached is not None:
            return cached

    session = session or _make_session()
    result: dict[str, dict] = {}
    try:
        resp = _get_with_retry(
            session, BINANCE_APEX_URL,
            params={"type": "technical", "interval": interval},
        )
        items = resp.json().get("data", {}).get("items", [])
        _TECHNICAL_KEYS = {
            f"technical_score_{interval}",
            f"technical_score_momentum_{interval}",
            f"technical_score_trend_{interval}",
            f"technical_score_volatility_{interval}",
            f"technical_score_volprice_{interval}",
        }
        for item in items:
            asset   = item.get("asset") or item.get("baseAsset")
            metrics = item.get("metrics", {})
            if not asset:
                continue
            parsed: dict[str, float] = {}
            for k in _TECHNICAL_KEYS:
                if k in metrics:
                    try:
                        parsed[k] = float(metrics[k]["value"])
                    except (KeyError, TypeError, ValueError):
                        pass
            result[asset] = parsed
    except Exception as exc:
        warnings.warn(
            f"fetch_binance_technical failed ({exc}); technical scores unavailable — "
            "attention bucket will renormalize to sentiment only"
        )

    _save_cache(result, key)
    return result


def fetch_binance_alpha_ticker(session: requests.Session | None = None,
                               refresh: bool = False) -> dict[str, dict]:
    """
    Return Binance Alpha token records keyed by SYMBOL (e.g. "VELVET").

    Apex sentiment/technical endpoints do not cover Alpha tokens at all — they only
    return regular spot tokens. For Alpha tokens the only available attention proxy
    is the `score` field from this endpoint (0–max, rank-based). The notebook uses
    `score` to compute s_att for Alpha tokens instead of the Apex signals.

    Each value dict:
      addr_key  — "0xCONTRACTADDRESS@CHAINID" (for Apex key remapping if ever needed)
      marketCap — used as MCap fallback (Alpha tokens absent from binance_spot_list)
      volume24h — DEX volume in USD, used as spot-side proxy for P/S ratio
      score     — Binance ranking score [0, 111]; use percentile rank for normalisation
      fdv, price, alphaId, chainId — contextual
    """
    key = "binance_alpha_ticker"
    if not refresh:
        cached = _load_cache(key, _TTL[key])
        if cached is not None:
            return cached

    session = session or _make_session()
    result: dict[str, dict] = {}
    try:
        resp = _get_with_retry(
            session, BINANCE_ALPHA_TICKER_URL,
            params={"dataType": "aggregate"},
        )
        items = resp.json().get("data", [])
        if not isinstance(items, list):
            items = []
        for item in items:
            addr  = item.get("contractAddress", "")
            chain = item.get("chainId", "")
            sym   = item.get("symbol", "")
            if not (addr and chain and sym):
                continue
            result[sym] = {
                # addr_key matches the Apex asset field format ("0x...@chainId") for
                # remapping if Apex ever starts returning Alpha tokens in bulk
                "addr_key":  f"{addr}@{chain}",
                "name":      item.get("name", ""),
                "marketCap": _safe_float(item.get("marketCap")),
                "fdv":       _safe_float(item.get("fdv")),
                "price":     _safe_float(item.get("price")),
                # volume24h in USD — spot-volume proxy for P/S ratio (Alpha tokens
                # have no regular Binance spot market and no spot klines)
                "volume24h": _safe_float(item.get("volume24h")),
                # score: Binance's composite ranking for this Alpha token (0–111).
                # Used as the s_att proxy when Apex sentiment/technical are unavailable.
                "score":     item.get("score"),
                "alphaId":   item.get("alphaId", ""),
                "chainId":   chain,
            }
    except Exception as exc:
        warnings.warn(
            f"fetch_binance_alpha_ticker failed ({exc}); "
            "Alpha token attention scores will be 0 and MCap fallback unavailable"
        )

    _save_cache(result, key)
    return result


def fetch_binance_alpha_sentiment(addr_key: str,
                                  session: requests.Session | None = None,
                                  refresh: bool = False) -> dict[str, float]:
    """
    Fetch Apex AI sentiment for a single Alpha token via the per-asset endpoint.

    addr_key — "0xCONTRACTADDRESS@CHAINID" (stored in alpha_ticker value["addr_key"]).
    Returns a flat dict of floats with the same keys as fetch_binance_sentiment entries:
      {sentiment_score, sentiment_score_kol, sentiment_24h_social_volume,
       sentiment_score_social, sentiment_score_news}
    Returns {} on any failure or if the endpoint returns no data.

    The bulk Apex endpoint (/opportunity/assets) never returns Alpha tokens; this
    per-asset endpoint is the only way to get sentiment for them.
    Technical scores (type=technical) return data=null for Alpha tokens — only
    sentiment is available here.
    """
    # Cache key: replace special chars so it works as a filename stem.
    safe_key = addr_key.replace("0x", "").replace("@", "_")
    cache_key = f"binance_alpha_sentiment_{safe_key}"
    if not refresh:
        cached = _load_cache(cache_key, _TTL["binance_alpha_sentiment"])
        if cached is not None:
            return cached

    session = session or _make_session()
    result: dict[str, float] = {}
    _SENTIMENT_KEYS = {
        "sentiment_score",
        "sentiment_score_kol",
        "sentiment_24h_social_volume",
        "sentiment_score_social",
        "sentiment_score_news",
    }
    try:
        resp = _get_with_retry(
            session, BINANCE_APEX_DETAIL_URL,
            params={"asset": addr_key, "type": "sentiment", "interval": "1h", "quote": "USDT"},
        )
        data    = resp.json().get("data") or {}
        metrics = data.get("metrics", {})
        for k in _SENTIMENT_KEYS:
            if k in metrics:
                try:
                    result[k] = float(metrics[k]["value"])
                except (KeyError, TypeError, ValueError):
                    pass
    except Exception as exc:
        warnings.warn(f"fetch_binance_alpha_sentiment({addr_key}): {exc}")

    _save_cache(result, cache_key)
    return result


def _safe_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _normalize_alpha_kline(k: list) -> list:
    """
    Pad a 7-field Alpha kline to the standard 12-field Binance kline format.

    Alpha kline: [openTime, open, high, low, close, baseVolume, closeTime]
    Standard:    [openTime, open, high, low, close, baseVolume, closeTime,
                  quoteVolume, tradeCount, takerBase, takerQuote, ignore]

    quoteVolume (index 7) = baseVolume × closePrice.
    This is an approximation; it matches how klines_quote_vols() reads vol in the notebook.
    """
    if len(k) < 7:
        return k
    quote_vol = str(float(k[5]) * float(k[4]))
    return [k[0], k[1], k[2], k[3], k[4], k[5], k[6],
            quote_vol, "0", "0", "0", "0"]


def fetch_binance_alpha_klines(
    contract_address: str,
    chain_id: str,
    session: requests.Session | None = None,
    refresh: bool = False,
    interval: str = "1d",
    limit: int = 35,
) -> list[list]:
    """
    Return klines for a Binance Alpha token (DEX, on-chain data).

    Alpha tokens have no regular Binance spot market, so binance_spot_klines
    returns []. This endpoint provides the equivalent: OHLCV history keyed by
    contract address + chain, used as the spot-side proxy for P/S ratio scoring
    and for direction / BTC correlation calculations.

    Returned candles are padded to 12-field standard Binance kline format:
      [openTime, open, high, low, close, baseVol, closeTime, quoteVol, 0, 0, 0, 0]
    quoteVol = baseVol × closePrice (approximation; close enough for vol acceleration).

    contract_address: raw hex address WITHOUT chain suffix (e.g. "0x8b1943...")
    chain_id: chain as string (e.g. "56" for BSC)
    """
    safe_addr = contract_address.lower().replace("/", "_")
    key = f"binance_alpha_klines_{safe_addr}_{chain_id}_{interval}"
    if not refresh:
        cached = _load_cache(key, _TTL["binance_alpha_klines"])
        if cached is not None:
            return cached

    session = session or _make_session()
    data: list[list] = []
    try:
        resp = _get_with_retry(
            session, BINANCE_ALPHA_KLINES_URL,
            params={
                "chainId":      chain_id,
                "tokenAddress": contract_address,
                "interval":     interval,
                "limit":        limit,
                "dataType":     "aggregate",
            },
        )
        raw = resp.json().get("data", {}).get("klineInfos", [])
        if not isinstance(raw, list):
            raw = []
        data = [_normalize_alpha_kline(k) for k in raw if isinstance(k, list) and len(k) >= 7]
    except Exception as exc:
        warnings.warn(
            f"fetch_binance_alpha_klines({contract_address}@{chain_id}) failed ({exc}); "
            "P/S ratio and direction scores will be unavailable for this Alpha token"
        )

    _save_cache(data, key)
    return data


def fetch_cmc_trending(session: requests.Session | None = None,
                       refresh: bool = False,
                       page_size: int = 200) -> list[dict]:
    """
    Return CMC unified-trending list (up to page_size tokens).
    Each dict: {symbol, tokenSymbol, slug, priceUsd, volume24h,
                pricePercentageChange24h, marketCap, listingRank, ...}
    Requires browser-like headers (origin/referer/platform); bare User-Agent gets 500.
    Returns [] on persistent failure — not a hard error, universe shrinks gracefully.
    Retries once after 5 s on first 500.
    """
    key = "cmc_trending"
    if not refresh:
        cached = _load_cache(key, _TTL[key])
        if cached is not None:
            return cached

    session = _make_cmc_session()  # always use CMC-specific headers, ignore passed session
    data: list[dict] = []
    try:
        resp = _get_with_retry(
            session, CMC_TRENDING_URL,
            params={"interval": "24h", "pageNum": 1, "pageSize": page_size},
            max_retries=1,
            backoff_base=5.0,
        )
        body = resp.json()
        raw = body.get("data", {}).get("list", [])
        if not isinstance(raw, list):
            raw = []
        # Normalise: add "symbol" alias for tokenSymbol so downstream code uses one key
        for item in raw:
            item.setdefault("symbol", item.get("tokenSymbol", ""))
        data = raw
    except Exception as exc:
        warnings.warn(
            f"fetch_cmc_trending failed ({exc}); CMC source excluded from universe"
        )

    _save_cache(data, key)
    return data


def fetch_cg_trending(session: requests.Session | None = None,
                      refresh: bool = False) -> dict:
    """
    Return CoinGecko trending response dict: {coins, nfts, categories}.
    coins: list of {item: {symbol, score, market_cap_rank, data: {price, ...}}}
    categories: list of {name, slug, data: {market_cap, total_volume, ...}}
    Returns {} on failure.
    """
    key = "cg_trending"
    if not refresh:
        cached = _load_cache(key, _TTL[key])
        if cached is not None:
            return cached

    session = session or _make_session()
    data: dict = {}
    try:
        resp = _get_with_retry(session, CG_TRENDING_URL)
        data = resp.json()
        if not isinstance(data, dict):
            data = {}
    except Exception as exc:
        warnings.warn(
            f"fetch_cg_trending failed ({exc}); CoinGecko source excluded from universe"
        )

    _save_cache(data, key)
    return data


def fetch_cg_markets(session: requests.Session | None = None,
                     refresh: bool = False,
                     per_page: int = 250) -> list[dict]:
    """
    Return CoinGecko /coins/markets top-N by market cap (USD, no API key required).
    Each dict: {id, symbol, name, current_price, market_cap, market_cap_rank, ...}
    Used as MCap fallback for tokens absent from Binance spot list (e.g. HYPE).
    Returns [] on failure.
    """
    key = "cg_markets"
    if not refresh:
        cached = _load_cache(key, _TTL[key])
        if cached is not None:
            return cached

    session = session or _make_session()
    data: list[dict] = []
    try:
        resp = _get_with_retry(
            session, CG_MARKETS_URL,
            params={"vs_currency": "usd", "order": "market_cap_desc",
                    "per_page": per_page, "page": 1, "sparkline": "false"},
        )
        data = resp.json()
        if not isinstance(data, list):
            data = []
    except Exception as exc:
        warnings.warn(f"fetch_cg_markets failed ({exc}); CoinGecko MCap fallback unavailable")

    _save_cache(data, key)
    return data


def fetch_defillama_protocols(session: requests.Session | None = None,
                               refresh: bool = False) -> list[dict]:
    """
    Return all DefiLlama protocol records (~4000+).
    Each dict: {id, name, symbol, category, tvl, change_1d, change_7d, slug, mcap, ...}
    Used for: s_fund percentile rank over full universe.
    Response is ~4 MB; cached for 30 min.
    Returns [] on failure.
    """
    key = "defillama_protocols"
    if not refresh:
        cached = _load_cache(key, _TTL[key])
        if cached is not None:
            return cached

    session = session or _make_session()
    data: list[dict] = []
    try:
        resp = _get_with_retry(session, DEFILLAMA_PROTOCOLS_URL)
        data = resp.json()
        if not isinstance(data, list):
            data = []
    except Exception as exc:
        warnings.warn(
            f"fetch_defillama_protocols failed ({exc}); protocol usage score unavailable"
        )

    _save_cache(data, key)
    return data


# ---------------------------------------------------------------------------
# Per-symbol fetch functions (called lazily per candidate in Phase 2.6)
# ---------------------------------------------------------------------------

def fetch_binance_oi_hist(
    symbol: str,
    session: requests.Session | None = None,
    refresh: bool = False,
    period: str = "1h",
    limit: int = 720,
) -> list[dict]:
    """
    Return hourly OI history for one Binance USD-M symbol.
    Each dict: {symbol, sumOpenInterest, sumOpenInterestValue, timestamp}
    limit=720 covers 30 days at 1h resolution.
    Used for: r_OI_24h, r_OI_7d, r_OI_30d calculation (find closest snapshot to each anchor).
    Weight: 1 per call.
    Returns [] if symbol has no futures market or on failure.
    """
    key = f"binance_oi_hist_{symbol}"
    if not refresh:
        cached = _load_cache(key, _TTL["binance_oi_hist"])
        if cached is not None:
            return cached

    session = session or _make_session()
    data: list[dict] = []
    try:
        resp = _get_with_retry(
            session, BINANCE_FAPI_OI_HIST,
            params={"symbol": symbol, "period": period, "limit": limit},
            is_fapi=True,
        )
        data = resp.json()
        if not isinstance(data, list):
            data = []
    except Exception as exc:
        warnings.warn(f"fetch_binance_oi_hist({symbol}) failed ({exc}); OI score = null")

    _save_cache(data, key)
    return data


def fetch_binance_klines(
    symbol: str,
    session: requests.Session | None = None,
    refresh: bool = False,
    interval: str = "1d",
    limit: int = 35,
) -> list[list]:
    """
    Return daily kline data for one Binance USD-M symbol.
    Each row: [open_time, open, high, low, close, base_vol, close_time,
               quote_vol, trade_count, taker_buy_base, taker_buy_quote, ignore]
    limit=35 provides 30 full days plus buffer for partial current day.
    Used for: 30d average volume (vol acceleration), 7d daily spot volume (P/S trend).
    Weight: 2 per call (limit ≤ 499).
    Returns [] on failure.
    """
    key = f"binance_klines_{symbol}_{interval}"
    if not refresh:
        cached = _load_cache(key, _TTL["binance_klines"])
        if cached is not None:
            return cached

    session = session or _make_session()
    data: list[list] = []
    try:
        resp = _get_with_retry(
            session, BINANCE_FAPI_KLINES,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            is_fapi=True,
        )
        data = resp.json()
        if not isinstance(data, list):
            data = []
    except Exception as exc:
        warnings.warn(f"fetch_binance_klines({symbol}) failed ({exc}); vol accel = null")

    _save_cache(data, key)
    return data


def fetch_binance_premium_index(
    symbol: str,
    session: requests.Session | None = None,
    refresh: bool = False,
) -> dict:
    """
    Return premiumIndex for one Binance USD-M symbol.
    Dict: {symbol, markPrice, indexPrice, lastFundingRate, nextFundingTime, ...}
    Used for: OI notional = sumOpenInterest * markPrice.
    Weight: 1 per call.
    Returns {} on failure.
    """
    key = f"binance_premium_{symbol}"
    if not refresh:
        cached = _load_cache(key, _TTL["binance_premium"])
        if cached is not None:
            return cached

    session = session or _make_session()
    data: dict = {}
    try:
        resp = _get_with_retry(
            session, BINANCE_FAPI_PREMIUM,
            params={"symbol": symbol},
            is_fapi=True,
        )
        data = resp.json()
        if not isinstance(data, dict):
            data = {}
    except Exception as exc:
        warnings.warn(
            f"fetch_binance_premium_index({symbol}) failed ({exc}); OI notional = null"
        )

    _save_cache(data, key)
    return data


def fetch_binance_spot_klines(
    symbol: str,
    session: requests.Session | None = None,
    refresh: bool = False,
    interval: str = "1d",
    limit: int = 35,
) -> list[list]:
    """
    Return daily klines for a Binance spot symbol (api.binance.com, not fapi).
    Same array format as futures klines: [open_time, open, high, low, close,
    base_vol, close_time, quote_vol, trade_count, ...]
    limit=35 covers 30 full days + buffer for partial current day.
    Used for: 7-day daily spot volumes in P/S ratio trend calculation.
    Returns [] on failure or if symbol has no spot market.
    """
    key = f"binance_spot_klines_{symbol}_{interval}"
    if not refresh:
        cached = _load_cache(key, _TTL["binance_spot_klines"])
        if cached is not None:
            return cached

    session = session or _make_session()
    data: list[list] = []
    try:
        resp = _get_with_retry(
            session, BINANCE_SPOT_KLINES_URL,
            params={"symbol": symbol, "interval": interval, "limit": limit},
        )
        data = resp.json()
        if not isinstance(data, list):
            data = []
    except Exception as exc:
        warnings.warn(f"fetch_binance_spot_klines({symbol}) failed ({exc}); P/S trend = null")

    _save_cache(data, key)
    return data


def fetch_binance_funding_hist(
    symbol: str,
    session: requests.Session | None = None,
    refresh: bool = False,
    limit: int = 21,
) -> list[dict]:
    """
    Return funding rate settlement history for one Binance USD-M symbol.
    Each dict: {symbol, fundingRate, fundingTime, markPrice}
    limit=21 covers ~7 days at 8h settlement intervals (21 x 8h = 168h).
    Used for: s_fund_level, s_flip (neg-to-pos detection), s_persist (fraction positive).
    Weight: 1 per call.
    Returns [] on failure.
    """
    key = f"binance_funding_hist_{symbol}"
    if not refresh:
        cached = _load_cache(key, _TTL["binance_funding_hist"])
        if cached is not None:
            return cached

    session = session or _make_session()
    data: list[dict] = []
    try:
        resp = _get_with_retry(
            session, BINANCE_FAPI_FUNDING,
            params={"symbol": symbol, "limit": limit},
            is_fapi=True,
        )
        data = resp.json()
        if not isinstance(data, list):
            data = []
    except Exception as exc:
        warnings.warn(
            f"fetch_binance_funding_hist({symbol}) failed ({exc}); leverage demand = null"
        )

    _save_cache(data, key)
    return data


# ---------------------------------------------------------------------------
# CoinGecko search — used by AUTO_FILL_IDS export cell, not in fetch_all_radar
# ---------------------------------------------------------------------------

def fetch_cg_search(
    symbol: str,
    session: requests.Session | None = None,
    sleep_before: float = 4.0,
) -> dict:
    """
    Search CoinGecko for a symbol and return the best-match coin dict.
    Returns {} if no match or on ambiguity/failure.

    sleep_before: seconds to sleep before the request (default 4s for free-tier
    rate limit ~30 req/min). Pass 0 only if caller manages its own pacing.

    Returned dict fields: {id (coingecko slug), name, symbol, market_cap_rank}
    """
    session = session or _make_session()
    if sleep_before > 0:
        time.sleep(sleep_before)
    try:
        for attempt in range(3):
            resp = session.get(CG_SEARCH_URL, params={"query": symbol}, timeout=30)
            if resp.status_code == 429:
                wait = 60 * (2 ** attempt)
                warnings.warn(f"CoinGecko search 429 for {symbol}; sleeping {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            coins = resp.json().get("coins", [])
            if not coins:
                return {}
            # Best match: exact symbol match preferred, else first result
            exact = [c for c in coins if c.get("symbol", "").upper() == symbol.upper()]
            match = exact[0] if exact else coins[0]
            return {
                "id":              match.get("id", ""),
                "name":            match.get("name", ""),
                "symbol":          match.get("symbol", ""),
                "market_cap_rank": match.get("market_cap_rank"),
                "ambiguous":       len(exact) != 1,
            }
    except Exception as exc:
        warnings.warn(f"fetch_cg_search({symbol}) failed ({exc})")
    return {}


# ---------------------------------------------------------------------------
# Parallel orchestrator — global fetches only
# ---------------------------------------------------------------------------

def fetch_all_radar(
    refresh: bool = False,
    workers: int = 8,
) -> dict[str, Any]:
    """
    Fetch all global (non-per-symbol) radar sources in parallel.

    Per-symbol fetches (fetch_binance_oi_hist, fetch_binance_klines,
    fetch_binance_premium_index) are NOT called here; invoke them per-candidate
    in the notebook preprocessing phase (supports ThreadPoolExecutor(max_workers=5)
    there too — total fapi weight stays well within 2400/min).

    Returns:
        {
            "risex_markets":          set[str],        # base symbols on RISEx
            "binance_futures_ticker": list[dict],      # all fapi 24hr tickers
            "binance_spot_list":      list[dict],      # bapi marketing list
            "binance_sentiment":      dict[str, dict], # symbol → sentiment metrics (Alpha remapped)
            "binance_technical":      dict[str, dict], # symbol → technical metrics (1h, Alpha remapped)
            "binance_technical_1d":   dict[str, dict], # symbol → technical metrics (1d, Alpha remapped)
            "binance_alpha_ticker":   dict[str, dict], # "0xADDR@CHAIN" → {symbol, marketCap, volume24h, ...}
            "cmc_trending":           list[dict],      # [] on 500
            "cg_trending":            dict,            # {} on failure
            "cg_markets":             list[dict],      # [] on failure; top-250 by MCap
            "defillama_protocols":    list[dict],      # [] on failure
            "_fetch_errors":          dict[str, str],  # source → error for failed calls
        }

    Post-fetch: sentiment/technical keys for Binance Alpha tokens arrive as
    "0xADDRESS@CHAINID" from the Apex API. fetch_binance_alpha_ticker resolves
    these to plain symbols; fetch_all_radar remaps the dicts after all parallel
    fetches complete so callers always look up by symbol.
    """
    tasks: dict[str, Any] = {
        "risex_markets":          lambda: fetch_risex_markets(refresh=refresh),
        "binance_futures_ticker": lambda: fetch_binance_futures_ticker(refresh=refresh),
        "binance_spot_list":      lambda: fetch_binance_spot_list(refresh=refresh),
        "binance_sentiment":      lambda: fetch_binance_sentiment(refresh=refresh),
        "binance_technical":      lambda: fetch_binance_technical(refresh=refresh, interval="1h"),
        "binance_technical_1d":   lambda: fetch_binance_technical(refresh=refresh, interval="1d"),
        "binance_alpha_ticker":   lambda: fetch_binance_alpha_ticker(refresh=refresh),
        "cmc_trending":           lambda: fetch_cmc_trending(refresh=refresh, page_size=200),
        "cg_trending":            lambda: fetch_cg_trending(refresh=refresh),
        "cg_markets":             lambda: fetch_cg_markets(refresh=refresh),
        "defillama_protocols":    lambda: fetch_defillama_protocols(refresh=refresh),
    }

    results:      dict[str, Any] = {}
    fetch_errors: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except RuntimeError as exc:
                # RuntimeError from 418 — propagate: a Binance IP ban affects all fapi calls
                if "418" in str(exc):
                    raise
                fetch_errors[name] = str(exc)
                results[name] = _empty_for(name)
            except Exception as exc:
                fetch_errors[name] = str(exc)
                results[name] = _empty_for(name)

    # Remap Apex sentiment/technical keys for Alpha tokens from "0xADDR@CHAIN" → symbol.
    # Apex endpoints currently never return Alpha tokens, but we apply the remap
    # defensively in case they start doing so. alpha_ticker is now keyed by symbol,
    # so build the addr→symbol map from the stored addr_key field.
    alpha_map: dict[str, str] = {
        info["addr_key"]: sym
        for sym, info in results.get("binance_alpha_ticker", {}).items()
        if info.get("addr_key")
    }
    if alpha_map:
        for src in ("binance_sentiment", "binance_technical", "binance_technical_1d"):
            orig = results.get(src, {})
            results[src] = {alpha_map.get(k, k): v for k, v in orig.items()}

    results["_fetch_errors"] = fetch_errors
    if fetch_errors:
        warnings.warn(
            f"fetch_all_radar: {len(fetch_errors)} source(s) failed: "
            + ", ".join(f"{k}: {v}" for k, v in fetch_errors.items())
        )
    return results


def _empty_for(name: str) -> Any:
    """Return a safe empty value for each result key."""
    if name == "risex_markets":
        return set()
    if name in ("binance_sentiment", "binance_technical", "binance_technical_1d",
                "binance_alpha_ticker", "cg_trending"):
        return {}
    return []  # list sources


# ---------------------------------------------------------------------------
# Diagnostic entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    refresh = "--refresh" in sys.argv
    print(f"Fetching all radar sources (refresh={refresh}) ...")
    raw = fetch_all_radar(refresh=refresh)

    errors = raw.pop("_fetch_errors", {})
    for name, val in raw.items():
        n = len(val) if isinstance(val, (list, dict, set)) else "?"
        print(f"  {name:<30} {n} items")

    if errors:
        print("\nFetch errors:")
        for name, err in errors.items():
            print(f"  {name}: {err}")
    else:
        print("\nAll sources fetched successfully.")
    print(f"\nBinance fapi weight used: {_weight_monitor.used_weight}/2400")
