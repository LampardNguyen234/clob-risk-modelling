"""
Radar threshold calibration from empirical signal distributions.

Rationale
---------
Current thresholds were set by judgment (e.g., "5× is exceptional").
This script asks: given the actual distribution of each raw signal across all
live Binance perp tokens, at what percentile does the current θ sit? And what θ
would give better score discrimination?

Design principle
----------------
For a listing radar, a good threshold sits near p80-p85:
  - Top 15-20% of universe gets near-full credit (score → 1.0)
  - Bottom 50% scores ≤ 0.5
  - std(scores) ≈ 0.28-0.35 across universe

Endpoints used (all public, no API key):
  /bapi/composite/v1/public/marketing/symbol/list  — MCap per token
  /api/v3/ticker/24hr                              — spot USDT quote volume (P/S denominator)
  /fapi/v1/ticker/24hr                             — perp USDT quote volume + price change
  /fapi/v1/premiumIndex                            — mark price + index price (for basis)
  /fapi/v1/klines                                  — futures 1d klines (vol accel, 90d history)
  /api/v3/klines                                   — spot 1d klines (spot spike, 90d history)
  /futures/data/openInterestHist                   — OI history (4h intervals, ~83 days max)
  /fapi/v1/fundingRate                             — funding rate history (~90 days)

OI history note: uses period=4h, limit=500 (Binance max = ~83 days at 4h).
  7-day  slice = last  42 entries (42 × 4h = 168h).
  30-day slice = last 180 entries.
  60-day slice = last 360 entries.
  Spearman rho is rank-based — time spacing does not affect the trend direction result.

Usage:
    python script/threshold_calibration.py
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from scipy import stats

BAPI     = "https://www.binance.com"
SPOT_API = "https://api.binance.com"
FAPI     = "https://fapi.binance.com"

SPOT_LIST_URL   = f"{BAPI}/bapi/composite/v1/public/marketing/symbol/list"
SPOT_TICKER_URL = f"{SPOT_API}/api/v3/ticker/24hr"
PERP_TICKER_URL = f"{FAPI}/fapi/v1/ticker/24hr"
PREM_INDEX_URL  = f"{FAPI}/fapi/v1/premiumIndex"

MIN_FDV      = 50e6  # exclude tokens with FDV < $50M
MIN_PERP_VOL = 5e6   # exclude tokens with perp 24h vol < $5M
MIN_OI       = 1e6   # exclude tokens with OI notional < $1M

# OI history: 4h candles; Binance max per call = 500 entries = ~83 days
OI_PERIOD    = "4h"
OI_LIMIT     = 500
OI_7D_SLOTS  =  42   # 7d  × 24h / 4h
OI_30D_SLOTS = 180   # 30d × 24h / 4h
OI_60D_SLOTS = 360   # 60d × 24h / 4h
OI_MIN_SLOTS = OI_7D_SLOTS  # require at least 7 days for a meaningful trend

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})

SEP  = "=" * 65
SEP2 = "-" * 65


def _get(url, params=None, timeout=20):
    r = SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ── Bulk fetchers ─────────────────────────────────────────────────────────────

def fetch_spot_list() -> dict[str, dict]:
    """Return {baseAsset: {mcap, fdv}} from Binance marketing API."""
    data  = _get(SPOT_LIST_URL)
    items = data.get("data") or []
    if isinstance(items, dict):
        items = items.get("list", [])
    out = {}
    for s in items:
        sym  = s.get("baseAsset")
        mcap = s.get("marketCap")
        if not (sym and mcap):
            continue
        fdv = s.get("fullyDilutedMarketCap")
        out[sym] = {"mcap": float(mcap), "fdv": float(fdv) if fdv else None}
    return out


def fetch_spot_ticker() -> dict[str, float]:
    """Return {baseAsset: quoteVolume24h USD} for all *USDT spot pairs."""
    tickers = _get(SPOT_TICKER_URL)
    return {t["symbol"][:-4]: float(t["quoteVolume"])
            for t in tickers if t["symbol"].endswith("USDT")}


def fetch_premium_index() -> dict[str, dict]:
    """Return {baseAsset: {markPrice, indexPrice}} for all *USDT perp pairs."""
    entries = _get(PREM_INDEX_URL)
    out = {}
    for e in entries:
        if not e["symbol"].endswith("USDT"):
            continue
        mark  = float(e.get("markPrice",  0) or 0)
        index = float(e.get("indexPrice", 0) or 0)
        if mark > 0 and index > 0:
            out[e["symbol"][:-4]] = {"markPrice": mark, "indexPrice": index}
    return out


def build_perp_universe(all_tickers: list[dict]) -> list[tuple[str, str]]:
    """Return [(base, sym)] for all active USDT/USDC perp pairs, deduped (USDT preferred)."""
    seen: dict[str, str] = {}
    for t in sorted(t["symbol"] for t in all_tickers
                    if t["symbol"].endswith(("USDT", "USDC"))):
        sym  = t
        base = sym
        for sfx in ("USDC", "USDT", "BUSD", "FDUSD"):
            if sym.endswith(sfx):
                base = sym[: -len(sfx)]
                break
        if base not in seen or sym.endswith("USDT"):
            seen[base] = sym
    return list(seen.items())


# ── Per-symbol fetcher ────────────────────────────────────────────────────────

def fetch_token_data(sym: str) -> dict:
    """Fetch all per-symbol live data. Each call is best-effort; errors → empty list."""
    base     = sym[:-4]  # strip USDT
    spot_sym = base + "USDT"
    result   = {"sym": sym, "base": base,
                "fut_klines": [], "spot_klines": [], "oi_hist": [], "fund_hist": []}

    # Futures 1d klines: 120 entries (90 complete days + buffer + in-progress)
    try:
        result["fut_klines"] = _get(f"{FAPI}/fapi/v1/klines",
                                     params={"symbol": sym, "interval": "1d", "limit": 120})
    except Exception:
        pass

    # Spot 1d klines: only exists for Binance-listed tokens (not Alpha)
    try:
        result["spot_klines"] = _get(f"{SPOT_API}/api/v3/klines",
                                      params={"symbol": spot_sym, "interval": "1d", "limit": 91})
    except Exception:
        pass

    # OI history: 4h candles, ~83 days (Binance max = 500 entries); sumOpenInterestValue is already USD
    try:
        result["oi_hist"] = _get(f"{FAPI}/futures/data/openInterestHist",
                                   params={"symbol": sym, "period": OI_PERIOD, "limit": OI_LIMIT})
    except Exception:
        pass

    # Funding rate history: ~90 days at 8h intervals
    try:
        result["fund_hist"] = _get(f"{FAPI}/fapi/v1/fundingRate",
                                    params={"symbol": sym, "limit": 270})
    except Exception:
        pass

    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def pct_of(arr, val):
    return float(stats.percentileofscore(arr, val, kind="weak"))


def score_linear(arr, theta):
    return np.clip(arr / theta, 0.0, 1.0)


def score_log_ceiling(arr, ceiling):
    """Log-clamp: score = log(x)/log(C), clamped to [0,1]. x<1 → 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.log(np.maximum(arr, 1e-12)) / np.log(ceiling)
    return np.clip(s, 0.0, 1.0)


def report(name, arr_raw, current_theta, score_fn, unit="", ceiling_mode=False):
    arr      = np.array([v for v in arr_raw if v is not None and np.isfinite(v)])
    positive = arr[arr > 0]
    if len(positive) < 5:
        print(f"\n{SEP2}\n  {name}: insufficient data (n={len(positive)})")
        return

    ps = np.percentile(positive, [25, 50, 75, 80, 85, 90, 95])
    p_theta = pct_of(positive, current_theta)

    scores_curr = score_fn(positive, current_theta)
    theta_p80   = max(ps[3], 1e-9)
    theta_p85   = max(ps[4], 1e-9)
    scores_p80  = score_fn(positive, theta_p80)
    scores_p85  = score_fn(positive, theta_p85)

    at_ceil_curr = 100 * (scores_curr >= 1.0).mean()
    at_ceil_p80  = 100 * (scores_p80  >= 1.0).mean()
    at_zero_curr = 100 * (scores_curr <= 0.0).mean()

    print(f"\n{SEP}")
    print(f"  {name}")
    print(f"{SEP}")
    print(f"  n = {len(positive)}   (positive values only)")
    print(f"  Raw:  mean={positive.mean():.5g}  std={positive.std():.5g}  "
          f"median={np.median(positive):.5g} {unit}")
    print(f"  Percentiles: p25={ps[0]:.5g}  p50={ps[1]:.5g}  p75={ps[2]:.5g}  "
          f"p80={ps[3]:.5g}  p85={ps[4]:.5g}  p90={ps[5]:.5g}  p95={ps[6]:.5g} {unit}")
    print()
    print(f"  Current θ = {current_theta}{unit}  →  p{p_theta:.0f} of universe")
    print(f"  {'(log-ceiling)' if ceiling_mode else '(linear clamp)'}")
    print()
    print(f"  Score under current θ:  mean={scores_curr.mean():.3f}  "
          f"std={scores_curr.std():.3f}  at_ceil={at_ceil_curr:.0f}%  at_zero={at_zero_curr:.0f}%")
    print(f"  Score under p80 θ={theta_p80:.5g}:  mean={scores_p80.mean():.3f}  "
          f"std={scores_p80.std():.3f}  at_ceil={at_ceil_p80:.0f}%")
    print(f"  Score under p85 θ={theta_p85:.5g}:  mean={scores_p85.mean():.3f}  "
          f"std={scores_p85.std():.3f}  at_ceil={100*(scores_p85>=1.0).mean():.0f}%")
    print()

    if at_ceil_curr > 30:
        verdict = f"⚠  TOO EASY: {at_ceil_curr:.0f}% of tokens at full score. Raise θ to p85={theta_p85:.5g}"
    elif at_ceil_curr < 5:
        verdict = f"⚠  TOO HARD: only {at_ceil_curr:.0f}% hit full score. Lower θ toward p80={theta_p80:.5g}"
    elif scores_curr.std() < 0.18:
        verdict = f"⚠  LOW SPREAD: std={scores_curr.std():.3f}. Recalibrate θ."
    else:
        verdict = f"✓  OK: {at_ceil_curr:.0f}% at ceiling, std={scores_curr.std():.3f}"
    print(f"  → {verdict}")

    return {
        "name": name, "n": len(positive),
        "current_theta": current_theta, "p_theta": p_theta,
        "p80_theta": float(theta_p80), "p85_theta": float(theta_p85),
        "std_curr": float(scores_curr.std()), "at_ceil_curr": float(at_ceil_curr),
        "std_p80": float(scores_p80.std()),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{SEP}")
    print("  Radar Threshold Calibration")
    print(SEP)
    print("  Fetching live data from Binance...")

    # Bulk fetches
    print("  spot list (MCap)...", end=" ", flush=True)
    spot_list = fetch_spot_list()
    print(f"{len(spot_list)} entries")

    print("  perp ticker...", end=" ", flush=True)
    all_perp_tickers = _get(PERP_TICKER_URL)
    # Index USDT pairs by base for quick lookup
    perp_ticker: dict[str, dict] = {
        t["symbol"][:-4]: t for t in all_perp_tickers if t["symbol"].endswith("USDT")
    }
    print(f"{len(perp_ticker)} USDT pairs")

    print("  spot ticker (vol)...", end=" ", flush=True)
    spot_ticker = fetch_spot_ticker()
    print(f"{len(spot_ticker)} USDT pairs")

    print("  premiumIndex (mark/index prices)...", end=" ", flush=True)
    prem_index = fetch_premium_index()
    print(f"{len(prem_index)} entries")

    # Universe: all USDT/USDC perp pairs, deduped by base asset (USDT preferred)
    universe = build_perp_universe(all_perp_tickers)
    # Filter to USDT-quoted symbols only (straightforward kline/OI fetching)
    # and exclude tokens with known FDV < MIN_FDV (keep if FDV not set in spot_list)
    usdt_universe = [
        (base, sym) for base, sym in universe
        if sym.endswith("USDT")
        and (spot_list.get(base, {}).get("fdv") is None
             or spot_list[base]["fdv"] >= MIN_FDV)
        and float(perp_ticker.get(base, {}).get("quoteVolume") or 0) >= MIN_PERP_VOL
    ]
    print(f"\n  Universe: {len(usdt_universe)} unique base assets "
          f"(USDT perp, FDV ≥ ${MIN_FDV/1e6:.0f}M or unknown, "
          f"PerpVol ≥ ${MIN_PERP_VOL/1e6:.0f}M, OI ≥ ${MIN_OI/1e6:.0f}M applied per-token)")

    # Per-symbol parallel fetch
    print(f"  Fetching per-symbol data ({len(usdt_universe)} symbols, parallel)...", flush=True)
    all_token_data: list[dict] = []
    with ThreadPoolExecutor(max_workers=20) as pool:
        futs = {pool.submit(fetch_token_data, sym): (base, sym)
                for base, sym in usdt_universe}
        for fut in as_completed(futs):
            td = fut.result()
            if td:
                all_token_data.append(td)
    print(f"  {len(all_token_data)} symbols fetched")

    # Signal accumulators
    vol_accel_7d_vals      = []   # avg_7d / avg_30d
    vol_accel_24h_vals     = []   # V_24h  / avg_30d
    vol_accel_7d_90_vals   = []   # avg_7d / avg_90d  (extended baseline)
    vol_accel_24h_90_vals  = []   # V_24h  / avg_90d  (extended baseline)
    spot_spike_vals        = []
    PS_level_vals          = []
    price_24h_vals         = []
    OI_MCap_vals           = []
    fund_level_vals        = []
    persist_vals           = []
    basis_vals             = []
    OI_trend_7d_vals       = []
    OI_trend_30d_vals      = []
    OI_trend_60d_vals      = []

    for td in all_token_data:
        base = td["base"]

        # ── Futures klines: vol accel ──────────────────────────────────────
        klines = td["fut_klines"]
        closes: list[float] = []
        if len(klines) >= 2:
            # k[7] = quoteAssetVolume = USDT-denominated perp trading volume
            quote_vols = [float(k[7]) for k in klines]
            closes     = [float(k[4]) for k in klines]
            n = len(quote_vols)
            # Averages over complete days only (exclude in-progress last candle)
            avg_30d = float(np.mean(quote_vols[max(0, n - 31):n - 1])) if n >= 2  else None
            avg_7d  = float(np.mean(quote_vols[max(0, n -  8):n - 1])) if n >= 8  else None
            avg_90d = float(np.mean(quote_vols[max(0, n - 91):n - 1])) if n >= 30 else None
            v_24h   = quote_vols[-1]  # in-progress candle = today's live volume
            if avg_30d and avg_30d > 0:
                vol_accel_24h_vals.append(v_24h / avg_30d)
                if avg_7d and avg_7d > 0:
                    vol_accel_7d_vals.append(avg_7d / avg_30d)
            if avg_90d and avg_90d > 0:
                vol_accel_24h_90_vals.append(v_24h / avg_90d)
                if avg_7d and avg_7d > 0:
                    vol_accel_7d_90_vals.append(avg_7d / avg_90d)

        # ── Price 24h: from live perp ticker (most accurate) ───────────────
        tick = perp_ticker.get(base, {})
        pcp  = tick.get("priceChangePercent")
        if pcp:
            # priceChangePercent is a string percentage (e.g. "1.23" = 1.23%)
            price_24h_vals.append(abs(float(pcp)) / 100)
        elif len(closes) >= 2 and closes[-2] > 0:
            price_24h_vals.append(abs(closes[-1] - closes[-2]) / closes[-2])

        # ── Spot spike: spot vol today vs 7-day average ───────────────────
        # k[7] = quoteAssetVolume = USDT spot volume
        sk = td["spot_klines"]
        if len(sk) >= 8:
            svols   = [float(k[7]) for k in sk]
            sv_24h  = svols[-1]                        # in-progress day
            sv_7avg = float(np.mean(svols[-8:-1]))     # last 7 complete days
            if sv_7avg > 0:
                spot_spike_vals.append(sv_24h / sv_7avg)

        # ── P/S level: both legs USDT quote volume ─────────────────────────
        perp_qvol = float(tick.get("quoteVolume") or 0)
        spot_qvol = spot_ticker.get(base, 0.0)
        if perp_qvol > 0 and spot_qvol > 0:
            PS_level_vals.append(perp_qvol / spot_qvol)

        # ── OI filter + OI/MCap + OI trend ────────────────────────────────
        oi_hist = td["oi_hist"]
        # Skip token if OI is confirmed below minimum; keep if OI unavailable (fetch error)
        if oi_hist:
            _latest_oi = float(oi_hist[-1]["sumOpenInterestValue"])
            if _latest_oi < MIN_OI:
                continue
        if oi_hist:
            # sumOpenInterestValue is already USD (contracts × mark price at snapshot time)
            latest_oi_usd = float(oi_hist[-1]["sumOpenInterestValue"])
            mcap = spot_list.get(base, {}).get("mcap")
            if mcap and mcap > 0 and latest_oi_usd > 0:
                OI_MCap_vals.append(latest_oi_usd / mcap)

            if len(oi_hist) >= OI_MIN_SLOTS:
                oi_vals = [float(h["sumOpenInterestValue"]) for h in oi_hist]
                t_idx   = np.arange(len(oi_vals))
                # Spearman rho is rank-based: time unit (4h) does not affect trend direction
                rho_7d,  _ = stats.spearmanr(t_idx[-OI_7D_SLOTS:],  oi_vals[-OI_7D_SLOTS:])
                rho_30d, _ = stats.spearmanr(t_idx[-OI_30D_SLOTS:], oi_vals[-OI_30D_SLOTS:])
                if np.isfinite(rho_7d):
                    OI_trend_7d_vals.append(rho_7d)
                if np.isfinite(rho_30d):
                    OI_trend_30d_vals.append(rho_30d)
                if len(oi_hist) >= OI_60D_SLOTS:
                    rho_60d, _ = stats.spearmanr(t_idx[-OI_60D_SLOTS:], oi_vals[-OI_60D_SLOTS:])
                    if np.isfinite(rho_60d):
                        OI_trend_60d_vals.append(rho_60d)

        # ── Funding ───────────────────────────────────────────────────────
        fund_hist = td["fund_hist"]
        if fund_hist:
            rates     = [float(h["fundingRate"]) for h in fund_hist]
            pos_rates = [r for r in rates if r > 0]
            fund_level_vals.append(float(np.mean(pos_rates)) if pos_rates else 0.0)
            persist_vals.append(sum(1 for r in rates if r > 0) / len(rates))

        # ── Basis: (mark - index) / index ─────────────────────────────────
        prem  = prem_index.get(base, {})
        mark  = prem.get("markPrice",  0.0)
        index = prem.get("indexPrice", 0.0)
        if mark > 0 and index > 0:
            basis_vals.append((mark - index) / index)

    # ── Report per metric ──────────────────────────────────────────────────────

    print(f"\n\n{SEP}")
    print("  SIGNAL DISTRIBUTIONS AND THRESHOLD CALIBRATION")
    print(f"{SEP}")

    results = []

    r = report("Vol Accel 7d  (avg_7d / avg_30d)",    vol_accel_7d_vals,     5.0, score_log_ceiling, "×", True)
    if r: results.append(r)
    r = report("Vol Accel 24h  (V_24h / avg_30d)",    vol_accel_24h_vals,    5.0, score_log_ceiling, "×", True)
    if r: results.append(r)
    r = report("Vol Accel 7d  (avg_7d / avg_90d)",    vol_accel_7d_90_vals,  5.0, score_log_ceiling, "×", True)
    if r: results.append(r)
    r = report("Vol Accel 24h  (V_24h / avg_90d)",    vol_accel_24h_90_vals, 5.0, score_log_ceiling, "×", True)
    if r: results.append(r)
    r = report("Spot Spike  (spot_V_24h / spot_V_7d_avg)", spot_spike_vals, 3.0,  score_log_ceiling, "×", True)
    if r: results.append(r)
    r = report("P/S Level  (perp_quoteVol_24h / spot_vol_24h)", PS_level_vals, 2.0, score_linear, "×")
    if r: results.append(r)
    r = report("Price Change 24h  (|ΔP| decimal)",    price_24h_vals,     0.20,   score_linear, " (dec)")
    if r: results.append(r)
    r = report("OI/MCap  (OI_USD / market_cap)",       OI_MCap_vals,       0.30,   score_linear)
    if r: results.append(r)
    r = report("Funding Level  (avg positive funding rate per 8h)", fund_level_vals, 0.0003, score_linear)
    if r: results.append(r)
    r = report("Basis  |(mark-index)/index|",           [abs(v) for v in basis_vals], 0.002, score_linear)
    if r: results.append(r)

    if persist_vals:
        arr = np.array(persist_vals)
        print(f"\n{SEP}")
        print("  s_persist  (fraction of positive funding settlements)")
        print(SEP)
        print(f"  n = {len(arr)}")
        print(f"  mean={arr.mean():.3f}  std={arr.std():.3f}  "
              f"p25={np.percentile(arr,25):.3f}  p50={np.percentile(arr,50):.3f}  "
              f"p75={np.percentile(arr,75):.3f}")
        print(f"  % tokens with persist ≥ 0.70: {100*(arr>=0.70).mean():.0f}%")
        print(f"  % tokens with persist ≥ 0.90: {100*(arr>=0.90).mean():.0f}%")
        if arr.std() < 0.15:
            print("  ⚠  LOW SPREAD: s_persist clusters tightly → poor discrimination")
        else:
            print("  ✓  Adequate spread.")

    if OI_trend_7d_vals and OI_trend_30d_vals:
        a7  = np.array(OI_trend_7d_vals)
        a30 = np.array(OI_trend_30d_vals)
        a60 = np.array(OI_trend_60d_vals) if OI_trend_60d_vals else None
        print(f"\n{SEP}")
        print("  OI Spearman ρ  (no θ; max(0,ρ) used directly)")
        print(SEP)
        print(f"  7d  rho: n={len(a7)}   mean={a7.mean():.3f}  std={a7.std():.3f}  "
              f"p25={np.percentile(a7,25):.3f}  p50={np.percentile(a7,50):.3f}  p75={np.percentile(a7,75):.3f}")
        print(f"  30d rho: n={len(a30)}  mean={a30.mean():.3f}  std={a30.std():.3f}  "
              f"p25={np.percentile(a30,25):.3f}  p50={np.percentile(a30,50):.3f}  p75={np.percentile(a30,75):.3f}")
        if a60 is not None:
            print(f"  60d rho: n={len(a60)}  mean={a60.mean():.3f}  std={a60.std():.3f}  "
                  f"p25={np.percentile(a60,25):.3f}  p50={np.percentile(a60,50):.3f}  p75={np.percentile(a60,75):.3f}")
        n_common = min(len(a7), len(a30))
        corr_7_30, _ = stats.pearsonr(a7[:n_common], a30[:n_common])
        print(f"  Pearson(rho_7d, rho_30d) = {corr_7_30:.3f}")
        if a60 is not None:
            n_c60 = min(n_common, len(a60))
            corr_30_60, _ = stats.pearsonr(a30[:n_c60], a60[:n_c60])
            print(f"  Pearson(rho_30d, rho_60d) = {corr_30_60:.3f}")
        s7  = np.clip(a7[:n_common],  0, 1)
        s30 = np.clip(a30[:n_common], 0, 1)
        blend = 0.40 * s7 + 0.60 * s30
        print(f"  Score (s_OI_trend = 0.40·rho_7d + 0.60·rho_30d):  "
              f"mean={blend.mean():.3f}  std={blend.std():.3f}")
        pct_pos = (f"7d={100*(a7>0).mean():.0f}%  30d={100*(a30>0).mean():.0f}%"
                   + (f"  60d={100*(a60>0).mean():.0f}%" if a60 is not None else ""))
        print(f"  % tokens with positive rho: {pct_pos}")

    # Summary table
    print(f"\n\n{SEP}")
    print("  SUMMARY: Current θ vs recommended range")
    print(SEP)
    print(f"  {'Metric':<40} {'Curr θ':>8} {'p(θ)':>6} {'at_ceil':>8} {'std':>6} {'p80_θ':>10} {'p85_θ':>10}")
    print(f"  {'-'*40} {'-'*8} {'-'*6} {'-'*8} {'-'*6} {'-'*10} {'-'*10}")
    for r in results:
        if r:
            print(f"  {r['name']:<40} {r['current_theta']:>8.5g} {r['p_theta']:>5.0f}% "
                  f"{r['at_ceil_curr']:>7.0f}% {r['std_curr']:>6.3f} "
                  f"{r['p80_theta']:>10.5g} {r['p85_theta']:>10.5g}")

    print()
    print("  Legend:")
    print("    p(θ)     : where current θ sits in the empirical distribution")
    print("    at_ceil  : % of tokens that score 1.0 under current θ (target: 15-25%)")
    print("    std      : score std dev (target: ≥ 0.25 for good discrimination)")
    print("    p80/p85_θ: threshold that would put only top 20%/15% at full score")


if __name__ == "__main__":
    main()
