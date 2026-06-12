"""
Forward-return proxy backtest for radar sub-scores.

Methodology
-----------
We lack labeled listing outcomes, so we use short-term (5d) price return as a
noisy proxy for "was this a good signal?" This is valid under one assumption:
tokens that the radar correctly identifies as momentum candidates should show
continued positive price action over a short horizon, at least more often than
tokens the radar correctly ranks low.

Procedure
---------
1. Fetch the most recent 35 complete daily klines per symbol. Split into:
   - Historical window: first 30 bars (T-35d to T-5d) — compute signals here
   - Forward window:    last  5 bars  (T-5d  to T)    — measure 5d return here

2. For each signal, sweep θ across a grid. Compute:
   - Spearman ρ(score, forward_return): is higher score associated with
     better forward returns?
   - Mann-Whitney U p-value: do high-score tokens (top tercile) have
     significantly better returns than low-score tokens (bottom tercile)?
   - % of high-score tokens with positive return (directional hit rate)

3. Report the θ that maximises Spearman ρ, and compare it to the current θ.

Live data note
--------------
Fetches all data live from Binance (no cache).

OI signal uses the last 30 days of 4h OI history (T-30d to T), which is
slightly misaligned with the historical klines window (T-35d to T-5d). This
is an acceptable approximation given the noisy nature of the proxy test.

P/S level uses live spot ticker quoteVolume (USDT) for both legs — the
original cache-based version used the BAPI marketing `volume` field which
is in base-asset units and is unreliable.

Limitations (read before interpreting results)
----------------------------------------------
- n ≈ 50-100 tokens, one time window → no statistical significance. Results
  are directionally indicative only. A truly reliable test requires ≥ 200
  labeled listing events across multiple market regimes.
- 5d price return ≠ listing quality.
- Survivorship bias: universe is selected for liquidity (perp vol ≥ $5M),
  biasing toward tokens with recent positive momentum.
- Forward window is only 5 bars. A 30-day forward window would be more
  meaningful but is unavailable from a single fetch.

Usage
-----
    python script/forward_return_proxy.py
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from scipy import stats

BAPI     = "https://www.binance.com"
FAPI     = "https://fapi.binance.com"
SPOT_API = "https://api.binance.com"

SPOT_LIST_URL   = f"{BAPI}/bapi/composite/v1/public/marketing/symbol/list"
SPOT_TICKER_URL = f"{SPOT_API}/api/v3/ticker/24hr"
PERP_TICKER_URL = f"{FAPI}/fapi/v1/ticker/24hr"

# Universe quality filters — consistent with notebook and other scripts
MIN_FDV      = 50e6   # FDV >= $50M
MIN_PERP_VOL =  5e6   # perp 24h quote volume >= $5M
MIN_OI       =  1e6   # OI notional >= $1M

# OI history: 4h candles, 30 days = 180 entries
OI_PERIOD    = "4h"
OI_LIMIT     = 180
OI_MIN_SLOTS = 12   # 12 × 4h = 48h minimum for Spearman

HIST_BARS = 30
FWD_BARS  =  5
MIN_BARS  = HIST_BARS + FWD_BARS  # 35 complete daily bars required

SEP  = "=" * 65
SEP2 = "-" * 65

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})


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
    """Return {baseAsset: quoteVolume24hUSD} for all *USDT spot pairs."""
    tickers = _get(SPOT_TICKER_URL)
    return {t["symbol"][:-4]: float(t["quoteVolume"])
            for t in tickers if t["symbol"].endswith("USDT")}


def build_perp_universe(all_tickers: list[dict]) -> list[tuple[str, str]]:
    """Return [(base, sym)] for all active USDT/USDC perp pairs, deduped (USDT preferred)."""
    seen: dict[str, str] = {}
    for t in sorted(all_tickers, key=lambda x: x["symbol"]):
        sym = t["symbol"]
        if not sym.endswith(("USDT", "USDC")):
            continue
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
    """Fetch per-symbol live data. Errors → empty list."""
    base     = sym[:-4]
    spot_sym = base + "USDT"
    result   = {"sym": sym, "base": base,
                "fut_klines": [], "spot_klines": [], "oi_hist": []}

    try:
        # MIN_BARS complete + 1 in-progress candle
        result["fut_klines"] = _get(f"{FAPI}/fapi/v1/klines",
                                     params={"symbol": sym, "interval": "1d",
                                             "limit": MIN_BARS + 1})
    except Exception:
        pass

    try:
        result["spot_klines"] = _get(f"{SPOT_API}/api/v3/klines",
                                      params={"symbol": spot_sym, "interval": "1d",
                                              "limit": MIN_BARS + 1})
    except Exception:
        pass  # Alpha tokens and some others have no spot listing

    try:
        result["oi_hist"] = _get(f"{FAPI}/futures/data/openInterestHist",
                                   params={"symbol": sym, "period": OI_PERIOD,
                                           "limit": OI_LIMIT})
    except Exception:
        pass

    return result


# ── Scoring functions ─────────────────────────────────────────────────────────

def score_linear(x, theta):
    if theta <= 0:
        return 0.0
    return float(np.clip(x / theta, 0.0, 1.0))


def score_log_ceiling(x, ceiling):
    if x <= 0 or ceiling <= 1:
        return 0.0
    return float(np.clip(np.log(x) / np.log(ceiling), 0.0, 1.0))


# ── Sweep ─────────────────────────────────────────────────────────────────────

def sweep_threshold(records, signal_key, theta_grid, score_fn, name,
                    current_theta, unit=""):
    vals = [(r[signal_key], r["fwd_return"]) for r in records
            if r[signal_key] is not None and np.isfinite(r[signal_key])
            and np.isfinite(r["fwd_return"])]
    if len(vals) < 10:
        print(f"\n  {name}: insufficient data (n={len(vals)})")
        return None

    raw_arr = np.array([v[0] for v in vals])
    ret_arr = np.array([v[1] for v in vals])

    print(f"\n{SEP}")
    print(f"  {name}   (n={len(vals)})")
    print(SEP)
    print(f"  Forward return dist: mean={ret_arr.mean():.3f}  std={ret_arr.std():.3f}  "
          f"p25={np.percentile(ret_arr,25):.3f}  p75={np.percentile(ret_arr,75):.3f}")
    print()
    print(f"  {'θ':>10}   {'Spearman ρ':>12}   {'p-value':>10}   "
          f"{'hit_rate(top3)':>15}   {'score_std':>10}")
    print(f"  {'-'*10}   {'-'*12}   {'-'*10}   {'-'*15}   {'-'*10}")

    best_rho   = -np.inf
    best_theta = None

    for theta in theta_grid:
        scores = np.array([score_fn(x, theta) for x in raw_arr])
        if scores.std() < 0.01:
            continue
        rho, pval = stats.spearmanr(scores, ret_arr)
        if not np.isfinite(rho):
            continue

        t33      = np.percentile(scores, 67)
        top_mask = scores >= t33
        hit_rate = (ret_arr[top_mask] > 0).mean() if top_mask.sum() >= 3 else np.nan

        marker = " ← current" if abs(theta - current_theta) < 1e-9 else ""
        print(f"  {theta:>10.4g}{unit}   {rho:>+12.4f}   {pval:>10.4f}   "
              f"{100*hit_rate:>13.0f}%   {scores.std():>10.4f}{marker}")

        if rho > best_rho:
            best_rho, best_theta = rho, theta

    if best_theta is not None:
        print(f"\n  → Best θ for Spearman ρ: {best_theta}{unit}  (ρ={best_rho:+.4f})")

    return best_theta, best_rho


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{SEP}")
    print("  Forward-Return Proxy Backtest")
    print(SEP)
    print(f"  Historical window : first {HIST_BARS} of {MIN_BARS} complete daily bars")
    print(f"  Forward window    : last  {FWD_BARS}  of {MIN_BARS} complete daily bars")
    print(f"  Universe filters  : FDV ≥ ${MIN_FDV/1e6:.0f}M, "
          f"PerpVol ≥ ${MIN_PERP_VOL/1e6:.0f}M, OI ≥ ${MIN_OI/1e6:.0f}M")

    print("\nFetching live data from Binance...")

    print("  spot list (FDV/MCap)...", end=" ", flush=True)
    spot_list = fetch_spot_list()
    print(f"{len(spot_list)} entries")

    print("  perp ticker...", end=" ", flush=True)
    all_perp_tickers = _get(PERP_TICKER_URL)
    perp_ticker = {t["symbol"][:-4]: t
                   for t in all_perp_tickers if t["symbol"].endswith("USDT")}
    print(f"{len(perp_ticker)} USDT pairs")

    print("  spot ticker (vol)...", end=" ", flush=True)
    spot_ticker = fetch_spot_ticker()
    print(f"{len(spot_ticker)} USDT pairs")

    # Build universe with FDV + perp vol filters (OI applied per-token below)
    universe = build_perp_universe(all_perp_tickers)
    usdt_universe = [
        (base, sym) for base, sym in universe
        if sym.endswith("USDT")
        and (spot_list.get(base, {}).get("fdv") is None
             or spot_list[base]["fdv"] >= MIN_FDV)
        and float(perp_ticker.get(base, {}).get("quoteVolume") or 0) >= MIN_PERP_VOL
    ]
    print(f"  Universe: {len(usdt_universe)} tokens after FDV + PerpVol filters")

    print(f"  Fetching per-symbol data ({len(usdt_universe)} symbols, parallel)...",
          flush=True)
    all_token_data: list[dict] = []
    with ThreadPoolExecutor(max_workers=20) as pool:
        futs = {pool.submit(fetch_token_data, sym): (base, sym)
                for base, sym in usdt_universe}
        for fut in as_completed(futs):
            td = fut.result()
            if td:
                all_token_data.append(td)
    print(f"  {len(all_token_data)} symbols fetched")

    # Build per-token records
    records        = []
    skipped_bars   = 0
    skipped_oi_low = 0

    for td in all_token_data:
        base = td["base"]

        # Drop in-progress last candle; require MIN_BARS complete bars
        klines   = td["fut_klines"]
        complete = klines[:-1] if len(klines) > 1 else klines
        if len(complete) < MIN_BARS:
            skipped_bars += 1
            continue

        # Use most recent MIN_BARS bars
        closes     = [float(k[4]) for k in complete[-MIN_BARS:]]
        quote_vols = [float(k[7]) for k in complete[-MIN_BARS:]]

        h_closes = closes[:HIST_BARS]
        h_vols   = quote_vols[:HIST_BARS]

        c_start = h_closes[-1]
        if c_start <= 0:
            skipped_bars += 1
            continue
        # c_end is close of the last bar in the forward window
        c_end      = closes[HIST_BARS + FWD_BARS - 1]
        fwd_return = (c_end - c_start) / c_start

        # OI filter: skip if confirmed below minimum
        oi_hist = td["oi_hist"]
        if oi_hist:
            latest_oi_usd = float(oi_hist[-1]["sumOpenInterestValue"])
            if latest_oi_usd < MIN_OI:
                skipped_oi_low += 1
                continue

        # ── Signals ────────────────────────────────────────────────────────

        # Vol accel 7d: avg(hist[-7:]) / avg(hist)
        avg_30d = float(np.mean(h_vols))
        avg_7d  = float(np.mean(h_vols[-7:]))
        va_7d   = avg_7d / avg_30d if avg_30d > 0 else None

        # Vol accel 24h: last hist bar / avg(hist[:-1])
        avg_excl = float(np.mean(h_vols[:-1])) if len(h_vols) > 1 else None
        va_24h   = h_vols[-1] / avg_excl if (avg_excl and avg_excl > 0) else None

        # Spot spike: spot_vol_24h / spot_vol_7d_avg
        # k[7] = quoteAssetVolume = USDT-denominated spot volume
        spot_spike  = None
        sk          = td["spot_klines"]
        sk_complete = sk[:-1] if len(sk) > 1 else sk
        if len(sk_complete) >= MIN_BARS:
            sv      = [float(k[7]) for k in sk_complete[-MIN_BARS:]][:HIST_BARS]
            sv_avg7 = float(np.mean(sv[-7:]))
            if sv_avg7 > 0:
                spot_spike = sv[-1] / sv_avg7

        # Price change 24h (absolute, decimal)
        price_24h = (abs(h_closes[-1] - h_closes[-2]) / h_closes[-2]
                     if len(h_closes) >= 2 and h_closes[-2] > 0 else None)

        # P/S level: both legs USDT quoteVolume from live tickers
        # (old version used BAPI marketing `volume` = base-asset units — wrong units)
        perp_qvol = float(perp_ticker.get(base, {}).get("quoteVolume") or 0)
        spot_qvol = spot_ticker.get(base, 0.0)
        PS_level  = perp_qvol / spot_qvol if perp_qvol > 0 and spot_qvol > 0 else None

        # OI Spearman trend (4h data, ~30 days; slightly misaligned with hist window — acceptable)
        OI_trend = None
        if oi_hist and len(oi_hist) >= OI_MIN_SLOTS:
            oi_vals = [float(h["sumOpenInterestValue"]) for h in oi_hist]
            rho, _  = stats.spearmanr(np.arange(len(oi_vals)), oi_vals)
            if np.isfinite(rho):
                OI_trend = rho

        records.append({
            "base":       base,
            "fwd_return": fwd_return,
            "va_7d":      va_7d,
            "va_24h":     va_24h,
            "spot_spike": spot_spike,
            "price_24h":  price_24h,
            "PS_level":   PS_level,
            "OI_trend":   OI_trend,
        })

    print(f"  Records: {len(records)}  "
          f"(skipped: {skipped_bars} insufficient bars, "
          f"{skipped_oi_low} OI < ${MIN_OI/1e6:.0f}M)")

    # ── Threshold sweep ───────────────────────────────────────────────────────

    print(f"\n\n{SEP}")
    print("  THRESHOLD SWEEP RESULTS")
    print("  (Spearman ρ between score and 5d forward return)")
    print(SEP)
    print()
    print("  Interpretation:")
    print("    ρ > 0 : higher score → better forward return (signal has momentum)")
    print("    ρ < 0 : higher score → worse forward return (mean-reversion)")
    print("    ρ ≈ 0 : no relationship")
    print("    hit_rate(top3): % of top-tercile tokens with positive 5d return")
    print()
    print("  NOTE: n is small, one window → not statistically significant.")
    print("  Spearman |ρ| < 0.28 is noise at this sample size (α=0.05 threshold).")

    sweep_threshold(records, "va_7d",
                    [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
                    score_log_ceiling,
                    "Vol Accel 7d  (avg_7d / avg_30d) — log-ceiling",
                    current_theta=5.0, unit="×")

    sweep_threshold(records, "va_24h",
                    [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
                    score_log_ceiling,
                    "Vol Accel 24h  (V_24h / avg_30d) — log-ceiling",
                    current_theta=5.0, unit="×")

    sweep_threshold(records, "spot_spike",
                    [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
                    score_log_ceiling,
                    "Spot Spike  (spot_V_24h / spot_V_7d_avg) — log-ceiling",
                    current_theta=3.0, unit="×")

    sweep_threshold(records, "price_24h",
                    [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30],
                    score_linear,
                    "Price Change 24h  (|ΔP| decimal) — linear",
                    current_theta=0.20)

    sweep_threshold(records, "PS_level",
                    [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
                    score_linear,
                    "P/S Level  (perp_vol / spot_vol) — linear",
                    current_theta=2.0, unit="×")

    # OI trend: no θ, just report ρ(OI_trend, fwd_return)
    oi_vals = [(r["OI_trend"], r["fwd_return"]) for r in records
               if r["OI_trend"] is not None and np.isfinite(r["OI_trend"])
               and np.isfinite(r["fwd_return"])]
    if len(oi_vals) >= 10:
        oi_rho_arr = np.array([v[0] for v in oi_vals])
        oi_ret_arr = np.array([v[1] for v in oi_vals])
        rho_oi, pval_oi = stats.spearmanr(oi_rho_arr, oi_ret_arr)
        print(f"\n{SEP}")
        print(f"  OI Spearman trend  (no θ; 4h OI last 30d)   n={len(oi_vals)}")
        print(SEP)
        print(f"  Spearman ρ(OI_trend, fwd_return) = {rho_oi:+.4f}  (p={pval_oi:.4f})")
        hit = (oi_ret_arr[oi_rho_arr > 0] > 0).mean() if (oi_rho_arr > 0).any() else np.nan
        print(f"  Hit rate (OI_trend>0 → fwd>0): {100*hit:.0f}%")
        print(f"  → {'confirms' if rho_oi > 0.1 else 'weak or no'} short-term momentum link.")

    # ── Composite score vs forward return ─────────────────────────────────────

    print(f"\n\n{SEP}")
    print("  COMPOSITE SCORE VS FORWARD RETURN")
    print("  Using current thresholds and v1.1 bucket weights")
    print(SEP)

    W = {"vol": 0.25, "price": 0.15, "OI": 0.25}

    comp_rows = []
    for r in records:
        if not np.isfinite(r["fwd_return"]):
            continue

        parts = {}
        if r["va_7d"]  is not None: parts["7d"]  = score_log_ceiling(r["va_7d"],  5.0)
        if r["va_24h"] is not None: parts["24h"] = score_log_ceiling(r["va_24h"], 5.0)
        if parts:
            w       = {"7d": 0.60, "24h": 0.40}
            total_w = sum(w[k] for k in parts)
            s_vol   = sum(parts[k] * w[k] for k in parts) / total_w
        else:
            s_vol = None

        s_price = score_linear(r["price_24h"], 0.20) if r["price_24h"] is not None else None
        s_oi    = max(0.0, r["OI_trend"])             if r["OI_trend"]  is not None else None

        available = {k: v for k, v in [("vol", s_vol), ("price", s_price), ("OI", s_oi)]
                     if v is not None}
        if len(available) < 2:
            continue
        total_w   = sum(W[k] for k in available)
        composite = sum(available[k] * W[k] for k in available) / total_w * 100

        comp_rows.append({"base": r["base"], "composite": composite,
                          "fwd_return": r["fwd_return"]})

    if len(comp_rows) >= 10:
        c_arr   = np.array([r["composite"]  for r in comp_rows])
        ret_arr = np.array([r["fwd_return"] for r in comp_rows])
        rho_c, pval_c = stats.spearmanr(c_arr, ret_arr)

        comp_rows.sort(key=lambda x: -x["composite"])
        print(f"\n  {'Token':<12} {'Score':>7} {'5d Return':>10}")
        print(f"  {'-'*12} {'-'*7} {'-'*10}")
        for row in comp_rows[:15]:
            print(f"  {row['base']:<12} {row['composite']:>7.1f} {row['fwd_return']:>+9.2%}")

        print(f"\n  n={len(comp_rows)}  Spearman ρ(composite, fwd_return) = "
              f"{rho_c:+.4f}  p={pval_c:.4f}")
        top_half = ret_arr[c_arr >= np.median(c_arr)]
        bot_half = ret_arr[c_arr <  np.median(c_arr)]
        print(f"\n  Top-half avg 5d return : {top_half.mean():+.2%}")
        print(f"  Bot-half avg 5d return : {bot_half.mean():+.2%}")
        _, mw_pval = stats.mannwhitneyu(top_half, bot_half, alternative="greater")
        print(f"  Mann-Whitney U p-value : {mw_pval:.4f}  "
              f"({'significant at α=0.10' if mw_pval < 0.10 else 'not significant'})")
        print()
        print("  Caveats: n is small; short-term return ≠ listing quality.")
        print("  Take direction not magnitude.")


if __name__ == "__main__":
    main()
