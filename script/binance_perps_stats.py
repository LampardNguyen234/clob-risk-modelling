"""
Fetch live Binance data and print perp market stats for tokens with MCap > MIN_MCAP.

Includes both regular spot-listed tokens and Binance Alpha tokens (on-chain DEX,
no Binance spot listing but have USDT perp futures).

Endpoints used (all public, no API key):
  Binance bapi:
    /bapi/composite/v1/public/marketing/symbol/list   — spot universe + MCap + FDV
    /bapi/defi/v1/public/alpha-trade/aggTicker24      — Alpha token universe + MCap + FDV + DEX vol
  Binance spot:
    /api/v3/ticker/24hr                               — spot 24h USDT quoteVolume
  Binance fapi:
    /fapi/v1/ticker/24hr                              — perp 24h USDT quoteVolume
    /fapi/v1/premiumIndex                             — mark price (for OI notional and price column)
    /fapi/v1/openInterest?symbol=X                    — current OI in base units

Stats reported (percentiles P10/P25/P50/P75/P90):
  - Perp/Spot volume ratio  — spot = Binance USDT spot for regular tokens,
                              DEX volume (volume24h, already USD) for Alpha tokens
  - OI notional (USD)       — contracts × mark price
  - OI/MCap ratio

Always prints top-20 and bottom-20 by perp volume with full metrics.

Usage:
  python script/binance_perps_stats.py [--min-mcap 50e6]
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

BAPI     = "https://www.binance.com"
SPOT_API = "https://api.binance.com"
FAPI     = "https://fapi.binance.com"

SPOT_LIST_URL   = f"{BAPI}/bapi/composite/v1/public/marketing/symbol/list"
ALPHA_LIST_URL  = f"{BAPI}/bapi/defi/v1/public/alpha-trade/aggTicker24"
SPOT_TICKER_URL = f"{SPOT_API}/api/v3/ticker/24hr"
PERP_TICKER_URL = f"{FAPI}/fapi/v1/ticker/24hr"
PREM_INDEX_URL  = f"{FAPI}/fapi/v1/premiumIndex"
OI_URL          = f"{FAPI}/fapi/v1/openInterest"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})


def _get(url, params=None, timeout=20):
    r = SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_spot_list() -> dict[str, dict]:
    """Return {baseAsset: {mcap, fdv}} from Binance spot listing."""
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
        out[sym] = {
            "mcap": float(mcap),
            "fdv":  float(fdv) if fdv else None,
        }
    return out


def fetch_alpha_ticker() -> dict[str, dict]:
    """Return {cexCoinName: {mcap, fdv, dex_vol}} for Alpha tokens with a perp listing.

    volume24h is already in USD.
    Only tokens with cexCoinName set have a Binance perp pair.
    """
    data  = _get(ALPHA_LIST_URL)
    items = data.get("data") or []
    out = {}
    for a in items:
        cex_sym = a.get("cexCoinName")
        if not cex_sym:
            continue
        mcap  = a.get("marketCap")
        vol   = a.get("volume24h")
        price = a.get("price")
        if not (mcap and vol and price):
            continue
        fdv = a.get("fdv")
        out[cex_sym] = {
            "mcap":    float(mcap),
            "fdv":     float(fdv) if fdv else None,
            "dex_vol": float(vol),  # already in USD
        }
    return out


def fetch_spot_vol() -> dict[str, float]:
    """Return {baseAsset: quoteVolume24hUSD} for all *USDT Binance spot pairs."""
    tickers = _get(SPOT_TICKER_URL)
    return {t["symbol"][:-4]: float(t["quoteVolume"])
            for t in tickers if t["symbol"].endswith("USDT")}


def fetch_perp_ticker() -> dict[str, float]:
    """Return {baseAsset: quoteVolume24hUSD} for all *USDT perp pairs."""
    tickers = _get(PERP_TICKER_URL)
    return {t["symbol"][:-4]: float(t["quoteVolume"])
            for t in tickers if t["symbol"].endswith("USDT")}


def fetch_mark_prices() -> dict[str, float]:
    """Return {baseAsset: markPrice} for all *USDT perp pairs."""
    entries = _get(PREM_INDEX_URL)
    return {e["symbol"][:-4]: float(e["markPrice"])
            for e in entries if e["symbol"].endswith("USDT")}


def fetch_oi_one(symbol: str) -> float | None:
    """Fetch current OI in base-asset units for one perp symbol (e.g. WLDUSDT)."""
    try:
        return float(_get(OI_URL, params={"symbol": symbol})["openInterest"])
    except Exception:
        return None


def fetch_oi_all(symbols: list[str], max_workers: int = 20) -> dict[str, float]:
    """Fetch OI for all symbols in parallel. Returns {baseAsset: oiContracts}."""
    result = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_oi_one, f"{sym}USDT"): sym for sym in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            val = fut.result()
            if val is not None:
                result[sym] = val
    return result


def fmt_usd(v):
    if v is None:   return "-"
    if v >= 1e9:    return f"${v/1e9:.2f}B"
    if v >= 1e6:    return f"${v/1e6:.1f}M"
    return f"${v/1e3:.0f}K"


def fmt_price(v):
    if v is None: return "-"
    if v >= 1000: return f"${v:,.0f}"
    if v >= 1:    return f"${v:.3f}"
    if v >= 0.01: return f"${v:.4f}"
    return f"${v:.6f}"


def percentile_table(values: list[float], label: str, fmt):
    a = np.array([v for v in values if v is not None and np.isfinite(v)])
    print(f"\n{label}  (n={len(a)})")
    if len(a) == 0:
        print("  (no data)")
        return
    print(f"  {'Pct':>4}  {'Value':>12}")
    for p in [10, 25, 50, 75, 90]:
        print(f"  P{p:02d}    {fmt(np.percentile(a, p)):>12}")


def print_token_table(rows: list[dict], title: str):
    """Print a fixed-width table of token metrics."""
    HDR = (f"  {'':1}  {'Token':<14}  {'Price':>10}  {'OI':>10}  {'OI/MCap':>8}  "
           f"{'OI/PVol':>8}  {'MCap':>10}  {'FDV':>10}  {'Spot Vol':>10}  "
           f"{'Perp Vol':>10}  {'P/S':>6}")
    print(f"\n{title}")
    print("  " + "-" * (len(HDR) - 2))
    print(HDR)
    print("  " + "-" * (len(HDR) - 2))
    for r in rows:
        tag    = "A" if r["is_alpha"] else " "
        oim_s  = f"{r['oi_mcap']:.2%}"      if r["oi_mcap"]     else "-"
        oipv_s = f"{r['oi_perp_vol']:.2f}x" if r["oi_perp_vol"] else "-"
        ps_s   = f"{r['ps_ratio']:.1f}x"    if r["ps_ratio"]    else "-"
        print(f"  {tag}  {r['sym']:<14}  {fmt_price(r['price']):>10}  "
              f"{fmt_usd(r['oi_usd']):>10}  {oim_s:>8}  {oipv_s:>8}  "
              f"{fmt_usd(r['mcap']):>10}  {fmt_usd(r['fdv']):>10}  "
              f"{fmt_usd(r['spot_vol']):>10}  {fmt_usd(r['perp_vol']):>10}  {ps_s:>6}")
    print("  " + "-" * (len(HDR) - 2))
    print("  A = Binance Alpha (DEX-native; Spot Vol = DEX vol)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-mcap", type=float, default=50e6,
                        help="Minimum market cap in USD (default: 50M)")
    parser.add_argument("--min-fdv", type=float, default=50e6,
                        help="Minimum fully diluted valuation in USD (default: 50M)")
    parser.add_argument("--min-perp-vol", type=float, default=5e6,
                        help="Minimum perp 24h quote volume in USD (default: 5M)")
    parser.add_argument("--min-oi", type=float, default=1e6,
                        help="Minimum OI notional in USD (default: 1M)")
    args = parser.parse_args()

    print("Fetching live data from Binance...", flush=True)

    print("  spot list (MCap + FDV)...", end=" ", flush=True)
    spot_list = fetch_spot_list()
    print(f"{len(spot_list)} entries")

    print("  alpha ticker...", end=" ", flush=True)
    alpha_data = fetch_alpha_ticker()
    print(f"{len(alpha_data)} entries with perp")

    print("  spot 24h ticker...", end=" ", flush=True)
    spot_vol = fetch_spot_vol()
    print(f"{len(spot_vol)} pairs")

    print("  perp 24h ticker...", end=" ", flush=True)
    perp_vol = fetch_perp_ticker()
    print(f"{len(perp_vol)} pairs")

    print("  premiumIndex (mark prices)...", end=" ", flush=True)
    mark_prices = fetch_mark_prices()
    print(f"{len(mark_prices)} pairs")

    # Regular tokens: Binance spot-listed, passes all thresholds, has USDT perp
    regular = {sym: {"mcap": d["mcap"], "fdv": d["fdv"], "is_alpha": False}
               for sym, d in spot_list.items()
               if d["mcap"] >= args.min_mcap
               and (d["fdv"] is None or d["fdv"] >= args.min_fdv)
               and perp_vol.get(sym, 0) >= args.min_perp_vol}

    # Alpha tokens: not in spot list, passes all thresholds, has USDT perp
    alpha = {sym: {"mcap": ad["mcap"], "fdv": ad["fdv"],
                   "is_alpha": True, "dex_vol": ad["dex_vol"]}
             for sym, ad in alpha_data.items()
             if sym not in spot_list
             and ad["mcap"] >= args.min_mcap
             and (ad["fdv"] is None or ad["fdv"] >= args.min_fdv)
             and perp_vol.get(sym, 0) >= args.min_perp_vol}

    candidates = {**regular, **alpha}

    n_reg   = sum(1 for v in candidates.values() if not v["is_alpha"])
    n_alpha = sum(1 for v in candidates.values() if v["is_alpha"])
    print(f"\nUniverse: {len(candidates)} tokens  "
          f"(MCap > ${args.min_mcap/1e6:.0f}M, FDV > ${args.min_fdv/1e6:.0f}M, "
          f"PerpVol > ${args.min_perp_vol/1e6:.0f}M, OI > ${args.min_oi/1e6:.0f}M)  "
          f"— {n_reg} regular + {n_alpha} Alpha")

    print(f"  fetching OI for {len(candidates)} symbols...", end=" ", flush=True)
    oi_contracts = fetch_oi_all(list(candidates.keys()))
    print(f"{len(oi_contracts)} fetched")

    # Build rows
    rows = []
    for sym, meta in candidates.items():
        mcap     = meta["mcap"]
        fdv      = meta.get("fdv")
        is_alpha = meta["is_alpha"]
        sv       = meta.get("dex_vol") if is_alpha else spot_vol.get(sym)
        pv       = perp_vol.get(sym)
        mark_p   = mark_prices.get(sym)
        oi_c     = oi_contracts.get(sym)

        oi_usd   = oi_c * mark_p if (oi_c and mark_p) else None
        if oi_usd is None or oi_usd < args.min_oi:
            continue
        ps_ratio    = pv / sv          if (pv and sv and sv > 0)  else None
        oi_mcap     = oi_usd / mcap    if (oi_usd and mcap > 0)   else None
        oi_perp_vol = oi_usd / pv      if (oi_usd and pv and pv > 0) else None

        rows.append({
            "sym":         sym,
            "is_alpha":    is_alpha,
            "price":       mark_p,
            "mcap":        mcap,
            "fdv":         fdv,
            "spot_vol":    sv,
            "perp_vol":    pv,
            "oi_usd":      oi_usd,
            "ps_ratio":    ps_ratio,
            "oi_mcap":     oi_mcap,
            "oi_perp_vol": oi_perp_vol,
        })

    def _rows(alpha_only=None):
        return rows if alpha_only is None else [r for r in rows if r["is_alpha"] == alpha_only]

    # Percentile stats
    for subset, label in [
        (rows,         "ALL"),
        (_rows(False), "Regular (Binance spot-listed)"),
        (_rows(True),  "Alpha (DEX-native, no spot listing)"),
    ]:
        print(f"\n{'='*60}")
        print(f"  {label}  ({len(subset)} tokens)")
        print(f"{'='*60}")
        note = "  * Alpha P/S uses DEX vol as spot proxy" if "Alpha" in label else ""
        percentile_table([r["ps_ratio"]    for r in subset],
                         f"Perp/Spot Vol Ratio{note}",
                         lambda v: f"{v:.2f}x")
        percentile_table([r["oi_usd"]      for r in subset], "OI Notional  (USD)", fmt_usd)
        percentile_table([r["oi_mcap"]     for r in subset], "OI / MCap",
                         lambda v: f"{v:.2%}")
        percentile_table([r["oi_perp_vol"] for r in subset], "OI / Perp Volume",
                         lambda v: f"{v:.2f}x")

    # Top-20 and bottom-20 by perp volume
    by_perp = sorted(rows, key=lambda r: r["perp_vol"] or 0, reverse=True)
    print_token_table(by_perp[:20],  "Top 20 by Perp Volume")
    print_token_table(by_perp[-20:], "Bottom 20 by Perp Volume")


if __name__ == "__main__":
    main()
