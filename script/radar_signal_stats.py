"""
Validates s_corr discrimination and signal variance across all live Binance perp tokens.

Fetches all data live (no cache dependency):
  BTC reference: Binance spot 1d klines (BTCUSDT)
  Token klines:  Binance futures 1d klines, one call per USDT perp symbol (parallel)

Usage:
    python script/radar_signal_stats.py
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT     = Path(__file__).parent.parent
BAPI     = "https://www.binance.com"
FAPI     = "https://fapi.binance.com"
SPOT_API = "https://api.binance.com"

SPOT_LIST_URL = f"{BAPI}/bapi/composite/v1/public/marketing/symbol/list"

MIN_FDV      = 50e6  # exclude tokens with FDV < $50M
MIN_PERP_VOL = 5e6   # exclude tokens with perp 24h vol < $5M

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})


def _get(url, params=None, timeout=20):
    r = SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_fdv_map() -> dict[str, float | None]:
    """Return {baseAsset: fdv} from Binance marketing API. fdv may be None if not set."""
    data  = _get(SPOT_LIST_URL)
    items = data.get("data") or []
    if isinstance(items, dict):
        items = items.get("list", [])
    out: dict[str, float | None] = {}
    for s in items:
        sym = s.get("baseAsset")
        if not sym:
            continue
        fdv = s.get("fullyDilutedMarketCap")
        out[sym] = float(fdv) if fdv else None
    return out


def fetch_btc_closes(limit: int = 60) -> list[float]:
    """Fetch BTC spot 1d closes. Excludes in-progress candle."""
    klines = _get(f"{SPOT_API}/api/v3/klines",
                  params={"symbol": "BTCUSDT", "interval": "1d", "limit": limit})
    return [float(k[4]) for k in klines[:-1]]


def fetch_perp_universe() -> tuple[dict[str, str], dict[str, float]]:
    """Return ({baseAsset: symbol}, {baseAsset: quoteVolume24hUSD}) for all USDT/USDC perp pairs.

    Deduped by base asset — USDT preferred over USDC.
    """
    tickers = _get(f"{FAPI}/fapi/v1/ticker/24hr")
    deduped:  dict[str, str]   = {}
    vol_usd:  dict[str, float] = {}
    for t in sorted(tickers, key=lambda x: x["symbol"]):
        sym = t["symbol"]
        if not sym.endswith(("USDT", "USDC")):
            continue
        base = sym
        for sfx in ("USDC", "USDT", "BUSD", "FDUSD"):
            if sym.endswith(sfx):
                base = sym[: -len(sfx)]
                break
        if base not in deduped or sym.endswith("USDT"):
            deduped[base] = sym
            vol_usd[base] = float(t.get("quoteVolume") or 0)
    return deduped, vol_usd


def fetch_futures_closes(sym: str, limit: int = 35) -> list[float] | None:
    """Fetch 1d futures closes. Excludes in-progress candle. Returns None on error."""
    try:
        klines = _get(f"{FAPI}/fapi/v1/klines",
                      params={"symbol": sym, "interval": "1d", "limit": limit})
        return [float(k[4]) for k in klines[:-1]]
    except Exception:
        return None


def main():
    print("Fetching live data from Binance...", flush=True)

    print("  BTC spot 1d closes...", end=" ", flush=True)
    btc_closes_all = fetch_btc_closes(60)
    print(f"{len(btc_closes_all)} closes")

    print("  FDV map...", end=" ", flush=True)
    fdv_map = fetch_fdv_map()
    print(f"{len(fdv_map)} entries")

    print("  perp universe...", end=" ", flush=True)
    universe, perp_vol_usd = fetch_perp_universe()
    # FDV filter: keep if unknown; exclude if known < MIN_FDV
    # Perp vol filter: exclude if < MIN_PERP_VOL (note: OI filter not applied here —
    #   it would require per-symbol OI fetches not relevant to correlation analysis)
    universe = {base: sym for base, sym in universe.items()
                if (fdv_map.get(base) is None or fdv_map[base] >= MIN_FDV)
                and perp_vol_usd.get(base, 0) >= MIN_PERP_VOL}
    print(f"{len(universe)} unique base assets "
          f"(FDV ≥ ${MIN_FDV/1e6:.0f}M or unknown, PerpVol ≥ ${MIN_PERP_VOL/1e6:.0f}M)")

    print(f"  futures 1d klines ({len(universe)} symbols in parallel)...", flush=True)
    closes_by_base: dict[str, list[float]] = {}
    with ThreadPoolExecutor(max_workers=20) as pool:
        futs = {pool.submit(fetch_futures_closes, sym): base
                for base, sym in universe.items()}
        for fut in as_completed(futs):
            base   = futs[fut]
            result = fut.result()
            if result:
                closes_by_base[base] = result
    print(f"  {len(closes_by_base)} symbols fetched")

    # Compute ρ_BTC for each token using 30-day log returns
    rows = []
    for base, token_closes in closes_by_base.items():
        n = min(len(token_closes), len(btc_closes_all), 31)
        if n < 16:
            continue
        t_rets = np.diff(np.log(np.maximum(token_closes[-n:], 1e-12)))
        b_rets = np.diff(np.log(np.maximum(btc_closes_all[-n:], 1e-12)))
        if len(t_rets) < 14:
            continue
        rho = float(np.corrcoef(t_rets, b_rets)[0, 1])
        if np.isnan(rho):
            continue
        s_corr = max(0.0, 1.0 - abs(rho))
        rows.append({"symbol": base, "rho_btc": round(rho, 4), "s_corr": round(s_corr, 4)})

    df = pd.DataFrame(rows).sort_values("rho_btc").reset_index(drop=True)

    SEP = "=" * 60
    print(f"\n{SEP}")
    print(f"BTC Correlation Check  ({len(df)} unique tokens)")
    print(SEP)

    print("\nρ_BTC distribution:")
    print(df["rho_btc"].describe().round(3).to_string())

    print("\ns_corr = max(0, 1 - |ρ|) distribution:")
    print(df["s_corr"].describe().round(3).to_string())

    std_scorr  = df["s_corr"].std()
    mean_scorr = df["s_corr"].mean()
    print(f"\n→ mean s_corr = {mean_scorr:.3f},  std = {std_scorr:.3f}")
    if std_scorr < 0.15:
        print("  ⚠  LOW DISCRIMINATION: std < 0.15. s_corr near-uniform. "
              "Confirms §6 alt-season concern.")
    else:
        print("  ✓  Adequate spread. s_corr is differentiating.")

    print("\nρ_BTC bins:")
    bins   = [-1.0, -0.3, 0.3, 0.6, 0.8, 1.0]
    labels = ["strong_neg (<-0.3)", "decorr (-0.3..0.3)", "moderate (0.3..0.6)",
              "high (0.6..0.8)", "very_high (>0.8)"]
    df["rho_bin"] = pd.cut(df["rho_btc"], bins=bins, labels=labels)
    print(df["rho_bin"].value_counts().sort_index().to_string())

    print("\nBottom 10 (most BTC-correlated, s_corr → 0):")
    print(df.head(10)[["symbol", "rho_btc", "s_corr"]].to_string(index=False))

    print("\nTop 10 (most decorrelated, s_corr → 1):")
    print(df.tail(10)[["symbol", "rho_btc", "s_corr"]].to_string(index=False))

    # Optional: cross-reference against radar notebook YAML export
    yaml_path = ROOT / "notes/token_listing/radar_export.yml"
    try:
        import re
        text = yaml_path.read_text()
        entries = []
        for block in text.split("  - symbol:")[1:]:
            sym_match  = re.match(r" (\S+)", block)
            r_match    = re.search(r"R=([\d.]+)", block)
            chip_match = re.search(r"# R=.+?\n    # (.+)", block)
            if sym_match and r_match:
                chips = chip_match.group(1).strip() if chip_match else ""
                entries.append({
                    "symbol": sym_match.group(1),
                    "R":      float(r_match.group(1)),
                    "chips":  chips,
                })
        if entries:
            scored_df = pd.DataFrame(entries)
            scored_df = scored_df.merge(df[["symbol", "rho_btc", "s_corr"]], on="symbol", how="left")

            print(f"\n{SEP}")
            print("Radar output × BTC correlation")
            print(SEP)
            print(scored_df[["symbol", "R", "rho_btc", "s_corr", "chips"]].to_string(index=False))

            chip_keywords = {
                "Vol Accel":  [r"Vol \d+", "Spot Spike"],
                "OI Growth":  ["OI rho=", "OI/MCap"],
                "Attention":  ["Sent ", "Technical"],
                "Leverage":   ["Fund ", "Persist", "Fund Flip", "Flip"],
                "Price":      ["4h Move", "24h Move", "Direction", "BTC corr"],
            }
            print(f"\n{SEP}")
            print("Bucket chip frequency (how often each bucket drives a top-3 reason):")
            print(SEP)
            for bucket, kws in chip_keywords.items():
                count = sum(1 for row in scored_df.itertuples()
                            if any(kw.lower() in row.chips.lower() for kw in kws))
                pct = count / len(scored_df) * 100
                bar = "█" * int(pct / 3)
                print(f"  {bucket:<12} {count:>3}/{len(scored_df)}  ({pct:4.0f}%)  {bar}")

    except Exception as e:
        print(f"\nCould not load YAML export: {e}")


if __name__ == "__main__":
    main()
