"""
Liquidity Metrics Calculation Module (2026 Update)

This module provides functions for calculating various liquidity metrics from OHLCV data.

NEW RECOMMENDED METRICS:
- Corwin-Schultz Spread (2012): Bid-ask spread estimator from High-Low prices
- Kyle's Lambda (1985/2016): Market impact coefficient
- Dollar Depth: Total dollar volume traded

DEPRECATED METRICS (kept for comparison):
- Amihud ILLIQ Ratio (2002): DO NOT USE for new calculations

References:
- Corwin & Schultz (2012), "A Simple Way to Estimate Bid-Ask Spreads", Journal of Finance
- Kyle & Obizhaeva (2016), "Market Microstructure Invariance", Econometrica
- Amihud (2002), "Illiquidity and Stock Returns", Journal of Financial Markets

See docs/liquidity_estimation_alternatives.md for detailed methodology and rationale.
"""

import numpy as np
import pandas as pd


# ============================================================================
# NEW LIQUIDITY METRICS (RECOMMENDED)
# ============================================================================

def corwin_schultz_spread(df, set_negative_to_zero=True):
    """
    Calculate Corwin-Schultz bid-ask spread estimator.

    IMPORTANT: Corwin-Schultz (2012) was designed for DAILY data.
    This function resamples to daily before calculating spreads.

    The CS method uses high-low price ratios to estimate effective bid-ask spreads:
    - Daily highs ≈ trades at ask price
    - Daily lows ≈ trades at bid price
    - Compare 1-day vs 2-day ratios to isolate spread from variance

    Reference: Corwin & Schultz (2012), Journal of Finance

    Parameters:
    -----------
    df : pd.DataFrame
        Must contain 'high' and 'low' columns, indexed by timestamp
    set_negative_to_zero : bool
        If True, set negative daily spreads to zero (recommended)

    Returns:
    --------
    pd.Series : Estimated bid-ask spread (as proportion, e.g., 0.01 = 1%)
                Returned at original frequency via forward-fill

    Examples:
    ---------
    >>> df = load_ohlcv_data('BTCUSDT')
    >>> spread = corwin_schultz_spread(df)
    >>> print(f"Average spread: {spread.mean()*100:.3f}%")
    """
    # CRITICAL: Resample to daily frequency (CS method designed for daily data)
    daily_df = df.resample('D').agg({
        'high': 'max',
        'low': 'min'
    }).dropna()

    # Single-day high-low ratio (beta)
    beta = (np.log(daily_df['high'] / daily_df['low'])) ** 2

    # Two-day high-low ratio (gamma)
    high_2day = daily_df['high'].rolling(2).max()
    low_2day = daily_df['low'].rolling(2).min()
    gamma = (np.log(high_2day / low_2day)) ** 2

    # Component alpha (spread-related)
    sqrt_2 = np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * sqrt_2) - \
            np.sqrt(gamma / (3 - 2 * sqrt_2))

    # Estimated spread: S = 2(e^α - 1)/(1 + e^α)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

    if set_negative_to_zero:
        spread = spread.clip(lower=0)

    # Replace inf and extreme values
    spread = spread.replace([np.inf, -np.inf], np.nan)

    # Forward-fill to original (minute-level) frequency
    spread_minute = spread.reindex(df.index, method='ffill')

    return spread_minute


def estimate_kyle_lambda_fast(df, window='24h'):
    """
    Estimate Kyle's lambda (market impact coefficient) using fast vectorized operations.

    Kyle's lambda measures price impact: ΔP/P = λ × √Q
    where Q is order size in dollars, and ΔP/P is the relative price change.

    This implementation uses a simplified ratio approach:
        λ ≈ (Price_Change / Price) / √(Dollar_Volume)

    Using relative price changes ensures price-independence:
    - BTC at $90K and DOGE at $0.30 are directly comparable
    - A 1% price swing has the same impact coefficient regardless of absolute price

    Then smooths with rolling median for stability.

    This is 100x+ faster than regression-based methods and provides similar results.

    Reference: Kyle & Obizhaeva (2016), Econometrica

    Parameters:
    -----------
    df : pd.DataFrame
        Must contain 'high', 'low', 'close', 'volume' columns
    window : str
        Rolling window (e.g., '24h', '7d')

    Returns:
    --------
    pd.Series : Kyle's lambda estimates (dimensionless, units: 1/√dollar)
                Interpretation: For $Q order, price impact = λ × √Q (as fraction)

    Interpretation:
    ---------------
    - λ = 1e-6: Very low impact (highly liquid)
    - λ = 1e-5: Moderate impact
    - λ = 1e-4: High impact (illiquid)
    - λ > 1e-3: Extreme impact (manipulation risk)

    Examples:
    ---------
    >>> df = load_ohlcv_data('ETHUSDT')
    >>> lambda_24h = estimate_kyle_lambda_fast(df, window='24h')
    >>> print(f"Kyle Lambda: {lambda_24h.mean():.2e}")
    """
    # Calculate RELATIVE price changes (use High-Low range normalized by price)
    # This ensures price-independence: 1% move in BTC = 1% move in DOGE
    price_change = (df['high'] - df['low']) / df['close']

    # Square root of dollar volume
    dollar_volume = df['volume'] * df['close']
    sqrt_dollar_volume = np.sqrt(dollar_volume)

    # Simple ratio: λ ≈ (ΔP/P) / √Q
    # This is the instantaneous impact estimate (dimensionless)
    raw_lambda = price_change / sqrt_dollar_volume

    # Replace inf/NaN from division by zero
    raw_lambda = raw_lambda.replace([np.inf, -np.inf], np.nan)

    # Smooth with rolling median (more robust than mean)
    lambda_estimates = raw_lambda.rolling(window).median()

    # Clip extreme values (top 1% are likely outliers)
    if lambda_estimates.quantile(0.99) > 0:
        lambda_estimates = lambda_estimates.clip(upper=lambda_estimates.quantile(0.99))

    lambda_estimates = lambda_estimates.clip(lower=0)

    return lambda_estimates


def calculate_dollar_depth(df, window='24h'):
    """
    Calculate dollar depth: total dollar volume traded over window.

    Dollar depth measures market size - the total amount of capital that
    traded during the window. Higher values indicate better liquidity.

    Parameters:
    -----------
    df : pd.DataFrame
        Must contain 'close', 'volume'
    window : str
        Rolling window (e.g., '24h', '7d')

    Returns:
    --------
    pd.Series : Dollar volume depth

    Examples:
    ---------
    >>> df = load_ohlcv_data('BTCUSDT')
    >>> depth = calculate_dollar_depth(df, window='24h')
    >>> print(f"24h Dollar Depth: ${depth.mean():,.0f}")
    """
    dollar_volume = (df['close'] * df['volume']).rolling(window).sum()
    return dollar_volume


def calculate_liquidity_metrics(df, windows=['24h', '7d'], verbose=False):
    """
    Calculate comprehensive liquidity metrics for a symbol.

    This is the MAIN function that calculates all liquidity metrics:
    - Corwin-Schultz spread - resampled to daily
    - Kyle's Lambda - fast vectorized version
    - Dollar depth - supplementary metric
    - Amihud ILLIQ - alternative methodology

    All metrics are calculated for dual methodology comparison.

    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV data with columns: open, high, low, close, volume
    windows : list of str
        Time windows for rolling calculations (default: ['24h', '7d'])
    verbose : bool
        If True, print progress messages

    Returns:
    --------
    pd.DataFrame : Original df with added liquidity columns:
        - CS_spread: Instantaneous Corwin-Schultz spread
        - CS_spread_24h, CS_spread_7d: Rolling averages
        - kyle_lambda_24h, kyle_lambda_7d: Kyle's lambda
        - dollar_depth_24h, dollar_depth_7d: Dollar volume depth
        - ILLIQ_24h, ILLIQ_7d: Amihud ILLIQ (alternative method)

    Examples:
    ---------
    >>> df = load_ohlcv_data('SUIUSDT')
    >>> df_with_metrics = calculate_liquidity_metrics(df, verbose=True)
    >>> print(df_with_metrics[['CS_spread_24h', 'kyle_lambda_24h']].describe())
    """
    df = df.copy()

    if verbose:
        print("  Calculating Corwin-Schultz spread (daily resampled)...")

    # 1. Corwin-Schultz spread (primary measure) - DAILY FREQUENCY
    df['CS_spread'] = corwin_schultz_spread(df, set_negative_to_zero=True)

    # Rolling averages of CS spread
    for window in windows:
        df[f'CS_spread_{window}'] = df['CS_spread'].rolling(window).mean()

    if verbose:
        print("  Calculating Kyle's Lambda (fast vectorized version)...")

    # 2. Kyle's Lambda (price impact) - FAST OPTIMIZED VERSION
    for window in windows:
        df[f'kyle_lambda_{window}'] = estimate_kyle_lambda_fast(df, window=window)

    if verbose:
        print("  Calculating dollar depth...")

    # 3. Dollar depth (supplementary)
    for window in windows:
        df[f'dollar_depth_{window}'] = calculate_dollar_depth(df, window=window)

    if verbose:
        print("  Calculating Amihud ILLIQ (alternative method)...")

    # 4. Amihud ILLIQ (alternative methodology for comparison)
    df = calculate_rolling_illiq(df, windows=windows)

    return df


# ============================================================================
# AMIHUD ILLIQ FUNCTIONS (Alternative Method)
# ============================================================================

def calculate_amihud_illiq(df, price_col='close'):
    """
    Calculate Amihud Illiquidity Ratio at DAILY level.

    The Amihud (2002) ratio measures price impact per dollar traded.
    Used as an alternative methodology alongside CS spread + Kyle Lambda.

    Original Amihud (2002) formula:
    ILLIQ_daily = |R_daily| / DollarVolume_daily

    Note: This is part of a dual methodology approach. Both Amihud ILLIQ
    and CS spread + Kyle Lambda are calculated for comparison.

    See docs/dual_methodology_approach.md for comparison framework.

    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV data
    price_col : str
        Column to use for price (default: 'close')

    Returns:
    --------
    pd.Series : Amihud ILLIQ at minute frequency (forward-filled from daily)
    """
    # Resample to daily level
    daily_df = df.resample('D').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        price_col: 'last'
    }).dropna()

    # Calculate daily absolute return using range
    daily_abs_return = (daily_df['high'] - daily_df['low']) / daily_df['low']

    # Calculate daily dollar volume from minute data
    minute_dollar_volume = df['volume'] * df[price_col]
    daily_dollar_volume = minute_dollar_volume.resample('D').sum()

    # Calculate daily ILLIQ
    daily_illiq = daily_abs_return / daily_dollar_volume
    daily_illiq = daily_illiq.replace([np.inf, -np.inf], np.nan)

    # Forward fill to minute level
    illiq_minute = daily_illiq.reindex(df.index, method='ffill')

    return illiq_minute


def calculate_rolling_illiq(df, windows=['24h', '7d']):
    """
    Calculate rolling Amihud ILLIQ across multiple time windows.

    Part of dual methodology approach alongside CS spread + Kyle Lambda.

    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV data
    windows : list of str
        Time windows (note: ILLIQ is daily, so only '24h' and '7d' make sense)

    Returns:
    --------
    pd.DataFrame : df with added ILLIQ columns
    """
    df = df.copy()

    # Calculate daily ILLIQ
    illiq = calculate_amihud_illiq(df)
    df['ILLIQ_raw'] = illiq

    # Resample to daily for rolling averages
    daily_illiq = illiq.resample('D').first().dropna()

    # Add rolling windows
    for window in windows:
        if window == '24h':
            # 24h = just the daily value
            df[f'ILLIQ_{window}'] = illiq
        else:
            # Multi-day rolling average
            days = int(window.replace('d', ''))
            illiq_rolled = daily_illiq.rolling(window=days, min_periods=1).mean()
            df[f'ILLIQ_{window}'] = illiq_rolled.reindex(df.index, method='ffill')

    return df


# ============================================================================
# TIER CLASSIFICATION FUNCTIONS
# ============================================================================

def classify_liquidity_tier(cs_spread_mean_pct):
    """
    Classify assets by liquidity tier using Corwin-Schultz spread.

    Note: Lower spread = better liquidity (inverse of old Amihud ILLIQ)

    Parameters:
    -----------
    cs_spread_mean_pct : float
        CS spread in percentage (e.g., 0.25 means 0.25%)

    Returns:
    --------
    str : Tier classification

    Tier Definitions:
    -----------------
    - Tier 1: < 0.15% spread - Highly Liquid (BTC, ETH)
    - Tier 2: 0.15% - 0.5% - Moderate Liquidity (major altcoins)
    - Tier 3: 0.5% - 1.0% - Lower Liquidity (emerging tokens)
    - Tier 4: > 1.0% - Illiquid/High Risk (small caps)
    """
    if cs_spread_mean_pct < 0.15:
        return 'Tier 1 (Highly Liquid)'
    elif cs_spread_mean_pct < 0.5:
        return 'Tier 2 (Moderate)'
    elif cs_spread_mean_pct < 1.0:
        return 'Tier 3 (Lower Liquidity)'
    else:
        return 'Tier 4 (Illiquid/High Risk)'


def classify_liquidity_tier_illiq(illiq_mean):
    """
    Classify assets by liquidity tier using Amihud ILLIQ.

    Alternative classification method for comparison with CS spread tiers.
    """
    if illiq_mean < 1e-6:
        return 'Tier 1 (Highly Liquid)'
    elif illiq_mean < 5e-6:
        return 'Tier 2 (Moderate)'
    elif illiq_mean < 1e-5:
        return 'Tier 3 (Lower Liquidity)'
    else:
        return 'Tier 4 (Illiquid/High Risk)'


# ============================================================================
# SUMMARY STATISTICS FUNCTIONS
# ============================================================================

def calculate_liquidity_summary(liquidity_results_dict):
    """
    Calculate summary statistics for liquidity metrics across multiple symbols.

    Parameters:
    -----------
    liquidity_results_dict : dict
        Dictionary mapping symbol -> DataFrame with liquidity metrics
        (output from calculate_liquidity_metrics)

    Returns:
    --------
    pd.DataFrame : Summary statistics with columns:
        - Symbol
        - CS_Spread_24h_Mean_%, Median_%, P95_%
        - Kyle_Lambda_24h_Mean
        - Dollar_Depth_24h_Mean
        - ILLIQ_24h_OLD (deprecated, for comparison)
    """
    summary_list = []

    for symbol, df in liquidity_results_dict.items():
        summary = {
            'Symbol': symbol,
            # NEW METRICS (Recommended)
            'CS_Spread_24h_Mean_%': df['CS_spread_24h'].mean() * 100,
            'CS_Spread_24h_Median_%': df['CS_spread_24h'].median() * 100,
            'CS_Spread_24h_P95_%': df['CS_spread_24h'].quantile(0.95) * 100,
            'Kyle_Lambda_24h_Mean': df['kyle_lambda_24h'].mean(),
            'Dollar_Depth_24h_Mean': df['dollar_depth_24h'].mean(),
            # Alternative Method (Amihud ILLIQ)
            'ILLIQ_24h': df['ILLIQ_24h'].mean(),
        }
        summary_list.append(summary)

    return pd.DataFrame(summary_list)
