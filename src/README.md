# CLOB Risk Modeling - Core Library (`src/`)

This directory contains reusable Python modules for cryptocurrency derivatives risk modeling. Functions are extracted from notebooks into importable modules for better code organization, testing, and reuse.

## Module Structure

```
src/
├── __init__.py                # Package initialization
├── liquidity_metrics.py       # Liquidity estimation functions
├── README.md                  # This file
└── __pycache__/              # Python bytecode cache
```

## Modules

### `liquidity_metrics.py`

Provides functions for calculating liquidity metrics from OHLCV data.

**NEW RECOMMENDED METRICS** (2026 Update):
- `corwin_schultz_spread()` - Bid-ask spread estimator from High-Low prices
- `estimate_kyle_lambda_fast()` - Market impact coefficient (fast vectorized)
- `calculate_dollar_depth()` - Total dollar volume traded
- `calculate_liquidity_metrics()` - **Main function** - calculates all metrics

**CLASSIFICATION**:
- `classify_liquidity_tier()` - Classify assets into liquidity tiers (1-4)
- `calculate_liquidity_summary()` - Generate summary statistics

**DEPRECATED** (kept for comparison):
- `calculate_amihud_illiq_DEPRECATED()` - Old Amihud ILLIQ ratio
- `calculate_rolling_illiq_DEPRECATED()` - Rolling ILLIQ averages
- `classify_liquidity_tier_OLD_DEPRECATED()` - Old tier classification

## Usage in Notebooks

### Import the module:

```python
import sys
from pathlib import Path

# Add src to Python path
src_path = Path('../../src')  # Adjust relative path as needed
sys.path.insert(0, str(src_path.resolve()))

# Import functions
from liquidity_metrics import (
    calculate_liquidity_metrics,
    classify_liquidity_tier,
    calculate_liquidity_summary
)
```

### Calculate liquidity metrics:

```python
# For a single symbol
df = load_ohlcv_data('BTCUSDT')
df_with_metrics = calculate_liquidity_metrics(df, windows=['24h', '7d'], verbose=True)

# Columns added:
# - CS_spread, CS_spread_24h, CS_spread_7d
# - kyle_lambda_24h, kyle_lambda_7d
# - dollar_depth_24h, dollar_depth_7d
# - ILLIQ_24h_DEPRECATED, ILLIQ_7d_DEPRECATED

# For multiple symbols
liquidity_results = {}
for symbol in ['BTCUSDT', 'ETHUSDT', 'SUIUSDT']:
    df = load_ohlcv_data(symbol)
    liquidity_results[symbol] = calculate_liquidity_metrics(df)

# Generate summary
summary_df = calculate_liquidity_summary(liquidity_results)
print(summary_df)
```

### Classify assets:

```python
# Classify based on CS spread
cs_spread_pct = 0.25  # 0.25%
tier = classify_liquidity_tier(cs_spread_pct)
print(tier)  # Output: "Tier 2 (Moderate)"
```

## Design Principles

### 1. **Separation of Concerns**
- Notebooks focus on **analysis and visualization**
- `src/` modules provide **reusable computation functions**
- Clear separation between "what to compute" (notebooks) and "how to compute" (src)

### 2. **Documentation**
- All functions have comprehensive docstrings
- Include parameters, returns, examples, and references
- Deprecation warnings clearly marked

### 3. **Backward Compatibility**
- Deprecated functions kept with `_DEPRECATED` suffix
- Clear warnings guide users to new implementations
- Side-by-side comparison possible during migration

### 4. **Performance**
- Vectorized pandas operations (100x+ faster than loops)
- Efficient resampling for daily-frequency calculations
- Memory-efficient rolling window operations

### 5. **Testing-Ready**
- Pure functions (input → output, no side effects)
- Minimal dependencies (numpy, pandas only)
- Easy to unit test with sample data

## Future Modules

Planned additions to `src/`:

- `volatility_metrics.py` - Parkinson, Garman-Klass, EWMA volatility
- `oi_limits.py` - OI limit calculation, regime adjustments
- `price_bands.py` - Dynamic price band construction
- `order_flow.py` - OFI, VPIN, trade classification
- `utils.py` - Common utilities (grid layout, data loading, etc.)

## References

**Academic Papers**:
- Corwin & Schultz (2012), "A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices", Journal of Finance
- Kyle & Obizhaeva (2016), "Market Microstructure Invariance: Empirical Hypotheses", Econometrica
- Amihud (2002), "Illiquidity and Stock Returns: Cross-Section and Time-Series Effects", Journal of Financial Markets

**Documentation**:
- `docs/liquidity_estimation_alternatives.md` - Comprehensive review of liquidity metrics
- `docs/amihud_deprecation_migration_guide.md` - Migration guide from Amihud to CS

## Contributing

When adding new functions to `src/`:

1. **Write comprehensive docstrings** (include examples!)
2. **Use type hints** where appropriate
3. **Add to `__init__.py`** for convenient imports
4. **Mark deprecated code** with `_DEPRECATED` suffix and warnings
5. **Update this README** with new module documentation

## Version History

- **v1.0.0** (2026-01-29): Initial release
  - `liquidity_metrics.py` extracted from Notebook 01
  - Includes CS spread, Kyle Lambda, Dollar Depth
  - Deprecates Amihud ILLIQ with comparison functions
