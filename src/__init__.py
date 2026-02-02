"""
CLOB Risk Modeling - Core Library

This package provides reusable functions for cryptocurrency derivatives risk modeling,
including liquidity metrics, volatility estimation, and OI limit calculations.

Modules:
--------
- liquidity_metrics: Liquidity estimation (Corwin-Schultz, Kyle Lambda, etc.)
- (future) volatility_metrics: Volatility estimation (Parkinson, Garman-Klass, etc.)
- (future) oi_limits: OI limit calculation and regime-based adjustments
"""

__version__ = "1.0.0"
__author__ = "CLOB Risk Team"

# Import main functions for convenient access
from .liquidity_metrics import (
    # Corwin-Schultz + Kyle Lambda Method
    corwin_schultz_spread,
    estimate_kyle_lambda_fast,
    calculate_dollar_depth,

    # Amihud ILLIQ Method (Alternative)
    calculate_amihud_illiq,
    calculate_rolling_illiq,

    # Main calculation function (calculates both methods)
    calculate_liquidity_metrics,

    # CLASSIFICATION
    classify_liquidity_tier,
    classify_liquidity_tier_illiq,
    calculate_liquidity_summary,
)

__all__ = [
    # CS + Kyle Lambda Method
    'corwin_schultz_spread',
    'estimate_kyle_lambda_fast',
    'calculate_dollar_depth',

    # Amihud Method (Alternative)
    'calculate_amihud_illiq',
    'calculate_rolling_illiq',

    # Main Function
    'calculate_liquidity_metrics',

    # Classification
    'classify_liquidity_tier',
    'classify_liquidity_tier_illiq',
    'calculate_liquidity_summary',
]
