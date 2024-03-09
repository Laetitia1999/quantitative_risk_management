import numpy as np
import pandas as pd


def compute_portfolio_returns(asset_prices, weights):
    asset_returns = asset_prices.pct_change()
    portfolio_returns = asset_returns.dot(weights)
    return portfolio_returns


def compute_annualized_volatility(returns_windowed):
    volatility_series = returns_windowed.std() * np.sqrt(252)
    return volatility_series
