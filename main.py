import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import model
import view
import repository


def main():
    # Load data
    portfolio = repository.load_portfolio_data('input\crisis_portfolio.csv')

    # Plot asset prices
    view.plot_asset_prices(portfolio)

    # Calculate portfolio returns
    weights = repository.get_weights()
    portfolio_returns = model.compute_portfolio_returns(portfolio, weights)

    # Plot portfolio returns
    view.plot_portfolio_returns(portfolio_returns)

    # Compute covariance matrix
    asset_returns = portfolio.pct_change()
    Covariance = asset_returns.cov() * 252
    print(f'Covariance={Covariance}')

    # Calculate portfolio volatility
    portfolio_variance = np.transpose(weights) @ Covariance @ weights
    portfolio_volatility = np.sqrt(portfolio_variance)
    print(f'Portfolio volatility={portfolio_volatility}')

    # Calculate 30-day rolling window volatility
    returns_windowed = portfolio_returns.rolling(30)
    volatility_series = model.compute_annualized_volatility(returns_windowed)

    # Plot volatility
    view.plot_volatility(volatility_series)


if __name__ == "__main__":
    main()
