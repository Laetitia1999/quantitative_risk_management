import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # get data -> repository
    portfolio = pd.read_csv(r'input\crisis_portfolio.csv', delimiter=",", index_col="Date", parse_dates=["Date"])

    # loc -> model
    # dates -> get_begin_date + get_begin_date --> respository
    # Select portfolio asset prices for the middle of the crisis, 2008-2009
    # date dans fihcier config
    assetPrices = portfolio.loc['2008-01-01':'2009-12-31']


    # Plot portfolio's asset prices during this time
    # plot -> view
    assetPrices.plot().set_ylabel('Closing Prices, USD')
    plt.show()

    # Compute the portfolio's daily returns
    # compute -> view
    assetReturns = assetPrices.pct_change()
    # get_weights--> repository
    weights = [0.25, 0.25, 0.25, 0.25]
    # compute --> model
    portfolio_returns = assetReturns.dot(weights)



    # Plot portfolio returns
    portfolio_returns.plot().set_ylabel("Daily Return, %")
    plt.show()

    # Generate the covariance matrix from portfolio asset's returns
    Covariance = assetReturns.cov()

    # Annualize the covariance using 252 trading days per year
    Covariance = Covariance * 252

    # Display the covariance matrix --> view
    print(f'Covariance={Covariance}')

    portfolio_variance = np.transpose(weights) @ Covariance @ weights
    portfolio_volatility = np.sqrt(portfolio_variance)
    print(f'Portfolio volatility={portfolio_volatility}')

    # Calculate the 30-day rolling window of portfolio returns
    returns_windowed = portfolio_returns.rolling(30)

# model
    # Compute the annualized volatility series
    volatility_series = returns_windowed.std() * np.sqrt(252)
# view
    # Plot the portfolio volatility
    volatility_series.plot().set_ylabel("Annualized Volatility, 30-day Window")
    # view
    plt.show()


if __name__ == "__main__":
    main()
