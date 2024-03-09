import pandas as pd

def load_portfolio_data(file_path):
    portfolio = pd.read_csv(file_path, delimiter=",", index_col="Date", parse_dates=["Date"])
    return portfolio.loc['2008-01-01':'2009-12-31']

def get_weights():
    return [0.25, 0.25, 0.25, 0.25]
