import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf

def fetch_and_prepare_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data found for {ticker} in the specified date range.")
    
    data['Adj Close'] = data.get('Adj Close', data['Close'])
    
    if 'Open' not in data.columns:
        raise ValueError(f"'Open' column is missing for {ticker} data.")
    
    data['Adjusted_Price'] = data['Adj Close']
    data.loc[data.index[0], 'Adjusted_Price'] = data['Open'].iloc[0]
    
    return data

def calculate_log_returns(data):
    data['Log_Return'] = np.log(data['Adjusted_Price'] / data['Adjusted_Price'].shift(1))
    return data.dropna()  

def calculate_cumulative_stats(log_returns, trading_days):
    mu_daily = log_returns.mean()
    sigma_daily = log_returns.std()
    mu_cumulative = trading_days * mu_daily
    sigma_cumulative = np.sqrt(trading_days) * sigma_daily
    return mu_cumulative, sigma_cumulative

def calculate_probability(threshold, mu_cumulative, sigma_cumulative):
    z_score = (threshold - mu_cumulative) / sigma_cumulative
    return 1 - norm.cdf(z_score)

# config 
start_date = '2024-10-29'
end_date = '2024-12-02'
tickers = ['DAL', 'HON']
threshold = np.log(1.10)  # 10% 

results = []
trading_days = None

for ticker in tickers:
    data = fetch_and_prepare_data(ticker, start_date, end_date)
    data = calculate_log_returns(data)
    
    if trading_days is None:
        trading_days = len(data)  # all stocks have the same number of trading days
    
    mu_cumulative, sigma_cumulative = calculate_cumulative_stats(data['Log_Return'], trading_days)
    probability = calculate_probability(threshold, mu_cumulative, sigma_cumulative)
    
    results.append({
        'Ticker': ticker,
        'Probability of 10% Gain (%)': round(probability * 100, 2)
    })

# Display Results
for result in results:
    print(f"Probability of {result['Ticker']} having a 10% gain: {result['Probability of 10% Gain (%)']}%")

