#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:32:26 2024

@author: daleyfraser
"""

import yfinance as yf
import os

# Create a list of 20 major stock tickers (you can adjust this list)
top_stocks = [
    # 'AAPL',  # Apple Inc.
    # 'MSFT',  # Microsoft Corporation
    # 'GOOGL', # Alphabet Inc. (Google)
    # 'AMZN',  # Amazon.com Inc.
    # 'TSLA',  # Tesla Inc.
    
    # 'META',    # Meta Platforms Inc. (Facebook)
    # 'NVDA',  # NVIDIA Corporation
    'ABBV', # Berkshire Hathaway Inc.
    # 'V',     # Visa Inc.
    # 'JNJ',   # Johnson & Johnson
    
    # 'WMT',   # Walmart Inc.
    # 'PG',    # Procter & Gamble Co.
    # 'JPM',   # JPMorgan Chase & Co.
    # 'MA',    # Mastercard Incorporated
    # 'DIS',   # The Walt Disney Company
    
    # 'PYPL',  # PayPal Holdings Inc.
    # 'NFLX',  # Netflix Inc.
    # 'KO',    # The Coca-Cola Company
    # 'XOM',   # Exxon Mobil Corporation
    # 'CSCO'   # Cisco Systems Inc.
]

# Create a folder to store the CSV files

# Define the date range (2 years, 2024 is holdout trading set)
# folder_name = '22-23 data'
# start_date = '2022-01-01'
# end_date = '2023-12-31'

folder_name = '24_trading_time'
start_date = '2024-01-01'
end_date = '2024-9-01'

os.makedirs(folder_name, exist_ok=True)

# Loop through each stock and download the data, saving it to a CSV file
for ticker in top_stocks:
    # Download the stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    
    # Define the CSV file path
    file_path = os.path.join(folder_name, f'{ticker}_stock_data.csv')
    
    # Save the data to the CSV file
    stock_data.to_csv(file_path)
    
    print(f'Saved {ticker} data to {file_path}')