# *****************************************************************************
# idea: TBD, clearly define inputs and outputs
# *****************************************************************************
  


# *****************************************************************************
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from statsmodels.tsa.stattools import acf
import pandas_ta as ta
import sys
import os
# *****************************************************************************



# *****************************************************************************
# functions
# *****************************************************************************
def calculate_start_end(df, start_idx):
    period_df = df.iloc[start_idx]
    start_price = period_df['avg_price']
    start_date = period_df['Date']
    return start_price, start_date
def calculate_day_difference(base_date, target_date):
    base_date = pd.to_datetime(base_date, format='%m/%d/%Y')
    target_date = pd.to_datetime(target_date, format='%m/%d/%Y')
    return (target_date - base_date).days
def williams_r(high, low, close, length):
    highest_high = high.rolling(window=length).max()
    lowest_low = low.rolling(window=length).min()
    return ((highest_high - close) / (highest_high - lowest_low)) * -100
def calculate_vwap(high, low, close, volume):
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap
# *****************************************************************************



# *****************************************************************************
# iterate through all the stocks in the folder
# *****************************************************************************
# input_folder = '/Users/daleyfraser/Documents/cs/stocks/sept_2024/large_cap/raw_stock_data/22-23 data/'
# output_folder = '/Users/daleyfraser/Documents/cs/stocks/sept_2024/large_cap/feature_extraction/22-23/'

input_folder = '/Users/daleyfraser/Documents/cs/stocks/sept_2024/large_cap/raw_stock_data/24_trading_time/'
output_folder = '/Users/daleyfraser/Documents/cs/stocks/sept_2024/large_cap/feature_extraction/24/'

for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):
        input_path = os.path.join(input_folder, file_name)
        output_file_name = f'feature_extracted_{file_name}'
        output_path = os.path.join(output_folder, output_file_name)
# *****************************************************************************



# *****************************************************************************
# pull data and define output df
# *****************************************************************************
    df = pd.read_csv(input_path)
    df['avg_price'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    res = pd.DataFrame()
# *****************************************************************************
    
    
    
# *****************************************************************************
# iterate through time period, ensure at least 20 market days
# *****************************************************************************
    for idx, row in df.iterrows():
        if idx < 20 or idx+10 >= len(df):
            continue  # need data from past month, 20 days is 4, 5 day weeks
                        # also need data ~ 2 weeks in advance for output
        
# *****************************************************************************
    
    
    
# *****************************************************************************
# data for inputs, about 1 month (20 market days)
# *****************************************************************************
        period_df = df.iloc[idx-20: idx]
        values = period_df['avg_price'].tolist()
# *****************************************************************************
    
    
    
# *****************************************************************************
# output data: following about 2 weeks (10 market days)
# 
# OUTPUT definition: binary --> 0 or 1
# a 0 signifies you should not buy the stock, it will go down
# 1 means that the stock rises at the 5% within the next 2 weeks on any day
# *****************************************************************************
        output_df = df.iloc[idx+1: idx+10]
        prices = output_df['avg_price'].tolist()
        high = max(prices)
        
        output = None  # 0 or 1
        
        if high / row['avg_price'] > 1.05:
            output = 1
        else:
            output = 0
        
# *****************************************************************************
    
    
    
    
# *****************************************************************************
# basic stats on preivous 20 days
# *****************************************************************************
        values = np.array(values)
        minimum = np.min(values)
        mean = np.mean(values)
        maximum = np.max(values)
        percentile_25 = np.percentile(values, 25)
        percentile_75 = np.percentile(values, 75)
        skewness = skew(values)
        std_dev = np.std(values)
        kurt = kurtosis(values)
        shannon_entropy = entropy(values)
        median = np.median(values)
        mean_abs_dev = np.mean(np.abs(values - np.mean(values)))
        coeff_of_var = std_dev / mean
        percentile_10 = np.percentile(values, 10)
        percentile_90 = np.percentile(values, 90)
        autocorr = acf(values, nlags=1)[1]
        cumulative_return = (values[-1] - values[0]) / values[0]
# *****************************************************************************
    
    
    
# *****************************************************************************
# curr price relative to previous prices
# *****************************************************************************
        curr_to_avg_one_month = df['avg_price'].iloc[idx] / mean
        prev_two_week = values[-10:]  # Last 10 days within the 20-day period
        prev_two_week_mean = np.mean(prev_two_week)
        curr_to_avg_one_two_week = df['avg_price'].iloc[idx] / prev_two_week_mean
        prev_one_week = values[-5:]  # Last 5 days within the 3-month period
        prev_one_week_mean = np.mean(prev_one_week)
        curr_to_avg_one_one_week = df['avg_price'].iloc[idx] / prev_one_week_mean
# *****************************************************************************
    
    
    
# *****************************************************************************
# adding more inputs
# *****************************************************************************
        momentum = df['avg_price'].pct_change(periods=5).iloc[idx]
        rsi = ta.rsi(pd.Series(values), length=14).iloc[-1]
        bbands_df = ta.bbands(pd.Series(values), length=20)
        upper_band_current = bbands_df['BBU_20_2.0'].iloc[-1]
        middle_band_current = bbands_df['BBM_20_2.0'].iloc[-1]
        lower_band_current = bbands_df['BBL_20_2.0'].iloc[-1]
        rolling_mean_7 = df['avg_price'].rolling(window=7).mean().iloc[idx]
        rolling_std_7 = df['avg_price'].rolling(window=7).std().iloc[idx]
        rolling_mean_14 = df['avg_price'].rolling(window=14).mean().iloc[idx]
        rolling_std_14 = df['avg_price'].rolling(window=14).std().iloc[idx]
        ema_7 = ta.ema(pd.Series(values), length=7).iloc[-1]
        ema_14 = ta.ema(pd.Series(values), length=14).iloc[-1]
        roc_7 = ta.roc(pd.Series(values), length=7).iloc[-1]
        roc_14 = ta.roc(pd.Series(values), length=14).iloc[-1]
        volume = df['Volume'].iloc[idx]
        volatility = np.std(np.diff(values))
        avg_volume_20 = df['Volume'].rolling(window=20).mean().iloc[idx]
        rvol = volume / avg_volume_20
        obv = ta.obv(df['avg_price'], df['Volume']).iloc[idx]
        cmf = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20).iloc[idx]
        vwap = calculate_vwap(df['High'], df['Low'], df['Close'], df['Volume']).iloc[idx]
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=14).iloc[idx]
        williams_r_value = williams_r(df['High'], df['Low'], df['Close'], length=14).iloc[idx]
        stoch_df = ta.stoch(df['High'], df['Low'], df['Close'], fast_k=14, slow_k=3, slow_d=3)
        stoch_k_current = stoch_df['STOCHk_14_3_3'].iloc[-1]
        stoch_d_current = stoch_df['STOCHd_14_3_3'].iloc[-1]
        cci = ta.cci(df['High'], df['Low'], df['Close'], length=20).iloc[idx]
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14).iloc[idx]
        high = df['High'].rolling(window=14).max().iloc[idx]
        low = df['Low'].rolling(window=14).min().iloc[idx]
        fib_levels = {
            'Fib_23.6': high - 0.236 * (high - low),
            'Fib_38.2': high - 0.382 * (high - low),
            'Fib_50': high - 0.5 * (high - low),
            'Fib_61.8': high - 0.618 * (high - low),
            'Fib_76.4': high - 0.764 * (high - low)
        }
        parabolic_sar = ta.psar(df['High'], df['Low'], df['Close']).iloc[idx]
        adj_close_momentum = df['Adj Close'].pct_change(periods=5).iloc[idx]
        adj_close_return = (df['Adj Close'].iloc[idx] - df['Adj Close'].iloc[idx-6]) / df['Adj Close'].iloc[idx-6]
        adj_close_7d_ma = df['Adj Close'].rolling(window=7).mean().iloc[idx]
        adj_close_14d_ma = df['Adj Close'].rolling(window=14).mean().iloc[idx]
        adj_close_volatility = df['Adj Close'].rolling(window=20).std().iloc[idx]
# *****************************************************************************
    
    
    
# *****************************************************************************
# add to df
# *****************************************************************************

        start_price, start_date= calculate_start_end(df, idx)
        result_date = df['Date'].iloc[idx]
        print('making new row', idx) if idx%100 == 0 else None
        new_row = {
            'date': result_date,
            'buy?': output,
            'todays_price': row['avg_price'],
            'Minimum': minimum,'Maximum': maximum, '25th Percentile': percentile_25,
            '75th Percentile': percentile_75, 'Skewness': skewness, 'Standard Deviation': std_dev,
            'Kurtosis': kurt, 'Shannon Entropy': shannon_entropy, 'Median': median,
            'Mean Absolute Deviation (MAD)': mean_abs_dev, 'Coefficient of Variation (CV)': coeff_of_var,
            '10th Percentile': percentile_10, '90th Percentile': percentile_90, 'Autocorrelation': autocorr,
            'Cumulative Return': cumulative_return,
            'curr_to_avg_one_month': curr_to_avg_one_month,
            'curr_to_avg_one_two_week': curr_to_avg_one_two_week,
            'curr_to_avg_one_one_week': curr_to_avg_one_one_week,
            'momentum': momentum, 'rsi': rsi, 'upper_band_current': upper_band_current,
            'middle_band_current': middle_band_current, 'lower_band_current': lower_band_current,
            'rolling_mean_7': rolling_mean_7, 'rolling_std_7': rolling_std_7,
            'rolling_mean_14': rolling_mean_14, 'rolling_std_14': rolling_std_14,
            'ema_7': ema_7, 'ema_14': ema_14, 'roc_7': roc_7, 'roc_14': roc_14,
            'Volume': volume, 'Volatility': volatility, 'Relative Volume (RVOL)': rvol,
            'On-Balance Volume (OBV)': obv, 'Williams %R': williams_r_value,
            'Stochastic Oscillator %K': stoch_k_current, 'Stochastic Oscillator %D': stoch_d_current,
            'Commodity Channel Index (CCI)': cci, 
            'ADX1': adx.iloc[1],
            'ADX2': adx.iloc[2],
            'Chaikin Money Flow (CMF)': cmf, 'Average True Range (ATR)': atr,
            'Fib_23.6': fib_levels['Fib_23.6'], 'Fib_38.2': fib_levels['Fib_38.2'],
            'Fib_50': fib_levels['Fib_50'], 'Fib_61.8': fib_levels['Fib_61.8'],
            'Fib_76.4': fib_levels['Fib_76.4'], 
            'Parabolic SAR 2': parabolic_sar.iloc[2],
            'Parabolic SAR 3': parabolic_sar.iloc[3],
            'VWAP': vwap,
            'Adj Close Momentum': adj_close_momentum, 'Adj Close Return': adj_close_return,
            'Adj Close 7d MA': adj_close_7d_ma, 'Adj Close 14d MA': adj_close_14d_ma,
            'Adj Close Volatility': adj_close_volatility
        }
    
        new_row_df = pd.DataFrame([new_row])
        res = pd.concat([res, new_row_df], ignore_index=True)
        idx += 1
     
    # res.dropna(inplace=True)
    res.to_csv(output_path, index=False)
    print(f"Processed CSV saved to {output_path}")
    print(res)
    print('\n')
    columns_with_nan = res.columns[res.isna().any()].tolist()
    print(columns_with_nan)
# *****************************************************************************
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


