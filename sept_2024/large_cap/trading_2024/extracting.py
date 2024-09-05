
# *****************************************************************************
#******************************************************************************import pandas as pd
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from statsmodels.tsa.stattools import acf
import pandas_ta as ta
import sys
# ******************************************************************************



# ******************************************************************************
# functions
# ******************************************************************************
def calculate_start_end(data, start_idx):
    period_data = data.iloc[start_idx]
    start_price = period_data['avg_price']
    start_date = period_data['Date']
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
def process_multiple_datasets(datasets):
    for dataset in datasets:
        process_dataset(dataset['input_path'], dataset['output_path'])
# ******************************************************************************



# ******************************************************************************
# main processing function
# ******************************************************************************
def process_dataset(input_path, output_path):
    data = pd.read_csv(input_path)
    print(data)

    # Initialize an empty DataFrame for results
    res = pd.DataFrame(columns=[
        'date', 'Minimum', 'Mean', 'Maximum', '25th Percentile', '75th Percentile',
        'Skewness', 'Standard Deviation', 'Kurtosis', 'Shannon Entropy',
        'Median', 'Mean Absolute Deviation (MAD)', 'Coefficient of Variation (CV)',
        '10th Percentile', '90th Percentile', 'Autocorrelation', 'Cumulative Return',
        'start_price', 'start_date', 'end_price', 'end_date', 'percent_change',
        'prev_day_to_avg_three_month', 'prev_day_to_avg_one_month', 'prev_day_to_avg_one_two_week',
        'prev_day_to_avg_one_one_week', 'momentum', 'rsi', 'macd_current',
        'macd_signal_current', 'upper_band_current', 'middle_band_current', 'lower_band_current',
        'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_std_14',
        'ema_7', 'ema_14', 'roc_7', 'roc_14',
        'Volume', 'Volatility', 'Relative Volume (RVOL)', 'On-Balance Volume (OBV)',
        'Williams %R', 'Stochastic Oscillator %K', 'Stochastic Oscillator %D',
        'Commodity Channel Index (CCI)', 'ADX',
        'Chaikin Money Flow (CMF)', 'Average True Range (ATR)',
        'Fib_23.6', 'Fib_38.2', 'Fib_50', 'Fib_61.8', 'Fib_76.4', 'Parabolic SAR',
        'VWAP'
    ])



    curr = 100 # process 3 months data
    n_days = len(data)

    while curr +1<= n_days: # out of bounds unless we do curr + 13 <= n_days
        
        period_data = data.iloc[curr-90: curr]
        
        values = period_data['avg_price'].tolist()
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


#******************************************************************************
#******************************************************************************
# adding more inputs
#******************************************************************************
#******************************************************************************

        prev_month = values[-30:]  # Last 30 days within the 3-month period
        prev_month_mean = np.mean(prev_month)
        prev_day_to_avg_one_month = data['avg_price'].iloc[curr-1] / prev_month_mean
        
        if(prev_month_mean) == 0:
            print("Avg was 0")
            sys.exit()
    
        prev_two_week = values[-14:]  # Last 14 days within the 3-month period
        prev_two_week_mean = np.mean(prev_two_week)
        prev_day_to_avg_one_two_week = data['avg_price'].iloc[curr-1] / prev_two_week_mean
        
        prev_one_week = values[-7:]  # Last 14 days within the 3-month period
        prev_one_week_mean = np.mean(prev_one_week)
        prev_day_to_avg_one_one_week = data['avg_price'].iloc[curr-1] / prev_one_week_mean
        
#******************************************************************************
#******************************************************************************
# end adding more inputs
#******************************************************************************
#******************************************************************************      



#******************************************************************************
#******************************************************************************
# even more inputs
#******************************************************************************
#******************************************************************************

        # Price Momentum
        momentum = data['avg_price'].pct_change(periods=5).iloc[curr-1]
        
        # Relative Strength Index (RSI)
        rsi = ta.rsi(pd.Series(values), length=14).iloc[-1]
        
        # Moving Average Convergence Divergence (MACD)
        macd_df = ta.macd(pd.Series(values), fast=12, slow=26, signal=9)
        macd_current = macd_df['MACD_12_26_9'].iloc[-1]
        macd_signal_current = macd_df['MACDs_12_26_9'].iloc[-1]
        
        # Bollinger Bands
        bbands_df = ta.bbands(pd.Series(values), length=20)
        upper_band_current = bbands_df['BBU_20_2.0'].iloc[-1]
        middle_band_current = bbands_df['BBM_20_2.0'].iloc[-1]
        lower_band_current = bbands_df['BBL_20_2.0'].iloc[-1]
        
        # Rolling Mean and Standard Deviation for 7 and 14 days
        rolling_mean_7 = data['avg_price'].rolling(window=7).mean().iloc[curr-1]
        rolling_std_7 = data['avg_price'].rolling(window=7).std().iloc[curr-1]
        rolling_mean_14 = data['avg_price'].rolling(window=14).mean().iloc[curr-1]
        rolling_std_14 = data['avg_price'].rolling(window=14).std().iloc[curr-1]
        
        # Exponential Moving Average (EMA) for 7 and 14 days
        ema_7 = ta.ema(pd.Series(values), length=7).iloc[-1]
        ema_14 = ta.ema(pd.Series(values), length=14).iloc[-1]
        
        # Price Rate of Change (ROC) for 7 and 14 days
        roc_7 = ta.roc(pd.Series(values), length=7).iloc[-1]
        roc_14 = ta.roc(pd.Series(values), length=14).iloc[-1]
        
        # Volume (assuming 'Volume' is a column in your dataset)
        volume = data['Volume'].iloc[curr-1]
        
        # Volatility (standard deviation of price changes)
        volatility = np.std(np.diff(values))
        
        # Relative Volume (RVOL) - ratio of current volume to average volume
        avg_volume_50 = data['Volume'].rolling(window=50).mean().iloc[curr-1]
        rvol = volume / avg_volume_50
        
        # On-Balance Volume (OBV)
        obv = ta.obv(data['avg_price'], data['Volume']).iloc[curr-1]
        cmf = ta.cmf(data['High'], data['Low'], data['Close'], data['Volume'], length=20).iloc[curr-1]
        vwap = calculate_vwap(data['High'], data['Low'], data['Close'], data['Volume']).iloc[curr-1]

       # Average True Range (ATR)
        atr = ta.atr(data['High'], data['Low'], data['Close'], length=14).iloc[curr-1]

        # Williams %R
        williams_r_value = williams_r(data['High'], data['Low'], data['Close'], length=14).iloc[curr-1]

        # Stochastic Oscillator
        stoch_df = ta.stoch(data['High'], data['Low'], data['Close'], fast_k=14, slow_k=3, slow_d=3)
        stoch_k_current = stoch_df['STOCHk_14_3_3'].iloc[-1]
        stoch_d_current = stoch_df['STOCHd_14_3_3'].iloc[-1]
        # Commodity Channel Index (CCI)
        cci = ta.cci(data['High'], data['Low'], data['Close'], length=20).iloc[curr-1]

        # Average Directional Index (ADX)
        adx = ta.adx(data['High'], data['Low'], data['Close'], length=14).iloc[curr-1]

        # Fibonacci Retracement Levels
        high = data['High'].rolling(window=14).max().iloc[curr-1]
        low = data['Low'].rolling(window=14).min().iloc[curr-1]
        fib_levels = {
            'Fib_23.6': high - 0.236 * (high - low),
            'Fib_38.2': high - 0.382 * (high - low),
            'Fib_50': high - 0.5 * (high - low),
            'Fib_61.8': high - 0.618 * (high - low),
            'Fib_76.4': high - 0.764 * (high - low)
        }

        # Parabolic SAR
        parabolic_sar = ta.psar(data['High'], data['Low'], data['Close']).iloc[curr-1]

       
        # Adjusted Close Momentum
        adj_close_momentum = data['Adj Close'].pct_change(periods=5).iloc[curr-1]
        # Adjusted Close Return
        adj_close_return = (data['Adj Close'].iloc[curr-1] - data['Adj Close'].iloc[curr-6]) / data['Adj Close'].iloc[curr-6]
        
        # Adjusted Close Moving Averages
        adj_close_7d_ma = data['Adj Close'].rolling(window=7).mean().iloc[curr-1]
        adj_close_14d_ma = data['Adj Close'].rolling(window=14).mean().iloc[curr-1]
        adj_close_30d_ma = data['Adj Close'].rolling(window=30).mean().iloc[curr-1]
        
        # Adjusted Close Volatility
        adj_close_volatility = data['Adj Close'].rolling(window=30).std().iloc[curr-1]

        
# ******************************************************************************
# end more inputs
# ******************************************************************************

        start_price, start_date= calculate_start_end(data, curr)
        result_date = data['Date'].iloc[curr]
        print('making new row', curr) if curr%100 == 0 else None
        new_row = {
            'date': result_date,
            'Minimum': minimum, 'Mean': mean, 'Maximum': maximum, '25th Percentile': percentile_25,
            '75th Percentile': percentile_75, 'Skewness': skewness, 'Standard Deviation': std_dev,
            'Kurtosis': kurt, 'Shannon Entropy': shannon_entropy, 'Median': median,
            'Mean Absolute Deviation (MAD)': mean_abs_dev, 'Coefficient of Variation (CV)': coeff_of_var,
            '10th Percentile': percentile_10, '90th Percentile': percentile_90, 'Autocorrelation': autocorr,
            'Cumulative Return': cumulative_return, 'start_price': start_price, 'start_date': start_date, 
            'prev_day_to_avg_one_month': prev_day_to_avg_one_month,
            'prev_day_to_avg_one_two_week': prev_day_to_avg_one_two_week,
            'prev_day_to_avg_one_one_week': prev_day_to_avg_one_one_week,
            'momentum': momentum, 'rsi': rsi, 'macd_current': macd_current,
            'macd_signal_current': macd_signal_current, 'upper_band_current': upper_band_current,
            'middle_band_current': middle_band_current, 'lower_band_current': lower_band_current,
            'rolling_mean_7': rolling_mean_7, 'rolling_std_7': rolling_std_7,
            'rolling_mean_14': rolling_mean_14, 'rolling_std_14': rolling_std_14,
            'ema_7': ema_7, 'ema_14': ema_14, 'roc_7': roc_7, 'roc_14': roc_14,
            'Volume': volume, 'Volatility': np.std(np.diff(values)), 'Relative Volume (RVOL)': rvol,
            'On-Balance Volume (OBV)': obv, 'Williams %R': williams_r_value,
            'Stochastic Oscillator %K': stoch_k_current, 'Stochastic Oscillator %D': stoch_d_current,
            'Commodity Channel Index (CCI)': cci, 'ADX': adx,
            'Chaikin Money Flow (CMF)': cmf, 'Average True Range (ATR)': atr,
            'Fib_23.6': fib_levels['Fib_23.6'], 'Fib_38.2': fib_levels['Fib_38.2'],
            'Fib_50': fib_levels['Fib_50'], 'Fib_61.8': fib_levels['Fib_61.8'],
            'Fib_76.4': fib_levels['Fib_76.4'], 'Parabolic SAR': parabolic_sar,
            'VWAP': vwap,
            'Adj Close Momentum': adj_close_momentum, 'Adj Close Return': adj_close_return,
            'Adj Close 7d MA': adj_close_7d_ma, 'Adj Close 14d MA': adj_close_14d_ma,
            'Adj Close 30d MA': adj_close_30d_ma, 'Adj Close Volatility': adj_close_volatility
        }

        new_row_df = pd.DataFrame([new_row])
        res = pd.concat([res, new_row_df], ignore_index=True)

        curr += 1
 
    # res.dropna(inplace=True)
    # res.to_csv(output_path, index=False)
    print(res)
    print(f"Processed CSV saved to {output_path}")
# ******************************************************************************



# ******************************************************************************
if __name__ == "__main__":
    datasets = [
        # apple
        # {
        #     'input_path': 'C:/Users/dfras/Documents/doge_coin/nvidia/apple_07_01_2024_to_05_28_2024.csv',
        #     'output_path': 'C:/Users/dfras/Documents/doge_coin/nvidia/masterCSV/apple.csv'
        # },
        
       #apple
        {
            'input_path': '/Users/daleyfraser/Documents/cs/stocks/sept_2024/large_cap/raw_stock_data/22-23 data/AAPL_stock_data.csv',
            'output_path': '/Users/daleyfraser/Documents/cs/stocks/sept_2024/large_cap/feature_extraction/22-23/'
            }
        
       
    ]

    process_multiple_datasets(datasets)








