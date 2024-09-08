# *****************************************************************************
# idea: tune to get highest acc and auroc,
# but also very low fp
# *****************************************************************************
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import pandas as pd
from xgboost import XGBClassifier
import random
import os
import sys
import numpy as np
# *****************************************************************************



# *****************************************************************************
# data
# *****************************************************************************
folder = '/Users/daleyfraser/Documents/cs/stocks/sept_2024/large_cap/feature_extraction/22-23'
dfs = []

for file_name in os.listdir(folder):
    # Check if the file is a CSV
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder, file_name)
        # Read the CSV into a DataFrame and append it to the list
        df = pd.read_csv(file_path)
        dfs.append(df)

test_folder = '/Users/daleyfraser/Documents/cs/stocks/sept_2024/large_cap/feature_extraction/24'
test_dfs = []
for file_name in os.listdir(test_folder):
    # Check if the file is a CSV
    if file_name.endswith('.csv'):
        file_path = os.path.join(test_folder, file_name)
        # Read the CSV into a DataFrame and append it to the list
        df = pd.read_csv(file_path)
        test_dfs.append(df)

train = pd.concat([dfs[j] for j in range(len(dfs))], ignore_index=True)
test = pd.concat([test_dfs[j] for j in range(len(test_dfs))], ignore_index=True)
# *****************************************************************************



# *****************************************************************************
# define inputs to train on
# *****************************************************************************
inputs = ['todays_price', 'Minimum', 'Maximum', '25th Percentile',
       '75th Percentile', 'Skewness', 'Standard Deviation', 'Kurtosis',
       'Shannon Entropy', 'Median', 'Mean Absolute Deviation (MAD)',
       'Coefficient of Variation (CV)', '10th Percentile', '90th Percentile',
       'Autocorrelation', 'Cumulative Return', 'curr_to_avg_one_month',
       'curr_to_avg_one_two_week', 'curr_to_avg_one_one_week', 'momentum',
       'rsi', 'upper_band_current', 'middle_band_current',
       'lower_band_current', 'rolling_mean_7', 'rolling_std_7',
       'rolling_mean_14', 'rolling_std_14', 'ema_7', 'ema_14', 'roc_7',
       'roc_14', 'Volume', 'Volatility', 'Relative Volume (RVOL)',
       'On-Balance Volume (OBV)', 'Williams %R', 'Stochastic Oscillator %K',
       'Stochastic Oscillator %D', 'Commodity Channel Index (CCI)', 'ADX1',
       'ADX2', 'Chaikin Money Flow (CMF)', 'Average True Range (ATR)',
       'Fib_23.6', 'Fib_38.2', 'Fib_50', 'Fib_61.8', 'Fib_76.4',
       'Parabolic SAR 2', 'Parabolic SAR 3', 'VWAP', 'Adj Close Momentum',
       'Adj Close Return', 'Adj Close 7d MA', 'Adj Close 14d MA',
       'Adj Close Volatility']
# *****************************************************************************


    
# *****************************************************************************



# *****************************************************************************
# deine input/outputs and scale inputs
# *****************************************************************************
x_train = train[inputs]
y_train = train['buy?']
x_test = test[inputs]
y_test = test['buy?']

# Scale features to be between 0 and 1
scaler = MinMaxScaler()
x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=inputs)
x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=inputs)
# *****************************************************************************



# *****************************************************************************
# train model
# *****************************************************************************
# Calculate the ratio of the negative and positive classes
neg_class = np.bincount(train['buy?'])[0]
pos_class = np.bincount(train['buy?'])[1]
scale_pos_weight = (neg_class / pos_class)*1

# Create and train the model with `scale_pos_weight`
model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=2, scale_pos_weight=scale_pos_weight)

model.fit(x_train_scaled, y_train)
y_proba = model.predict_proba(x_test_scaled)[:, 1]  
# *****************************************************************************



# *****************************************************************************
# find optimal threshold
# *****************************************************************************
fpr, tpr, thresholds = roc_curve(y_test, y_proba, drop_intermediate=False)
yj = sorted(zip(tpr-fpr, thresholds))[-1][1]
# print(f'Optimal Threshold by Youden\'s J: {yj}\n')

# Make predictions based on the optimal threshold
y_pred = (y_proba >= yj).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# *****************************************************************************



# *****************************************************************************
# results
# *****************************************************************************
accuracy = accuracy_score(y_test, y_pred)
auroc = roc_auc_score(y_test, y_proba)

performances = []
performances.append((round(accuracy, 4), round(auroc, 4), tp, tn, fp, fn))

                
performances.sort(key=lambda x: (x[2], x[1], x[3], x[4]), reverse=True)
all_samples =test
print('Distribution:')
value_counts = all_samples['buy?'].value_counts()
higher = 0
total = 0
for index, value in value_counts.items():
    higher = max(value, higher)
    total += value
    print(f"{index}: {value}")
print('standard to beat:', higher / total)

print('\nAverage performance')
print(tabulate(performances, headers=["Accuracy", "AUROC", "tp", "tn", "fp", "fn"], tablefmt="pretty"))
















