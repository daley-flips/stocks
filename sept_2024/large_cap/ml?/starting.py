from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import pandas as pd
import os
import sys
# *****************************************************************************



# *****************************************************************************
# data
# *****************************************************************************
train = pd.read_csv('/Users/daleyfraser/Documents/cs/stocks/sept_2024/large_cap/feature_extraction/22-23/feature_extracted_AAPL_stock_data.csv')
val = pd.read_csv('/Users/daleyfraser/Documents/cs/stocks/sept_2024/large_cap/feature_extraction/22-23/feature_extracted_ABBV_stock_data.csv')
# *****************************************************************************



# *****************************************************************************

# ensure no nan values
# *****************************************************************************
samples = pd.concat([train, val])
has_nan = samples.isnull().any().any()
if has_nan:
    print("The DataFrame contains NaN values.")
    sys.exit()
else:
    print("")
# *****************************************************************************



# *****************************************************************************
# scale variables to train on
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
    

x_train = train[inputs]
y_train = train['buy?']
x_val = val[inputs]
y_val = val['buy?']

# Scale features to be between 0 and 1
scaler = MinMaxScaler()
x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=inputs)
x_val_scaled = pd.DataFrame(scaler.transform(x_val), columns=inputs)
# *****************************************************************************



# *****************************************************************************
# train model
# *****************************************************************************
performances = []

model = LogisticRegression()
model.fit(x_train_scaled, y_train)
y_proba = model.predict_proba(x_val_scaled)[:, 1]  
# *****************************************************************************



# *****************************************************************************
# find optimal threshold
# *****************************************************************************
fpr, tpr, thresholds = roc_curve(y_val, y_proba, drop_intermediate=False)
yj = sorted(zip(tpr-fpr, thresholds))[-1][1]
# print(f'Optimal Threshold by Youden\'s J: {yj}\n')

# Make predictions based on the optimal threshold
y_pred = (y_proba >= yj).astype(int)

tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

# Calculate metrics
accuracy = accuracy_score(y_val, y_pred)
sensitivity =tp / (tp + fn)  # Sensitivity, recall, or true positive rate
specificity = tn / (tn + fp) # Specificity or true negative rate
auroc = roc_auc_score(y_val, y_proba)

# print(f'Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}')
# print(f'Accuracy: {accuracy}')
# print(f'Sensitivity: {sensitivity}')
# print(f'Specificity: {specificity}')
# print(f'AUROC: {auroc}')
# *****************************************************************************



# *****************************************************************************
# results
# *****************************************************************************
performances.append(('Validation', round(accuracy, 4), round(auroc, 4), round(sensitivity, 4), round(specificity, 4)))
print('\nResults for Logistic Regression')
print('Distribution:')
value_counts = val['buy?'].value_counts()
higher = 0
total = 0
for index, value in value_counts.items():
    higher = max(value, higher)
    total += value
    print(f"{index}: {value}")
print('standard to beat:', higher / total)
print(tabulate(performances, headers=["Model", "Accuracy", "AUROC", "Sensitivity", "Specificity", "Params"], tablefmt="pretty"))
# print(f'Model Coefficients:\n{model.coef_}')
print(f'Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}')

















