from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import pandas as pd
import random
import os
import sys
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
# define models to try
# *****************************************************************************
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

models = [
    # LogisticRegression(max_iter=1000),
    # DecisionTreeClassifier(),
    # RandomForestClassifier(),
    GradientBoostingClassifier(),
    # XGBClassifier(),
    # SVC(probability=True),
    # GaussianNB(),
    # MLPClassifier(max_iter=1000),
]

performances = []

for model in models:
# *****************************************************************************







# *****************************************************************************
# LOO
# *****************************************************************************
    accs = []
    aurocs = []
    sens = []
    specs = []
    
    tps = 0
    tns = 0
    fps = 0
    fns = 0
    
    for i in range(len(dfs)):
        val = dfs[i]  # leave one out
        train = pd.concat([dfs[j] for j in range(len(dfs)) if j != i], ignore_index=True)
# *****************************************************************************



# *****************************************************************************
# deine input/outputs and scale inputs
# *****************************************************************************
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
        
        tps += tp
        tns += tn
        fps += fp
        fns += fn
        
        # Calculate metrics
        accs.append(accuracy_score(y_val, y_pred))
        aurocs.append(roc_auc_score(y_val, y_proba))

        # sens.append(tp / (tp + fn))  # Sensitivity, recall, or true positive rate
        # specs.append(tn / (tn + fp)) # Specificity or true negative rate
        
# *****************************************************************************



# *****************************************************************************
# results
# *****************************************************************************
    accuracy = sum(accs)/len(accs)
    auroc = sum(aurocs)/len(aurocs)
    # sensitivity = sum(sens)/len(sens)
    # specificity = sum(specs)/len(specs)
    
    model = str(model)[:3]
    performances.append((model, round(accuracy, 4), round(auroc, 4), tps, tns, fps, fns))
    
all_samples = pd.concat(dfs, ignore_index=True)
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
print(tabulate(performances, headers=["Model", "Accuracy", "AUROC", "tp", "tn", "fp", "fn"], tablefmt="pretty"))
















