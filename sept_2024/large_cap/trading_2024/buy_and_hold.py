# *****************************************************************************
# idea: 
# you have $100 on January 1st, 2024 you put $5 into the 20 most popular stocks
# how much would you make if you sold them all on September 1st, 2024
# *****************************************************************************



# *****************************************************************************
import pandas as pd
import os
import sys
# *****************************************************************************



# *****************************************************************************
init_port = 100
port = init_port  # portfolio
buy = 5  # Price Per Stock
# *****************************************************************************



# *****************************************************************************
# iterate through files in 24_trading_time
# *****************************************************************************
directory_path = '/Users/daleyfraser/Documents/cs/stocks/sept_2024/large_cap/raw_stock_data/24_trading_time'
for file in os.listdir(directory_path):
    
    if file.startswith('.'):
        continue  #  skip hidden files 
    file_path = os.path.join(directory_path, file)  
    df = pd.read_csv(file_path)
    
    if len(df) < 100 or df.isna().any().any():
        print(f"{file} idata is invalid")
        sys.exit()
# *****************************************************************************



# *****************************************************************************
# create an average price for that day, calculate total earnings
# *****************************************************************************
    df['avg_price'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    
    start = df.iloc[0]['avg_price']
    end = df.iloc[-1]['avg_price'] 
    roi = end / start
# *****************************************************************************



# *****************************************************************************
# run our simulation on this hold stock
# *****************************************************************************
    port -= buy  # invest $5
    sell = buy*roi
    port += sell
# *****************************************************************************



# *****************************************************************************
# results
# *****************************************************************************
print(f'final portfolio is {round(port, 2)} for a profit of ${round(port-init_port, 2)}')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  
    
    
    



