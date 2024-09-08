# *****************************************************************************
# idea: 
# I need to benchmark the ML trading system. See how well someone randomly
# buy stocks could do. We will let the selling be a bit more strategic though

# methods:
# buy on a random day
# sell when the stock has risin y%, or lowered z% (stop loss)
# adjust y and z to see which gives the most profit on average
# *****************************************************************************



# *****************************************************************************
import pandas as pd
import os
import sys
import random
# *****************************************************************************



# *****************************************************************************
init_port = 100
port = init_port  # portfolio
buy = 5  # Price Per Stock
# *****************************************************************************



# *****************************************************************************
# iterate through files in 24_trading_time
# *****************************************************************************
directory_path = '/Users/daleyfraser/Documents/cs/stocks/sept_2024/large_cap/feature_extraction/24'
for file in os.listdir(directory_path):
    
    if file.startswith('.'):
        continue  #  skip hidden files 
    file_path = os.path.join(directory_path, file)  
    df = pd.read_csv(file_path)
    
    if len(df) < 100 or df.isna().any().any():
        print(f"{file} idata is invalid")
        sys.exit()
    
    # match ML test data
    df = df[20:]
    df = df[:-10]
    # print(df)
    # sys.exit()
# *****************************************************************************



# *****************************************************************************
# create an average price for that day
# *****************************************************************************
    df['avg_price'] = df[['todays_price']]
    
# *****************************************************************************



# *****************************************************************************
# run our simulation on this hold stock
# *****************************************************************************
    
    hold = False  # do we currently own the stock
    days_held = 0
    
    start = None
    end = None
    
    port -= buy  # invest $5
    single_port = buy  # each stock starts at 5, and will buy more with growth
    
    sell_at = 1.05  # 5% gain
    stop_loss = .90  # stop loss after reaching 90% of original value

    for idx, row in df.iterrows():
        
        # print(idx)
        
        
        if not hold and idx-19!= len(df):
            if random.randint(1,5) == 5:  # 20% chance of buying
                start = row['avg_price']
                hold = True
        
        else:
            # we are holding, should we sell yet?
            days_held += 1
            end = row['avg_price'] 
            roi = end / start
            
            if (roi >= sell_at or  # moneyyyyyy
                roi <= stop_loss or  # LLLLLLLL
                idx-19 == len(df) or  # always sell on last day
                days_held == 14):  # 2 week hold max
                
                single_port *= roi  # sell
                hold = False
                days_held = 0
    if hold:
        # print(idx-19)
        # print(len(df))
        print('^how did that occur')
    
                
    port += single_port
    
# *****************************************************************************



# *****************************************************************************
# results
# *****************************************************************************
print(f'final portfolio is {round(port, 2)} for a profit of ${round(port-init_port, 2)}')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  
    
    
    



