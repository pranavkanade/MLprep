import pandas as pd
import quandl
import math
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

#following are features
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

#print(df.head)

#Adj. Close => this wont be our lable as we cannot predict the
#Adj close value of the next day based on the current one before staritng the trade  

#what would be the lable => price of the stock

forecast_col = 'Adj. Close'

#missing data is filled with following
df.fillna(-99999, inplace=True)

#print("******",len(df),"\n") -> this gives total no of rows in dataset

#So, according to the following what we are doing is considering the-
# '.1' percent of the data and predicting what should be the closing
# price
forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.head())
 
