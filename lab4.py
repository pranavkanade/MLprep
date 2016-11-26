import pandas as pd
import quandl, math
import numpy as np #lets us use arrays

from sklearn import preprocessing, model_selection, svm
#cross_validation
from sklearn.linear_model import LinearRegression


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
forecast_out = int(math.ceil(0.001*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.tail())
print(forecast_out)

# generally 'X' => Features
# And 'y' => labels


X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

X = preprocessing.scale(X)
#above stmt is doing normalization of the data

#X = X[:-forecast_out+1]
#df.dropna(inplace=True)


y = np.array(df['label'])

print(len(X), len(y))
 
#cross_validation is used to train and test your data - what it does is it splits the provided data and shuffles it to make it unbiased
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#classifier
#clf = LinearRegression()
clf = LinearRegression(n_jobs=-1) #n_jobs attr is for no of threads in parallel '-1' is max possible.
#clf = svm.SVR()    #support vector regression
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)
