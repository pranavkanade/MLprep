import pandas as pd
import quandl, math, datetime
import numpy as np #lets us use arrays

from sklearn import preprocessing, model_selection, svm
#cross_validation
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

#pickling is the serialization of the object
import pickle

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
#print(df.tail())

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

print(df.tail())
print(forecast_out)

# generally 'X' => Features
# And 'y' => labels


X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:] #we dont have y val for all these
X = X[:-forecast_out]
#above stmt is doing normalization of the data

#X = X[:-forecast_out+1]
#df.dropna(inplace=True)
df.dropna(inplace=True)
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

#pickling the classifier results and saving the results of training
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf, f)


# Once the classifier is trained there is no need to run the above stmt for classifier
# As all the results needed for the classifier are stored in the file.
pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)



accuracy = clf.score(X_test, y_test)

print(accuracy)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day


#following loop shows dates at the base of the graph

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

#df.loc -> is the index and if it exist then replace the val in above loop


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# pickel works as file and use it when required and we are saving the trained results
# and use them with the new incomming data as the training will cost more than the
# prediction itself hence we will store the training results with pickle.
