import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd
import pickle

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)


clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

# following can be used to store the whole the training classification result and then ues when required
# with open('KNeighborsClassif.pickle', 'wb') as f:
#     pickle.dump(clf, f)

# pickle_in = open('KNeighborsClassif.pickle', 'rb')
# clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)

print(accuracy)

ex_measure = np.array([4,2,1,1,1,2,3,2,1])
ex_measure = ex_measure.reshape(1, -1)
# Try commenting above
# the first att is no of test data points in this case = 2, to generalize use following
# the data array should be list of lists.
# ex_measure = ex_measure.reshape(len(ex_measure), -1)


prediction = clf.predict(ex_measure)
print(prediction)
