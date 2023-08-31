import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
import pandas as pd

df = pd.read_csv('dataset.txt')

df.replace('?', -99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)  # We dropped it as the id has no effect on the label
# print(df.head())

X = np.array(df.drop(['class'], axis=1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1]])

pred = clf.predict(example_measures)
print(pred)
