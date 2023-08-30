# In this file we will be doing linear regression by hand
# In regression we try to model data with y = mx + b and we try to get best m and b
import pandas as pd
import quandl,math,datetime
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
# What we want is to get the adj.close from a next dat
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)  # to not lose data and the value will be treated as an outlier

forecast_out = int(math.ceil(0.01 * len(df)))  # We will try to predict 10 percent of our data frame

df['label'] = df[forecast_col].shift(
    -forecast_out)  # We shift the columns upwars , which mean that each label that need to be learnt is the adjusted close 10 days to the future
print(df.head())

X = np.array(df.drop(['label'], axis=1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
y = np.array(df["label"])
y = y[:-forecast_out]
df.dropna(inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=10)  # Default n_jobs is 1 , which is the amount of threading we do for running the model
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

# Predict on the x data
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Pickling using scikit learn
# Pickling is seralization of the model
with open('google_linear_regression.pickle' , 'wb') as f:
    pickle.dump(clf,f)

pickle_in = open('google_linear_regression.pickle', 'rb')
clf = pickle.load(pickle_in)
acc = clf.score(X_test, y_test)
print(acc)