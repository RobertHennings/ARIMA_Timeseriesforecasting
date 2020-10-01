#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:01:03 2020

@author: Robert_Hennings
"""

#Als Erweiterung der Moving Averages folgt nun ein ARIMA Modell
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import datetime
from pandas.plotting import autocorrelation_plot

df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv')
print(df)

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
 
series = read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
print(series.head())
series.plot()
plt.show()
#Die Saleszahlen 36 an der Zahl sind nun nochmal im eigenen Dataframe mit den angegebene Datetimes welche auf den Start 1900 normiert wurden
#Wie im Plot zu sehen ist, gibt es einen deutlichen Aufwärtstrend zu beobachten
#Dies zeigt aber auch dass der Datensatz nicht stationär ist , er muss differenziert werden um ihn stationär zu machen mindestents mit einer Differenzierungsorder von 1
#Außerdem kann der Datensatz auf eine Autocorrelation hin  beobachtet werden  


def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
 
series = read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
autocorrelation_plot(series)
plt.show()

#Zu sehen ist eine positive Korrelation bei den ersten 10-12 Lags ab dann ca 0 bzw. eine negative, die positive Korrelation ist vermutlich für die ersten 5 Lags tatsächlich signifikant
#Die ersten 5 Lags bieten also daher einen guten Startpunkt für das ARIMA Modell

from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
 
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
 
series = read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# fit model
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())




from sklearn.metrics import mean_squared_error
 
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
 
series = read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()