#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 21:25:39 2020

@author: Robert_Hennings
"""

#Verschiedene Beispiele für Moving Averages in Python
import pandas as pd
import numpy as np


#Stock Price over 12 Months, Months :1-12 with the Price
stock = {'month' : [1,2,3,4,5,6,7,8,9,10,11,12],'Price':[290,260,288,300,310,303,329,340,316,330,308,310]}
df = pd.DataFrame(stock)
df.head()
df.plot()

#SMA mit dem Intervall 3, es werden 3 Daten zur  Kalkulation herangezogen, der älteste Wert wird durch den neusten erseetzt
#Mithilfe der iloc Funktion da nur die Spalte Price gebraucht wird, die Reihen sind variabel als i und werden solange durchkalkuliert bis zum Ende des dfs
for i in range(0,df.shape[0]-2):
    df.loc[df.index[i+2],'SMA_3'] = np.round(((df.iloc[i,1]+ df.iloc[i+1,1] +df.iloc[i+2,1])/3),1)
    
df.head()

#Zum Überprüfen wird die bereits bestehende Rolling Funktion von Pandas benutzt
df['pandas_SMA_3'] = df.iloc[:,1].rolling(window=3).mean()
df.head()
#Vergleich der beiden Spalten mit den ermittelten Werten zeigt die gute Übereinstimmung der selbst berechneten mit denen der eigenen eingebauten Funktion

#Nun nochmal mit dem Fenster von 4 anstatt 3:
for i in range(0,df.shape[0]-3):
    df.loc[df.index[i+3],'SMA_4'] = np.round(((df.iloc[i,1]+ df.iloc[i+1,1] +df.iloc[i+2,1]+df.iloc[i+3,1])/4),1)
    
    df.head()
    
df['pandas_SMA_4'] = df.iloc[:,1].rolling(window=4).mean()
#Vergelich zeigt auch hier die richtige Übereinstimmung
#Nun Visualisieren mit Matplotlib:
import matplotlib.pyplot as plt

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(df['Price'],label='data')
plt.plot(df['SMA_3'],label='SMA 3 Months')
plt.plot(df['SMA_4'],label='SMA 4 Months')
plt.legend(loc=2)



#Zweite Variante: Cumulative Moving Average
#Dafür wird nun ein anderer Datensatz verwendet, der sich mit Luftqualität und verscheidenen Attributen beschäftigt
#ZIP File runtergelden und im entsprechenden Ordner abgelegt zum Öffnen


df2 = pd.read_csv('/Users/Robert_Hennings/Dokumente/IT Weiterbildung/Python Data Analysis /AbbeV/Meine Projekte/Moving Averages in Python/AirQualityUCI/AirQualityUCI.csv', sep = ";", decimal = ",")
df2 = df2.iloc[ : , 0:14]
print(df2)

#Als ersten Data Processing Schritt soll geklärt werden ob acuh alle DAten vorhanden sind und es keine fehlenenden Daten als NaN gibt:
#Sollten welche gefunden werden können diese z.B. mit einer 0 ersetzt werden oder ganz heraus gelassen werden
df2.isna().sum()

#Es befinden sich definitiv NaN Werte im DataSet die ersetzt oder raus sollen

df2.dropna(inplace=True)
df2.isna().sum()
#Jetzt sind alle NaN Werte herausgefallen
#Der Movung Average soll auf die Temperature Spalte angewendet werden, diese soll herausgetrennt werden
df2_T = pd.DataFrame(df2.iloc[:,-2])
df2_T.head()  #Wurde ein neuer Dataframe geschaffen nur mit der interessierenden Spalte der Temperatur T
#Für den CMA wird die expanding Funktionvon Pandas angewendet:
df2_T['CMA_4'] = df2_T.expanding(min_periods=4).mean()
df2_T.head(10) #Eine neue Spalte ist im df2_T hinzugekommen als CMA Werte 
#Die Temperatur und die CMA Werte sollen nun noch in Kontext mit der aufgezeichneten ZEit gebracht werden, wie folgt:
import datetime

df2['DateTime'] = (df2.Date) + ' ' + (df2.Time)
df2.DateTime = df2.DateTime.apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y %H.%M.%S'))

df2_T.index = df2.DateTime #Die Index Spalte des df2_T wurde nun durch die entsprechenden Zeitangaben ersetzt
#Das GAnze soll nun einmal aufgeplottet werden:
plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(df2_T['T'],label='temperature')
plt.plot(df2_T['CMA_4'],label='CMA_4')
plt.legend(loc=2)

#Als Dritte Variante soll der EMA konstruiert werden:
df2_T['EMA'] = df2_T.iloc[:,0].ewm(span=40,adjust=False).mean()
df2_T.head()  #Im bekannten df2_T ist nun zusätzlich die Spalte mit en EMA Werten hinzugekommen wie zu sehen ist
#Ebenfalls nochmal aufplotten das Ganze
plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(df2_T['T'],label='temperature')
plt.plot(df2_T['CMA_4'],label='CMA_4')
plt.plot(df2_T['EMA'],label='EMA')
plt.legend(loc=2)
