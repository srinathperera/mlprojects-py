import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as math

def drawLineChart(df, feilds):
    ts = df[feilds[0]]
    plt.plot(range(len(ts)), ts, "r--")
    # df.plot(y='Sales')

    plt.show()

#http://pandas.pydata.org/pandas-docs/stable/visualization.html
#df = pd.read_csv("data/rossmann.csv", dtype={'Promo': np.int32, 'SchoolHoliday': np.int32})
#print(df.describe())

df = pd.read_csv("forecastdata.csv")
print(df.describe())


df['LR_ER'] = abs(df['LR'] - df['actual'])
df['RFR_ER'] = abs(df['RFR'] - df['actual'])

#http://pandas.pydata.org/pandas-docs/stable/indexing.html
df[['LR_ER', 'RFR_ER']].plot()

#df.drop('# seq',1).head(500).plot()
plt.show()



#df = df[(df['Store'] <= 100)]

#df.to_csv('data/rossmann300stores.csv')

#print(df['Store'].median())

#print(len(df['Store'].unique()))

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.hist(df['Sales'], bins = 10, range = (df['Sales'].min(),df['Sales'].max()))
#plt.title('Sales distribution')
#plt.xlabel('Sales')
#plt.ylabel('Count of Bins')
#plt.show()

#df.boxplot(column='Sales', by = 'Store')
#df.boxplot(column='Sales')

#df = df[(df['Store'] == 1)]\

#df = df.head(1000)

#print(df.shape)

#sales = df['Sales'][:200]
#print(len(sales))

