import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




#productDf = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/producto_tabla.csv')
#productDf = productDf[(productDf['Producto_ID'] <= 300)]

#if 1000 then 68 products and 35M rows
# if 2000, the 138 products

#print productDf.shape
#df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/train.csv')

#df = df[(df['Producto_ID'] <= 300)]

#print df.shape

#df.to_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems300.csv')

#appleSotcksDf = appleSotcksDf.fillna(method='pad')
#np.savetxt('temp.csv', appleSotcksDf, fmt='%s', delimiter=',', header="Date,Close")


df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems300.csv')


grouped = df.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])['Demanda_uni_equil']
#salesByDeport = grouped.count()
#print salesByDeport.head(10)

def avgdiff(group):
    group = group.values
    if len(group) > 1:
        return np.mean([group[i] - group[i-1] for i in range(1, len(group))])
    else:
        return 0

values = grouped.apply(avgdiff)

#print values[1110, 7, 3301, 50395, 145]

slopes = []
for index, row in df.iterrows():
    slopes.append(values[row['Agencia_ID'], row['Canal_ID'], row['Ruta_SAK'], row['Cliente_ID'], row['Producto_ID']])

print slopes

#print df.describe()

salesByDeport = df.groupby(['Agencia_ID'])['Demanda_uni_equil'].mean()
#print salesByDeport.describe()
#print type(salesByDeport)
plt.figure(1, figsize=(20,10))
plt.subplot(321)
plt.xlabel('SalesDeport', fontsize=18)
plt.ylabel('MeanSales', fontsize=18)
#plt.yscale('log')
#plt.xscale('log')
plt.scatter(salesByDeport.index.values, salesByDeport.values, alpha=0.5)

salesByDeport = df.groupby(['Canal_ID'])['Demanda_uni_equil'].mean()
plt.subplot(322)
plt.xlabel('Channels', fontsize=18)
plt.ylabel('Sales', fontsize=18)
plt.scatter(salesByDeport.index.values, salesByDeport.values, alpha=0.5)

salesByDeport = df.groupby(['Ruta_SAK'])['Demanda_uni_equil'].mean()
plt.subplot(323)
plt.xlabel('Route', fontsize=18)
plt.ylabel('Sales', fontsize=18)
plt.scatter(salesByDeport.index.values, salesByDeport.values, alpha=0.5)


salesByDeport = df.groupby(['Cliente_ID'])['Demanda_uni_equil'].mean()
plt.subplot(324)
plt.xlabel('Client', fontsize=18)
plt.ylabel('Sales', fontsize=18)
plt.scatter(salesByDeport.index.values, salesByDeport.values, alpha=0.5)


salesByDeport = df.groupby(['Producto_ID'])['Demanda_uni_equil'].mean()
plt.subplot(325)
plt.xlabel('Product', fontsize=18)
plt.ylabel('Sales', fontsize=18)
plt.scatter(salesByDeport.index.values, salesByDeport.values, alpha=0.5)


salesByDeport = df.groupby(['Producto_ID'])['Demanda_uni_equil'].mean()
plt.subplot(326)
plt.xlabel('Client', fontsize=18)
plt.ylabel('Route', fontsize=18)
plt.scatter(df['Cliente_ID'].values, df['Ruta_SAK'], c=df['Demanda_uni_equil'], alpha=0.5)



#plt.tight_layout()
#plt.show()