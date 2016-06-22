import numpy as np
import pandas as pd




productDf = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/producto_tabla.csv')
productDf = productDf[(productDf['Producto_ID'] <= 300)]

#if 1000 then 68 products and 35M rows
# if 2000, the 138 products


print productDf.shape


df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/train.csv')

df = df[(df['Producto_ID'] <= 300)]

print df.shape

df.to_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems300.csv')

#appleSotcksDf = appleSotcksDf.fillna(method='pad')
#np.savetxt('temp.csv', appleSotcksDf, fmt='%s', delimiter=',', header="Date,Close")
