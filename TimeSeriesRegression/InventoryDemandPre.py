import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mltools import apply_zeroMeanUnit2D, preprocess2DtoZeroMeanUnit, undo_zeroMeanUnit2D
import time




def print_base_stats(df):
    print df.describe()


def create_small_datafile(limit, df):
    productDf = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/producto_tabla.csv')
    productDf = productDf[(productDf['Producto_ID'] <= limit)]

    #if 1000 then 68 products and 35M rows
    # if 2000, the 138 products
    df = df[(df['Producto_ID'] <= limit)]

    print "Size", df.shape
    df.to_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems300.csv', index=False)




#print productDf.shape


#appleSotcksDf = appleSotcksDf.fillna(method='pad')
#np.savetxt('temp.csv', appleSotcksDf, fmt='%s', delimiter=',', header="Date,Close")


#df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems300.csv')




#grouped = df.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])['Demanda_uni_equil']
#salesByDeport = grouped.count()
#print salesByDeport.head(10)

#def avgdiff(group):
#    group = group.values
#    if len(group) > 1:
#        #return [np.mean([group[i] - group[i-1] for i in range(1, len(group))]) , np.mean(group), np.std(group)]
#        return np.mean(group)
#    else:
#        return 0

#values = grouped.apply(avgdiff)

#t = values[1110, 7, 3301, 50395, 145]

#print t

#print values.index
#print t[0], t[1], t[2]

#valuesDf =  pd.DataFrame(values)

#http://discuss.analyticsvidhya.com/t/how-to-convert-the-multi-index-series-into-a-data-frame-in-python/5119/2
#valuesDf = values.to_frame("Mean")
#valuesDf.reset_index(inplace=True)
#valuesDf['Std'] = grouped.std().values

#print valuesDf.head(10)


#start_ts = time.time()
#merged = pd.merge(df, valuesDf, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])

#print "took", (time.time() - start_ts)

#print merged.head(10)



#slopes = []
#for index, row in df.iterrows():
#    slopes.append(values[row['Agencia_ID'], row['Canal_ID'], row['Ruta_SAK'], row['Cliente_ID'], row['Producto_ID']])

#print slopes

#print df.describe()

def show_stat_graphs(df):
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

def test2Dtransfrom(df):
    half = int(df.shape[0]/2)
    df1 = df[:half]
    df2 = df[-1*(df.shape[0]-half):]

    d1 = df1.values.copy()
    d2 = df2.values.copy()
    t1,pn = preprocess2DtoZeroMeanUnit(d1)
    t2 = apply_zeroMeanUnit2D(d2, pn)

    f1 = undo_zeroMeanUnit2D(t1,pn)
    f2 = undo_zeroMeanUnit2D(t2,pn)

    print "d1=f1", np.allclose(d1, f1, atol=0.01)
    print "d2=f2", np.allclose(d2, f2, atol=0.01)

    print "d1", d1.shape, "t1", t1.shape, "f1", f1.shape
    print "d2", d2.shape, "t2", t2.shape, "f1", f2.shape






#df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/train.csv')
df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems300.csv')


#print_base_stats(df)
test2Dtransfrom(df)

#plt.tight_layout()
#plt.show()