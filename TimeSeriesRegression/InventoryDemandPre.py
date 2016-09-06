import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mltools import apply_zeroMeanUnit2D, preprocess2DtoZeroMeanUnit, undo_zeroMeanUnit2D
import time

import re
from matplotlib import cm as CM


from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans
from data_explore import *


from inventory_demand import *
from inventory_demand_features import *
from inventory_demand_ensambles import *

def print_base_stats(df):
    print df.describe()


def create_small_datafile(low, high, df):
    #productDf = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/producto_tabla.csv')
    #productDf = productDf[(productDf['Producto_ID'] < high) & (productDf['Producto_ID'] >= low)]
    #print len(productDf)

    #if 1000 then 68 products and 35M rows
    # if 2000, the 138 products
    df = df[(df['Producto_ID'] < high) & (df['Producto_ID'] >= low)]

    print df['Producto_ID'].min(), df['Producto_ID'].max()

    print "Size", df.shape
    df.to_csv('data/trainitems'+ str(low) +'_'+ str(high) +'.csv', index=False)


def create_small_testfile(low, high, df):
    df = df[(df['Producto_ID'] < high) & (df['Producto_ID'] >= low)]

    print df['Producto_ID'].min(), df['Producto_ID'].max()
    print low, " ", high, " Size", df.shape
    df.to_csv('test_'+ str(low) +'_'+ str(high) +'.csv', index=False)

def break_test_dataset(df):
    for i in range(0, 50000, 5000):
        create_small_testfile(i,i+5000, df)




def create_small_test_datafile(low, high):
    df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/train.csv')
    #if 1000 then 68 products and 35M rows
    # if 2000, the 138 products
    df = df[(df['Producto_ID'] < high) & (df['Producto_ID'] >= low)]

    print df['Producto_ID'].min(), df['Producto_ID'].max()

    print "Size", df.shape
    df.to_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/test'+ str(low) +'_'+ str(high) +'.csv', index=False)



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

    plt.tight_layout()
    plt.show()

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


def merge_outputs():
    testDf = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/test.csv')
    testDf1k = testDf[(testDf['Producto_ID'] < 1000)]
    results1k = pd.read_csv('sub1/submission1k.csv')
    results1k['id'] = testDf1k['id']

    testDf2k = testDf[(testDf['Producto_ID'] < 2000) & (testDf['Producto_ID'] >= 1000)]
    results2k = pd.read_csv('sub1/submission1k_2k.csv')
    results2k['id'] = testDf2k['id']

    testDf3k = testDf[(testDf['Producto_ID'] < 10000) & (testDf['Producto_ID'] >= 2000)]
    results3k = pd.read_csv('sub1/submission2k_10k.csv')
    results3k['id'] = testDf3k['id']

    result = pd.concat([results1k, results2k, results3k]).sort_values(by=['id'])

    result.to_csv('final.csv', index=False)

    print result.shape


def merge_submissions(path="./"):
    submission_files = [pd.read_csv(path+'en_submission'+str(i)+'.csv') for i in range(4)]

    tot = 0
    for df in submission_files:
        print "size",df.shape[0]
        tot = tot + df.shape[0]
    print "submission size", tot

    result = pd.concat(submission_files).sort_values(by=['id'])
    result.to_csv('en_final.csv', index=False)
    print "submission file of shape ", result.shape, " created"




def merge_csv_files(list, output_file_name, root_dir):
    submission_files = [pd.read_csv(root_dir+ f) for f in list]
    result = pd.concat(submission_files)
    result.to_csv(output_file_name, index=False)
    print output_file_name, "submission file of shape ", result.shape, " created"


def merge_data_and_test_files():
    train_files = ["trainitems5000_10000.csv", "trainitems35000_40000.csv",  "trainitems45000_50000.csv"]
    merge_csv_files(train_files, "trainitems5_10_35_40_45_50k.csv", "data/")

    test_files = ["test_5000_10000.csv", "test_35000_40000.csv", "test_45000_50000.csv"]
    merge_csv_files(test_files, "test_5_10_35_40_45_50k.csv", "data/")





def test_falttern_reshape():
    x = np.array([[[1], [2,3]], [[4], [5], [5, 6,7], [], [8]]])
    print "before", x
    y = x.ravel()
    print "X", y, "len",  y.shape

    #print x.flatten().reshape(-1, 4)

def find_missing_products():
    train = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/train.csv')
    train_ids = train['Producto_ID'].unique()
    test = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/test.csv')
    test_ids = test['Producto_ID'].unique()

    missing_ids = pd.Index(test_ids).difference(pd.Index(train_ids))
    print "missing ID count ", len(missing_ids)

    missing_ids_df =  pd.DataFrame(missing_ids, columns=["Producto_ID"])
    missing_ids_df.to_csv('missing_ids.csv', index=False)

    entries_with_missing = pd.merge(test, missing_ids_df, on='Producto_ID')

    print "Mising entries=", entries_with_missing.shape[0], "percentage=", entries_with_missing.shape[0]*100/test.shape[0]

    print "full entries count", test.shape[0]



def testset_stats():
    train_df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/train.csv')
    counts = train_df.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])['Demanda_uni_equil'].count()
    valuesDf = counts.to_frame("Count")
    valuesDf.reset_index(inplace=True)

    test_df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/test.csv')
    merged = pd.merge(test_df, valuesDf, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])

    print "np.inf=", np.where(np.isnan(merged))

    missing_count = merged["Count"].isnull().sum()

    print float(missing_count)/ test_df.shape[0], " entries are known ", missing_count, ",", test_df.shape[0]


def build_sample_dataset():
    #
    #227,3,1110,7,3301,1263779,145,1,19.75,0,0.0,1

    data = []
    for agency in range(0,5):
        for week in range(6,10):
            for product in range(1,200):
                data.append([week, agency, 1, 1, 1, product, 1, 0, 0, 0, week*product/(agency +1)])

    df =  pd.DataFrame(data, columns=['Semana', 'Agencia_ID','Canal_ID','Ruta_SAK', 'Cliente_ID', 'Producto_ID',
                                                  'Venta_uni_hoy', 'Venta_hoy','Dev_uni_proxima', 'Dev_proxima', 'Demanda_uni_equil'])


    grouped = df.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
    print "all 4", grouped.mean(), grouped.std()

    groupe1d = df.groupby(['Agencia_ID'])
    print "Agencia_ID", groupe1d.mean(), groupe1d.std()

    groupe2d = df.groupby(['Ruta_SAK'])
    print "Ruta_SAK", groupe2d.mean(), groupe2d.std()

    df.to_csv('sample_dataset.csv', index=False)

def break_dataset():
    df = pd.read_csv('data/train.csv')
    for i in range(0, 50000, 5000):
        create_small_datafile(i,i+5000, df)




def analyze_error():
    df = pd.read_csv('forecast_with_data.csv')

    df['error'] = np.abs(np.log(df['actual'].values +1) - np.log(df['predictions'].values + 1))
    df['Slopes'] = np.round(df['Slopes'].values)

    print df.describe()



    plt.figure(1, figsize=(20,10))
    plt.subplot(321)
    plt.xlabel('Error', fontsize=18)
    plt.ylabel('Slope', fontsize=18)
    #plt.yscale('log')
    #plt.xscale('log')
    plt.scatter(df['Slopes'].values, df['error'].values, alpha=0.5)

    groupe2d = df.groupby(['Slopes'])['error'].mean()
    plt.subplot(322)
    plt.xlabel('Slope', fontsize=18)
    plt.ylabel('Mean Error', fontsize=18)
    plt.scatter(groupe2d.index.values, groupe2d.values, alpha=0.5)

    df['groupedStd'] = np.round(df['groupedStd'].values)
    groupe2d = df.groupby(['groupedStd'])['error'].mean()
    plt.subplot(323)
    plt.xlabel('groupedStd', fontsize=18)
    plt.ylabel('Mean Error', fontsize=18)
    plt.scatter(groupe2d.index.values, groupe2d.values, alpha=0.5)

    df['groupedMeans'] = np.round(df['groupedMeans'].values)
    groupe2d = df.groupby(['groupedMeans'])['error'].mean()
    plt.subplot(324)
    plt.xlabel('groupedMeans', fontsize=18)
    plt.ylabel('Mean Error', fontsize=18)
    plt.scatter(groupe2d.index.values, groupe2d.values, alpha=0.5)





def product_stats():
    df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems300.csv')

    grouped = df.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])['Demanda_uni_equil']

    slopeMap = grouped.apply(avgdiff)
    groupedMeanMap = grouped.mean()
    groupedStddevMap = grouped.std()

    valuesDf = slopeMap.to_frame("Slopes")
    valuesDf.reset_index(inplace=True)
    valuesDf["groupedMeans"] = groupedMeanMap.values
    valuesDf["groupedStd"] = groupedStddevMap.values



    plt.figure(1, figsize=(20,10))
    plt.subplot(321)
    plt.xlabel('Producto_ID', fontsize=18)
    plt.ylabel('Mean', fontsize=18)
    #plt.yscale('log')
    #plt.xscale('log')
    plt.scatter(valuesDf['Producto_ID'].values, valuesDf['groupedMeans'].values, alpha=0.5)

#   groupe2d = df.groupby(['Slopes'])['error'].mean()
    plt.subplot(322)
    plt.xlabel('Agencia_ID', fontsize=18)
    plt.ylabel('Mean', fontsize=18)
    plt.scatter(valuesDf['Agencia_ID'].values, valuesDf['groupedMeans'].values, alpha=0.5)

    plt.subplot(323)
    plt.xlabel('Agencia_ID', fontsize=18)
    plt.ylabel('Slope', fontsize=18)
    plt.scatter(valuesDf['Agencia_ID'].values, valuesDf['Slopes'].values, alpha=0.5)

    plt.subplot(324)
    plt.xlabel('Producto_ID', fontsize=18)
    plt.ylabel('Slope', fontsize=18)
    plt.scatter(valuesDf['Producto_ID'].values, valuesDf['Slopes'].values, alpha=0.5)


    plt.tight_layout()
    plt.show()





## category tests #######

def find_top10_category_coverage(df, feild_name):
    counts = df[feild_name].value_counts()
    counts = counts.sort_values(ascending=False)
    top10 = counts[0:min(len(counts), 200)]

    return float(top10.sum())/counts.sum()


def category_histograms(df):
    plt.figure(1, figsize=(20,10))


    find_top10_category_coverage(df, 'Producto_ID' )

    print "Producto_ID",len(df['Producto_ID'].unique()), "coverage", find_top10_category_coverage(df, 'Producto_ID' ), df['Producto_ID'].max()
    print "Agencia_ID", len(df['Agencia_ID'].unique()), "coverage", find_top10_category_coverage(df, 'Agencia_ID' ), df['Agencia_ID'].max()
    print "Canal_ID", len(df['Canal_ID'].unique()), "coverage", find_top10_category_coverage(df, 'Canal_ID' ), df['Canal_ID'].max()
    print "Ruta_SAK", len(df['Ruta_SAK'].unique()), "coverage", find_top10_category_coverage(df, 'Ruta_SAK' ), df['Ruta_SAK'].max()
    print "Cliente_ID", len(df['Cliente_ID'].unique()) , "coverage", find_top10_category_coverage(df, 'Cliente_ID' ), df['Cliente_ID'].max()


    plt.subplot(321)
    plt.xlabel('Producto_ID', fontsize=12)
    plt.ylabel('Mean', fontsize=18)
    #plt.yscale('log')
    #plt.xscale('log')
    plt.hist(df['Producto_ID'], bins=1000)

    plt.subplot(322)
    plt.xlabel('Agencia_ID', fontsize=12)
    plt.ylabel('Mean', fontsize=18)
    #plt.yscale('log')
    #plt.xscale('log')
    plt.hist(df['Agencia_ID'], bins=1000)


    plt.subplot(323)
    plt.xlabel('Canal_ID', fontsize=12)
    plt.ylabel('Mean', fontsize=18)
    #plt.yscale('log')
    #plt.xscale('log')
    plt.hist(df['Canal_ID'], bins=100)


    plt.subplot(324)
    plt.xlabel('Ruta_SAK', fontsize=12)
    plt.ylabel('Mean', fontsize=18)
    #plt.yscale('log')
    #plt.xscale('log')
    plt.hist(df['Ruta_SAK'], bins=100)


    plt.subplot(325)
    plt.xlabel('Cliente_ID', fontsize=12)
    plt.ylabel('Mean', fontsize=18)
    #plt.yscale('log')
    #plt.xscale('log')
    plt.hist(df['Cliente_ID'], bins=100)

    plt.show()



def write_graph(values, names, file):
    data = np.hstack(values)
    to_saveDf =  pd.DataFrame(data, columns=names)
    to_saveDf.to_csv(file, index=False)


def agency_product_stats(df):
    feild_name = 'Agencia_ID_Producto_ID'
    counts = df.groupby(['Agencia_ID', 'Producto_ID'])['Demanda_uni_equil'].count()
    valuesDf = counts.to_frame("Count")
    valuesDf.reset_index(inplace=True)
    valuesDf = valuesDf.sort_values(by=['Count'], ascending=False)
    valuesDf['c'+feild_name] = range(valuesDf.shape[0])
    valuesDf.to_csv('Agencia_ID_Producto_ID_data.csv', index=False)
    print valuesDf.describe()
    print feild_name+ '_data.csv Done'

    valuesDf['Source'] = valuesDf['Agencia_ID']
    valuesDf['Target'] = valuesDf['Producto_ID']
    valuesDf = drop_feilds_1df(valuesDf, ['Agencia_ID', 'Producto_ID'])
    valuesDf.to_csv('Agencia_ID_Producto_ID_data_g.csv', index=False)

    feild_name = 'Agencia_ID_Canal_ID'
    counts = df.groupby(['Agencia_ID', 'Canal_ID'])['Demanda_uni_equil'].count()
    valuesDf = counts.to_frame("Count")
    valuesDf.reset_index(inplace=True)
    valuesDf = valuesDf.sort_values(by=['Count'], ascending=False)
    valuesDf['c'+feild_name] = range(valuesDf.shape[0])
    valuesDf.to_csv('Agencia_ID_Canal_ID_data.csv', index=False)
    print valuesDf.describe()
    print feild_name+ '_data.csv Done'



def product_raw_stats():
    df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/train.csv')

    #df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems300.csv')

    grouped = df.groupby(['Semana'])['Demanda_uni_equil'].mean()

    plt.figure(1, figsize=(20,10))
    plt.subplot(321)
    plt.xlabel('Semana', fontsize=18)
    plt.ylabel('Mean', fontsize=18)
    #plt.yscale('log')
    #plt.xscale('log')
    plt.scatter(df['Semana'].values, df['Demanda_uni_equil'].values, alpha=0.5)

    grouped = df.groupby(['Semana'])['Demanda_uni_equil'].mean()
    plt.subplot(322)
    plt.xlabel('Slope', fontsize=18)
    plt.ylabel('Mean Error', fontsize=18)
    plt.scatter(grouped.index.values, grouped.values, alpha=0.5)

    grouped = df.groupby(['Semana', 'Producto_ID'])['Demanda_uni_equil'].mean()
    valuesDf = grouped.to_frame("Mean")
    valuesDf.reset_index(inplace=True)


    plt.subplot(323)
    plt.xlabel('Semana', fontsize=18)
    plt.ylabel('Mean Error', fontsize=18)
    plt.scatter(valuesDf['Semana'], valuesDf["Mean"], alpha=0.5)


    grouped = df.groupby(['Semana', 'Producto_ID'])['Demanda_uni_equil'].count()
    valuesDf = grouped.to_frame("count")
    valuesDf.reset_index(inplace=True)


    plt.subplot(324)

    X = valuesDf['Semana']
    Y = valuesDf['Producto_ID']
    plt.hexbin(valuesDf['Semana'], valuesDf['Producto_ID'], C=valuesDf['count'], cmap=CM.jet, gridsize=30, bins=50)
    plt.axis([X.min(), X.max(), Y.min(), Y.max()])

    cb = plt.colorbar()
    cb.set_label('mean value')

    plt.tight_layout()
    plt.show()


def calculate_and_save_aggrigates(df):
    grouped = df.groupby(['Producto_ID'])['Demanda_uni_equil'].count()
    count_df = group_to_df(grouped, "count").sort_values(by=["count"])
    count_df.to_csv('product_id_count.csv', index=False)

    grouped = df.groupby(['Agencia_ID'])['Demanda_uni_equil'].count()
    count_df = group_to_df(grouped, "count").sort_values(by=["count"])
    count_df.to_csv('agency_count.csv', index=False)

def add_precentage():
    id_count_df = pd.read_csv('product_id_count.csv')
    total = id_count_df["count"].sum()
    id_count_df["percent"] = np.round(float(100)*id_count_df["count"].values/total, decimals=2)
    id_count_df.to_csv('product_id_count_full.csv', index=False)


    id_count_df = pd.read_csv('agency_count.csv')
    total = id_count_df["count"].sum()
    id_count_df["percent"] = np.round(float(100)*id_count_df["count"].values/total, decimals=2)
    id_count_df.to_csv('agency_count_full.csv', index=False)


def test_one_hot():
    df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems300.csv')

    training_set_size = int(0.7*df.shape[0])
    test_set_size = df.shape[0] - training_set_size
    train_df = df[:training_set_size]
    test_df = df[-1*test_set_size:]

    #vec = DictVectorizer()
    #vec.fit(train_df[['Agencia_ID', 'Canal_ID']].to_dict(orient='records'))

    #transformed_train = vec.transform(train_df[['Agencia_ID', 'Canal_ID']].to_dict(orient='records'))
    #print "transformed_train", transformed_train.shape

    #transformed_test = vec.transform(test_df[['Agencia_ID', 'Canal_ID']].to_dict(orient='records'))
    #print "transformed_test", transformed_test.shape


    #transformed_train2 = vec.transform(train_df[['Agencia_ID', 'Canal_ID']].to_dict(orient='records'))
    #print "transformed_train", transformed_train.shape

    #print type(transformed_train)
    #print np.allclose(transformed_train.toarray(), transformed_train2.toarray())

    train_df_encoded, vec = encode_onehot(train_df, ['Agencia_ID', 'Canal_ID'])

    print "train_df, train_df_encoded", train_df.shape, train_df_encoded.shape
    print list(train_df_encoded)

    test_df_encoded, vec = encode_onehot(test_df, ['Agencia_ID', 'Canal_ID'], vec)
    print "test_df, test_df_encoded", test_df.shape, test_df_encoded.shape


def test_return_multiple_apply(df):
    df = df.head(100000)
    grouped = df.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
    slopeMap = grouped.apply(calculate_slope)

    valuesDf = slopeMap.to_frame("Slopes")
    valuesDf.reset_index(inplace=True)

    #print "slopeMap", valuesDf.head()
    print expand_array_feild_and_add_df(valuesDf, "Slopes", ["mean", "slope"])


def submission_stats():
    df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/test.csv')
    sub1 = pd.read_csv('/Users/srinath/Desktop/final.csv')

    missing = list(set(df['id']) - set(sub1['id']))

    print len(missing), " missing"

    print df[missing]


    #print df.describe()

'''
def test_ensambles():
    df = pd.read_csv('forecasts4ensamble_s1.csv')
    actual = df['actual'].values
    df = drop_feilds_1df(df, ['actual'])

    print "size 22", df.shape

    rmsle_values = [0.65659655949269957, 0.66699743799227806, 0.68701745810105874, 0.62905236144713017]
    avg_models(rmsle_values, df.values, actual)

    print 'hello'
'''

def create_random_file():
    df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/train.csv')
    subset_df = df.sample(10000)
    subset_df.to_csv("train-rsample-10k.csv", index=False)


def parse_feature_importance(file='/Users/srinath/playground/data-science/BimboInventoryDemand/logs/out.txt'):
    file = open(file,'r')
    data =  file.read()

    #data = data.replace('\n','')
    data = re.sub(r'\[=+\].*?s', '', data)
    #28. feature 27 =Producto_ID_Dev_proxima_StdDev (0.002047)

    p1 = re.compile('[0-9]+.*?feature.*?=([A-z0-9_]+)\s+\(([0-9.]+)\)')
    p2 = re.compile('[0-9]+\s+([A-z0-9_]+)\s+([0-9.]+)')

    dict = {}
    sumdict = {}

    lines = data.split('\n')
    print "lines", len(lines), "found"

    for t in lines:
        for match in p1.finditer(t):
            f = match.group(1)
            score = match.group(2)
            if len(f) > 2:
                list = dict.get(f, [])
                list.append(float(score))
                dict[f] = list

            #print f,"=",score
        for match in p2.finditer(t):
            f = match.group(1)
            score = match.group(2)
            if len(f) > 2:
                list = dict.get(f, [])
                list.append(float(score))
                dict[f] = list

    df_data = []
    #for t in dict.items():
    for (k,v) in dict.items():
        #print t
        #df_data.append([t[0], np.sum(t[1])])
        sum = np.sum(v)
        df_data.append([k, sum])
        sumdict[k] = sum

    print df_data

    feature_importance_df =  pd.DataFrame(np.row_stack(df_data), columns=["f","score"])
    feature_importance_df.index = range(0, feature_importance_df.shape[0])
    feature_importance_df = feature_importance_df.sort_values(by=['score'], ascending=False)
    print feature_importance_df




    groups = [
        ['Agencia_ID_Demanda_uni_equil_Mean', 'Agencia_ID_Demanda_uni_equilci', 'Agencia_ID_Demanda_uni_equil_median']
            + ['Agencia_ID_Dev_proxima_Mean', 'Agencia_ID_Dev_proximaci', 'Agencia_ID_Dev_proxima_median']
            + ['Agencia_ID_Venta_hoy_Mean', 'Agencia_ID_Venta_hoyci', 'Agencia_ID_Venta_hoy_median'],
        ['clients_combined_Mean', 'clients_combined_kurtosis', 'clients_combinedci', 'clients_combined_median']
            + ['clients_combined_vh_Mean', 'clients_combined_vhci', 'clients_combined_vh_median']
            + ['clients_combined_dp_Mean', 'clients_combined_dpci', 'clients_combined_dp_median'],
        ['client_nn_Mean', 'client_nnci', 'client_nn_median']
            + ['client_nn_vh_Mean', 'client_nn_vhci', 'client_nn_vh_median']
            + ['client_nn_dp_Mean', 'client_nn_dpci', 'client_nn_dp_median'],
        ['client_nn_agency_Mean', 'client_nn_agencyci', 'client_nn_agency_median']
            + ['client_nn_agency_vh_Mean', 'client_nn_agency_vhci', 'client_nn_agency_vh_median']
            + ['client_nn_agency_dp_Mean', 'client_nn_agency_dpci', 'client_nn_agency_dp_median'],
        ['Producto_ID_Demanda_uni_equil_Mean', 'Producto_ID_Demanda_uni_equilci', 'Producto_ID_Demanda_uni_equil_median']
            + ['Producto_ID_Venta_hoy_Mean', 'Producto_ID_Venta_hoyci', 'Producto_ID_Venta_hoy_median']
            + ['Producto_ID_Dev_proxima_Mean', 'Producto_ID_Dev_proximaci', 'Producto_ID_Dev_proxima_median'],
        ['weight', 'pieces', 'has_choco', 'has_vanilla', 'has_multigrain'],
        ['brand_id_Demanda_uni_equil_Mean', 'brand_id_Demanda_uni_equilci', 'brand_id_Demanda_uni_equil_median','product_word_Demanda_uni_equil_Mean', 'product_word_Demanda_uni_equilci', 'product_word_Demanda_uni_equil_median'],
        ['Town_id_Demanda_uni_equil_Mean', 'Town_id_Demanda_uni_equilci', 'Town_id_Demanda_uni_equil_median', 'State_id_Demanda_uni_equil_Mean', 'State_id_Demanda_uni_equilci', 'State_id_Demanda_uni_equil_median'],
        ['agc_product_Mean', 'agc_productci', 'agc_product_median'],
        ['routes_combined_Mean', 'routes_combinedci', 'routes_combined_median'],
        ['clients_route_agc_Mean', 'clients_route_agcci', 'clients_route_agc_median']
    ]

    all_f_count = 0
    found_f_count = 0
    for g in groups:
        gk = []
        for f in g:
            score = sumdict.get(f, -1)
            all_f_count += 1
            if score > 1.5:
                found_f_count += 1
                gk.append(f)
        print gk

    print "Feature data", all_f_count, found_f_count







def parse_map_from_str(data):
    p2 = re.compile('\'([A-z0-9_]+)\'\s*:\s*([0-9.]+)')
    dict = {}
    for match in p2.finditer(data):
        key = match.group(1)
        value = float(match.group(2))
        dict[key] = value
    return dict


def parse_parameter_sweep(file='/Users/srinath/playground/data-science/BimboInventoryDemand/logs/xgboost_params-explore-case4.txt'):
    file = open(file,'r')
    data =  file.read()

    data = data.replace('\n','')
    data = re.sub(r'\[=+\'\].*?s', '', data)
    #28. feature 27 =Producto_ID_Dev_proxima_StdDev (0.002047)

    p1 = re.compile('Run ([0-9+]) XGBoost_nocv {(.*?)} .*?rmsle=([0-9.]+)')

    readings = []
    for match in p1.finditer(data):
        data_index = int(match.group(1))
        params_as_str = match.group(2)
        rmsle = float(match.group(3))
        print data_index, rmsle, params_as_str

        kvmap = parse_map_from_str(params_as_str)
        print kvmap
        readings.append([data_index, rmsle, kvmap['eta'], kvmap['max_depth'], kvmap['min_child_weight'], kvmap['gamma'],
                         kvmap['subsample'], kvmap['colsample_bytree']])

    df_data = np.row_stack(readings)
    para_sweep_df= pd.DataFrame(df_data, columns=['data_index' , 'rmsle', 'eta', 'max_depth', 'min_child_weight', 'gamma',
                         'subsample', 'colsample_bytree'])
    print para_sweep_df
    return para_sweep_df

def analyze_parameter_sweep():
    create_fig()
    para_sweep_df = parse_parameter_sweep()
    y_val = para_sweep_df['rmsle'].values
    draw_simple_scatterplot(para_sweep_df['max_depth'].values, y_val, 'max_depth', 321 )
    draw_simple_scatterplot(para_sweep_df['min_child_weight'], y_val, 'min_child_weight', 322)
    draw_simple_scatterplot(para_sweep_df['gamma'], y_val, 'gamma', 323)
    draw_simple_scatterplot(para_sweep_df['subsample'], y_val, 'subsample', 324)
    draw_simple_scatterplot(para_sweep_df['colsample_bytree'], y_val, 'colsample_bytree', 325)
    draw_simple_scatterplot(para_sweep_df['data_index'], y_val, 'data_index', 326)
    plt.tight_layout()
    plt.savefig('xgboost-pswp.png')





'''
def test_voting_ensamble():
    #[0.79032504607649889, 0.82079594626169383, 0.8203164757712077, 0.83624063520333292]

    df = pd.read_csv('individual_forecasts6-test.csv')

    y_actual = df['actual'].values.copy()
    df = drop_feilds_1df(df, ['actual'])


    vf_start = time.time()
    forecasts = df.values
    best_forecast_index = 0
    vote_forecast = vote_based_forecast(forecasts, best_forecast_index, y_actual)
    calculate_accuracy("vote_forecast", y_actual, vote_forecast)
    print "vf tooks", (time.time() - vf_start)



#parse_feature_importance()
test_voting_ensamble()
'''


def test_merge_datasets():
    cmd = 6
    train_df, test_df, testDf = load_train_data('agr_cat', cmd)
    train_df, test_df, testDf = merge_another_dataset(train_df, test_df, testDf, 'fg_stats', cmd,["mean_sales", "sales_count",
                                            "sales_stddev", "median_sales", "last_sale", "last_sale_week", "returns"])
    print list(train_df)


def shaffle_data():
    data_files = [
        ["trainitems0_5000.csv", 0, 5000, "test_0_5000.csv"], #1.4G, 0
        ["trainitems5_10_35_40_45_50k.csv", 5000, 10000, "test_5_10_35_40_45_50k.csv"], #534M, 1
        ["trainitems30000_35000.csv", 30000, 35000, "test_30000_35000.csv"], #559M, 2
        #["trainitems30000_35000.csv", 30000, 35000, "trainitems5_10_35_40_45_50k.csv"], #559M # to remove ** pass #1 as #2 test
        ["trainitems40000_45000.csv", 40000, 45000, "test_40000_45000.csv"], #640M, 2
        ["trainitems5000_15000.csv", -1, -1, "test0_100.csv"], #4
        ["train-rsample-10m.csv", -1, -1, "test0_100.csv"], #5
        ["train-rsample-500k.csv", -1, -1, "test0_100.csv"], #6
        ["train-rsample-15m.csv", -1, -1, "test.csv"], #7
        ["train-rsample-10k.csv", -1, -1, "test0_100.csv"] #8
    ]

    merge_feilds = ['Semana', 'Agencia_ID' , 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID']
    #for command in range(4):
    for command in [8]:
        ordered_data = pd.concat([load_file('all_features', command, f) for f in ['train', 'test']])
        df_raw, _ = read_datafiles(command, False)
        shuffled_df = pd.merge(ordered_data[merge_feilds], df_raw, how='left', on=merge_feilds)
        find_NA_rows_percent(shuffled_df, "shaffle_data")
        shuffled_df.to_csv(data_files[command][0], index=False)


def compare_datasets(folder1, folder2, command):
    train_df1, test_df1, testDf1, y_actual_train1, y_actual_test1 = load_train_data(folder1, command)
    if train_df1 is None or test_df1 is None or testDf1 is None:
        raise ValueError('data not found for '+ folder1)
    else:
        print "reusing train data", folder1

    train_df2, test_df2, testDf2, y_actual_train2, y_actual_test2 = load_train_data(folder2, command)
    if train_df2 is None or test_df2 is None or testDf2 is None:
        raise ValueError('data not found for '+ folder2)
    else:
        print "reusing train data", folder2

    data_list1 = [train_df1, test_df1, testDf1]
    data_list2 = [train_df2, test_df2, testDf2]

    print data_list1[0].size(), "=", data_list2[0].size()
    print list(data_list1[0])
    print list(data_list2[0])

    for i in range(len(data_list1)):
        if not data_list1[0].equals(data_list2[0]):
            for f in list(data_list1[0]):
                if "_Mean" in f:
                   print i, data_list1[0][f].mean(), "=", data_list2[0][f].mean()
        else:
            print i, " equal"


compare_datasets('/home/wso2test/MLProjects/mlprojects-py/TimeSeriesRegression/agr_cat', '/home/wso2test/MLProjects/tags/mlprojects-py/TimeSeriesRegression/agr_cat/', 1)



#shaffle_data()
#test_merge_datasets()
#find_similar_products()
#analyze_parameter_sweep()
#create_random_file()

#parse_feature_importance(file='/Users/srinath/playground/data-science/BimboInventoryDemand/logs/feature-explore4.txt')
#parse_feature_explore_outputs()

#compare_submissions(
#    ['submissions_parts/xgb_ensamble_submission_1472484817.68.csv',
#     'submissions_parts/xgb_ensamble_submission_1472490059.13.csv',
#     'submissions_parts/xgb_ensamble_submission_1472495260.59.csv',
#     'submissions_parts/xgb_ensamble_submission_1472500466.16.csv'])

#analyze_error()

#product_raw_stats()
#analyze_error()
#break_dataset()
#build_sample_dataset()

#df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems5000_10000.csv')


#df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems300.csv')
#df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems0_10.csv')


#category_histograms(df)

#test_return_multiple_apply(df)

#add_precentage()
#parse_product_data()

#cluster_to_find_similar_products()
#test_one_hot()

#calculate_and_save_aggrigates(df)

#find_missing_products()
#print_base_stats(df)
#test2Dtransfrom(df)

#create_small_datafile(0,75, df)

# for i in range(0, 50000, 5000):
#     create_small_datafile(i,i+5000, df)
#create_small_test_datafile(0,100)
#create_small_datafile(0,1000, df)
#create_small_datafile(1000,2000, df)
#create_small_datafile(2000, 10000, df)



#df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/train.csv')
#agency_product_stats(df)

#testset_stats()

#build_product_info()

#merge_data_and_test_files()

##submission_stats()

#testDf = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/test.csv')
#break_test_dataset(testDf)

#
#print testDf.describe()
#print testDf['Producto_ID'].min(), testDf['Producto_ID'].max()

#test_falttern_reshape()


#df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/train.csv')
#create_small_datafile(5000,15000, df)

#cluster_to_find_similar_products()
#build_product_info()

#build_state_info()


#analyze_error()
#raw_df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems5_10_35_40_45_50k.csv')
#raw_df = raw_df.sort_values(by=['Semana'])
#raw_df.to_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems5_10_35_40_45_50k2.csv', index=False)

#product_raw_stats()

#test_ensambles()
#merge_submissions()