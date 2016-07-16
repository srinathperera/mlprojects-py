import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mltools import apply_zeroMeanUnit2D, preprocess2DtoZeroMeanUnit, undo_zeroMeanUnit2D
import time

import re

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans


from inventory_demand import *

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


def merge_submissions():
    submission_files = [pd.read_csv('submission'+str(i)+'.csv') for i in range(5)]

    tot = 0
    for df in submission_files:
        print "size",df.shape[0]
        tot = tot + df.shape[0]
    print "submission size", tot

    result = pd.concat(submission_files).sort_values(by=['id'])
    result.to_csv('final.csv', index=False)
    print "submission file of shape ", result.shape, " created"




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

#Tostada Ondulada Reg 30p 360g CU2 MR 6085
#p = re.compile('loss.*?\[(.*?)\].*?val_loss.*?\[(.*?)\].*?NN(.*?)AC_errorRate=([.0-9]+) RMSEP=([.0-9]+)')
p = re.compile('([^0-9]+) ([0-9][0-9pKkgG\s]{1,}).*')


def contains_pattern(str, pattern):
    #regexp = re.compile(r'ba[r|z|d]')
    regexp = re.compile(pattern)
    return regexp.search(str) is not None


def extract_data(product_str):
    tokens = product_str.split()
    brand = tokens[-2]
    description = None
    weight = -1
    pieces = -1

    m = p.search(product_str)
    if m:
        description = m.group(1)
        weight_size = m.group(2)
        if(weight_size):
            tokens = weight_size.split()
            for t in tokens:
                if len(t) > 1:
                    if t.endswith('p'):
                        pieces = int(t.replace('p',''))
                    elif t.endswith('kg') or t.endswith('Kg'):
                        weight = 1000*int(t.replace('Kg','').replace('kg', ''))
                    elif t.endswith('g'):
                        weight = int(t.replace('g',''))

    #    print 'G1: ',
    #    print "G2: ",
    else:
        description = product_str

    has_choco = contains_pattern(description, r'Choco')
    has_vanilla = contains_pattern(description, r'Va(i)?nilla')
    has_multigrain = contains_pattern(description, r'Multigrano')
    return [description, brand, weight, pieces]




def parse_product_data():
    df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/producto_tabla.csv')
    for p in df['NombreProducto'].values:
        print extract_data(p)
    print "Done"


def cluster_to_find_similar_products():
    df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/producto_tabla.csv')
    labels = df['Producto_ID']
    extracted_features = [extract_data(p) for p in df['NombreProducto'].values]
    extracted_features_np = np.row_stack(extracted_features)

    extracted_features_df =  pd.DataFrame(extracted_features_np, columns=['description', 'brand', 'weight', 'pieces'])

    print "have " + str(df.shape[0]) + "products"

    #vectorize names
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=200,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
    X = vectorizer.fit_transform(extracted_features_df['description'])

    print("n_samples: %d, n_features: %d" % X.shape)

    print("Performing dimensionality reduction using LSA")
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(5)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)


    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print("new size", X.shape)
    print type(X)


    extracted_features_df = encode_onehot(extracted_features_df, ['brand'])
    extracted_features_df = drop_feilds_1df(extracted_features_df,['description'])

    print "X,df", X.shape, extracted_features_df.values.shape

    X = np.hstack((X, extracted_features_df.values))



    # Do the actual clustering
    km = KMeans(n_clusters=10, init='k-means++', max_iter=100, n_init=1,
                verbose=True)

    print("Clustering sparse data with %s" % km)
    #km.fit(X)

    results = km.fit_predict(X)
    print len(results), results



    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index: %.3f"
        % metrics.adjusted_rand_score(labels, km.labels_))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, km.labels_, sample_size=1000))

    print()

    products_clusters = np.column_stack([labels, results])
    to_saveDf =  pd.DataFrame(products_clusters, columns=["Producto_ID","Cluster"])
    to_saveDf.to_csv('product_clusters.csv', index=False)

    to_saveDf['NombreProducto']= df['NombreProducto']

    #grouped = to_saveDf.groupby(['Cluster'])['NombreProducto']
    #grouped.apply(print_cluster)



def find_similar_products():
    df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/producto_tabla.csv')

    labels = df['Producto_ID']
    print "have " + str(df.shape[0]) + "products"

    vectorizer = TfidfVectorizer(max_df=0.5, max_features=200,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
    X = vectorizer.fit_transform(df['NombreProducto'])

    print("n_samples: %d, n_features: %d" % X.shape)

    print type(X)
    print X



    print("Performing dimensionality reduction using LSA")
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(5)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)


    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print("new size", X.shape)
    print type(X)
    print X



    # Do the actual clustering
    km = KMeans(n_clusters=30, init='k-means++', max_iter=100, n_init=1,
                verbose=True)

    print("Clustering sparse data with %s" % km)
    #km.fit(X)

    results = km.fit_predict(X)
    print len(results), results



    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index: %.3f"
        % metrics.adjusted_rand_score(labels, km.labels_))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, km.labels_, sample_size=1000))

    print()

    products_clusters = np.column_stack([labels, results])
    to_saveDf =  pd.DataFrame(products_clusters, columns=["Producto_ID","Cluster"])
    to_saveDf.to_csv('product_clusters.csv', index=False)

    to_saveDf['NombreProducto']= df['NombreProducto']

    grouped = to_saveDf.groupby(['Cluster'])['NombreProducto']
    grouped.apply(print_cluster)

def print_cluster(group):
    print group.values
    return None


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





#find_similar_products()

#analyze_error()

#product_raw_stats()
#analyze_error()
#break_dataset()
#build_sample_dataset()

#df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems5000_10000.csv')
#df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/train.csv')


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

merge_submissions()

##submission_stats()

#testDf = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/test.csv')
#break_test_dataset(testDf)

#
#print testDf.describe()
#print testDf['Producto_ID'].min(), testDf['Producto_ID'].max()

#test_falttern_reshape()


