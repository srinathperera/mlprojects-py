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

from inventory_demand import *
from sklearn.cluster import KMeans, MiniBatchKMeans

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
    return [description, brand, weight, pieces, has_choco, has_vanilla, has_multigrain]








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

    extracted_features_df =  pd.DataFrame(extracted_features_np, columns=['description', 'brand', 'weight', 'pieces', "has_choco", "has_vanilla", "has_multigrain"])

    print "have " + str(df.shape[0]) + "products"

    #vectorize names
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=200,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
    X = vectorizer.fit_transform(extracted_features_df['description'])

    print X

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



def convert_cat_feild2_interger(df, feild_name):
    brands = df[feild_name].unique()
    brands_data = np.column_stack([brands, range(len(brands))])
    brand_df =  pd.DataFrame(brands_data, columns=[feild_name, feild_name + '_id'])
    merged = pd.merge(df, brand_df, how='left', on=[feild_name])
    return drop_feilds_1df(merged, [feild_name])



def build_state_info():
    df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/town_state.csv')
    merged = convert_cat_feild2_interger(df, 'Town')
    merged = convert_cat_feild2_interger(merged, 'State')
    merged.to_csv('agency_data.csv', index=False)




def build_product_info():
    df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/producto_tabla.csv')
    extracted_features = [extract_data(p) for p in df['NombreProducto'].values]
    extracted_features_np = np.row_stack(extracted_features)
    extracted_features_df =  pd.DataFrame(extracted_features_np, columns=['description', 'brand', 'weight', 'pieces',
                                                                          "has_choco", "has_vanilla", "has_multigrain"])

    extracted_features_df['Producto_ID'] = df['Producto_ID']

    words = []
    for  w in  extracted_features_df['description']:
        wl = w.lower().split()
        words = words + wl
    uniqWords = sorted(set(words)) #remove duplicate words and sort
    wordcounts = [[word, words.count(word)] for word in uniqWords]

    wordcounts_np = np.row_stack(wordcounts)
    wordcounts_df =  pd.DataFrame(wordcounts_np, columns=['word', 'count'])
    wordcounts_df["count"] = wordcounts_df["count"].astype(int)

    wordcounts_df = wordcounts_df.sort_values(by=['count'], ascending=False)

    wordcounts_dict = dict(wordcounts_df.values)

    dic_id = dict(zip(wordcounts_df['word'], range(len(wordcounts_df['word']))))

    print wordcounts_dict


    word_type = []
    for  w in  extracted_features_df['description']:
        max_word = None
        max_count = 0
        for t in w.lower().split():
            c = wordcounts_dict.get(t, 0)
            if c > max_count:
                max_count = c
                max_word = t
        word_type.append(dic_id[max_word])

    extracted_features_df['product_word'] = word_type



    print extracted_features_df.shape
    extracted_features_df = drop_feilds_1df(extracted_features_df, ['description'])

    #replace brand with ID
    brands = extracted_features_df['brand'].unique()
    brands_data = np.column_stack([brands, range(len(brands))])
    print brands_data.shape
    brand_df =  pd.DataFrame(brands_data, columns=['brand', 'brand_id'])
    merged = pd.merge(extracted_features_df, brand_df, how='left', on=['brand'])
    merged = drop_feilds_1df(merged, ['brand'])

    #product based features
    df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/train.csv')

    #add time between features
    df1 = df[df['Demanda_uni_equil'] > 0]
    grouped = df1.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])['Semana']
    tbdelivery = grouped.apply(time_between_delivery)

    valuesDf = tbdelivery.to_frame("time_between_delivery_all")
    valuesDf.reset_index(inplace=True)
    tbdelivery_by_product = valuesDf.groupby(['Producto_ID'])['time_between_delivery_all'].mean()

    tbdelivery_by_product_df = tbdelivery_by_product.to_frame("time_between_delivery")
    tbdelivery_by_product_df.reset_index(inplace=True)

    merged = pd.merge(merged, tbdelivery_by_product_df, how='left', on=['Producto_ID'])

    #returns_by_product = df.groupby(['Producto_ID'])['Dev_uni_proxima'].sum()

    print merged.describe()

    merged = merged.fillna(1)
    #write to file
    merged.to_csv('product_data.csv', index=False)


def find_similar_client_data():
    df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/train.csv')

    clientMean = df.groupby(['Ruta_SAK', 'Cliente_ID'])['Demanda_uni_equil'].mean()
    clientMean_df = clientMean.to_frame("mean_sales")
    clientMean_df.reset_index(inplace=True)
    clientMean_df = clientMean_df.sort_values(by=['mean_sales'])
    clientMean_df['index'] = range(0, clientMean_df.shape[0])
    clientMean_df.to_csv('similar_client_data.csv', index=False)

    print clientMean_df.describe()

find_similar_client_data()



