import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mltools import apply_zeroMeanUnit2D, preprocess2DtoZeroMeanUnit, undo_zeroMeanUnit2D
import time


from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans


from inventory_demand import *




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


def show_timeline(plotid, x, y1, y2):
    plt.subplot(plotid)
    plt.scatter(x, y1, color='b')
    plt.scatter(x, y2, color='r')



def show_errors_by_feature(error_data):
    plt.figure(1, figsize=(20,10))
    plt.subplot(321)
    plt.xlabel('Producto_ID', fontsize=18)
    plt.ylabel('Error', fontsize=18)
    #plt.yscale('log')
    #plt.xscale('log')
    plt.scatter(error_data['Producto_ID'].values, error_data['mean'].values, alpha=0.5)

    plt.subplot(322)
    plt.xlabel('Cluster', fontsize=18)
    plt.ylabel('Error Sum', fontsize=18)
    plt.scatter(error_data['Cluster'].values, error_data['mean'].values, alpha=0.5)

    plt.subplot(323)
    plt.xlabel('Percent', fontsize=18)
    plt.ylabel('Error Sum', fontsize=18)
    plt.scatter(error_data['percent'].values, error_data['mean'].values, alpha=0.5)

    plt.tight_layout()
    plt.show()

def show_errors_by_timeline(df):
    plt.figure(1, figsize=(20,10))

    show_timeline(321, df['Semana'].values, df['actual'].values, df['predictions'].values)

    plt.tight_layout()
    plt.show()





def show_error_by_feature(df, feature_name, chartloc):
    group1 = df.groupby([feature_name])['error']
    errors_by_feature = group_to_df_sum_mean(group1).sort("mean")

    plt.subplot(chartloc)
    plt.xlabel(feature_name + " in sorted order by Mean", fontsize=18)
    plt.ylabel('Mean Error', fontsize=18)
    #plt.yscale('log')
    #plt.xscale('log')
    #plt.scatter(errors_by_feature[feature_name].values, errors_by_feature['mean'].values, alpha=0.5)
    plt.scatter(range(0, errors_by_feature.shape[0]), errors_by_feature['mean'].values, alpha=0.5, color='b')
    plt.scatter(range(0, errors_by_feature.shape[0]), np.log(errors_by_feature['count'].values), alpha=0.5, color='r')

    return errors_by_feature

def show_error_by_features():
    df = pd.read_csv('forecast_with_data.csv')
    df['error'] = np.abs(np.log(df['actual'].values +1) - np.log(df['predictions'].values + 1))

    plt.figure(1, figsize=(20,10))

    err_by_products = show_error_by_feature(df, 'Producto_ID', 321)
    show_error_by_feature(df, 'Agencia_ID', 322)
    show_error_by_feature(df, 'Cliente_ID', 323)
    show_error_by_feature(df, 'Canal_ID', 324)
    show_error_by_feature(df, 'Ruta_SAK', 325)


    clusters = pd.read_csv('product_clusters.csv')
    df_with_cluster = pd.merge(df, clusters, how='left', on=['Producto_ID'])
    print "df_with_cluster", df_with_cluster.shape
    show_error_by_feature(df_with_cluster, 'Cluster', 326)


    plt.tight_layout()
    plt.show()



def top_errors():
    df = pd.read_csv('forecast_with_data.csv')
    df['error'] = np.abs(np.log(df['actual'].values +1) - np.log(df['predictions'].values + 1))

    print df.describe()

    #75% of error is 0.558213


    neg_error = df[df['predictions'] > np.exp(1)-1]
    print neg_error['error'].count(), neg_error['error'].mean()



    top_error_df = df[df['error'] > 558213]
    df = df[df['error'] > 0.6]


    print df.groupby(['Semana'])['error'].sum()

    print df.groupby(['Cluster'])['error'].sum()


    group1 = df.groupby(['Producto_ID'])['error']
    erbypid = group_to_df_sum_mean(group1).sort("rank")

    #print erbypid

    #plot top product's real and forecasted values



    clusters = pd.read_csv('product_clusters.csv')
    erbypid_with_clusters = pd.merge(erbypid, clusters, how='left', on=['Producto_ID'])

    products = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/producto_tabla.csv')
    erbypid_with_clusters_products = pd.merge(erbypid_with_clusters, products, how='left', on=['Producto_ID'])


    product_use = pd.read_csv('product_id_count_full.csv')
    with_use_df = pd.merge(erbypid_with_clusters_products, product_use, how='left', on=['Producto_ID'])

    final_df = drop_feilds_1df(with_use_df, ['NombreProducto', 'count_y', 'sum', 'mean', 'count_x', "rank"])
    #final_df = drop_feilds_1df(with_use_df, ['NombreProducto', 'count_y', 'sum', 'count_x'])
    final_df.to_csv('product_more_data.csv', index=False)


    print with_use_df

    #show_errors_by_feature(with_use_df)

    products_most_errors_df = erbypid.tail(100)
    print products_most_errors_df
    entries4products = pd.merge(products_most_errors_df, df, how='left', on=['Producto_ID'])
    g = entries4products.groupby(['Producto_ID'])

    mean_diff_for_top =  pd.DataFrame(np.column_stack([g['actual'].mean().values, g['predictions'].mean().values]), columns=["actual","predictions"])
    print mean_diff_for_top

    show_errors_by_timeline(entries4products)







show_error_by_features()

