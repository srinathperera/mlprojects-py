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

from inventory_demand import *
from data_explore import *

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

def time_between_delivery2(group):
    deliveries = np.sort(group['Semana'].values)
    sales_count = np.sort(group['Venta_uni_hoy'].values)
    sales = np.sort(group['Venta_hoy'].values)



    if len(deliveries) == 0:
        return [100, 0, 0]
    rep = np.sum([math.pow(2,w-3) for w in deliveries])
    if len(deliveries) == 1:
        return [8, 0, rep]
    time_between_delivery = [deliveries[i] - deliveries[i-1] for i in range(1,len(deliveries))]
    #print "W", deliveries, np.median(time_between_delivery), np.mean(time_between_delivery), np.std(time_between_delivery)
    #print "S",sales
    #print "SU",sales_count
    return [np.median(time_between_delivery), np.std(time_between_delivery), rep]



def explore_returns_and_delivery_freq():
    df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems5000_15000.csv')

    df1 = df[df['Venta_uni_hoy'] > 0]
    grouped = df1.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
    tbdelivery = grouped.apply(time_between_delivery2)

    tbdelivery_df = tbdelivery.to_frame("delivery_data")
    tbdelivery_df.reset_index(inplace=True)
    tbdelivery_df = expand_array_feild_and_add_df(tbdelivery_df, 'delivery_data',
                                                  ["mean", "stddev", "rep"])
    print tbdelivery_df.describe()

    print tbdelivery_df.groupby('rep')['rep'].count()

    '''

    valuesDf = tbdelivery.to_frame("pg_time_between_delivery_mean")
    valuesDf.reset_index(inplace=True)
    tbdelivery_by_product = valuesDf.groupby(['Producto_ID'])['time_between_delivery'].mean()

    plt.figure(1, figsize=(20,10))
    plt.subplot(321)
    plt.xlabel('Producto_ID', fontsize=18)
    plt.ylabel('time_between_delivery', fontsize=18)
    #plt.yscale('log')
    #plt.xscale('log')
    plt.scatter(tbdelivery_by_product.index, tbdelivery_by_product.values, alpha=0.5)

    returns_by_product = df.groupby(['Producto_ID'])['Dev_uni_proxima'].sum()
    plt.subplot(322)
    plt.xlabel('Producto_ID', fontsize=18)
    plt.ylabel('time_between_delivery', fontsize=18)
    plt.scatter(returns_by_product.index, returns_by_product.values, alpha=0.5)


    #tbdelivery = grouped.apply(time_between_delivery)


    plt.tight_layout()
    plt.show()
    '''



def explore_product_stats():
    df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems300.csv')

    grouped = df.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])['Demanda_uni_equil']

    slopeMap = grouped.apply(np.mean)
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

def show_product_stats(df, feature_name, redo_x=True):
    start  = time.time()
    create_fig()
    group = df.groupby([feature_name])
    group1 = group['Demanda_uni_equil']

    countData = group1.count()
    valuesDf = countData.to_frame("count")
    valuesDf.reset_index(inplace=True)
    #valuesDf['sum'] = group1.sum()
    #valuesDf['sum'] = group1.std()
    #valuesDf['sum'] = group1.mean()


    plt.subplot(331)
    plt.xlabel(feature_name + " in sorted order by Mean", fontsize=18)
    plt.ylabel('Count', fontsize=18)
    if redo_x:
        x = range(0, valuesDf.shape[0])
    else:
        x = valuesDf[feature_name]
    plt.scatter(x, valuesDf['count'], alpha=0.3, color='b')

    plt.subplot(332)
    plt.xlabel(feature_name + "Index", fontsize=18)
    plt.ylabel('Sum', fontsize=18)
    plt.scatter(x, group1.sum(), alpha=0.3, color='r')

    plt.subplot(333)
    plt.xlabel(feature_name + "Index", fontsize=18)
    plt.ylabel('Std', fontsize=18)
    plt.scatter(x, group1.std(), alpha=0.3, color='r')

    plt.subplot(334)
    #plt.xlabel(feature_name + "Index", fontsize=18)
    #plt.ylabel('Demanda_uni_equil, Percentiles', fontsize=18)
    #plt.scatter(x, group1.mean(), alpha=0.5, color='r')
    #plt.scatter(x, group1.median(), alpha=0.5, color='b')
    #plt.scatter(x, group1.quantile(0.1, interpolation='nearest'), alpha=0.3, color='r')
    #plt.scatter(x, group1.quantile(0.9, interpolation='nearest'), alpha=0.3, color='b')
    draw_error_bars(334, x, group1.mean(), group1.quantile(0.1, interpolation='nearest'),
                    group1.quantile(0.9, interpolation='nearest'), feature_name, y_label='Venta_hoy, Percentiles')


    group2 = group['Venta_hoy']
    #plt.subplot(335)
    #plt.xlabel(feature_name + "Index", fontsize=18)
    #plt.ylabel('Venta_hoy Percentiles', fontsize=18)
    #plt.scatter(x, group2.mean(), alpha=0.5, color='r')
    #plt.scatter(x, group2.median(), alpha=0.5, color='b')
    #plt.scatter(x, group2.quantile(0.1, interpolation='nearest'), alpha=0.3, color='r')
    #plt.scatter(x, group2.quantile(0.9, interpolation='nearest'), alpha=0.3, color='b')

    draw_error_bars(335, x, group2.mean(), group2.quantile(0.1, interpolation='nearest'),
                    group2.quantile(0.9, interpolation='nearest'), feature_name, y_label='Venta_hoy, Percentiles', do_log=False)



    group3 = group['Dev_proxima']
    #plt.subplot(336)
    #plt.xlabel(feature_name + "Index", fontsize=18)
    #plt.ylabel('Dev_proxima, Percentiles', fontsize=18)
    #plt.scatter(x, group3.mean(), alpha=0.5, color='r')
    #plt.scatter(x, group3.median(), alpha=0.5, color='b')
    #plt.scatter(x, group3.quantile(0.1, interpolation='nearest'), alpha=0.3, color='r')
    #plt.scatter(x, group3.quantile(0.9, interpolation='nearest'), alpha=0.3, color='b')

    draw_error_bars(336, x, group3.mean(), group3.quantile(0.1, interpolation='nearest'),
                    group3.quantile(0.9, interpolation='nearest'), feature_name, y_label='Dev_proxima, Percentiles', do_log=False)



    group3 = group['Venta_uni_hoy']
    ax = plt.subplot(337)
    plt.xlabel(feature_name + "Index", fontsize=18)
    plt.ylabel('Venta_uni_hoy, Percentiles', fontsize=18)
    ax.set_yscale('log')

    plt.errorbar(x, group3.mean(), yerr=[group3.quantile(0.1, interpolation='nearest'), group3.quantile(0.9, interpolation='nearest')],
                 ms=5, mew=2, fmt='--o')
    #plt.scatter(x, group3.mean(), alpha=0.5, color='r')
    #plt.scatter(x, group3.median(), alpha=0.5, color='b')
    #plt.scatter(x, group3.quantile(0.1, interpolation='nearest'), alpha=0.3, color='r')
    #plt.scatter(x, group3.quantile(0.9, interpolation='nearest'), alpha=0.3, color='b')
    plt.savefig('stats-'+feature_name+'.png')


    group3 = group['Dev_uni_proxima']
    draw_error_bars(338, x, group3.mean(), group3.quantile(0.1, interpolation='nearest'),
                    group3.quantile(0.9, interpolation='nearest'), feature_name, y_label='Dev_uni_proxima, Percentiles',
                    do_log=False)

    #plt.subplot(338)
    #plt.xlabel(feature_name + "Index", fontsize=18)
    #plt.ylabel('Dev_uni_proxima, Percentiles', fontsize=18)
    #plt.scatter(x, group3.mean(), alpha=0.5, color='r')
    #plt.scatter(x, group3.median(), alpha=0.5, color='b')
    #plt.scatter(x, group3.quantile(0.1, interpolation='nearest'), alpha=0.3, color='r')
    #plt.scatter(x, group3.quantile(0.9, interpolation='nearest'), alpha=0.3, color='b')
    plt.savefig('stats-'+feature_name+'.png')

    print feature_name, "took", (time.time() - start), "s"


def show_feature_stats():
    #df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems300.csv')

    df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/train.csv')
    df = df.sample(1000000)
    for f in ['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID']:
        print "processing ", f
        show_product_stats(df,f)

#show_feature_stats()

#explore_returns_and_delivery_freq()
