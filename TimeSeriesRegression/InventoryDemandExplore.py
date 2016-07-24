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

explore_returns_and_delivery_freq()
