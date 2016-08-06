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
from data_explore import *


np.set_printoptions(precision=1, suppress=True)

def g2df_sum_mean(group):
    sum = group.sum()
    mean = group.mean()
    count = group.count()
    valuesDf = sum.to_frame("sum")
    valuesDf.reset_index(inplace=True)
    valuesDf['mean'] = mean.values
    valuesDf['count'] = count.values
    valuesDf['rank'] = valuesDf['mean']*np.log(1+valuesDf['count'])

    return valuesDf


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


def show_scatterplot(x,y, x_title, y_title, subplotnum):
    plt.subplot(subplotnum)
    plt.xlabel(x_title, fontsize=18)
    plt.ylabel(y_title, fontsize=18)
    plt.scatter(x, y, alpha=0.5, s=5)

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


#figure_id

def show_raw_error_by_feature(df, feature_name, chartloc):
    start  = time.time()
    plt.subplot(chartloc)
    plt.xlabel(feature_name , fontsize=18)
    plt.ylabel('Error', fontsize=18)
    x = df[feature_name] + 0.01*np.random.normal(size=df.shape[0])
    plt.scatter(x, df['error'].values, alpha=0.5, color='b')
    plt.scatter(x, df['error'], alpha=0.5, color='r')
    print "show_raw_by_feature", feature_name, "took", (time.time() - start), "s"




def show_error_by_feature(df, feature_name, chartloc, redo_x = True):
    start  = time.time()
    group1 = df.groupby([feature_name])['error']
    errors_by_feature = g2df_sum_mean(group1).sort("mean")


    #plt.subplot(chartloc)
    #plt.xlabel(feature_name + " in sorted order by Mean", fontsize=18)
    #plt.ylabel('Mean Error/Rank', fontsize=18)
    #plt.yscale('log')
    #plt.xscale('log')
    #plt.scatter(errors_by_feature[feature_name].values, errors_by_feature['mean'].values, alpha=0.5)
    if redo_x:
        x = range(0, errors_by_feature.shape[0])
    else:
        x = errors_by_feature[feature_name]

    data = [(x, errors_by_feature['mean'].values), (x, np.log(errors_by_feature['rank'].values))]
    draw_scatterplot(data, "Mean Error, Rank vs. "+ feature_name, chartloc, c =['b', 'r'])

    #plt.scatter(x, errors_by_feature['mean'].values, alpha=0.5, color='b', s=5)
    #plt.scatter(x, np.log(errors_by_feature['rank'].values), alpha=0.5, color='r', s=5)

    print "show_error_by_feature", feature_name, "took", (time.time() - start), "s"

    return errors_by_feature


def print_data(group):
    print ">",group.values

def find_missing_feild_stats(df, feild_name):
    print ">>",feild_name
    training_set_size = int(0.7*df.shape[0])
    test_set_size = df.shape[0] - training_set_size


    train = df[:training_set_size]
    test = df[-1*test_set_size:]

    #train = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/train.csv')
    train_ids = train[feild_name].unique()
    #test = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/test.csv')
    test_ids = test[feild_name].unique()

    missing_ids = pd.Index(test_ids).difference(pd.Index(train_ids))
    print "missing ID count ", len(missing_ids)

    missing_ids_df =  pd.DataFrame(missing_ids, columns=[feild_name])
    #missing_ids_df.to_csv('missing_ids.csv', index=False)

    print "missing ", feild_name, " count", missing_ids_df.shape

    entries_with_missing = pd.merge(test, missing_ids_df, on=[feild_name])

    print "Mising entries=", entries_with_missing.shape[0], "percentage=", float(entries_with_missing.shape[0])*100/test.shape[0]

    print "full entries count", test.shape[0]

def find_missing_data():
    df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems5_10_35_40_45_50k.csv')
    find_missing_feild_stats(df, 'Producto_ID')
    find_missing_feild_stats(df, 'Canal_ID')
    find_missing_feild_stats(df, 'Ruta_SAK')
    find_missing_feild_stats(df, 'Cliente_ID')
    find_missing_feild_stats(df, 'Agencia_ID')





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
    erbypid = g2df_sum_mean(group1).sort("rank")

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



def show_timelines_toperrors(errordf, full_df, top_errors_count=50, cmd=0):
    group1 = errordf.groupby(['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID'])['error']
    top_error_groups = g2df_sum_mean(group1).sort_values(by=['sum'], ascending=False)

    top_error_groups = top_error_groups.head(top_errors_count)
    #top_error_groups = top_error_groups.head(1000)

    #print top_error_groups

    merged_with_raw = pd.merge(full_df, top_error_groups, on=['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID'])
    print merged_with_raw

    merged_with_raw.groupby(['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID'])['Demanda_uni_equil'].apply(print_data)

    merged_with_forecasts = pd.merge(errordf, top_error_groups, on=['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID'])
    print merged_with_forecasts

    plt.figure(2, figsize=(20,10))

    plt.subplot(321)
    merged_with_raw['Semana'] = merged_with_raw['Semana'] + np.random.normal(size=merged_with_raw.shape[0])

    plt.xlabel("Demnad", fontsize=18)
    plt.ylabel('Semana', fontsize=18)
    #plt.scatter(merged_with_raw['Semana'], merged_with_raw['Demanda_uni_equil'], color='b', alpha=0.3)
    plt.plot(merged_with_raw['Semana'], merged_with_raw['Demanda_uni_equil'], color='b', alpha=0.6)
    plt.plot(merged_with_forecasts['Semana'], merged_with_forecasts['predictions'], color='r', alpha=0.3)

    merged_with_forecasts['Semana'] = merged_with_forecasts['Semana'] + np.random.normal(size=merged_with_forecasts.shape[0])
    plt.subplot(322)
    plt.xlabel("Demnad", fontsize=18)
    plt.ylabel('Semana', fontsize=18)
    plt.plot(merged_with_forecasts['Semana'], merged_with_forecasts['predictions'], color='r', alpha=0.3)

    plt.tight_layout()
    plt.savefig('top_error_timeline-'+str(cmd)+'.png')

def show_raw_error_by_features(df, cmd):
    create_fig()
    df = df.sample(min(100000, df.shape[0]))
    show_raw_error_by_feature(df, 'Producto_ID', 321)
    show_raw_error_by_feature(df, 'Agencia_ID', 322)
    show_raw_error_by_feature(df, 'Cliente_ID', 323)
    show_raw_error_by_feature(df, 'Canal_ID', 324)
    show_raw_error_by_feature(df, 'Ruta_SAK', 325)

    show_raw_error_by_feature(df, 'Semana', 326)

    plt.tight_layout()
    plt.savefig('error_raw_by_features-'+str(cmd)+'.png')



def show_error_by_features(df, cmd):
    create_fig()
    err_by_products = show_error_by_feature(df, 'Producto_ID', 331)
    show_error_by_feature(df, 'Agencia_ID', 332)
    show_error_by_feature(df, 'Cliente_ID', 333)
    show_error_by_feature(df, 'Canal_ID', 334)
    show_error_by_feature(df, 'Ruta_SAK', 335)

    show_error_by_feature(df, 'Semana', 336, redo_x=False)

    #show_scatterplot(df['actual'],df['error'], 'actual', 'error', 337)
    #show_scatterplot(df['predictions'],df['error'], 'predictions', 'error', 338)

    plt.subplot(337)
    plt.xlabel("demand_value", fontsize=18)
    plt.ylabel("error", fontsize=18)
    plt.scatter(df['actual'], df['error'], alpha=0.5, s=5, c='b')
    plt.scatter(df['predictions'], df['error'], alpha=0.5, s=5, c='r')


    plt.tight_layout()
    plt.savefig('error_by_features-'+str(cmd)+'.png')

def show_error_distribution(errors):
    plt.hist(errors, 50, normed=1, facecolor='green', alpha=0.75)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    plt.tight_layout()
    plt.savefig('error_dist.png')


def show_error_distribution_all_forecasts(forecasts_df):
    column_names = list(forecasts_df)
    error_list = [calculate_error(forecasts_df['actual'], forecasts_df[c]) for c in column_names[:-1]]

    #error dist on one
    #error by id on one
    #error by value one one
    colors_list = ['r', 'b', 'g', 'y','c', 'm', 'k']

    create_fig()

    plt.subplot(331)
    plt.xlabel('Error')
    plt.ylabel('Frequency')

    for i, e in enumerate(error_list):
        plt.hist(e, 50, normed=1, facecolor=colors_list[i], alpha=0.75)


    x = range(forecasts_df.shape[0])


    draw_scatterplot([(x, e) for e in error_list], "Id", 332, c=colors_list)

    draw_scatterplot([(forecasts_df['actual'], e) for e in error_list], "Actual Value", 333, c=colors_list)


    plt.tight_layout()
    plt.savefig('error_dist.png')


def calculate_error(forecast, actual):
    return np.log(forecast + 1) - np.log(actual + 1)



def do_error_analysis(df, cmd, full_df):
    df['error'] = np.log(df['actual'].values +1) - np.log(df['predictions'].values + 1)
    print df.describe()
    show_error_distribution(df['error'])
    show_error_by_features(df, cmd)
    show_raw_error_by_features(df, cmd)
    show_timelines_toperrors(df, full_df, cmd=cmd)


def print_submission_data(sub_df=None, sub_file=None, show=False, command=0):
    if sub_df is None:
        sub_df = pd.read_csv(sub_file)
    print sub_df.describe()

    sub_df = sub_df.sample(100000)

    plt.figure(1, figsize=(20,10))
    plt.subplot(321)
    plt.xlabel('id', fontsize=18)
    plt.ylabel('Demanda_uni_equil', fontsize=18)
    #plt.yscale('log')
    #plt.xscale('log')
    plt.scatter(sub_df['id'].values, sub_df['Demanda_uni_equil'].values, alpha=0.5)
    plt.tight_layout()
    plt.savefig('submission_stats-'+str(command)+'.png')
    if(show):
        plt.show()

def test_error_analysis():
    error_df = pd.read_csv('prediction_with_data5.csv')
    full_df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/train-rsample-10m.csv')
    error_df = error_df.sample(10000)
    do_error_analysis(error_df, 0, full_df)
    #error_df = error_df.head(10000)

#print_submission_data(sub_file='/Users/srinath/playground/data-science/BimboInventoryDemand/submission/temp/en_submission_final.csv', show=True)
#print_submission_data('/Users/srinath/playground/data-science/BimboInventoryDemand/submission/final.csv', show=True)

#find_missing_data()

def test_scatter_plot():
    create_fig()
    count = 99
    data1 = (np.random.rand(count,1), np.random.rand(count,1))
    data2 = (np.random.rand(count,1), np.random.rand(count,1))
    draw_scatterplot([data1, data2], "X vs. Y", 311, c=['b', 'r'])
    plt.tight_layout()
    plt.savefig('test.png')


#test_error_analysis()

forecasts_df = pd.read_csv('/Users/srinath/code/workspace/mlprojects-py/TimeSeriesRegression/agr_cat1/model_forecasts6.csv')
show_error_distribution_all_forecasts(forecasts_df)

#show_error_by_features()
#show_timelines_toperrors()

#test_scatter_plot()