import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import time
import math
from sklearn.externals import joblib

from mltools import undoPreprocessing
from keras.optimizers import Adam
from sklearn.linear_model import LinearRegression
from tsforecasttools import run_timeseries_froecasts, regression_with_xgboost, regression_with_xgboost_no_cv
from sklearn.feature_extraction import DictVectorizer
from sklearn import  preprocessing


from mltools import preprocess2DtoZeroMeanUnit, preprocess1DtoZeroMeanUnit, train_test_split, print_feature_importance, apply_zeroMeanUnit2D
from mltools import calculate_rmsle, almost_correct_based_accuracy, MLConfigs, print_regression_model_summary, regression_with_dl, apply_zeroMeanUnit, undo_zeroMeanUnit2D
from keras.optimizers import Adam

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from inventory_demand import *

from mltools import *
import scipy

class CompositeFeatures:
    def __init__(self, featues_names, default_stats, is_single=True, short_name=None, aggr_feild='Demanda_uni_equil',
                 fops=FeatureOps()):
        self.featues_names = featues_names
        self.is_single = is_single
        if short_name is None:
            if not is_single:
                raise ValueError("coposite features must have a short name")
            self.short_name = featues_names[0]+"_"+aggr_feild
        else:
            self.short_name = short_name
        self.aggr_feild = aggr_feild
        self.fops = fops
        self.default_stats=default_stats




def generate_features_with_stats(conf, train_df, df_list, f_start_map=None, feature_details=None):
    start = time.time()

    default_demand_stats = DefaultStats(mean=train_df['Demanda_uni_equil'].mean(), count=train_df['Demanda_uni_equil'].count(),
                                        stddev=train_df['Demanda_uni_equil'].std())
    default_venta_hoy_stats = DefaultStats(train_df['Venta_hoy'].mean(), train_df['Venta_hoy'].count(),
                                        train_df['Venta_hoy'].std())
    default_dev_proxima_stats = DefaultStats(train_df['Dev_proxima'].mean(), train_df['Dev_proxima'].count(),
                                        train_df['Dev_proxima'].std())

    if feature_details is None:
        feature_details = [
            CompositeFeatures(['Agencia_ID'], default_demand_stats, is_single=True),
            CompositeFeatures(['Ruta_SAK', 'Cliente_ID'], default_demand_stats, is_single=False, short_name='clients_combined')
        ]

    if f_start_map is None:
        f_start_map = dict()

        for f in feature_details:
            if f.is_single:
                fstats = calculate_feild_stats(train_df, f.featues_names[0], f.aggr_feild, f.default_stats, f.fops)
            else:
                fstats = calculate_multiple_feild_stats(train_df, f.featues_name, f.agr_feild, f.default_stats,
                                                    f.short_name, f.fops)
            f_start_map[f.short_name] = fstats


    full_df = df_list + train_df
    converted_df_list = []
    for tdf in full_df:
        for f in feature_details:
            if f.is_single:
                tdf = add_single_feild_stats(tdf, fstats, f.featues_names[0], f.default_stats,f.fops, drop=False,
                            agr_feild=f.agr_feild)
            else:
                tdf = add_multiple_feild_stats(tdf, fstats, f.feild_names, f.short_name, f.default_stats)
        converted_df_list.append(tdf)

    print "generate_features took ", (time.time() - start), "s"
    return converted_df_list

    '''
    use_slope = False
    use_group_aggrigate = True
    use_product_features = True
    use_agency_features = False
    use_sales_data = False

    testDf = subdf

    if use_slope:
        train_df, test_df, testDf = addSlopes(train_df, test_df, testDf)



    #add mean and stddev by groups

    groups = ('Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID')
    measures = ('Dev_proxima', 'Dev_uni_proxima', 'Demanda_uni_equil', 'unit_prize', 'Venta_uni_hoy', 'Venta_hoy')
    #for t in itertools.product(groups, measures):
    #for t in [('Cliente_ID', 'Demanda_uni_equil'), ('Cliente_ID', 'Venta_uni_hoy'), ('Cliente_ID', 'Venta_hoy'),
    #    ('Ruta_SAK', 'unit_prize')]:
    #        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df,
    #                                test_df,t[0], testDf, drop=False, agr_feild=t[1])

    if use_group_aggrigate:
        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Agencia_ID', testDf, default_demand_stats,
                                                            FeatureOps(hmean=True, stddev=True, count=True))
        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Canal_ID', testDf, drop=False)
        #*train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Ruta_SAK', testDf, drop=False)
        #*train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Cliente_ID', testDf, drop=False) #duplicated
        train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Ruta_SAK', 'Cliente_ID'],
                'Demanda_uni_equil', "clients_combined", default_demand_stats,
                                                              FeatureOps(sum= True, kurtosis=True, stddev=True, count=True, p90=10, p10=True, hmean=True))

        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, default_demand_stats,
                                                            FeatureOps(stddev=True, p90=True, hmean=True,p10=True, count=True))

        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, drop=False, agr_feild='Venta_hoy',
                                                            default_stats=default_venta_hoy_stats, fops=FeatureOps(stddev=True, count=True))
        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Canal_ID', testDf, drop=False, agr_feild='Venta_hoy')
        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Ruta_SAK', testDf, drop=False, agr_feild='Venta_hoy')
        #*train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Cliente_ID', testDf, drop=False, agr_feild='Venta_hoy')
        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, drop=False, agr_feild='Venta_hoy')

        train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Ruta_SAK', 'Cliente_ID'],
            'Venta_hoy', "clients_combined_vh", default_venta_hoy_stats, FeatureOps(sum=True, hmean=True, p90=True, stddev=True))



        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, drop=False,
                                                            agr_feild='Dev_proxima', default_stats=default_dev_proxima_stats,
                                                            fops=FeatureOps(count=True))

        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Canal_ID', testDf, default_demand_stats,
        #                                                    drop=False, agr_feild='Dev_proxima', fops=FeatureOps())
        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Ruta_SAK', testDf, drop=False, agr_feild='Dev_proxima')
        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Cliente_ID', testDf, drop=False, agr_feild='Dev_proxima')
        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Agencia_ID', testDf, drop=False, agr_feild='Dev_proxima')


    if use_product_features:
        product_data_df = read_productdata_file('product_data.csv')
        #remove  unused feilds
        #product_data_df = drop_feilds_1df(product_data_df, ['has_vanilla','has_multigrain', 'has_choco', 'weight','pieces'])
        product_data_df = drop_feilds_1df(product_data_df, ['has_vanilla','has_multigrain', 'has_choco', "time_between_delivery"])

        train_df = pd.merge(train_df, product_data_df, how='left', on=['Producto_ID'])
        test_df = pd.merge(test_df, product_data_df, how='left', on=['Producto_ID'])
        testDf = pd.merge(testDf, product_data_df, how='left', on=['Producto_ID'])

        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'brand_id', testDf, drop=False,
        #    default_stats=default_demand_stats)
        train_df, test_df, testDf = drop_feilds(train_df, test_df, testDf, ['brand_id'])

        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'product_word', testDf, drop=False,
            default_stats=default_demand_stats, fops=FeatureOps(hmean=True))
        train_df, test_df, testDf = drop_feilds(train_df, test_df, testDf, ['product_word'])

    if use_agency_features:
        agency_data_df = read_productdata_file('agency_data.csv')
        train_df, test_df, testDf =  merge_csv_by_feild(train_df, test_df, testDf, agency_data_df, 'Agencia_ID')

        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Town_id', testDf, drop=False, fops=FeatureOps())
        train_df, test_df, testDf = drop_feilds(train_df, test_df, testDf, ['Town_id'])

        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'State_id', testDf, drop=False,
                                                            default_stats=default_demand_stats, fops=FeatureOps())
        train_df, test_df, testDf = drop_feilds(train_df, test_df, testDf, ['State_id'])

    if use_sales_data:
        train_df, test_df, testDf = add_time_bwt_delivery(train_df, test_df, testDf)
        train_df, test_df, testDf = add_last_sale_and_week(train_df, test_df, testDf)

    #train_df, test_df, testDf = add_five_grouped_stats(train_df, test_df, testDf)

    #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, drop=False, agr_feild='Dev_proxima')
    #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, drop=False, agr_feild='Venta_hoy')

    #train_df, test_df, testDf =  merge_clusters(train_df, test_df, testDf)
    #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Cluster', testDf, drop=False)
    #train_df, test_df, testDf = do_one_hot_all(train_df, test_df, testDf, ['Cluster'])

    #train_df, test_df, testDf = do_one_hot_all(train_df, test_df, testDf, ['Canal_ID'])

    #train_df, test_df, testDf = only_keep_top_categories(train_df, test_df,testDf, 'Producto_ID', 30)
    #train_df, test_df, testDf = do_one_hot_all(train_df, test_df, testDf, ['Producto_ID'])



    #train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Producto_ID', 'Agencia_ID'],
    #                                                      'Demanda_uni_equil', "agc_product", demand_val_mean, demand_val_stddev)

    #train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Canal_ID', 'Ruta_SAK', 'Cliente_ID'],
    #                                                      'Demanda_uni_equil', "routes_combined", demand_val_mean, demand_val_stddev)



    test_df_before_dropping_features = test_df

    #train_df, test_df, testDf = do_one_hot_all(train_df, test_df, testDf, ['Agencia_ID', 'Cliente_ID'])

    train_df, test_df, testDf = drop_feilds(train_df, test_df, testDf, ['Canal_ID','Cliente_ID','Producto_ID', 'Agencia_ID', 'Ruta_SAK'])
    #train_df, test_df, testDf = drop_feilds(train_df, test_df, testDf, ['Canal_ID','Cliente_ID','Producto_ID', 'Ruta_SAK'])
    #train_df, test_df, testDf = drop_feilds(train_df, test_df, testDf, ['Canal_ID','Cliente_ID', 'Ruta_SAK'])
    #train_df, test_df, testDf = drop_feilds(train_df, test_df, testDf, ['Canal_ID','Cliente_ID','Producto_ID'])


    #TODO explore this more http://pandas.pydata.org/pandas-docs/stable/missing_data.html
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    #following are feilds in test data that is not used
    train_df, test_df = drop_column(train_df, test_df, 'Venta_uni_hoy')
    train_df, test_df = drop_column(train_df, test_df, 'Venta_hoy')
    train_df, test_df = drop_column(train_df, test_df, 'Dev_uni_proxima')
    train_df, test_df = drop_column(train_df, test_df, 'Dev_proxima')

    print "generate_features took ", (time.time() - start), "s"
    return train_df, test_df, testDf, y_actual_test, test_df_before_dropping_features
    '''



def add_single_feild_stats(base_df, feild_stats, feild_name, default_stats,fops, drop=False,
                            agr_feild='Demanda_uni_equil'):
    start = time.time()

    calculate_ts = time.time()
    merged = pd.merge(base_df, feild_stats, how='left', on=[feild_name])
    #TODO is this ok?
    merged.fillna(0, inplace=True)

    merged[feild_name+"_"+agr_feild+"_Mean"].fillna(default_stats.mean, inplace=True)
    if fops.stddev:
        merged[feild_name+"_"+agr_feild+"_StdDev"].fillna(default_stats.stddev, inplace=True)
    if fops.count:
        merged[feild_name+"_"+agr_feild+"_Count"].fillna(default_stats.count, inplace=True)

    if drop:
        merged = merged.drop(feild_name,1)
    merged.fillna(0, inplace=True)

    print "addFeildStatsAsFeatures() "+ feild_name+ " took %f (%f, %f), size %s %f" %(time.time()-start, calculate_ts- start,
                                           time.time() - calculate_ts, feild_name, merged.shape[0])
    return merged


def calculate_multiple_feild_stats(bdf, feild_names, agr_feild, default_stats, name, fops):
    start = time.time()
    groupData = bdf.groupby(feild_names)[agr_feild]
    meanData = groupData.mean()

    #TODO check why data is NA
    valuesDf = meanData.to_frame(name+"_Mean")
    valuesDf.fillna(default_stats.mean, inplace=True)
    valuesDf.reset_index(inplace=True)
    if fops.stddev:
        stddevData = groupData.std()
        valuesDf[name+"_StdDev"] = stddevData.values
        valuesDf.fillna(default_stats.stddev, inplace=True)
    if fops.count:
        countData = groupData.count()
        valuesDf[name+"_Count"] = countData.values
        valuesDf.fillna(default_stats.count, inplace=True)
    if fops.sum:
        sumData = groupData.sum()
        valuesDf[name+"_sum"] = sumData.values
    start2_start = time.time()
    print "join_multiple_feild_stats: base stats took", (start2_start - start)
    print "start complex stat", (time.time() - start)
    if fops.p10:
        pcerntile10 = groupData.quantile(0.1, interpolation='nearest')
        valuesDf[name+"_pcerntile10"] = np.where(np.isnan(pcerntile10), 0, pcerntile10)
    print "took p10", (time.time() - start)
    if fops.p90:
        pcerntile90 = groupData.quantile(0.9, interpolation='nearest')
        valuesDf[name + "_pcerntile90"] = np.where(np.isnan(pcerntile90), 0, pcerntile90)
    print "took p90", (time.time() - start)
    if fops.kurtosis:
        kurtosis = groupData.apply(lambda x: min(scipy.stats.kurtosis(x), 10000))
        valuesDf[name+"_kurtosis"] = np.where(np.isnan(kurtosis), 0, kurtosis)
    print "took kurtosis", (time.time() - start)
    if fops.hmean:
        hmean = groupData.apply(calcuate_hmean)
        valuesDf[name+"_hMean"] = np.where(np.isnan(hmean), 0, hmean)
    print "took hmean", (time.time() - start)
    if fops.entropy:
        entropy = groupData.apply(lambda x: min(scipy.stats.entropy(x), 10000))
        valuesDf[name+"_entropy"] = np.where(np.isnan(entropy), 0, np.where(np.isinf(entropy), 10, entropy))
    print "took entropy", (time.time() - start)
    #if fops.use_close_products_missing and feild_names[0] == 'Ruta_SAK' and feild_names[1] == 'Cliente_ID':
    #    to_merge = pd.concat([testdf[['Ruta_SAK','Cliente_ID']], subdf[['Ruta_SAK','Cliente_ID']]])
    #    to_merge = to_merge.drop_duplicates()
    #    valuesDf = find_alt_for_missing(to_merge, valuesDf)
    #    print "Using close values for missing values"
    return valuesDf


def add_multiple_feild_stats(bdf, feild_stats, feild_names, name, default_stats):
    merge_start = time.time()
    merged = merge__multiple_feilds_stats_with_df(name, bdf, feild_stats, feild_names, default_stats)
    print "join_multiple_feild_stats: merge took", (time.time() - merge_start)

    return merged


def generate_all_features(conf, train_df, test_df, subdf, y_actual_test):
    start = time.time()
    default_demand_stats = DefaultStats(mean=train_df['Demanda_uni_equil'].mean(), count=train_df['Demanda_uni_equil'].count(),
                                        stddev=train_df['Demanda_uni_equil'].std())
    default_venta_hoy_stats = DefaultStats(train_df['Venta_hoy'].mean(), train_df['Venta_hoy'].count(),
                                        train_df['Venta_hoy'].std())
    default_dev_proxima_stats = DefaultStats(train_df['Dev_proxima'].mean(), train_df['Dev_proxima'].count(),
                                        train_df['Dev_proxima'].std())

    #this is to drop all features in one go
    feilds_to_drop = []

    #agency
    train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Agencia_ID', subdf, default_demand_stats,
                                                        FeatureOps(hmean=True, stddev=True, count=True))
    train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Agencia_ID', testDf, default_dev_proxima_stats, fops=FeatureOps(), agr_feild='Dev_proxima')
    train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Agencia_ID', testDf, default_venta_hoy_stats, fops=FeatureOps(), agr_feild='Venta_hoy')

    #client
    train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Ruta_SAK', 'Cliente_ID'],
            'Demanda_uni_equil', "clients_combined", default_demand_stats,
                                                          FeatureOps(sum= True, kurtosis=True, stddev=True, count=True, p90=10, p10=True, hmean=True))
    train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Ruta_SAK', 'Cliente_ID'],
            'Venta_hoy', "clients_combined_vh", default_demand_stats, FeatureOps())
    train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Ruta_SAK', 'Cliente_ID'],
            'Dev_proxima', "clients_combined_dp", default_demand_stats, FeatureOps())

    #client NN
    train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Producto_ID', 'Agencia_ID', 'Canal_ID',  'Ruta_SAK'],
            'Demanda_uni_equil', "client_nn", default_demand_stats,FeatureOps())
    train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Producto_ID', 'Agencia_ID', 'Canal_ID',  'Ruta_SAK'],
            'Venta_hoy', "client_nn_vh", default_demand_stats, FeatureOps())
    train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Producto_ID', 'Agencia_ID', 'Canal_ID',  'Ruta_SAK'],
            'Dev_proxima', "client_nn_dp", default_demand_stats, FeatureOps())

    train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Producto_ID', 'Agencia_ID', 'Canal_ID'],
            'Demanda_uni_equil', "client_nn_agency", default_demand_stats,FeatureOps())
    train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Producto_ID', 'Agencia_ID', 'Canal_ID'],
            'Venta_hoy', "client_nn_agency_vh", default_demand_stats, FeatureOps())
    train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Producto_ID', 'Agencia_ID', 'Canal_ID'],
            'Dev_proxima', "client_nn_agency_dp", default_demand_stats, FeatureOps())



    #product
    train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, default_demand_stats,
                                                        FeatureOps(stddev=True, p90=True, hmean=True,p10=True, count=True))
    train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, drop=False, agr_feild='Venta_hoy',
                                                        default_stats=default_venta_hoy_stats, fops=FeatureOps(stddev=True, count=True))
    train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, drop=False,
                                                        agr_feild='Dev_proxima', default_stats=default_dev_proxima_stats,
                                                        fops=FeatureOps(count=True))
    '''
    #Canal_ID
    train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Canal_ID', testDf, default_demand_stats, fops=FeatureOps())
    train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Canal_ID', testDf, default_venta_hoy_stats, fops=FeatureOps(), agr_feild='Venta_hoy')
    train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Canal_ID', testDf, default_dev_proxima_stats,
                                                        fops=FeatureOps(), agr_feild='Dev_proxima')

    #Ruta_SAK
    train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Ruta_SAK', testDf, default_demand_stats, fops=FeatureOps())
    train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Ruta_SAK', testDf, default_venta_hoy_stats, fops=FeatureOps(), agr_feild='Venta_hoy')
    train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Ruta_SAK', testDf, default_dev_proxima_stats, fops=FeatureOps(), agr_feild='Dev_proxima')
    '''

    ############ add product data
    product_data_df = read_productdata_file('product_data.csv')
    #remove  unused feilds
    product_data_df = drop_feilds_1df(product_data_df, ["time_between_delivery"])

    if 'weight' in product_data_df:
        weight = product_data_df['weight']
        product_data_df['weight'] = np.where(weight < 0, 0, weight)
    if 'pieces' in product_data_df:
        pieces = product_data_df['pieces']
        product_data_df['pieces'] = np.where(pieces < 0, 0, pieces)
    train_df = pd.merge(train_df, product_data_df, how='left', on=['Producto_ID'])
    test_df = pd.merge(test_df, product_data_df, how='left', on=['Producto_ID'])
    testDf = pd.merge(testDf, product_data_df, how='left', on=['Producto_ID'])

    #add product data aggrigates by groups
    train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'brand_id', testDf,
        default_stats=default_demand_stats, fops=FeatureOps())
    train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'product_word', testDf, drop=False,
        default_stats=default_demand_stats, fops=FeatureOps(hmean=True))
    feilds_to_drop = feilds_to_drop + ['product_word', 'brand_id']

    agency_data_df = read_productdata_file('agency_data.csv')
    train_df, test_df, testDf =  merge_csv_by_feild(train_df, test_df, testDf, agency_data_df, 'Agencia_ID')
    train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Town_id', testDf, default_demand_stats, fops=FeatureOps())
    train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'State_id', testDf, default_demand_stats, fops=FeatureOps())
    feilds_to_drop = feilds_to_drop + ['Town_id', 'State_id']

    #bdf, testdf, subdf, feild_names, agr_feild, name, default_stats, fops

    train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Producto_ID', 'Agencia_ID'],
                                                          'Demanda_uni_equil', "agc_product", default_demand_stats, fops=FeatureOps())
    train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Canal_ID', 'Ruta_SAK', 'Cliente_ID'],
                                                          'Demanda_uni_equil', "routes_combined", default_demand_stats, fops=FeatureOps())
    train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Agencia_ID', 'Ruta_SAK', 'Cliente_ID'],
                                                          'Demanda_uni_equil', "clients_route_agc", default_demand_stats, fops=FeatureOps())

    #train_df, test_df, testDf =  merge_clusters(train_df, test_df, testDf)
    #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Cluster', testDf, drop=False)
    #train_df, test_df, testDf = only_keep_top_categories(train_df, test_df,testDf, 'Producto_ID', 30)
    #train_df, test_df, testDf = do_one_hot_all(train_df, test_df, testDf, ['Producto_ID'])

    test_df_before_dropping_features = test_df

    train_data_feilds_to_drop = ['Venta_uni_hoy', 'Venta_hoy', 'Dev_uni_proxima', 'Dev_proxima', 'Demanda_uni_equil']
    train_df, test_df, _ = drop_feilds(train_df, test_df, None, feilds_to_drop + train_data_feilds_to_drop)
    testDf = drop_feilds_1df(testDf, feilds_to_drop)

    #TODO explore this more http://pandas.pydata.org/pandas-docs/stable/missing_data.html
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    print "generate_features took ", (time.time() - start), "s"
    return train_df, test_df, testDf, y_actual_test, test_df_before_dropping_features

def parse_list_from_str(list_as_str):
    list_as_str = list_as_str.replace('[','')
    list_as_str = list_as_str.replace(']','')
    list_as_str = list_as_str.replace('\'','')
    list_as_str = list_as_str.replace(' ','')

    items = list_as_str.split(',')
    return items


def parse_feature_explore_output(file_name, feature_importance_map):
    #[IDF1] ['clients_combined_vh_Mean_x', 'clients_combined_vhci_x', 'clients_combined_vh_median_x', 'Producto_ID_Venta_hoy_Mean', 'Producto_ID_Venta_hoyci', 'Producto_ID_Venta_hoy_median', 'Producto_ID_Dev_proxima_Mean', 'Producto_ID_Dev_proximaci', 'Producto_ID_Dev_proxima_median', 'agc_product_Mean', 'agc_productci', 'agc_product_median'] XGB 0.584072902792

    file = open(file_name,'r')
    data =  file.read()

    data = data.replace('\n','')
    data = re.sub(r'\[=+\'\].*?s', '', data)
    #28. feature 27 =Producto_ID_Dev_proxima_StdDev (0.002047)

    p1 = re.compile('\[IDF1\] (\[.*?\]) XGB ([0-9.]+)')

    readings = []
    for match in p1.finditer(data):
        feature_set = match.group(1)
        rmsle = float(match.group(2))
        if 0.56 < rmsle < 0.57:
            for f in parse_list_from_str(feature_set):
                count = feature_importance_map.get(f, 0)
                count += 1
                feature_importance_map[f] = count
        readings.append([feature_set, rmsle])

    df_data = np.row_stack(readings)
    para_sweep_df= pd.DataFrame(df_data, columns=['feature_set' , 'rmsle'])
    return para_sweep_df


def parse_feature_explore_outputs():
    feature_importance_map = dict()

    #data_df1 = parse_feature_explore_output('/Users/srinath/playground/data-science/BimboInventoryDemand/logs/feature-explore1.txt', feature_importance_map)
    #data_df2 = parse_feature_explore_output('/Users/srinath/playground/data-science/BimboInventoryDemand/logs/feature-explore.txt', feature_importance_map)
    #data_df3 = parse_feature_explore_output('/Users/srinath/playground/data-science/BimboInventoryDemand/logs/feature-explore2.txt', feature_importance_map)
    #data_df4 = parse_feature_explore_output('/Users/srinath/playground/data-science/BimboInventoryDemand/logs/feature-explore3.txt', feature_importance_map)
    data_df5 = parse_feature_explore_output('/Users/srinath/playground/data-science/BimboInventoryDemand/logs/feature-explore4.txt', feature_importance_map)


    #data_df = pd.concat([data_df1, data_df2, data_df3, data_df4])
    data_df = data_df5

    data_df = data_df.sort_values(by=['rmsle'])
    print data_df.head(20)

    feature_importance_df = pd.DataFrame(feature_importance_map.items(), columns=['feature', 'count'])
    feature_importance_df = feature_importance_df.sort_values(by=['count'], ascending=False)

    print feature_importance_df.head(20)


def select_2tier_features():
    #'clients_combined_Mean', 'clients_combined_kurtosis', 'clients_combinedci', 'clients_combined_median',
    # 'clients_combined_vh_Mean', 'clients_combined_vhci', 'clients_combined_vh_median',
    # 'clients_combined_dp_Mean', 'clients_combined_dpci', 'clients_combined_dp_median',
    # 'client_nn_Mean', 'client_nnci', 'client_nn_median',
    # 'client_nn_vh_Mean', 'client_nn_vhci', 'client_nn_vh_median',
    # 'client_nn_dp_Mean', 'client_nn_dpci', 'client_nn_dp_median'
    groups =[
        ['Agencia_ID_Demanda_uni_equil_Mean', 'Agencia_ID_Demanda_uni_equilci', 'Agencia_ID_Demanda_uni_equil_median', 'Agencia_ID_Dev_proxima_Mean', 'Agencia_ID_Dev_proximaci', 'Agencia_ID_Venta_hoy_Mean', 'Agencia_ID_Venta_hoyci', 'Agencia_ID_Venta_hoy_median'],
        #['clients_combined_Mean', 'clients_combined_kurtosis', 'clients_combinedci', 'clients_combined_median', 'clients_combined_vh_Mean', 'clients_combined_vhci', 'clients_combined_vh_median', 'clients_combined_dp_Mean', 'clients_combined_dpci'],
        #['client_nn_Mean', 'client_nnci', 'client_nn_median', 'client_nn_vh_Mean', 'client_nn_vhci', 'client_nn_vh_median', 'client_nn_dp_Mean', 'client_nn_dpci'],
        ['client_nn_agency_Mean', 'client_nn_agencyci', 'client_nn_agency_median', 'client_nn_agency_vh_Mean', 'client_nn_agency_vhci', 'client_nn_agency_vh_median', 'client_nn_agency_dp_Mean', 'client_nn_agency_dpci'],
        ['Producto_ID_Demanda_uni_equil_Mean', 'Producto_ID_Demanda_uni_equilci', 'Producto_ID_Demanda_uni_equil_median', 'Producto_ID_Venta_hoy_Mean', 'Producto_ID_Venta_hoyci', 'Producto_ID_Venta_hoy_median', 'Producto_ID_Dev_proxima_Mean', 'Producto_ID_Dev_proximaci'],
        ['weight', 'pieces'],
        ['product_word_Demanda_uni_equil_Mean', 'product_word_Demanda_uni_equilci'],
        ['Town_id_Demanda_uni_equil_Mean', 'Town_id_Demanda_uni_equilci', 'State_id_Demanda_uni_equil_Mean', 'State_id_Demanda_uni_equilci'],
        ['agc_product_Mean', 'agc_productci', 'agc_product_median'],
        ['routes_combined_Mean', 'routes_combinedci', 'routes_combined_median'],
        ["mean_sales", "sales_count", "sales_stddev", "median_sales", "hmean", "ci", "kurtosis"],
        ["last_sale", "last_sale_week", 'Semana'],
        ["entropy", "corr", "mean_autocorr", "mean_corss_points_count"] + ["returns", "signature"]


    ]

    features = []
    for t in list(itertools.combinations(range(len(groups)), 4)):
        #flist = top_group[random.randint(0, len(top_group)-1)] + groups[t[0]] + groups[t[1]] + groups[t[2]] + groups[t[3]]
        #fset = list(set(flist))
        #features.append(fset)
        flist = groups[t[0]] + groups[t[1]] + groups[t[2]] + groups[t[3]]
        features.append(flist)

    np.random.shuffle(features)
    features = features[:10]
    return features



def select_from_all_features():
    '''
    groups = [
        ['Agencia_ID_Demanda_uni_equil_Mean', 'Agencia_ID_Demanda_uni_equilci', 'Agencia_ID_Demanda_uni_equil_median'],
        ['clients_combined_Mean', 'clients_combined_kurtosis', 'clients_combinedci', 'clients_combined_median'],
        ['Producto_ID_Demanda_uni_equil_Mean', 'Producto_ID_Demanda_uni_equilci', 'Producto_ID_Demanda_uni_equil_median', 'Producto_ID_Venta_hoy_Mean', 'Producto_ID_Venta_hoyci', 'Producto_ID_Venta_hoy_median'],
        ['clients_combined_vh_Mean', 'clients_combined_vhci', 'clients_combined_vh_median'],
        ['Producto_ID_Dev_proxima_Mean', 'Producto_ID_Dev_proximaci', 'Producto_ID_Dev_proxima_median'],
        ['weight', 'pieces'],
        ['product_word_Demanda_uni_equil_Mean', 'product_word_Demanda_uni_equilci', 'product_word_Demanda_uni_equil_median'],
        ['mean_sales', 'sales_count', 'sales_stddev', 'median_sales', 'hmean'],
        ['last_sale', 'last_sale_week'],
        ['returns'],
        ['signature'],
        ['kurtosis'],
        ['entropy']
    ]
    '''

    top_group2 = [
        ['clients_combined_dp_Mean', 'clients_combined_dpci', 'clients_combined_dp_median'],
        ['clients_combined_vh_Mean_x', 'clients_combined_vhci_x', 'clients_combined_vh_median_x'],
        ['Agencia_ID_Demanda_uni_equil_Mean', 'Agencia_ID_Demanda_uni_equilci', 'Agencia_ID_Demanda_uni_equil_median'],
        ['Producto_ID_Demanda_uni_equil_Mean', 'Producto_ID_Demanda_uni_equilci', 'Producto_ID_Demanda_uni_equil_median', 'Producto_ID_Venta_hoy_Mean', 'Producto_ID_Venta_hoyci', 'Producto_ID_Venta_hoy_median'],
        ['clients_combined_kurtosis', 'clients_combinedci', 'clients_combined_median', 'clients_combined_Mean']
    ]


    groups2 = [
        ['clients_combined_dp_Mean', 'clients_combined_dpci', 'clients_combined_dp_median'],
        ['clients_combined_vh_Mean_x', 'clients_combined_vhci_x', 'clients_combined_vh_median_x'],
        ['Agencia_ID_Demanda_uni_equil_Mean', 'Agencia_ID_Demanda_uni_equilci', 'Agencia_ID_Demanda_uni_equil_median'],
        ['Producto_ID_Demanda_uni_equilci', 'Producto_ID_Demanda_uni_equil_median', 'Producto_ID_Demanda_uni_equil_Mean'],
        ['clients_combined_kurtosis', 'clients_combinedci', 'clients_combined_median', 'clients_combined_Mean'],

        ['Agencia_ID_Dev_proxima_Mean', 'Agencia_ID_Dev_proximaci', 'Agencia_ID_Dev_proxima_median'],
        ['Agencia_ID_Venta_hoy_Mean', 'Agencia_ID_Venta_hoyci', 'Agencia_ID_Venta_hoy_median'],
        ['Producto_ID_Venta_hoy_Mean', 'Producto_ID_Venta_hoyci', 'Producto_ID_Venta_hoy_median'],
        ['Producto_ID_Dev_proxima_Mean', 'Producto_ID_Dev_proximaci', 'Producto_ID_Dev_proxima_median'],
        ['Canal_ID_Demanda_uni_equil_Mean', 'Canal_ID_Demanda_uni_equilci', 'Canal_ID_Demanda_uni_equil_median'],
        ['Canal_ID_Venta_hoy_Mean', 'Canal_ID_Venta_hoyci', 'Canal_ID_Venta_hoy_median'],
        ['Canal_ID_Dev_proxima_Mean', 'Canal_ID_Dev_proximaci', 'Canal_ID_Dev_proxima_median'],
        ['Ruta_SAK_Demanda_uni_equil_Mean', 'Ruta_SAK_Demanda_uni_equilci', 'Ruta_SAK_Demanda_uni_equil_median'],
        ['Ruta_SAK_Venta_hoy_Mean', 'Ruta_SAK_Venta_hoyci', 'Ruta_SAK_Venta_hoy_median'],
        ['Ruta_SAK_Dev_proxima_Mean', 'Ruta_SAK_Dev_proximaci', 'Ruta_SAK_Dev_proxima_median'],
        ['weight', 'pieces', 'has_choco', 'has_vanilla', 'has_multigrain'],
        ['brand_id_Demanda_uni_equil_Mean', 'brand_id_Demanda_uni_equilci', 'brand_id_Demanda_uni_equil_median'],
        ['product_word_Demanda_uni_equil_Mean', 'product_word_Demanda_uni_equilci', 'product_word_Demanda_uni_equil_median'],
        ['Town_id_Demanda_uni_equil_Mean', 'Town_id_Demanda_uni_equilci', 'Town_id_Demanda_uni_equil_median'],
        ['State_id_Demanda_uni_equil_Mean', 'State_id_Demanda_uni_equilci', 'State_id_Demanda_uni_equil_median'],
        ['agc_product_Mean', 'agc_productci', 'agc_product_median'],
        ['routes_combined_Mean', 'routes_combinedci', 'routes_combined_median'],
        ['clients_route_agc_Mean', 'clients_route_agcci', 'clients_route_agc_median'],
        ["mean_sales", "sales_count", "sales_stddev", "median_sales", "hmean", "ci", "kurtosis"],
        ["last_sale", "last_sale_week", 'Semana'],
        ["returns", "signature"],
        ["entropy", "corr", "mean_autocorr", "mean_corss_points_count"]
    ]

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

    features = []
    for t in list(itertools.combinations(range(len(groups)), 4)):
        #flist = top_group[random.randint(0, len(top_group)-1)] + groups[t[0]] + groups[t[1]] + groups[t[2]] + groups[t[3]]
        #fset = list(set(flist))
        #features.append(fset)
        flist = groups[t[0]] + groups[t[1]] + groups[t[2]] + groups[t[3]]
        features.append(flist)

    np.random.shuffle(features)
    features = features[:200]
    return features


def five_group_stats(group):
    sales = np.array(group['Demanda_uni_equil'].values)
    samana = group['Semana'].values
    max_index = np.argmax(samana)
    returns = group['Dev_proxima'].mean()
    #this is signature on when slaes happens

    sorted_samana_index = np.argsort(samana)
    sorted_sales = sales[sorted_samana_index]

    signature = np.sum([ math.pow(2,s-3) for s in samana])
    kurtosis = fillna_and_inf(scipy.stats.kurtosis(sorted_sales))
    hmean = fillna_and_inf(scipy.stats.hmean(np.where(sales <0, 0.1, sales)))
    entropy = fillna_and_inf(scipy.stats.entropy(sales))
    std = fillna_and_inf(np.std(sales))
    N = len(sales)
    ci = fillna_and_inf(calculate_ci(std, N))
    corr = fillna_and_inf(scipy.stats.pearsonr(range(N), sorted_sales)[0])

    autocorr_list = np.correlate(sorted_sales, sorted_sales, mode='same')
    mean_autocorr = fillna_and_inf(np.mean(autocorr_list))

    mean = np.mean(sales)

    mean_corss_points_count = 0
    if N > 1:
        high_than_mean = mean < sorted_sales[0]
        for i in range(1,N):
            if (high_than_mean and mean > sorted_sales[i]) or (not high_than_mean and mean > sorted_sales[i]):
                mean_corss_points_count += mean_corss_points_count
            high_than_mean = mean < sorted_sales[i]

    return mean, N, std, np.median(sales), sales[max_index], samana[max_index], \
           returns, signature, kurtosis, hmean, entropy, ci, corr, mean_autocorr, mean_corss_points_count


def add_five_grouped_stats(train_df, test_df, testDf):
    start_ts = time.time()

    #we first remove any entry that has only returns
    sales_df = train_df[train_df['Demanda_uni_equil'] > 0]
    grouped = sales_df.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])

    slope_data_df = grouped.apply(five_group_stats)
    sales_data_df = slope_data_df.to_frame("sales_data")
    sales_data_df.reset_index(inplace=True)
    valuesDf = expand_array_feild_and_add_df(sales_data_df, 'sales_data', ["mean_sales", "sales_count", "sales_stddev",
                    "median_sales", "last_sale", "last_sale_week", "returns", "signature", "kurtosis", "hmean", "entropy", "ci", "corr", "mean_autocorr", "mean_corss_points_count"])
    find_NA_rows_percent(valuesDf, "valuesDf base stats")

    #valuesDf = expand_array_feild_and_add_df(sales_data_df, 'sales_data', ["sales_count"])

    #now we merge the data
    sale_data_aggr_time = time.time()

    train_df_m = pd.merge(train_df, valuesDf, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
    default_sales_mean = train_df_m['mean_sales'].mean()
    train_df_m['mean_sales'].fillna(default_sales_mean, inplace=True)
    train_df_m['median_sales'].fillna(default_sales_mean, inplace=True)
    train_df_m['last_sale'].fillna(default_sales_mean, inplace=True)
    train_df_m.fillna(0, inplace=True)

    test_df_m = pd.merge(test_df, valuesDf, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
    test_df_m['mean_sales'].fillna(default_sales_mean, inplace=True)
    test_df_m['median_sales'].fillna(default_sales_mean, inplace=True)
    test_df_m['last_sale'].fillna(default_sales_mean, inplace=True)
    test_df_m.fillna(0, inplace=True)

    if testDf is not None:
        testDf = pd.merge(testDf, valuesDf, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
        testDf['mean_sales'].fillna(default_sales_mean, inplace=True)
        testDf['median_sales'].fillna(default_sales_mean, inplace=True)
        testDf['last_sale'].fillna(default_sales_mean, inplace=True)
        testDf.fillna(0, inplace=True)

    '''
    #these are top aggrigate features
    default_demand_stats = DefaultStats(mean=train_df['Demanda_uni_equil'].mean(), count=train_df['Demanda_uni_equil'].count(),
                                        stddev=train_df['Demanda_uni_equil'].std())
    train_df_m, test_df_m, testDf = addFeildStatsAsFeatures(train_df_m, test_df_m,'Agencia_ID', testDf, default_demand_stats,
                                                        FeatureOps(stddev=True))
    train_df_m, test_df_m, testDf = join_multiple_feild_stats(train_df_m, test_df_m, testDf, ['Ruta_SAK', 'Cliente_ID'],
                'Demanda_uni_equil', "clients_combined", default_demand_stats, FeatureOps(kurtosis=True, stddev=True))
    train_df_m, test_df_m, testDf = addFeildStatsAsFeatures(train_df_m, test_df_m,'Producto_ID', testDf, default_demand_stats,
                                                            FeatureOps(stddev=True))
    '''
    train_data_feilds_to_drop = ['Venta_uni_hoy', 'Venta_hoy', 'Dev_uni_proxima', 'Dev_proxima']
    train_df_m, test_df_m, _ = drop_feilds(train_df_m, test_df_m, None, train_data_feilds_to_drop)

    slopes_time = time.time()

    print "Add Sales Data took %f (%f, %f)" %(slopes_time - start_ts, sale_data_aggr_time-start_ts, slopes_time-sale_data_aggr_time)
    return train_df_m, test_df_m, testDf