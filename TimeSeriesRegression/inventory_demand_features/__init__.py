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
