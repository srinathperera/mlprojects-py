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
import sys


analysis_type = 'fg_stats'



def five_group_stats(group):
    sales = np.array(group['Demanda_uni_equil'].values)
    samana = group['Semana'].values
    max_index = np.argmax(samana)
    returns = group['Dev_proxima'].mean()
    #this is signature on when slaes happens
    signature = np.sum([ math.pow(2,s-3) for s in samana])
    kurtosis = fillna_and_inf(scipy.stats.kurtosis(sales))
    hmean = fillna_and_inf(scipy.stats.hmean(np.where(sales <0, 0.1, sales)))
    entropy = fillna_and_inf(scipy.stats.entropy(sales))


    return np.mean(sales), len(sales), np.std(sales), np.median(sales), sales[max_index], samana[max_index], \
           returns, signature, kurtosis, hmean, entropy




def add_five_grouped_stats(train_df, test_df, testDf):
    start_ts = time.time()

    #we first remove any entry that has only returns
    sales_df = train_df[train_df['Demanda_uni_equil'] > 0]
    grouped = sales_df.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])

    slope_data_df = grouped.apply(five_group_stats)
    sales_data_df = slope_data_df.to_frame("sales_data")
    sales_data_df.reset_index(inplace=True)
    valuesDf = expand_array_feild_and_add_df(sales_data_df, 'sales_data', ["mean_sales", "sales_count", "sales_stddev",
                    "median_sales", "last_sale", "last_sale_week", "returns", "signature", "kurtosis", "hmean", "entropy"])
    find_NA_rows_percent(valuesDf, "valuesDf base stats")

    #valuesDf = expand_array_feild_and_add_df(sales_data_df, 'sales_data', ["sales_count"])

    #now we merge the data
    sale_data_aggr_time = time.time()
    train_df_m = pd.merge(train_df, valuesDf, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
    train_df_m.fillna(0, inplace=True)
    test_df_m = pd.merge(test_df, valuesDf, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
    test_df_m.fillna(0, inplace=True)

    if testDf is not None:
        testDf = pd.merge(testDf, valuesDf, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
        testDf.fillna(0, inplace=True)

    #these are top aggrigate features
    default_demand_stats = DefaultStats(mean=train_df['Demanda_uni_equil'].mean(), count=train_df['Demanda_uni_equil'].count(),
                                        stddev=train_df['Demanda_uni_equil'].std())
    train_df_m, test_df_m, testDf = addFeildStatsAsFeatures(train_df_m, test_df_m,'Agencia_ID', testDf, default_demand_stats,
                                                        FeatureOps(stddev=True))
    train_df_m, test_df_m, testDf = join_multiple_feild_stats(train_df_m, test_df_m, testDf, ['Ruta_SAK', 'Cliente_ID'],
                'Demanda_uni_equil', "clients_combined", default_demand_stats, FeatureOps(kurtosis=True, stddev=True))
    train_df_m, test_df_m, testDf = addFeildStatsAsFeatures(train_df_m, test_df_m,'Producto_ID', testDf, default_demand_stats,
                                                            FeatureOps(stddev=True))

    train_data_feilds_to_drop = ['Venta_uni_hoy', 'Venta_hoy', 'Dev_uni_proxima', 'Dev_proxima']
    train_df_m, test_df_m, _ = drop_feilds(train_df_m, test_df_m, None, train_data_feilds_to_drop)

    slopes_time = time.time()

    print "Add Sales Data took %f (%f, %f)" %(slopes_time - start_ts, sale_data_aggr_time-start_ts, slopes_time-sale_data_aggr_time)
    return train_df_m, test_df_m, testDf

def check_accuracy_from_model_output(conf, y_pred_raw, y_actual_test, label):
    y_pred_final = modeloutput2predictions(y_pred_raw, parmsFromNormalization=conf.parmsFromNormalization)
    if conf.target_as_log:
        y_pred_final = retransfrom_from_log(y_pred_final)

    error_ac, rmsep, mape, rmse = almost_correct_based_accuracy(y_actual_test, y_pred_final, 10)
    rmsle = calculate_rmsle(y_actual_test, y_pred_final)
    print ">> %s AC_errorRate=%.1f RMSEP=%.6f MAPE=%6f RMSE=%6f rmsle=%.5f" %("Run " + str(conf.command)+ " "
                                                                              + label, error_ac, rmsep, mape, rmse, rmsle)


def do_simple_models(conf, train_df_raw, test_df_raw, subdf_raw, y_actual_train_o, y_actual_test_o):
    train_df, test_df, testDf, y_actual_train, y_actual_test = load_train_data(analysis_type, conf.command)
    if train_df is None or test_df is None or testDf is None:
        train_df, test_df, testDf = add_five_grouped_stats(train_df_raw, test_df_raw, subdf_raw)
        save_train_data(analysis_type, conf.command, train_df, test_df, testDf, y_actual_train_o, y_actual_test_o)
        y_actual_train = y_actual_train_o
        y_actual_test = y_actual_test_o
        print "create and save train data", analysis_type
    else:
        print "reusing train data", analysis_type

    #drop the base feilds from forecasts
    feilds_to_drop =  ['Canal_ID','Cliente_ID','Producto_ID', 'Agencia_ID', 'Ruta_SAK']
    train_df, test_df, _ = drop_feilds(train_df, test_df, None, feilds_to_drop + ['Demanda_uni_equil'] )
    testDf = drop_feilds_1df(testDf, feilds_to_drop)

    mean_forecast = test_df['mean_sales']
    calculate_accuracy("mean_forecast", y_actual_test, mean_forecast)

    median_forecast = test_df['median_sales']
    calculate_accuracy("median_forecast", y_actual_test, median_forecast)

    if conf.target_as_log:
        train_df, test_df, testDf = tranform_train_data_to_log(train_df, test_df, testDf, skip_field_patterns=['kurtosis', 'id'])
        y_train_raw, y_test_raw = transfrom_to_log(y_actual_train), transfrom_to_log(y_actual_test)
    else:
        y_train_raw, y_test_raw = y_actual_train, y_actual_test


    #conf, train_df, test_df, y_train_raw, y_test_raw, y_actual_test
    models, forecasts, test_df, parmsFromNormalization, parmsFromNormalization2D = do_forecast(conf, train_df, test_df,
                                                                                               y_train_raw, y_test_raw, y_actual_test)

    #TODO save submission




def test_simple_model(command):
    conf = IDConfigs(target_as_log=True, normalize=True, save_predictions_with_data=True, generate_submission=True)
    conf.command = command

    df, subdf = read_datafiles(command, test_run=False)

    training_set_size = int(0.6*df.shape[0])
    test_set_size = df.shape[0] - training_set_size

    y_all = df['Demanda_uni_equil'].values

    #if conf.target_as_log and not conf.log_target_only:
    #then all values are done as logs
    #    df['Demanda_uni_equil'] = transfrom_to_log(df['Demanda_uni_equil'].values)

    train_df = df[:training_set_size]
    test_df = df[-1*test_set_size:]

    y_actual_train = y_all[:training_set_size]
    y_actual_test = y_all[-1*test_set_size:]

    do_simple_models(conf, train_df, test_df, subdf, y_actual_train, y_actual_test)



print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

command = -2
if len(sys.argv) > 1:
    command = int(sys.argv[1])

test_simple_model(command)