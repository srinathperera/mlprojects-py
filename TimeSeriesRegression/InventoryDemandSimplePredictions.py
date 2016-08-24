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

from inventory_demand_features import *

from inventory_demand import *
import sys


analysis_type = 'fg_stats'





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

    testDf.fillna(0, inplace=True)
    ids = testDf['id']
    testDf.drop('id',axis=1, inplace=True)

    #tmodels, tforecasts, tsubmission_forecasts = do_forecast(conf, train_df, test_df, testDf, y_actual_train, y_actual_test)

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