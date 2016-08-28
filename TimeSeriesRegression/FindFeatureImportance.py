
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import time
from sklearn.externals import joblib

from mltools import undoPreprocessing
import itertools
import gc


from tsforecasttools import run_timeseries_froecasts, regression_with_xgboost
from mltools import preprocess2DtoZeroMeanUnit, preprocess1DtoZeroMeanUnit, train_test_split, print_feature_importance, apply_zeroMeanUnit2D
from mltools import calculate_rmsle, almost_correct_based_accuracy, MLConfigs, print_regression_model_summary, \
    regression_with_dl, apply_zeroMeanUnit, undo_zeroMeanUnit2D
from keras.optimizers import Adam
from sklearn.linear_model import LinearRegression

from inventory_demand import *
from mltools import *
#from mlpreprocessing import feather2df

from inventory_demand_ensambles import *
from inventory_demand_features import *
import os

import sys
print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

command = -2

if len(sys.argv) > 1:
    command = int(sys.argv[1])
if len(sys.argv) > 2:
    feature_set = sys.argv[2]
else:
    feature_set = None


def expand_features(base_df, feild_name):
    data = base_df[feild_name].values
    base_df[feild_name+"_inv"] = fillna_and_inf(1/data)
    base_df[feild_name+"_sq"] = fillna_and_inf(data*data)
    return base_df

conf = IDConfigs(target_as_log=True, normalize=True, save_predictions_with_data=True, generate_submission=True)
conf.command = command

singles_features = ['Semana', 'clients_combined_dp_Mean', 'clients_combined_vh_Mean', 'Agencia_ID_Demanda_uni_equil_Mean',
                    'Producto_ID_Demanda_uni_equil_Mean', 'clients_combined_Mean', 'Agencia_ID_Dev_proxima_Mean', 'Agencia_ID_Venta_hoy_Mean',
                    'Producto_ID_Venta_hoy_Mean', 'Producto_ID_Dev_proxima_Mean', 'Canal_ID_Demanda_uni_equil_Mean', 'Canal_ID_Venta_hoy_Mean',
                    'Canal_ID_Dev_proxima_Mean', 'Ruta_SAK_Demanda_uni_equil_Mean' 'Ruta_SAK_Venta_hoy_Mean', 'Ruta_SAK_Dev_proxima_Mean', 'weight', 'pieces',
                    'has_choco', 'has_vanilla', 'has_multigrain', 'brand_id_Demanda_uni_equil_Mean', 'product_word_Demanda_uni_equil_Mean', 'Town_id_Demanda_uni_equil_Mean',
                    'State_id_Demanda_uni_equil_Mean', 'agc_product_Mean', 'routes_combined_Mean', 'clients_route_agc_Mean', 'mean_sales', 'last_sale', 'returns'
                    ]

features = [[f] for f in singles_features ]


fg_feilds = ["mean_sales", "sales_count", "sales_stddev",
                        "median_sales", "last_sale", "last_sale_week", "returns", "signature", "kurtosis", "hmean", "entropy", "ci", "corr", "mean_autocorr", "mean_corss_points_count"]

#lets load all the data
train_df, test_df, testDf, y_actual_train, y_actual_test = load_train_data('all_features', conf.command, throw_error=True, load_sub_id=True)
train_df, test_df, testDf = merge_another_dataset(train_df, test_df, testDf, 'fg_stats', conf.command, fg_feilds)

print_mem_usage("after data set 2")
print "X",train_df.shape, "Y", y_actual_train.shape


train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)
testDf.fillna(0, inplace=True)

ml_models = get_models4xgboost_only(conf)

ids = testDf['id']
testDf.drop('id',axis=1, inplace=True)

print "train_df", train_df.shape, "test_df", test_df.shape
verify_forecasting_data(train_df.values, y_actual_train, test_df.values, y_actual_test)

for i, fset in enumerate(features):
    forecasts = []
    submissions = []
    model_names = []
    model_rmsle = []

    try:
        for m in ml_models:
            print "using ", fset, "from", list(train_df)
            train_dft = train_df[fset].copy(); test_dft = test_df[fset].copy(); testDft = testDf[fset].copy()
            train_dft=expand_features(train_dft, fset[0])
            test_dft=expand_features(test_dft, fset[0])
            testDft=expand_features(testDft, fset[0])

            print_mem_usage("before running model " + m.name)
            tmodels, tforecasts, tsubmission_forecasts = do_forecast(conf, train_dft, test_dft, testDft, y_actual_train, y_actual_test, models=[m])
            forecasts.append(tforecasts[:, 0])
            t_model = tmodels[0]
            submissions.append(tsubmission_forecasts[:, 0])
            model_names.append(t_model.name)
            model_rmsle.append(t_model.rmsle)
            t_model.cleanup()
            gc.collect() #to free up memory
            print_mem_usage("after model " + t_model.name)
            print "[IDF"+str(conf.command)+"] Feild Feature Importance=", fset, t_model.name, t_model.rmsle

    except Exception, error:
        print "An exception was thrown!"
        print str(error)


        #model, parmsFromNormalization, parmsFromNormalization2D, best_forecast = do_forecast(conf, train_df, test_df, y_actual_test)

