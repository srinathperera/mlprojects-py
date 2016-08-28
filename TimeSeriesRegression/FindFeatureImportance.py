
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

conf = IDConfigs(target_as_log=True, normalize=True, save_predictions_with_data=True, generate_submission=True)
conf.command = command


features = ['clients_combined_dp_Mean', 'clients_combined_vh_Mean']
#, Agencia_ID_Demanda_uni_equil_Mean, Producto_ID_Demanda_uni_equil_Mean, clients_combined_Mean
#Agencia_ID_Dev_proxima_Mean, Agencia_ID_Venta_hoy_Mean, Producto_ID_Venta_hoy_Mean, Producto_ID_Dev_proxima_Mean, Canal_ID_Demanda_uni_equil_Mean
#Canal_ID_Venta_hoy_Mean, Canal_ID_Dev_proxima_Mean, Ruta_SAK_Demanda_uni_equil_Mean, Ruta_SAK_Venta_hoy_Mean, Ruta_SAK_Dev_proxima_Mean, weight', 'pieces', 'has_choco', 'has_vanilla', 'has_multigrain'
#brand_id_Demanda_uni_equil_Mean, product_word_Demanda_uni_equil_Mean, Town_id_Demanda_uni_equil_Mean
#State_id_Demanda_uni_equil_Mean, agc_product_Mean, routes_combined_Mean, clients_route_agc_Mean, mean_sales, last_sale, returns


    all_feilds = features[0]
    feilds_in_first_ds = [f for f in all_feilds if f not in fg_feilds]
    feilds_in_second_ds = [f for f in all_feilds if f not in feilds_in_first_ds]
    #feilds_in_first_ds = list(set(all_feilds) - set(fg_feilds))
    #feilds_in_second_ds = list(set(all_feilds) - set(feilds_in_first_ds))

    print "feilds_in_first_ds", feilds_in_first_ds
    print "feilds_in_second_ds", feilds_in_second_ds

    if len(feilds_in_first_ds) + len(feilds_in_second_ds) != len(all_feilds):
        raise ValueError(feilds_in_first_ds, "+", feilds_in_second_ds, "!=", all_feilds)

    #following are join features
    feilds_in_first_ds = feilds_in_first_ds + blend_features

    train_df, test_df, testDf, y_actual_train, y_actual_test = load_train_data('all_features', conf.command, throw_error=True, fields=feilds_in_first_ds, load_sub_id=True)
    print "reusing train data", analysis_type, "X",train_df.shape, "Y", y_actual_train.shape, "test_df",test_df.shape, "Y test", y_actual_test.shape

    print_mem_usage("after data set 1")

    if len(feilds_in_second_ds) > 0:
        train_df, test_df, testDf = merge_another_dataset(train_df, test_df, testDf, 'fg_stats', conf.command, feilds_in_second_ds)


print_mem_usage("after data set 2")
print "X",train_df.shape, "Y", y_actual_train.shape

train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)
testDf.fillna(0, inplace=True)


#drop five feild
train_df, test_df, testDf = drop_feilds(train_df, test_df, testDf, feilds_to_drop)


testDf.fillna(0, inplace=True)
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
            print "[IDF"+str(conf.command)+"]", fset, t_model.name, t_model.rmsle

        best_model_index = np.argmin(model_rmsle)
        print str(i)+"[IDF"+str(conf.command)+"]Best Single Model has rmsle=", model_rmsle[best_model_index]

    except Exception, error:
        print "An exception was thrown!"
        print str(error)


        #model, parmsFromNormalization, parmsFromNormalization2D, best_forecast = do_forecast(conf, train_df, test_df, y_actual_test)




m_time = time.time()

print_time_took(s_time, "forecasting")
