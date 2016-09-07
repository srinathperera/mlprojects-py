
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

if feature_set is None or feature_set == "feature-explore":
    feature_set = "feature-explore"
    features = select_2tier_features()
    ml_models = get_models4xgboost_only(conf)
    load_all_data = True
else:
    core = ['clients_combined_Mean', 'clients_combined_kurtosis', 'clients_combinedci', 'clients_combined_median', 'clients_combined_vh_Mean', 'clients_combined_vhci', 'clients_combined_vh_median', 'clients_combined_dp_Mean', 'clients_combined_dpci', 'clients_combined_dp_median', 'client_nn_Mean', 'client_nnci', 'client_nn_median', 'client_nn_vh_Mean', 'client_nn_vhci', 'client_nn_vh_median', 'client_nn_dp_Mean', 'client_nn_dpci', 'client_nn_dp_median']
    if feature_set == "fg-vhmean-product":
        flist = ['Producto_ID_Demanda_uni_equil_Mean', 'Producto_ID_Demanda_uni_equilci', 'Producto_ID_Demanda_uni_equil_median', 'Producto_ID_Venta_hoy_Mean', 'Producto_ID_Venta_hoyci', 'Producto_ID_Venta_hoy_median', 'clients_combined_vh_Mean', 'clients_combined_vhci', 'clients_combined_vh_median', 'mean_sales', 'sales_count', 'sales_stddev', 'median_sales', 'hmean', 'entropy']
    elif feature_set == "vh-mean-product":
        flist = ['clients_combined_vh_Mean', 'clients_combined_vhci', 'clients_combined_vh_median', 'Producto_ID_Dev_proxima_Mean', 'Producto_ID_Dev_proximaci', 'Producto_ID_Dev_proxima_median', 'weight', 'pieces', 'signature']
    elif feature_set == "nn_features": #0.5737
        flist = ['client_nn_Mean', 'client_nnci', 'client_nn_median']\
            + ['client_nn_vh_Mean', 'client_nn_vhci', 'client_nn_vh_median']\
            + ['client_nn_dp_Mean', 'client_nn_dpci', 'client_nn_dp_median']\
            + ['client_nn_agency_Mean', 'client_nn_agencyci', 'client_nn_agency_median'] \
            + ['client_nn_agency_vh_Mean', 'client_nn_agency_vhci', 'client_nn_agency_vh_median']\
            + ['client_nn_agency_dp_Mean', 'client_nn_agency_dpci', 'client_nn_agency_dp_median']\
            + ['Producto_ID_Demanda_uni_equil_Mean', 'Producto_ID_Demanda_uni_equilci', 'Producto_ID_Demanda_uni_equil_median']\
            + ['Producto_ID_Venta_hoy_Mean', 'Producto_ID_Venta_hoyci', 'Producto_ID_Venta_hoy_median']\
            + ['Producto_ID_Dev_proxima_Mean', 'Producto_ID_Dev_proximaci', 'Producto_ID_Dev_proxima_median']\
            + ["mean_sales", "sales_count", "sales_stddev", "median_sales", "hmean", "ci", "kurtosis"]
    elif feature_set == "nn_features-agency": #0.5737
        flist = core + ['client_nn_agency_Mean', 'client_nn_agencyci', 'client_nn_agency_median', 'client_nn_agency_vh_Mean', 'client_nn_agency_vhci', 'client_nn_agency_vh_median', 'client_nn_agency_dp_Mean', 'client_nn_agency_dpci', 'client_nn_agency_dp_median']
    elif feature_set == "nn_features-product": #0.5737
        flist = core + ['Producto_ID_Demanda_uni_equil_Mean', 'Producto_ID_Demanda_uni_equilci', 'Producto_ID_Demanda_uni_equil_median', 'Producto_ID_Venta_hoy_Mean', 'Producto_ID_Venta_hoyci', 'Producto_ID_Venta_hoy_median', 'Producto_ID_Dev_proxima_Mean', 'Producto_ID_Dev_proximaci', 'Producto_ID_Dev_proxima_median']
    elif feature_set == "nn_features-brand": #0.5737
        flist = core + ['brand_id_Demanda_uni_equil_Mean', 'brand_id_Demanda_uni_equilci', 'brand_id_Demanda_uni_equil_median', 'product_word_Demanda_uni_equil_Mean', 'product_word_Demanda_uni_equilci', 'product_word_Demanda_uni_equil_median'] + ['Town_id_Demanda_uni_equil_Mean', 'Town_id_Demanda_uni_equilci', 'Town_id_Demanda_uni_equil_median', 'State_id_Demanda_uni_equil_Mean', 'State_id_Demanda_uni_equilci', 'State_id_Demanda_uni_equil_median']
    elif feature_set == "features-agc-pp": #0.5737
        flist = core + ['agc_product_Mean', 'agc_productci', 'agc_product_median'] + ['weight', 'pieces', 'has_choco', 'has_vanilla', 'has_multigrain']
    elif feature_set == "features-agency": #0.5737
        flist = core + ['Agencia_ID_Demanda_uni_equil_Mean', 'Agencia_ID_Demanda_uni_equilci', 'Agencia_ID_Demanda_uni_equil_median', 'Agencia_ID_Dev_proxima_Mean', 'Agencia_ID_Dev_proximaci', 'Agencia_ID_Dev_proxima_median', 'Agencia_ID_Venta_hoy_Mean', 'Agencia_ID_Venta_hoyci', 'Agencia_ID_Venta_hoy_median']
    elif feature_set == "cc-cnn-agc": #0.492059235749
        flist = ['clients_combined_Mean', 'clients_combined_kurtosis', 'clients_combinedci', 'clients_combined_median', 'clients_combined_vh_Mean', 'clients_combined_vhci', 'clients_combined_vh_median', 'clients_combined_dp_Mean', 'clients_combined_dpci', 'clients_combined_dp_median', 'client_nn_Mean', 'client_nnci', 'client_nn_median', 'client_nn_vh_Mean', 'client_nn_vhci', 'client_nn_vh_median', 'client_nn_dp_Mean', 'client_nn_dpci', 'client_nn_dp_median', 'agc_product_Mean',
                 'agc_productci', 'agc_product_median', 'weight', 'pieces', 'has_choco', 'has_vanilla', 'has_multigrain']
    elif feature_set == "misc-features": #0.585399746915
        flist = ['weight', 'pieces', 'product_word_Demanda_uni_equil_Mean', 'product_word_Demanda_uni_equilci', 'agc_product_Mean', 'agc_productci', 'agc_product_median', 'routes_combined_Mean', 'routes_combinedci', 'routes_combined_median']
    else:
        raise ValueError("Unknown feature set "+ feature_set)

    features = [flist]
    ml_models = get_models4ensamble(conf)
    load_all_data = False
    #ml_models = get_models4xgboost_only(conf)


analysis_type = feature_set
conf.analysis_type = analysis_type

s_time = time.time()

fg_feilds = ["mean_sales", "sales_count", "sales_stddev",
                        "median_sales", "last_sale", "last_sale_week", "returns", "signature", "kurtosis", "hmean", "entropy", "ci", "corr", "mean_autocorr", "mean_corss_points_count"]

feilds_to_drop = ['Canal_ID','Cliente_ID','Producto_ID', 'Agencia_ID', 'Ruta_SAK']

# we added following data when we write the submission file
blend_features = feilds_to_drop + ['Semana']


if load_all_data:
    #load first dataset
    train_df, test_df, testDf, y_actual_train, y_actual_test = load_train_data('all_features', conf.command, throw_error=True, load_sub_id=True)
    print "reusing train data", analysis_type, "X",train_df.shape, "Y", y_actual_train.shape, "test_df",test_df.shape, "Y test", y_actual_test.shape

    print_mem_usage("after data set 1")
    train_df, test_df, testDf = merge_another_dataset(train_df, test_df, testDf, 'fg_stats', conf.command, fg_feilds)
else:
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

if conf.command == 0:
    size = int(train_df.shape[0]*0.6)
    train_df = train_df[:size]
    y_actual_train = y_actual_train[:size]

print_mem_usage("after data set 2")
print "X",train_df.shape, "Y", y_actual_train.shape

#TODO add or handle if needed
#train_df.fillna(0, inplace=True)
#test_df.fillna(0, inplace=True)
#testDf.fillna(0, inplace=True)

blend_test_data_keys = test_df[blend_features]
blend_submission_data_keys = testDf[blend_features]

#drop five feild
train_df, test_df, testDf = drop_feilds(train_df, test_df, testDf, feilds_to_drop)

train_df, test_df, testDf = replace_inf(train_df, test_df, testDf, 100000)
train_df, test_df, testDf = replace_na_dfs_with_mean(train_df, test_df, testDf)


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
        if not os.path.exists(analysis_type):
            os.makedirs(analysis_type)

        submission_file = analysis_type +'/submission'+str(conf.command)+ '.csv'
        save_submission_file(submission_file, ids, submissions[best_model_index])
        #convert the values to numpy arrays
        forecasts = np.column_stack(forecasts)
        submissions = np.column_stack(submissions)

        #create and save predictions for each model so we can build an ensamble later
        if forecasts.shape[1] > 1:
            model_forecasts_data = np.column_stack([blend_test_data_keys, forecasts, y_actual_test])
            to_saveDf =  pd.DataFrame(model_forecasts_data, columns=blend_features + model_names + ["actual"])
            metadata_map = {'rmsle':model_rmsle}
            save_file(analysis_type, command, to_saveDf, 'model_forecasts', metadata=metadata_map)
            print "## model_forecasts ##"
            print to_saveDf.describe()

            submission_data = np.column_stack([ids, blend_submission_data_keys, submissions])
            to_saveDf =  pd.DataFrame(submission_data, columns=[["id"] + blend_features +model_names])
            save_file(analysis_type, command, to_saveDf, 'model_submissions')
            print "## model_submissions ##"
            print to_saveDf.describe()

    except Exception, error:
        print "An exception was thrown!"
        print str(error)


        #model, parmsFromNormalization, parmsFromNormalization2D, best_forecast = do_forecast(conf, train_df, test_df, y_actual_test)




m_time = time.time()

print_time_took(s_time, "forecasting")
