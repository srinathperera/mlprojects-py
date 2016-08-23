
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import time
from sklearn.externals import joblib

from mltools import undoPreprocessing
import itertools



from tsforecasttools import run_timeseries_froecasts, regression_with_xgboost
from mltools import preprocess2DtoZeroMeanUnit, preprocess1DtoZeroMeanUnit, train_test_split, print_feature_importance, apply_zeroMeanUnit2D
from mltools import calculate_rmsle, almost_correct_based_accuracy, MLConfigs, print_regression_model_summary, \
    regression_with_dl, apply_zeroMeanUnit, undo_zeroMeanUnit2D
from keras.optimizers import Adam
from sklearn.linear_model import LinearRegression

from inventory_demand import *
from mltools import *
from inventory_demand_ensambles import *

from os import listdir
from os.path import isfile, join

import sys
print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)


command = -2
if len(sys.argv) > 1:
    command = int(sys.argv[1])


def load__single_file(file):
    df = pd.read_csv(file)
    y_actual = df['actual'].values.copy()
    df = drop_feilds_1df(df, ['actual'])
    best_forecast_index = 0
    return df, y_actual, best_forecast_index


def merge_a_df(base_df, model_name, file_name, command, feilds_to_use=None):
    new_df = load_file(model_name, command, file_name)
    print "merging ", model_name, " feilds=", str(list(new_df)) + " with base fields=", list(base_df)
    merge_feilds = ['Semana', 'Agencia_ID' , 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID']
    if feilds_to_use is not None:
        new_df = new_df[merge_feilds+feilds_to_use]
    print "merging ", model_name, " feilds=", list(new_df)
    full_df = pd.merge(base_df, new_df, how='left', on=merge_feilds)
    return full_df


def load__model_results_from_store(model_names_list, name, use_agr_features=True):
    first_dir = model_names_list[0]
    files_list = [f for f in listdir(first_dir) if str(f).startswith(name) and
                    str(f).endswith(".csv")]
    print name, "found files", files_list

    df_list = []
    for f in files_list:
        base_df = pd.read_csv(first_dir+'/'+str(f))
        cmd = extract_regx('[0-9]+', f)
        if name == "model_forecasts":
            fname = "test"
            addtional_data_feild = 'actual'
        elif name == "model_submissions":
            fname = "sub"
            addtional_data_feild = 'id'
        else:
            raise ValueError("")

        for i in range(1,len(model_names_list)):
            model_name = model_names_list[i]
            feilds_to_use = None
            file_name = name
            if model_name == "fg_stats":
                feilds_to_use = ["mean_sales", "sales_count", "sales_stddev",
                    "median_sales", "last_sale", "last_sale_week", "returns", "signature", "kurtosis", "hmean", "entropy"]
                file_name = fname
            base_df = merge_a_df(base_df, model_name, file_name, cmd, feilds_to_use)

        feilds_to_remove = [f for f in list(base_df) if str(f).startswith(addtional_data_feild)]
        addtional_data_feild = feilds_to_remove[0]
        feilds_to_remove = feilds_to_remove[1:]

        merge_feilds = ['Semana', 'Agencia_ID' , 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID']
        base_df = drop_feilds_1df(base_df, merge_feilds + feilds_to_remove)
        print "remove feilds", feilds_to_remove
        df_list.append(base_df)

    if len(df_list) > 0:
        final_df = pd.concat(df_list)

        #this is criticial as otherwise few stores will depend all in testing data
        final_df = final_df.sample(frac=1)

         #if we do not copy, we get errors due to shafflinf
        #find_NA_rows_percent(final_df, "loading data for ensamble")
        final_df.fillna(0, inplace=True)

        print "final feilds", list(final_df)
        addtional_data = final_df[addtional_data_feild].values.copy()
        final_df = drop_feilds_1df(final_df, [addtional_data_feild])

        print "load files", model_names_list, name, "shape=", final_df.shape, "with features", list(final_df)
        return final_df, addtional_data
    else:
        return None


def load__from_store(model_type, name, use_agr_features=True):
    dir = model_type
    files_list = [f for f in listdir(dir) if str(f).startswith(name) and
                    str(f).endswith(".csv")]
    print name, "found files", files_list

    df_list = []
    for f in files_list:
        base_df = pd.read_csv(dir+'/'+str(f))
        cmd = extract_regx('[0-9]+', f)
        if name == "model_forecasts":
            fname = "test"
            addtional_data_feild = 'actual'
        elif name == "model_submissions":
            fname = "sub"
            addtional_data_feild = 'id'

        merge_feilds = ['Semana', 'Agencia_ID' , 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID']
        if use_agr_features:
            #then features are in order, we just concatinate
            blend_df = load_file('agr_cat',cmd, fname)
            feilds_to_use = ["Agencia_ID_Demanda_uni_equil_Mean", "Agencia_ID_Demanda_uni_equilci" , "Agencia_ID_Demanda_uni_equil_median", "clients_combined_Mean",
                             "clients_combinedci", "clients_combined_median", "Producto_ID_Demanda_uni_equil_Mean" , "Producto_ID_Demanda_uni_equilci" ,
                             "Producto_ID_Demanda_uni_equil_median", "Producto_ID_Venta_hoy_Mean" ,"Producto_ID_Venta_hoyci" , "Producto_ID_Dev_proxima_Mean", "Producto_ID_Dev_proximaci" ,
                             "Producto_ID_Dev_proxima_median"]
            blend_df = blend_df[feilds_to_use]

            print list(base_df)
            base_df = drop_feilds_1df(base_df, merge_feilds)
            if base_df.shape[0] != blend_df.shape[0]:
                raise ValueError("two data frame sizes does not match "+ str(base_df.shape) + " " + str(blend_df))
            feature_df = pd.concat([base_df, blend_df], axis=1)
        else:
            blend_df = load_file('fg_stats',cmd, fname)
            feilds_to_use = ["mean_sales", "sales_count", "sales_stddev",
                    "median_sales", "last_sale", "last_sale_week", "returns", "signature", "kurtosis", "hmean", "entropy"]
            feilds_to_use = feilds_to_use + merge_feilds
            blend_df = blend_df[feilds_to_use]
            feature_df = pd.merge(base_df, blend_df, how='left', on=merge_feilds)
            feature_df = drop_feilds_1df(feature_df, merge_feilds)

        df_list.append(feature_df)

    if len(df_list) > 0:
        final_df = pd.concat(df_list)

        #this is criticial as otherwise few stores will depend all in testing data
        final_df = final_df.sample(frac=1)

         #if we do not copy, we get errors due to shafflinf
        find_NA_rows_percent(final_df, "loading data for ensamble")
        final_df.fillna(0, inplace=True)

        addtional_data = final_df[addtional_data_feild].values.copy()
        final_df = drop_feilds_1df(final_df, [addtional_data_feild])

        print "load file", model_type, name, "shape=", final_df.shape, "with features", list(feature_df)
        return final_df, addtional_data
    else:
        return None



def do_ensamble(conf, forecasts, best_forecast_index, y_actual, submissions_ids, submissions):
    vf_start = time.time()

    ensmbales = [
        SimpleAvgEnsamble(conf, "mean"),
        SimpleAvgEnsamble(conf, "median"),
        BestPairEnsamble(conf),
        BestThreeEnsamble(conf)
    ]

    for en in ensmbales:
        en.fit(forecasts, best_forecast_index, y_actual)

    #vote_forecast = vote_based_forecast(forecasts, best_forecast_index, y_actual)
    #calculate_accuracy("vote_forecast "+ str(conf.command), y_actual, vote_forecast)

    #vote_with_lr(conf, forecasts, best_forecast_index, y_actual)

    print "vf tooks", (time.time() - vf_start)

    if submissions_ids is not None and submissions is not None:
        best_pair_ensamble_forecasts = ensmbales[2].predict(submissions, best_forecast_index)
        save_submission_file("best_pair_submission.csv", submissions_ids, best_pair_ensamble_forecasts)

        best_pair_ensamble_forecasts = ensmbales[3].predict(submissions, best_forecast_index)
        save_submission_file("best_triple_submission.csv", submissions_ids, best_pair_ensamble_forecasts)
    else:
        print "submissions not found"

    #avg models also save the submission
    #avg_models(conf, models, forecasts, y_actual_test, test_df, submission_forecasts=kaggale_predicted_list, submission_ids=ids, sub_df=testDf)


def find_best_forecast(forecasts, y_actual):
    start = time.time()
    forecasts_rmsle = []
    for i in range(forecasts.shape[1]):
        rmsle = calculate_accuracy("vote_forecast "+ str(i), y_actual, forecasts[:, i])
        forecasts_rmsle.append(rmsle)
        print "[full data] forecast "+str(i)+" rmsle=", rmsle, " stats\n"

    model_index_by_acc = np.argsort(forecasts_rmsle)
    print "model index order", model_index_by_acc
    best_findex = model_index_by_acc[0]
    print "best single model forecast is", best_findex, "rmsle=", forecasts_rmsle[best_findex]
    print_time_took(start, "find_best_forecast")

    print "best single model forecast stats\n", basic_stats_as_str(forecasts[best_findex])
    print "y actual\n", basic_stats_as_str(y_actual)

    return model_index_by_acc



def run_ensambles(rcommand):
    conf = IDConfigs(target_as_log=True, normalize=True, save_predictions_with_data=True, generate_submission=True)
    conf.command=-2

    #load forecast data
    forecasts_df, y_actual = load__from_store('agr_cat', "model_forecasts")

    #not using LR forecasts
    print list(forecasts_df)

    forecasts = forecasts_df[['XGB', 'RFR','ETR']].values


    #load submission data
    subdf, submissions_ids = load__from_store('agr_cat', "model_submissions")
    submissions = subdf[['XGB', 'RFR','ETR']].values

    model_index_by_acc = find_best_forecast(forecasts, y_actual)

    best_forecast_index = model_index_by_acc[0]
    print_mem_usage("before simple ensamble")
    do_ensamble(conf, forecasts, best_forecast_index, y_actual, submissions_ids ,submissions)
    #blend_models(conf, forecasts, model_index_by_acc, y_actual, submissions_ids, submissions,
    #             blend_data, blend_data_submission)
    print_mem_usage("avg models")
    avg_models(conf, forecasts_df, y_actual, subdf, submission_ids=None)


def run_ensambles_on_multiple_models(command):
    conf = IDConfigs(target_as_log=True, normalize=True, save_predictions_with_data=True, generate_submission=True)
    conf.command=-2

    model_list = ['agr_cat', 'fg-vhmean-product', 'fg_stats', 'nn_features-product', 'nn_features-agency']

    #load forecast data
    forecasts_df, y_actual = load__model_results_from_store(model_list, "model_forecasts")

    #not using LR forecasts
    print "Using feilds", list(forecasts_df)

    feilds_to_keep = [f for f in list(forecasts_df) if str(f).startswith("XGB") or str(f).startswith("RFR") or str(f).startswith("ETR")]
    forecasts = forecasts_df[feilds_to_keep].values


    #load submission data
    subdf, submissions_ids = load__model_results_from_store(model_list, "model_submissions")
    submissions = subdf[feilds_to_keep].values

    model_index_by_acc = find_best_forecast(forecasts, y_actual)

    best_forecast_index = model_index_by_acc[0]
    print_mem_usage("before simple ensamble")
    do_ensamble(conf, forecasts, best_forecast_index, y_actual, submissions_ids ,submissions)
    #blend_models(conf, forecasts, model_index_by_acc, y_actual, submissions_ids, submissions,
    #             blend_data, blend_data_submission)
    print_mem_usage("avg models")
    avg_models(conf, forecasts_df, y_actual, subdf, submission_ids=submissions_ids)
    avg_models_with_ml(conf, forecasts_df, y_actual, subdf, submission_ids=submissions_ids)

run_ensambles_on_multiple_models(command)