
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

run_ensambles(command)