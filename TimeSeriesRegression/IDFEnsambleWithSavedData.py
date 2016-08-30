
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
import gc
print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)


command = -2
if len(sys.argv) > 1:
    command = int(sys.argv[1])



def load_ensamble_data(name):
    model_type = "all_ensamble"
    command = 0
    train_df = load_file(model_type, command, name, throw_error=True)
    ytrain_df = load_file(model_type, command, 'y_'+name, throw_error=True)
    y_train = ytrain_df['target'].values

    return train_df, y_train


def per_product_forecast():
    #per_product_forecast, per_product_forecast_submission = find_best_forecast_per_product(train_fold1, y_actual_fold1, sub_with_blend_df,
    #                                                                                product_data, product_data_submission, submissions_ids)
    #forecasts_with_blend_df['ppf'] = per_product_forecast
    #sub_with_blend_df['ppf'] = per_product_forecast_submission
    #avg_models(conf, forecasts_with_blend_df, y_actual, sub_with_blend_df, submission_ids=submissions_ids, frac=0.5)
    print "hello"

def log_centrality_forecasts(conf, forecasts_data, y_actual):
    non_log_mean_forecast = np.mean(forecasts_data, axis=1)
    calculate_accuracy("non_log_mean", y_actual, non_log_mean_forecast)

    forecasts_data = transfrom_to_log2d(forecasts_data)
    mean_forecast = retransfrom_from_log(np.mean(forecasts_data, axis=1))
    calculate_accuracy("log_mean", y_actual, mean_forecast)

    median_forecast = retransfrom_from_log(np.median(forecasts_data, axis=1))
    calculate_accuracy("log_median", y_actual, median_forecast)

    #mean_forecast = retransfrom_from_log(np.mean(forecasts_data, axis=1))
    #calculate_accuracy("log_mean", y_actual, mean_forecast)

def best_pair_forecast(conf, forecasts_data, y_actual, submission_data, submissions_ids):
    best_pair_ensmbale = BestPairLogEnsamble(conf)
    best_pair_ensmbale_forecast = best_pair_ensmbale.fit(forecasts_data, y_actual)
    best_pair_ensamble_submission = best_pair_ensmbale.predict(submission_data)
    save_submission_file("best_pair_submission.csv", submissions_ids, best_pair_ensamble_submission)



def xgb_k_ensamble(conf, all_feilds, forecasts_with_blend_df, y_actual, sub_with_blend_df, submissions_ids):
    data_size = forecasts_with_blend_df.shape[0]
    fold_size = int(math.ceil(data_size/4))

    forecast_data = forecasts_with_blend_df[all_feilds].values
    train_folds = []
    y_folds = []
    for fs in range(0, data_size, fold_size):
        train_folds.append(forecast_data[fs:min(fs+fold_size, data_size)])
        y_folds.append(y_actual[fs:min(fs+fold_size, data_size)])

    submission_forecasts = []
    xgb_forecasts = []
    y_actuals = []

    for i in range(len(y_folds)):
        train_df = pd.DataFrame(train_folds[i], columns=all_feilds)
        y_data = y_folds[i]
        print "fold data", train_df.shape, y_data.shape
        try:
            xgb_forecast, y_actual_test, submission_forecast = avg_models(conf, train_df, y_data, sub_with_blend_df, submission_ids=submissions_ids, do_cv=True)
            submission_forecasts.append(submission_forecast)
            xgb_forecasts.append(xgb_forecast)
            y_actuals.append(y_actual_test)
        except Exception, error:
            print "An exception was thrown!"
            print str(error)

    calculate_accuracy("overall avg forecast", np.concatenate(y_actuals), np.concatenate(xgb_forecasts))


    all_submission_data = np.column_stack(submission_forecasts)
    avg_submission = np.mean(all_submission_data, axis=1)

    avg_submission = np.where(avg_submission < 0, 0, avg_submission)
    to_save = np.column_stack((submissions_ids, avg_submission))
    to_saveDf =  pd.DataFrame(to_save, columns=["id","Demanda_uni_equil"])
    to_saveDf = to_saveDf.fillna(0)
    to_saveDf["id"] = to_saveDf["id"].astype(int)
    submission_file = 'avg_xgb_ensamble_submission.csv'
    to_saveDf.to_csv(submission_file, index=False)

    print "Best Ensamble Submission Stats", submission_file


def run_ensambles_on_multiple_models(command):
    conf = IDConfigs(target_as_log=True, normalize=True, save_predictions_with_data=True, generate_submission=True)
    conf.command=-2

    forecasts_with_blend_df, y_actual = load_ensamble_data("model_forecasts")
    sub_with_blend_df, submissions_ids = load_ensamble_data("model_submissions")

    data_feilds = ["mean_sales", "sales_count", "sales_stddev",
                    "median_sales", "last_sale", "last_sale_week", "returns", "signature", "kurtosis", "hmean", "entropy"]
    forecast_feilds = [f for f in list(forecasts_with_blend_df) if "." in f]
    all_feilds = data_feilds+forecast_feilds
    print_mem_usage("before models")

    gc.collect()

    #product_data = forecasts_with_blend_df['Producto_ID']
    #product_data_submission = sub_with_blend_df['Producto_ID']

    #xgb_k_ensamble(conf, all_feilds, forecasts_with_blend_df, y_actual, sub_with_blend_df, submissions_ids)
    #xgb_forecast_feilds = [f for f in list(forecasts_with_blend_df) if ".XGB" in f]
    #log_centrality_forecasts(conf, forecasts_with_blend_df[forecast_feilds].values, y_actual)

    forecast_feilds_data_only = forecasts_with_blend_df[forecast_feilds].values
    best_pair_forecast(conf, forecast_feilds_data_only, y_actual, sub_with_blend_df[forecast_feilds].values, submissions_ids)

    predict_using_veriation(forecast_feilds_data_only, forecasts_with_blend_df['cc-cnn-agc.XGB'].values, y_actual)
    print_mem_usage("after models")

run_ensambles_on_multiple_models(command)