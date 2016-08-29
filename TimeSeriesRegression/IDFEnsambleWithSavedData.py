
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



def run_ensambles_on_multiple_models(command):
    conf = IDConfigs(target_as_log=True, normalize=True, save_predictions_with_data=True, generate_submission=True)
    conf.command=-2

    forecasts_with_blend_df, y_actual = load_ensamble_data("model_forecasts")
    sub_with_blend_df, submissions_ids = load_ensamble_data("model_submissions")

    data_feilds = ["mean_sales", "sales_count", "sales_stddev",
                    "median_sales", "last_sale", "last_sale_week", "returns", "signature", "kurtosis", "hmean", "entropy"]
    forecast_feilds = [f for f in list(forecasts_with_blend_df) if "." in f]
    all_feilds = data_feilds+forecast_feilds

    product_data = forecasts_with_blend_df['Producto_ID']
    product_data_submission = sub_with_blend_df['Producto_ID']


    data_size = forecasts_with_blend_df.shape[0]
    fold_size = int(math.ceil(data_size/4))

    forecast_data = forecasts_with_blend_df[all_feilds].values
    train_folds = []
    y_folds = []
    for fs in range(0, data_size, fold_size):
        train_folds.append(forecast_data[fs:min(fs+fold_size, data_size)])
        y_folds.append(y_actual[fs:min(fs+fold_size, data_size)])

    submission_forecasts = []
    for i in range(len(y_folds)):
        train_df = pd.DataFrame(train_folds[i], columns=all_feilds)
        y_data = y_folds[i]
        print "fold data", train_df.shape, y_data.shape
        _, submission_forecast = avg_models(conf, train_df, y_data, sub_with_blend_df, submission_ids=submissions_ids, do_cv=False)
        submission_forecasts.append(submission_forecast)

    all_submission_data = np.column_stack(submission_forecasts)
    avg_submission = np.mean(all_submission_data, axis=1)
    to_save = np.column_stack((submissions_ids, avg_submission))
    to_saveDf =  pd.DataFrame(to_save, columns=["id","Demanda_uni_equil"])
    to_saveDf = to_saveDf.fillna(0)
    to_saveDf["id"] = to_saveDf["id"].astype(int)
    submission_file = 'avg_xgb_ensamble_submission.csv'
    to_saveDf.to_csv(submission_file, index=False)

    print "Best Ensamble Submission Stats", submission_file
    print_mem_usage("after models")

run_ensambles_on_multiple_models(command)