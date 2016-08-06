
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


def load__from_store(model_type, name):
    dir = model_type
    files_list = [f for f in listdir(dir) if str(f).startswith(name) and
                    str(f).endswith(".csv")]
    print name, "found files", files_list
    df_list = [pd.read_csv(dir+'/'+str(f)) for f in files_list]
    if len(df_list) > 0:
        final_df = pd.concat(df_list)
        return final_df
    else:
        return None

    #df, metadata = load_file_with_metadata(model_type, command, name)
    #rmsle_values = metadata['rmsle']
    #bf_index = np.argmin(rmsle_values)
    #print "best single model rmsle=", rmsle_values[bf_index]
    #y_actual = df['actual'].values.copy()
    #df = drop_feilds_1df(df, ['actual'])
    #return df, y_actual, bf_index



def do_ensamble(conf, forecasts, best_forecast_index, y_actual, submissions_ids, submissions):
    vf_start = time.time()

    ensmbales = [
        SimpleAvgEnsamble(conf, "mean"),
        SimpleAvgEnsamble(conf, "median"),
        BestPairEnsamble(conf)
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
        print "forecast "+str(i)+" rmsle="+ rmsle + " stats\n", pd.Series(forecasts[i]).describe()

    best_findex = np.argmin(forecasts_rmsle)
    print "best single model forecast is", best_findex, "rmsle=", forecasts_rmsle[best_findex]
    print_time_took(start, "find_best_forecast")

    print "best single model forecast stats\n", pd.Series(forecasts[best_findex]).describe()
    print "y actual\n", pd.Series(y_actual).describe()

    return best_findex



def run_ensambles(rcommand):
    conf = IDConfigs(target_as_log=True, normalize=True, save_predictions_with_data=True, generate_submission=True)
    conf.command=-2

    forecasts_df = load__from_store('agr_cat1', "model_forecasts")
    y_actual = forecasts_df['actual'].values.copy()
    forecasts_df = drop_feilds_1df(forecasts_df, ['actual'])
    forecasts = forecasts_df.values

    subdf = load__from_store('agr_cat', "model_submissions")
    if subdf is not None:
        submissions_ids = subdf['id']
        subdf = drop_feilds_1df(subdf, ['id'])
        submissions = subdf.values
    else:
        submissions_ids = None
        submissions =None
    best_forecast_index = find_best_forecast(forecasts, y_actual)
    do_ensamble(conf, forecasts, best_forecast_index, y_actual, submissions_ids ,submissions)

run_ensambles(command)