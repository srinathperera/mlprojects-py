
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

def load__from_store(model_type, command, name):
    df, metadata = load_file_with_metadata(model_type, command, name)
    rmsle_values = metadata['rmsle']
    bf_index = np.argmin(rmsle_values)
    print "best single model rmsle=", rmsle_values[bf_index]
    y_actual = df['actual'].values.copy()
    df = drop_feilds_1df(df, ['actual'])
    return df, y_actual, bf_index


def do_ensamble(conf, forecasts, best_forecast_index, y_actual):
    vf_start = time.time()

    ensmbales = [
        SimpleAvgEnsamble(conf, "mean"),
        SimpleAvgEnsamble(conf, "median"),
        BestPairEnsamble(conf)
    ]

    for en in ensmbales:
        en.fit(forecasts, best_forecast_index, y_actual)

    vote_forecast = vote_based_forecast(forecasts, best_forecast_index, y_actual)
    calculate_accuracy("vote_forecast "+ str(conf.command), y_actual, vote_forecast)

    vote_with_lr(conf, forecasts, best_forecast_index, y_actual)

    print "vf tooks", (time.time() - vf_start)



    #avg models also save the submission
    #avg_models(conf, models, forecasts, y_actual_test, test_df, submission_forecasts=kaggale_predicted_list, submission_ids=ids, sub_df=testDf)

def run_ensambles(rcommand):
    conf = IDConfigs(target_as_log=True, normalize=True, save_predictions_with_data=True, generate_submission=True)
    conf.command=rcommand
    df, y_actual, best_forecast_index = load__from_store('agr_cat', conf.command, "model_forecasts")
    forcasts = df.values
    do_ensamble(conf, forcasts, best_forecast_index, y_actual)

run_ensambles(command)