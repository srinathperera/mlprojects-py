
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


def compare_submission_files(file_list, name_list):
    submissions_df_list = [read_submission_file(f) for f in file_list]


    basedf = submissions_df_list[0]
    basedf.rename(columns={"Demanda_uni_equil": name_list[0]}, inplace=True)
    for i in range(1, len(submissions_df_list)):
        mdf = submissions_df_list[i]
        mdf.rename(columns={"Demanda_uni_equil": name_list[i]}, inplace=True)
        basedf = pd.merge(basedf, mdf, how='left', on=["id"])

    submission_ids = basedf['id']
    basedf = drop_feilds_1df(basedf, ['id'])

    feature_list = list(basedf)
    for f in feature_list:
        data = basedf[f].values
        print f,"data.shape", data.shape
        print f, basic_stats_as_str(data), np.histogram(data)

    for i in range(1,len(feature_list)):
        calculate_accuracy(feature_list[i], basedf[feature_list[0]].values, basedf[feature_list[i]].values)
        print_error_distribution(basedf[feature_list[0]].values, basedf[feature_list[i]].values)



print compare_submission_files([
    'submissions_parts/submission-0.46.csv',
    'submissions_parts/avg_xgb_ensamble_submission.csv',
    'submissions_parts/xgb_ensamble_submission_1472490059.13.csv',
    'submissions_parts/best_pair_submission.csv',
    'submissions_parts/mean_log_ensamble_forecast.csv',
    'submissions_parts/avg_xgb_ensamble_submission-d3.csv'],
    ["best", "avg_xgb", "part", "best_pair", "mean log", "d3"]
)
