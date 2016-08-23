
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

import sys


def split_df(s_df, parts):
    l = s_df.shape[0]
    part_size = int(l/parts)
    parts = []
    for i in range(0, l, part_size):
        if l - i > 2*part_size:
            parts.append(s_df[i:i+part_size])
        elif l -i > part_size and l -i < 2*part_size:
            parts.append(s_df[i:l])
        #else do nothing
    return parts


def train_and_find_top10error(conf, traindf1, y_train1,  traindf_check, y_train_check, test_df, y_test):
    models = get_models4xgboost_only(conf)
    print "Forecast with", "traindf1", traindf1.shape, "y_train1", y_train1.shape, "test_df", test_df.shape, "y_test", y_test.shape, "traindf_check", traindf_check.shape

    #conf, train_df, test_df, sub_df, y_train, y_test, models=None
    tmodels, tforecasts, tsubmission_forecasts = do_forecast(conf, traindf1, test_df, traindf_check, y_train1, y_test, models=models)

    rmsle = tmodels[0].rmsle

    errors = np.log(1+y_train_check) - np.log(1+ tsubmission_forecasts[:, 0])
    error90percentile = np.percentile(errors, 90)

    train_df_feilds = list(traindf_check)
    traindf_check_data = np.column_stack([traindf_check.values, errors, y_train_check])
    traindf_check = pd.DataFrame(traindf_check_data, columns=list(traindf_check) + ['error', 'target'])

    train_df_delta = traindf_check[traindf_check['error'] > error90percentile]
    check_y_to_add = train_df_delta['target'].values
    train_df_delta = train_df_delta[train_df_feilds]

    traindf1 = pd.concat([traindf1, train_df_delta])
    y_train1 = np.concatenate([y_train1, check_y_to_add])

    traindf_check_left = traindf_check[traindf_check['error'] <= error90percentile]
    y_train_check_left = traindf_check_left['target']
    traindf_check_left = traindf_check_left[train_df_feilds]
    print "Forecast Return", "traindf1", traindf1.shape, "y_train1", y_train1.shape, "test_df", "traindf_check_left", traindf_check_left.shape, "y_train_check_left", y_train_check_left.shape
    return traindf1, y_train1, traindf_check_left, y_train_check_left, rmsle

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

df, sub_dfo = read_datafiles(command, False)
y_all = df['Demanda_uni_equil']
train_df, test_df, y_actual_train,y_actual_test = prepare_train_and_test_data(df, y_all, split_frac=0.6)

train_df1, train_df_check, y_train1, y_train_check = prepare_train_and_test_data(df, y_all, split_frac=0.2)

train_df_check = drop_feilds_1df(train_df_check, ['Venta_uni_hoy', 'Venta_hoy', 'Dev_uni_proxima', 'Dev_proxima', 'Demanda_uni_equil'])

y_actual_test = y_actual_test.values

accuracy = []

for i in range(5):
    train_df1, test_df, train_df_check, _, _ = generate_features(conf, train_df1, test_df, train_df_check, y_actual_test)
    train_df1, test_df, train_df_check = drop_feilds(train_df1, test_df, train_df_check, ['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID'])
    print "train_df_check",list(train_df_check)
    train_df1, y_train1, train_df_check, y_train_check, rmsle = train_and_find_top10error(conf, train_df1, y_train1,
            train_df_check, y_train_check, test_df, y_actual_test)
    accuracy.append(rmsle)
    print "try ", i ," done", rmsle

print 'train_df.shape', train_df.shape
print "Done accuracy=", accuracy

