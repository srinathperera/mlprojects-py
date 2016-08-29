
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


def load_saved_ensamble_data(file_name):
    data_df = load_file("all_ensamble", 0, file_name)

    if file_name == "model_forecasts":
        addtional_data_feild = 'actual'
    elif file_name == "model_submissions":
        addtional_data_feild = 'id'

    additional_feild_data = data_df[addtional_data_feild].values
    return data_df, additional_feild_data

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

    forecasts_with_blend_df, y_actual = load_saved_ensamble_data("model_forecasts")
    sub_with_blend_df, submissions_ids = load_saved_ensamble_data("model_submissions")

    forecast_feilds = [f for f in list(forecasts_with_blend_df) if "." in f]

    product_data = forecasts_with_blend_df['Producto_ID']
    product_data_submission = sub_with_blend_df['Producto_ID']


    kf = KFold(data_df.shape[0], n_folds=fold_count)
    y_all = data_df[y_feild].values

    folds =[]
    for train_index, test_index in kf:
        train_df, test_df = data_df.ix[train_index], data_df.ix[test_index]
        y_train, y_test = y_all[train_index], y_all[test_index]
        folds.append((train_df, test_df, y_train,y_test))

    return folds, y_all



    data_size = forecasts_with_blend_df.shape[0]
    fold_size = math.ceil(data_size/4)

    forecast_data = forecasts_with_blend_df.values
    train_folds = []
    y_folds = []
    for fs in range(0, data_size, fold_size):
        train_folds.append(forecast_data[fs:min(fs+fold_size, data_size)])
        y_folds.append(y_actual[fs:min(fs+fold_size, data_size)])

    for i in range(len(y_folds)):
        train_df = pd.DataFrame(train_folds[i], columns=)
        avg_models(conf, forecasts_with_blend_df, y_actual, sub_with_blend_df, submission_ids=submissions_ids)



    print_mem_usage("after models")
    #mean_ensmbale_forecast, mean_top4_submission, best_pair_ensmbale_forecast, best_pair_ensamble_submission = \
    #    do_ensamble(conf, forecasts_only_df, top_forecast_feilds, y_actual, submissions_ids ,submissions_only_df)

    #to_saveDf =  pd.DataFrame(to_save, columns=["id","Demanda_uni_equil"])

    '''
    new_column_list = list(forecasts_with_blend_df) + ['mean_forecast', 'best_pair_forecast']
    forecasts_with_blend_df = pd.DataFrame(np.column_stack([forecasts_with_blend_df.values, mean_ensmbale_forecast, best_pair_ensmbale_forecast]),
                                                           columns=new_column_list)
    sub_with_blend_df = pd.DataFrame(np.column_stack([sub_with_blend_df.values, mean_top4_submission, best_pair_ensamble_submission]),
                                                           columns=new_column_list)

    avg_models(conf, forecasts_with_blend_df, y_actual, sub_with_blend_df, submission_ids=submissions_ids)
    '''

    #blend_models(conf, forecasts, model_index_by_acc, y_actual, submissions_ids, submissions,
    #             blend_data, blend_data_submission)
    #print_mem_usage("avg models")

    #avg_models_with_ml(conf, forecasts_df, y_actual, subdf, submission_ids=submissions_ids)

run_ensambles_on_multiple_models(command)