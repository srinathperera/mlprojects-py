
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
print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)


command = -2
if len(sys.argv) > 1:
    command = int(sys.argv[1])
if len(sys.argv) > 2:
    test_run = (int(sys.argv[2]) == 1)
else:
    test_run = False

analysis_type = 'agr_cat'

conf = IDConfigs(target_as_log=True, normalize=True, save_predictions_with_data=True, generate_submission=True)
conf.command = command
conf.analysis_type = analysis_type

s_time = time.time()

#load first dataset
train_df, test_df, testDf, y_actual_train, y_actual_test = load_train_data(analysis_type, conf.command, throw_error=True)
print "reusing train data", analysis_type

print "X",train_df.shape, "Y", y_actual_train.shape

#load second dataset
train_df, test_df, testDf = merge_another_dataset(train_df, test_df, testDf, 'fg_stats', conf.command,
    ["median_sales", "returns", "signature", "kurtosis"])

print "X",train_df.shape, "Y", y_actual_train.shape

train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)
testDf.fillna(0, inplace=True)

#"mean_sales", "sales_count", "sales_stddev", "hmean", "entropy"

#drop five feild
feilds_to_drop = ['Canal_ID','Cliente_ID','Producto_ID', 'Agencia_ID', 'Ruta_SAK']
train_df, test_df, testDf = drop_feilds(train_df, test_df, testDf, feilds_to_drop)

if conf.target_as_log:
    train_df, test_df, testDf = tranform_train_data_to_log(train_df, test_df, testDf, skip_field_patterns=['kurtosis', 'id'])
    y_train_raw, y_test_raw = transfrom_to_log(y_actual_train), transfrom_to_log(y_actual_test)
else:
    y_train_raw, y_test_raw = y_actual_train, y_actual_test

testDf.fillna(0, inplace=True)
ids = testDf['id']
testDf.drop('id',axis=1, inplace=True)
sub_X_all = testDf.values


ml_models = get_models4ensamble(conf)

forecasts = []
submissions = []
model_names = []
model_rmsle = []

for m in ml_models:
    #model, parmsFromNormalization, parmsFromNormalization2D, best_forecast = do_forecast(conf, train_df, test_df, y_actual_test)
    print_mem_usage("before running model " + m.name)
    tmlist, tforecasts, _, parmsFromNormalization, parmsFromNormalization2D = do_forecast(conf, train_df, test_df,
                                                                                           y_train_raw, y_test_raw, y_actual_test, models=[m])
    forecasts.append(tforecasts[:, 0])
    t_model = tmlist[0]
    y_forecast_submission = create_submission(conf, t_model, ids, sub_X_all, parmsFromNormalization, parmsFromNormalization2D)
    submissions.append(y_forecast_submission)
    model_names.append(t_model.name)
    model_rmsle.append(t_model.rmsle)
    t_model.cleanup()
    gc.collect() #to free up memory
    print_mem_usage("after model " + t_model.name)

best_model_index = np.argmin(model_rmsle)
print "[IDF"+str(conf.command)+"]Best Single Model has rmsle=", model_rmsle[best_model_index]
submission_file = 'submission'+str(conf.command)+ '.csv'
save_submission_file(submission_file, ids, submissions[best_model_index])

#convert the values to numpy arrays
forecasts = np.column_stack(forecasts)
submissions = np.column_stack(submissions)

#create and save predictions for each model so we can build an ensamble later
if forecasts.shape[1] > 1:
    blend_features = get_blend_features()
    blend_data_test = test_df[blend_features].values

    print "shapes", blend_data_test.shape, forecasts.shape, y_actual_test.shape
    model_forecasts_data = np.column_stack([blend_data_test, forecasts, y_actual_test])
    to_saveDf =  pd.DataFrame(model_forecasts_data, columns=blend_features + model_names + ["actual"])
    metadata_map = {'rmsle':model_rmsle}
    save_file(analysis_type, command, to_saveDf, 'model_forecasts', metadata=metadata_map)
    print "## model_forecasts ##"
    print to_saveDf.describe()

    blend_data_submission = testDf[blend_features].values
    #ids, kaggale_predicted_list = create_per_model_submission(conf, models, testDf, parmsFromNormalization, parmsFromNormalization2D )

    submission_data = np.column_stack([ids, blend_data_submission, submissions])
    to_saveDf =  pd.DataFrame(submission_data, columns=[["id"] + blend_features +model_names])
    save_file(analysis_type, command, to_saveDf, 'model_submissions')
    print "## model_submissions ##"
    print to_saveDf.describe()


m_time = time.time()

print_time_took(s_time, "forecasting")
