
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

use_preprocessed_file = False
save_preprocessed_file = False
target_as_log = True
preprocessed_file_name = "_data.csv"
save_predictions_with_data = False
verify_sub_data = False

s_time = time.time()

np.set_printoptions(precision=1, suppress=True)

df, testDf = read_datafiles(command, test_run)


y_actual_2nd_verification = None
if 'Demanda_uni_equil' in testDf:
    #then this is a datafile passed as submission file
    y_actual_2nd_verification = testDf['Demanda_uni_equil']
    testDf = testDf[['Semana','Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID']]
    testDf['id'] = range(testDf.shape[0])



r_time = time.time()

print "read took %f" %(r_time-s_time)

conf = IDConfigs(target_as_log=True, normalize=True, save_predictions_with_data=True, generate_submission=True)
conf.command = command
conf.analysis_type = analysis_type


#df = remove_rare_categories(df)

df = df[df['Producto_ID'] > 0]

df = df.sample(frac=1)


#df['unit_prize'] = df['Venta_hoy']/df['Venta_uni_hoy']

find_NA_rows_percent(df, "data set stats")


training_set_size = int(0.6*df.shape[0])
test_set_size = df.shape[0] - training_set_size

y_all = df['Demanda_uni_equil'].values


train_df = df[:training_set_size]
test_df = df[-1*test_set_size:]

y_actual_train = y_all[:training_set_size]
y_actual_test = y_all[-1*test_set_size:]


#train_df = train_df[train_df['Demanda_uni_equil'] > 0] # to remove
if verify_sub_data:
    testDf = train_df.copy()[['Semana','Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID']]
    testDf['id'] = range(testDf.shape[0])


#print "train", train_df['Semana'].unique(), train_df.shape,"test", test_df['Semana'].unique(), test_df.shape
print_mem_usage("before features")
#objgraph.show_growth()
#if test_run:
#    do_simple_models(conf, train_df,test_df, testDf, y_actual_test)

train_df, test_df, testDf, y_actual_test, test_df_before_dropping_features = generate_features(conf, train_df,
                                                                                               test_df, testDf, y_actual_test)

print "after features", test_df['Semana'].unique(), test_df.shape
print "after features bd", test_df_before_dropping_features['Semana'].unique(), test_df_before_dropping_features.shape

prep_time = time.time()

if test_run:
    print train_df.describe()



print_mem_usage("before forecast")
#print "Memory: train, test, sub",  object_size(train_df), object_size(test_df), object_size(testDf)
#objgraph.show_most_common_types()
#objgraph.show_growth()

if conf.target_as_log:
    train_df, test_df, testDf = tranform_train_data_to_log(train_df, test_df, testDf, skip_field_patterns=['kurtosis', 'id'])
    y_train_raw, y_test_raw = transfrom_to_log(y_actual_train), transfrom_to_log(y_actual_test)
else:
    y_train_raw, y_test_raw = y_actual_train, y_actual_test

#model, parmsFromNormalization, parmsFromNormalization2D, best_forecast = do_forecast(conf, train_df, test_df, y_actual_test)
models, forecasts, test_df, parmsFromNormalization, parmsFromNormalization2D = do_forecast(conf, train_df, test_df,
                                                                                           y_train_raw, y_test_raw, y_actual_test)

print_mem_usage("after forecast")

best_model_index = np.argmin([m.rmsle for m in models])
best_model = models[best_model_index]
print "[IDF"+str(conf.command)+"]Best Single Model has rmsle=", best_model.rmsle

print best_model_index,
best_forecast = forecasts[:,best_model_index]

#save submission based on best model
if conf.generate_submission:
    y_forecast_submission = create_submission(conf, best_model, testDf, parmsFromNormalization, parmsFromNormalization2D)
    if y_actual_2nd_verification is not None:
        rmsle = calculate_rmsle(y_actual_2nd_verification, y_forecast_submission)
        print "2nd Verification rmsle=", rmsle

if verify_sub_data:
    print "train/sub shapes", train_df.shape, testDf.shape
    print train_df.head(10)['clients_combined_Mean']
    print testDf.head(10)['clients_combined_Mean']

    for f in list(test_df):
        if not np.allclose(train_df[f], testDf[f], equal_nan=True):
            print "#### Does not match", f

#create and save predictions for each model so we can build an ensamble later
if forecasts.shape[1] > 1:
    blend_features = get_blend_features()
    blend_data_test = test_df[blend_features].values


    model_names = [m.name for m in models]
        #first we will save all individual model results for reuse
    model_rmsle = [m.rmsle for m in models]
    model_forecasts_data = np.column_stack([blend_data_test, forecasts, y_actual_test])
    to_saveDf =  pd.DataFrame(model_forecasts_data, columns=blend_features + model_names + ["actual"])
    metadata_map = {'rmsle':model_rmsle}
    save_file(analysis_type, command, to_saveDf, 'model_forecasts', metadata=metadata_map)
    print "## model_forecasts ##"
    print to_saveDf.describe()

    blend_data_submission = testDf[blend_features].values
    ids, kaggale_predicted_list = create_per_model_submission(conf, models, testDf, parmsFromNormalization, parmsFromNormalization2D )
    submission_data = np.column_stack([ids, blend_data_submission, kaggale_predicted_list])
    to_saveDf =  pd.DataFrame(submission_data, columns=[["id"] + blend_features +model_names])
    save_file(analysis_type, command, to_saveDf, 'model_submissions')
    print "## model_submissions ##"
    print to_saveDf.describe()


m_time = time.time()

if conf.save_predictions_with_data:
    print "Sizes", test_df_before_dropping_features.shape, best_forecast.shape
    test_df_before_dropping_features['predictions'] = best_forecast
    test_df_before_dropping_features['actual'] = y_actual_test
    test_df_before_dropping_features.to_csv('prediction_with_data'+str(conf.command)+'.csv', index=False)
    #do_error_analysis(test_df_before_dropping_features, conf.command, df)


#print "top aggrigate count", len(slopeMap)
print "total=%f, read=%fs, preporcess=%fs, model=%fs" \
      %(m_time-s_time, (r_time-s_time), (prep_time-r_time), (m_time-prep_time))


