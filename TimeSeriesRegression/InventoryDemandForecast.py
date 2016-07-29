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
from InventoryDemandSimplePredictions import do_simple_models
#from InventoryDemandErrorAnalysis import do_error_analysis

import sys
print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

command = -2
if len(sys.argv) > 1:
    command = int(sys.argv[1])
if len(sys.argv) > 2:
    test_run = int(sys.argv[1]) == 1
else:
    test_run = False

use_preprocessed_file = False
save_preprocessed_file = False
target_as_log = True
preprocessed_file_name = "_data.csv"
save_predictions_with_data = False
verify_sub_data = False

s_time = time.time()

np.set_printoptions(precision=1, suppress=True)

data_files = [
    ["trainitems0_5000.csv", 0, 5000, "test_0_5000.csv"], #1.4G
    ["trainitems5_10_35_40_45_50k.csv", 5000, 10000, "test_5_10_35_40_45_50k.csv"], #534M
    ["trainitems30000_35000.csv", 30000, 35000, "test_30000_35000.csv"], #559M
    #["trainitems30000_35000.csv", 30000, 35000, "trainitems5_10_35_40_45_50k.csv"], #559M # to remove ** pass #1 as #2 test
    ["trainitems40000_45000.csv", 40000, 45000, "test_40000_45000.csv"], #640M
    ["trainitems5000_15000.csv", -1, -1, "test_40000_45000.csv"],
    ["train-rsample-10k.csv", -1, -1, "test_40000_45000.csv"]
]



if command == -2:
    df = read_train_file('data/train.csv')
    testDf = pd.read_csv('data/test.csv')
elif command == -1:
    df = read_train_file('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems300.csv')
    testDf = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/test.csv')
    testDf = testDf[(testDf['Producto_ID'] <= 300)]
else:
    dir = None
    if test_run:
        dir = "/Users/srinath/playground/data-science/BimboInventoryDemand/"
    else:
        dir = "data/"

    df = read_train_file(dir + data_files[command][0])
    testDf = pd.read_csv(dir +data_files[command][3])
    print "using ", dir + data_files[command][0], " and ", dir +data_files[command][3]
    print "testDf read", testDf.shape


y_actual_2nd_verification = None
if 'Demanda_uni_equil' in testDf:
    #then this is a datafile passed as submission file
    y_actual_2nd_verification = testDf['Demanda_uni_equil']
    testDf = testDf[['Semana','Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID']]
    testDf['id'] = range(testDf.shape[0])



r_time = time.time()

print "read took %f" %(r_time-s_time)

conf = IDConfigs(target_as_log=True, normalize=True, save_predictions_with_data=True, generate_submission=True)
conf.log_target_only = False
conf.command = command


#df = remove_rare_categories(df)

df = df[df['Producto_ID'] > 0]

#df['unit_prize'] = df['Venta_hoy']/df['Venta_uni_hoy']

find_NA_rows_percent(df, "data set stats")


training_set_size = int(0.7*df.shape[0])
test_set_size = df.shape[0] - training_set_size

y_all = df['Demanda_uni_equil'].values

if conf.target_as_log and not conf.log_target_only:
    #then all values are done as logs
    df['Demanda_uni_equil'] = transfrom_to_log(df['Demanda_uni_equil'].values)

train_df = df[:training_set_size]
test_df = df[-1*test_set_size:]

y_actual_train = y_all[:training_set_size]
y_actual_test = y_all[-1*test_set_size:]


#train_df = train_df[train_df['Demanda_uni_equil'] > 0] # to remove
if verify_sub_data:
    testDf = train_df.copy()[['Semana','Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID']]
    testDf['id'] = range(testDf.shape[0])


print "train", train_df['Semana'].unique(), train_df.shape,"test", test_df['Semana'].unique(), test_df.shape

#if test_run:
#    do_simple_models(conf, train_df,test_df, testDf, y_actual_test)

train_df, test_df, testDf, y_actual_test, test_df_before_dropping_features = generate_features(conf, train_df,
                                                                                               test_df, testDf, y_actual_test)

print "after features", test_df['Semana'].unique(), test_df.shape
print "after features bd", test_df_before_dropping_features['Semana'].unique(), test_df_before_dropping_features.shape

prep_time = time.time()

#model, parmsFromNormalization, parmsFromNormalization2D, best_forecast = do_forecast(conf, train_df, test_df, y_actual_test)
models, forecasts, test_df, parmsFromNormalization, parmsFromNormalization2D = do_forecast(conf, train_df, test_df, y_actual_test)


best_model_index = np.argmin([m.rmsle for m in models])
best_model = models[best_model_index]
print "[IDF]Best Single Model has rmsle=", best_model.rmsle
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

#create and save submission based on ensamble
if len(forecasts) > 1:
    ids, kaggale_predicted_list = create_per_model_submission(conf, models, testDf, parmsFromNormalization, parmsFromNormalization2D )
    #avg models also save the submission
    avg_models(conf, models, forecasts, y_actual_test, test_df, submission_forecasts=kaggale_predicted_list, submission_ids=ids, sub_df=testDf)


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

