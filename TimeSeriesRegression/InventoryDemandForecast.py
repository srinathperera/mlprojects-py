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

import sys
print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

command = -2
if len(sys.argv) > 1:
    command = int(sys.argv[1])


test_run = False
use_preprocessed_file = False
save_preprocessed_file = False
target_as_log = True
preprocessed_file_name = "_data.csv"
save_predictions_with_data = False

s_time = time.time()

np.set_printoptions(precision=1, suppress=True)


data_files = [
    ["trainitems0_5000.csv", 0, 5000, "test_0_5000.csv"], #1.4G
    ["trainitems5000_10000.csv", 5000, 10000, "test_5000_10000.csv"], #76M
    ["trainitems30000_35000.csv", 30000, 35000, "test_30000_35000.csv"], #559N
    ["trainitems35000_40000.csv", 35000, 40000, "test_35000_40000.csv"], #336M
    ["trainitems40000_45000.csv", 40000, 45000, "test_40000_45000.csv"], #640M
    ["trainitems45000_50000.csv", 450000, 50000, "test_45000_50000.csv"], #123M
]



y_actual = None
if command == -2:
    df = pd.read_csv('data/train.csv')
    testDf = pd.read_csv('data/test.csv')
elif command == -1:
    df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems300.csv')
    testDf = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/test.csv')
    testDf = testDf[(testDf['Producto_ID'] <= 300)]
else:
    dir = None
    if test_run:
        dir = "/Users/srinath/playground/data-science/BimboInventoryDemand/"
    else:
        dir = "data/"

    df = pd.read_csv(dir + data_files[command][0])
    testDf = pd.read_csv(dir +data_files[command][3])
    print "testDf read", testDf.shape


r_time = time.time()

print "read took %f" %(r_time-s_time)

conf = IDConfigs(target_as_log=True, normalize=True, save_predictions_with_data=True, generate_submission=True)
conf.command = command

train_df, test_df, testDf, y_actual_test = generate_features(conf, df, testDf)

prep_time = time.time()


model, parmsFromNormalization, parmsFromNormalization2D = do_forecast(conf, train_df, test_df, y_actual_test)

#if save_predictions_with_data:
#    test_df_before_dropping_features['predictions'] = y_pred_final
#    test_df_before_dropping_features['actual'] = y_actual_test
#    test_df_before_dropping_features.to_csv('forecast_with_data.csv', index=False)

#model = None


    #rmsle = None
    #if target_as_log:
    #    mean_rmsle = calculate_rmsle(y_actual_test, retransfrom_from_log(test_df["groupedMeans"]))
    #    median_rmsle = calculate_rmsle(y_actual_test, retransfrom_from_log(test_df["groupedMedian"]))
    #else:
    #    mean_rmsle = calculate_rmsle(y_actual_test, test_df["groupedMeans"])
    #    median_rmsle = calculate_rmsle(y_actual_test, test_df["groupedMedian"])
    #print "rmsle for mean prediction ", rmsle


m_time = time.time()

#print "top aggrigate count", len(slopeMap)
print "total=%f, read=%fs, preporcess=%fs, model=%fs" \
      %(m_time-s_time, (r_time-s_time), (prep_time-r_time), (m_time-prep_time))

