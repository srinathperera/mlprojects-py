
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
from inventory_demand_features import generate_all_features

from inventory_demand_ensambles import *

import sys
print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

analysis_type = 'agr_cat'

command = -2
if len(sys.argv) > 1:
    command = int(sys.argv[1])
if len(sys.argv) > 2:
    analysis_type = sys.argv[2]
    test_run = False
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

if analysis_type == 'agr_cat':
    train_df, test_df, testDf, y_actual_test, test_df_before_dropping_features = generate_features(conf, train_df,
                                                                                               test_df, testDf, y_actual_test)
elif analysis_type == 'all_features':
    train_df, test_df, testDf, y_actual_test, test_df_before_dropping_features = generate_all_features(conf, train_df,
                                                                                               test_df, testDf, y_actual_test)
else:
    raise ValueError("Unknown analysis type" + analysis_type)


print "after features", test_df['Semana'].unique(), test_df.shape
print "after features bd", test_df_before_dropping_features['Semana'].unique(), test_df_before_dropping_features.shape

prep_time = time.time()

if test_run:
    print train_df.describe()
print_mem_usage("end of feture generation")

save_train_data(analysis_type, conf.command, train_df, test_df, testDf, y_actual_train, y_actual_test)


#print "Memory: train, test, sub",  object_size(train_df), object_size(test_df), object_size(testDf)
#objgraph.show_most_common_types()
#objgraph.show_growth()



