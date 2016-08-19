
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
    feature_set = sys.argv[2]
else:
    feature_set = None

conf = IDConfigs(target_as_log=True, normalize=True, save_predictions_with_data=True, generate_submission=True)
conf.command = command


if feature_set == "fg-vhmean-product":
    list = ['Producto_ID_Demanda_uni_equil_Mean', 'Producto_ID_Demanda_uni_equilci', 'Producto_ID_Demanda_uni_equil_median', 'Producto_ID_Venta_hoy_Mean', 'Producto_ID_Venta_hoyci', 'Producto_ID_Venta_hoy_median', 'clients_combined_vh_Mean', 'clients_combined_vhci', 'clients_combined_vh_median', 'mean_sales', 'sales_count', 'sales_stddev', 'median_sales', 'hmean', 'entropy']
elif feature_set == "vh-mean-product":
    list = ['clients_combined_vh_Mean', 'clients_combined_vhci', 'clients_combined_vh_median', 'Producto_ID_Dev_proxima_Mean', 'Producto_ID_Dev_proximaci', 'Producto_ID_Dev_proxima_median', 'weight', 'pieces', 'signature']
else:
    raise ValueError("Unknown feature set "+ feature_set)

features = [list]
ml_models = get_models4ensamble(conf)

analysis_type = feature_set
conf.analysis_type = analysis_type

s_time = time.time()

#load first dataset
train_df, test_df, testDf, y_actual_train, y_actual_test = load_train_data('all_features', conf.command, throw_error=True)
print "reusing train data", analysis_type

print "X",train_df.shape, "Y", y_actual_train.shape, "test_df",test_df.shape, "Y test", y_actual_test.shape

#load second dataset
#train_df, test_df, testDf = merge_another_dataset(train_df, test_df, testDf, 'fg_stats', conf.command,
#    ["median_sales", "returns", "signature", "kurtosis"])

train_df, test_df, testDf = merge_another_dataset(train_df, test_df, testDf, 'fg_stats', conf.command, ["mean_sales", "sales_count", "sales_stddev",
                    "median_sales", "last_sale", "last_sale_week", "returns", "signature", "kurtosis", "hmean", "entropy"])


print "X",train_df.shape, "Y", y_actual_train.shape

train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)
testDf.fillna(0, inplace=True)

feilds_to_drop = ['Canal_ID','Cliente_ID','Producto_ID', 'Agencia_ID', 'Ruta_SAK']

# we added following data when we write the submission file
blend_features = feilds_to_drop + ['Semana']
blend_test_data_keys = test_df[blend_features]
blend_submission_data_keys = testDf[blend_features]

#drop five feild
train_df, test_df, testDf = drop_feilds(train_df, test_df, testDf, feilds_to_drop)


testDf.fillna(0, inplace=True)
ids = testDf['id']
testDf.drop('id',axis=1, inplace=True)

print "train_df", train_df.shape, "test_df", test_df.shape
verify_forecasting_data(train_df.values, y_actual_train, test_df.values, y_actual_test)

for fset in features:
    forecasts = []
    submissions = []
    model_names = []
    model_rmsle = []

    try:
        for m in ml_models:
            print "using ", fset, "from", list(train_df)
            train_dft = train_df[fset].copy(); test_dft = test_df[fset].copy(); testDft = testDf[fset].copy()
            print_mem_usage("before running model " + m.name)
            tmodels, tforecasts, tsubmission_forecasts = do_forecast(conf, train_dft, test_dft, testDft, y_actual_train, y_actual_test, models=[m])
            forecasts.append(tforecasts[:, 0])
            t_model = tmodels[0]
            submissions.append(tsubmission_forecasts[:, 0])
            model_names.append(t_model.name)
            model_rmsle.append(t_model.rmsle)
            t_model.cleanup()
            gc.collect() #to free up memory
            print_mem_usage("after model " + t_model.name)
            print "[IDF"+str(conf.command)+"]", fset, t_model.name, t_model.rmsle

        best_model_index = np.argmin(model_rmsle)
        print "[IDF"+str(conf.command)+"]Best Single Model has rmsle=", model_rmsle[best_model_index]
        submission_file = 'submission'+str(conf.command)+ '.csv'
        save_submission_file(submission_file, ids, submissions[best_model_index])
        #convert the values to numpy arrays
        forecasts = np.column_stack(forecasts)
        submissions = np.column_stack(submissions)

        #create and save predictions for each model so we can build an ensamble later
        if forecasts.shape[1] > 1:
            model_forecasts_data = np.column_stack([blend_test_data_keys, forecasts, y_actual_test])
            to_saveDf =  pd.DataFrame(model_forecasts_data, columns=blend_features + model_names + ["actual"])
            metadata_map = {'rmsle':model_rmsle}
            save_file(analysis_type, command, to_saveDf, 'model_forecasts', metadata=metadata_map)
            print "## model_forecasts ##"
            print to_saveDf.describe()

            submission_data = np.column_stack([ids, blend_submission_data_keys, submissions])
            to_saveDf =  pd.DataFrame(submission_data, columns=[["id"] + blend_features +model_names])
            save_file(analysis_type, command, to_saveDf, 'model_submissions')
            print "## model_submissions ##"
            print to_saveDf.describe()

    except Exception, error:
        print "An exception was thrown!"
        print str(error)


        #model, parmsFromNormalization, parmsFromNormalization2D, best_forecast = do_forecast(conf, train_df, test_df, y_actual_test)




m_time = time.time()

print_time_took(s_time, "forecasting")
