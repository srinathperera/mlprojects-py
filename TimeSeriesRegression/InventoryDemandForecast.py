import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import time
from sklearn.externals import joblib

from mltools import undoPreprocessing


from tsforecasttools import run_timeseries_froecasts, regression_with_xgboost
from mltools import preprocess2DtoZeroMeanUnit, preprocess1DtoZeroMeanUnit, train_test_split, print_feature_importance, apply_zeroMeanUnit2D
from mltools import calculate_rmsle, almost_correct_based_accuracy, MLConfigs, print_regression_model_summary, \
    regression_with_dl, apply_zeroMeanUnit, undo_zeroMeanUnit2D
from keras.optimizers import Adam

#from mlpreprocessing import feather2df

import sys
print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

def addFeildStatsAsFeatures(train_df, test_df, feild_name, testDf, drop=False, default_mean=None, default_stddev=None):
    start = time.time()
    groupData = train_df.groupby([feild_name])['Demanda_uni_equil']
    meanData = groupData.mean()
    stddevData = groupData.std()

    valuesDf = meanData.to_frame(feild_name+"_Mean")
    valuesDf.reset_index(inplace=True)
    valuesDf[feild_name+"_StdDev"] = stddevData.values


    calculate_ts = time.time()
    train_df_m = pd.merge(train_df, valuesDf, how='left', on=[feild_name])
    test_df_m = pd.merge(test_df, valuesDf, how='left', on=[feild_name])

    if drop:
        train_df_m = train_df_m.drop(feild_name,1)
        test_df_m = test_df_m.drop(feild_name,1)

    if testDf is not None:
        testDf = pd.merge(testDf, valuesDf, how='left', on=[feild_name])
        if default_mean is not None:
            testDf[feild_name+"_Mean"].fillna(default_mean, inplace=True)
        if default_stddev is not None:
            testDf[feild_name+"_StdDev"].fillna(default_stddev, inplace=True)
        if drop:
            testDf = testDf.drop(feild_name,1)
    print "took %f (%f, %f), size %s %f" %(time.time()-start, calculate_ts- start,
                                           time.time() - calculate_ts, feild_name, len(meanData))

    return train_df_m, test_df_m, testDf

def avgdiff(group):
    group = group.values
    if len(group) > 1:
        return np.mean([group[i] - group[i-1] for i in range(1, len(group))])
    else:
        return 0

#TODO fix defaults

def modeloutput2predictions(model_forecast, parmsFromNormalization, negative_allowed=False):
    y_pred_corrected = undoPreprocessing(model_forecast, parmsFromNormalization)
    if not negative_allowed:
        y_pred_corrected = np.where(y_pred_corrected < 0, 1, y_pred_corrected)
    return y_pred_corrected


def addSlopes(train_df, test_df, testDf):
    start_ts = time.time()
    #TODO do all operations in one go (see Pre)

    #calculate average slope
    grouped = train_df.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])['Demanda_uni_equil']

    slopeMap = grouped.apply(avgdiff)
    groupedMeanMap = grouped.mean()
    groupedStddevMap = grouped.std()

    valuesDf = slopeMap.to_frame("Slopes")
    valuesDf.reset_index(inplace=True)
    valuesDf["groupedMeans"] = groupedMeanMap.values
    valuesDf["groupedStd"] = groupedStddevMap.values

    slopes_aggr_time = time.time()
    train_df_m = pd.merge(train_df, valuesDf, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
    train_df_m.fillna(0, inplace=True)

    test_df_m = pd.merge(test_df, valuesDf, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
    test_df_m.fillna(0, inplace=True)

    if testDf is not None:
        testDf = pd.merge(testDf, valuesDf, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
        testDf.fillna(0, inplace=True)

    slopes_time = time.time()
    print "Slopes took %f (%f, %f)" %(slopes_time - start_ts, slopes_aggr_time-start_ts, slopes_time-slopes_aggr_time)
    return train_df_m, test_df_m, testDf


def run_rfr(X_train, Y_train, X_test, y_test):
    rfr = RandomForestRegressor()
    rfr.fit(X_train, Y_train)

    print_feature_importance(rfr.feature_importances_)

    #save model
    joblib.dump(rfr, 'model.pkl')
    return rfr


def run_dl(X_train, y_train, X_test, y_test):
    c = MLConfigs(nodes_in_layer=20, number_of_hidden_layers=3, dropout=0.0, activation_fn='relu', loss="mse",
             epoch_count=50, optimizer=Adam(lr=0.0001), regularization=0.001)
    model, y_pred_dl = regression_with_dl(X_train, y_train, X_test, y_test, c)
    return model


def run_xgboost(X_train, Y_train, X_test, y_actual):
    model, y_pred = regression_with_xgboost(X_train, Y_train, X_test)
    #undo the normalization

    return model

def drop_column(df1, df2, feild_name):
    df1 = df1.drop(feild_name,1)
    df2 = df2.drop(feild_name,1)
    return df1, df2


def transfrom_to_log(data):
    return np.log(data + 1)

def retransfrom_from_log(data):
    return np.exp(data) - 1


test_run = False
use_preprocessed_file = False
save_preprocessed_file = False
target_as_log = True
preprocessed_file_name = "_data.csv"

s_time = time.time()

y_actual = None
if not use_preprocessed_file:
    if test_run:
        df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems300.csv')
        testDf = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/test.csv')
        testDf = testDf[(testDf['Producto_ID'] <= 300)]
        print "testDf read", testDf.shape
    else:
        df = pd.read_csv('data/train.csv')
        testDf = pd.read_csv('data/test.csv')
else:
    if test_run:
        df = pd.read_csv(preprocessed_file_name)
        y_actual = df['Demanda_uni_equil'].values
        df = df.drop('Demanda_uni_equil',1)

        testDf = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/test.csv')
        testDf = testDf[(testDf['Producto_ID'] <= 300)]
    else:
        df = pd.read_csv(preprocessed_file_name)
        y_actual = df['Demanda_uni_equil'].values

        df = df.drop('Demanda_uni_equil',1)
        testDf = pd.read_csv('data/test.csv')


r_time = time.time()

print "read took %f" %(r_time-s_time)

y_actual_train = None
y_actual_test = None



if not use_preprocessed_file:
    #print "shapes train, test", df.shape, testDf.shape
    print "shapes train, test", df.shape

    training_set_size = int(0.7*df.shape[0])
    test_set_size = df.shape[0] - training_set_size

    y_all = df['Demanda_uni_equil'].values

    if target_as_log:
        #then all values are done as logs
        df['Demanda_uni_equil'] = transfrom_to_log(df['Demanda_uni_equil'].values)

    train_df = df[:training_set_size]
    test_df = df[-1*test_set_size:]

    y_actual_train = y_all[:training_set_size]
    y_actual_test = y_all[-1*test_set_size:]

    train_df, test_df, testDf = addSlopes(train_df, test_df, testDf)

    demand_val_mean = df['Demanda_uni_equil'].mean()
    demand_val_stddev = df['Demanda_uni_equil'].std()
    #add mean and stddev by groups
    train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Agencia_ID', testDf, drop=True)
    train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Canal_ID', testDf, drop=True)
    train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Ruta_SAK', testDf, drop=True)
    train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Cliente_ID', testDf, drop=True)
    train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, True, demand_val_mean, demand_val_stddev)

    #TODO explore this more http://pandas.pydata.org/pandas-docs/stable/missing_data.html
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    train_df, test_df = drop_column(train_df, test_df, 'Venta_uni_hoy')
    train_df, test_df = drop_column(train_df, test_df, 'Venta_hoy')
    train_df, test_df = drop_column(train_df, test_df, 'Dev_uni_proxima')
    train_df, test_df = drop_column(train_df, test_df, 'Dev_proxima')

    if save_preprocessed_file:
        df.to_csv(preprocessed_file_name, index=False)

    y_train_row = train_df['Demanda_uni_equil'].values
    y_test_row = test_df['Demanda_uni_equil'].values

    train_df, test_df = drop_column(train_df, test_df, 'Demanda_uni_equil')

    if train_df.shape[1] != test_df.shape[1]:
        print "train and test does not match " + list(train_df) + " " + list(test_df)

    print "Forecasting Feilds", list(train_df)

y_train, parmsFromNormalization = preprocess1DtoZeroMeanUnit(y_train_row)
y_test = apply_zeroMeanUnit(y_test_row, parmsFromNormalization)

X_train, parmsFromNormalization2D = preprocess2DtoZeroMeanUnit(train_df.values.copy())
x_test_raw = test_df.values.copy()
X_test = apply_zeroMeanUnit2D(x_test_raw, parmsFromNormalization2D)


prep_time = time.time()

#run_timeseries_froecasts(X_train, y_train, X_test, y_test, -1, 10, parmsFromNormalization)
#regression_with_RFR(X_train, y_train, X_test, y_test, parmsFromNormalization)

model = run_rfr(X_train, y_train, X_test, y_actual_test)
#model, y_pred_corrected = run_xgboost(X_train, y_train, X_test, y_actual_test)
#model, y_pred_corrected = run_dl(X_train, y_train, X_test, y_test)


y_pred_raw = model.predict(X_test)
#undo the normalization
y_pred_final = modeloutput2predictions(y_pred_raw, parmsFromNormalization=parmsFromNormalization)
if target_as_log:
    y_pred_final = retransfrom_from_log(y_pred_final)

error_ac, rmsep, mape, rmse = almost_correct_based_accuracy(y_actual_test, y_pred_final, 10)
rmsle = calculate_rmsle(y_actual_test, y_pred_final)
print ">> %s AC_errorRate=%.1f RMSEP=%.6f MAPE=%6f RMSE=%6f rmsle=%.5f" %("RFR", error_ac, rmsep, mape, rmse, rmsle)


#model = None


if test_run:
    corrected_Y_test = modeloutput2predictions(y_test, parmsFromNormalization=parmsFromNormalization)
    if target_as_log:
        corrected_Y_test = retransfrom_from_log(corrected_Y_test)
    print "Undo normalization test passed", np.allclose(corrected_Y_test, y_actual_test, atol=0.01)

    print "Undo X data test test passed", \
        np.allclose(x_test_raw, undo_zeroMeanUnit2D(X_test, parmsFromNormalization2D), atol=0.01)

    if target_as_log:
        rmsle = calculate_rmsle(y_actual_test, retransfrom_from_log(test_df["groupedMeans"]))
    else:
        rmsle = calculate_rmsle(y_actual_test, test_df["groupedMeans"])
    print "rmsle for mean prediction", rmsle


m_time = time.time()

if testDf is not None and model is not None:
    temp = testDf.drop('id',1)

    if train_df.shape[1] != temp.shape[1]:
        print "train and test does not match " + list(train_df) + " " + list(temp)

    #pd.colnames(temp)[pd.colSums(is.na(temp)) > 0]
    #print temp.describe()
    #print df.isnull().any()
    temp = temp.fillna(0)

    print "forecasting values",  temp.shape

    kaggale_test = apply_zeroMeanUnit2D(temp.values.copy(), parmsFromNormalization2D)

    print "kaggale_test", kaggale_test.shape

    kaggale_predicted_raw = model.predict(kaggale_test)
    kaggale_predicted = modeloutput2predictions(kaggale_predicted_raw, parmsFromNormalization=parmsFromNormalization)

    print "kaggale_predicted", kaggale_test.shape

    if target_as_log:
        kaggale_predicted = retransfrom_from_log(kaggale_predicted)

    print "log retransform", kaggale_test.shape

    to_save = np.column_stack((np.array(list(range(len(kaggale_predicted)))), kaggale_predicted))
    to_saveDf =  pd.DataFrame(to_save, columns=["id","Demanda_uni_equil"])
    to_saveDf = to_saveDf.fillna(0)
    to_saveDf["id"] = to_saveDf["id"].astype(int)
    to_saveDf.to_csv('submission.csv', index=False)

    to_saveDf["groupedMeans"] = testDf["groupedMeans"]
    to_saveDf["groupedStd"] = testDf["groupedStd"]
    to_saveDf["Slopes"] = testDf["Slopes"]
    to_saveDf.to_csv('prediction_detailed.csv', index=False)

    #np.savetxt('submission.csv', to_save, delimiter=',', header="id,Demanda_uni_equil", fmt='%d')   # X is an array

#print "top aggrigate count", len(slopeMap)
print "total=%f, read=%fs, preporcess=%fs, model=%fs" \
      %(m_time-s_time, (r_time-s_time), (prep_time-r_time), (m_time-prep_time))

