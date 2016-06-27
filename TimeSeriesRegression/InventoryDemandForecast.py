import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import time
from sklearn.externals import joblib

from mltools import undoPreprocessing


from tsforecasttools import run_timeseries_froecasts, regression_with_xgboost
from mltools import preprocess2DtoZeroMeanUnit, preprocess1DtoZeroMeanUnit, train_test_split, print_feature_importance
from mltools import calculate_rmsle, almost_correct_based_accuracy, MLConfigs, print_regression_model_summary, regression_with_dl
from keras.optimizers import Adam

#from mlpreprocessing import feather2df

import sys
print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)


def addFeildStatsAsFeatures(df, feild_name, testDf, drop=False, default_mean=None, default_stddev=None):
    start = time.time()
    groupData = df.groupby([feild_name])['Demanda_uni_equil']
    meanData = groupData.mean()
    stddevData = groupData.std()

    valuesDf = meanData.to_frame(feild_name+"_Mean")
    valuesDf.reset_index(inplace=True)
    valuesDf[feild_name+"_StdDev"] = stddevData.values


    calculate_ts = time.time()
    df = pd.merge(df, valuesDf, how='left', on=[feild_name])

    #df[feild_name+"_Mean"] = [meanData.get(x, default_mean) for x in df[feild_name]]
    #df[feild_name+"_StdDev"] = [stddevData.get(x, default_stddev) for x in df[feild_name]]
    if drop:
        newDf = df.drop(feild_name,1)

    if testDf is not None:
        testDf = pd.merge(testDf, valuesDf, how='left', on=[feild_name])
        if default_mean is not None:
            testDf[feild_name+"_Mean"].fillna(default_mean, inplace=True)
        if default_mean is not None:
            testDf[feild_name+"_StdDev"].fillna(default_stddev, inplace=True)
        #testDf[feild_name+"_Mean"] = [meanData.get(x, default_mean) for x in testDf[feild_name]]
        #testDf[feild_name+"_StdDev"] = [stddevData.get(x, default_stddev) for x in testDf[feild_name]]
        if drop:
            testDf = testDf.drop(feild_name,1)
    print "took %f (%f, %f), size %s %f" %(time.time()-start, calculate_ts- start,
                                           time.time() - calculate_ts, feild_name, len(meanData))

    return newDf, testDf

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


def addSlopes(df, testDf):
    start_ts = time.time()
    #TODO do all operations in one go (see Pre)

    #calculate average slope
    grouped = df.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])['Demanda_uni_equil']

    slopeMap = grouped.apply(avgdiff)
    groupedMeanMap = grouped.mean()
    groupedStddevMap = grouped.std()

    valuesDf = slopeMap.to_frame("Slopes")
    valuesDf.reset_index(inplace=True)
    valuesDf["groupedMeans"] = groupedMeanMap.values
    valuesDf["groupedStd"] = groupedStddevMap.values

    slopes_aggr_time = time.time()
    df = pd.merge(df, valuesDf, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
    df.fillna(0, inplace=True)
    #slopes = []
    #groupedMeans = []
    #groupedStddevs = []
    #for index, row in df.iterrows():
    #    slopes.append(slopeMap[row['Agencia_ID'], row['Canal_ID'], row['Ruta_SAK'], row['Cliente_ID'], row['Producto_ID']])
    #    groupedMeans.append(groupedMeanMap[row['Agencia_ID'], row['Canal_ID'], row['Ruta_SAK'], row['Cliente_ID'], row['Producto_ID']])
    #    groupedStddevs.append(groupedStddevMap[row['Agencia_ID'], row['Canal_ID'], row['Ruta_SAK'], row['Cliente_ID'], row['Producto_ID']])

    #df['Slopes']= slopes
    #df['groupedMeans']= groupedMeans
    #df['groupedStd']= groupedStddevs
    if testDf is not None:
        testDf = pd.merge(testDf, valuesDf, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
        testDf.fillna(0, inplace=True)

    slopes_time = time.time()
    print "Slopes took %f (%f, %f)" %(slopes_time - start_ts, slopes_aggr_time-start_ts, slopes_time-slopes_aggr_time)
    return df, testDf


def run_rfr(X_train, Y_train, X_test, y_actual, parmsFromNormalization):
    rfr = RandomForestRegressor()
    rfr.fit(X_train, Y_train)

    #predict data set
    y_pred_rfr = rfr.predict(X_test)

    print_feature_importance(X_test, y_test,rfr.feature_importances_)

    #undo the normalization
    y_actual_4test = y_actual[-1*len(y_pred_rfr):]
    y_pred_final = modeloutput2predictions(y_pred_rfr, parmsFromNormalization=parmsFromNormalization)

    error_ac, rmsep, mape, rmse = almost_correct_based_accuracy(y_actual_4test, y_pred_final, 10)
    rmsle = calculate_rmsle(y_actual_4test, y_pred_final)
    print ">> %s AC_errorRate=%.1f RMSEP=%.6f MAPE=%6f RMSE=%6f rmsle=%.5f" %("RFR", error_ac, rmsep, mape, rmse, rmsle)

    #save model
    joblib.dump(y_pred_rfr, 'model.pkl')
    
    return rfr, y_pred_final

def run_xgboost(X_train, Y_train, X_test, y_actual, parmsFromNormalization):
    model, y_pred = regression_with_xgboost(X_train, Y_train, X_test)
    #undo the normalization
    y_actual = y_actual[-1*len(y_pred):]

    y_pred_corrected = modeloutput2predictions(y_pred, parmsFromNormalization=parmsFromNormalization)

    error_AC, rmsep, mape, rmse = almost_correct_based_accuracy(y_actual, y_pred_corrected, 10)
    rmsle = calculate_rmsle(y_actual, y_pred_corrected)
    print ">> %s AC_errorRate=%.1f RMSEP=%.6f MAPE=%6f RMSE=%6f rmsle=%.5f" %("XGBoost", error_AC, rmsep, mape, rmse, rmsle)
    return model, y_pred_corrected

test_run = True
use_preprocessed_file = False
save_preprocessed_file = False
preprocessed_file_name = "_data.csv"

s_time = time.time()

y_actual = None
if not use_preprocessed_file:
    if test_run:
        df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems300.csv')
        testDf = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/test.csv')
        testDf = testDf[(testDf['Producto_ID'] <= 300)]
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

if not use_preprocessed_file:
    #print "shapes train, test", df.shape, testDf.shape
    print "shapes train, test", df.shape

    df, testDf = addSlopes(df, testDf)

    demand_val_mean = df['Demanda_uni_equil'].mean()
    demand_val_stddev = df['Demanda_uni_equil'].std()
    #add mean and stddev by groups
    df, testDf = addFeildStatsAsFeatures(df,'Agencia_ID', testDf, drop=True)
    df, testDf = addFeildStatsAsFeatures(df,'Canal_ID', testDf, drop=True)
    df, testDf = addFeildStatsAsFeatures(df,'Ruta_SAK', testDf, drop=True)
    df, testDf = addFeildStatsAsFeatures(df,'Cliente_ID', testDf, drop=True)
    df, testDf = addFeildStatsAsFeatures(df,'Producto_ID', testDf, True, demand_val_mean, demand_val_stddev)

    print df.head(10)

    #TODO explore this more http://pandas.pydata.org/pandas-docs/stable/missing_data.html
    df = df.fillna(0)

    y_actual = df['Demanda_uni_equil'].values
    Y_all, parmsFromNormalization = preprocess1DtoZeroMeanUnit(y_actual)
    df = df.drop('Venta_uni_hoy',1)
    df = df.drop('Venta_hoy',1)
    df = df.drop('Dev_uni_proxima',1)
    df = df.drop('Dev_proxima',1)

    if save_preprocessed_file:
        df.to_csv(preprocessed_file_name, index=False)

    df = df.drop('Demanda_uni_equil',1)

    print "df feilds", list(df)
    #print "testDf", list(testDf)


X_all = preprocess2DtoZeroMeanUnit(df.values.copy())

training_set_size = int(0.7*X_all.shape[0])
X_train, X_test, y_train, y_test = train_test_split(training_set_size, X_all, Y_all)

prep_time = time.time()

#run_timeseries_froecasts(X_train, y_train, X_test, y_test, -1, 10, parmsFromNormalization)
#regression_with_RFR(X_train, y_train, X_test, y_test, parmsFromNormalization)

model, y_pred_corrected = run_rfr(X_train, y_train, X_test, y_actual, parmsFromNormalization)
#run_xgboost(X_train, y_train, X_test, y_actual, parmsFromNormalization)

#c = MLConfigs(nodes_in_layer=20, number_of_hidden_layers=3, dropout=0.0, activation_fn='relu', loss="mse",
#              epoch_count=50, optimizer=Adam(lr=0.0001), regularization=0.001)
#y_pred_dl = regression_with_dl(X_train, y_train, X_test, y_test, c)
#print ">> %d %s" %(index, str(c.tostr()))
#print_regression_model_summary("DL", y_test, y_pred_dl, parmsFromNormalization)


if test_run:
    corrected_Y_test = modeloutput2predictions(y_test, parmsFromNormalization=parmsFromNormalization)
    y_actual_test = y_actual[-1*len(corrected_Y_test):]
    print "Undo normalization test passed", np.allclose(corrected_Y_test, y_actual_test, atol=0.01)

m_time = time.time()

if testDf is not None:
    temp = testDf.drop('id',1)
    #pd.colnames(temp)[pd.colSums(is.na(temp)) > 0]
    #print temp.describe()
    #print df.isnull().any()
    temp = temp.fillna(0)

    kaggale_test = preprocess2DtoZeroMeanUnit(temp.values.copy())
    kaggale_predicted_raw = model.predict(kaggale_test)
    kaggale_predicted = modeloutput2predictions(kaggale_predicted_raw, parmsFromNormalization=parmsFromNormalization)

    to_save = np.column_stack((np.array(list(range(len(kaggale_predicted)))), kaggale_predicted))
    to_saveDf =  pd.DataFrame(to_save, columns=["id","Demanda_uni_equil"])
    to_saveDf = to_saveDf.fillna(0)
    to_saveDf["id"] = to_saveDf["id"].astype(int)
    to_saveDf.to_csv('submission.csv', index=False)
    #np.savetxt('submission.csv', to_save, delimiter=',', header="id,Demanda_uni_equil", fmt='%d')   # X is an array

#print "top aggrigate count", len(slopeMap)
print "total=%f, read=%fs, preporcess=%fs, model=%fs" \
      %(m_time-s_time, (r_time-s_time), (prep_time-r_time), (m_time-prep_time))

