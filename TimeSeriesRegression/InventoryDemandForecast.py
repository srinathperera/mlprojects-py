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

#from mlpreprocessing import feather2df

import sys
print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)



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
        #df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems0_1000.csv')

        #df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems2000_10000.csv')
        #df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems1000_2000.csv')

        testDf = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/test.csv')
        #testDf = testDf[(testDf['Producto_ID'] <= 1000)]
        testDf = testDf[(testDf['Producto_ID'] <= 300)]
        #testDf = testDf[(testDf['Producto_ID'] < 2000) & (testDf['Producto_ID'] >= 1000)]

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
    df['unit_prize'] = df['Venta_hoy']/df['Venta_uni_hoy']

    #df['UClient'] =

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
    #demand_val_mean = df['Demanda_uni_equil'].median()
    demand_val_stddev = df['Demanda_uni_equil'].std()
    #add mean and stddev by groups

    groups = ('Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID')
    measures = ('Dev_proxima', 'Dev_uni_proxima', 'Demanda_uni_equil', 'unit_prize', 'Venta_uni_hoy', 'Venta_hoy')
    #for t in itertools.product(groups, measures):
    for t in [('Cliente_ID', 'Demanda_uni_equil'), ('Cliente_ID', 'Venta_uni_hoy'), ('Cliente_ID', 'Venta_hoy'),
        ('Ruta_SAK', 'unit_prize')]:
            train_df, test_df, testDf = addFeildStatsAsFeatures(train_df,
                                    test_df,t[0], testDf, drop=False, agr_feild=t[1])


    #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Agencia_ID', testDf, drop=False)
    #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Canal_ID', testDf, drop=False)
    #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Ruta_SAK', testDf, drop=False)
    #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Cliente_ID', testDf, drop=False) #duplicated
    #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, False, demand_val_mean, demand_val_stddev)

    #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, drop=False, agr_feild='unit_prize')
    #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, drop=False, agr_feild='Dev_proxima')
    #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, drop=False, agr_feild='Venta_hoy')




    train_df, test_df, testDf = drop_feilds(train_df, test_df, testDf, ['Canal_ID','Cliente_ID','Producto_ID', 'Agencia_ID', 'Ruta_SAK'])


    #TODO explore this more http://pandas.pydata.org/pandas-docs/stable/missing_data.html
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    train_df, test_df = drop_column(train_df, test_df, 'Venta_uni_hoy')
    train_df, test_df = drop_column(train_df, test_df, 'Venta_hoy')
    train_df, test_df = drop_column(train_df, test_df, 'Dev_uni_proxima')
    train_df, test_df = drop_column(train_df, test_df, 'Dev_proxima')
    train_df, test_df = drop_column(train_df, test_df, 'unit_prize')


    if save_preprocessed_file:
        df.to_csv(preprocessed_file_name, index=False)

    y_train_row = train_df['Demanda_uni_equil'].values
    y_test_row = test_df['Demanda_uni_equil'].values

    train_df, test_df = drop_column(train_df, test_df, 'Demanda_uni_equil')

    if train_df.shape[1] != test_df.shape[1]:
        print "train and test does not match " + list(train_df) + " " + list(test_df)

    forecasting_feilds = list(train_df)
    print "Forecasting Feilds", [ "("+str(i)+")" + forecasting_feilds[i] for i in range(len(forecasting_feilds))]

y_train, parmsFromNormalization = preprocess1DtoZeroMeanUnit(y_train_row)
y_test = apply_zeroMeanUnit(y_test_row, parmsFromNormalization)

X_train, parmsFromNormalization2D = preprocess2DtoZeroMeanUnit(train_df.values.copy())
x_test_raw = test_df.values.copy()
X_test = apply_zeroMeanUnit2D(x_test_raw, parmsFromNormalization2D)


prep_time = time.time()

#run_timeseries_froecasts(X_train, y_train, X_test, y_test, -1, 10, parmsFromNormalization)
#regression_with_RFR(X_train, y_train, X_test, y_test, parmsFromNormalization)

if X_train.shape[1] != X_test.shape[1]:
    print " columns not aligned X_train, y_train, X_test, y_test", X_train.shape, y_train.shape, X_test.shape, y_test.shape

if X_train.shape[0] != y_train.shape[0] or y_test.shape[0] != X_test.shape[0]:
    print "rows not aligned X_train, y_train, X_test, y_test", X_train.shape, y_train.shape, X_test.shape, y_test.shape

#model = run_rfr(X_train, y_train, X_test, y_test, forecasting_feilds)
#model = run_lr(X_train, y_train, X_test, y_test)
model = run_xgboost(X_train, y_train, X_test, y_test)
#model = run_dl(X_train, y_train, X_test, y_test)


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
    ids = testDf['id']
    temp = testDf.drop('id',1)

    if train_df.shape[1] != temp.shape[1]:
        print "train and test does not match " + str(list(train_df)) + " " + str(list(temp))

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

    to_save = np.column_stack((ids, kaggale_predicted))
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

