import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import time
import math
from sklearn.externals import joblib

from mltools import undoPreprocessing
from keras.optimizers import Adam
from sklearn.linear_model import LinearRegression
from tsforecasttools import run_timeseries_froecasts, regression_with_xgboost



from mltools import preprocess2DtoZeroMeanUnit, preprocess1DtoZeroMeanUnit, train_test_split, print_feature_importance, apply_zeroMeanUnit2D
from mltools import calculate_rmsle, almost_correct_based_accuracy, MLConfigs, print_regression_model_summary, regression_with_dl, apply_zeroMeanUnit, undo_zeroMeanUnit2D
from keras.optimizers import Adam


def calculate_feild_stats(bdf, feild_name, agr_feild):
    groupData = bdf.groupby([feild_name])[agr_feild]
    meanData = groupData.mean()
    stddevData = groupData.std()

    valuesDf = meanData.to_frame(feild_name+"_"+agr_feild+"_Mean")
    valuesDf.reset_index(inplace=True)
    valuesDf[feild_name+"_"+agr_feild+"_StdDev"] = stddevData.values
    return valuesDf


def merge_stats_with_df(bdf, stat_df, feild_name, default_mean=None, default_stddev=None, agr_feild='Demanda_uni_equil'):
    merged = pd.merge(bdf, stat_df, how='left', on=[feild_name])

    if default_mean is not None:
        merged[feild_name+"_"+agr_feild+"_Mean"].fillna(default_mean, inplace=True)
    if default_stddev is not None:
        merged[feild_name+"_"+agr_feild++"_StdDev"].fillna(default_stddev, inplace=True)
    return merged


def addFeildStatsAsFeatures(train_df, test_df, feild_name, testDf, drop=False, default_mean=None,
                            default_stddev=None, agr_feild='Demanda_uni_equil'):
    start = time.time()
    valuesDf = calculate_feild_stats(train_df, feild_name, agr_feild)

    calculate_ts = time.time()
    train_df_m = pd.merge(train_df, valuesDf, how='left', on=[feild_name])
    test_df_m = pd.merge(test_df, valuesDf, how='left', on=[feild_name])

    if drop:
        train_df_m = train_df_m.drop(feild_name,1)
        test_df_m = test_df_m.drop(feild_name,1)

    if testDf is not None:
        testDf = pd.merge(testDf, valuesDf, how='left', on=[feild_name])
        if default_mean is not None:
            testDf[feild_name+"_"+agr_feild+"_Mean"].fillna(default_mean, inplace=True)
        if default_stddev is not None:
            testDf[feild_name+"_"+agr_feild+"_StdDev"].fillna(default_stddev, inplace=True)
        if drop:
            testDf = testDf.drop(feild_name,1)
    print "took %f (%f, %f), size %s %f" %(time.time()-start, calculate_ts- start,
                                           time.time() - calculate_ts, feild_name, valuesDf.shape[0])

    return train_df_m, test_df_m, testDf

def avgdiff(group):
    group = group.values
    if len(group) > 1:
        return np.mean([group[i] - group[i-1] for i in range(1, len(group))])
    else:
        return 0

#TODO fix defaults

def modeloutput2predictions(model_forecast, parmsFromNormalization, default_forecast, negative_allowed=False):
    y_pred_corrected = undoPreprocessing(model_forecast, parmsFromNormalization)
    zero_frecasts = np.where(y_pred_corrected < 0)[0]
    print "zero forecasts=%d, %f precent" %(zero_frecasts.shape[0], float(zero_frecasts.shape[0])/y_pred_corrected.shape[0])
    if not negative_allowed:
        for i in range(y_pred_corrected.shape[0]):
            if y_pred_corrected[i] < 0:
                #y_pred_corrected[i] = default_forecast[i]
                y_pred_corrected[i] = 1
        #y_pred_corrected = np.where(y_pred_corrected < 0, 1, y_pred_corrected)
    return y_pred_corrected


def drop_feilds(train_df, test_df, testDf, feilds):
    train_df_t = train_df
    test_df_t = test_df
    testDf_t = testDf
    for feild_name in feilds:
        train_df_t = train_df_t.drop(feild_name,1)
        test_df_t = test_df_t.drop(feild_name,1)
        testDf_t = testDf_t.drop(feild_name,1)

    return train_df_t, test_df_t, testDf_t

def drop_feilds_1df(df, feilds):
    df_t = df
    for feild_name in feilds:
        df_t = df_t.drop(feild_name,1)
    return df_t



def addSlopes(train_df, test_df, testDf):
    start_ts = time.time()
    #TODO do all operations in one go (see Pre)

    #calculate average slope
    grouped = train_df.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])['Demanda_uni_equil']

    slopeMap = grouped.apply(avgdiff)
    groupedMeanMap = grouped.mean()
    groupedStddevMap = grouped.std()
    groupedmedianMap = grouped.median()

    valuesDf = slopeMap.to_frame("Slopes")
    valuesDf.reset_index(inplace=True)
    valuesDf["groupedMeans"] = groupedMeanMap.values
    valuesDf["groupedStd"] = groupedStddevMap.values
    valuesDf["groupedMedian"] = groupedmedianMap.values

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


def run_rfr(X_train, Y_train, X_test, y_test, forecasting_feilds=None):
    #X_train = X_train.astype('float32')
    #X_train = np.nan_to_num(X_train)

    print "Running RFR"
    rfr = RandomForestRegressor()
    rfr.fit(X_train, Y_train)

    print_feature_importance(rfr.feature_importances_, forecasting_feilds)

    #save model
    joblib.dump(rfr, 'model.pkl')
    return rfr


def run_lr(X_train, Y_train, X_test, y_test):
    print "Running LR"
    lr = LinearRegression(normalize=True)
    lr.fit(X_train, Y_train)
    return lr



def run_dl(X_train, y_train, X_test, y_test):
    print "Running DL"
    c = MLConfigs(nodes_in_layer=30, number_of_hidden_layers=3, dropout=0.1, activation_fn='relu', loss="mse",
             epoch_count=30, optimizer=Adam(lr=0.0001), regularization=0.01)
    model, y_pred_dl = regression_with_dl(X_train, y_train, X_test, y_test, c)
    return model


def run_xgboost(X_train, Y_train, X_test, Y_test, forecasting_feilds=None):
    print "Running XG Boost"
    model, y_pred = regression_with_xgboost(X_train, Y_train, X_test, Y_test, forecasting_feilds)
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


class InventoryDemandPredictor:
    def __init__(self, model):
        self.model = model

    def preprocess_x(self, x_data):
        return None

    def preprocess_y(self, y_data):
        return None


    def add_features2train(self, x_train):
        return None

    def transfrom4prediction(self, data):
        return None


    #models should be able to correct normalizations
    def train(self, models, x_train, y_train, x_test, y_test):
        x_train_predictions = np.hstack([m.predict(x_train) for m in models])
        x_test_predictions = np.hstack([m.predict(x_test) for m in models])

        #return lr

    def predict(self, x_data):
        return self.model.predict(x_data)


def avgdiff(group):
    group = group.values
    if len(group) > 1:
        slope =  np.mean([group[i] - group[i-1] for i in range(1, len(group))])
        return slope
    else:
        return 0

