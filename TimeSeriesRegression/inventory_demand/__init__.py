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
from sklearn.feature_extraction import DictVectorizer
from sklearn import  preprocessing


from mltools import preprocess2DtoZeroMeanUnit, preprocess1DtoZeroMeanUnit, train_test_split, print_feature_importance, apply_zeroMeanUnit2D
from mltools import calculate_rmsle, almost_correct_based_accuracy, MLConfigs, print_regression_model_summary, regression_with_dl, apply_zeroMeanUnit, undo_zeroMeanUnit2D
from keras.optimizers import Adam



def join_multiple_feild_stats(bdf, testdf, subdf, feild_names, agr_feild, name, default_mean=None, default_stddev=None):
    groupData = bdf.groupby(feild_names)[agr_feild]
    meanData = groupData.mean()
    stddevData = groupData.std()
    countData = groupData.count()

    valuesDf = meanData.to_frame(name+"_Mean")
    valuesDf.reset_index(inplace=True)
    valuesDf[name+"_StdDev"] = stddevData.values
    valuesDf[name+"_"+agr_feild+"_Count"] = countData.values

    bdf = merge__multiple_feilds_stats_with_df(name, bdf, valuesDf, feild_names, default_mean, default_stddev)
    testdf = merge__multiple_feilds_stats_with_df(name, testdf, valuesDf, feild_names, default_mean, default_stddev)
    subdf = merge__multiple_feilds_stats_with_df(name, subdf, valuesDf, feild_names, default_mean, default_stddev)

    return bdf, testdf, subdf


def merge__multiple_feilds_stats_with_df(name, bdf, stat_df, feild_names, default_mean=None, default_stddev=None, agr_feild='Demanda_uni_equil'):
    merged = pd.merge(bdf, stat_df, how='left', on=feild_names)

    if default_mean is not None:
        merged[name+"_Mean"].fillna(default_mean, inplace=True)
    if default_stddev is not None:
        merged[name+"_StdDev"].fillna(default_stddev, inplace=True)
    return merged



def calculate_feild_stats(bdf, feild_name, agr_feild):
    groupData = bdf.groupby([feild_name])[agr_feild]
    meanData = groupData.mean()
    stddevData = groupData.std()
    countData = groupData.count()

    valuesDf = meanData.to_frame(feild_name+"_"+agr_feild+"_Mean")
    valuesDf.reset_index(inplace=True)
    valuesDf[feild_name+"_"+agr_feild+"_StdDev"] = stddevData.values
    valuesDf[feild_name+"_"+agr_feild+"_Count"] = countData.values
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

def modeloutput2predictions(model_forecast, parmsFromNormalization, negative_allowed=False):
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
    groupedCountMap = grouped.count()


    valuesDf = slopeMap.to_frame("Slopes")
    valuesDf.reset_index(inplace=True)
    valuesDf["groupedMeans"] = groupedMeanMap.values
    valuesDf["groupedStd"] = groupedStddevMap.values
    valuesDf["groupedMedian"] = groupedmedianMap.values
    valuesDf["groupedCount"] = groupedCountMap.values

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



def run_dl(X_train, y_train, X_test, y_test, c=None):
    print "Running DL"
    if c == None:
        c = MLConfigs(nodes_in_layer=10, number_of_hidden_layers=2, dropout=0.2, activation_fn='relu', loss="mse",
             epoch_count=10, optimizer=Adam(lr=0.0001), regularization=0.1)
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


def check_accuracy(label, model, X_test, parmsFromNormalization, test_df, target_as_log, y_actual_test, command):
    y_pred_raw = model.predict(X_test)
    #undo the normalization
    y_pred_final = modeloutput2predictions(y_pred_raw, parmsFromNormalization=parmsFromNormalization)
    if target_as_log:
        y_pred_final = retransfrom_from_log(y_pred_final)

    error_ac, rmsep, mape, rmse = almost_correct_based_accuracy(y_actual_test, y_pred_final, 10)
    rmsle = calculate_rmsle(y_actual_test, y_pred_final)
    print ">> %s AC_errorRate=%.1f RMSEP=%.6f MAPE=%6f RMSE=%6f rmsle=%.5f" %("Run " + str(command)+ " "+ label, error_ac, rmsep, mape, rmse, rmsle)
    return y_pred_final


def merge_clusters(traindf, testdf, subdf):
    clusterdf = pd.read_csv('product_clusters.csv')
    #clusterdf = pd.read_csv('product_more_data.csv')

    train_df_m = pd.merge(traindf, clusterdf, how='left', on=['Producto_ID'])
    test_df_m = pd.merge(testdf, clusterdf, how='left', on=['Producto_ID'])
    sub_df_m = pd.merge(subdf, clusterdf, how='left', on=['Producto_ID'])

    return train_df_m, test_df_m, sub_df_m

#def do_one_hot(base_df, feild_name):
#    one_hot = base_df.get_dummies(base_df[feild_name])
#    base_df = base_df.drop(feild_name, axis=1)
#    return base_df.join(one_hot)

def doPCA():
    #DO PCA on the data and use it to transform
    svd = TruncatedSVD(5)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)


def encode_onehot(df, cols, vec=None):
    """
    One-hot encoding is applied to columns specified in a pandas DataFrame.

    Modified from: https://gist.github.com/kljensen/5452382

    Details:

    http://en.wikipedia.org/wiki/One-hot
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    x_data = df[cols]
    if vec is None:
        vec = preprocessing.OneHotEncoder()
        results = vec.fit_transform(x_data).toarray()
    else:
        results = vec.transform(x_data).toarray()

    '''

    '''


    size = results.shape[1]
    vec_data = pd.DataFrame(results)
    vec_data.columns = ["f"+str(i) for i in range(0, size)]
    vec_data.index = df.index

    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df, vec


def do_one_hot_all(traindf, testdf, subdf, feild_names):
    traindf_, vec = encode_onehot(traindf, feild_names)
    testdf_, _ = encode_onehot(testdf, feild_names, vec)
    subdf_, _ = encode_onehot(subdf, feild_names, vec)
    return traindf_, testdf_, subdf_


def group_to_df(group, name):
    valuesDf = group.to_frame(name)
    valuesDf.reset_index(inplace=True)
    return valuesDf


def group_to_df_sum_mean(group):
    sum = group.sum()
    mean = group.mean()
    count = group.count()
    valuesDf = sum.to_frame("sum")
    valuesDf.reset_index(inplace=True)
    valuesDf['mean'] = mean.values
    valuesDf['count'] = count.values
    valuesDf['rank'] = valuesDf['mean']*np.log(1+valuesDf['count'])

    return valuesDf