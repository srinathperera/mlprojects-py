import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import time
import math
from sklearn.externals import joblib

from mltools import undoPreprocessing
from keras.optimizers import Adam
from sklearn.linear_model import LinearRegression
from tsforecasttools import run_timeseries_froecasts, regression_with_xgboost, regression_with_xgboost_no_cv
from sklearn.feature_extraction import DictVectorizer
from sklearn import  preprocessing


from mltools import preprocess2DtoZeroMeanUnit, preprocess1DtoZeroMeanUnit, train_test_split, print_feature_importance, apply_zeroMeanUnit2D
from mltools import calculate_rmsle, almost_correct_based_accuracy, MLConfigs, print_regression_model_summary, regression_with_dl, apply_zeroMeanUnit, undo_zeroMeanUnit2D
from keras.optimizers import Adam

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from mltools import *
import scipy

def read_train_file(file_name):
    #extended memory  https://xgboost.readthedocs.io/en/latest//how_to/external_memory.html
    train = pd.read_csv(file_name,
                    dtype  = {'Semana' : 'int32',
                              'Agencia_ID' :'int32',
                              'Canal_ID' : 'int32',
                              'Ruta_SAK' : 'int32',
                              'Cliente-ID' : 'int64',
                              'Producto_ID':'int32',
                              'Venta_hoy':'float32',
                              'Venta_uni_hoy': 'int32',
                              'Dev_uni_proxima':'int32',
                              'Dev_proxima':'float32',
                              'Demanda_uni_equil':'int32'})
    return train

def read_productdata_file(file_name):
    #weight,pieces,has_choco,has_vanilla,has_multigrain,Producto_ID,brand_id
    train = pd.read_csv(file_name,
                    dtype  = {'weight' : 'int16',
                              'pieces' :'int16',
                              'has_choco' : 'int16',
                              'has_vanilla' : 'int16',
                              'has_multigrain' : 'int16',
                              'Producto_ID':'int32',
                              'brand_id':'float32'})
    return train




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

def calcuate_hmean(group):
    x = group.values
    non_zero_x = np.where(x > 0)
    if len(non_zero_x) > 0:
        #print non_zero_x
        return scipy.stats.hmean(x[non_zero_x])
    else:
        return 0


def calculate_feild_stats(bdf, feild_name, agr_feild, do_count=True, do_stddev=True):
    groupData = bdf.groupby([feild_name])[agr_feild]
    #meanData = groupData.mean()

    meanData = groupData.apply(calcuate_hmean)
    valuesDf = meanData.to_frame(feild_name+"_"+agr_feild+"_Mean")
    valuesDf.reset_index(inplace=True)

    #valuesDf[feild_name+"_"+agr_feild+"_hMean"] =
    #valuesDf = drop_feilds_1df(valuesDf, [feild_name+"_"+agr_feild+"_Mean"])
    #meanData = groupData.median()

    if do_stddev:
        stddevData = groupData.std()
        valuesDf[feild_name+"_"+agr_feild+"_StdDev"] = stddevData.values
    if do_count:
        countData = groupData.count()
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
                            default_stddev=None, agr_feild='Demanda_uni_equil', do_count=True, do_stddev=True):
    start = time.time()
    valuesDf = calculate_feild_stats(train_df, feild_name, agr_feild, do_count=do_count, do_stddev=do_stddev)

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
        if default_stddev is not None and feild_name+"_"+agr_feild+"_StdDev" in testDf:
            testDf[feild_name+"_"+agr_feild+"_StdDev"].fillna(default_stddev, inplace=True)
        if drop:
            testDf = testDf.drop(feild_name,1)
    print "took %f (%f, %f), size %s %f" %(time.time()-start, calculate_ts- start,
                                           time.time() - calculate_ts, feild_name, valuesDf.shape[0])

    return train_df_m, test_df_m, testDf


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

def avgdiff(group):
    group = group.values
    if len(group) > 1:
        slope = np.mean([group[i] - group[i-1] for i in range(1, len(group))])
        return slope
    else:
        return 0

def calcuate_slope_stats(group):
    sales = np.array(group['Demanda_uni_equil'].values)
    samana = group['Semana'].values

    #agencia_ID = group['Demanda_uni_equil'].values[0]
    #canal_ID = group['Canal_ID'].values[0]
    #ruta_SAK = group['Ruta_SAK'].values[0]
    #cliente_ID = group['Cliente_ID'].values[0]
    #producto_ID = group['Producto_ID'].values[0]

    slopes = []
    tot_slope = 0
    min_slope = 100
    max_slope = -100

    if len(sales) > 1:
        for i in range(1, len(sales)):
            if (samana[i] - samana[i-1]) > 0:
                slope = (sales[i] - sales[i-1])/(samana[i] - samana[i-1])
                if slope < min_slope: min_slope = slope
                if slope > max_slope: max_slope = slope
                tot_slope = tot_slope + slope
                slopes.append(slope)
            else:
                print "same weeks happens again", samana[i], samana[i-1]

        slopes = np.array(slopes)
        mean_slope = tot_slope/ len(slopes)
        #"Mean_Slope", "Min_Slope", "Max_slope", "Stddev_Slope","Last_Sale", "Last_Sale_Week", "Sales_Mean"
        #Stddev_Sales, "Median_Sales", "Count"
        return [mean_slope, min_slope, max_slope, np.std(slopes), sales[-1], samana[-1], np.mean(sales),
                np.std(sales), np.median(sales), len(sales)]
    elif len(sales) == 1:
        return [0, 0, 0, 0, sales[-1], samana[-1], sales[0], 0, sales[0], 1]
    else:
        return [0, 0, 0, 0, sales[-1], samana[-1], 0, 0, 0, 0]
    #return [np.mean(slopes), np.min(slopes), np.max(slopes), np.std(slopes), sales[-1], samana[-1],
    #    np.mean(sales), np.std(sales), np.median(sales), np.min(sales), np.max(sales)]

def process_merged_df(dfp):
    dfp["Time_Since_Last_Sale_Value"] = dfp['Semana'] - dfp["Last_Sale_Week"]
    dfp["Simple_Forecast"] = dfp['Last_Sale'] + dfp["Mean_Slope"]*dfp["Time_Since_Last_Sale_Value"]
    dfp = drop_feilds_1df(dfp, ["Last_Sale_Week"])
    return dfp



def addSlopes(train_df, test_df, testDf):
    start_ts = time.time()
    #TODO do all operations in one go (see Pre)


    #calculate average slope
    #grouped = train_df.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])['Demanda_uni_equil']
    grouped = train_df.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
    slope_data_df = grouped.apply(calcuate_slope_stats)

    valuesDf = slope_data_df.to_frame("SlopeData")
    valuesDf.reset_index(inplace=True)

    valuesDf = expand_array_feild_and_add_df(valuesDf, 'SlopeData', ["Mean_Slope", "Min_Slope", "Max_slope", "Stddev_Slope",
        "Last_Sale", "Last_Sale_Week", "Sales_Mean", "Stddev_Sales", "Median_Sales", "Count"])
    print valuesDf.head(10)


    '''


    print "slopeMap", slopeMap.head()
    print type(slopeMap)


    groupedMeanMap = grouped['Demanda_uni_equil'].mean()
    groupedStddevMap = grouped['Demanda_uni_equil'].std()
    groupedmedianMap = grouped['Demanda_uni_equil'].median()
    groupedCountMap = grouped['Demanda_uni_equil'].count()


    valuesDf = slopeMap.to_frame("Slopes")
    valuesDf.reset_index(inplace=True)

    print "valuesDf", valuesDf.head(10)
    valuesDf["groupedMeans"] = groupedMeanMap.values
    valuesDf["groupedStd"] = groupedStddevMap.values
    valuesDf["groupedMedian"] = groupedmedianMap.values
    valuesDf["groupedCount"] = groupedCountMap.values

    t1 = time.time()
    grouped = train_df.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
    slope_stats_results = []
    grouped.apply(calcuate_slope_stats, slope_stats_results)

    slope_stats_results_np = np.vstack(slope_stats_results)
    check4nan(slope_stats_results_np)

    print "valuesDf", "New", valuesDf.shape, slope_stats_results_np.shape


    valuesDf["Mean_Slope"] = slope_stats_results_np[:,5]
    valuesDf["Min_Slope"] = slope_stats_results_np[:,5]
    valuesDf["Max_slope"] = slope_stats_results_np[:,5]
    valuesDf["Stddev_Slope"] = slope_stats_results_np[:,5]
    valuesDf["Last_Sale"] = slope_stats_results_np[:,5]
    valuesDf["Last_Sale_Week"] = slope_stats_results_np[:,5]

    t2 = time.time()

    print "second group took", (t2-t1), " seconds"
    columns = ['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID', "Mean_Slope", "Min_Slope", "Max_slope", "Stddev_Slope",
        "Last_Sale", "Last_Sale_Week"]
    '''




    slopes_aggr_time = time.time()
    train_df_m = pd.merge(train_df, valuesDf, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
    train_df_m.fillna(0, inplace=True)

    test_df_m = pd.merge(test_df, valuesDf, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
    test_df_m.fillna(0, inplace=True)

    if testDf is not None:
        testDf = pd.merge(testDf, valuesDf, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
        testDf.fillna(0, inplace=True)

    train_df_m = process_merged_df(train_df_m)
    test_df_m = process_merged_df(test_df_m)
    testDf = process_merged_df(testDf)

    slopes_time = time.time()
    print "Slopes took %f (%f, %f)" %(slopes_time - start_ts, slopes_aggr_time-start_ts, slopes_time-slopes_aggr_time)
    return train_df_m, test_df_m, testDf



class RFRModel:
    def __init__(self, conf):
        self.conf = conf
    def fit(self, X_train, y_train, X_test, y_test, y_actual, forecasting_feilds=None):
        self.model = RandomForestRegressor(n_jobs=4)
        self.model.fit(X_train, y_train)
        print_feature_importance(self.model.feature_importances_, forecasting_feilds)

        #save model
        #joblib.dump(rfr, 'model.pkl')
        y_pred_final, rmsle = check_accuracy("RFR", self.model, X_test, self.conf.parmsFromNormalization,
                                      self.conf.target_as_log, y_actual, self.conf.command)
        self.rmsle =  rmsle
        return y_pred_final

    def predict(self, X_test):
        return self.model.predict(X_test)



class XGBoostModel:
    def __init__(self, conf, xgb_params):
        self.conf = conf
        self.xgb_params = xgb_params
    def fit(self, X_train, y_train, X_test, y_test, y_actual, forecasting_feilds=None):
        model, y_pred = regression_with_xgboost_no_cv(X_train, y_train, X_test, y_test, features=forecasting_feilds,
                                                      xgb_params=self.xgb_params,num_rounds=200)
        self.model = model
        y_pred_final, rmsle = check_accuracy("XGBoost", self.model, X_test, self.conf.parmsFromNormalization,
                                      self.conf.target_as_log, y_actual, self.conf.command)
        self.rmsle =  rmsle
        return y_pred_final

    def predict(self, X_test):
        return self.model.predict(X_test)

class LRModel:
    def __init__(self, conf):
        self.conf = conf
    def fit(self, X_train, y_train, X_test, y_test, y_actual, forecasting_feilds=None):
        self.model = LinearRegression(normalize=True)
        self.model.fit(X_train, y_train)

        y_pred_final, rmsle = check_accuracy("LR", self.model, X_test, self.conf.parmsFromNormalization,
                                      self.conf.target_as_log, y_actual, self.conf.command)
        self.rmsle =  rmsle
        return y_pred_final

    def predict(self, X_test):
        return self.model.predict(X_test)


class DLModel:
    def __init__(self, conf, dlconf=None):
        self.conf = conf
        self.dlconf = dlconf
    def fit(self, X_train, y_train, X_test, y_test, y_actual, forecasting_feilds=None):
        if self.dlconf == None:
            self.dlconf = MLConfigs(nodes_in_layer=10, number_of_hidden_layers=2, dropout=0.3, activation_fn='relu', loss="mse",
                epoch_count=10, optimizer=Adam(lr=0.0001), regularization=0.2)
        model, y_pred_dl = regression_with_dl(X_train, y_train, X_test, y_test, self.dlconf)
        self.model = model

        y_pred_final, rmsle = check_accuracy("LR", self.model, X_test, self.conf.parmsFromNormalization,
                                      self.conf.target_as_log, y_actual, self.conf.command)
        self.rmsle =  rmsle
        return y_pred_final

    def predict(self, X_test):
        return self.model.predict(X_test)


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


def avgdiff1(group):
    group = group.values
    if len(group) > 1:
        slope =  np.mean([group[i] - group[i-1] for i in range(1, len(group))])
        return slope
    else:
        return 0

def calculate_slope(group):
    group = group['Demanda_uni_equil'].values
    if len(group) > 1:
        slope =  np.mean([group[i] - group[i-1] for i in range(1, len(group))])
        #return pd.Series({'slope': slope, 'mean': 0})
        return [slope, 0]
    else:
        #return pd.Series({'slope': 0, 'mean': 0})
        return [0, 0]


def check_accuracy(label, model, X_test, parmsFromNormalization, target_as_log, y_actual_test, command):
    y_pred_raw = model.predict(X_test)
    #undo the normalization
    y_pred_final = modeloutput2predictions(y_pred_raw, parmsFromNormalization=parmsFromNormalization)
    if target_as_log:
        y_pred_final = retransfrom_from_log(y_pred_final)

    error_ac, rmsep, mape, rmse = almost_correct_based_accuracy(y_actual_test, y_pred_final, 10)
    rmsle = calculate_rmsle(y_actual_test, y_pred_final)
    print ">> %s AC_errorRate=%.1f RMSEP=%.6f MAPE=%6f RMSE=%6f rmsle=%.5f" %("Run " + str(command)+ " "+ label, error_ac, rmsep, mape, rmse, rmsle)
    return y_pred_final, rmsle


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

def doPCA(X, output_columns_count):
    #DO PCA on the data and use it to transform
    svd = TruncatedSVD(output_columns_count)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)
    return X


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

    result_size = results.shape[1]

    #TODO bug in pca code, find and fix
    #after_pca_size = 4*len(cols)
    #if(result_size > 5 and after_pca_size < result_size):
    #    results = results[:min(500000,  results.shape[0])]
    #    results = doPCA(results, after_pca_size)
    #    result_size = after_pca_size

    vec_data = pd.DataFrame(results)
    vec_data.columns = ["f"+str(i) for i in range(0, result_size)]
    vec_data.index = df.index

    #df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df, vec


def do_one_hot_all(traindf, testdf, subdf, feild_names):
    traindf_, vec = encode_onehot(traindf, feild_names)
    testdf_, _ = encode_onehot(testdf, feild_names, vec)
    subdf_, _ = encode_onehot(subdf, feild_names, vec)
    return traindf_, testdf_, subdf_


def only_keep_top_categories(traindf, testdf, subdf, feild_name, category_count):
    counts = traindf[feild_name].value_counts()
    counts = counts.sort_values(ascending=False)
    top = counts[0:category_count]

    print "coverage", feild_name, float(top.sum())/counts.sum()

    mapf = dict(top)
    traindf[feild_name] = [ mapf.get(k, 0) for k in traindf[feild_name].values]
    testdf[feild_name] = [ mapf.get(k, 0) for k in testdf[feild_name].values]
    subdf[feild_name] = [ mapf.get(k, 0) for k in subdf[feild_name].values]
    return traindf, testdf, subdf


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


def expand_array_feild_and_add_df(tdf, composite_feild, feilds):
    npa = np.vstack(tdf[composite_feild].values)

    #valuesDf['mean'] = npa[:,1]
    index = 0
    for f in feilds:
        tdf[f] = npa[:,index]
        index = index +1

    tdf = drop_feilds_1df(tdf, [composite_feild])

    return tdf

def merge_csv_by_feild(train_df, test_df, testDf, base_df, feild_name ):
    train_df = pd.merge(train_df, base_df, how='left', on=[feild_name])
    test_df = pd.merge(test_df, base_df, how='left', on=[feild_name])
    testDf = pd.merge(testDf, base_df, how='left', on=[feild_name])
    return train_df, test_df, testDf


class IDConfigs:
    def __init__(self, target_as_log, normalize, save_predictions_with_data, generate_submission):
        self.target_as_log = target_as_log
        self.normalize = normalize
        self.save_predictions_with_data = save_predictions_with_data
        self.generate_submission = generate_submission
        self.parmsFromNormalization = None
        self.verify_data = False


def generate_features(conf, df, subdf):
    use_slope = False
    use_group_aggrigate = True
    use_product_features = True

    use_agency_features = False


    #print "shapes train, test", df.shape, testDf.shape
    df = df[df['Producto_ID'] > 0]
    print "shapes train, test", df.shape
    df['unit_prize'] = df['Venta_hoy']/df['Venta_uni_hoy']

    training_set_size = int(0.7*df.shape[0])
    test_set_size = df.shape[0] - training_set_size

    y_all = df['Demanda_uni_equil'].values

    if conf.target_as_log:
        #then all values are done as logs
        df['Demanda_uni_equil'] = transfrom_to_log(df['Demanda_uni_equil'].values)

    train_df = df[:training_set_size]
    test_df = df[-1*test_set_size:]

    y_actual_train = y_all[:training_set_size]
    y_actual_test = y_all[-1*test_set_size:]

    testDf = subdf

    if use_slope:
        train_df, test_df, testDf = addSlopes(train_df, test_df, testDf)

    demand_val_mean = df['Demanda_uni_equil'].mean()
    #demand_val_mean = df['Demanda_uni_equil'].median()
    demand_val_stddev = df['Demanda_uni_equil'].std()
    #add mean and stddev by groups

    groups = ('Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID')
    measures = ('Dev_proxima', 'Dev_uni_proxima', 'Demanda_uni_equil', 'unit_prize', 'Venta_uni_hoy', 'Venta_hoy')
    #for t in itertools.product(groups, measures):
    #for t in [('Cliente_ID', 'Demanda_uni_equil'), ('Cliente_ID', 'Venta_uni_hoy'), ('Cliente_ID', 'Venta_hoy'),
    #    ('Ruta_SAK', 'unit_prize')]:
    #        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df,
    #                                test_df,t[0], testDf, drop=False, agr_feild=t[1])

    if use_group_aggrigate:
        #removed for agency state etc *train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Agencia_ID', testDf, drop=False)
        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Canal_ID', testDf, drop=False)
        #*train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Ruta_SAK', testDf, drop=False)
        #*train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Cliente_ID', testDf, drop=False) #duplicated
        train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Ruta_SAK', 'Cliente_ID'],
                                                          'Demanda_uni_equil', "clients_combined", demand_val_mean, demand_val_stddev)

        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, False, demand_val_mean, demand_val_stddev)

        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, drop=False, agr_feild='Venta_hoy')
        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Canal_ID', testDf, drop=False, agr_feild='Venta_hoy')
        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Ruta_SAK', testDf, drop=False, agr_feild='Venta_hoy')
        #*train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Cliente_ID', testDf, drop=False, agr_feild='Venta_hoy')
        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, drop=False, agr_feild='Venta_hoy')

        train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Ruta_SAK', 'Cliente_ID'],
                                                          'Venta_hoy', "clients_combined_vh", demand_val_mean, demand_val_stddev)


        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, drop=False,
                                                            agr_feild='Dev_proxima', do_count=False, do_stddev=False)
        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Canal_ID', testDf, drop=False, agr_feild='Dev_proxima')
        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Ruta_SAK', testDf, drop=False, agr_feild='Dev_proxima')
        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Cliente_ID', testDf, drop=False, agr_feild='Dev_proxima')
        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Agencia_ID', testDf, drop=False, agr_feild='Dev_proxima')


    if use_product_features:
        product_data_df = read_productdata_file('product_data.csv')
        #remove  unused feilds
        #product_data_df = drop_feilds_1df(product_data_df, ['has_vanilla','has_multigrain', 'has_choco', 'weight','pieces'])
        product_data_df = drop_feilds_1df(product_data_df, ['has_vanilla','has_multigrain', 'has_choco'])

        train_df = pd.merge(train_df, product_data_df, how='left', on=['Producto_ID'])
        test_df = pd.merge(test_df, product_data_df, how='left', on=['Producto_ID'])
        testDf = pd.merge(testDf, product_data_df, how='left', on=['Producto_ID'])

        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'brand_id', testDf, drop=False)
        train_df, test_df, testDf = drop_feilds(train_df, test_df, testDf, ['brand_id'])

        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'product_word', testDf, drop=False)
        train_df, test_df, testDf = drop_feilds(train_df, test_df, testDf, ['product_word'])

    if use_agency_features:
        agency_data_df = read_productdata_file('agency_data.csv')
        train_df, test_df, testDf =  merge_csv_by_feild(train_df, test_df, testDf, agency_data_df, 'Agencia_ID')

        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Town_id', testDf, drop=False)
        train_df, test_df, testDf = drop_feilds(train_df, test_df, testDf, ['Town_id'])

        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'State_id', testDf, drop=False)
        train_df, test_df, testDf = drop_feilds(train_df, test_df, testDf, ['State_id'])



    #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, drop=False, agr_feild='Dev_proxima')
    #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, drop=False, agr_feild='Venta_hoy')

    #train_df, test_df, testDf =  merge_clusters(train_df, test_df, testDf)
    #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Cluster', testDf, drop=False)
    #train_df, test_df, testDf = do_one_hot_all(train_df, test_df, testDf, ['Cluster'])

    #train_df, test_df, testDf = do_one_hot_all(train_df, test_df, testDf, ['Canal_ID'])

    #train_df, test_df, testDf = only_keep_top_categories(train_df, test_df,testDf, 'Producto_ID', 30)
    #train_df, test_df, testDf = do_one_hot_all(train_df, test_df, testDf, ['Producto_ID'])



    #train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Producto_ID', 'Agencia_ID'],
    #                                                      'Demanda_uni_equil', "agc_product", demand_val_mean, demand_val_stddev)

    #train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Canal_ID', 'Ruta_SAK', 'Cliente_ID'],
    #                                                      'Demanda_uni_equil', "routes_combined", demand_val_mean, demand_val_stddev)



    #test_df_before_dropping_features = test_df

    #train_df, test_df, testDf = do_one_hot_all(train_df, test_df, testDf, ['Agencia_ID', 'Cliente_ID'])

    train_df, test_df, testDf = drop_feilds(train_df, test_df, testDf, ['Canal_ID','Cliente_ID','Producto_ID', 'Agencia_ID', 'Ruta_SAK'])
    #train_df, test_df, testDf = drop_feilds(train_df, test_df, testDf, ['Canal_ID','Cliente_ID','Producto_ID', 'Ruta_SAK'])
    #train_df, test_df, testDf = drop_feilds(train_df, test_df, testDf, ['Canal_ID','Cliente_ID', 'Ruta_SAK'])
    #train_df, test_df, testDf = drop_feilds(train_df, test_df, testDf, ['Canal_ID','Cliente_ID','Producto_ID'])


    #TODO explore this more http://pandas.pydata.org/pandas-docs/stable/missing_data.html
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    #following are feilds in test data that is not used
    train_df, test_df = drop_column(train_df, test_df, 'Venta_uni_hoy')
    train_df, test_df = drop_column(train_df, test_df, 'Venta_hoy')
    train_df, test_df = drop_column(train_df, test_df, 'Dev_uni_proxima')
    train_df, test_df = drop_column(train_df, test_df, 'Dev_proxima')
    train_df, test_df = drop_column(train_df, test_df, 'unit_prize')

#    if conf.save_preprocessed_file:
#        df.to_csv(preprocessed_file_name, index=False)

    return train_df, test_df, testDf, y_actual_test


def do_forecast(conf, train_df, test_df, y_actual_test):
    if train_df.shape[1] != test_df.shape[1]:
        print "train and test does not match " + list(train_df) + " " + list(test_df)

    y_train_row = train_df['Demanda_uni_equil'].values
    y_test_row = test_df['Demanda_uni_equil'].values

    train_df, test_df = drop_column(train_df, test_df, 'Demanda_uni_equil')

    forecasting_feilds = list(train_df)
    print "Forecasting Feilds", [ "("+str(i)+")" + forecasting_feilds[i] for i in range(len(forecasting_feilds))]

    y_train, parmsFromNormalization = preprocess1DtoZeroMeanUnit(y_train_row)
    y_test = apply_zeroMeanUnit(y_test_row, parmsFromNormalization)

    X_train, parmsFromNormalization2D = preprocess2DtoZeroMeanUnit(train_df.values.copy())
    x_test_raw = test_df.values.copy()
    X_test = apply_zeroMeanUnit2D(x_test_raw, parmsFromNormalization2D)

    conf.parmsFromNormalization = parmsFromNormalization



    if X_train.shape[1] != X_test.shape[1]:
        print " columns not aligned X_train, y_train, X_test, y_test", X_train.shape, y_train.shape, X_test.shape, y_test.shape

    if X_train.shape[0] != y_train.shape[0] or y_test.shape[0] != X_test.shape[0]:
        print "rows not aligned X_train, y_train, X_test, y_test", X_train.shape, y_train.shape, X_test.shape, y_test.shape

    #print_xy_sample(X_train, y_train)
    #print_xy_sample(X_test, y_test)

    check4nan(X_train)

    de_normalized_forecasts = []

    #models = []
    models = [RFRModel(conf), DLModel(conf), LRModel(conf)]

    xgb_params = {"objective": "reg:linear", "booster":"gbtree", "max_depth":3, "eta":0.1, "min_child_weight":5,
            "subsample":0.5, "nthread":4, "colsample_bytree":0.5, "num_parallel_tree":1, 'gamma':0}
    for md  in [0]:
        #xgb_params['max_depth'] = md[0]
        #xgb_params['subsample'] = md[1]
        #xgb_params['min_child_weight'] = md[2]
        models.append(XGBoostModel(conf, xgb_params))


    for m in models:
        den_forecasted_data = m.fit(X_train, y_train, X_test, y_test, y_actual_test, forecasting_feilds=forecasting_feilds)
        de_normalized_forecasts.append(den_forecasted_data)

    if len(de_normalized_forecasts) > 1:
        avg_models(np.column_stack(de_normalized_forecasts), y_actual_test)



    #model = run_rfr(X_train, y_train, X_test, y_test, forecasting_feilds)
    #model = run_lr(X_train, y_train, X_test, y_test)
    #model = run_xgboost(X_train, y_train, X_test, y_test, forecasting_feilds=forecasting_feilds)
    #model, y_pred = regression_with_xgboost_no_cv(X_train, y_train, X_test, y_test, features=forecasting_feilds, num_rounds=20)

    #ntdepths, ntwidths, dropouts, reglur, lr, trialcount
    #configs = create_rondomsearch_configs4DL((1,2,3), (10, 20), (0.1, 0.2, 0.4),
    #                                       (0.1, 0.2, 0.3), (0.001, 0.0001), 10)
    #for c  in configs:
    #    c.epoch_count = 5
    #model = run_dl(X_train, y_train, X_test, y_test,c)
    #    print c.tostr() + "rmsle"
    #    y_pred_final = check_accuracy(c.tostr(), model, X_test, parmsFromNormalization, test_df, target_as_log, y_actual_test, command)

    #model = run_dl(X_train, y_train, X_test, y_test)
    #y_pred_final = check_accuracy("Linear Booster", model, X_test, parmsFromNormalization, test_df, conf.target_as_log,
    #                             y_actual_test, conf.command)


    '''
    for t in itertools.product((3, 5, 10), (0,0.1,0.2), (0,0.1,0.2)):
        xgb_params = {"objective": "reg:linear", "booster":"gblinear", "max_depth":t[0], "lambda":t[1], "lambda_bias":t[2], "alpha":0, "nthread":4}
        print t, xgb_params
        model, y_pred = regression_with_xgboost(X_train, y_train, X_test, y_test, features=forecasting_feilds, xgb_params=xgb_params)
        y_pred_final = check_accuracy("Linear Booster" + str(t), model, X_test, parmsFromNormalization, test_df, target_as_log, y_actual_test, command)
    '''
    #xgboost(data = X, booster = "gbtree", objective = "binary:logistic", max.depth = 5, eta = 0.5, nthread = 2, nround = 2,  min_child_weight = 1, subsample = 0.5, colsample_bytree = 1,num_parallel_tree = 1)

    #xgb_params['subsample'] = 0.5 #0.5-1, Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting.
    #xgb_params['min_child_weight'] = 3 #      #Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
    #xgb_params['max_depth'] = 3 #Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.

    '''
    xgb_params = {"objective": "reg:linear", "booster":"gbtree", "max_depth":3, "eta":0.1, "min_child_weight":5,
            "subsample":0.5, "nthread":4, "colsample_bytree":0.5, "num_parallel_tree":1, 'gamma':0}
    #for md  in [(3, 0.5, 5), (5, 0.5, 10), (8, 0.7, 10)]:
    #for md  in [(0.5, 0.5)]:
    #for md  in [1, 5, 10]:
    for md  in [0]:
        #xgb_params['max_depth'] = md[0]
        #xgb_params['subsample'] = md[1]
        #xgb_params['min_child_weight'] = md[2]

        #xgb_params['max_delta_step'] = 1

        print md, xgb_params
        #model, y_pred = regression_with_xgboost(X_train, y_train, X_test, y_test, features=forecasting_feilds, xgb_params=xgb_params)
        model, y_pred = regression_with_xgboost_no_cv(X_train, y_train, X_test, y_test, features=forecasting_feilds,
                                                      xgb_params=xgb_params,num_rounds=200)
        y_pred_final = check_accuracy("Linear Booster " + str(md), model, X_test, parmsFromNormalization,
                                      conf.target_as_log, y_actual_test, conf.command)
    '''


    if conf.verify_data:
        corrected_Y_test = modeloutput2predictions(y_test, parmsFromNormalization=parmsFromNormalization)
        if conf.target_as_log:
            corrected_Y_test = retransfrom_from_log(corrected_Y_test)
        print "Undo normalization test passed", np.allclose(corrected_Y_test, y_actual_test, atol=0.01)

        print "Undo X data test test passed", \
            np.allclose(x_test_raw, undo_zeroMeanUnit2D(X_test, parmsFromNormalization2D), atol=0.01)

    #find best model
    best_model = None
    best_error = 100000
    for m in models:
        if m.rmsle < best_error:
            best_error = m.rmsle
            best_model = m

    return best_model, parmsFromNormalization, parmsFromNormalization2D
    #return model, parmsFromNormalization, parmsFromNormalization2D


def calculate_accuracy(label, y_actual_test, y_forecast):
    error_ac, rmsep, mape, rmse = almost_correct_based_accuracy(y_actual_test, y_forecast, 10)
    rmsle = calculate_rmsle(y_actual_test, y_forecast)
    print ">> %s AC_errorRate=%.1f RMSEP=%.6f MAPE=%6f RMSE=%6f rmsle=%.5f" %("Run " + label, error_ac, rmsep, mape, rmse, rmsle)


def avg_models(forecasts, y_actual):
    median_forecast = np.median(forecasts, axis=1)
    calculate_accuracy("median_forecast", y_actual, median_forecast)

    hmean_forecast = scipy.stats.hmean(forecasts, axis=1)
    calculate_accuracy("hmean_forecast", y_actual, hmean_forecast)


def create_submission(conf, model, testDf, parmsFromNormalization, parmsFromNormalization2D ):
    ids = testDf['id']
    temp = testDf.drop('id',1)
    print "creating submission for ", len(ids), "values"

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

    if conf.target_as_log:
        kaggale_predicted = retransfrom_from_log(kaggale_predicted)

    print "log retransform", kaggale_test.shape

    to_save = np.column_stack((ids, kaggale_predicted))
    to_saveDf =  pd.DataFrame(to_save, columns=["id","Demanda_uni_equil"])
    to_saveDf = to_saveDf.fillna(0)
    to_saveDf["id"] = to_saveDf["id"].astype(int)
    to_saveDf.to_csv('submission'+str(conf.command)+ '.csv', index=False)

    print "Submission done for ", to_saveDf.shape[0], "values"

    #to_saveDf["groupedMeans"] = testDf["groupedMeans"]
    #to_saveDf["groupedStd"] = testDf["groupedStd"]
    #to_saveDf["Slopes"] = testDf["Slopes"]
    #to_saveDf.to_csv('prediction_detailed.csv', index=False)

    #np.savetxt('submission.csv', to_save, delimiter=',', header="id,Demanda_uni_equil", fmt='%d')   # X is an array









