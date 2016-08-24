import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor

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
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from mltools import *
import scipy

import resource

from pympler import asizeof
import objgraph
import os
import pickle
import re


def read_datafiles(command, test_run=True):
    data_files = [
        ["trainitems0_5000.csv", 0, 5000, "test_0_5000.csv"], #1.4G, 0
        ["trainitems5_10_35_40_45_50k.csv", 5000, 10000, "test_5_10_35_40_45_50k.csv"], #534M, 1
        ["trainitems30000_35000.csv", 30000, 35000, "test_30000_35000.csv"], #559M, 2
        #["trainitems30000_35000.csv", 30000, 35000, "trainitems5_10_35_40_45_50k.csv"], #559M # to remove ** pass #1 as #2 test
        ["trainitems40000_45000.csv", 40000, 45000, "test_40000_45000.csv"], #640M, 2
        ["trainitems5000_15000.csv", -1, -1, "test0_100.csv"], #4
        ["train-rsample-10m.csv", -1, -1, "test0_100.csv"], #5
        ["train-rsample-500k.csv", -1, -1, "test0_100.csv"], #6
        ["train-rsample-15m.csv", -1, -1, "test.csv"], #7
        ["train-rsample-10k.csv", -1, -1, "test0_100.csv"] #8
    ]

    if command == -2:
        df = read_train_file('data/train.csv')
        subdf = pd.read_csv('data/test.csv')
    elif command == -1:
        df = read_train_file('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems300.csv')
        testDf = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/test.csv')
        subdf = testDf[(testDf['Producto_ID'] <= 300)]
    else:
        if test_run:
            dir = "/Users/srinath/playground/data-science/BimboInventoryDemand/"
        else:
            dir = "data/"

        df = read_train_file(dir + data_files[command][0])
        subdf = pd.read_csv(dir +data_files[command][3])
        print "using ", dir + data_files[command][0], " and ", dir +data_files[command][3]
        print "testDf read", subdf.shape

    return df, subdf



def print_mem_usage(label=""):
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000000
    print label, ":mem_usage", mem_mb, "mb"
    #usage=resource.getrusage(resource.RUSAGE_SELF)
    #print '''%s: usertime=%s systime=%s mem=%s gb
    #       '''%(point,usage[0],usage[1],
    #            (float(usage[2]*resource.getpagesize()))/1024*1024*1024 )


def object_size(obj):
    return asizeof.asizeof(obj)/1024*1024


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


def fillna_and_inf(data, na_r=0, inf_r=100000):
    return np.where(np.isnan(data), na_r, np.where(np.isinf(data), inf_r, data))


def fillna_if_feildexists(df, feild_name, replacement=-1):
    if feild_name in df:
        if replacement != -1:
            df[feild_name].fillna(replacement, inplace=True)
        else:
            df[feild_name].fillna(df[feild_name].mean(), inplace=True)
    return df


def basic_stats_as_str(data):
    stats = pd.Series(data).describe(percentiles=[0.5, .25, .5, .75, 0.95])
    start_as_str = str(stats).replace('\n', ',')
    start_as_str = re.sub('\s+', '=', start_as_str)
    return ">>"+start_as_str


def print_time_took(start_time, label, silient_on_small=False):
    time_took = time.time() - start_time
    if silient_on_small:
        if time_took > 10:
            print label, "took", time_took, "s"
    else:
        print label, "took", time_took, "s"


def extract_regx(p, str):
    p1 = re.compile(p)
    for match in p1.finditer(str):
        return match.group(0)


def extract_regx_grps(p, str, groups):
    p1 = re.compile(p)
    for match in p1.finditer(str):
        return [ match.group(g) for g in groups]


def sample_train_dataset(X_train, y_train, X_test, y_test, maxentries=5000000):
    x_size = X_train.shape[0]
    sample_indexes_train = np.random.randint(0, X_train.shape[0], min(maxentries, x_size))
    X_train = X_train[sample_indexes_train].copy()
    y_train = y_train[sample_indexes_train].copy()

    sample_indexes_test = np.random.randint(0, X_test.shape[0], min(maxentries, x_size))
    X_test = X_test[sample_indexes_test].copy()
    y_test = y_test[sample_indexes_test].copy()
    return X_train, y_train, X_test, y_test
'''
We can switch to XGboost files later if needed
http://xgboost.readthedocs.io/en/latest/python/python_intro.html
'''

def save_file(model_type, command, df, name, metadata=None):
    if not os.path.exists(model_type):
        os.makedirs(model_type)

    submission_file = model_type+ '/' + name+str(command)+ '.csv'
    df.to_csv(submission_file, index=False)

    if metadata is not None:
        metadata_file = model_type+ '/' + name+str(command)+ '.pickle'
        file = open(metadata_file, 'wb')
        pickle.dump(metadata, file)
    feature_list = list(df)
    print "saved", submission_file, len(feature_list), " feilds=", feature_list


def load_file(model_type, command, name, throw_error=True, fields=None):
    submission_file = model_type+ '/' + name+str(command)+ '.csv'
    if not throw_error:
        if not os.path.exists(submission_file):
            return None
    print "loading", submission_file
    if fields is not None:
        return pd.read_csv(submission_file, usecols=fields)
    else:
        return  pd.read_csv(submission_file)


def load_file_with_metadata(model_type, command, name):
    load_df = load_file(model_type, command, name)

    hardcode = False

    if hardcode:
        metadata = {'rmsle':[0.82079594626169383, 0.8203164757712077, 0.83624063520333292, 0.63268]}
    else:
        metadata_file = model_type+ '/' + name+str(command)+ '.pickle'
        file = open(metadata_file, 'rb')
        metadata = pickle.load(file)
    return load_df, metadata


def save_submission_file(submission_file, ids, submissions):
    uids = np.unique(ids)
    if int(uids.shape[0]) != int(ids.shape[0]):
        print uids.shape, ids.shape, uids.shape[0], ids.shape[0]
        raise ValueError('submission ids are not unique')
    start = time.time()
    to_save = np.column_stack((ids, submissions))
    to_saveDf =  pd.DataFrame(to_save, columns=["id","Demanda_uni_equil"])
    to_saveDf = to_saveDf.fillna(0)
    to_saveDf["id"] = to_saveDf["id"].astype(int)
    to_saveDf.to_csv(submission_file, index=False)

    print "## Submissions details ##"
    print to_saveDf.describe()

    print "Submission done for ", to_saveDf.shape[0], "file", submission_file
    print_time_took(start, "create_submission took ")


def save_train_data(model_type, command, train_df, test_df, sub_df, y_train, y_test):
    save_file(model_type, command,train_df, 'train')
    save_file(model_type, command,test_df, 'test')
    save_file(model_type, command,sub_df, 'sub')

    ytrain_df =  pd.DataFrame(y_train.reshape(-1,1), columns=["target"])
    save_file(model_type, command, ytrain_df, 'y_train')

    ytest_df =  pd.DataFrame(y_test.reshape(-1,1), columns=["target"])
    save_file(model_type, command, ytest_df, 'y_test')


def load_train_data(model_type, command, throw_error=False, fields=None, load_sub_id=True):
    train_df = load_file(model_type, command,'train', throw_error=throw_error, fields=fields)
    test_df = load_file(model_type, command, 'test', throw_error=throw_error, fields=fields)
    if load_sub_id:
        if fields is None:
            fields = ['id']
        else:
            fields = fields + ['id']
    sub_df = load_file(model_type, command, 'sub', throw_error=throw_error, fields=fields)

    ytrain_df = load_file(model_type, command, 'y_train', throw_error=throw_error)
    ytest_df = load_file(model_type, command, 'y_test', throw_error=throw_error)

    if ytest_df is not None and ytest_df is not None:
        y_train = ytrain_df['target'].values
        y_test = ytest_df['target'].values
        if train_df.shape[0] != y_train.shape[0] or y_test.shape[0] != test_df.shape[0]:
            raise ValueError("columns not aligned X_train, y_train, X_test, y_test " + train_df.shape + " "+ y_train.shape
                         + " " + test_df.shape + " " +y_test.shape)

        return train_df, test_df, sub_df, y_train, y_test
    else:
        if throw_error:
            raise ValueError("cannot find ytrain and ytest data in store")
        else:
            return train_df, test_df, sub_df, None, None


def find_alt_for_missing(to_merge, seen_with_stats):
    print "feilds 0", list(to_merge), list(seen_with_stats)
    seen_without_stat = seen_with_stats[['Ruta_SAK', 'Cliente_ID']]
    seen_without_stat = seen_without_stat.copy() #TODO might need to improve and do without copy
    seen_without_stat['has_seen'] = 1
    merged = pd.merge(to_merge, seen_without_stat, how='left', on=['Ruta_SAK', 'Cliente_ID'])
    merged = merged.fillna(0)
    #merged['id'] = range(len(merged))

    #workdf = merged[['has_seen', 'Ruta_SAK', 'Cliente_ID', 'id']]

    similar_clients_df = pd.read_csv('similar_client_data.csv');
    similar_clients_df = drop_feilds_1df(similar_clients_df, ['mean_sales'])

    merged_with_order = pd.merge(merged, similar_clients_df, how='left', on=['Ruta_SAK', 'Cliente_ID'])
    merged_with_order = merged_with_order.sort_values(by=['index'])
    merged_with_order = drop_feilds_1df(merged_with_order, ['index'])

    merged_with_order['Ruta_SAK_p'] = merged_with_order['Ruta_SAK']
    merged_with_order['Cliente_ID_p'] = merged_with_order['Cliente_ID']

    print "feilds 1", list(merged_with_order) #['Ruta_SAK', 'Cliente_ID', 'has_seen', 'Ruta_SAK_p', 'Cliente_ID_p']
    merged_with_order_np = merged_with_order.values

    #now we find replacements

    #first find one seen value
    seen_index = -1
    for i in range(len(merged_with_order_np)):
        if merged_with_order_np[i][2] == 1:
            seen_index = i
    #walk though and replace with old seen value
    for i in range(len(merged_with_order_np)):
        if merged_with_order_np[i][2] == 0:
            merged_with_order_np[i][0] = merged_with_order_np[seen_index][0]
            merged_with_order_np[i][1] = merged_with_order_np[seen_index][1]
        else:
            seen_index = i

    merged_with_alt =  pd.DataFrame(merged_with_order_np, columns=['Ruta_SAK', 'Cliente_ID', 'has_seen', 'Ruta_SAK_p', 'Cliente_ID_p'])

    merged_withstats = pd.merge(merged_with_alt, seen_with_stats, how='left', on=['Ruta_SAK', 'Cliente_ID'])
    merged_withstats = drop_feilds_1df(merged_withstats, ['has_seen', 'Ruta_SAK', 'Cliente_ID'])
    merged_withstats.rename(columns = {'Ruta_SAK_p':'Ruta_SAK'}, inplace=True)
    merged_withstats.rename(columns = {'Cliente_ID_p':'Cliente_ID'}, inplace=True)

    #make sure no NA's in the list
    print "find replacement done", seen_with_stats.shape, "->", merged_withstats.shape, " feilds", list(merged_withstats)

    #make sure all values are contained
    missing_ids_count = setdiff_counts_froms_dfs(to_merge, merged_withstats[['Ruta_SAK','Cliente_ID']])
    if missing_ids_count > 0:
        raise "missing from target ", missing_ids_count
    missing_ids_count = setdiff_counts_froms_dfs(seen_with_stats[['Ruta_SAK','Cliente_ID']], merged_withstats[['Ruta_SAK','Cliente_ID']])
    if missing_ids_count > 0:
        raise "missing from seen ", missing_ids_count

    return merged_withstats


def setdiff_counts_froms_dfs(df1, df2):
    #missing_ids = pd.Index(to_merge).difference(pd.Index(merged_withstats[['Ruta_SAK','Cliente_ID']]))
    ds1 = set([ tuple(line) for line in df1.values.tolist()])
    ds2 = set([ tuple(line) for line in df2.values.tolist()])
    ds1.difference(ds2)


def calculate_ci(std, N):
    #based on N size, it is a normal or stddev
    return np.where(N < 30, 1.645*np.divide(std ,np.sqrt(N)), 2.920*np.divide(std ,np.sqrt(N)))


def join_multiple_feild_stats(bdf, testdf, subdf, feild_names, agr_feild, name, default_stats, fops):
    start = time.time()
    groupData = bdf.groupby(feild_names)[agr_feild]

    meanData = groupData.mean()


    #TODO check why data is NA
    valuesDf = meanData.to_frame(name+"_Mean")
    valuesDf.fillna(meanData.mean(), inplace=True)
    valuesDf.reset_index(inplace=True)

    stddevData = groupData.std()
    if fops.stddev:
        valuesDf[name+"_StdDev"] = stddevData.values
        valuesDf.fillna(10000, inplace=True)

    countData = groupData.count()
    if fops.count:
        valuesDf[name+"_Count"] = countData.values
        valuesDf.fillna(0, inplace=True)
    if fops.sum:
        sumData = groupData.sum()
        valuesDf[name+"_sum"] = sumData.values
    start2_start = time.time()
    print "join_multiple_feild_stats: base stats took", (start2_start - start)
    print "start complex stat", (time.time() - start)
    if fops.p10:
        pcerntile10 = groupData.quantile(0.1, interpolation='nearest')
        valuesDf[name+"_pcerntile10"] = np.where(np.isnan(pcerntile10), 0, pcerntile10)
    print "took p10", (time.time() - start)
    if fops.p90:
        pcerntile90 = groupData.quantile(0.9, interpolation='nearest')
        valuesDf[name + "_pcerntile90"] = np.where(np.isnan(pcerntile90), 0, pcerntile90)
    print "took p90", (time.time() - start)
    if fops.kurtosis:
        kurtosis = groupData.apply(lambda x: min(scipy.stats.kurtosis(x), 10000))
        valuesDf[name+"_kurtosis"] = fillna_and_inf(kurtosis)
    print "took kurtosis", (time.time() - start)
    if fops.hmean:
        hmean = groupData.apply(calcuate_hmean)
        valuesDf[name+"_hMean"] = fillna_and_inf(hmean)
    print "took hmean", (time.time() - start)
    if fops.entropy:
        entropy = groupData.apply(lambda x: min(scipy.stats.entropy(x), 10000))
        valuesDf[name+"_entropy"] = np.where(np.isnan(entropy), 0, np.where(np.isinf(entropy), 10, entropy))
    if fops.ci:
        valuesDf[name+"ci"] = calculate_ci(stddevData,countData)
    if fops.median:
        median = groupData.median()
        valuesDf[name +"_median"] = fillna_and_inf(median)

    #valuesDf = calculate_group_stats(groupData, name, default_stats, fops)
    print "took entropy", (time.time() - start)
    if fops.use_close_products_missing and feild_names[0] == 'Ruta_SAK' and feild_names[1] == 'Cliente_ID':
        to_merge = pd.concat([testdf[['Ruta_SAK','Cliente_ID']], subdf[['Ruta_SAK','Cliente_ID']]])
        to_merge = to_merge.drop_duplicates()
        valuesDf = find_alt_for_missing(to_merge, valuesDf)
        print "Using close values for missing values"

    merge_start = time.time()
    print "join_multiple_feild_stats: complex stats took", (merge_start - start2_start)
    bdf = merge__multiple_feilds_stats_with_df(name, bdf, valuesDf, feild_names, default_stats)
    testdf = merge__multiple_feilds_stats_with_df(name, testdf, valuesDf, feild_names, default_stats)
    subdf = merge__multiple_feilds_stats_with_df(name, subdf, valuesDf, feild_names, default_stats)
    print "join_multiple_feild_stats: merge took", (time.time() - merge_start)

    print "join_multiple_feild_stats", str(feild_names), " took ", (time.time() - start), "s"

    return bdf, testdf, subdf


def find_NA_rows_percent(df_check, label):
    start = time.time()
    all_rows = df_check.shape[0]
    na_rows = df_check[df_check.isnull().any(axis=1)]
    na_rows_count = na_rows.shape[0]
    na_percent = float(na_rows_count)/all_rows
    if na_rows_count > 0:
        na_cols = df_check.isnull().sum(axis=0)
        column_names = list(df_check)
        na_colmuns = []
        na_colmuns_actual = []
        for i in range(len(column_names)):
            if na_cols[i] > 0:
                na_colmuns.append(column_names[i] + "=" + str(na_cols[i]))
                na_colmuns_actual.append(column_names[i])
        print "NA in ", label, "count=", na_rows_count, "(", na_percent, ")", na_colmuns
        #print na_rows[na_colmuns_actual].sample(10)
        #print df_check.sample(10)

    for f in list(df_check):
        X = df_check[f]
        if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
            #raise ValueError("feild" + f + "Input contains NaN, infinity"
            #             " or a value too large for %r." % X.dtype)
            error = "feild " + f + " contains " + "np.inf=" + str(np.where(np.isnan(X))) \
                    + "is.inf=" + str(np.where(np.isinf(X))) + "np.max=" + str(np.max(abs(X)))
            print error, "shape", "stats", X.shape, pd.Series(X).describe()
            raise ValueError(error)
    print "find_NA_rows_percent took", (time.time() - start), "s"
    return na_percent


def merge__multiple_feilds_stats_with_df(name, bdf, stat_df, feild_names, default_stats):
    merged = pd.merge(bdf, stat_df, how='left', on=feild_names)
    merged = fillna_if_feildexists(merged, name+"_Mean")
    merged = fillna_if_feildexists(merged, name+"_StdDev", 10000)
    merged = fillna_if_feildexists(merged, name+"_Count", 0)
    #replace rest with zero
    merged.fillna(0, inplace=True)
    return merged

def calcuate_hmean(group):
    x = group.values
    non_zero_x = np.where(x > 0)
    if len(non_zero_x) > 0:
        #print non_zero_x
        return min(scipy.stats.hmean(x[non_zero_x]), 100000)
    else:
        return 0


def calculate_group_stats(grouped_data, name, default_stats, fops):
    meanData = grouped_data.mean()
    valuesDf = meanData.to_frame(name +"_Mean")
    valuesDf.reset_index(inplace=True)

    if fops.sum:
        valuesDf[name +"_sum"] = grouped_data.sum()

    stddevData = grouped_data.std()
    if fops.stddev or fops.count:
        valuesDf[ name +"_StdDev"] = stddevData.values
        valuesDf.fillna(10000, inplace=True)

    if fops.ci or fops.count:
        countData = grouped_data.count()
    if fops.count:
        valuesDf[name + "_Count"] = countData.values
        valuesDf.fillna(0, inplace=True)
    if fops.ci:
        valuesDf[name+"ci"] = calculate_ci(stddevData.values,countData.values)

    if fops.p10:
        pcerntile10 = grouped_data.quantile(0.1, interpolation='nearest')
        valuesDf[name + "_pcerntile10"] = np.where(np.isnan(pcerntile10), 0, pcerntile10)
    if fops.p90:
        pcerntile90 = grouped_data.quantile(0.9, interpolation='nearest')
        valuesDf[name + "_pcerntile90"] = np.where(np.isnan(pcerntile90), 0, pcerntile90)
    if fops.kurtosis:
        kurtosis = grouped_data.apply(lambda x: min(scipy.stats.kurtosis(x), 10000))
        valuesDf[name + "_kurtosis"] = fillna_and_inf(kurtosis)
    if fops.hmean:
        hmean = grouped_data.apply(calcuate_hmean)
        valuesDf[name + "_hMean"] = np.where(np.isnan(hmean), 0, hmean)
    if fops.entropy:
        entropy = grouped_data.apply(lambda x: min(scipy.stats.entropy(x), 10000))
        valuesDf[name +"_entropy"] =  np.where(np.isnan(entropy), 0, np.where(np.isinf(entropy), 10, entropy))
    if fops.median:
        median = grouped_data.median()
        valuesDf[name +"_median"] = fillna_and_inf(median)


    valuesDf.fillna(default_stats.mean, inplace=True)
    return valuesDf



def calculate_feild_stats(bdf, feild_name, agr_feild, default_stats, fops):
    groupData = bdf.groupby([feild_name])[agr_feild]
    return calculate_group_stats(groupData, feild_name+"_"+agr_feild, default_stats, fops)


def merge_stats_with_df(bdf, stat_df, feild_name, default_mean=None, default_stddev=None, agr_feild='Demanda_uni_equil'):
    merged = pd.merge(bdf, stat_df, how='left', on=[feild_name])

    if default_mean is not None:
        merged[feild_name+"_"+agr_feild+"_Mean"].fillna(default_mean, inplace=True)
    if default_stddev is not None:
        merged[feild_name+"_"+agr_feild++"_StdDev"].fillna(default_stddev, inplace=True)

    merged.fillna(0, inplace=True)
    return merged


def addFeildStatsAsFeatures(train_df, test_df, feild_name, testDf, default_stats,fops, drop=False,
                            agr_feild='Demanda_uni_equil'):
    start = time.time()
    valuesDf = calculate_feild_stats(train_df, feild_name, agr_feild, default_stats, fops)

    calculate_ts = time.time()
    train_df_m = pd.merge(train_df, valuesDf, how='left', on=[feild_name])
    test_df_m = pd.merge(test_df, valuesDf, how='left', on=[feild_name])

    train_df_m.fillna(0, inplace=True)
    test_df_m.fillna(0, inplace=True)

    if testDf is not None:
        name = feild_name+"_"+agr_feild
        testDf = pd.merge(testDf, valuesDf, how='left', on=[feild_name])
        testDf[name + "_Mean"].fillna(testDf[name + "_Mean"].mean(), inplace=True)
        if fops.stddev:
            testDf[name + "_StdDev"].fillna(10000, inplace=True)
        if fops.count:
            testDf[name + "_Count"].fillna(0, inplace=True)
        if drop:
            testDf = testDf.drop(feild_name,1)
        testDf.fillna(0, inplace=True)

    print "addFeildStatsAsFeatures() "+ feild_name+ " took %f (%f, %f), size %s %f" %(time.time()-start, calculate_ts- start,
                                           time.time() - calculate_ts, feild_name, valuesDf.shape[0])
    return train_df_m, test_df_m, testDf


#TODO fix defaults

def modeloutput2predictions(model_forecast, parmsFromNormalization):
    y_pred_corrected = undoPreprocessing(model_forecast, parmsFromNormalization)
#    zero_frecasts = np.where(y_pred_corrected < 0)[0]
#    print "zero forecasts=%d, %f precent" %(zero_frecasts.shape[0], float(zero_frecasts.shape[0])/y_pred_corrected.shape[0])
#    if not negative_allowed:
#        for i in range(y_pred_corrected.shape[0]):
#            if y_pred_corrected[i] < 0:
#                y_pred_corrected[i] = 1
    return y_pred_corrected


def drop_feilds(train_df, test_df, testDf, feilds):
    train_df = train_df.drop(feilds, axis=1)
    test_df = test_df.drop(feilds, axis=1)
    if testDf is not None:
        testDf = testDf.drop(feilds, axis=1)

    return train_df, test_df, testDf


def drop_feilds_1df(df, feilds):
    return df.drop(feilds, axis=1)

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
    #dfp["Simple_Forecast"] = dfp['Last_Sale'] + dfp["Mean_Slope"]*dfp["Time_Since_Last_Sale_Value"]
    #dfp = drop_feilds_1df(dfp, ["Last_Sale_Week"])
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

def calculate_last_sale_and_week(group):
    sales = np.array(group['Demanda_uni_equil'].values)
    samana = group['Semana'].values
    max_index = np.argmax(samana)
    return sales[max_index], samana[max_index]


def calculate_delivery_probability_this_week(base_df):
    timebtdlv = base_df['pg_time_between_delivery_mean']
    this_week = base_df['Semana']
    start = np.mod((this_week - base_df['Last_Sale_Week']),timebtdlv)

    delivery_probability_this_week = np.where(this_week - start < timebtdlv/2,
            1 - 2*(this_week - start)/timebtdlv,
            2*(this_week - start)/timebtdlv - 1)

    base_df['delivery_probability_this_week'] = delivery_probability_this_week
    base_df['time_since_last_sale_cycle'] = start
    return base_df


def time_between_delivery(group):
    deliveries = np.sort(group.values)
    if len(deliveries) == 0:
        return [100, 0]
    if len(deliveries) == 1:
        return [8, 1]
    if deliveries[0] != 3:
        deliveries = [3] + deliveries
    if deliveries[-1] != 7:
        deliveries = deliveries + [8]

    time_between_delivery = [deliveries[i] - deliveries[i-1] for i in range(1,len(deliveries))]
    return [np.mean(time_between_delivery), np.std(time_between_delivery)]


def expand_array_feild_and_add_df(tdf, composite_feild, feilds):
    npa = np.vstack(tdf[composite_feild].values)

    #valuesDf['mean'] = npa[:,1]
    index = 0
    for f in feilds:
        tdf[f] = npa[:,index]
        index = index +1

    tdf = drop_feilds_1df(tdf, [composite_feild])

    return tdf

def add_time_bwt_delivery(train_df, test_df, testDf):
    start_ts = time.time()
    df1 = train_df[train_df['Venta_uni_hoy'] > 0]
    grouped = df1.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])['Semana']
    tbdelivery = grouped.apply(time_between_delivery)

    tbdelivery_df = tbdelivery.to_frame("delivery_data")
    tbdelivery_df.reset_index(inplace=True)
    tbdelivery_df = expand_array_feild_and_add_df(tbdelivery_df, 'delivery_data',
                                                  ["pg_time_between_delivery_mean", "pg_time_between_delivery_stddev"])

    train_df_m = pd.merge(train_df, tbdelivery_df, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
    train_df_m.fillna(0, inplace=True)
    test_df_m = pd.merge(test_df, tbdelivery_df, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
    test_df_m.fillna(0, inplace=True)

    if testDf is not None:
        testDf = pd.merge(testDf, tbdelivery_df, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
        testDf.fillna(0, inplace=True)
    print "add_time_bwt_delivery took ", (time.time() - start_ts)
    return train_df_m, test_df_m, testDf


def add_last_sale_and_week(train_df, test_df, testDf):
    start_ts = time.time()

    #we first remove any entry that has only returns
    sales_df = train_df[train_df['Demanda_uni_equil'] > 0]
    grouped = sales_df.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])

    #following code if we need multiple features
    #slope_data_df = grouped.apply(calculate_last_sale_and_week)
    #sales_data_df = slope_data_df.to_frame("sales_data")
    #sales_data_df.reset_index(inplace=True)
    #valuesDf = expand_array_feild_and_add_df(sales_data_df, 'sales_data', ["Last_Sale", "Last_Sale_Week"])

    #this is to one feature
    slope_data_df = grouped['Semana'].max()
    sales_data_df = slope_data_df.to_frame("Last_Sale_Week")
    sales_data_df.reset_index(inplace=True)
    valuesDf = sales_data_df

    #now we merge the data
    sale_data_aggr_time = time.time()
    train_df_m = pd.merge(train_df, valuesDf, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
    train_df_m.fillna(0, inplace=True)
    test_df_m = pd.merge(test_df, valuesDf, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
    test_df_m.fillna(0, inplace=True)

    if testDf is not None:
        testDf = pd.merge(testDf, valuesDf, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
        testDf.fillna(0, inplace=True)

    train_df_m = calculate_delivery_probability_this_week(train_df_m)
    test_df_m = calculate_delivery_probability_this_week(test_df_m)
    testDf = calculate_delivery_probability_this_week(testDf)

    slopes_time = time.time()
    print "Add Sales Data took %f (%f, %f)" %(slopes_time - start_ts, sale_data_aggr_time-start_ts, slopes_time-sale_data_aggr_time)
    return train_df_m, test_df_m, testDf


class RFRModel:
    def __init__(self, conf, model=None):
        self.conf = conf
        self.name = "RFR"
        if model is None:
            self.model = RandomForestRegressor(n_jobs=4)
        else:
            self.model = model

    def fit(self, X_train, y_train, X_test, y_test, y_actual, forecasting_feilds=None):
        start = time.time()
        self.model.fit(X_train, y_train)
        print_feature_importance(self.model.feature_importances_, forecasting_feilds)
        #save model
        #joblib.dump(rfr, 'model.pkl')
        y_pred_final, rmsle = check_accuracy("RFR "+ str(self.model), self.model, X_test, self.conf.parmsFromNormalization,
                                      self.conf.target_as_log, y_actual, self.conf.command)
        self.rmsle =  rmsle
        print "RFR model took", (time.time() - start), "s"
        return y_pred_final

    def predict(self, X_test):
        return self.model.predict(X_test)

    def cleanup(self):
        self.model = None





'''
http://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/

max_features: - These are the maximum number of features Random Forest is allowed to try in individual tree. ncreasing
max_features generally improves the performance of the model as at each node now we have a higher number of options
to be considered. However, this is not necessarily true as this decreases the diversity of individual tree which is the
USP of random forest. But, for sure, you decrease the speed of algorithm by increasing the max_features. Hence,
you need to strike the right balance and choose the optimal max_features.
n_estimators :This is the number of trees you want to build before taking the maximum voting or averages of predictions.
Higher number of trees give you better performance but makes your code slower.
min_sample_leaf : If you have built a decision tree before, you can appreciate the importance of minimum sample leaf size.
Leaf is the end node of a decision tree. A smaller leaf makes the model more prone to capturing noise in train data.
oob_score : This is a random forest cross validation method. It is very similar to leave one out validation technique,
however, this is so much faster. This method simply tags every observation used in different tress. And then it finds
out a maximum vote score for every observation based on only trees which did not use this particular observation to train itself.
'''


class ETRModel:
    def __init__(self, conf, model=None):
        self.conf = conf
        self.name = "ETR"
        if model is None:
            self.model = ExtraTreesRegressor(n_jobs=4)
        else:
            self.model = model

    def fit(self, X_train, y_train, X_test, y_test, y_actual, forecasting_feilds=None):
        start = time.time()
        self.model.fit(X_train, y_train)
        #print_feature_importance(self.model.feature_importances_, forecasting_feilds)
        #save model
        #joblib.dump(rfr, 'model.pkl')
        y_pred_final, rmsle = check_accuracy("ETR "+ str(self.model), self.model, X_test, self.conf.parmsFromNormalization,
                                      self.conf.target_as_log, y_actual, self.conf.command)
        self.rmsle =  rmsle
        print "ETR model took", (time.time() - start), "s"
        return y_pred_final

    def predict(self, X_test):
        return self.model.predict(X_test)

    def cleanup(self):
        self.model = None



class AdaBoostRModel:
    def __init__(self, conf, model=None):
        self.conf = conf
        self.name = "AdaBoostR"
        if model is None:
            self.model = AdaBoostRegressor(loss='square')
        else:
            self.model = model

    def fit(self, X_train, y_train, X_test, y_test, y_actual, forecasting_feilds=None):
        start = time.time()
        self.model.fit(X_train, y_train)
        #print_feature_importance(self.model.feature_importances_, forecasting_feilds)
        #save model
        #joblib.dump(rfr, 'model.pkl')
        y_pred_final, rmsle = check_accuracy("AdaBoostR "+ str(self.model), self.model, X_test, self.conf.parmsFromNormalization,
                                      self.conf.target_as_log, y_actual, self.conf.command)
        self.rmsle =  rmsle
        print "AdaBoostR model took", (time.time() - start), "s"
        return y_pred_final

    def predict(self, X_test):
        return self.model.predict(X_test)

    def cleanup(self):
        self.model = None



def create_rfr_params(n_estimator=[200], oob_score=[False], max_features=["auto"],
                          min_samples_leaf=[50]):
    rfr_models =[]
    for t in itertools.product(n_estimator, oob_score, max_features, min_samples_leaf):
        rfr = RandomForestRegressor(n_jobs=4, n_estimators=t[0], oob_score=t[1], max_features=t[2],
                                    min_samples_leaf=t[3])
        rfr_models.append(rfr)
    return rfr_models



def create_xgboost_params(trialcount, maxdepth=[5], eta=[0.1], min_child_weight=[1],
                          gamma=[0], subsample=[0.8], colsample_bytree=[0.8], reg_alpha=[0], reg_lambda=[0]):
    xg_configs =[]
    for t in itertools.product(maxdepth, eta, min_child_weight, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda, ):
        xgb_params = {"objective": "reg:linear", "booster":"gbtree", "max_depth":t[0], "eta":t[1], "min_child_weight":t[2],
            "subsample":t[3], "nthread":4, "colsample_bytree":t[4], 'gamma':t[5],
            "alpha":t[6], "lambda":t[7]}
        xg_configs.append(xgb_params)

    if trialcount <= 0:
        return xg_configs
    else:
        count2remove = len(xg_configs) - trialcount
        print "explore %2f of search space" %(float(trialcount)/len(xg_configs))
        #indexes2remove = random.shuffle(range(len(all_dl_configs)))
        random.shuffle(xg_configs)
        return xg_configs[0:trialcount]


class XGBoostModel:
    def __init__(self, conf, xgb_params, use_cv=False):
        self.conf = conf
        self.name = "XGB"
        self.xgb_params = xgb_params
        self.use_cv = use_cv
        self.model = None

    def fit(self, X_train, y_train, X_test, y_test, y_actual, forecasting_feilds=None):
        start = time.time()
        if self.use_cv:
            model, y_pred = regression_with_xgboost(X_train, y_train, X_test, y_test, features=forecasting_feilds, use_cv=True,
                use_sklean=False, xgb_params=self.xgb_params)
        else:
            model, y_pred = regression_with_xgboost_no_cv(X_train, y_train, X_test, y_test, features=forecasting_feilds,
                xgb_params=self.xgb_params,num_rounds=200)
        self.model = model
        y_pred_final, rmsle = check_accuracy("XGBoost_nocv "+ str(self.xgb_params), self.model, X_test, self.conf.parmsFromNormalization,
                                      self.conf.target_as_log, y_actual, self.conf.command)
        self.rmsle =  rmsle
        print "Xgboost model took", (time.time() - start), "s"
        return y_pred_final

    def predict(self, X_test):
        return self.model.predict(X_test)

    def cleanup(self):
        self.model = None


class LRModel:
    def __init__(self, conf, model=None):
        self.conf = conf
        self.name = "LR"
        if model is None:
            self.model = LinearRegression(normalize=False)
        else:
            self.model = model

    def fit(self, X_train, y_train, X_test, y_test, y_actual, forecasting_feilds=None):
        start = time.time()
        self.model.fit(X_train, y_train)

        y_pred_final, rmsle = check_accuracy("LR"+str(self.model), self.model, X_test, self.conf.parmsFromNormalization,
                                      self.conf.target_as_log, y_actual, self.conf.command)
        self.rmsle =  rmsle
        print "LR model took", (time.time() - start), "s"
        return y_pred_final

    def predict(self, X_test):
        return self.model.predict(X_test)

    def cleanup(self):
        self.model = None



class DLModel:
    def __init__(self, conf, dlconf=None):
        self.conf = conf
        self.dlconf = dlconf
        self.name = "DL"
        self.model = None
    def fit(self, X_train, y_train, X_test, y_test, y_actual, forecasting_feilds=None):
        start = time.time()
        if self.dlconf == None:
            self.dlconf = MLConfigs(nodes_in_layer=10, number_of_hidden_layers=2, dropout=0.3, activation_fn='relu', loss="mse",
                epoch_count=20, optimizer=Adam(lr=0.0001), regularization=0.2)
        model, y_pred_dl = regression_with_dl(X_train, y_train, X_test, y_test, self.dlconf)
        self.model = model

        y_pred_final, rmsle = check_accuracy("DL", self.model, X_test, self.conf.parmsFromNormalization,
                                      self.conf.target_as_log, y_actual, self.conf.command)
        self.rmsle =  rmsle
        print "DL", (time.time() - start), "s"
        return y_pred_final

    def predict(self, X_test):
        return self.model.predict(X_test)

    def cleanup(self):
        self.model = None



def run_xgboost(X_train, Y_train, X_test, Y_test, forecasting_feilds=None):
    print "Running XG Boost"
    model, y_pred = regression_with_xgboost(X_train, Y_train, X_test, Y_test, forecasting_feilds)
    #undo the normalization

    return model

def drop_column(df1, df2, feild_name):
    df1 = df1.drop(feild_name,1)
    df2 = df2.drop(feild_name,1)
    return df1, df2


def transfrom_to_log2d(data):
    datat = []
    for i in range(data.shape[1]):
        datat.append(transfrom_to_log(data[:, i]))
    return np.column_stack(datat)




def transfrom_to_log(data):
    #print "->to log\n", pd.Series(data).describe()
    data_as_log = np.log(data + 1)
    return data_as_log

def retransfrom_from_log(data):
    data_from_log =  np.exp(data) - 1
    #print "<-from log\n", pd.Series(data_from_log).describe()
    return data_from_log


def tranform_train_data_to_log(train_df, test_df, sub_df, skip_field_patterns=[]):
    if sub_df is not None:
        sub_df = transform_df_to_log(sub_df, skip_field_patterns)
    return transform_df_to_log(train_df, skip_field_patterns), \
           transform_df_to_log(test_df, skip_field_patterns), \
           sub_df

    #if conf.target_as_log and not conf.log_target_only:
    #then all values are done as logs
    #df['Demanda_uni_equil'] = transfrom_to_log(df['Demanda_uni_equil'].values)


def transform_df_to_log(df, skip_field_patterns):
    columns_list = list(df)
    transformed_data = []

    for c in columns_list:
        skip = False
        for p in skip_field_patterns:
            if p in c:
                skip = True
        if not skip:
            log_vals = transfrom_to_log(df[c].values)
        else:
            log_vals = df[c].values
            print "skip log transform ", c
        transformed_data.append(log_vals)
    return pd.DataFrame(np.column_stack(transformed_data), columns=columns_list)

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


def merge_another_dataset(train_df, test_df, sub_df, analysis_type, cmd, feilds_to_use=None):
    column_count_before_merge = train_df.shape[1]
    merge_feilds = ['Semana', 'Agencia_ID' , 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID']
    feilds_to_use = feilds_to_use + merge_feilds

    sup_train_df, sup_test_df, sup_sub_df, _, _ = load_train_data(analysis_type, cmd, throw_error=True, fields=feilds_to_use)

    #sup_train_df = sup_train_df[feilds_to_use]
    #sup_test_df = sup_test_df[feilds_to_use]
    #sup_sub_df = sup_sub_df[feilds_to_use]

    #if left has duplicates, that will increase left side. this fix it
    #sup_train_df = sup_train_df.drop_duplicates(subset=merge_feilds)

    #print "X",train_df.shape, "X _actual", train_df.values.shape
    #print train_df[['Agencia_ID' , 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID']].head(5)
    #print train_df[['Agencia_ID' , 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID']].tail(5)
    train_df = pd.merge(train_df, sup_train_df, how='left', on=merge_feilds)
    #print "X",train_df.shape, "Y"
    #print train_df[['Agencia_ID' , 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID']].head(5)
    #print train_df[['Agencia_ID' , 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID']].tail(5)

    test_df = pd.merge(test_df, sup_test_df, how='left', on=merge_feilds)
    sub_df = pd.merge(sub_df, sup_sub_df, how='left', on=merge_feilds)

    if train_df.shape[1] <= column_count_before_merge:
        raise ValueError("Join failed, no new feilds added, feilds after merge=", list(train_df))

    #find_NA_rows_percent(train_df, "merging two tables")
    #ind_NA_rows_percent(test_df, "merging two tables")
    #find_NA_rows_percent(sub_df, "merging two tables")
    return train_df, test_df, sub_df


def check_accuracy(label, model, X_test, parmsFromNormalization, target_as_log, y_actual_test, command):
    y_pred_raw = model.predict(X_test)
    #undo the normalization
    y_pred_final = modeloutput2predictions(y_pred_raw, parmsFromNormalization=parmsFromNormalization)
    if target_as_log:
        y_pred_final = retransfrom_from_log(y_pred_final)
    else:
        print 'no log retransform'

    #error_ac, rmsep, mape, rmse = almost_correct_based_accuracy(y_actual_test, y_pred_final, 10)
    rmsle = calculate_rmsle(y_actual_test, y_pred_final)
    #print ">> %s AC_errorRate=%.1f RMSEP=%.6f MAPE=%6f RMSE=%6f rmsle=%.5f" %("Run " + str(command)+ " "+ label, error_ac, rmsep, mape, rmse, rmsle)
    print ">> %s rmsle=%.5f" %("Run " + str(command)+ " "+ label, rmsle)
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


def verify_forecasting_data(X_train, y_train, X_test, y_test):
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("columns not aligned X_train, y_train, X_test, y_test " + X_train.shape + " "+ y_train.shape
                         + " " + X_test.shape + " " +y_test.shape)

    if X_train.shape[0] != y_train.shape[0] or y_test.shape[0] != X_test.shape[0]:
        raise ValueError("columns not aligned X_train, y_train, X_test, y_test " + str(X_train.shape) + " "+ str(y_train.shape)
                         + " " + str(X_test.shape) + " " + str(y_test.shape))


        #print_xy_sample(X_train, y_train)
    #print_xy_sample(X_test, y_test)

    #print "X_train"
    #check4nan(X_train)
    #print "X_test"
    #check4nan(X_test)
    #print "y_train"
    #check4nan(y_train)
    #print "y_test"
    #check4nan(y_test)




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





def merge_csv_by_feild(train_df, test_df, testDf, base_df, feild_name ):
    train_df = pd.merge(train_df, base_df, how='left', on=[feild_name])
    test_df = pd.merge(test_df, base_df, how='left', on=[feild_name])
    testDf = pd.merge(testDf, base_df, how='left', on=[feild_name])
    return train_df, test_df, testDf


class IDConfigs:
    def __init__(self, target_as_log, normalize, save_predictions_with_data, generate_submission, analysis_type='base'):
        self.target_as_log = target_as_log
        self.normalize = normalize
        self.save_predictions_with_data = save_predictions_with_data
        self.generate_submission = generate_submission
        self.parmsFromNormalization = None
        self.verify_data = False
        self.analysis_type = analysis_type


class DefaultStats:
    def __init__(self, mean, count, stddev):
        self.mean = mean
        self.count = count
        self.stddev = stddev


class FeatureOps:
    def __init__(self, count=False, stddev=False, sum=False, p10=False, p90=False, kurtosis=False,
                 hmean=False, entropy=False, ci=True):
        '''
        #self.sum = sum
        self.sum = sum
        self.count = count
        self.stddev = stddev
        #follow two are too expensive
        self.p10 =False
        self.p90 = False
        self.kurtosis = kurtosis
        self.hmean = hmean
        self.entropy=entropy
        self.use_close_products_missing=False
        self.ci = ci
        '''

        self.sum = False
        self.count = False
        self.stddev = False
        #follow two are too expensive
        self.p10 =False
        self.p90 = False
        self.kurtosis = kurtosis
        self.hmean = False
        self.entropy=False
        self.median=True
        self.use_close_products_missing=False
        self.ci = ci



def generate_features(conf, train_df, test_df, subdf, y_actual_test):
    start = time.time()
    use_slope = False
    use_group_aggrigate = True
    use_product_features = True
    use_agency_features = False
    use_sales_data = False

    testDf = subdf

    if use_slope:
        train_df, test_df, testDf = addSlopes(train_df, test_df, testDf)

    default_demand_stats = DefaultStats(mean=train_df['Demanda_uni_equil'].mean(), count=train_df['Demanda_uni_equil'].count(),
                                        stddev=train_df['Demanda_uni_equil'].std())
    default_venta_hoy_stats = DefaultStats(train_df['Venta_hoy'].mean(), train_df['Venta_hoy'].count(),
                                        train_df['Venta_hoy'].std())
    default_dev_proxima_stats = DefaultStats(train_df['Dev_proxima'].mean(), train_df['Dev_proxima'].count(),
                                        train_df['Dev_proxima'].std())

    #this is to drop all features in one go
    feilds_to_drop = []

    #add mean and stddev by groups

    groups = ('Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID')
    measures = ('Dev_proxima', 'Dev_uni_proxima', 'Demanda_uni_equil', 'unit_prize', 'Venta_uni_hoy', 'Venta_hoy')
    #for t in itertools.product(groups, measures):
    #for t in [('Cliente_ID', 'Demanda_uni_equil'), ('Cliente_ID', 'Venta_uni_hoy'), ('Cliente_ID', 'Venta_hoy'),
    #    ('Ruta_SAK', 'unit_prize')]:
    #        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df,
    #                                test_df,t[0], testDf, drop=False, agr_feild=t[1])

    if use_group_aggrigate:
        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Agencia_ID', testDf, default_demand_stats,
                                                            FeatureOps(hmean=True, stddev=True, count=True))
        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Canal_ID', testDf, drop=False)
        #*train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Ruta_SAK', testDf, drop=False)
        #*train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Cliente_ID', testDf, drop=False) #duplicated
        train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Ruta_SAK', 'Cliente_ID'],
                'Demanda_uni_equil', "clients_combined", default_demand_stats,
                                                              FeatureOps(sum= True, kurtosis=True, stddev=True, count=True, p90=10, p10=True, hmean=True))

        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, default_demand_stats,
                                                            FeatureOps(stddev=True, p90=True, hmean=True,p10=True, count=True))

        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, drop=False, agr_feild='Venta_hoy',
                                                            default_stats=default_venta_hoy_stats, fops=FeatureOps(stddev=True, count=True))
        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Canal_ID', testDf, drop=False, agr_feild='Venta_hoy')
        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Ruta_SAK', testDf, drop=False, agr_feild='Venta_hoy')
        #*train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Cliente_ID', testDf, drop=False, agr_feild='Venta_hoy')
        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, drop=False, agr_feild='Venta_hoy')

        train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Ruta_SAK', 'Cliente_ID'],
            'Venta_hoy', "clients_combined_vh", default_venta_hoy_stats, FeatureOps(sum=True, hmean=True, p90=True, stddev=True))



        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, drop=False,
                                                            agr_feild='Dev_proxima', default_stats=default_dev_proxima_stats,
                                                            fops=FeatureOps(count=True))

        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Canal_ID', testDf, default_demand_stats,
        #                                                    drop=False, agr_feild='Dev_proxima', fops=FeatureOps())
        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Ruta_SAK', testDf, drop=False, agr_feild='Dev_proxima')
        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Cliente_ID', testDf, drop=False, agr_feild='Dev_proxima')
        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Agencia_ID', testDf, drop=False, agr_feild='Dev_proxima')


    if use_product_features:
        product_data_df = read_productdata_file('product_data.csv')

        #remove  unused feilds
        #product_data_df = drop_feilds_1df(product_data_df, ['has_vanilla','has_multigrain', 'has_choco', 'weight','pieces'])
        product_data_df = drop_feilds_1df(product_data_df, ['has_vanilla','has_multigrain', 'has_choco', "time_between_delivery"])

        if 'weight' in product_data_df:
            weight = product_data_df['weight']
            product_data_df['weight'] = np.where(weight < 0, 0, weight)
        if 'pieces' in product_data_df:
            pieces = product_data_df['pieces']
            product_data_df['pieces'] = np.where(pieces < 0, 0, pieces)


        train_df = pd.merge(train_df, product_data_df, how='left', on=['Producto_ID'])
        test_df = pd.merge(test_df, product_data_df, how='left', on=['Producto_ID'])
        testDf = pd.merge(testDf, product_data_df, how='left', on=['Producto_ID'])

        #add aggrigates by groups
        #train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'brand_id', testDf, drop=False,
        #    default_stats=default_demand_stats)
        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'product_word', testDf, drop=False,
            default_stats=default_demand_stats, fops=FeatureOps(hmean=True))

        #remove group ids
        feilds_to_drop = feilds_to_drop + ['product_word', 'brand_id']

    if use_agency_features:
        agency_data_df = read_productdata_file('agency_data.csv')
        train_df, test_df, testDf =  merge_csv_by_feild(train_df, test_df, testDf, agency_data_df, 'Agencia_ID')

        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Town_id', testDf, drop=False, fops=FeatureOps())
        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'State_id', testDf, drop=False,
                                                            default_stats=default_demand_stats, fops=FeatureOps())
        feilds_to_drop = feilds_to_drop + ['Town_id', 'State_id']

    if use_sales_data:
        train_df, test_df, testDf = add_time_bwt_delivery(train_df, test_df, testDf)
        train_df, test_df, testDf = add_last_sale_and_week(train_df, test_df, testDf)

    #train_df, test_df, testDf = add_five_grouped_stats(train_df, test_df, testDf)

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

    test_df_before_dropping_features = test_df

    #save_train_data(conf.analysis_type, conf.command, train_df, test_df, testDf)
    #train_df, test_df, testDf = merge_another_dataset(train_df, test_df, testDf, 'fg_stats', conf.command,["mean_sales", "sales_count",
    #                                        "sales_stddev", "median_sales", "last_sale", "last_sale_week", "returns"])

    #train_df, test_df, testDf = do_one_hot_all(train_df, test_df, testDf, ['Agencia_ID', 'Cliente_ID'])
    train_data_feilds_to_drop = ['Venta_uni_hoy', 'Venta_hoy', 'Dev_uni_proxima', 'Dev_proxima', 'Demanda_uni_equil']
    train_df, test_df, _ = drop_feilds(train_df, test_df, None, feilds_to_drop + train_data_feilds_to_drop)
    testDf = drop_feilds_1df(testDf, feilds_to_drop)

    #TODO explore this more http://pandas.pydata.org/pandas-docs/stable/missing_data.html
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)


    print "generate_features took ", (time.time() - start), "s"
    return train_df, test_df, testDf, y_actual_test, test_df_before_dropping_features

def get_models4xgboost_only(conf):
    models = []
#    xgb_params = {"objective": "reg:linear", "booster":"gbtree", "max_depth":3, "eta":0.1, "min_child_weight":5,
#            "subsample":0.5, "nthread":4, "colsample_bytree":0.5, "num_parallel_tree":1, 'gamma':0}
    xgb_params = {"objective": "reg:linear", "booster":"gbtree", "max_depth":10, "eta":0.1, "min_child_weight":8,
            "subsample":0.5, "nthread":4, "colsample_bytree":0.5, "num_parallel_tree":1, 'gamma':0}

    models.append(XGBoostModel(conf, xgb_params))
    return models


def get_models4dl_only(conf):
    MLConfigs(nodes_in_layer=20, number_of_hidden_layers=2, dropout=0.0, activation_fn='relu', loss="mse",
                epoch_count=100, optimizer=Adam(lr=0.0001), regularization=0.1)
    return [DLModel(conf)]


def get_models4ensamble(conf):
    models = []
    #models = [RFRModel(conf), DLModel(conf), LRModel(conf)]
    #models = [LRModel(conf)]
    # see http://scikit-learn.org/stable/modules/linear_model.html

    #0 was too big to run with depth set to 1, and 1 was overfitting a bit
    if conf.command == 0 or conf.command == 1:
        xgb_params = {"objective": "reg:linear", "booster":"gbtree", "max_depth":3, "eta":0.1, "min_child_weight":5,
            "subsample":0.5, "nthread":4, "colsample_bytree":0.5, "num_parallel_tree":1, 'gamma':0}
    else:
        xgb_params = {"objective": "reg:linear", "booster":"gbtree", "max_depth":10, "eta":0.1, "min_child_weight":8,
            "subsample":0.5, "nthread":4, "colsample_bytree":0.5, "num_parallel_tree":1, 'gamma':0}

    models = [
                #DLModel(conf),

                #LRModel(conf, model=linear_model.BayesianRidge()),
                #LRModel(conf, model=linear_model.LassoLars(alpha=.1)),
                #LRModel(conf, model=linear_model.Lasso(alpha = 0.1)),
                #LRModel(conf, model=Pipeline([('poly', PolynomialFeatures(degree=3)),
                #LRModel(conf, model=linear_model.Ridge (alpha = .5))
                #   ('linear', LinearRegression(fit_intercept=False))])),
                XGBoostModel(conf, xgb_params, use_cv=True),
                LRModel(conf, model=linear_model.Lasso(alpha = 0.3)),
                RFRModel(conf, RandomForestRegressor(oob_score=True, n_jobs=4)),
                #LRModel(conf, model=linear_model.Lasso(alpha = 0.2)),
                ETRModel(conf, model=ExtraTreesRegressor(n_jobs=4)),
                #AdaBoostRModel(conf, model=AdaBoostRegressor(loss='square'))
              ]
    return models


def get_models4xgboost_tunning(conf):
    #http://www.voidcn.com/blog/mmc2015/article/p-5751771.html
    case = 4

    if case == 0:
        xgb_params = {"objective": "reg:linear", "booster":"gbtree", "max_depth":5, "eta":0.1, "min_child_weight":1,
            "subsample":0.8, "nthread":4, "colsample_bytree":0.8, "num_parallel_tree":1, 'gamma':0}
        xgb_params_list = [xgb_params]
    elif case == 1:
        eta = 0.1
        #tune maxdepth and min child weight #eta decide before
        xgb_params_list = create_xgboost_params(0, maxdepth=[3, 5, 8, 10], eta=[eta], min_child_weight=[1, 3, 5],
            gamma=[0], subsample=[0.8], colsample_bytree=[0.8], reg_alpha=[0], reg_lambda=[0])
    elif case == 2:
        #tune gamma
        eta = 0.1
        maxdepth = 10
        min_child_weight = 5
        xgb_params_list = create_xgboost_params(0, maxdepth=[maxdepth], eta=[eta], min_child_weight=[min_child_weight],
            gamma=[0.5, 0.3, 0.1], subsample=[0.8], colsample_bytree=[0.8], reg_alpha=[0], reg_lambda=[0])
    elif case == 3:
        #Tune subsample and colsample_bytree
        eta = 0.1
        maxdepth = 10
        min_child_weight = 5
        gamma = 0.1
        xgb_params_list = create_xgboost_params(0, maxdepth=[maxdepth], eta=[eta], min_child_weight=[min_child_weight],
            gamma=[gamma], subsample=[0.6, 0.8, 1.0], colsample_bytree=[0.6, 0.8, 1.0], reg_alpha=[0], reg_lambda=[0])

    elif case == 4:
        #Run 7 XGBoost_nocv {'reg_alpha': 0, 'booster': 'gbtree', 'colsample_bytree': 0.8, 'nthread': 4, 'min_child_weight': 5,
        # 'subsample': 0.6, 'reg_lambda': 0, 'eta': 0.1, 'objective': 'reg:linear', 'max_depth': 10, 'gamma': 0.1} AC_errorRate=88.0 RMSEP=130.365780 MAPE=67.375561 RMSE=15.882261 rmsle=0.60238
        eta = 0.1
        maxdepth = 10
        min_child_weight = 5
        gamma = 0.1
        xgb_params_list = create_xgboost_params(0, maxdepth=[3, 10, 15], eta=[eta], min_child_weight=[5, 8, 10],
            gamma=[0.1, 0.2], subsample=[0.6], colsample_bytree=[0.8], reg_alpha=[0], reg_lambda=[0])
    else:
        raise ValueError("Unknown case "+ str(case))

    #xgb_params_list = create_xgboost_params(0, maxdepth=[3, 5], eta=[0.1, 0.05], min_child_weight=[5],
    #                      gamma=[0], subsample=[0.5], colsample_bytree=[0.5],
    #                      reg_alpha=[0], reg_lambda=[0])


    models = []
    for xgb_params in xgb_params_list:
        #xgb_params['max_depth'] = md[0]
        #xgb_params['subsample'] = md[1]
        #xgb_params['min_child_weight'] = md[2]
        models.append(XGBoostModel(conf, xgb_params, use_cv=True))
        #xgb_params['seed'] = 347
        #models.append(XGBoostModel(conf, xgb_params))

    return models


def get_models4rfr_tunning(conf):
    configs = create_rfr_params(n_estimator=[100,200, 300, 500], oob_score=[True, False], min_samples_leaf=[50, 100, 200])
    models = [RFRModel(conf, rfr) for rfr in configs]
    return models;


#class ModelSet:



#takes data as no transformation and returns results after undoing all tranformatins
def do_forecast(conf, train_df, test_df, sub_df, y_train, y_test, models=None):
    y_actual_test = y_test

    if conf.target_as_log:
        train_df, test_df, sub_df = tranform_train_data_to_log(train_df, test_df, sub_df, skip_field_patterns=['kurtosis', 'id'])
        y_train, y_test = transfrom_to_log(y_train), transfrom_to_log(y_test)

    start = time.time()
    if train_df.shape[1] != test_df.shape[1]:
        raise ValueError("train and test does not match " + str(list(train_df)) + " " + str(list(test_df)))

    find_NA_rows_percent(train_df, "train_df before forecast")
    find_NA_rows_percent(test_df, "test before forecast")

    forecasting_feilds = list(train_df)
    print "Forecasting Feilds", [ "("+str(i)+")" + forecasting_feilds[i] for i in range(len(forecasting_feilds))]

    y_train, parmsFromNormalization = preprocess1DtoZeroMeanUnit(y_train)
    y_test = apply_zeroMeanUnit(y_test, parmsFromNormalization)

    X_train, parmsFromNormalization2D = preprocess2DtoZeroMeanUnit(train_df.values.copy())
    x_test_raw = test_df.values.copy()

    X_test = apply_zeroMeanUnit2D(x_test_raw, parmsFromNormalization2D)
    conf.parmsFromNormalization = parmsFromNormalization

    verify_forecasting_data(X_train, y_train, X_test, y_test)

    de_normalized_forecasts = []

    tune_paramers = False
    if models is None:
        if tune_paramers:
            models = get_models4xgboost_tunning(conf)
            #models = get_models4rfr_tunning(conf)
        else:
            #models = get_models4xgboost_only(conf)
            models = get_models4ensamble(conf)

    print_mem_usage("before running models")
    print "trying ", len(models), " Models"
    for m in models:
        print "running model", m.name
        m_start = time.time()
        den_forecasted_data = m.fit(X_train, y_train, X_test, y_test, y_actual_test, forecasting_feilds=forecasting_feilds)
        print "model took ", (time.time() - m_start), "seconds"
        de_normalized_forecasts.append(den_forecasted_data)

    if conf.verify_data:
        corrected_Y_test = modeloutput2predictions(y_test, parmsFromNormalization=parmsFromNormalization)
        if conf.target_as_log:
            corrected_Y_test = retransfrom_from_log(corrected_Y_test)
        print "Undo normalization test passed", np.allclose(corrected_Y_test, y_actual_test, atol=0.01)

        print "Undo X data test test passed", \
            np.allclose(x_test_raw, undo_zeroMeanUnit2D(X_test, parmsFromNormalization2D), atol=0.01)

    if len(de_normalized_forecasts) == 1:
        forecasts = np.column_stack(de_normalized_forecasts)
    else:
        forecasts = np.column_stack(de_normalized_forecasts)
    print "do_forecast took ", (time.time() - start), "s"

    #lets generate submissions
    submission_forecasts = None
    if sub_df is not None:
        sub_X_all = sub_df.values
        submission_forecasts = [create_submission(conf, m, sub_X_all, parmsFromNormalization, parmsFromNormalization2D) for m in models]
        submission_forecasts = np.column_stack(submission_forecasts)

    print str(len(models)), " models forecasts=", forecasts.shape, ", submission_forecasts=", submission_forecasts.shape
    return models, forecasts, submission_forecasts


    #return best_model, parmsFromNormalization, parmsFromNormalization2D, best_forecast
    #return model, parmsFromNormalization, parmsFromNormalization2D, best_forecast


def calculate_accuracy(label, y_actual_test, y_forecast):
    #error_ac, rmsep, mape, rmse = almost_correct_based_accuracy(y_actual_test, y_forecast, 10)
    rmsle = calculate_rmsle(y_actual_test, y_forecast)
    #print ">> %s AC_errorRate=%.1f RMSEP=%.6f MAPE=%6f RMSE=%6f rmsle=%.5f" %("Run " + label, error_ac, rmsep, mape, rmse, rmsle)
    print "%s :rmsle=%.5f" %(label, rmsle)
    return rmsle


def find_range(rmsle, forecast):
    h = np.exp(rmsle)*(forecast+1) - 1
    l = (forecast+1)/np.exp(rmsle) - 1
    return l, h



def create_per_model_submission(conf, models, testDf, parmsFromNormalization, parmsFromNormalization2D ):
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

    kaggale_predicted_list = []
    kaggale_predicted_as_log_raw_list = [m.predict(kaggale_test) for m in models]
    for kpr in kaggale_predicted_as_log_raw_list:
        kaggale_predicted = modeloutput2predictions(kpr, parmsFromNormalization=parmsFromNormalization)
        if conf.target_as_log:
            kaggale_predicted = retransfrom_from_log(kaggale_predicted)
        kaggale_predicted_list.append(kaggale_predicted)

    return ids, np.column_stack(kaggale_predicted_list)


def g2df_sum_mean(group):
    sum = group.sum()
    mean = group.mean()
    count = group.count()
    valuesDf = sum.to_frame("sum")
    valuesDf.reset_index(inplace=True)
    valuesDf['mean'] = mean.values
    valuesDf['count'] = count.values
    valuesDf['rank'] = valuesDf['mean']*np.log(1+valuesDf['count'])

    return valuesDf



def create_submission(conf, model, sub_X_all, parmsFromNormalization, parmsFromNormalization2D ):
    start = time.time()
    print "creating submission for ", sub_X_all.shape[0], "values"

    #pd.colnames(temp)[pd.colSums(is.na(temp)) > 0]
    #print temp.describe()
    #print df.isnull().any()

    kaggale_test = apply_zeroMeanUnit2D(sub_X_all, parmsFromNormalization2D)

    print "kaggale_test", kaggale_test.shape

    kaggale_predicted_raw = model.predict(kaggale_test)
    kaggale_predicted = modeloutput2predictions(kaggale_predicted_raw, parmsFromNormalization=parmsFromNormalization)

    print "kaggale_predicted", kaggale_test.shape

    if conf.target_as_log:
        kaggale_predicted = retransfrom_from_log(kaggale_predicted)

    print "log retransform", kaggale_test.shape

    #submission_file = 'submission'+str(conf.command)+ '.csv'
    #save_submission_file(submission_file, ids, kaggale_predicted)

    #to_save = np.column_stack((ids, kaggale_predicted))
    #to_saveDf =  pd.DataFrame(to_save, columns=["id","Demanda_uni_equil"])
    #to_saveDf = to_saveDf.fillna(0)
    #to_saveDf["id"] = to_saveDf["id"].astype(int)
    #save_file(conf.analysis_type, conf.command, to_saveDf, "submission")
    #to_saveDf.to_csv(submission_file, index=False)

    #print "Best Model Submission Stats"
    #print_submission_data(sub_df=to_saveDf, command=conf.command)
#    print "Submission done for ", to_saveDf.shape[0], "file", submission_file
#    print "create_submission took ", (time.time() - start), "s"

    return kaggale_predicted

    #to_saveDf["groupedMeans"] = testDf["groupedMeans"]
    #to_saveDf["groupedStd"] = testDf["groupedStd"]
    #to_saveDf["Slopes"] = testDf["Slopes"]
    #to_saveDf.to_csv('prediction_detailed.csv', index=False)

    #np.savetxt('submission.csv', to_save, delimiter=',', header="id,Demanda_uni_equil", fmt='%d')   # X is an array









