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
from sklearn.linear_model import LinearRegression

#from mlpreprocessing import feather2df

from inventory_demand import *
from mltools import *
import scipy

import sys
print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)




#Note that groupby will preserve the order in which observations are sorted within each group. For example, the groups created by groupby() below are in the order the appeared in the original DataFrame:

#grouped = df.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])['Demanda_uni_equil']


def check_increasing(list):
    start = -1
    for i in list:
        if start > i:
            return False
        else:
            start = i
    return True


def build_windw(group, list):
    #if not check_increasing(group['Semana']):
    #    raise "group by is not increasing"
    sales = group['Demanda_uni_equil'].values
    last = -1
    last_value = -1
    filled_data = []
    index = 0

    for d in group['Semana']:
        if last != -1 and d - last > 1:
            for i in range(last+1, d, 1):
                filled_data.append(last_value)

        filled_data.append(sales[index])
        last_value = sales[index]
        last = d
        index = index + 1

    for i in range(3,len(filled_data)):
        feilds_t = extract_feilds_from_group(group, filled_data[:i])
        window_t = [filled_data[i-3], filled_data[i-2], filled_data[i-1], filled_data[i]]
        list.append(np.concatenate([feilds_t, window_t]))

    return None


def build_window_dataset(train_df):
    grouped = train_df.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
    collected_list = [] #apply will collect data here
    grouped.apply(build_windw, list=collected_list)

    all_data = np.vstack(collected_list)

    print all_data[np.random.randint(all_data.shape[0],size=10)]
    x_all = all_data[:, :-1]
    y_all = all_data[:, -1:]
    return x_all, y_all


def extract_feilds_from_group(group, sales):
    agencia_id_mean = group['Agencia_ID_Demanda_uni_equil_Mean'].values[0]
    agencia_id_stddev = group['Agencia_ID_Demanda_uni_equil_StdDev'].values[0]
    ruta_sak_mean = group['Ruta_SAK_Demanda_uni_equil_Mean'].values[0]
    ruta_sak_stddev = group['Ruta_SAK_Demanda_uni_equil_StdDev'].values[0]

    mean  = np.mean(sales)
    stddev = np.std(sales)
    median = np.median(sales)
    #entropy = scipy.stats.entropy(sales)

    data_as_num = np.nan_to_num([mean, stddev, median, agencia_id_mean, agencia_id_stddev, ruta_sak_mean, ruta_sak_stddev])
    data_maxed = np.where(data_as_num > 10000, 10000, data_as_num)

    return data_maxed


def build_test_window(group, default_value):
    # build or guess missing values

    week2predict = group['Semana_x'].values[0]
    sales = group['Demanda_uni_equil'].values
    last = week2predict-3
    last_value = default_value
    filled_data = []
    index = 0

    days = group['Semana_y'].astype(int)
    for d in days:
        if d >= week2predict -3:
            if d - last > 1:
                for i in range(last+1, d, 1):
                    filled_data.append(last_value)

            filled_data.append(sales[index])
            last = d

        last_value = sales[index]
        index = index + 1
    filled_data_size = len(filled_data)
    if filled_data_size == 0:
        filled_data = [default_value, default_value, default_value]
    elif filled_data_size == 1:
        filled_data = [filled_data[0], filled_data[0], filled_data[0]]
    elif filled_data_size == 2:
        filled_data = [filled_data[0]] + filled_data

    if len(filled_data) != 3:
        raise Exception("unexpected size %d" %(len(filled_data)))

    return np.concatenate([extract_feilds_from_group(group, sales), filled_data])


def forecast_batch(model,  train_df, test_df, default_sales):
    test_df_t = test_df[['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'Semana']]

    merged = pd.merge(test_df_t, train_df, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])

    print "train", list(train_df)
    print "merged", list(merged)


    merged_grouped = merged.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'Semana_y'])
    test_data = merged_grouped.apply(build_test_window, default_sales)

    #test_data = test_data.fillna(0) #TODO check what this is needed

    #pd.colnames(test_data)[pd.colSums(np.isna(test_data)) > 0]
    #print temp.describe()
    #print df.isnull().any()

    x_test = np.vstack(test_data.values)
    #print "xtest sample"
    #print x_test[np.random.randint(x_test.shape[0],size=10)]

    y_predictions = model.predict(x_test)

    predicted_data_frame = test_data.to_frame("Data")
    predicted_data_frame.reset_index(inplace=True) #convert index to feilds

    predicted_data_frame = drop_feilds_1df(predicted_data_frame, ['Data'])
    predicted_data_frame["Demanda_uni_equil"] = y_predictions

    return predicted_data_frame

def forecast(model, train_df, test_df, default_sales):
    test_df7 = test_df[test_df['Semana'] == 7]
    test_df8 = test_df[test_df['Semana'] == 8]
    test_df9 = test_df[test_df['Semana'] == 9]

    #we predict day by day
    y_predictions7_df = forecast_batch(model, train_df, test_df7, default_sales)

    y_predictions7_df.rename(columns={'Semana_y': 'Semana'}, inplace=True)
    train_df = pd.concat([train_df, y_predictions7_df])
    y_predictions8_df = forecast_batch(model, train_df, test_df8, default_sales)

    y_predictions8_df.rename(columns={'Semana_y': 'Semana'}, inplace=True)
    train_df = pd.concat([train_df, y_predictions8_df])
    y_predictions9_df = forecast_batch(model, train_df, test_df9, default_sales)

    y_predictions9_df.rename(columns={'Semana_y': 'Semana'}, inplace=True)
    prediction_df = pd.concat([y_predictions7_df, y_predictions8_df, y_predictions9_df])


    prediction_df_with_actual = pd.merge(test_df, prediction_df, how='left', on=['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])
    return prediction_df_with_actual

do_normalize_data = False #making this true reduce accuracy, test later

#df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems0_100.csv')
df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems300.csv')

#preporcessing
df = drop_feilds_1df(df, ['Venta_uni_hoy', 'Venta_hoy', 'Dev_uni_proxima', 'Dev_proxima'])
y_actual = df['Demanda_uni_equil'].values

training_set_size = int(0.7*df.shape[0])
test_set_size = df.shape[0] - training_set_size

y_actual_test = y_actual[-1*test_set_size:] # this we used for validation later

y_actual_log = transfrom_to_log(df['Demanda_uni_equil'].values)


if do_normalize_data:
    #normlization we do here to avoid warning overwriting copied data frame
    y_actual_log_train = y_actual[:training_set_size]
    y_actual_log_test = y_actual[-1*test_set_size:]
    y_actual_log_train_norm, parmsFromNormalization = preprocess1DtoZeroMeanUnit(y_actual_log_train)
    y_actual_log_test_norm = apply_zeroMeanUnit2D(y_actual_log_test, parmsFromNormalization)
    df['Demanda_uni_equil'] = np.concatenate((y_actual_log_train_norm,y_actual_log_test_norm))
else:
    df['Demanda_uni_equil'] = y_actual_log

#break training and test
train_df = df[:training_set_size]
test_df = df[-1*test_set_size:]

default_sales = train_df['Demanda_uni_equil'].median()

stat_df = calculate_feild_stats(train_df, 'Agencia_ID', 'Demanda_uni_equil')
train_df = merge_stats_with_df(train_df, stat_df, 'Agencia_ID', default_mean=default_sales, default_stddev=None)
test_df = merge_stats_with_df(test_df, stat_df, 'Agencia_ID', default_mean=default_sales, default_stddev=None)

stat_df = calculate_feild_stats(train_df, 'Ruta_SAK', 'Demanda_uni_equil')
train_df = merge_stats_with_df(train_df, stat_df, 'Ruta_SAK', default_mean=default_sales, default_stddev=None)
test_df = merge_stats_with_df(test_df, stat_df, 'Ruta_SAK', default_mean=default_sales, default_stddev=None)





if train_df.shape[1] != test_df.shape[1]:
    raise Exception("train and test df does not match")
print "train feilds=", list(train_df)
print "test feilds=", list(train_df)

x_train, y_train = build_window_dataset(train_df)
y_train = np.ravel(y_train)
#train the model
print "Shapes", x_train.shape, y_train.shape

check4nan(x_train)
model = run_lr(x_train, y_train, None, None)
model = run_rfr(x_train, y_train, None, None)
#model = run_xgboost(x_train, y_train, None, None)


#verify
test_df = drop_feilds_1df(test_df, ['Demanda_uni_equil'])
prediction_df_with_actual = forecast(model, train_df, test_df, default_sales)

#undo changes, first noralization, then log
predecited_retransformed = None
if do_normalize_data:
    undo_norm_forecasts = modeloutput2predictions(
        prediction_df_with_actual["Demanda_uni_equil"], parmsFromNormalization)
    predecited_retransformed = retransfrom_from_log(undo_norm_forecasts)
else:
    predecited_retransformed = retransfrom_from_log(prediction_df_with_actual["Demanda_uni_equil"])

rmsle = calculate_rmsle(y_actual_test, predecited_retransformed)
print "rmsle =", rmsle

