import numpy as np
import csv
import pandas as pd
from sklearn import preprocessing
import math

from scipy import stats
from tsforecasttools import run_timeseries_froecasts, aggregate_hl
from sklearn.linear_model import LinearRegression


from mltools import build_rolling_window_dataset, l2norm, regression_with_GBR, regression_with_LR, regression_with_RFR
from mltools import train_test_split,print_graph_test,almost_correct_based_accuracy, shuffle_data
from mltools import regression_with_dl, print_regression_model_summary, preprocess1DtoZeroMeanUnit, MLConfigs, check_assending

from datetime import datetime
from keras.optimizers import Adam


#df = pd.read_csv("household_power_consumption200k.txt", delimiter=';')
#print(df.head())
#power_data = df['Global_active_power'].values
#print (power_data)

#https://datamarket.com/data/set/22ox/monthly-milk-production-pounds-per-cow-jan-62-dec-75
milk_production_data = pd.read_csv('./data/milk_production.csv')
milk_production = milk_production_data['Production'].values.flatten()


#bike sharing
bikesharingDf = pd.read_csv('./data/bikesharing_hourly.csv')
check_assending(bikesharingDf, 'dteday','%Y-%m-%d')
bikesharing = bikesharingDf['cnt']


#dollar euro exchange rate - https://research.stlouisfed.org/fred2/series/DEXUSEU#
excahngeRateDf = pd.read_csv('./data/USDvsEUExchangeRateFixed.csv')
check_assending(excahngeRateDf, 'day','%Y-%m-%d')
excahngeRateDf = excahngeRateDf.interpolate() #interploate missing values
exchangeRate = excahngeRateDf['ExchangeRate']


#http://finance.yahoo.com/q/hp?s=AAPL&a=11&b=12&c=1980&d=05&e=1&f=2016&g=d
appleSotcksDf = pd.read_csv('./data/applestocksfixed.csv')
check_assending(appleSotcksDf, 'Date','%Y-%m-%d')
appleSotcks = appleSotcksDf['Close']

datasetsNames = ["milk_production", "bikesharing", "exchangeRate", "appleSotcks"]
datasets = [milk_production, bikesharing, exchangeRate, appleSotcks]

window_size = 14

configs = [
    #lr=0.01
    MLConfigs(nodes_in_layer=20, number_of_hidden_layers=3, dropout=0, activation_fn='relu', loss="mse",
              epoch_count=200, optimizer=Adam(lr=0.001)),
    #MLConfigs(nodes_in_layer=20, number_of_hidden_layers=3, dropout=0, activation_fn='relu', loss="mse",
    #          epoch_count=200, optimizer=Adam(lr=0.001), regularization=0.005),
    ]

datasetIndex = 0
for dataset in datasets:
    #check for assending
    dataset = np.array(dataset)
    dataset, parmsFromNormalization = preprocess1DtoZeroMeanUnit(dataset)
    X_all, Y_all = build_rolling_window_dataset(dataset, window_size)

    row_count = X_all.shape[0];
    training_set_size = int(0.7*row_count)
    #print("X_all.shape", X_all.shape)

    #X_all = np.column_stack((X_all, cycle_data, zscore_vals, entropy_vals, mavg1_vals, mavg2_vals, mavg4_vals, mavg8_vals, mavg16_vals))
    X_all, Y_all = shuffle_data(X_all, Y_all)

    X_train, X_test, y_train, y_test = train_test_split(training_set_size, X_all, Y_all)

    print ">> Dataset %s" %(datasetsNames[datasetIndex])
    #run_timeseries_froecasts(X_train, y_train, X_test, y_test, window_size, epoch_count=10, parmsFromNormalization=parmsFromNormalization)

    index = 0
    for c in configs:
        c.epoch_count = 200
        y_pred_dl = regression_with_dl(X_train, y_train, X_test, y_test, c)
        print ">> %d %s" %(index, str(c.tostr()))
        print_regression_model_summary("DL", y_test, y_pred_dl, parmsFromNormalization)
        index = index + 1
    datasetIndex = datasetIndex +1
