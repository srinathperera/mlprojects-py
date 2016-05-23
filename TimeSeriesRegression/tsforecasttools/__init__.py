import numpy as np
import time as time

import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


from mltools import rolling_univariate_window, build_rolling_window_dataset
from mltools import train_test_split,print_graph_test,almost_correct_based_accuracy
from mltools import regression_with_dl, print_regression_model_summary, report_scores, regression_with_GBR, regression_with_LR, regression_with_RFR


def aggregate_hl(mavg1_vals, window_size):
    hl_mvavg = []
    for i in range(len(mavg1_vals)):
        start_index = max(i-window_size,0)
        if start_index == i:
            hl_mvavg.append(mavg1_vals[i])
        else:
            hl_mvavg.append(np.mean(mavg1_vals[start_index:i:1]))
    return np.array(hl_mvavg)


def run_timeseries_froecasts(X_train, y_train, X_test, y_test, window_size, epoch_count):
    start_time = time.time()
    print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape,
      "X_test.shape", X_test.shape, "Y_test.shape", y_test.shape)

    #Deep Learning
    nodes_in_layer = 200
    number_of_hidden_layers = 5
    #nodes_in_layer = 400
    #number_of_hidden_layers = 40
    droput = 0.1
    activation_fn='relu'
    #activation_fn='sigmoid'

    #y_pred_dl = regression_with_dl(X_train, y_train, X_test, y_test, nodes_in_layer,
    #                   number_of_hidden_layers, droput, activation_fn, epoch_count)
    #print_regression_model_summary("DL", y_test, y_pred_dl)

    #parameter sweep http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html
    #rfr = RandomForestRegressor()
    # specify parameters and distributions to sample from
    param_dist = {"n_estimators": [150, 50, 300],
              "min_samples_split": [1,5,10]}

    # run randomized search
    #random_search = RandomizedSearchCV(rfr, param_distributions=param_dist,
    #                                   n_iter=3)
    #random_search = GridSearchCV(rfr, param_grid=param_dist)
    #random_search.fit(X_train, y_train)
    #report_scores(random_search.grid_scores_)
    #print("===================")

    #Linear Regression
    #lr = LinearRegression(normalize=True)
    #lr.fit(X_train, y_train)
    #y_pred_lr = lr.predict(X_test)

    #print_regression_model_summary("LR", y_test, y_pred_lr)
    y_pred_lr = regression_with_LR(X_train, y_train, X_test, y_test)


    #Random Forest Regressor
    #rfr = RandomForestRegressor(n_estimators=300, min_samples_split=10)
    #rfr = RandomForestRegressor()
    #rfr.fit(X_train, y_train)

    #y_pred_rfr = rfr.predict(X_test)


    #print_regression_model_summary("RFR", y_test, y_pred_rfr)
    y_pred_rfr = regression_with_RFR(X_train, y_train, X_test, y_test)

    #GradientBoostingRegressor
    #params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
    #      'learning_rate': 0.01, 'loss': 'ls'}
    #gfr = GradientBoostingRegressor(**params)
    #gfr.fit(X_train, y_train)
    #y_pred_gbr = gfr.predict(X_test)
    #print_regression_model_summary("GBR", y_test, y_pred_gbr)

    y_pred_gbr = regression_with_GBR(X_train, y_train, X_test, y_test)

    print("All forecasts are done")

    #average forecast
    #all_forecasts = np.column_stack((y_pred_lr, y_pred_dl, y_pred_rfr, y_pred_gbr))
    all_forecasts = np.column_stack((y_pred_lr, y_pred_rfr, y_pred_gbr))

    #combine_models(all_forecasts, y_test)

    forecast_data = np.column_stack((np.array(list(range(0,y_test.shape[0],1))), y_test, y_pred_lr, y_pred_rfr, y_pred_gbr, X_test))

    headers=''
    for i in range(X_test.shape[1]):
        headers = headers + ",X"+i

    #print (forecast_data)
    np.savetxt('forecastdata.csv', forecast_data, delimiter=',', header="seq,actual,LR,RFR,GBR,"+headers)   # X is an array
    #print("All forecasts written to forecastdata.csv")
    print("run took %f seconds" %((time.time() - start_time)))

    #print_graph_test(y_test, y_pred_lr, y_pred_lr, 500)


def combine_models(models, y_test):
    combined_forecast = [ np.median(models[i]) for i in range(models.shape[0])]
    print_regression_model_summary("CombinedMedian", y_test, combined_forecast)

    combined_forecast = [ np.mean(models[i]) for i in range(models.shape[0])]
    print_regression_model_summary("CombinedMean", y_test, combined_forecast)