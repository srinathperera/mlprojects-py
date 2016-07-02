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
import xgboost as xgb
from mltools import undoPreprocessing, ParmsFromNormalization

def aggregate_hl(mavg1_vals, window_size):
    hl_mvavg = []
    for i in range(len(mavg1_vals)):
        start_index = max(i-window_size,0)
        if start_index == i:
            hl_mvavg.append(mavg1_vals[i])
        else:
            hl_mvavg.append(np.mean(mavg1_vals[start_index:i:1]))
    return np.array(hl_mvavg)


class XGBoostModel:
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        return self.model.predict(xgb.DMatrix(data))


def regression_with_xgboost1(X_train, Y_train, X_test):
    #http://datascience.stackexchange.com/questions/9483/xgboost-linear-regression-output-incorrect
    #http://xgboost.readthedocs.io/en/latest/get_started/index.html
    #https://www.kaggle.com/c/higgs-boson/forums/t/10286/customize-loss-function-in-xgboost

    T_train_xgb = xgb.DMatrix(X_train, Y_train)

    params = {"objective": "reg:linear", "booster":"gblinear"}
    gbm = xgb.train(dtrain=T_train_xgb,params=params)
    y_pred = gbm.predict(xgb.DMatrix(X_test))
    return XGBoostModel(gbm), y_pred


def regression_with_xgboost2(X_train, Y_train, X_test, Y_test):
    xlf = xgb.XGBRegressor(objective="reg:linear", seed=1729)
    xlf.fit(X_train,Y_train, eval_metric='rmse', verbose = True, eval_set = [(X_test, Y_test)])

    # calculate the auc score
    y_pred = xlf.predict(X_test)
    #print('\nMean Square error" ', mean_squared_error(y_test,preds))
    return XGBoostModel(xlf), y_pred


def regression_with_xgboost(X_train, Y_train, X_test, Y_test, forecasting_feilds=None):
    #http://datascience.stackexchange.com/questions/9483/xgboost-linear-regression-output-incorrect
    #http://xgboost.readthedocs.io/en/latest/get_started/index.html
    #https://www.kaggle.com/c/higgs-boson/forums/t/10286/customize-loss-function-in-xgboost


    #http://xgboost.readthedocs.io/en/latest/python/python_intro.html
    #Python API http://xgboost.readthedocs.io/en/latest/python/python_api.html
    #is this correct

    if X_test is not None and Y_test is not None:
        train_data = xgb.DMatrix(X_train, Y_train)
        test_data = xgb.DMatrix(X_test, Y_test)
        #parameters https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
        #basic version
        #params = {"objective": "reg:linear", "booster":"gblinear"}
        #params = {"objective": "reg:linear", "booster":"gblinear", 'eta':0.1, 'max_depth':2, 'alpha':0.2, 'lambda':0.8}
        params = {"objective": "reg:linear", "booster":"gbtree", 'eta':0.3, 'gamma':1.0 , 'max_depth':1, 'min_child_weight':3, 'subsample':0.99,
                  'alpha':0.8, 'lambda':0.8}
        #params = {"objective": "reg:linear", "booster":"gblinear", 'eta':0.1, 'gamma':1.0 , 'max_depth':3, 'min_child_weight':1}

        params['nthread'] = 4

        evallist  = [(test_data,'eval'), (train_data,'train')]
        num_round =20
        gbm = xgb.train( params, train_data, num_round, evallist, verbose_eval = True, early_stopping_rounds=5)
        y_pred = gbm.predict(xgb.DMatrix(X_test))

        gbm.dump_model('xgb.fmap')
        importance = gbm.get_fscore(fmap='xgb.fmap')
        print "importance=", importance
        return XGBoostModel(gbm), y_pred
    else:
        train_data = xgb.DMatrix(X_train, Y_train)

        params = {"objective": "reg:linear", "booster":"gblinear"}
        params['nthread'] = 4

        evallist  = [(train_data,'train')]
        num_round = 10
        gbm = xgb.train( params, train_data, num_round, evallist, verbose_eval = True)
        y_pred = gbm.predict(xgb.DMatrix(X_test))
        return XGBoostModel(gbm), y_pred





def run_timeseries_froecasts(X_train, y_train, X_test, y_test, window_size, epoch_count, parmsFromNormalization):
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
    y_pred_lr = regression_with_LR(X_train, y_train, X_test, y_test, parmsFromNormalization)


    #Random Forest Regressor
    #rfr = RandomForestRegressor(n_estimators=300, min_samples_split=10)
    #rfr = RandomForestRegressor()
    #rfr.fit(X_train, y_train)

    #y_pred_rfr = rfr.predict(X_test)


    #print_regression_model_summary("RFR", y_test, y_pred_rfr)
    y_pred_rfr = regression_with_RFR(X_train, y_train, X_test, y_test, parmsFromNormalization)

    #GradientBoostingRegressor
    #params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
    #      'learning_rate': 0.01, 'loss': 'ls'}
    #gfr = GradientBoostingRegressor(**params)
    #gfr.fit(X_train, y_train)
    #y_pred_gbr = gfr.predict(X_test)
    #print_regression_model_summary("GBR", y_test, y_pred_gbr)

    y_pred_gbr = regression_with_GBR(X_train, y_train, X_test, y_test, parmsFromNormalization)

    print("All forecasts are done")

    #average forecast
    #all_forecasts = np.column_stack((y_pred_lr, y_pred_dl, y_pred_rfr, y_pred_gbr))
    all_forecasts = np.column_stack((y_pred_lr, y_pred_rfr, y_pred_gbr))
    run_regression_ensamble(all_forecasts, y_test, parmsFromNormalization)

    #combine_models(all_forecasts, y_test)

    forecast_data = np.column_stack((np.array(list(range(0,y_test.shape[0],1))), y_test, y_pred_lr, y_pred_rfr, y_pred_gbr, X_test))

    headers='X0'
    for i in range(1,X_test.shape[1]):
        headers = headers + ",X"+ str(i)

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


def run_regression_ensamble(models, y_test, parmsFromNormalization):
    training_set_size = int(len(y_test)*.7)
    X_train, X_test, y_train, y_test = train_test_split(training_set_size, models, y_test)
    print("results for combined Models")
    y_pred_lr = regression_with_LR(X_train, y_train, X_test, y_test, parmsFromNormalization)


class RegressionEnsamble:
    def __init__(self, model):
        self.model = model

    #models should be able to correct normalizations
    def train(self, models, x_train, y_train, x_test, y_test):
        x_train_predictions = np.hstack([m.predict(x_train) for m in models])
        x_test_predictions = np.hstack([m.predict(x_test) for m in models])

        lr = LinearRegression(normalize=True)
        lr.fit(x_train_predictions, y_train)
        y_pred_lr = lr.predict(x_test_predictions)
        print_regression_model_summary("LR", y_test, y_pred_lr, ParmsFromNormalization(mean=0,std=1,sqrtx2=1)())
        self.model = lr
        return lr

    def predict(self, x_data):
        return self.model.predict(x_data)