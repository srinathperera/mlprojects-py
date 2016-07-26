import numpy as np
import time as time

import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from xgboost import plot_importance

import operator
from matplotlib import pylab as plt
import xgboost




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
    def __init__(self, model, isgbtree=False):
        self.model = model
        self.isgbtree = isgbtree

    def predict(self, data):
        if self.isgbtree:
            return self.model.predict(xgb.DMatrix(data), ntree_limit=self.model.best_ntree_limit)
        else:
            return self.model.predict(xgb.DMatrix(data))
'''
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


def regression_with_xgboost2(X_train, Y_train, X_test, Y_test, forecasting_feilds=None):
    #http://datascience.stackexchange.com/questions/9483/xgboost-linear-regression-output-incorrect
    #http://xgboost.readthedocs.io/en/latest/get_started/index.html
    #https://www.kaggle.com/c/higgs-boson/forums/t/10286/customize-loss-function-in-xgboost


    #http://xgboost.readthedocs.io/en/latest/python/python_intro.html
    #Python API http://xgboost.readthedocs.io/en/latest/python/python_api.html
    #is this correct

    if X_test is not None and Y_test is not None:
        train_data = xgb.DMatrix(X_train, Y_train)
        test_data = xgb.DMatrix(X_test, Y_test)


        params['nthread'] = 4

        evallist  = [(test_data,'eval'), (train_data,'train')]
        num_round =50
        gbm = xgb.train( params, train_data, num_round, evallist, verbose_eval = True, early_stopping_rounds=5)
        y_pred = gbm.predict(xgb.DMatrix(X_test))


        #importance = gbm.get_fscore()
        #plot_importance(importance)
        #gbm.dump_model('xgb.fmap')
        #importance = gbm.get_fscore(fmap='xgb.fmap')
        #importance = sorted(importance.items(), key=operator.itemgetter(1))

        #print "importance=", importance

        return XGBoostModel(gbm), y_pred
    else:
        train_data = xgb.DMatrix(X_train, Y_train)

        params = {"objective": "reg:linear", "booster":"gblinear"}
        #params = {"objective": "reg:linear", "booster":"gblinear"}
        params['nthread'] = 4

        evallist  = [(train_data,'train')]
        num_round = 10
        gbm = xgb.train( params, train_data, num_round, evallist, verbose_eval = True)
        y_pred = gbm.predict(xgb.DMatrix(X_test))
        return XGBoostModel(gbm), y_pred

'''


def ceate_feature_map_for_feature_importance(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()

def show_feature_importance(gbdt):
    importance = gbdt.get_fscore(fmap='xgb.fmap')
    print importance
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    print "feature importance", df

    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.gcf().savefig('feature_importance_xgb.png')


def get_default_xgboost_params():
    #xgb_params = {"objective": "reg:linear", "booster":"gblinear"}
    #xgb_params = {"objective": "reg:linear", "eta": 0.1, "max_depth": 10, "seed": 42, "silent": 0}


    #basic version #parameters https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
    #xgb_params = {"objective": "reg:linear", "booster":"gblinear"}
    #xgb_params = {"objective": "reg:linear", "booster":"gblinear", 'eta':0.1, 'max_depth':2, 'alpha':0.2, 'lambda':0.8}
    #xgb_params = {"objective": "reg:linear", "booster":"gblinear", 'eta':0.1, 'gamma':1.0 , 'max_depth':3, 'min_child_weight':1}

    xgb_params = {"objective": "reg:linear", "booster":"gbtree", 'eta':0.1}
    #xgb_params['subsample'] = 0.5 #0.5-1, Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting.
    #xgb_params['min_child_weight'] = 3 #      #Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
    #xgb_params['max_depth'] = 3 #Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
    #params['gamma'] = 0
    #params['alpha'] = 0.2 #L1
    #params['lambda'] = 0.2 #L2

    xgb_params['nthread'] = 4
    return xgb_params

def regression_with_xgboost_no_cv(x_train, y_train, X_test, Y_test, features=None, xgb_params=None, num_rounds = 10):
    train_data = xgb.DMatrix(x_train, label=y_train)
    test_data = xgb.DMatrix(X_test, Y_test)
    evallist  = [(train_data,'train'), (test_data,'eval')]

    if xgb_params is None:
        xgb_params = get_default_xgboost_params()
        print "xgb_params not found"

    print "XGBoost, using param", xgb_params
    gbdt = xgb.train(xgb_params, train_data, num_rounds, evallist, verbose_eval = True, early_stopping_rounds=5)

    isgbtree = xgb_params["booster"] == "gbtree"
    if isgbtree :
        ceate_feature_map_for_feature_importance(features)
        show_feature_importance(gbdt)
        y_pred = gbdt.predict(xgb.DMatrix(X_test), ntree_limit=gbdt.best_ntree_limit)
    else:
        y_pred = gbdt.predict(xgb.DMatrix(X_test))

    return XGBoostModel(gbdt), y_pred


def regression_with_xgboost(x_train, y_train, X_test, Y_test, features=None, use_cv=True, use_sklean=False, xgb_params=None):

    train_data = xgb.DMatrix(x_train, label=y_train)
    test_data = xgb.DMatrix(X_test, Y_test)
    evallist  = [(test_data,'eval'), (train_data,'train')]

    if xgb_params == None:
        xgb_params = get_default_xgboost_params()

    if not use_cv:
        num_rounds = 10
    else:
        cvresult = xgb.cv(xgb_params, train_data, num_boost_round=30, nfold=5,
            metrics={'rmse'}, show_progress=True)
        print cvresult
        num_rounds = len(cvresult)
    gbdt = None
    if(use_sklean):
        #gbdt = xgboost.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='reg:linear', nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0, missing=None)
        xgb_params['n_estimators'] = num_rounds
        gbdt = xgboost.XGBRegressor(xgb_params)

        gbdt.fit(x_train, y_train)
        y_pred = gbdt.predict(X_test)

        return gbdt, y_pred
    else:
        #gbdt = xgb.train( xgb_params, train_data, num_rounds, evallist, verbose_eval = True, early_stopping_rounds=5)
        gbdt = xgb.train( xgb_params, train_data, num_rounds, evallist, verbose_eval = True)

        #ceate_feature_map_for_feature_importance(features)
        #show_feature_importance(gbdt)

        y_pred = gbdt.predict(xgb.DMatrix(X_test))
        return XGBoostModel(gbdt), y_pred





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