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
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from mltools import *
from inventory_demand import *
import scipy

def cal_rmsle(a,b):
    val = np.abs(np.log(a+1) - np.log(b+1))
    return val

def is_forecast_improved(org, new, actual):
    return cal_rmsle(org, actual) - cal_rmsle(new, actual) > 0


class SimpleAvgEnsamble:
    def __init__(self, conf, method):
        self.conf = conf
        self.method = method
        self.name = "SimpleAvg_"+method

    def fit(self, forecasts, best_rmsle_index, y_actual):
        start = time.time()
        forecast = self.predict(forecasts, best_rmsle_index)
        calculate_accuracy(self.method + "_forecast " + str(self.conf.command), y_actual, forecast)

        #hmean_forecast = scipy.stats.hmean(forecasts, axis=1)
        #calculate_accuracy("hmean_forecast", y_actual, hmean_forecast)
        print_time_took(start, self.method + "_forecast " + str(self.conf.command))
        return forecast

    def predict(self, forecasts, best_rmsle_index):
        if self.method == 'median':
            forecast = np.median(forecasts, axis=1)
        elif self.method == 'mean':
            forecast = np.mean(forecasts, axis=1)
        return forecast


class BestPairEnsamble:
    def __init__(self, conf, method="mean"):
        self.conf = conf
        self.method = method
        self.name = "BestPairEnsamble_"+method
    def fit(self, forecasts, best_rmsle_index, y_actual):
        start = time.time()
        comb = list(itertools.combinations(range(forecasts.shape[1]),2))
        rmsle_values = []
        for (a,b) in comb:
            forecast = self.predict_pair(forecasts[:,a], forecasts[:,b])
            rmsle = calculate_accuracy("try " +self.method + " pair " + str((a,b)) , y_actual, forecast)
            rmsle_values.append(rmsle)

        best_index = np.argmin(rmsle_values)
        self.best_pair = comb[best_index]
        print "best mean pair value, " + str(self.conf.command), str(self.best_pair), 'rmsle=', rmsle_values[best_index]
        print_time_took(start, self.method + "_forecast " + str(self.conf.command))
        return forecast

    def predict(self, forecasts, best_rmsle_index):
        (a,b) = self.best_pair
        return self.predict_pair(forecasts[:,a], forecasts[:,b])

    def predict_pair(self, f1,f2 ):
        forecasts = np.column_stack([f1,f2])
        if self.method == 'median':
            forecast = np.median(forecasts, axis=1)
        elif self.method == 'mean':
            forecast = np.mean(forecasts, axis=1)
        return forecast


class BestThreeEnsamble:
    def __init__(self, conf, method="mean"):
        self.conf = conf
        self.method = method
        self.name = "BestThreeEnsamble_"+method
        self.best_triple = None

    def fit(self, forecasts, best_rmsle_index, y_actual):
        start = time.time()
        comb = list(itertools.combinations(range(forecasts.shape[1]),3))
        rmsle_values = []
        for (a,b,c) in comb:
            forecast = self.predict_triple(forecasts[:,a], forecasts[:,b], forecasts[:,b])
            rmsle = calculate_accuracy("try " +self.method + " triple " + str((a,b)) , y_actual, forecast)
            rmsle_values.append(rmsle)

        best_index = np.argmin(rmsle_values)
        self.best_triple = comb[best_index]
        print "best triple, " + str(self.conf.command), str(self.best_triple), 'rmsle=', rmsle_values[best_index]
        print_time_took(start, self.name + str(self.conf.command))
        return forecast

    def predict(self, forecasts, best_rmsle_index):
        (a,b, c) = self.best_triple
        return self.predict_triple(forecasts[:, a], forecasts[:, b], forecasts[:, c])

    def predict_triple(self, f1,f2, f3):
        forecasts = np.column_stack([f1,f2, f3])
        if self.method == 'median':
            forecast = np.median(forecasts, axis=1)
        elif self.method == 'mean':
            forecast = np.mean(forecasts, axis=1)
        return forecast


def generate_forecast_features(forecasts, model_index_by_acc):
    sorted_forecats = forecasts[:,model_index_by_acc]
    sorted_forecats = transfrom_to_log2d(sorted_forecats)
    diff_best_two = np.abs(sorted_forecats[:, 1] - sorted_forecats[:, 0])
    min_diff_to_best = np.min(np.abs(sorted_forecats[:, 2:] - sorted_forecats[:, 0].reshape((-1,1))), axis=1)
    min_diff_to_second = np.min(np.abs(sorted_forecats[:, 2:] - sorted_forecats[:, 1].reshape((-1,1))), axis=1)
    avg_two = np.mean(sorted_forecats[:, 0:2], axis=1)
    std_all = np.mean(sorted_forecats, axis=1)
    weighted_mean = 0.34*sorted_forecats[:, 0] + 0.33*sorted_forecats[:, 0] + 0.33*np.median(sorted_forecats[:, 2:])

    kurtosis = scipy.stats.kurtosis(sorted_forecats, axis=1)
    kurtosis = np.where(np.isnan(kurtosis), 0, kurtosis)
    hmean = scipy.stats.hmean(np.where(sorted_forecats <= 0, 0.1, sorted_forecats), axis=1)
    hmean = np.where(np.isnan(hmean), 0, hmean)
    #entropy = scipy.stats.entropy(sorted_forecats, axis=1)
    #entropy = np.where(np.isnan(entropy), 0, entropy)


    return np.column_stack([sorted_forecats, kurtosis, hmean, diff_best_two, min_diff_to_best,
            min_diff_to_second, avg_two, std_all,weighted_mean]),['f'+str(f) for f in range(sorted_forecats.shape[1])] \
            + ["kurtosis", "hmean", "diff_best_two", "min_diff_to_best", "min_diff_to_second", "avg_two", "std_all","weighted_mean"]


def vote_based_forecast(forecasts, best_model_index, y_actual=None):
    start = time.time()
    limit = 0.1
    best_forecast = forecasts[:, best_model_index]

    forecasts = np.sort(np.delete(forecasts, best_model_index, axis=1), axis=1)
    forecasts = np.where(forecasts <=0, 0.1, forecasts)

    final_forecast = np.zeros((forecasts.shape[0],))
    same = 0
    replaced = 0
    averaged = 0

    same_c = 0
    replaced_c = 0
    averaged_c = 0

    data_train = []

    for i in range(forecasts.shape[0]):
        f_row = forecasts[i,]
        min_diff_to_best = np.min([cal_rmsle(best_forecast[i], f) for f in f_row])
        comb = list(itertools.combinations(f_row,2))
        avg_error = scipy.stats.hmean([cal_rmsle(x,y) for (x,y) in comb])

        if min_diff_to_best < limit:
            final_forecast[i] = best_forecast[i]
            same = same +1
            if y_actual is not None:
                same_c = same_c + 1 if is_forecast_improved(best_forecast[i], final_forecast[i], y_actual[i]) else 0
        else:
            if avg_error < 2*limit:
                final_forecast[i] = np.median(f_row)
                replaced = replaced +1
                if y_actual is not None and is_forecast_improved(best_forecast[i], final_forecast[i], y_actual[i]):
                    replaced_c = replaced_c + 1
                #print best_forecast[i], '->', final_forecast[i], y_actual[i], cal_rmsle(final_forecast[i], y_actual[i])
            else:
                final_forecast[i] = 0.6*best_forecast[i] + 0.4*scipy.stats.hmean(f_row)
                averaged = averaged + 1 if is_forecast_improved(best_forecast[i], final_forecast[i], y_actual[i]) else 0
                if y_actual is not None and is_forecast_improved(best_forecast[i], final_forecast[i], y_actual[i]):
                    averaged_c = averaged_c + 1 if cal_rmsle(final_forecast[i], y_actual[i]) < 0.1 else 0
        data_train.append([min_diff_to_best, avg_error, scipy.stats.hmean(f_row), np.median(f_row), np.std(f_row)])

    print "same, replaced, averaged", same, replaced, averaged
    print "same_c, replaced_c, averaged_c", same_c, replaced_c, averaged_c
    print_time_took(start, "vote_based_forecast")
    return final_forecast

    #ff = np.when(np.abs(best_model_index - best_forecast) < limit, best_forecast,
    #)


def vote_with_lr(conf, forecasts, best_model_index, y_actual):
    start = time.time()
    best_forecast = forecasts[:, best_model_index]
    forecasts = np.sort(np.delete(forecasts, best_model_index, axis=1), axis=1)
    forecasts = np.where(forecasts <=0, 0.1, forecasts)

    data_train = []

    for i in range(forecasts.shape[0]):
        f_row = forecasts[i,]
        min_diff_to_best = np.min([cal_rmsle(best_forecast[i], f) for f in f_row])
        comb = list(itertools.combinations(f_row,2))
        avg_error = scipy.stats.hmean([cal_rmsle(x,y) for (x,y) in comb])
        data_train.append([min_diff_to_best, avg_error, scipy.stats.hmean(f_row), np.median(f_row), np.std(f_row)])


    X_all = np.column_stack([np.row_stack(data_train), best_forecast])
    if conf.target_as_log:
        y_actual = transfrom_to_log(y_actual)
    #we use 10% full data to train the ensamble and 30% for evalaution
    no_of_training_instances = int(round(len(y_actual)*0.25))
    X_train, X_test, y_train, y_test = train_test_split(no_of_training_instances, X_all, y_actual)
    y_actual_test = y_actual[no_of_training_instances:]

    lr_model =linear_model.Lasso(alpha = 0.2)
    lr_model.fit(X_train, y_train)
    lr_forecast = lr_model.predict(X_test)
    lr_forcast_revered = retransfrom_from_log(lr_forecast)
    calculate_accuracy("vote__lr_forecast " + str(conf.command), y_actual_test, lr_forcast_revered)
    print_time_took(start, "vote_with_lr")
    return lr_forcast_revered

def get_blend_features():
     return ['Semana','clients_combined_Mean', 'Producto_ID_Demanda_uni_equil_Mean']


def blend_models(conf, forecasts, model_index_by_acc, y_actual, submissions_ids, submissions,
                 blend_data, blend_data_submission):
    use_complex_features = False
    if use_complex_features:
        X_all, forecasting_feilds = generate_forecast_features(forecasts, model_index_by_acc)
    else:
        X_all,forecasting_feilds = forecasts, ["f"+str(f) for f in range(forecasts.shape[1])]

    X_all = np.column_stack([X_all, blend_data])
    forecasting_feilds = forecasting_feilds + get_blend_features()

    #removing NaN and inf if there is any
    X_all = np.where(np.isnan(X_all), 0, np.where(np.isinf(X_all), 10000, X_all))
    y_actual_saved = y_actual
    if conf.target_as_log:
        y_actual = transfrom_to_log(y_actual)

    #we use 10% full data to train the ensamble and 30% for evalaution
    no_of_training_instances = int(round(len(y_actual)*0.50))
    X_train, X_test, y_train, y_test = train_test_split(no_of_training_instances, X_all, y_actual)
    y_actual_test = y_actual_saved[no_of_training_instances:]

    rfr = RandomForestRegressor(n_jobs=4, oob_score=True)
    rfr.fit(X_train, y_train)
    print_feature_importance(rfr.feature_importances_, forecasting_feilds)
    rfr_forecast_as_log = rfr.predict(X_test)
    rfr_forecast = retransfrom_from_log(rfr_forecast_as_log)
    rmsle = calculate_accuracy("rfr_forecast", y_actual_test, rfr_forecast)

    if submissions_ids is not None and submissions is not None:
        if use_complex_features:
            submissions, _ = generate_forecast_features(submissions, model_index_by_acc)
        submissions = np.column_stack([submissions, blend_data_submission])

        submissions = np.where(np.isnan(submissions), 0, np.where(np.isinf(submissions), 10000, submissions))
        rfr_ensamble_forecasts = rfr.predict(submissions)
        if conf.target_as_log:
            rfr_ensamble_forecasts = retransfrom_from_log(rfr_ensamble_forecasts)
        save_submission_file("rfr_blend_submission.csv", submissions_ids, rfr_ensamble_forecasts)
    else:
        print "submissions not found"

    lr_model =linear_model.Lasso(alpha = 0.1)
    lr_model.fit(X_train, y_train)
    lr_forecast = lr_model.predict(X_test)
    lr_forcast_revered = retransfrom_from_log(lr_forecast)
    calculate_accuracy("vote__lr_forecast " + str(conf.command), y_actual_test, lr_forcast_revered)

    return rfr_forecast, rmsle


def avg_models(conf, forecasts, y_actual, blend_features, submission_forecasts=None, test=False, submission_ids=None, sub_df=None):
    start = time.time()

    #add few more features
    use_features = False
    if use_features:
        X_all = np.column_stack([forecasts, blend_features])

        forecasting_feilds = ["f"+str(f) for f in range(forecasts.shape[1])] \
                             + ["Semana", "clients_combined_Mean", 'Producto_ID_Demanda_uni_equil_Mean']
    else:
        X_all = forecasts
        forecasting_feilds = ["f"+str(f) for f in range(forecasts.shape[1])]


    #removing NaN and inf if there is any
    X_all = np.where(np.isnan(X_all), 0, np.where(np.isinf(X_all), 10000, X_all))
    print "X_all"
    check4nan(X_all)
    print "Y_all"
    check4nan(y_actual)

    y_actual_saved = y_actual

    target_as_log = True
    if target_as_log:
        y_actual = transfrom_to_log(y_actual)



    #we use 10% full data to train the ensamble and 30% for evalaution
    no_of_training_instances = int(round(len(y_actual)*0.25))
    X_train, X_test, y_train, y_test = train_test_split(no_of_training_instances, X_all, y_actual)
    y_actual_test = y_actual_saved[no_of_training_instances:]

    ensambles = []

    rfr = RandomForestRegressor(n_jobs=4, oob_score=True, max_depth=3)
    rfr.fit(X_train, y_train)
    print_feature_importance(rfr.feature_importances_, forecasting_feilds)
    rfr_forecast = rfr.predict(X_test)
    rmsle = calculate_accuracy("rfr_forecast", y_actual_test, retransfrom_from_log(rfr_forecast))
    ensambles.append((rmsle, rfr, "rfr ensamble"))

    xgb_params = {"objective": "reg:linear", "booster":"gbtree", "eta":0.1, "nthread":4 }
    model, y_pred = regression_with_xgboost(X_train, y_train, X_test, y_test, features=forecasting_feilds, use_cv=True,
                            use_sklean=False, xgb_params=xgb_params)
    #model, y_pred = regression_with_xgboost_no_cv(X_train, y_train, X_test, y_test, features=forecasting_feilds,
    #                                                  xgb_params=xgb_params,num_rounds=20)
    xgb_forecast = model.predict(X_test)
    rmsle = calculate_accuracy("xgb_forecast", y_actual_test, retransfrom_from_log(xgb_forecast))
    ensambles.append((rmsle, model, "xgboost ensamble"))


    lr_model =linear_model.Lasso(alpha = 0.2)
    lr_model.fit(X_train, y_train)
    lr_forecast = lr_model.predict(X_test)
    rmsle = calculate_accuracy("lr_forecast", y_actual_test, retransfrom_from_log(lr_forecast))
    ensambles.append((rmsle, lr_model, "rfr ensamble"))



    best_ensamble_index = np.argmin([t[0] for t in ensambles])
    best_ensamble = ensambles[best_ensamble_index][1]
    print "[IDF]Best Ensamble", ensambles[best_ensamble_index][2], ensambles[best_ensamble_index][0]


    if submission_forecasts is not None:
        median_forecast = np.median(submission_forecasts, axis=1)
        if use_features:
            list = [submission_forecasts, median_forecast, sub_df['Semana'],
                             sub_df['clients_combined_Mean'], sub_df['Producto_ID_Demanda_uni_equil_Mean']]
            print "sizes", [ a.shape for a in list]
            sub_x_all = np.column_stack(list)
        else:
            sub_x_all = submission_forecasts

        ensamble_forecast = best_ensamble.predict(sub_x_all)

        to_save = np.column_stack((submission_ids, ensamble_forecast))
        to_saveDf =  pd.DataFrame(to_save, columns=["id","Demanda_uni_equil"])
        to_saveDf = to_saveDf.fillna(0)
        to_saveDf["id"] = to_saveDf["id"].astype(int)
        submission_file = 'en_submission'+str(conf.command)+ '.csv'
        to_saveDf.to_csv(submission_file, index=False)

        print "Best Ensamble Submission Stats"
        print to_saveDf.describe()




    print "avg_models took ", (time.time() - start), "s"
    '''
    if len(models) >= 3:

        print "toprmsle_valuesrmsle", rmsle_values
        sorted_index = np.argsort(rmsle_values)
        print "sorted_index", sorted_index
        bestindexes = sorted_index[0:3]
        print "top3indexes", bestindexes
        top3forecasts = forecasts[:,bestindexes]

        weighted_forecast = top3forecasts[:,0]*0.6+ top3forecasts[:,1]*0.25+ top3forecasts[:,2]*0.15
        calculate_accuracy("weighted_forecast", y_actual, weighted_forecast)

        weighted_forecast1 = top3forecasts[:,0]*0.5+ top3forecasts[:,1]*0.3+ top3forecasts[:,2]*0.2
        calculate_accuracy("weighted_forecast1", y_actual, weighted_forecast1)
        top3rmsle =  rmsle_values[top3forecasts]


        #top3rmsle = forecasts[:,top3index]
        l1, h1 = find_range(top3rmsle[0], top3forecasts[0])
        l2, h2 = find_range(top3rmsle[1], top3forecasts[1])

        math_based_forecast = np.where(l2 < h1, (l2-h1)/2, np.where(l1 < h2, (l1-h2)/2,top3forecasts) )
        calculate_accuracy("math_based_forecast", y_actual, weighted_forecast1)
        #if l2 < h1:(h1+l2)/2
        #if l1 < h2:(h2+l1)/2
        #else top3forecasts[2]

        '''
