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
from mlpreprocessing import *
import scipy

def cal_rmsle(a,b):
    val = np.abs(np.log(a+1) - np.log(b+1))
    return val

def is_forecast_improved(org, new, actual):
    return cal_rmsle(org, actual) - cal_rmsle(new, actual) > 0


class SimpleAvgEnsamble:
    def __init__(self, conf, method, models2use=4):
        self.conf = conf
        self.method = method
        self.name = "SimpleAvg_"+method
        self.models2use = models2use

    def fit(self, forecasts, best_rmsle_index, y_actual):
        start = time.time()
        forecast = self.predict(forecasts, best_rmsle_index)
        calculate_accuracy(self.method + "_forecast " + str(self.conf.command), y_actual, forecast)

        #hmean_forecast = scipy.stats.hmean(forecasts, axis=1)
        #calculate_accuracy("hmean_forecast", y_actual, hmean_forecast)
        print_time_took(start, self.method + "_forecast " + str(self.conf.command))
        return forecast

    def predict(self, forecasts, best_rmsle_index):
        forecasts = forecasts[:, :self.models2use]
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
        print "[IDF]best mean pair value, " + str(self.conf.command), str(self.best_pair), 'rmsle=', rmsle_values[best_index]
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

class BestPairLogEnsamble:
    def __init__(self, conf, method="mean"):
        self.conf = conf
        self.method = method
        self.name = "BestPairEnsamble_"+method

    def fit(self, forecasts, y_actual):
        start = time.time()
        comb = list(itertools.combinations(range(forecasts.shape[1]),2))
        random.shuffle(comb)
        comb = comb[:min(50, len(comb))]
        rmsle_values = []
        for i, (a,b) in enumerate(comb):
            x = transfrom_to_log(forecasts[:,a])
            y = transfrom_to_log(forecasts[:,b])
            forecast = self.predict_pair(x,y)
            forecast = retransfrom_from_log(forecast)
            rmsle = calculate_accuracy(str(i) +" try " +self.method + " pair " + str((a,b)) , y_actual, forecast)
            rmsle_values.append(rmsle)

        best_index = np.argmin(rmsle_values)
        self.best_pair = comb[best_index]
        print "[IDF]best mean pair value, " + str(self.conf.command), str(self.best_pair), 'rmsle=', rmsle_values[best_index]
        print_time_took(start, self.method + "_forecast " + str(self.conf.command))
        return forecast

    def predict(self, forecasts):
        forecasts = transfrom_to_log(forecasts)
        (a,b) = self.best_pair
        final_forecast = self.predict_pair(forecasts[:,a], forecasts[:,b])
        final_forecast = retransfrom_from_log(final_forecast)
        return final_forecast


    def predict_pair(self, f1,f2):
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
            rmsle = calculate_accuracy("try " +self.method + " triple " + str((a,b,c)) , y_actual, forecast)
            rmsle_values.append(rmsle)

        best_index = np.argmin(rmsle_values)
        self.best_triple = comb[best_index]
        print "[IDF]best triple, " + str(self.conf.command), str(self.best_triple), 'rmsle=', rmsle_values[best_index]
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


def predict_using_veriation(forecasts_data, best_forecast, y_actual, frac = 1.0):
    size_to_keep = int(forecasts_data.shape[0]*frac)
    forecasts_data = forecasts_data[:size_to_keep]
    y_actual = y_actual[:size_to_keep]

    forecasts_data = transfrom_to_log2d(forecasts_data)
    forecasts_stdev = np.std(forecasts_data, axis=1)
    forecasts_mean = np.mean(forecasts_data, axis=1)
    forecasts_hmean = fillna_and_inf(scipy.stats.hmean(np.where(forecasts_data <= 0, 0.1, forecasts_data), axis=1))
    min_diff_to_best = np.min(np.abs(forecasts_data - best_forecast.reshape((-1,1))), axis=1)
    diff_best_to_mean = np.abs(best_forecast - forecasts_mean)

    print "forecasts_stdev", basic_stats_as_str(forecasts_stdev)
    print "forecasts_mean", basic_stats_as_str(forecasts_mean)
    print "diff_best_to_mean", basic_stats_as_str(diff_best_to_mean)
    print "min_diff_to_best", basic_stats_as_str(min_diff_to_best)

    final_forecast = np.zeros(size_to_keep)
    for i in range(size_to_keep):
        if min_diff_to_best[i] < 0.2 or diff_best_to_mean[i] < 0.3:
            final_forecast[i] = best_forecast[i]
        elif forecasts_stdev[i] < 0.3:
            final_forecast[i] = forecasts_mean[i]
        else:
            final_forecast[i] = (forecasts_hmean[i] + best_forecast[i])/2

    calculate_accuracy("predict_using_veriation", y_actual, final_forecast)
    return final_forecast



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
    use_complex_features = True
    if use_complex_features:
        X_all, forecasting_feilds = generate_forecast_features(forecasts, model_index_by_acc)
    else:
        X_all,forecasting_feilds = forecasts, ["f"+str(f) for f in range(forecasts.shape[1])]

    X_all = np.column_stack([X_all, blend_data])
    forecasting_feilds = forecasting_feilds + get_blend_features()

    #removing NaN and inf if there is any
    y_actual_saved = y_actual
    if conf.target_as_log:
        X_all = transfrom_to_log2d(X_all)
        y_actual = transfrom_to_log(y_actual)

    X_all = fillna_and_inf(X_all)
    y_actual = fillna_and_inf(y_actual)

    #we use 10% full data to train the ensamble and 30% for evalaution
    no_of_training_instances = int(round(len(y_actual)*0.50))
    X_train, X_test, y_train, y_test = train_test_split(no_of_training_instances, X_all, y_actual)
    y_actual_test = y_actual_saved[no_of_training_instances:]

    '''
    rfr = RandomForestRegressor(n_jobs=4, oob_score=True)
    rfr.fit(X_train, y_train)
    print_feature_importance(rfr.feature_importances_, forecasting_feilds)
    rfr_forecast_as_log = rfr.predict(X_test)
    rfr_forecast = retransfrom_from_log(rfr_forecast_as_log)
    rmsle = calculate_accuracy("rfr_forecast", y_actual_test, rfr_forecast)


    lr_model =linear_model.Lasso(alpha = 0.1)
    lr_model.fit(X_train, y_train)
    lr_forecast = lr_model.predict(X_test)
    lr_forcast_revered = retransfrom_from_log(lr_forecast)
    calculate_accuracy("vote__lr_forecast " + str(conf.command), y_actual_test, lr_forcast_revered)
    '''

    xgb_params = {"objective": "reg:linear", "booster":"gbtree", "eta":0.1, "nthread":4, 'min_child_weight':5}
    model, y_pred = regression_with_xgboost(X_train, y_train, X_test, y_test, features=forecasting_feilds, use_cv=True,
                            use_sklean=False, xgb_params=xgb_params)
    #model, y_pred = regression_with_xgboost_no_cv(X_train, y_train, X_test, y_test, features=forecasting_feilds,
    #                                                  xgb_params=xgb_params,num_rounds=100)
    xgb_forecast = model.predict(X_test)
    xgb_forecast = retransfrom_from_log(xgb_forecast)
    calculate_accuracy("xgb_forecast", y_actual_test, xgb_forecast)

    if submissions_ids is not None and submissions is not None:
        if use_complex_features:
            submissions, _ = generate_forecast_features(submissions, model_index_by_acc)
        submissions = np.column_stack([submissions, blend_data_submission])
        submissions = np.where(np.isnan(submissions), 0, np.where(np.isinf(submissions), 10000, submissions))
        rfr_ensamble_forecasts = model.predict(submissions)
        if conf.target_as_log:
            rfr_ensamble_forecasts = retransfrom_from_log(rfr_ensamble_forecasts)
        save_submission_file("rfr_blend_submission.csv", submissions_ids, rfr_ensamble_forecasts)
    else:
        print "submissions not found"

    #we randomly select 5 million values
    x_size = X_train.shape[0]
    sample_indexes = np.random.randint(0, X_train.shape[0], min(5000000, x_size))
    X_train = X_train[sample_indexes]
    y_train = y_train[sample_indexes]

    dlconf = MLConfigs(nodes_in_layer=10, number_of_hidden_layers=2, dropout=0.3, activation_fn='relu', loss="mse",
                epoch_count=4, optimizer=Adam(lr=0.0001), regularization=0.2)
    y_train, parmsFromNormalization = preprocess1DtoZeroMeanUnit(y_train)
    y_test = apply_zeroMeanUnit(y_test, parmsFromNormalization)
    X_train, parmsFromNormalization2D = preprocess2DtoZeroMeanUnit(X_train)
    X_test = apply_zeroMeanUnit2D(X_test, parmsFromNormalization2D)

    model, y_forecast = regression_with_dl(X_train, y_train, X_test, y_test, dlconf)

    y_forecast = undoPreprocessing(y_forecast, parmsFromNormalization)
    y_forecast = retransfrom_from_log(y_forecast)
    rmsle = calculate_accuracy("ml_forecast", y_actual_test, y_forecast)


    #return xgb_forecast, rmsle

def avg_models(conf, blend_forecasts_df, y_actual, submission_forecasts_df, submission_ids=None, xgb_params=None, do_cv=True, frac=1.0):
    print "start avg models"
    start = time.time()

    forecasting_feilds = list(blend_forecasts_df)
    print "Using features", forecasting_feilds

    X_all = blend_forecasts_df.values
    sub_X_all = submission_forecasts_df.values

    if frac < 1:
        data_size = int(blend_forecasts_df.shape[0]*frac)
        X_all = X_all[:data_size, :]
        y_actual = y_actual[:data_size]


    #removing NaN and inf if there is any
    X_all = fillna_and_inf(X_all)

    y_actual_saved = y_actual

    target_as_log = True
    if target_as_log:
        y_actual = transfrom_to_log(y_actual)

    #we use 10% full data to train the ensamble and 30% for evalaution
    no_of_training_instances = int(round(len(y_actual)*0.5))
    X_train, X_test, y_train, y_test = train_test_split(no_of_training_instances, X_all, y_actual)
    y_actual_test = y_actual_saved[no_of_training_instances:]

    ensambles = []
    '''
    rfr = RandomForestRegressor(n_jobs=4, oob_score=True, max_depth=3)
    rfr.fit(X_train, y_train)
    print_feature_importance(rfr.feature_importances_, forecasting_feilds)
    rfr_forecast = rfr.predict(X_test)
    rmsle = calculate_accuracy("rfr_forecast", y_actual_test, retransfrom_from_log(rfr_forecast))
    ensambles.append((rmsle, rfr, "rfr ensamble"))

    lr_model =linear_model.Lasso(alpha = 0.2)
    lr_model.fit(X_train, y_train)
    lr_forecast = lr_model.predict(X_test)
    rmsle = calculate_accuracy("lr_forecast", y_actual_test, retransfrom_from_log(lr_forecast))
    ensambles.append((rmsle, lr_model, "lr ensamble"))

    '''
    do_xgb = True

    if do_xgb:
        if xgb_params is None:
            xgb_params = {"objective": "reg:linear", "booster":"gbtree", "eta":0.1, "nthread":4 }
        if do_cv:
            model, y_pred = regression_with_xgboost(X_train, y_train, X_test, y_test, features=forecasting_feilds, use_cv=True,
                                use_sklean=False, xgb_params=xgb_params)
        else:
            model, y_pred = regression_with_xgboost_no_cv(X_train, y_train, X_test, y_test, features=forecasting_feilds,
                                                          xgb_params=xgb_params,num_rounds=200)
        xgb_forecast = model.predict(X_test)
        xgb_forecast_actual = retransfrom_from_log(xgb_forecast)
        rmsle = calculate_accuracy(str(xgb_params) + "[IDF]xgb_forecast", y_actual_test, xgb_forecast_actual)
        ensambles.append((rmsle, model, "xgboost ensamble"))

        best_ensamble_index = np.argmin([t[0] for t in ensambles])
        best_ensamble = ensambles[best_ensamble_index][1]
        print "[IDF]Best Ensamble", ensambles[best_ensamble_index][2], ensambles[best_ensamble_index][0]

        if sub_X_all is not None:
            ensamble_forecast = best_ensamble.predict(sub_X_all)
            ensamble_forecast = retransfrom_from_log(ensamble_forecast)

            #becouse forecast cannot be negative
            ensamble_forecast = np.where(ensamble_forecast < 0, 0, ensamble_forecast)

            to_save = np.column_stack((submission_ids, ensamble_forecast))
            to_saveDf =  pd.DataFrame(to_save, columns=["id","Demanda_uni_equil"])
            to_saveDf = to_saveDf.fillna(0)
            to_saveDf["id"] = to_saveDf["id"].astype(int)
            submission_file = 'xgb_ensamble_submission_'+ str(time.time()) +'.csv'
            to_saveDf.to_csv(submission_file, index=False)

            print "Best Ensamble Submission Stats", submission_file
            print to_saveDf.describe()


        print "avg_models took ", (time.time() - start), "s"
        return xgb_forecast_actual, y_actual_test, ensamble_forecast



def avg_models_with_ml(conf, blend_forecasts_df, y_actual, submission_forecasts_df, submission_ids=None):
    blend_forecasts_df['target'] = y_actual

    blend_forecasts_df = blend_forecasts_df.sample(min(5000000, blend_forecasts_df.shape[0]))
    y_all = blend_forecasts_df['target'].values
    blend_forecasts_df = drop_feilds_1df(blend_forecasts_df, ['target'])
    blend_forecasts_df = blend_forecasts_df.copy()
    submission_forecasts_df = submission_forecasts_df.copy()

    train_df, test_df, y_actual_train, y_actual_test = prepare_train_and_test_data(blend_forecasts_df,
                                                                                  y_actual_data=y_all, split_frac=0.6)

    models = get_models4dl_only(conf)
    models, forecasts, submission_forecasts = do_forecast(conf, train_df, test_df, submission_forecasts_df,
                                                          y_actual_train, y_actual_test, models=models)
    y_forecast = forecasts[:, 0]
    print_mem_usage("after forecast")
    try:
        y_forecast_df =  pd.DataFrame(y_forecast.reshape(-1,1), columns=["target"])
        save_file("temp", 0, y_forecast_df, 'forecast_ml')
        y_forecast_df = load_file("temp", 0, 'forecast_ml', throw_error=True)
        calculate_accuracy("dl_forecast", y_actual_test, y_forecast_df['target'])
    except:
        print_mem_usage("after error")
        print "Unexpected error:"

    submission_forecast = submission_forecasts[:, 0]
    submission_forecast = np.where(submission_forecast < 0, 0, submission_forecast)
    submission_file = 'en_dl_submission'+str(conf.command)+ '.csv'
    save_submission_file(submission_file, submission_ids, submission_forecast)


def find_best_forecast_per_product(data_df, y_actual, sub_data_df, product_data, product_data_submission, submission_ids, frac=0.25):
    if frac < 1:
        full_data_size = data_df.shape[0]
        sample_size = int(full_data_size*frac)
        data_df = data_df.head(sample_size)
        product_data = product_data[:sample_size]
        y_actual = y_actual[:sample_size]

    feilds = {k: data_df[k] for k in list(data_df)}
    feilds['Producto_ID'] = pd.Series(product_data)
    data_df = pd.DataFrame(feilds)

    forecast_feilds = [f for f in list(data_df) if "." in f]
    errors = [product_data.values]
    errors = errors + [np.abs(np.log(1+data_df[f]) - np.log(1+y_actual)) for f in forecast_feilds]
    error_df = create_df_from_list(['Producto_ID'] + forecast_feilds, errors)

    grouped_error = error_df.groupby(['Producto_ID']).mean()
    grouped_error_vals = grouped_error.values
    best_forecast_index = np.argmin(grouped_error_vals, axis=1)
    best_forecast_index_df = create_df_from_list(['Producto_ID', "forecast_index"], [grouped_error.index, best_forecast_index])

    basedf = pd.merge(data_df, best_forecast_index_df, how='left', on=['Producto_ID'])

    best_forecast_index = basedf['forecast_index'].values
    forecast_options = basedf[forecast_feilds].values
    print forecast_options.shape
    print best_forecast_index.shape
    print basic_stats_as_str(best_forecast_index)

    forecast_size = best_forecast_index.shape[0]

    '''
    for i in range(forecast_size):
        findex = best_forecast_index[i]
        print findex
        per_product_forecast[i] = forecast_options[i, int(findex)]
    '''

    per_product_forecast = [forecast_options[i, int(best_forecast_index[i])] for i in range(forecast_size)]
    calculate_accuracy("best_forecast_per_product", y_actual, per_product_forecast)

    feilds = {k: sub_data_df[k] for k in list(sub_data_df)}
    feilds['Producto_ID'] = pd.Series(product_data_submission.values)
    sub_data_df = pd.DataFrame(feilds)

    sub_data_df = pd.merge(sub_data_df, best_forecast_index_df, how='left', on=['Producto_ID'])
    best_forecast_index_sub = fillna_and_inf(sub_data_df['forecast_index'].values)
    forecast_options_sub = fillna_and_inf(sub_data_df[forecast_feilds].values)
    per_product_forecast_submission = [forecast_options_sub[i, int(best_forecast_index_sub[i])] for i in range(forecast_options_sub.shape[0])]

    to_save = np.column_stack((submission_ids, per_product_forecast_submission))
    to_saveDf =  pd.DataFrame(to_save, columns=["id","Demanda_uni_equil"])
    to_saveDf = to_saveDf.fillna(0)
    to_saveDf["id"] = to_saveDf["id"].astype(int)
    submission_file = 'per_product_submission.csv'
    to_saveDf.to_csv(submission_file, index=False)
    print "Best Ensamble Submission Stats", submission_file

    return per_product_forecast, per_product_forecast_submission


def compare_submissions(file_list):
    submissions_df_list = [read_submission_file(f) for f in file_list]

    basedf = submissions_df_list[0]
    for i in range(1, len(submissions_df_list)):
        basedf = pd.merge(basedf, submissions_df_list[i], how='left', on=["id"])

    submission_ids = basedf['id']
    basedf = drop_feilds_1df(basedf, ['id'])

    submissions_data = basedf.values
    stddev_list = np.std(submissions_data, axis=1)
    print basic_stats_as_str(stddev_list)

    submissions_data = transfrom_to_log2d(submissions_data)
    mean_log_ensamble_forecast = retransfrom_from_log(np.mean(submissions_data, axis=1))
    mean_log_ensamble_forecast = np.where(mean_log_ensamble_forecast < 0, 0, mean_log_ensamble_forecast)

    to_save = np.column_stack([submission_ids, mean_log_ensamble_forecast])
    to_saveDf =  pd.DataFrame(to_save, columns=["id","Demanda_uni_equil"])
    to_saveDf = to_saveDf.fillna(0)
    to_saveDf["id"] = to_saveDf["id"].astype(int)
    submission_file = 'mean_log_ensamble_forecast.csv'
    to_saveDf.to_csv(submission_file, index=False)
    print "Best Ensamble Submission Stats", submission_file

