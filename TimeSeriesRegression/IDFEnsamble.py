
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import time
from sklearn.externals import joblib

from mltools import undoPreprocessing
import itertools



from tsforecasttools import run_timeseries_froecasts, regression_with_xgboost
from mltools import preprocess2DtoZeroMeanUnit, preprocess1DtoZeroMeanUnit, train_test_split, print_feature_importance, apply_zeroMeanUnit2D
from mltools import calculate_rmsle, almost_correct_based_accuracy, MLConfigs, print_regression_model_summary, \
    regression_with_dl, apply_zeroMeanUnit, undo_zeroMeanUnit2D
from keras.optimizers import Adam
from sklearn.linear_model import LinearRegression

from inventory_demand import *
from mltools import *
from inventory_demand_ensambles import *

from os import listdir
from os.path import isfile, join

import sys
print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)


command = -2
if len(sys.argv) > 1:
    command = int(sys.argv[1])


def load__single_file(file):
    df = pd.read_csv(file)
    y_actual = df['actual'].values.copy()
    df = drop_feilds_1df(df, ['actual'])
    best_forecast_index = 0
    return df, y_actual, best_forecast_index


def merge_a_df(base_df, model_name, file_name, command, feilds_to_use, merge_feilds):
    new_df = load_file(model_name, command, file_name, fields=merge_feilds + feilds_to_use)
    print "merging ", model_name, " feilds=", list(new_df)
    rmap = {f: model_name+"."+f for f in feilds_to_use}
    new_df.rename(columns=rmap, inplace=True)
    full_df = pd.merge(base_df, new_df, how='left', on=merge_feilds)
    return full_df


merge_feilds = ['Semana', 'Agencia_ID' , 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID']
forecast_feilds = ['XGB', 'LR', 'RFR', 'ETR']


def load_data_for_each_run(model_name, file_name, feild_names=None):

    files_list = [f for f in listdir(model_name) if str(f).startswith(file_name) and
                    str(f).endswith(".csv")]
    print file_name, "found files", files_list

    df_list = []
    for f in files_list:
        '''
        cmd = extract_regx('[0-9]+', f)
        if file_name == "model_forecasts":
            addtional_data_feild = 'actual'
        elif file_name == "model_submissions":
            addtional_data_feild = 'id'
        else:
            raise ValueError("")
        '''
        if feild_names is not None:
            base_df = pd.read_csv(model_name+'/'+str(f), usecols=feild_names)
        else:
            base_df = pd.read_csv(model_name+'/'+str(f))
        df_list.append(base_df)

    if len(df_list) > 0:
        final_df = pd.concat(df_list)

        #final_df = final_df.sample(frac=1)

         #if we do not copy, we get errors due to shafflinf
        #find_NA_rows_percent(final_df, "loading data for ensamble")
        final_df.fillna(0, inplace=True)


        #this is criticial as otherwise few stores will depend all in testing data
        #print "final feilds", list(final_df)
        #addtional_data = final_df[addtional_data_feild].values.copy()
        #final_df = drop_feilds_1df(final_df, [addtional_data_feild])

        print "load files", model_name, "shape=", final_df.shape, "with features", list(final_df)
        return final_df
    else:
        return None


def load_all_forecast_data(model_names_list, file_name):
    if file_name == "model_forecasts":
        fname = "test"
        addtional_data_feild = 'actual'
    elif file_name == "model_submissions":
        fname = "sub"
        addtional_data_feild = 'id'

    #basedf = load_data_for_each_run('fg_stats', fname, feild_names=merge_feilds + ["mean_sales", "sales_count", "sales_stddev",
    #                "median_sales", "last_sale", "last_sale_week", "returns", "signature", "kurtosis", "hmean", "entropy"])
    #basedf = load_data_for_each_run('fg_stats', fname, feild_names=merge_feilds + ["sales_count", "sales_stddev",
    #                "returns", "signature", "kurtosis", "entropy"])
    basedf = load_data_for_each_run('fg_stats', fname, feild_names=merge_feilds + ["signature", "kurtosis", "hmean", "entropy", "sales_count"])

    first_actual_feild = None
    data_size = 0

    feilds_to_remove = []
    forecast_data_feilds = []

    for model_name in model_names_list:
        forecast_df = load_data_for_each_run(model_name, file_name, merge_feilds + forecast_feilds + [addtional_data_feild])
        #rename so feilds will not clash
        rmap = {f: model_name+"."+f for f in forecast_feilds + [addtional_data_feild]}
        forecast_df.rename(columns=rmap, inplace=True)

        basedf = pd.merge(basedf, forecast_df, how='left', on=merge_feilds)
        basedf.fillna(0, inplace=True)

        y_feild =  model_name +"."+ addtional_data_feild

        if addtional_data_feild == 'actual':
            for f in forecast_feilds:
                ff = model_name +"."+ f
                forecast_data_feilds.append(ff)
                calculate_accuracy(ff + " loading", basedf[y_feild].values, basedf[ff].values)

        feilds_to_remove.append(y_feild)
        if first_actual_feild is None:
            first_actual_feild = y_feild
            data_size = forecast_df.shape[0]
        else:
            if forecast_df.shape[0] != data_size:
                raise ValueError("data sizes does not match for " + model_name + " " + str(forecast_df.shape[0]) + " != " + str(data_size))
            if not np.allclose(basedf[y_feild].values, basedf[first_actual_feild].values):
                raise ValueError("additiona feild not aligned " + str(basedf[y_feild].values[:10]), basedf[first_actual_feild].values[:10])

    #important, if not shufflued some products will end up in the test data fully
    basedf = basedf.sample(frac=1)
    #basedf = basedf.sample(frac=0.25)

    additional_feild_data = basedf[first_actual_feild].values
    basedf = drop_feilds_1df(basedf, feilds_to_remove + merge_feilds)

    return basedf, additional_feild_data, forecast_data_feilds


def load__model_results_from_store(model_names_list, name, use_agr_features=True):
    merge_feilds = ['Semana', 'Agencia_ID' , 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID']
    forecast_feilds = ['XGB', 'LR', 'RFR', 'ETR']
    #forecast_feilds = ['XGB', 'RFR', 'ETR']

    first_dir = model_names_list[0]
    files_list = [f for f in listdir(first_dir) if str(f).startswith(name) and
                    str(f).endswith(".csv")]
    print name, "found files", files_list

    df_list = []
    for f in files_list:
        cmd = extract_regx('[0-9]+', f)
        if name == "model_forecasts":
            fname = "test"
            addtional_data_feild = 'actual'
        elif name == "model_submissions":
            fname = "sub"
            addtional_data_feild = 'id'
        else:
            raise ValueError("")

        base_df = pd.read_csv(first_dir+'/'+str(f), usecols=merge_feilds + forecast_feilds + [addtional_data_feild])
        for i in range(1,len(model_names_list)):
            model_name = model_names_list[i]
            feilds_to_use = forecast_feilds
            file_name = name
            if model_name == "fg_stats":
                feilds_to_use = ["mean_sales", "sales_count", "sales_stddev",
                    "median_sales", "last_sale", "last_sale_week", "returns", "signature", "kurtosis", "hmean", "entropy"]
                file_name = fname
            base_df = merge_a_df(base_df, model_name, file_name, cmd, feilds_to_use, merge_feilds)

        base_df = drop_feilds_1df(base_df, merge_feilds)
        df_list.append(base_df)

    if len(df_list) > 0:
        '''
        #to make sure all feilds are in order
        for df in df_list:
            print "concat", list(df)
        fileds = list(df_list[0])
        df_list = [dft[fileds] for dft in df_list]
        for df in df_list:
            print "concat after", list(df)
        '''

        final_df = pd.concat(df_list)

        #this is criticial as otherwise few stores will depend all in testing data
        final_df = final_df.sample(frac=1)

         #if we do not copy, we get errors due to shafflinf
        #find_NA_rows_percent(final_df, "loading data for ensamble")
        final_df.fillna(0, inplace=True)

        print "final feilds", list(final_df)
        addtional_data = final_df[addtional_data_feild].values.copy()
        final_df = drop_feilds_1df(final_df, [addtional_data_feild])

        print "load files", model_names_list, name, "shape=", final_df.shape, "with features", list(final_df)
        return final_df, addtional_data
    else:
        return None

'''
def load__from_store(model_type, name, use_agr_features=True):
    dir = model_type
    files_list = [f for f in listdir(dir) if str(f).startswith(name) and
                    str(f).endswith(".csv")]
    print name, "found files", files_list

    df_list = []
    for f in files_list:
        base_df = pd.read_csv(dir+'/'+str(f))
        cmd = extract_regx('[0-9]+', f)
        if name == "model_forecasts":
            fname = "test"
            addtional_data_feild = 'actual'
        elif name == "model_submissions":
            fname = "sub"
            addtional_data_feild = 'id'

        merge_feilds = ['Semana', 'Agencia_ID' , 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID']
        if use_agr_features:
            #then features are in order, we just concatinate
            blend_df = load_file('agr_cat',cmd, fname)
            feilds_to_use = ["Agencia_ID_Demanda_uni_equil_Mean", "Agencia_ID_Demanda_uni_equilci" , "Agencia_ID_Demanda_uni_equil_median", "clients_combined_Mean",
                             "clients_combinedci", "clients_combined_median", "Producto_ID_Demanda_uni_equil_Mean" , "Producto_ID_Demanda_uni_equilci" ,
                             "Producto_ID_Demanda_uni_equil_median", "Producto_ID_Venta_hoy_Mean" ,"Producto_ID_Venta_hoyci" , "Producto_ID_Dev_proxima_Mean", "Producto_ID_Dev_proximaci" ,
                             "Producto_ID_Dev_proxima_median"]
            blend_df = blend_df[feilds_to_use]

            print list(base_df)
            base_df = drop_feilds_1df(base_df, merge_feilds)
            if base_df.shape[0] != blend_df.shape[0]:
                raise ValueError("two data frame sizes does not match "+ str(base_df.shape) + " " + str(blend_df))
            feature_df = pd.concat([base_df, blend_df], axis=1)
        else:
            blend_df = load_file('fg_stats',cmd, fname)
            feilds_to_use = ["mean_sales", "sales_count", "sales_stddev",
                    "median_sales", "last_sale", "last_sale_week", "returns", "signature", "kurtosis", "hmean", "entropy"]
            feilds_to_use = feilds_to_use + merge_feilds
            blend_df = blend_df[feilds_to_use]
            feature_df = pd.merge(base_df, blend_df, how='left', on=merge_feilds)
            feature_df = drop_feilds_1df(feature_df, merge_feilds)

        df_list.append(feature_df)

    if len(df_list) > 0:
        final_df = pd.concat(df_list)

        #this is criticial as otherwise few stores will depend all in testing data
        final_df = final_df.sample(frac=1)

         #if we do not copy, we get errors due to shafflinf
        find_NA_rows_percent(final_df, "loading data for ensamble")
        final_df.fillna(0, inplace=True)

        addtional_data = final_df[addtional_data_feild].values.copy()
        final_df = drop_feilds_1df(final_df, [addtional_data_feild])

        print "load file", model_type, name, "shape=", final_df.shape, "with features", list(feature_df)
        return final_df, addtional_data
    else:
        return None
'''


def do_ensamble(conf, forecasts_only_df, top_forecast_feilds, y_actual, submissions_ids, submissions_only_df):
    top_forecast_feilds = top_forecast_feilds[:10]
    sorted_forecasts = forecasts_only_df[top_forecast_feilds].values
    sorted_submissions = submissions_only_df[top_forecast_feilds].values

    best_forecast_index = [0]
    vf_start = time.time()

    mean_ensmbale = SimpleAvgEnsamble(conf, "mean")
    mean_ensmbale_forecast = mean_ensmbale.fit(sorted_forecasts, best_forecast_index, y_actual)
    mean_top4_submission = mean_ensmbale.predict(sorted_submissions, best_forecast_index)
    save_submission_file("mean_top4_submission.csv", submissions_ids, mean_top4_submission)


    best_pair_ensmbale = BestPairEnsamble(conf)
    best_pair_ensmbale_forecast = best_pair_ensmbale.fit(sorted_forecasts, best_forecast_index, y_actual)
    best_pair_ensamble_submission = best_pair_ensmbale.predict(sorted_submissions, best_forecast_index)
    save_submission_file("best_pair_submission.csv", submissions_ids, best_pair_ensamble_submission)


    #median_ensmbale = SimpleAvgEnsamble(conf, "median")
    #best_triple_ensmbale = BestThreeEnsamble(conf)

    #vote_forecast = vote_based_forecast(forecasts, best_forecast_index, y_actual)
    #calculate_accuracy("vote_forecast "+ str(conf.command), y_actual, vote_forecast)

    #vote_with_lr(conf, forecasts, best_forecast_index, y_actual)

    print "vf tooks", (time.time() - vf_start)
    return mean_ensmbale_forecast, mean_top4_submission, best_pair_ensmbale_forecast, best_pair_ensamble_submission


def find_best_forecast(forecasts_df, y_actual):
    start = time.time()
    forecasts_rmsle = []
    forecasts_fields = []
    for f in list(forecasts_df):
        rmsle = calculate_accuracy(f + " find best", y_actual, forecasts_df[f].values)
        forecasts_rmsle.append(rmsle)
        forecasts_fields.append(f)

    model_index_by_acc = np.argsort(forecasts_rmsle)
    top_forecast_feilds = [forecasts_fields[i] for i in model_index_by_acc]

    print "[IDF]full data rmsle=", forecasts_rmsle, forecasts_fields
    print "[IDF] sorted forecasts=", top_forecast_feilds

    best_findex = model_index_by_acc[0]
    print "[IDF]best single model forecast is", forecasts_fields[best_findex], "rmsle=", forecasts_rmsle[best_findex]
    print_time_took(start, "find_best_forecast")

    #print "best single model forecast stats\n", basic_stats_as_str(forecasts[best_findex])
    #print "y actual\n", basic_stats_as_str(y_actual)
    return top_forecast_feilds


'''
def run_ensambles(rcommand):
    conf = IDConfigs(target_as_log=True, normalize=True, save_predictions_with_data=True, generate_submission=True)
    conf.command=-2

    #load forecast data
    forecasts_df, y_actual = load__from_store('agr_cat', "model_forecasts")

    #not using LR forecasts
    print list(forecasts_df)

    forecasts = forecasts_df[['XGB', 'RFR','ETR']].values


    #load submission data
    subdf, submissions_ids = load__from_store('agr_cat', "model_submissions")
    submissions = subdf[['XGB', 'RFR','ETR']].values

    model_index_by_acc = find_best_forecast(forecasts, y_actual)

    best_forecast_index = model_index_by_acc[0]
    print_mem_usage("before simple ensamble")
    do_ensamble(conf, forecasts, best_forecast_index, y_actual, submissions_ids ,submissions)
    #blend_models(conf, forecasts, model_index_by_acc, y_actual, submissions_ids, submissions,
    #             blend_data, blend_data_submission)
    print_mem_usage("avg models")
    avg_models(conf, forecasts_df, y_actual, subdf, submission_ids=None)
'''


def run_ensambles_on_multiple_models(command):
    conf = IDConfigs(target_as_log=True, normalize=True, save_predictions_with_data=True, generate_submission=True)
    conf.command=-2

    '''
    #model_list = ['agr_cat', 'fg-vhmean-product', 'fg_stats', 'nn_features-product', 'nn_features-agency', "nn_features-brand"]
    model_list = ['agr_cat', 'fg-vhmean-product', 'fg_stats']


    #load forecast data
    forecasts_df, y_actual = load__model_results_from_store(model_list, "model_forecasts")

    #not using LR forecasts
    print "Using feilds", list(forecasts_df)

    forecasts = forecasts_df.values


    #load submission data
    subdf, submissions_ids = load__model_results_from_store(model_list, "model_submissions")
    submissions = subdf.values
    '''

    #model_list = ['agr_cat', 'fg-vhmean-product']
    model_list = ['nn_features-product', 'nn_features-agency', "nn_features-brand", "features-agc-pp", "agr_cat", "features-agency", "cc-cnn-agc"]
    #features-agency

    forecasts_with_blend_df, y_actual, forecast_feilds = load_all_forecast_data(model_list, "model_forecasts")
    forecasts_only_df = forecasts_with_blend_df[forecast_feilds]

    #load submission data
    sub_with_blend_df, submissions_ids, _ = load_all_forecast_data(model_list, "model_submissions")
    submissions_only_df = sub_with_blend_df[forecast_feilds]

    print "Using forecasting feilds", forecast_feilds
    top_forecast_feilds = find_best_forecast(forecasts_only_df, y_actual)

    #do the second level forecast
    #models = get_models4xgboost_tunning(conf, case=3)
    #for m in models:
    #    avg_models(conf, forecasts_with_blend_df, y_actual, sub_with_blend_df, submission_ids=submissions_ids,
    #               xgb_params=m.xgb_params, do_cv=False)

    print_mem_usage("before running models")
    print "new xgb configs 15"
    xgb_params = {'alpha': 0, 'booster': 'gbtree', 'colsample_bytree': 0.8, 'nthread': 4, 'min_child_weight': 10,
                  'subsample': 1.0, 'eta': 0.1, 'objective': 'reg:linear', 'max_depth': 15, 'gamma': 0.3, 'lambda': 0}
    avg_models(conf, forecasts_with_blend_df, y_actual, sub_with_blend_df, submission_ids=submissions_ids, xgb_params=xgb_params, frac=0.5)

    print "default xgb configs"
    avg_models(conf, forecasts_with_blend_df, y_actual, sub_with_blend_df, submission_ids=submissions_ids, frac=0.5)

    print "new xgb configs 10"
    xgb_params['max_depth'] = 10
    avg_models(conf, forecasts_with_blend_df, y_actual, sub_with_blend_df, submission_ids=submissions_ids, xgb_params=xgb_params, frac=0.5)


    print_mem_usage("after models")
    #mean_ensmbale_forecast, mean_top4_submission, best_pair_ensmbale_forecast, best_pair_ensamble_submission = \
    #    do_ensamble(conf, forecasts_only_df, top_forecast_feilds, y_actual, submissions_ids ,submissions_only_df)

    #to_saveDf =  pd.DataFrame(to_save, columns=["id","Demanda_uni_equil"])

    '''
    new_column_list = list(forecasts_with_blend_df) + ['mean_forecast', 'best_pair_forecast']
    forecasts_with_blend_df = pd.DataFrame(np.column_stack([forecasts_with_blend_df.values, mean_ensmbale_forecast, best_pair_ensmbale_forecast]),
                                                           columns=new_column_list)
    sub_with_blend_df = pd.DataFrame(np.column_stack([sub_with_blend_df.values, mean_top4_submission, best_pair_ensamble_submission]),
                                                           columns=new_column_list)

    avg_models(conf, forecasts_with_blend_df, y_actual, sub_with_blend_df, submission_ids=submissions_ids)
    '''

    #blend_models(conf, forecasts, model_index_by_acc, y_actual, submissions_ids, submissions,
    #             blend_data, blend_data_submission)
    #print_mem_usage("avg models")

    #avg_models_with_ml(conf, forecasts_df, y_actual, subdf, submission_ids=submissions_ids)

run_ensambles_on_multiple_models(command)