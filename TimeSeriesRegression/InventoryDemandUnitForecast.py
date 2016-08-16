
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
#from mlpreprocessing import feather2df


from inventory_demand_ensambles import *
from mlpreprocessing import *

import sys
print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

def generate_unit_features(conf, train_df, test_df, subdf):
    start = time.time()
    use_slope = False
    use_group_aggrigate = True
    use_product_features = True
    use_agency_features = False
    use_sales_data = False

    testDf = subdf

    if use_slope:
        train_df, test_df, testDf = addSlopes(train_df, test_df, testDf)

    #Venta_uni_hoy
    #Dev_uni_proxima


    default_demand_stats = DefaultStats(mean=train_df['Demanda_uni_equil'].mean(), count=train_df['Demanda_uni_equil'].count(),
                                        stddev=train_df['Demanda_uni_equil'].std())
    default_venta_hoy_stats = DefaultStats(train_df['Venta_hoy'].mean(), train_df['Venta_hoy'].count(),
                                        train_df['Venta_hoy'].std())
    default_dev_proxima_stats = DefaultStats(train_df['Dev_proxima'].mean(), train_df['Dev_proxima'].count(),
                                        train_df['Dev_proxima'].std())

    #this is to drop all features in one go
    feilds_to_drop = []

    default_demand_stats = DefaultStats(mean=train_df['sales_units'].mean(), count=train_df['sales_units'].count(),
                                        stddev=train_df['sales_units'].std())
    default_unit_price_stats = DefaultStats(train_df['unit_price'].mean(), train_df['unit_price'].count(),
                                        train_df['unit_price'].std())

    train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Canal_ID','Cliente_ID','Producto_ID', 'Agencia_ID', 'Ruta_SAK'],
                'unit_price', "unit_price", default_unit_price_stats,
                FeatureOps())

    if use_group_aggrigate:
        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Agencia_ID', testDf, default_demand_stats,
                                                            FeatureOps(hmean=True, stddev=True, count=True), agr_feild='sales_units')
        train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Ruta_SAK', 'Cliente_ID'],
                'sales_units', "clients_combined", default_demand_stats,
                FeatureOps(sum= True, kurtosis=True, stddev=True, count=True, p90=10, p10=True, hmean=True))
        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, default_demand_stats,
                FeatureOps(stddev=True, p90=True, hmean=True,p10=True, count=True), agr_feild='sales_units')
        train_df, test_df, testDf = join_multiple_feild_stats(train_df, test_df, testDf, ['Ruta_SAK', 'Cliente_ID'],
            'Dev_uni_proxima', "clients_combined_vh", default_venta_hoy_stats, FeatureOps(sum=True, hmean=True, p90=True, stddev=True))
        train_df, test_df, testDf = addFeildStatsAsFeatures(train_df, test_df,'Producto_ID', testDf, drop=False,
                                                            agr_feild='Dev_uni_proxima', default_stats=default_dev_proxima_stats,
                                                            fops=FeatureOps(count=True))



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

    #we use these features for second stage
    f2_features = ['unit_price_Mean', 'unit_priceci', 'unit_price_median']
    f2_train_df = train_df[f2_features]; f2_test_df = test_df[f2_features]; f2_sub_df= testDf[f2_features]

    train_data_feilds_to_drop = ['Venta_uni_hoy', 'Venta_hoy', 'Dev_uni_proxima', 'Dev_proxima', 'Demanda_uni_equil', 'sales_units', 'unit_price'] + f2_features
    feilds_to_drop = feilds_to_drop + ['Canal_ID','Cliente_ID','Producto_ID', 'Agencia_ID', 'Ruta_SAK']
    train_df, test_df, _ = drop_feilds(train_df, test_df, None, feilds_to_drop + train_data_feilds_to_drop)
    testDf = drop_feilds_1df(testDf, feilds_to_drop)

    #TODO explore this more http://pandas.pydata.org/pandas-docs/stable/missing_data.html
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    print "generate_features took ", (time.time() - start), "s"
    return train_df, test_df, testDf, f2_train_df, f2_test_df, f2_sub_df



command = -2
if len(sys.argv) > 1:
    command = int(sys.argv[1])
if len(sys.argv) > 2:
    test_run = (int(sys.argv[2]) == 1)
else:
    test_run = False

analysis_type = 'agr_unit'

use_preprocessed_file = False
save_preprocessed_file = False
target_as_log = True
save_predictions_with_data = False
verify_sub_data = False

s_time = time.time()

np.set_printoptions(precision=1, suppress=True)

df, sub_dfo = read_datafiles(command, test_run)

r_time = time.time()
print "read took %f" %(r_time-s_time)

conf = IDConfigs(target_as_log=True, normalize=True, save_predictions_with_data=True, generate_submission=True)
conf.command = command
conf.analysis_type = analysis_type


#df = remove_rare_categories(df)

df = df[df['Producto_ID'] > 0]
df = df.sample(frac=1)

#unit value
sales_unit_np = df['Venta_uni_hoy'] - df['Dev_uni_proxima']
df['sales_units'] = np.where(sales_unit_np <0, 0, sales_unit_np)


sales_unit_price = df['Venta_hoy']/np.where(df['Venta_uni_hoy'] == 0, 1, df['Venta_uni_hoy'])
df['unit_price'] = sales_unit_price


find_NA_rows_percent(df, "data set stats")

'''
training_set_size = int(0.6*df.shape[0])
test_set_size = df.shape[0] - training_set_size

train_df = df[:training_set_size]
test_df = df[-1*test_set_size:]

print_mem_usage("before features")
train_df, test_df, testDf = generate_unit_features(conf, train_df, test_df, testDf)

y_all = df['sales_units'].values
y_actual_train = y_all[:training_set_size]
y_actual_test = y_all[-1*test_set_size:]
'''
sub_dfo.fillna(0, inplace=True)
ids = sub_dfo['id']
sub_dfo.drop('id',axis=1, inplace=True)


fold_list, y_all = prepare_train_and_test_data_with_folds(df, 'sales_units', fold_count=2)
fold_forecasts = []
submissions_forecasts = []

for fold_id, (train_dfo, test_dfo, y_actual_train,y_actual_test) in enumerate(fold_list):
    train_df = train_dfo; test_df =test_dfo; testDf = sub_dfo.copy()



    print_mem_usage(str(fold_id) + ": before features")
    train_df, test_df, testDf, f2_train_df, f2_test_df, f2_sub_df = generate_unit_features(conf, train_df, test_df, testDf)

    prep_time = time.time()
    print_mem_usage(str(fold_id) + ": before forecast")
    if conf.target_as_log:
        train_df, test_df, testDf = tranform_train_data_to_log(train_df, test_df, testDf, skip_field_patterns=['kurtosis', 'id'])
        y_train_raw, y_test_raw = transfrom_to_log(y_actual_train), transfrom_to_log(y_actual_test)
    else:
        y_train_raw, y_test_raw = y_actual_train, y_actual_test

    models, forecasts, test_df, parmsFromNormalization, parmsFromNormalization2D = do_forecast(conf, train_df, test_df,
                                                                                           y_train_raw, y_test_raw, y_actual_test)

    print_mem_usage(str(fold_id) + ":after forecast")

    best_model_index = np.argmin([m.rmsle for m in models])
    best_model = models[best_model_index]
    print "[IDF"+str(conf.command)+" fold="+ str(fold_id) +"]Best Single Model has rmsle=", best_model.rmsle

    print best_model_index,
    best_forecast = forecasts[:,best_model_index]
    fold_forecasts.append(best_forecast)

    '''
    #lets train second model
    f2_train_df['units_sold'] = y_actual_train; f2_y_train = train_dfo['Demanda_uni_equil']
    f2_test_df['units_sold'] = y_actual_test; f2_y_test = test_dfo['Demanda_uni_equil']

    if conf.target_as_log:
        f2_train_df, f2_test_df, _ = tranform_train_data_to_log(train_df, test_df, testDf, skip_field_patterns=['kurtosis', 'id'])
        f2_y_train, f2_y_test = transfrom_to_log(f2_y_train), transfrom_to_log(f2_y_train)


    f2_m = get_models4xgboost_only()
    f2models, f2forecasts, _, f2parmsFromNormalization, f2parmsFromNormalization2D = do_forecast(conf, f2_train_df, f2_test_df,
                                                                                           f2_y_train, f2_y_test, test_dfo['Demanda_uni_equil'], models=f2_m)
    f2models[0].predict()
    '''

    sub_X_all = testDf.values
    y_forecast_submission = create_submission(conf, best_model, ids, sub_X_all, parmsFromNormalization, parmsFromNormalization2D)
    submissions_forecasts.append(y_forecast_submission)

fold_forecasts.reverse()
final_forecast = np.column_stack(fold_forecasts).flatten()
print "results shapes, actual=", y_all.shape, "forecast=", final_forecast.shape
calculate_accuracy("kfold unit forecast", y_all, final_forecast)

submissions_forecasts_np = np.column_stack(submissions_forecasts)
final_submission = np.mean(submissions_forecasts_np, axis=1)
print "submission shapes=", final_submission.shape





'''
#save submission based on best model
if conf.generate_submission:
    y_forecast_submission = create_submission(conf, best_model, testDf, parmsFromNormalization, parmsFromNormalization2D)

#create and save predictions for each model so we can build an ensamble later
if forecasts.shape[1] > 1:
    blend_features = get_blend_features()
    blend_data_test = test_df[blend_features].values


    model_names = [m.name for m in models]
        #first we will save all individual model results for reuse
    model_rmsle = [m.rmsle for m in models]
    model_forecasts_data = np.column_stack([blend_data_test, forecasts, y_actual_test])
    to_saveDf =  pd.DataFrame(model_forecasts_data, columns=blend_features + model_names + ["actual"])
    metadata_map = {'rmsle':model_rmsle}
    save_file(analysis_type, command, to_saveDf, 'model_forecasts', metadata=metadata_map)
    print "## model_forecasts ##"
    print to_saveDf.describe()

    blend_data_submission = testDf[blend_features].values
    ids, kaggale_predicted_list = create_per_model_submission(conf, models, testDf, parmsFromNormalization, parmsFromNormalization2D )
    submission_data = np.column_stack([ids, blend_data_submission, kaggale_predicted_list])
    to_saveDf =  pd.DataFrame(submission_data, columns=[["id"] + blend_features +model_names])
    save_file(analysis_type, command, to_saveDf, 'model_submissions')
    print "## model_submissions ##"
    print to_saveDf.describe()
'''

m_time = time.time()

print "total=%f, read=%fs, preporcess=%fs, model=%fs" \
      %(m_time-s_time, (r_time-s_time), (prep_time-r_time), (m_time-prep_time))


