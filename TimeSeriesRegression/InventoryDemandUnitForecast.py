
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

import sys
print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

def generate_unit_features(conf, train_df, test_df, subdf, y_actual_test):
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
    feilds_to_drop = feilds_to_drop + ['Canal_ID','Cliente_ID','Producto_ID', 'Agencia_ID', 'Ruta_SAK']
    train_df, test_df, _ = drop_feilds(train_df, test_df, None, feilds_to_drop + train_data_feilds_to_drop)
    testDf = drop_feilds_1df(testDf, feilds_to_drop)

    #TODO explore this more http://pandas.pydata.org/pandas-docs/stable/missing_data.html
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)


    print "generate_features took ", (time.time() - start), "s"
    return train_df, test_df, testDf, y_actual_test, test_df_before_dropping_features



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

df, testDf = read_datafiles(command, test_run)

r_time = time.time()
print "read took %f" %(r_time-s_time)

conf = IDConfigs(target_as_log=True, normalize=True, save_predictions_with_data=True, generate_submission=True)
conf.command = command
conf.analysis_type = analysis_type


#df = remove_rare_categories(df)

df = df[df['Producto_ID'] > 0]

df = df.sample(frac=1)

find_NA_rows_percent(df, "data set stats")


training_set_size = int(0.6*df.shape[0])
test_set_size = df.shape[0] - training_set_size

y_all = df['Demanda_uni_equil'].values


train_df = df[:training_set_size]
test_df = df[-1*test_set_size:]

y_actual_train = y_all[:training_set_size]
y_actual_test = y_all[-1*test_set_size:]


print_mem_usage("before features")

train_df, test_df, testDf, y_actual_test, test_df_before_dropping_features = generate_features(conf, train_df,
                                                                                               test_df, testDf, y_actual_test)

print "after features", test_df['Semana'].unique(), test_df.shape
print "after features bd", test_df_before_dropping_features['Semana'].unique(), test_df_before_dropping_features.shape

prep_time = time.time()

if test_run:
    print train_df.describe()

print_mem_usage("before forecast")

if conf.target_as_log:
    train_df, test_df, testDf = tranform_train_data_to_log(train_df, test_df, testDf, skip_field_patterns=['kurtosis', 'id'])
    y_train_raw, y_test_raw = transfrom_to_log(y_actual_train), transfrom_to_log(y_actual_test)
else:
    y_train_raw, y_test_raw = y_actual_train, y_actual_test

models, forecasts, test_df, parmsFromNormalization, parmsFromNormalization2D = do_forecast(conf, train_df, test_df,
                                                                                           y_train_raw, y_test_raw, y_actual_test)

print_mem_usage("after forecast")

best_model_index = np.argmin([m.rmsle for m in models])
best_model = models[best_model_index]
print "[IDF"+str(conf.command)+"]Best Single Model has rmsle=", best_model.rmsle

print best_model_index,
best_forecast = forecasts[:,best_model_index]

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


m_time = time.time()

print "total=%f, read=%fs, preporcess=%fs, model=%fs" \
      %(m_time-s_time, (r_time-s_time), (prep_time-r_time), (m_time-prep_time))


