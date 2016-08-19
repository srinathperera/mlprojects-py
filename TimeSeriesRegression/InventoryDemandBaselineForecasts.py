import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mltools import apply_zeroMeanUnit2D, preprocess2DtoZeroMeanUnit, undo_zeroMeanUnit2D
import time

import re
from matplotlib import cm as CM


from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

from inventory_demand import *
import sys
from data_explore import *

analysis_type = 'fg_stats1'





def run_simple_models(conf):
    train_df, test_df, testDf, y_actual_train, y_actual_test = load_train_data(analysis_type, conf.command)
    if train_df is None or test_df is None or testDf is None:
        print "Data not found", analysis_type
    else:
        print "reusing train data", analysis_type

    #drop the base feilds from forecasts
    feilds_to_drop =  ['Canal_ID','Cliente_ID','Producto_ID', 'Agencia_ID', 'Ruta_SAK']
    train_df, test_df, _ = drop_feilds(train_df, test_df, None, feilds_to_drop + ['Demanda_uni_equil'] )
    testDf = drop_feilds_1df(testDf, feilds_to_drop)

    mean_forecast = test_df['mean_sales']
    calculate_accuracy("mean_forecast", y_actual_test, mean_forecast)
    print basic_stats_as_str(mean_forecast)

    create_fig()
    error = np.abs(np.log(y_actual_test +1) - np.log(mean_forecast + 1))
    draw_simple_scatterplot(test_df['sales_count'], error, 'sales_count', 321)

    plt.tight_layout()
    plt.savefig('error_dist.png')

    error_df = test_df.copy()
    error_df['error'] = error
    error_df['sales_stddev'] = np.ceil(error_df['sales_stddev'])
    group1 = error_df.groupby(['sales_count'])['error']
    error_data = g2df_sum_mean(group1).sort_values(by=['sum'], ascending=False)
    print error_data

    group1 = error_df.groupby(['sales_stddev'])['error']
    error_data = g2df_sum_mean(group1).sort_values(by=['sum'], ascending=False)
    print error_data




    median_forecast = test_df['median_sales']
    calculate_accuracy("median_forecast", y_actual_test, median_forecast)

    last_sale_forecast = test_df['last_sale']
    calculate_accuracy("last_sale", y_actual_test, last_sale_forecast)
    print basic_stats_as_str(last_sale_forecast)







def test_simple_model(command):
    conf = IDConfigs(target_as_log=True, normalize=True, save_predictions_with_data=True, generate_submission=True)
    conf.command = command
    run_simple_models(conf)



print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

command = -2
if len(sys.argv) > 1:
    command = int(sys.argv[1])

test_simple_model(command)