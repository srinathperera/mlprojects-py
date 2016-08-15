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


analysis_type = 'fg_stats'





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

    median_forecast = test_df['median_sales']
    calculate_accuracy("median_forecast", y_actual_test, median_forecast)





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