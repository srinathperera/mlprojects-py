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
from InventoryDemandSimplePredictions import do_simple_models

import sys
print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

command = -2
if len(sys.argv) > 1:
    command = int(sys.argv[1])


def merge_submissions(path="./", file_name_prefix="submission"):
    submission_files = [pd.read_csv(path+file_name_prefix+str(i)+'.csv') for i in range(4)]

    tot = 0
    for df in submission_files:
        print "size",df.shape[0]
        tot = tot + df.shape[0]
    print "submission size", tot

    result = pd.concat(submission_files).sort_values(by=['id'])
    result.to_csv(file_name_prefix+ '_final.csv', index=False)
    print "submission file of shape ", result.shape, " created"


merge_submissions(file_name_prefix='submission')
merge_submissions(file_name_prefix='en_submission')