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

import sys

from os import listdir

print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

command = -2
if len(sys.argv) > 1:
    command = int(sys.argv[1])

np.set_printoptions(precision=1, suppress=True)
pd.options(scipen = 50)

dir = 'submissions'
files_list = [f for f in listdir(dir) if str(f).endswith(".csv")]
print "Found files", files_list

for f in files_list:
    df = pd.read_csv(dir+'/'+str(f))
    print str(f)
    print df.describe()
