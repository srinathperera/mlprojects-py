import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import time
from sklearn.externals import joblib

from mltools import undoPreprocessing


from tsforecasttools import run_timeseries_froecasts, regression_with_xgboost
from mltools import preprocess2DtoZeroMeanUnit, preprocess1DtoZeroMeanUnit, train_test_split, print_feature_importance, apply_zeroMeanUnit2D
from mltools import calculate_rmsle, almost_correct_based_accuracy, MLConfigs, print_regression_model_summary, \
    regression_with_dl, apply_zeroMeanUnit, undo_zeroMeanUnit2D
from keras.optimizers import Adam




class InventoryDemandPredictor:
    def __init__(self, model):
        self.model = model

    def preprocess_x(self, x_data):
        return None

    def preprocess_y(self, y_data):
        return None


    def add_features2train(self, x_train):
        return None

    def transfrom4prediction(self, data):
        return None


    #models should be able to correct normalizations
    def train(self, models, x_train, y_train, x_test, y_test):
        x_train_predictions = np.hstack([m.predict(x_train) for m in models])
        x_test_predictions = np.hstack([m.predict(x_test) for m in models])

        #return lr

    def predict(self, x_data):
        return self.model.predict(x_data)


def avgdiff(group):
    group = group.values
    if len(group) > 1:
        return np.mean([group[i] - group[i-1] for i in range(1, len(group))])
    else:
        return 0

