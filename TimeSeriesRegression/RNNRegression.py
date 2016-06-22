import numpy as np

import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from mltools import transform_dataset4rnn
from keras.regularizers import l2, activity_l2

#Resources
############
#https://github.com/Vict0rSch/deep_learning/tree/master/keras/recurrent
#http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
#http://danielhnyk.cz/predicting-sequences-vectors-keras-using-rnn-lstm/

from mltools import rolling_univariate_window, build_rolling_window_dataset
from mltools import train_test_split,print_graph_test,almost_correct_based_accuracy
from mltools import regression_with_dl, print_regression_model_summary, preprocess1DtoZeroMeanUnit

df = pd.read_csv("data/rossmann300stores.csv")

print("data frame Shape", df.shape)

#takes numpy array from data frame
sales_data = df['Sales'].values
row_count = len(df.index);


sales_data, parmsFromNormalization = preprocess1DtoZeroMeanUnit(sales_data)
df.Sales = sales_data
lb2IntEncoder = preprocessing.LabelEncoder()
df.StateHoliday  = lb2IntEncoder.fit_transform(df.StateHoliday)

df = df.drop('Date',1)
df = df.drop('Customers',1)
#df = df.drop('SchoolHoliday',1)
#df = df.drop('StateHoliday',1)
#df = df.drop('Sales',1)

#remove last row from df and remove first from sales data
#df = df.drop(df.index[[row_count-1]])
print df.head(10)
X_all = df.values.copy()

#print("X_all.shape", X_all.shape)
#X_train, X_test, y_train, y_test = train_test_split(row_count*.7, X_all, Y_all)
#fix this later. Rearrange data so it will work with function given

epochs = 20
ratio = 0.5
sequence_length = 50


stores = df['Store'].unique()

#only take first 50 stores
#stores = list(range(1,50,1))

X_train = None
y_train = None
X_test = None
y_test = None

X_train_list = []
y_train_list = []
X_test_list = []
y_test_list = []

isFirst = True
for s in stores:
    dataset = []
    X_all = df[(df['Store'] == s)].values.copy()
    for i in range(X_all.shape[1]):
        dataset.append(X_all[:, i])
    X_train_t, y_train_t, X_test_t, y_test_t = transform_dataset4rnn(dataset, sequence_length, 6)

    if isFirst:
        X_train = X_train_t
        y_train = y_train_t
        X_test = X_test_t
        y_test = y_test_t
        isFirst = False
    else:
        print "Stroeshapes ", len(X_train_t)
        X_train = np.vstack((X_train, X_train_t))
        y_train = np.append(y_train, y_train_t)
        X_test = np.vstack((X_test, X_test_t))
        y_test = np.append(y_test, y_test_t)
        #print "fpp"
    #print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape,
    #  "X_test.shape", X_test.shape, "Y_test.shape", y_test.shape)

#for s in stores:
#    dataset = []
#    X_all = df[(df['Store'] == s)].values.copy()
#    for i in range(X_all.shape[1]):
#        dataset.append(X_all[:, i])
#    X_train_t, y_train_t, X_test_t, y_test_t = transform_dataset4rnn(dataset, sequence_length, 6)
#    X_train_list.append(X_train_t)
#    y_train_list.append(y_train_t)
#    X_test_list.append(X_test_t)
#    y_test_list.append(y_test_t)

#X_train = np.vstack(X_train_list)
#print [x.shape for x in y_train_list ]
#y_train = np.vstack(y_train_list)
#X_test = np.vstack(X_test_list)
#y_test = np.vstack(y_test_list)

print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape,
      "X_test.shape", X_test.shape, "Y_test.shape", y_test.shape)

model = Sequential()
layers = [7, 10, 20, 1]
dropout = 0.1
rgw = 0.001

#

model.add(LSTM(
    input_dim=layers[0],
    output_dim=layers[1],
    return_sequences=True, W_regularizer=l2(rgw), U_regularizer=l2(rgw)))
model.add(Dropout(dropout))

model.add(LSTM(
    layers[2],
    return_sequences=False, W_regularizer=l2(rgw), U_regularizer=l2(rgw)))
model.add(Dropout(dropout))

model.add(Dense(
    output_dim=layers[3]))
model.add(Activation("linear"))

model.compile(loss="mse", optimizer="rmsprop")

print("compiltion done")

model.fit(X_train, y_train, batch_size=512, nb_epoch=epochs, validation_split=0.05)
Y_predicted = model.predict(X_test)
print("Y_predicted shape", Y_predicted.shape)

print_regression_model_summary("RNN", y_test, Y_predicted, parmsFromNormalization)



