#import theano
#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'


import numpy as np

import pandas as pd
from sklearn import preprocessing

from tsforecasttools import run_timeseries_froecasts

from mltools import rolling_univariate_window, build_rolling_window_dataset, verify_window_dataset
from mltools import train_test_split,print_graph_test,almost_correct_based_accuracy, MLConfigs
from mltools import regression_with_dl, print_regression_model_summary, report_scores

from keras.optimizers import Adam, SGD

from sklearn.utils import shuffle

df = pd.read_csv("data/rossmann300stores.csv")

#We take 10 from first store
#df = df[(df['Store'] == 1)].head(100)

print("data frame Shape", df.shape)

lb2IntEncoder = preprocessing.LabelEncoder()
df.StateHoliday  = lb2IntEncoder.fit_transform(df.StateHoliday)

df = df.drop('Date',1)
df = df.drop('Customers',1)
#df = df.drop('SchoolHoliday',1)
#df = df.drop('StateHoliday',1)

sales_data = df['Sales'].values
df['Sales'] = preprocessing.normalize(sales_data.astype("float32"), norm='l2')[0]

window_size = 14
storeList = df['Store'].unique()

storeData = []
target = []

for s in storeList:
    dfS = df[(df['Store'] == s)]
    sales_data = dfS['Sales'].values
    X_all_t, Y_all_t = build_rolling_window_dataset(sales_data, window_size)
    #verify_window_dataset(X_all_t,Y_all_t)

    dfS = dfS.drop('Sales',1)

    #remove first window_size-1 and last from data frame
    row_count = len(dfS.index);
    index2remove = [row_count-1];
    for i in xrange(0,window_size-1):
        index2remove.append(i)

    dfS = dfS.drop(dfS.index[index2remove])

    #covert data frame to numpy array
    X_without_sales = dfS.values.copy()
    print("X_all.shape", X_all_t.shape, "X_without_sales.shape", X_without_sales.shape)
    #join two arrays ( also can do np.row_stack )
    X_all_t = np.column_stack((X_without_sales,X_all_t))

    print("X_all.shape", X_all_t.shape)
    storeData.append(X_all_t)
    target.append(Y_all_t)

X_all = storeData[0]
Y_all = target[0]
for i in range(1,len(storeData)):
    #print("DebugX", i, X_all.shape, sales_data[i].shape)
    X_all = np.row_stack((X_all, storeData[i]))
    #print("Debug", i,Y_all.shape, target[i].shape)
    Y_all = np.concatenate((Y_all, target[i]), 0)

if X_all.shape[0] != Y_all.shape[0]:
    raise  ValueError("X_all and Y_all does not match")


#takes numpy array from data frame
row_count = X_all.shape[0]
print("sales_data.shape", sales_data.shape)
training_set_size = int(0.7*row_count)

#X_all = shuffle(X_all)

X_train, X_test, y_train, y_test = train_test_split(training_set_size, X_all, Y_all)

run_timeseries_froecasts(X_train, y_train, X_test, y_test, window_size, 10)





configs = [
#    MLConfigs(nodes_in_layer = 500, number_of_hidden_layers = 2, droput = 0, activation_fn='relu', loss= "mse",
#        epoch_count = 10, optimizer = SGD(lr=0.1, decay=1e-3, momentum=0.99, nesterov=True)),
    MLConfigs(nodes_in_layer = 1000, number_of_hidden_layers = 2, droput = 3, activation_fn='relu', loss= "mse",
        epoch_count = 20, optimizer = Adam()),
    MLConfigs(nodes_in_layer = 1000, number_of_hidden_layers = 2, droput = 2, activation_fn='relu', loss= "mse",
        epoch_count = 20, optimizer = SGD(lr=0.01, decay=0, momentum=0.9, nesterov=True)),

    #MLConfigs(nodes_in_layer = 500, number_of_hidden_layers = 2, droput = 0, activation_fn='relu', loss= "mse",
    #    epoch_count = 10, optimizer = SGD(lr=0.1, decay=1e-3, momentum=0.99, nesterov=True)),
    #MLConfigs(nodes_in_layer = 500, number_of_hidden_layers = 2, droput = 0, activation_fn='relu', loss= "mse",
    #    epoch_count = 10, optimizer = SGD(lr=0.1, decay=1e-3, momentum=0.99, nesterov=True)),
    #MLConfigs(nodes_in_layer = 500, number_of_hidden_layers = 2, droput = 0, activation_fn='relu', loss= "mse",
    #    epoch_count = 10, optimizer = SGD(lr=0.1, decay=1e-3, momentum=0.99, nesterov=True)),
    #MLConfigs(nodes_in_layer = 500, number_of_hidden_layers = 2, droput = 0, activation_fn='relu', loss= "mse",
    #    epoch_count = 10, optimizer = SGD(lr=0.1, decay=1e-3, momentum=0.99, nesterov=True)),
    #MLConfigs(nodes_in_layer = 500, number_of_hidden_layers = 10, droput = 0, activation_fn='relu', loss= "mse",
    #    epoch_count = 10, optimizer = SGD(lr=0.1, decay=1e-3, momentum=0.99, nesterov=True))
    ]

#for c in configs:
#    y_pred_dl = regression_with_dl(X_train, y_train, X_test, y_test, c)
#    print_regression_model_summary("DL" + str(c.tostr()), y_test, y_pred_dl)

