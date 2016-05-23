#import theano
#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'


import numpy as np

import pandas as pd
from sklearn import preprocessing

from tsforecasttools import run_timeseries_froecasts

from mltools import rolling_univariate_window, build_rolling_window_dataset, verify_window_dataset
from mltools import train_test_split,print_graph_test,almost_correct_based_accuracy
from mltools import regression_with_dl, print_regression_model_summary, report_scores


df = pd.read_csv("data/rossmann300stores.csv")

#We take 10 from first store
#df = df[(df['Store'] == 1)].head(100)
df = df[(df['Store'] == 1)]

print("data frame Shape", df.shape)

#takes numpy array from data frame
sales_data = df['Sales'].values
row_count = len(df.index);

#this return an 2D array, make it one
sales_data = preprocessing.normalize(sales_data.astype("float32"), norm='l2')[0]

print("sales_data.shape", sales_data.shape)

window_size = 4
training_set_size = int(0.7*row_count)

#build rolling window. It is of size row_count - window_size ( due to first window_size -1 due to lack of data, and last value
#due to lack of training (Y) data
X_all, Y_all = build_rolling_window_dataset(sales_data, window_size)

verify_window_dataset(X_all, Y_all)



#print(X_all)
#print(Y_all)
#print(df)

lb2IntEncoder = preprocessing.LabelEncoder()

df.StateHoliday  = lb2IntEncoder.fit_transform(df.StateHoliday)

df = df.drop('Date',1)
df = df.drop('Customers',1)
#df = df.drop('SchoolHoliday',1)
#df = df.drop('StateHoliday',1)
df = df.drop('Sales',1)

#remove first window_size-1 and last from data frame
index2remove = [row_count-1];
for i in xrange(1,window_size):
    index2remove.append(i)

print("index2remove", index2remove)
df = df.drop(df.index[index2remove])


print("Df", df.head())
#covert data frame to numpy array
X_without_sales = df.values.copy()
print("X_all.shape", X_all.shape, "X_without_sales.shape", X_without_sales.shape)
#join two arrays ( also can do np.row_stack )
X_all = np.column_stack((X_without_sales,X_all))

print("X_all.shape", X_all.shape)

#X_train_raw = preprocessing.normalize(X_train_raw.astype("float32"), norm='l2')

X_train, X_test, y_train, y_test = train_test_split(training_set_size, X_all, Y_all)

run_timeseries_froecasts(X_train, y_train, X_test, y_test, window_size, 10)


nodes_in_layer = 500
number_of_hidden_layers = 2
droput = 0.05
activation_fn='relu'
#y_pred_dl = regression_with_dl(X_train, y_train, X_test, y_test, nodes_in_layer,
#                      number_of_hidden_layers, droput, activation_fn, 100)
#print_regression_model_summary("DL", y_test, y_pred_dl)




#print_graph_test(y_test, y_pred_dl, y_pred_lr, 500)


#all_results = np.column_stack((all_results,y_pred))
#print(all_results)

#all_results = np.column_stack((y_test,y_pred))

#Alomost correct predictions - Perhaps the easiest metric to interpret, especially for less technical folks, is the percent of estimates that differ from the true value by no more than X%.
#Evaluation - http://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error