from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

from keras.regularizers import l2, activity_l2

import numpy as np

import pandas as pd
from sklearn import preprocessing

import matplotlib.pylab as plt
import seaborn as sns
sns.set_style("darkgrid")
sns.set_context("poster")


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from mltools import print_regression_model_summary,regression_with_dl

from tsforecasttools import run_timeseries_froecasts


from mltools import rolling_univariate_window, build_rolling_window_dataset
from mltools import train_test_split,print_graph_test,almost_correct_based_accuracy

df = pd.read_csv("data/rossmannW.csv")
#df = pd.read_csv("rossmannW200k.csv")

print("data frame Shape", df.shape)

#takes numpy array from data frame
#sales_data = df['Sales'].values
row_count = len(df.index);

#this return an 2D array, make it one
#sales_data = preprocessing.normalize(sales_data.astype("float32"), norm='l2')[0]

#print("sales_data.shape", sales_data.shape)

window_size = 4
training_set_size = 0.7*row_count

#build rolling window. It is of size row_count - window_size ( due to first window_size -1 due to lack of data, and last value
#due to lack of training (Y) data
#X_all, Y_all = build_rolling_window_dataset(sales_data, window_size)

#print(X_all)
#print(Y_all)
#print(df)

#lb2IntEncoder = preprocessing.LabelEncoder()

#df.StateHoliday  = lb2IntEncoder.fit_transform(df.StateHoliday)

print(df.head())
print(list(df.columns.values))
Y_all = df['Sales'].values
Y_all = preprocessing.normalize(Y_all.astype("float32"), norm='l2')[0]

#df = df.drop('Date',1)
#df = df.drop('Customers',1)
#df = df.drop('SchoolHoliday',1)
#df = df.drop('StateHoliday',1)
df = df.drop('Sales',1)
df = df.drop('Date', 1)
df = df.drop('Customers', 1)
df = df.drop('StateHoliday', 1)



#remove first window_size-1 and last from data frame
#index2remove = [row_count-1];
#for i in xrange(1,window_size):
#    index2remove.append(i)

#print("index2remove", index2remove)
#df = df.drop(df.index[index2remove])


print("Df", df.head())
print(df.dtypes)
#covert data frame to numpy array
X_all = df.values.copy()

X_all = preprocessing.normalize(X_all.astype("float32"), norm='l2')


#print("X_all.shape", X_all.shape, "X_without_sales.shape", X_without_sales.shape)

#join two arrays ( also can do np.row_stack )
#X_all = np.column_stack((X_without_sales,X_all))

#print(X_all)
print("X_all.shape", X_all.shape)


X_train, X_test, y_train, y_test = train_test_split(training_set_size, X_all, Y_all)

#run_timeseries_froecasts(X_train, y_train, X_test, y_test, window_size, epoch_count=10)


nodes_in_layer = 500
number_of_hidden_layers = 2
droput = 0.3
activation_fn='relu'
y_pred_dl = regression_with_dl(X_train, y_train, X_test, y_test, nodes_in_layer,
                      number_of_hidden_layers, droput, activation_fn, 10)
print_regression_model_summary("DL", y_test, y_pred_dl)






#print(Y_all)

#X_train_raw = preprocessing.normalize(X_train_raw.astype("float32"), norm='l2')

#input_feilds_count = 8 + window_size
#nodes_in_layer = 200
#number_of_hidden_layers = 20

#nodes_in_layer = 400
#number_of_hidden_layers = 40
#droput = 0.1
#activation_fn='relu'
#activation_fn='sigmoid'

#model = Sequential()
#model.add(Dense(nodes_in_layer, input_dim=input_feilds_count,activation=activation_fn))
#for i in xrange(0, number_of_hidden_layers):
#    model.add(Dense(nodes_in_layer, activation=activation_fn))
#    #model.add(Dense(nodes_in_layer, activation=activation_fn, W_regularizer=l2(0.001))) #http://keras.io/regularizers/
#    model.add(Dropout(droput)) # add dropout

#model.add(Dense(20, activation='relu'))
#model.add(Dense(1))
#model.compile(loss='mse', optimizer='adam') # loss 10-5
#model.compile(loss='mae', optimizer='adam') #loss 10-1
#model.compile(loss='mse', optimizer='sgd') #loss 10-2
#model.compile(loss='mape', optimizer='adam') # loss 10-5


#optimizer='sgd, RMSprop( usually for RNN), Adagrad, Adadelta, Adam' , see http://keras.io/optimizers/ ( momentum, learning rate etc is calcuated here)
#loss functions http://keras.io/objectives/
#activations http://keras.io/activations/



#hist = model.fit(X_train, y_train,
#          nb_epoch=2, batch_size=50, validation_data=(X_test, y_test))
#print(hist.history)


#score = model.evaluate(X_test, y_test, batch_size=16)

#loss = model.evaluate(X_test, y_test, batch_size=16, verbose=True)

#print("DL Loss", "{:.8f}".format(loss))

#y_pred = model.predict(X_test)

#print_regression_model_summary("LR", y_test, y_pred)

#print("window_size",window_size, "nodes_in_layer", nodes_in_layer, "number_of_hidden_layers", number_of_hidden_layers, "droput", droput)
#print "DL mean_squared_error: %8f" %(mean_squared_error(y_test, y_pred))
#dl_error_AC, p90 = almost_correct_based_accuracy(y_test, y_pred, 10)
#print("DL_AC_errorRate", dl_error_AC, "p90", p90)



#all_results = np.column_stack((y_test,y_pred))
#y_pred1 = y_pred;


#lr = LinearRegression(normalize=True)
#lr.fit(X_train, y_train)
#y_pred = lr.predict(X_test)
#print_regression_model_summary("LR", y_test, y_pred)



#print_graph_test(y_test, y_pred1, y_pred, 500)


#all_results = np.column_stack((all_results,y_pred))
#print(all_results)


#Alomost correct predictions - Perhaps the easiest metric to interpret, especially for less technical folks, is the percent of estimates that differ from the true value by no more than X%.
#Evaluation - http://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error