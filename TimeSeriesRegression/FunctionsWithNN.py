from mltools import  regression_with_dl, print_regression_model_summary, MLConfigs, train_test_split, shuffle_data
from keras.optimizers import Adam, SGD
from sklearn import preprocessing

import numpy as np


def normlaize_data(X_all, Y_all):
    return preprocessing.normalize(X_all.astype("float32"), norm='l2'), preprocessing.normalize(Y_all.astype("float32"), norm='l2')[0]



#for c in configs:
#    y_pred_dl = regression_with_dl(X_train, y_train, X_test, y_test, c)
#    print_regression_model_summary("DL" + str(c.tostr()), y_test, y_pred_dl)

size = 10000
x = np.random.zipf(2, size)
y = np.random.normal(10, 1, size)
xy = [ x[i]*y[i] for i in range(size)]
xbyy = [ x[i]/y[i] if y[i] != 0 else 1 for i in range(size)]



#target = [ 2*(2*x[i] + y[i])/3*y[i] for i in range(size)]
target = [ 2*(2*x[i] + y[i])/3*y[i] for i in range(size)]

train_set_size = int(size*0.7)

X_all, Y_all = normlaize_data(np.column_stack((x,y, xy, xbyy)), np.array(target))
X_all, Y_all = shuffle_data(X_all,Y_all)


X_train, X_test, y_train, y_test = train_test_split(train_set_size, X_all, Y_all)

c = MLConfigs(nodes_in_layer=500, number_of_hidden_layers=5, dropout=0, activation_fn='relu', loss="mse",
              epoch_count=15, optimizer=Adam())
y_pred_dl = regression_with_dl(X_train, y_train, X_test, y_test, c)
print_regression_model_summary("DL" + str(c.tostr()), y_test, y_pred_dl)

