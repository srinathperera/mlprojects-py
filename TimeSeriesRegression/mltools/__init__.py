import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
sns.set_style("darkgrid")
sns.set_context("poster")

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam, SGD
from math import sqrt

from keras.callbacks import EarlyStopping, Callback
from keras.objectives import mean_squared_error

from keras.regularizers import l2, activity_l2

from sklearn.metrics import mean_squared_error
from operator import itemgetter

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn import cross_validation


class MLConfigs:
    nodes_in_layer = 500
    number_of_hidden_layers = 5
    droput = 0
    activation_fn='relu'
    loss= "mse"
    epoch_count = 10
    optimizer = Adam()

    def __init__(self, nodes_in_layer = 500, number_of_hidden_layers = 5, droput = 0, activation_fn='relu', loss= "mse",
              epoch_count = 10, optimizer = Adam()):
        self.nodes_in_layer = nodes_in_layer
        self.number_of_hidden_layers = number_of_hidden_layers
        self.droput = droput
        self.activation_fn = activation_fn
        self.epoch_count = epoch_count
        self.optimizer = optimizer
        self.loss = loss

    def tostr(self):
        return "NN %dX%d droput=%d at=%s loss=%s" %(self.nodes_in_layer,self.number_of_hidden_layers, self.droput,
                                            self.activation_fn, self.loss),


class LogFile:
    def __init__(self):
        self.log = open('log.txt', 'a')

    def close(self):
        self.log.close()

    def write(self,m):
        self.log.write(m)

log = LogFile()


class LearningRateLogger(Callback):
    '''
    learning rate printer
    '''
    def on_epoch_end(self, epoch, logs={}):
        if hasattr(self.model.optimizer, 'decay'):
            lr = self.model.optimizer.lr.get_value()
            it = self.model.optimizer.iterations.get_value()
            decay = self.model.optimizer.decay.get_value()
            print(" lr=", lr * (1.0 / (1.0 + decay * it)))


def rolling_univariate_window(time_series, window_size):
    shape = (time_series.shape[0] - window_size + 1, window_size)
    strides = time_series.strides + (time_series.strides[-1],)
    return np.lib.stride_tricks.as_strided(time_series, shape=shape, strides=strides)



def build_rolling_window_dataset(time_series, window_size):
    last_element = time_series[-1]
    time_series = time_series[:-1]
    X_train = rolling_univariate_window(time_series, window_size)
    y_train = np.array([X_train[i, window_size-1] for i in range(1, X_train.shape[0])])

    return X_train, np.hstack((y_train, last_element))


def train_test_split(no_of_training_instances, X_all, y_all):
    X_train = X_all[0:no_of_training_instances, :]
    X_test = X_all[no_of_training_instances:, :]
    y_train = y_all[0:no_of_training_instances]
    y_test = y_all[no_of_training_instances:]

    return X_train, X_test, y_train, y_test

def print_graph(X_all, X_test, y_all, y_pred1, y_pred2):
    training_size = X_all.shape[0] - X_test.shape[0]
    x_full_limit = np.linspace(1, X_all.shape[0], X_all.shape[0])
    y_pred_limit = np.linspace(training_size+1, training_size + 1 + X_test.shape[0], X_test.shape[0])
    plt.plot(x_full_limit, y_all, label='actual', color='b', linewidth=1)
    plt.plot(y_pred_limit, y_pred1, '--', color='r', linewidth=2, label='prediction1')
    plt.plot(y_pred_limit, y_pred2, '--', color='g', linewidth=2, label='prediction2')
    plt.legend(loc=0)
    plt.show()


def print_graph_test(y_test, y_pred1, y_pred2, maxEntries=50):
    #y_pred_limit = min(maxEntries, len(y_test))
    length = min(maxEntries,len(y_test))
    y_pred_limit = np.linspace(1, length, length)
    plt.plot(y_pred_limit, y_test, label='actual', color='b', linewidth=1)
    plt.plot(y_pred_limit, y_pred1, '--', color='r', linewidth=2, label='prediction1')
    plt.plot(y_pred_limit, y_pred2, '--', color='g', linewidth=2, label='prediction2')
    plt.legend(loc=0)
    plt.show()


def almost_correct_based_accuracy(Y_actual, Y_predicted, percentage_cutoof):
    total_count = 0
    error_count = 0
    rmsep_sum = []
    mape_sum = []
    rmse_sum = []
    mean = np.mean(Y_actual)
    print("Mean Actual Prediction Value", mean)

    for i in xrange(0,len(Y_actual)):
        total_count += 1
        percent = 100*abs(Y_predicted[i] - Y_actual[i])/Y_actual[i];
        if Y_actual[i] > 0:
            errorp = (Y_predicted[i] - Y_actual[i])/ Y_actual[i]
            rmsep_sum.append(errorp*errorp)
            mape_sum.append(abs(errorp))
            rmse_sum.append((Y_predicted[i] - Y_actual[i])*(Y_predicted[i] - Y_actual[i]))
        #error_values.append(abs(Y_predicted[i] - Y_actual[i]))
        if percent > percentage_cutoof:
            error_count += 1
        #np.percentile(error_values, 50)

    overall_error_percent = 100*error_count/total_count
    rmsep = sqrt(sum(rmsep_sum)/total_count)
    mape = np.mean(mape_sum)
    rmse = sqrt(sum(rmse_sum)/total_count)
    return overall_error_percent, rmsep, mape,rmse


def ac_loss(Y_actual, Y_predicted):
    print("Y_actual.shape", Y_actual.shape, "Y_predicted.shape", Y_predicted)
    total_count = 0
    error_count = 0
    for i in xrange(0,len(Y_actual)):
        total_count += 1
        percent = 100*abs(Y_predicted[i] - Y_actual[i])/Y_actual[i];
        if percent > 10:
            error_count += 1
    return error_count/total_count


def regression_with_dl(X_train, y_train, X_test, y_test, nodes_in_layer=200,
                       number_of_hidden_layers=5, droput=0.1, activation_fn='relu', epoch_count=100):
    config = MLConfigs()
    config.nodes_in_layer = nodes_in_layer
    config.number_of_hidden_layers = number_of_hidden_layers
    config.droput = droput
    config.activation_fn = activation_fn
    config.epoch_count = epoch_count
    regression_with_dl(X_train, y_train, X_test, y_test, config)


def regression_with_dl(X_train, y_train, X_test, y_test, config):
    model = Sequential()
    model.add(Dense(config.nodes_in_layer, input_dim=X_train.shape[1],activation=config.activation_fn))
    #model.add(Dropout(0.1)) # add dropout
    for i in xrange(0, config.number_of_hidden_layers):
        model.add(Dense(config.nodes_in_layer, activation=config.activation_fn))
        #model.add(Dense(nodes_in_layer, activation=activation_fn, W_regularizer=l2(0.001))) #http://keras.io/regularizers/
        if config.droput > 0:
            model.add(Dropout(config.droput)) # add dropout

    #model.add(Dense(20, activation='relu'))
    model.add(Dense(1))

    #http://sebastianruder.com/optimizing-gradient-descent/index.html
    #adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08) #does not work, but used in other places with binary_crossentropy, categorical_crossentropy
    #adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07) # works
    #adam = Adam(lr=0.01, beta_1=0.99, beta_2=0.999, epsilon=1e-07)
    #adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-06) #No
    #adam = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=True)
    #SGD updated based on lr * (1.0 / (1.0 + decay * it) where it is each run so deacy should be small
    #adam = SGD(lr=0.1, momentum=0.9, decay=0.04, nesterov=True)#very good 51% error (same as ARIMA)
    #adam = SGD(lr=0.1, momentum=0.9, decay=0.001, nesterov=True)#65 but RMSEP improved
    #adam = SGD(lr=0.2, momentum=0.9, decay=0.001, nesterov=True)#56 RMSEP improved
    #adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-07) # slow??
    #adam = SGD(lr=0.1, momentum=0.9, decay=0.001, nesterov=True)#51 RMSEP improved

    #adam = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) #62 RMSEP=0.324359 however, learning rate was almost 0.1 though the time
    #adam = SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True) #27 RMSEP=0.348015 with 1000 per layer
    #adam = SGD(lr=0.1, decay=1e-3, momentum=0.9, nesterov=True) #28 RMSEP=0.341843 with 500 per layer 0.05 dropout

    # this is the best for power forecast
    #adam = SGD(lr=0.1, decay=1e-3, momentum=0.9, nesterov=True) #28 RMSEP=0.341843/ with 500 per layer 39/0.374848 with 0.01 droput, epoches 50 - 0.05 dropout 30/ 0.351425
    #full data set with 0.05 droput 24.0 RMSEP=0.254612
    ##

    if config.optimizer == None:
        #next
        #adam = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        opf = SGD(lr=0.1, decay=0.0005, momentum=0.99, nesterov=True)
        #adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        #one hidden layer
    else:
        opf = config.optimizer

    model.compile(loss=config.loss, optimizer=opf)
    #following passes a custom
    #model.compile(loss=ac_loss, optimizer=opf)





    #default
    #model.compile(loss='mse', optimizer='adam') # loss 10-5

    #model.compile(loss='mae', optimizer='adam') #loss 10-1
    #model.compile(loss='mse', optimizer='sgd') #loss 10-2
    #model.compile(loss='mse', optimizer='rmsprop') #loss 10-2
    #model.compile(loss='mape', optimizer='rmsprop') #loss 10-2 - OK

    #try loss='mean_absolute_percentage_error', optimizer=sgd, loss="mean_absolute_percentage_error", optimizer="rmsprop"

    #writing a custom loss function - https://github.com/fchollet/keras/issues/369



    #optimizer='sgd, RMSprop( usually for RNN), Adagrad, Adadelta, Adam' , see http://keras.io/optimizers/ ( momentum, learning rate etc is calcuated here)
    #loss functions http://keras.io/objectives/
    #activations http://keras.io/activations/
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    lr_logger = LearningRateLogger()
    hist = model.fit(X_train, y_train, nb_epoch=config.epoch_count, batch_size=16, validation_data=(X_test, y_test),
                     callbacks=[early_stop, lr_logger])
    print("history",hist.history)

#    score = model.evaluate(X_test, y_test, batch_size=16)

    loss = model.evaluate(X_test, y_test, batch_size=16, verbose=True)

    #print("DL Loss", "{:.6f}".format(loss))
    y_pred = model.predict(X_test)
    return y_pred


def print_regression_model_summary(prefix, y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    error_AC, rmsep, mape, rmse = almost_correct_based_accuracy(y_test, y_pred, 10)
    print "%s AC_errorRate=%.1f RMSEP=%.6f MAPE=%6f RMSE=%6f mse=%f" %(prefix, error_AC, rmsep, mape, rmse, mse)
    log.write("%s AC_errorRate=%.1f RMSEP=%.6f MAPE=%6f RMSE=%6f mse=%f" %(prefix, error_AC, rmsep, mape, rmse, mse))


# Utility function to report best scores
def report_scores(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


def transform_feild4rnn(power, sequence_length):
    result = []
    for index in range(len(power) - sequence_length):
        result.append(power[index: index + sequence_length])
    result = np.array(result)  # shape (2049230, 50)

    #print("result",result)
    #print("result.shape", result.shape)

    #result_mean = result.mean()
    #result -= result_mean
    #print "Shift : ", result_mean
    #print "Data  : ", result.shape

    row = round(0.9 * result.shape[0])
    train = result[:row, :]
    #np.random.shuffle(train)
    #print("train", train.shape)
    #print("train", train)

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = result[row:, :-1]
    y_test = result[row:, -1]
    return [X_train, y_train, X_test, y_test]


def transform_dataset4rnn(dataset, sequence_length, prediction_index):
    column_count = len(dataset)

    X_train_raw = []
    X_test_raw = []
    for index in range(column_count):
        X_train_t, y_train_t, X_test_t, y_test_t = transform_feild4rnn(dataset[index], sequence_length)
        X_train_raw.append(X_train_t)
        X_test_raw.append(X_test_t)
        if index == prediction_index:
            y_train = np.array(y_train_t)
            y_test = np.array(y_test_t)

    row_count = len(X_train_raw[0])
    #print ("row_count", row_count)
    #print("X_train_raw", X_train_raw)


    X_train = []
    for i in range(row_count):
        temp = []
        window_size = len(X_train_raw[0][i])
        for j in range(window_size):
            list = []
            for k in range(column_count):
                list.append(X_train_raw[k][i][j])
            temp.append(list)
        X_train.append(temp);

    row_count = len(X_test_raw[0])
    X_test = []
    for i in range(row_count):
        temp = []
        window_size = len(X_test_raw[0][i])
        for j in range(window_size):
            list = []
            for k in range(column_count):
                list.append(X_test_raw[k][i][j])
            temp.append(list)
        X_test.append(temp);

    X_train = np.array(X_train)
    X_test = np.array(X_test)



    #X_train, y_train, X_test, y_test = transform_data4rnn(power, sequence_length)
    #X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    #X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    #print("y_train", y_train)
    #print("X_train",X_train)

    return [X_train, y_train, X_test, y_test]


# Utility function to report best scores
def l2norm(nparray):
    return sqrt(np.sum([ x*x for x in nparray]))


def doassert(condtion, msg):
    if not condtion : print msg


def assertSame(val1, val2, msg):
    if np.allclose([val1], [val2]) != True:
        print msg, "but %f != %f" %(val1, val2)



def verify_window_dataset(x_all, y_all):
    #print "================="
    #print("x_all", x_all)
    #print("y_all", y_all)
    #print "================="
    doassert(x_all.shape[0] == len(y_all), "x_all and Y-all are not of same height %d !=  %d" %(x_all.shape[0], len(y_all)))
    for i in range(x_all.shape[0] - 1):
        for j in range(1,x_all.shape[1],1):
            val1 = x_all[i][j]
            val2 = x_all[i+1][j-1]
            assertSame(val1, val2, "row %d,%d " %(i, j))
        assertSame(x_all[i+1][x_all.shape[1]-1], y_all[i], "last value of row %d" %(i+1))




def print_feature_importance(X_test, y_test, importances):
    # print feature importance
    # http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    #std = np.std([tree.feature_importances_ for tree in rfr.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]

#    for i in range(20):
#        print(X_test[i][6],"->", y_test[i])

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_test.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


def regression_with_GBR(X_train, y_train, X_test, y_test, params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
          'learning_rate': 0.01, 'loss': 'ls'}):
        #GradientBoostingRegressor
    gfr = GradientBoostingRegressor(**params)
    gfr.fit(X_train, y_train)
    y_pred_gbr = gfr.predict(X_test)
    print_regression_model_summary("GBR", y_test, y_pred_gbr)
    print_feature_importance(X_test, y_test,gfr.feature_importances_)

    #cross validation ( not sure this make sense for regression
    #http://scikit-learn.org/stable/modules/cross_validation.html
    #gfr = GradientBoostingRegressor(**params)
    #scores = cross_validation.cross_val_score(gfr, X_train, y_train, cv=5)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return y_pred_gbr


def regression_with_LR(X_train, y_train, X_test, y_test):
    lr = LinearRegression(normalize=True)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print_regression_model_summary("LR", y_test, y_pred_lr)
    return y_pred_lr



def regression_with_RFR(X_train, y_train, X_test, y_test):
    rfr = RandomForestRegressor()
    rfr.fit(X_train, y_train)
    y_pred_rfr = rfr.predict(X_test)
    print_regression_model_summary("RFR", y_test, y_pred_rfr)
    print_feature_importance(X_test, y_test,rfr.feature_importances_)
    return y_pred_rfr



