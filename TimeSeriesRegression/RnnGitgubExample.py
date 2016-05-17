import matplotlib.pyplot as plt
import numpy as np
import time
import math
import csv
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from mltools import transform_dataset4rnn, transform_feild4rnn
np.random.seed(1234)





def data_power_consumption(path_to_dataset,
                           sequence_length=50,
                           ratio=1.0):

    max_values = ratio * 2049280

    with open(path_to_dataset) as f:
        data = csv.reader(f, delimiter=";")
        power = []
        nb_of_values = 0
        for line in data:
            try:
                power.append(float(line[2]))
                nb_of_values += 1
            except ValueError:
                pass
            # 2049280.0 is the total number of valid values, i.e. ratio = 1.0
            if nb_of_values >= max_values:
                break

    #power = list(range(1, 100, 1))*1.0
    data_size = 200
    #power1 = [math.sin(math.pi *x/16) for x in range(0, data_size)]
    #power2 = [math.sin(math.pi *x/32) for x in range(0, data_size)]
    power1 = [x * 1.0 for x in range(0, data_size)]
    power2 = [x * 0.1 for x in range(0, data_size)]


    #print("power1",power1)
    print "Data loaded from csv. Formatting..."

    print("power.shape", len(power1))

    return transform_dataset4rnn([power1, power2], sequence_length, 0)




def build_model():
    model = Sequential()
    layers = [2, 50, 100, 1]

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print "Compilation Time : ", time.time() - start
    return model


def run_network(model=None, data=None):
    global_start_time = time.time()
    epochs = 10
    ratio = 0.5
    sequence_length = 50
    path_to_dataset = 'household_power_consumption.txt'

    if data is None:
        print 'Loading data... '
        X_train, y_train, X_test, y_test = data_power_consumption(
            path_to_dataset, sequence_length, ratio)
    else:
        X_train, y_train, X_test, y_test = data

    print '\nData Loaded. Compiling...\n'

    if model is None:
        model = build_model()

    try:
        print("X_train.shape", X_train.shape)
        print("Y_train.shape", y_train.shape)
        model.fit(
            X_train, y_train,
            batch_size=512, nb_epoch=epochs, validation_split=0.05)
        predicted = model.predict(X_test)
        print("predicted2 shape", predicted.shape)
        predicted = np.reshape(predicted, (predicted.size,))
        print("predicted2 shape", predicted.shape)
    except KeyboardInterrupt:
        print 'Training duration (s) : ', time.time() - global_start_time
        return model, y_test, 0

    try:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(y_test[:100, 0])
        plt.plot(predicted[:100, 0])
        plt.show()
    except Exception as e:
        print str(e)
    print 'Training duration (s) : ', time.time() - global_start_time

    return model, y_test, predicted

data = data_power_consumption('household_power_consumption1000.csv')
model = build_model()
run_network(model, data)