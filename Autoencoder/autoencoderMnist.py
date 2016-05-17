import numpy as np
import pandas as pd

from keras.layers import containers, AutoEncoder, Dense
from keras import models
from keras.datasets import mnist

# input shape: (nb_samples, 32)
encoder = containers.Sequential([Dense(300, input_dim=784), Dense(10)])
decoder = containers.Sequential([Dense(300, input_dim=10), Dense(784)])


# the data, shuffled and split between train and test sets
(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

autoencoder = AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True)
model = models.Sequential()
model.add(autoencoder)

# training the autoencoder:
model.compile(optimizer='sgd', loss='mse')
model.fit(X_train, X_train, nb_epoch=10)

X_predicted = model.predict(X_train)

print('train: ', X_train.shape, "predicted", X_predicted.shape)

score = model.evaluate(X_train, X_predicted,
                       show_accuracy=True, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print(X_train)
print(X_predicted)


# predicting compressed representations of inputs:
autoencoder.output_reconstruction = False  # the model has to be recompiled after modifying this property
model.compile(optimizer='sgd', loss='mse')
X_predicted = model.predict(X_train)

print('train: ', X_predicted.shape);
