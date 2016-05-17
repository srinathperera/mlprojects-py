import numpy as np
import pandas as pd

from keras.layers import containers, AutoEncoder, Dense
from keras import models
from sklearn import preprocessing

# input shape: (nb_samples, 32)
encoder = containers.Sequential([Dense(4, input_dim=4), Dense(2)])
decoder = containers.Sequential([Dense(4, input_dim=2), Dense(4)])

df = pd.read_csv("8.csv")
X_train = df.values.copy()

#loss in 0.23
#X_train = preprocessing.scale(X_train.astype("float32"))

#loss is 0.02
X_train = preprocessing.normalize(X_train.astype("float32"), norm='l2')


autoencoder = AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True)
model = models.Sequential()
model.add(autoencoder)

# training the autoencoder:
model.compile(optimizer='sgd', loss='mse')
model.fit(X_train, X_train, nb_epoch=20)

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

np.savetxt("output.csv", X_predicted, delimiter=",")
