import numpy
import pickle

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Reshape
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from inventory_demand import *
from mltools import *



def save_embeddings(model, saved_embeddings_fname):
    weights = model.get_weights()
    store_embedding = weights[0]
    dow_embedding = weights[1]
    year_embedding = weights[4]
    month_embedding = weights[5]
    day_embedding = weights[6]
    german_states_embedding = weights[7]
    with open(saved_embeddings_fname, 'wb') as f:
        pickle.dump([store_embedding, dow_embedding, year_embedding,
                    month_embedding, day_embedding, german_states_embedding], f, -1)


def split_features(X):
    X_list = []
    #Forecasting Feilds ['(0)Semana', '(1)Agencia_ID', '(2)Canal_ID', '(3)Ruta_SAK', '(4)Cliente_ID', '(5)Producto_ID']

    week = X[..., 0]
    X_list.append(week)

    agency = X[..., [1]]
    X_list.append(agency)

    canal = X[..., [2]]
    X_list.append(canal)

    ruta_sak = X[..., [3]]
    X_list.append(ruta_sak)

    client = X[..., [4]]
    X_list.append(client)

    product = X[..., [5]]
    X_list.append(product)

    #print week.shape
    print agency.shape
    print canal.shape

    return X_list

def train_model_with_embeddings(X_train, y_train, X_test, y_test):
    models = []
    #    groups = ('Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID')
    #Forecasting Feilds ['(0)Semana', '(1)Agencia_ID', '(2)Canal_ID', '(3)Ruta_SAK', '(4)Cliente_ID', '(5)Producto_ID']
    model_week = Sequential()
    model_week.add(Dense(1, input_dim=1))
    models.append(model_week)

    model_agency = Sequential()
    model_agency.add(Embedding(552, 5, input_length=1))
    model_agency.add(Reshape((5,)))
    models.append(model_agency)

    model_canal = Sequential()
    model_canal.add(Embedding(9, 3, input_length=1))
    model_canal.add(Reshape((3,)))
    models.append(model_canal)

    model_ruta_sak = Sequential()
    model_ruta_sak.add(Embedding(3603, 7, input_length=1))
    model_ruta_sak.add(Reshape((7,)))
    models.append(model_ruta_sak)

    model_client = Sequential()
    model_client.add(Embedding(880604, 20, input_length=1))
    model_client.add(Reshape((20,)))
    models.append(model_client)

    model_product = Sequential()
    model_product.add(Embedding(1799, 10, input_length=1))
    model_product.add(Reshape((10,)))
    models.append(model_product)

    model = Sequential()
    model.add(Merge(models, mode='concat'))
    model.add(Dense(1000, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dense(500, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='mean_absolute_error', optimizer='adam')

    model.fit(X_train, y_train,
                       validation_data=(X_test, y_test),
                       nb_epoch=10, batch_size=128)




def embed_features(X, saved_embeddings_fname):
    # f_embeddings = open("embeddings_shuffled.pickle", "rb")
    f_embeddings = open(saved_embeddings_fname, "rb")
    embeddings = pickle.load(f_embeddings)

    index_embedding_mapping = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5}
    X_embedded = []

    (num_records, num_features) = X.shape
    for record in X:
        embedded_features = []
        for i, feat in enumerate(record):
            feat = int(feat)
            if i not in index_embedding_mapping.keys():
                embedded_features += [feat]
            else:
                embedding_index = index_embedding_mapping[i]
                embedded_features += embeddings[embedding_index][feat].tolist()

        X_embedded.append(embedded_features)

    return numpy.array(X_embedded)


class NN_with_EntityEmbedding():

    def __init__(self, X_train, y_train, X_val, y_val):
        self.nb_epoch = 10
        self.checkpointer = ModelCheckpoint(filepath="best_model_weights.hdf5", verbose=1, save_best_only=True)
        self.max_log_y = max(numpy.max(numpy.log(y_train)), numpy.max(numpy.log(y_val)))
        self.__build_keras_model()
        self.fit(X_train, y_train, X_val, y_val)

    #def preprocessing(self, X):
    #    X_list = split_features(X)
    #    return X_list

    def __build_keras_model(self):
        models = []

        model_store = Sequential()
        model_store.add(Embedding(1115, 10, input_length=1))
        model_store.add(Reshape(target_shape=(10,)))
        models.append(model_store)

        model_dow = Sequential()
        model_dow.add(Embedding(7, 6, input_length=1))
        model_dow.add(Reshape(target_shape=(6,)))
        models.append(model_dow)

        model_promo = Sequential()
        model_promo.add(Dense(1, input_dim=1))
        models.append(model_promo)

        model_year = Sequential()
        model_year.add(Embedding(3, 2, input_length=1))
        model_year.add(Reshape(target_shape=(2,)))
        models.append(model_year)

        model_month = Sequential()
        model_month.add(Embedding(12, 6, input_length=1))
        model_month.add(Reshape(target_shape=(6,)))
        models.append(model_month)

        model_day = Sequential()
        model_day.add(Embedding(31, 10, input_length=1))
        model_day.add(Reshape(target_shape=(10,)))
        models.append(model_day)

        model_germanstate = Sequential()
        model_germanstate.add(Embedding(12, 6, input_length=1))
        model_germanstate.add(Reshape(target_shape=(6,)))
        models.append(model_germanstate)

        self.model = Sequential()
        self.model.add(Merge(models, mode='concat'))
        self.model.add(Dense(1000, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(500, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='mean_absolute_error', optimizer='adam')

    def evaluate(self, X_val, y_val):
        assert(min(y_val) > 0)
        guessed_sales = self.guess(X_val)
        relative_err = numpy.absolute((y_val - guessed_sales) / y_val)
        result = numpy.sum(relative_err) / len(y_val)
        return result


    def _val_for_fit(self, val):
        val = numpy.log(val) / self.max_log_y
        return val

    def _val_for_pred(self, val):
        return numpy.exp(val * self.max_log_y)

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(self.preprocessing(X_train), self._val_for_fit(y_train),
                       validation_data=(self.preprocessing(X_val), self._val_for_fit(y_val)),
                       nb_epoch=self.nb_epoch, batch_size=128,
                       # callbacks=[self.checkpointer],
                       )
        # self.model.load_weights('best_model_weights.hdf5')
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, features):
        features = self.preprocessing(features)
        result = self.model.predict(features).flatten()
        return self._val_for_pred(result)

def run_model():
    df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems5000_10000.csv')
    df = drop_feilds_1df(df, ['Venta_uni_hoy', 'Venta_hoy', 'Dev_uni_proxima', 'Dev_proxima'])

    training_set_size = int(0.7*df.shape[0])
    test_set_size = df.shape[0] - training_set_size

    y_all = df['Demanda_uni_equil'].values

    df['Demanda_uni_equil'] = transfrom_to_log(df['Demanda_uni_equil'].values)

    train_df = df[:training_set_size]
    test_df = df[-1*test_set_size:]

    y_actual_train = y_all[:training_set_size]
    y_actual_test = y_all[-1*test_set_size:]

    train_df, test_df = drop_column(train_df, test_df, 'Demanda_uni_equil')

    forecasting_feilds = list(train_df)
    print "Forecasting Feilds", [ "("+str(i)+")" + forecasting_feilds[i] for i in range(len(forecasting_feilds))]

    y_train, parmsFromNormalization = preprocess1DtoZeroMeanUnit(y_actual_train)
    y_test = apply_zeroMeanUnit(y_actual_test, parmsFromNormalization)

    X_train = split_features(train_df.values.copy())
    X_test = split_features(test_df.values.copy())

    train_model_with_embeddings(X_train, y_train, X_test, y_test)



    #y_pred_final = check_accuracy("DL with embedding", X_test, parmsFromNormalization, test_df,
    #                                  True, y_actual_test, "1")


run_model()