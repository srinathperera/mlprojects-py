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
    X_list = [X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1),
                    X[:, 2].reshape(-1, 1), X[:, 3].reshape(-1, 1),
                    X[:, 4].reshape(-1, 1)
              ]


    '''
    X_list = []
    #Forecasting Feilds ['(0)Semana', '(1)Agencia_ID', '(2)Canal_ID', '(3)Ruta_SAK', '(4)Cliente_ID', '(5)Producto_ID']

    week = X[..., [0]]
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
    '''

    return X_list

def train_model_with_embeddings(X_train, y_train, X_test, y_test):
    models = []
    #    groups = ('Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID')
    #Forecasting Feilds ['(0)Semana', '(1)Agencia_ID', '(2)Canal_ID', '(3)Ruta_SAK', '(4)Cliente_ID', '(5)Producto_ID']

    '''
    Producto_ID 1799 coverage 0.225799153804 49997
Agencia_ID 552 coverage 0.085824901284 25759
Canal_ID 9 coverage 1.0 11
Ruta_SAK 3603 coverage 0.0545328888749 9991
Cliente_ID 880604 coverage 0.00190191584674 2015152015
    '''
    model_week = Sequential()
    model_week.add(Dense(1, input_dim=1))
    models.append(model_week)

    model_agency = Sequential()
    model_agency.add(Embedding(552, 2, input_length=1))
    model_agency.add(Reshape(dims=(2,)))
    models.append(model_agency)

    model_canal = Sequential()
    model_canal.add(Embedding(9, 2, input_length=1))
    model_canal.add(Reshape(dims=(2,)))
    models.append(model_canal)

    #model_ruta_sak = Sequential()
    #model_ruta_sak.add(Embedding(1905, 7, input_length=1)) #3603
    #model_ruta_sak.add(Reshape(dims=(7,)))
    #models.append(model_ruta_sak)

    #model_client = Sequential()
    #model_client.add(Embedding(657092, 20, input_length=1)) #880604
    #model_client.add(Reshape(dims=(20,)))
    #models.append(model_client)

    model_product = Sequential()
    model_product.add(Embedding(1799, 3, input_length=1))
    model_product.add(Reshape(dims=(3,)))
    models.append(model_product)

    model_rs_client = Sequential()
    model_rs_client.add(Embedding(1700000, 3, input_length=1)) #880604
    model_rs_client.add(Reshape(dims=(3,)))
    models.append(model_rs_client)



    model = Sequential()
    model.add(Merge(models, mode='concat'))
    model.add(Dense(100, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(100, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='mean_absolute_error', optimizer='adam')

    model.fit(X_train, y_train,
                       validation_data=(X_test, y_test),
                       nb_epoch=5, batch_size=4096)

    return model




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


def combine_routes_client_categories2continous_intergers(df):
    feild_name = 'Ruta_SAK_Cliente_ID'
    counts = df.groupby(['Ruta_SAK', 'Cliente_ID'])['Demanda_uni_equil'].count()
    valuesDf = counts.to_frame("Count")
    valuesDf.reset_index(inplace=True)
    valuesDf = valuesDf.sort_values(by=['Count'], ascending=False)
    valuesDf['c'+feild_name] = range(valuesDf.shape[0])
    valuesDf.to_csv(feild_name+ '_data.csv', index=False)
    print feild_name+ '_data.csv Done'






def categories2continous_intergers(df):
    feilds = ['Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID']
    for feild_name in feilds:
        counts = df[feild_name].value_counts()
        counts = counts.sort_values(ascending=False)
        valuesDf = counts.to_frame("Count")
        valuesDf.reset_index(inplace=True)
        valuesDf['c'+feild_name] = range(valuesDf.shape[0])
        valuesDf.to_csv(feild_name+ '_data.csv', index=False)
        print feild_name+ '_data.csv Done'


def replace_ids(df, feild_name, data_file_name, cutoff):
    t = time.time()
    id_df = pd.read_csv(data_file_name)
    id_df = id_df[id_df['Count'] > cutoff]
    id_df = drop_feilds_1df(id_df, ['Count'])
    merged = pd.merge(df, id_df, on=[feild_name])
    merged = drop_feilds_1df(merged, [feild_name])
    print "took ", (time.time() -t), "s"
    return merged

def replace_ruta_sak_clientid(df, data_file_name, cutoff):
    t = time.time()
    id_df = pd.read_csv(data_file_name)
    id_df = id_df[id_df['Count'] > cutoff]
    id_df = drop_feilds_1df(id_df, ['Count'])
    merged = pd.merge(df, id_df, on=['Ruta_SAK', 'Cliente_ID'])
    merged = drop_feilds_1df(merged, ['Ruta_SAK', 'Cliente_ID'])
    print "took ", (time.time() -t), "s"
    return merged





def print_category_stats():
    test_df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/test.csv')
    tot_test_count = test_df.shape[0]

    #files = ['Agencia_ID_data.csv', 'Canal_ID_data.csv', 'Ruta_SAK_data.csv', 'Cliente_ID_data.csv', 'Producto_ID_data.csv']
    files = ['Agencia_ID_data.csv', 'Ruta_SAK_data.csv', 'Cliente_ID_data.csv', 'Producto_ID_data.csv']
    feilds = ['Agencia_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID']
    cutoff = [500, 500, 20, 250]
    index = 0
    for f in files:
        id_df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/category_data/'+f)
        id_df = id_df[id_df['Count'] > cutoff[index]]
        print id_df.describe()

        merged = pd.merge(test_df, id_df,  on=[feilds[index]])
        print feilds[index], " top covers "+ str(float(merged.shape[0])/tot_test_count) + "of tests"
        index = index +1

    id_df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/category_data/Ruta_SAK_Cliente_ID_data.csv')
    id_df = id_df[id_df['Count'] > 10]
    print id_df.describe()

    merged = pd.merge(test_df, id_df,  on=['Ruta_SAK', 'Cliente_ID'])
    print "Ruta_SAK_Cliente_ID_data top covers "+ str(float(merged.shape[0])/tot_test_count) + "of tests"



    '''
    train_df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/train.csv')
    train_size = train_df.shape[0]
    merged = train_df
    index = 0
    for f in files:
        feild_name = feilds[index]
        id_df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/category_data/'+f)
        id_df = id_df[id_df['Count'] > cutoff[index]]
        merged = merged[merged[feilds[index]].isin(id_df[feilds[index]])]
        #merged = pd.merge(merged, id_df,  on=[feild_name])

        print "after ",feild_name, " size=", float(merged.shape[0])/train_size
        index = index +1
    '''


def remove_rare_categories(train_df):
    files = ['Agencia_ID_data.csv', 'Producto_ID_data.csv']
    feilds = ['Agencia_ID', 'Producto_ID']
    cutoff = [500, 250]

    train_size = train_df.shape[0]
    merged = train_df
    index = 0
    for f in files:
        feild_name = feilds[index]
        id_df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/category_data/'+f)
        id_df = id_df[id_df['Count'] > cutoff[index]]
        merged = merged[merged[feilds[index]].isin(id_df[feilds[index]])]
        #merged = pd.merge(merged, id_df,  on=[feild_name])

        print "after ",feild_name, " size=", float(merged.shape[0])/train_size
        index = index +1

    id_df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/category_data/Ruta_SAK_Cliente_ID_data.csv')
    id_df = id_df[id_df['Count'] > 10]
    merged = pd.merge(merged, id_df,  on=['Ruta_SAK', 'Cliente_ID'])
    print "after Ruta_SAK_Cliente_ID size=", float(merged.shape[0])/train_size

    return merged


def run_model():
    target_as_log = True
    df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems30000_35000.csv')
    df = drop_feilds_1df(df, ['Venta_uni_hoy', 'Venta_hoy', 'Dev_uni_proxima', 'Dev_proxima'])

    #df = remove_rare_categories(df) # not needed, done in inner join

    #Agencia_ID', '(2)Canal_ID', '(3)Ruta_SAK', '(4)Cliente_ID', '(5)Producto_ID
    df = replace_ids(df, 'Agencia_ID', '/Users/srinath/playground/data-science/BimboInventoryDemand/category_data/Agencia_ID_data.csv', 500)
    df = replace_ids(df, 'Canal_ID', '/Users/srinath/playground/data-science/BimboInventoryDemand/category_data/Canal_ID_data.csv', 0)
    #df = replace_ids(df, 'Ruta_SAK', '/Users/srinath/playground/data-science/BimboInventoryDemand/category_data/Ruta_SAK_data.csv')
    #df = replace_ids(df, 'Cliente_ID', '/Users/srinath/playground/data-science/BimboInventoryDemand/category_data/Cliente_ID_data.csv')
    df = replace_ids(df, 'Producto_ID', '/Users/srinath/playground/data-science/BimboInventoryDemand/category_data/Producto_ID_data.csv', 250)

    df = replace_ruta_sak_clientid(df, '/Users/srinath/playground/data-science/BimboInventoryDemand/category_data/Ruta_SAK_Cliente_ID_data.csv', 10)


    training_set_size = int(0.7*df.shape[0])
    test_set_size = df.shape[0] - training_set_size

    y_all = df['Demanda_uni_equil'].values

    train_df = df[:training_set_size]
    test_df = df[-1*test_set_size:]

    y_actual_train = y_all[:training_set_size]
    y_actual_test = y_all[-1*test_set_size:]

    y_train = y_actual_train
    y_test = y_actual_test
    if target_as_log:
        y_train = transfrom_to_log(y_train)
        y_test = transfrom_to_log(y_test)

    train_df, test_df = drop_column(train_df, test_df, 'Demanda_uni_equil')

    forecasting_feilds = list(train_df)
    print "Forecasting Feilds", [ "("+str(i)+")" + forecasting_feilds[i] for i in range(len(forecasting_feilds))]

    y_train, parmsFromNormalization = preprocess1DtoZeroMeanUnit(y_train)
    y_test = apply_zeroMeanUnit(y_test, parmsFromNormalization)

    X_train = split_features(train_df.values.copy())
    X_test = split_features(test_df.values.copy())

    model = train_model_with_embeddings(X_train, y_train, X_test, y_test)

    #test_embeddings(train_df, y_train)

    check_accuracy("DL with embedding", model, X_test, parmsFromNormalization, test_df, target_as_log, y_actual_test, "-1")

#https://groups.google.com/forum/#!topic/keras-users/4k2VMUApfoM
def test_embeddings(df, y_train):

    df = df[['Agencia_ID','Canal_ID']]
    X = np.array(df.values.copy())
    y = y_train
    print "Before", X.shape, y.shape

    print np.max(X[:,0]), np.max(X[:,1])

    #X = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [3, 4], [5,6]])
    X_list = [X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1)]

    #y = np.array([0, 0, 0, 1, 1, 6])

    print "After", X.shape, y.shape, X_list[0].shape, X_list[1].shape


    embed1 = Sequential()
    embed1.add(Embedding(30000, 5, input_length=1))
    embed1.add(Reshape(dims=(5,)))

    embed2 = Sequential()
    embed2.add(Embedding(12, 3, input_length=1))
    embed2.add(Reshape(dims=(3,)))


    model = Sequential()
    model.add(Merge([embed1, embed2], mode='concat'))
    model.add(Dense(32, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dense(1, init='uniform'))
    model.add(Activation('sigmoid'))
    model.compile(loss='mean_absolute_error', optimizer='adam')

    model.fit(X_list, y, nb_epoch=10, batch_size=4)

#print_category_stats()
#test_embeddings()
run_model()

#df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/train.csv')
#categories2continous_intergers(df)
#combine_routes_client_categories2continous_intergers(df)


