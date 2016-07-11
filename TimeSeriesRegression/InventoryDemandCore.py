




class MLConfigs:
    def __init__(self, nodes_in_layer = 20, number_of_hidden_layers = 3, dropout = 0, activation_fn='relu', loss= "mse",
              epoch_count = 10, optimizer = Adam(), regularization=0):
        self.nodes_in_layer = nodes_in_layer
        self.number_of_hidden_layers = number_of_hidden_layers
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.epoch_count = epoch_count
        self.optimizer = optimizer
        self.loss = loss
        self.regularization = regularization

    def tostr(self):
        return "NN %dX%d dp=%2f/%4f at=%s loss=%s op=%s epoches=%d" %(self.nodes_in_layer,self.number_of_hidden_layers, self.dropout,
                            self.regularization, self.activation_fn, self.loss, self.optimizer.get_config(), self.epoch_count)
