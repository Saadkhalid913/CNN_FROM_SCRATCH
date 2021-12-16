import numpy as np
from sklearn.datasets import make_moons 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from Activations import sigmoid
from Loss_Functions import MSELoss, MAELoss
from Dense import Dense




class Model():
    def __init__(self, epochs: int, learning_rate=0.0, loss=MSELoss):
        self.layers = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss = loss 

    def add(self, layer: Dense):
        self.layers.append(layer)

    def _forward(self, x):
        Z1 = None # unactivated output  
        A1 = x    # activated output 
        for layer in self.layers:
            Z1, A1 = layer.Forward(A1)
        
        return Z1, A1 
    
    

    def Backprop(self, x, y):
        for i in range(self.epochs):
            # We keep track of the activated and unactivated output 
            Z, A = self._forward(x)

            # the initial gradient will be 
            # the derivative of the loss with 
            # respect to y_pred 
            gradient = self.loss(A, y, derivative=True)

            

            for i in range(len(self.layers) -1 , -1, -1):
                # we interate through each layer in the 
                # network and calculate the gradients 
                # with respect to the input values 
                layer = self.layers[i]

                # We extract the unactivated output 
                # as well as the inputs of the layer 
                # from the internal state, this is used 
                # for backprop in the previous layer
                A0 = layer.Old_A0
                Z1 = layer.Old_Z1
                gradient = layer.Backprop(Z1, A0, gradient, learning_rate = self.learning_rate)



    def fit(self, x, y):

        # we extract the shape of the inputs 
        n_samples_x, input_neurons = x.shape
        n_samples_y, output_neuons = y.shape


        # we assert that the validation data
        # has the same number of samples as 
        # the input data 
        assert n_samples_y == n_samples_x 


        # we create input layers as well as the output 
        # layer to correspond to the features of x and 
        # and the shape of y 
        input_layer = Dense(input_neurons, self.layers[0].input_neurons, activation=sigmoid)
        self.layers.insert(0, input_layer)

        output_layer = Dense(self.layers[-1].output_neurons, output_neuons, activation=sigmoid)
        self.layers.insert(-1, output_layer)

        

if __name__ == "__main__":
    for i in range(5):
        n_features = 2 
        n_samples = 10000
        n_classes = 2
        lr = 0.01

        x,y = make_moons(n_samples)
        x_train, x_test, y_train, y_test = train_test_split(x,y)

        n_samples = x_train.shape[0]

        y_truth = np.zeros((n_samples, 2))

        for i in range(n_samples):
            y_truth[i][y_train[i]] = 1


        m = Model(epochs=1000, learning_rate=0.005, loss=MAELoss)
        L1 = Dense(2, 10, sigmoid)
        L2 = Dense(10, 2, sigmoid)
        m.add(L1)
        m.add(L2)
        m.Backprop(x_train, y_truth)

        Z1, A1 = m._forward(x_test)


        results = np.argmax(A1, axis = 1)
        print("MAELoss", 100 * np.sum(results == y_test) / len(y_test) , "%")

    for i in range(5):
        n_features = 2 
        n_samples = 10000
        n_classes = 2
        lr = 0.01

        x,y = make_moons(n_samples)
        x_train, x_test, y_train, y_test = train_test_split(x,y)

        n_samples = x_train.shape[0]

        y_truth = np.zeros((n_samples, 2))

        for i in range(n_samples):
            y_truth[i][y_train[i]] = 1


        m = Model(epochs=1000, learning_rate=0.005, loss=MSELoss)
        L1 = Dense(2, 10, sigmoid)
        L2 = Dense(10, 2, sigmoid)
        m.add(L1)
        m.add(L2)
        m.Backprop(x_train, y_truth)

        Z1, A1 = m._forward(x_test)


        results = np.argmax(A1, axis = 1)
        print("MSELoss", 100 * np.sum(results == y_test) / len(y_test) , "%")

