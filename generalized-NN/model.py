from re import X
import numpy as np

from sklearn.datasets import make_moons 
# import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from dense import Dense


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

def MSELoss(y_pred, y_truth, derivative = False):
    if derivative:
      return -1 * (y_truth - y_pred)
    return np.power((y_truth - y_pred), 2) / 2


class Model():
    def __init__(self, epochs: int, learning_rate=0.001):
        self.layers = []
        self.learning_rate = learning_rate
        self.epochs = epochs

    def add(self, layer: Dense):
        self.layers.append(layer)

    def _forward(self, x):
        Z1 = None
        A1 = x 
        for layer in self.layers:
            Z1, A1 = layer.Forward(A1)
        
        return Z1, A1

    def Backprop(self, x, y):
        for i in range(self.epochs):
            Z, A = self._forward(x)
            gradient = MSELoss(A, y, derivative=True)
            for i in range(len(self.layers) -1 , -1, -1):
                layer = self.layers[i]
                A0 = layer.Old_A0
                Z1 = layer.Old_Z1
                gradient = layer.Backprop(Z1, A0, gradient)



    def fit(self, x, y):
        n_samples_x, input_neurons = x.shape
        n_samples_y, output_neuons = y.shape

        assert n_samples_y == n_samples_x 

        input_layer = Dense(input_neurons, self.layers[0].input_neurons, activation=sigmoid)
        self.layers.insert(0, input_layer)

        output_layer = Dense(self.layers[-1].output_neurons, output_neuons, activation=sigmoid)
        self.layers.insert(-1, output_layer)

        

def sigmoid(x, derivative = False):
    if derivative:
      return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))


m = Model(epochs=1000, learning_rate=0.001)
m.add(Dense(2, 10, sigmoid))
m.add(Dense(10, 2, sigmoid))

m.Backprop(x_train, y_truth)

Z1, A1 = m._forward(x_test)


results = np.argmax(A1, axis = 1)
print(np.sum(results == y_test) / len(y_test))

