
import numpy as np

from sklearn.datasets import make_moons 
# import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


n_features = 2 
n_samples = 2000
n_classes = 2
lr = 0.001

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

def sigmoid(x, derivative = False):
    if derivative:
      return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))





class Dense():
    def __init__(self, input_neurons: int, output_neurons: int, activation, learning_rate: float):
        self.Weights = np.random.randn(input_neurons, output_neurons)
        self.Biases = np.random.randn(1, output_neurons)
        self.Activation = activation
        self.LR = learning_rate

    def Forward(self, x: np.array):
        assert x.shape[1] == self.Weights.shape[0]
        Z1 = np.dot(x, self.Weights)
        A1 = self.Activation(Z1)

        return Z1, A1


    def Backprop(self, Z1: np.array, A1: np.array, gradient: np.array, next_weights=None):
        assert gradient.shape[1] == self.Weights.shape[1]
        
        if not next_weights:
            delta = self.Activation(Z1, derivative = True) *  gradient # (samples, output_weights) 
        else:
            # (samples, output_weights) * (output_weights, next_layer_weights) DOT (samples, output_weights)
            delta = self.Activation(Z1, derivative = True) *  np.dot(next_weights, gradient) 

        WeightUpdate = np.dot(A1.T, delta) 
        BiasUpdate = np.sum(delta, axis = 0, keepdims=True)

        self.Weights += -self.LR * WeightUpdate
        self.Biases += -self.LR * BiasUpdate

        return np.dot(delta,self.Weights.T)

D1 = Dense(input_neurons=2, output_neurons=10, activation=sigmoid, learning_rate=1)
D2 = Dense(input_neurons=10, output_neurons=2, activation=sigmoid, learning_rate=1)

Z1, A1 = D1.Forward(x_train)
Z2, A2 = D2.Forward(A1)



loss = MSELoss(y_truth, A2, derivative=True)

gradient = D2.Backprop(Z2, A1, loss)

print(gradient.shape)