
import numpy as np

from sklearn.datasets import make_moons 
# import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import numpy as np 

class Dense():
    def __init__(self, input_neurons: int, output_neurons: int, activation):
        # Create weights of shape (num_inputs, num_outputs)
        self.Weights = np.random.randn(input_neurons, output_neurons)

        # Create bias of shape (1, num_outputs)
        self.Biases = np.random.randn(1, output_neurons)

        # Define Activation Functions 
        self.Activation = activation

        self.input_neruons = input_neurons

        self.output_neurons = output_neurons

        self.Old_A0 = None
        self.Old_Z1 = None

    def Forward(self, x: np.array):

        # Check if dot product between inputs 
        # and weights is defined 
        assert x.shape[1] == self.Weights.shape[0]

        # Unactivated Dot Product 
        Z1 = np.dot(x, self.Weights)

        self.Old_A0 = x
        self.Old_Z1 = Z1

        # Activated Dot Product 
        A1 = self.Activation(Z1)

        return Z1, A1


    def Backprop(self, Z1: np.array, A0: np.array, gradient: np.array, learning_rate = 0.0):
        # Z1: Unactivated outputs for this layer 
        # A0: Activated outputs of previous layer 
        # Gradient: Gradient with respect to the weights 

        ## Assert that the gradient
        ## has the same number of neurons 
        ## as the outputs 
        assert gradient.shape[1] == self.Weights.shape[1]

        ## Copy the weights to pass to previous 
        ## Layers gradient 
        OldWeights = self.Weights.copy()

        ## if there are no forward weights, 
        ## we take the gradient only 

        delta = self.Activation(Z1, derivative = True) *  gradient # (samples, output_weights) 
        # else:
        #     # (samples, output_weights) * (output_weights, next_layer_weights) DOT (samples, output_weights)
        #     # The total error is defined as element-wise multiplication of 
        #     # derivative of activation with respect to its input 
        #     # and the gradient of each respective weight 
        #     print(next_weights.shape, gradient.shape)
        #     delta = self.Activation(Z1, derivative = True) *  np.dot(next_weights, gradient) 
        
        # weight is updated with respect 
        # to the error times each input 
        # value of the neuron in the previous 
        # layer 
        WeightUpdate = np.dot(A0.T, delta)  
        BiasUpdate = np.sum(delta, axis = 0, keepdims=True)

        # we update both of the weights
        self.Weights += -learning_rate * WeightUpdate
        self.Biases += -learning_rate * BiasUpdate
        
        # we return the gradient 
        # with respect to the current weights 
        return np.dot(delta, OldWeights.T)




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

L1 = Dense(2, 10, sigmoid)
L2 = Dense(10, 2, sigmoid)


for i in range(10000):
    Z1, A1, = L1.Forward(x_train)
    Z2, A2, = L2.Forward(A1)

    loss = MSELoss(A2, y_truth, derivative=True)

    grad = L2.Backprop(Z2, A1, loss, learning_rate=0.001) 
    L1.Backprop(Z1, x_train, grad, learning_rate=0.001)


Z1, A1, = L1.Forward(x_test)
Z2, A2, = L2.Forward(A1)

pred = np.argmax(A2, axis = 1)
print(np.sum(pred == y_test) / len(y_test))
