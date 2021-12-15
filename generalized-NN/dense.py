import numpy as np 

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