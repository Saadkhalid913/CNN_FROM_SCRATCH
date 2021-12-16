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


    def Backprop(self, Z1: np.array, A0: np.array, gradient: np.array, learning_rate = 0.001, next_weights=None):
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
        if not next_weights:
            delta = self.Activation(Z1, derivative = True) *  gradient # (samples, output_weights) 
        else:
            # (samples, output_weights) * (output_weights, next_layer_weights) DOT (samples, output_weights)
            # The total error is defined as element-wise multiplication of 
            # derivative of activation with respect to its input 
            # and the gradient of each respective weight 
            delta = self.Activation(Z1, derivative = True) *  np.dot(next_weights, gradient) 
        
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