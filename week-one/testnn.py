# -*- coding: utf-8 -*-
"""testNN

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12jgXpF6e_SIpUNE0WILdl4Txwn4k8QLx
"""

'''
  This is the second iteration of the neural 
  network, generalized to have 3 layers. 
'''

import numpy as np
from sklearn.datasets import make_moons 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

n_features = 2 
n_samples = 20000
n_classes = 2
lr = 0.001

x,y = make_moons(n_samples)
x_train, x_test, y_train, y_test = train_test_split(x,y)

n_samples = x_train.shape[0]

y_truth = np.zeros((n_samples,2))
for i in range(n_samples):
  y_truth[i][y_train[i]] = 1

class NN():
  def __init__(self):
    hidden_layer_neurons = 20 

    # we initialize a set of random weights and biases 
    self.W1 = np.random.randn(n_features, hidden_layer_neurons )
    self.B1 = np.random.randn(1, hidden_layer_neurons ) # bias must be of shape (1, num_layer_nodes)
    self.W2 = np.random.randn(hidden_layer_neurons , n_classes)
    self.B2 = np.random.randn(1, n_classes) # bias must be of shape (1, num_layer_nodes)


  # sigmoid function as well as
  # sigmoid derivative with respect to x 
  def sigmoid(self, x, derivative = False):
    if derivative:
      return self.sigmoid(x) * (1 - self.sigmoid(x))
    return 1 / (1 + np.exp(-x))


  def forward(self, x):
    Z1 = np.dot(x, self.W1) + self.B1 # unactivated L1 values 
    A1 = self.sigmoid(Z1) # activated L1 values 
    Z2 = np.dot(A1, self.W2) + self.B2 # unactivated L2 values 
    A2 = self.sigmoid(Z2) # activated L1 values 

    return Z1, A1, Z2, A2

  def predict(self, x):
    Z1 = np.dot(x, self.W1) + self.B1
    A1 = self.sigmoid(Z1)
    Z2 = np.dot(A1, self.W2) + self.B2
    A2 = self.sigmoid(Z2)
    # sigmoid_output = self.sigmoid(A2)

    return A2


  ## this is the derivative of the loss with respect to y_pred 
  ## this will return an array of shape (num_samples, num_output_neurons)
  def MSELoss(self, y_pred, y_truth, derivative = False):
    if derivative:
      return -1 * (y_truth - y_pred)
    return np.power((y_truth - y_pred), 2) / 2

  def Backprop(self, n_iters=100):
    for i in range(n_iters):
      Z1, A1, Z2, A2 = self.forward(x_train)
      oldW2 = self.W2.copy() # we copy these weights for later (10, 2)
      DLoss = self.MSELoss(y_truth, A2,derivative = True) # Dloss/Dsigmoid output (num_samples, 2)
      DSigmoid = self.sigmoid(Z2, derivative = True) #Dsigmoid / Dneuron_out_put_sum (num_samples, 2)
      DWeight = A1 # Dneuron_sum/Dweights (num_samples, 10)

      delta1 = DLoss * DSigmoid # (num_samples, 2)

      UpdateW2 = np.dot(DWeight.T, delta1)          # we transpose the weights to perform
                                                     # dot product on the element-wise product 
                                                    # of the loss & the sigmoid derivative
                                                    # this is done because we need to take dot product 
                                                    # of each sample along with the delta (loss * sigmoid)
                                                    # that it creates for each neuron. 
                                                    # this is then summed up (as part of dot product)
                                                    # and becomes the jth value in the ith row of the 
                                                    # delta matrix (4x2) in this case or (hidden_nodes * output_nodes)
                                                    # this is then done for each neuron and its respective weights, resulting
                                                    # in a (4x2) matrix which is the update


      # to update the biases for the final layer, we simply
      # take the sum of the delta on the first axis 
      # which is a (1,2) shaped vector 
      UpdateB2 = np.sum(delta1, axis = 0, keepdims=True) / n_samples


      # updating the weights 
      self.W2 += -1 * lr * UpdateW2
      self.B2 += -1 * lr * UpdateB2

      # matrix shapes 
      # (num_samples, 2) (10, 2).T = (num_samples, 10)
      # we take the dot product of the delta
      # with the old weights, this is distributing the
      # gradient with respect to each old weight in the 
      # hidden layer 
      delta2 = np.dot(delta1, oldW2.T) 


      ## gradient with respect to W1

      # we then multiply the above gradient with the 
      # derivative of the gradient with respect to the
      # unactivated sigmoid input (Z1)
      GradientWRT_Z1 = delta2 * self.sigmoid(Z1, derivative = True)

      # the sum of this gradient with respect to the unactivated inputs
      # is the bias 
      UpdateB1 = np.sum(GradientWRT_Z1, axis = 0, keepdims=True) 

      # we calculate the loss of each neuron in the 
      # input layer (this is just the training data)
      # and that is the update for W1
      UpdateW1 = np.dot(x_train.T, GradientWRT_Z1)

      self.W1 += -1 * lr * UpdateW1
      self.B1 += -1 * lr * UpdateB1


n = NN()
n.Backprop()
pred = np.argmax(n.predict(x_test), axis = 1)
print(accuracy_score(pred, y_test))