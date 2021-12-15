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

y_truth = np.zeros((n_samples,2))
for i in range(n_samples):
  y_truth[i][y_train[i]] = 1

## num neurons in each hidden layer 
HL1 = 20 
HL2 = 12

class NN():
  def __init__(self):
    self.W1 = np.random.randn(2, HL1)
    self.B1 = np.random.randn(1, HL1)
    self.W2 = np.random.randn(HL1, HL2)
    self.B2 = np.random.randn(1, HL2)
    self.W3 = np.random.randn(HL2, 2)
    self.B3 = np.random.randn(1, 2)
    



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
    Z3 = np.dot(A2, self.W3) + self.B3 
    A3 = self.sigmoid(Z3)

    return Z1, A1, Z2, A2, Z3, A3

  def predict(self, x):
    Z1 = np.dot(x, self.W1) + self.B1 # unactivated L1 values 
    A1 = self.sigmoid(Z1) # activated L1 values 
    Z2 = np.dot(A1, self.W2) + self.B2 # unactivated L2 values 
    A2 = self.sigmoid(Z2) # activated L1 values 
    Z3 = np.dot(A2, self.W3) + self.B3 
    A3 = self.sigmoid(Z3)

    return A3


  ## this is the derivative of the loss with respect to y_pred 
  ## this will return an array of shape (num_samples, num_output_neurons)
  def MSELoss(self, y_pred, y_truth, derivative = False):
    if derivative:
      return -1 * (y_truth - y_pred)
    return np.power((y_truth - y_pred), 2) / 2

  def Backprop(self, n_iters=100):

    for i in range(n_iters):
      Z1, A1, Z2, A2, Z3, A3  = self.forward(x_train)
      
      # copying weights 
      oldW3 = self.W3.copy()
      oldW2 = self.W2.copy()

      ## finding derivative of loss with respect to acivated layer 2 outputs 
      LossDerivativeWRT_Sigmoid = self.MSELoss(A3, y_truth, derivative = True)
      SigmoidDerivativeWRT_L2 = self.sigmoid(Z3, derivative = True)

      ## gradient of layer 2 
      Delta3 = LossDerivativeWRT_Sigmoid * SigmoidDerivativeWRT_L2

      ## we transpose the output of the second layers and 
      ## update weights with the dot product of the 
      ## transpose and the gradient 
      Update_W3 = np.dot(A2.T, Delta3)
      Update_B3 = np.sum(Delta3, axis = 0, keepdims=True)

      self.W3 += -lr * Update_W3
      self.B3 += -lr * Update_B3

      ## calculating gradient WRT L1 (Sigmoid outputs)
      Delta2 = np.dot(Delta3, oldW3.T)

      ## we use chain rule and multiply above gradient
      ## to get gradient with respect to Z2 (unactivated)

      GradientWRT_Z2 = Delta2 * self.sigmoid(Z2, derivative = True)

      ## the update for the bias is simply going to be the sum of 
      ## this term along the 0th axis 

      Update_B2 = np.sum(GradientWRT_Z2, axis = 0, keepdims = True) 
      Update_W2 = np.dot(A1.T, GradientWRT_Z2)

      self.W2 += -lr * Update_W2
      self.B2 += -lr * Update_B2

      ## we calculate the gradient of the old weights 
      ## of L2
      Delta1 = np.dot(Delta2, oldW2.T)
      GradientWRT_Z1 = Delta1 * self.sigmoid(Z1, derivative = True)

      Update_B1 = np.sum(GradientWRT_Z1, axis = 0, keepdims=True)

      ## we calculate the gradient with respect to the input layer, 
      ## we must transpose it just the same as the other neuron 
      ## layers in order to get dimentions to match 

      Update_W1 = np.dot(x_train.T, GradientWRT_Z1)

      self.W1 += -lr * Update_W1
      self.B1 += -lr * Update_B1   






      


n = NN()
n.Backprop()
pred = np.argmax(n.predict(x_test), axis = 1)
print(accuracy_score(pred, y_test))