import numpy as np 

def sigmoid(x, derivative = False):
    if derivative:
      return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))

def Relu(x, derivative = False):
    if derivative:
      return np.where(x > 1, 1, -1)
    return np.where(x > 1, x, -x)



__all__ = ["sigmoid", "Relu"]