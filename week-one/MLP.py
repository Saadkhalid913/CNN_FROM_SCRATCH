import numpy as np
from sklearn.datasets import make_blobs
import os 
import json 
import pickle
import math
import itertools

class NN():
    def __init__(self, x, y):
        self.W1 = np.random.rand((x.shape[1], 4))
        self.W2 = np.random.rand((6, y.shape[1]))

    def Sigmoid(self, x, derivative = False):
        if derivative:
            return self.Sigmoid(x) * (1 - self.Sigmoid(x))
        return 1 / 1 + np.exp(-x)

    def forward(self, x):
        out = np.dot(x, self.W1)
        out = self.Sigmoid(out)
        out = np.dot(out, self.W2)

def CtoF(x):
    return (x * 1.8) + 32

print(np.random.randint((1, 150)))