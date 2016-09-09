from __future__ import division
from nn import NN
import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoidDerivate(x):
	return sigmoid(x) * (1 - sigmoid(x))

layers = list([2,3,1])

nn = NN(layers, sigmoid, sigmoidDerivate, 0.7)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
Z = np.array([0,0,0,1])

for i in range(5000):
	print(nn.f_batch(X,Z))
