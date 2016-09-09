from __future__ import division
#from nnb import NNB as NN
from nn import NN
import numpy as np
from DatasetNormalizer import DatasetNormalizer

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoidDerivate(x):
	return sigmoid(x) * (1 - sigmoid(x))

def andDataset():
	X = np.array([[0,0],[0,1],[1,0],[1,1]])
	Z = np.array([0,0,0,1])
	return (X,Z)

def orDataset():
	X = np.array([[0,0],[0,1],[1,0],[1,1]])
	Z = np.array([0,1,1,1])
	return (X,Z)

def xorDataset():
	X = np.array([[0,0],[0,1],[1,0],[1,1]])
	Z = np.array([0,1,1,0])
	layers = list([2,3,1])
	return (X,Z,layers)

def ej1():
	dn = DatasetNormalizer('./data/tp1_ej1_training.csv')
	X = dn.discretization(dn.data[:,1:], dn.EJ1_COLUMNS_NAME, save_path='./ej1.pkl')
	# X = dn.data[:,1:]
	Z = dn.data[:,0]
	layers = list([X.shape[1],5,1])
	return (X,Z,layers)


X, Z, layers = xorDataset()
X, Z, layers = ej1()

nn = NN(layers, sigmoid, sigmoidDerivate, 0.001)

acum = 0
for i in range(500000):
	if(i%100 == 0):
		print(acum/100.0)
		acum = 0
	#print(nn.f_batch(X,Z))
	acum += nn.random_batch(X,Z)

