from __future__ import division
#from nnb import NNB as NN
from nn import NN
import numpy as np
from DatasetNormalizer import DatasetNormalizer
from random import randint

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
	dn = DatasetNormalizer('./data/tp1_ej1_training.csv', 'ej1')
	X = dn.discretization(dn.data[:,1:], dn.EJ1_COLUMNS_NAME, save_path='./ej1.pkl')
	# X = dn.data[:,1:]
	Z = dn.data[:,0]
	# layers = list([X.shape[1],10,10,1])
	layers = list([X.shape[1],200,1])
	return (X,Z,layers)

def ej2(hiddenLayer=1000):
	dn = DatasetNormalizer('./data/tp1_ej2_training.csv', 'ej2')
	X = dn.discretization(dn.data[:,0:8], dn.EJ2_COLUMNS_NAME, save_path='./ej2.pkl')
	# X = dn.data[:,1:]
	Z = dn.data[:,8:]
	layers = list([X.shape[1],hiddenLayer,2])
	#layers = list([X.shape[1],200,1])
	return (X,Z,layers)

def auc(positivePredictions, negativePredictions):
	repeateTimes = 100 * max(len(positivePredictions),len(negativePredictions))
	acum = 0.0
	for i in range(repeateTimes):
		positiveIndex = randint(0,len(positivePredictions) -1)
		negativeIndex = randint(0,len(negativePredictions) -1)
		acum += positivePredictions[positiveIndex] > negativePredictions[negativeIndex]
	return(acum/repeateTimes)

def calcMetrics(Z, Zhat):
	print("positivePredictions", Zhat[Z[0:10]==1])
	print("negativePredictions", Zhat[Z[0:10]==0])
	# print("Zhat[0:10]", Zhat[0:10])
	# print("Z[0:10]", Z[0:10])
	positivePredictions = Zhat[Z[0:100]==1]
	negativePredictions = Zhat[Z[0:100]==0]
	print("Roc area %f " % auc(positivePredictions,negativePredictions))

#X, Z, layers = xorDataset()

X, Z, layers = ej2()

# Ej 1
# nn = NN(layers, sigmoid, sigmoidDerivate, 0.01)

# Ej 2

def trainEj1():
	X, Z, layers = ej1()
	nn = NN(layers, sigmoid, sigmoidDerivate, 0.01)
	acum = 0
	interval = 1000
	for i in range(500000):
		if(i%interval == 1):
			print(acum/interval)
			acum = 0
			# EJ 1
			Zhat = nn.predict(X[0:100,:])
			calcMetrics(Z[0:100], Zhat)
		acum += nn.random_batch(X,Z)


def compareResults(z, zhat):
	for j in range(10):
		print("%f - %f ====== %f - %f" %(z[j,0], zhat[j,0], z[j,1], zhat[j,1]))

def trainEj2():
	for lr in [0.1,0.01,0.001, 0.001, 0.0001]:
		for neurons in [400, 1000, 5000]:
			print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>"
			print "lr: %f, neurons: %d" % (lr, neurons)
			print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>"
			X, Z, layers = ej2(neurons)
			nn = NN(layers, sigmoid, sigmoidDerivate, lr)
			##
			acum = 0
			interval = 1000
			for i in range(100000):
				if(i%interval == 1):
					print(acum/interval)
					acum = 0
					Zhat = nn.predict(X[0:10,:])
					Zt = Z[0:10,:]
					compareResults(Zt, Zhat[:,0,:])
				acum += nn.random_batch(X,Z)

#trainEj1()

trainEj2()
