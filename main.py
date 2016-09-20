from __future__ import division
#from nnb import NNB as NN
from nn import NN
import numpy as np
from DatasetNormalizer import DatasetNormalizer
from random import randint
from sklearn import preprocessing

def sigmoid(x):
	return 1 / (1 + np.exp(-np.clip(x,-100, 100))) # Acoto los valores ya que por fuera de estos valores deberia saturar en 0 o 1

def sigmoidDerivate(x):
	return sigmoid(x) * (1 - sigmoid(x))

# Segun Wikipedia la derivada de sofPlus es sigmoid: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
def softPlus(x):
	return np.log(1 + np.exp(np.clip(x,-100, 100)))

def linear(x):
	return x

def linearDerivate(x):
	return np.ones(x.shape)

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
	#X = preprocessing.normalize(dn.data[:,0:8])
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

def calcAuc(Z, Zhat):
	print("positivePredictions", Zhat[Z[0:10]==1])
	print("negativePredictions", Zhat[Z[0:10]==0])
	# print("Zhat[0:10]", Zhat[0:10])
	# print("Z[0:10]", Z[0:10])
	positivePredictions = Zhat[Z[0:100]==1]
	negativePredictions = Zhat[Z[0:100]==0]
	print("Roc area %f " % auc(positivePredictions,negativePredictions))

def splitSet(X, Z, testProportion = 0.2):
	n = X.shape[0]
	trainIndex = list(range(n))
	np.random.shuffle(trainIndex)
	n_train = int(n*(1-testProportion))
	X_train = X[trainIndex[0:n_train], :]
	Z_train = Z[trainIndex[0:n_train]]
	X_test = X[trainIndex[n_train:], :]
	Z_test = Z[trainIndex[n_train:]]
	return(X_train,Z_train,X_test,Z_test)


def rmse(Z, Zhat):
	error = Z - Zhat
	error*=error
	error = np.mean(error)
	error = np.power(error,1.0/2.0)
	return(error)

def calcRmseEj2(Z, Zhat):
	print("First Column RMSE: ",rmse(Z[:,0], Zhat[:,0, 0]))
	print("Second Column RMSE: ",rmse(Z[:,1], Zhat[:,0, 1]))

#X, Z, layers = xorDataset()

#X, Z, layers = ej2()

# Ej 1
# nn = NN(layers, sigmoid, sigmoidDerivate, 0.01)

# Ej 2

def trainEj1():
	X, Z, layers = ej1()
	nn = NN(layers, [(sigmoid, sigmoidDerivate), (sigmoid, sigmoidDerivate), (sigmoid, sigmoidDerivate)], 0.01)
	acum = 0
	interval = 1000
	for i in range(500000):
		if(i%interval == 1):
			print(acum/interval)
			acum = 0
			# EJ 1
			Zhat = nn.predict(X[0:100,:])
			calcAuc(Z[0:100], Zhat)
		acum += nn.random_batch(X,Z)


def compareResults(z, zhat):
	for j in range(10):
		print("%f - %f ====== %f - %f" %(z[j,0], zhat[j,0], z[j,1], zhat[j,1]))

def trainEj2():
	# lr = 0.0001
	# neurons = 1000
	for lr in [0.01]:
		for neurons in [10,50,100,500,2000]:
			print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>"
			print "lr: %f, neurons: %d" % (lr, neurons)
			print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>"
			X, Z, layers = ej2(neurons)
			X_train, Z_train, X_test, Z_test = splitSet(X, Z, testProportion=0.2)
			minTrain = Z_train.min()
			maxTrain= Z_train.max()
			Z_train = (Z_train - minTrain)/(maxTrain-minTrain)
			#nn = NN(layers, [(sigmoid, sigmoidDerivate), (sigmoid, sigmoidDerivate), (softPlus, sigmoid)], lr)
			nn = NN(layers, [(sigmoid, sigmoidDerivate), (sigmoid, sigmoidDerivate), (sigmoid, sigmoidDerivate)], lr)
			##
			acum = 0
			interval = 20
			for i in range(2000):
				e = nn.mini_batch(X_train,Z_train)
				if(i%100 == 99):
					lr /= 1.1
					print("NEW LR IS: ", lr)
				#print("Epoc: %d Error: %f" %(i, e))
				if(i%interval == 0):
					#Zhat = nn.predict(X_train[0:10,:])
					#Zt = Z_train[0:10,:]
					#compareResults(Zt, Zhat[:,0,:])
					print("Train RMSE")
					Zhat = nn.predict(X_train)
					calcRmseEj2(Z_train, Zhat)
					print("Test RMSE")
					Zhat = nn.predict(X_test)
					Zhat = Zhat*(maxTrain-minTrain)+minTrain
					calcRmseEj2(Z_test, Zhat)

#trainEj1()

trainEj2()
