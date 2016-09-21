#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from nn import NN
import numpy as np
from DatasetNormalizer import DatasetNormalizer
from random import randint
import cPickle
import argparse

"""
Ejemplos:

Train Ej: 1
python main.py --problem 1 --mode training --input ./data/tp1_ej1_training.csv --layers 200 --lr 0.01 --epocs 3000 --save asd

Test Ej: 1
python main.py --problem 1 --mode test --input ./data/tp1_ej1_test.csv --load nn-ej1

Train Ej: 2
python main.py --problem 2 --mode training --input ./data/tp1_ej2_training.csv --layers 500 --lr 0.01 --epocs 3000 --save nn-ej2 --testProportion 0.1

Test Ej: 2
python main.py --problem 2 --mode test --input ./data/tp1_ej2_test.csv --load nn-ej2
"""

parser = argparse.ArgumentParser(description='TP 1 de Redes Neuronales')
parser.add_argument('--problem', metavar='problem', type=int, help='Problema 1 o 2. Valores esperados: 1 o 2')
parser.add_argument('--mode', metavar='mode', type=str, help='Modo en que se utilizara: "training"/"test"')
parser.add_argument('--input', type=str, help='File .csv con la informacion que sera utilizada')

# Parametros solo necesarios para el modo: test
parser.add_argument('--load', type=str, help='Ruta del dump de la red.')

# Parametros solo necesarios para el modo: train
parser.add_argument('--save', type=str, help='Ruta donde se guardara la red entrenada.')
parser.add_argument('--layers', metavar='layers', type=int, nargs='+', help='Solo requerido en modo: training. Lista de int con el tamanio de las capas intermedias de la red. Ej: "--layers 2 3" resulta en la red [_,2,3,1] para el problema 1.') # Esto se hace asi ya que al discretizar las variables generamos un numero mayor y no podemos permitir que se parametricen la entrada ni tampoco tiene sentido la salida.
parser.add_argument('--lr', metavar='lr', type=float, help='Learning Rate')
parser.add_argument('--epocs', type=int, help='Cantidad de epocs de entrenamiento.')
parser.add_argument('--testProportion', type=float, help='Aplica a Problema 2 unicamente. Si se setea este parametro entonces se divide el train set en dos partes y se mide el Error Cuadratico Medio')

args = parser.parse_args()

print args

###############################################################################
######################## ACTIVATION FUNCTIONS #################################
###############################################################################

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

###############################################################################
###################### DATASETS TRIVIALES DE PRUEBA ###########################
###############################################################################

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


###############################################################################
########################## CALCULAR PERFORMANCE ###############################
###############################################################################

def auc(positivePredictions, negativePredictions):
	repeateTimes = 100 * max(len(positivePredictions),len(negativePredictions))
	acum = 0.0
	for i in range(repeateTimes):
		positiveIndex = randint(0,len(positivePredictions) -1)
		negativeIndex = randint(0,len(negativePredictions) -1)
		acum += positivePredictions[positiveIndex] > negativePredictions[negativeIndex]
	return(acum/repeateTimes)

def calcAuc(Z, Zhat):
	# print("positivePredictions", Zhat[Z[0:10]==1])
	# print("negativePredictions", Zhat[Z[0:10]==0])
	# # print("Zhat[0:10]", Zhat[0:10])
	# print("Z[0:10]", Z[0:10])
	positivePredictions = Zhat[Z[0:100]==1]
	negativePredictions = Zhat[Z[0:100]==0]
	print("Roc area %f " % auc(positivePredictions,negativePredictions))

def rmse(Z, Zhat):
	error = Z - Zhat
	error*=error
	error = np.mean(error)
	error = np.power(error,1.0/2.0)
	return(error)

def calcRmseEj2(Z, Zhat):
	print("First Column RMSE: ",rmse(Z[:,0], Zhat[:,0, 0]))
	print("Second Column RMSE: ",rmse(Z[:,1], Zhat[:,0, 1]))

def compareResults(z, zhat):
	for j in range(10):
		print("%f - %f ====== %f - %f" %(z[j,0], zhat[j,0], z[j,1], zhat[j,1]))

###############################################################################
############################ DATASET ADMINISTRATOR ############################
###############################################################################

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

# Normaliza vectores y los centra entre 0 y 1. Devuelve los vectorres normalizados y los mínimos y máximos
def normalize(Z):
	minTrain = Z.min()
	maxTrain = Z.max()
	return ((Z - minTrain)/(maxTrain-minTrain), minTrain, maxTrain)

# Desnormaliza el vector para volverlo a los rangos iniciales
def denormalize(Z, minTrain, maxTrain):
	return Z*(maxTrain-minTrain)+minTrain

###############################################################################
################################ DUMPING ######################################
###############################################################################

def saveAs(saveIn, obj):
	print "Guardando la red en %s" % saveIn
	with open(saveIn, 'wb') as f:
		cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)

def load(loadFrom):
	print "Cargando red desde %s" % loadFrom
	with open(loadFrom, 'rb') as f:
		return cPickle.load(f)

###############################################################################
################################ TRAINING #####################################
###############################################################################

# Devuelve un par de sigmoid y su derivada por cada layer
def getSigmoidFunctions(size):
	activationFunctions = list()
	for i in range(size):
		activationFunctions.append((sigmoid, sigmoidDerivate))
	return activationFunctions

def getTrainingParamsEj1(inputLayers, filenameInput, saveIn):
	saveIn = saveIn if saveIn else './' # Si no viene definodo el parametro tomo ./ como default
	dn = DatasetNormalizer(filenameInput, 'ej1')
	X = dn.discretization(dn.data[:,1:], dn.EJ1_COLUMNS_NAME, save_path=saveIn+'-normalizer.pkl')
	Z = dn.data[:,0]
	layers = [X.shape[1]] + inputLayers + [1]
	activationFunctions = getSigmoidFunctions(len(layers))
	return (X,Z,layers,activationFunctions)

def getTrainingParamsEj2(inputLayers, filenameInput, saveIn):
	saveIn = saveIn if saveIn else './' # Si no viene definodo el parametro tomo ./ como default
	dn = DatasetNormalizer(filenameInput, 'ej2')
	X = dn.discretization(dn.data[:,0:8], dn.EJ2_COLUMNS_NAME, save_path=saveIn+'-normalizer.pkl')
	Z = dn.data[:,8:]
	layers = [X.shape[1]] + inputLayers + [2]
	activationFunctions = getSigmoidFunctions(len(layers))
	return (X,Z,layers,activationFunctions)

def trainEj1(inputLayers, lr, filenameInput, epocs, saveIn=None):
	X, Z, layers, activationFunctions = getTrainingParamsEj1(inputLayers, filenameInput, saveIn)
	nn = NN(layers, activationFunctions, lr)
	for i in range(epocs):
		e = nn.mini_batch(X,Z)
		print "Epoc %d error: %f" % (i, e)
		if (i % 50 == 0):
			Zhat = nn.predict(X[0:100,:])
			calcAuc(Z[0:100], Zhat)
	if (saveIn):
		saveAs(saveIn, nn)

def trainEj2(inputLayers, lr, filenameInput, epocs, saveIn=None, testProportion=None):
	X, Z, layers, activationFunctions = getTrainingParamsEj2(inputLayers, filenameInput, saveIn)
	tp = testProportion if testProportion else 0.0
	X_train, Z_train, X_test, Z_test = splitSet(X, Z, testProportion=tp)
	Z_train, minTrain, maxTrain = normalize(Z_train)
	nn = NN(layers, activationFunctions, lr)
	##
	acum = 0
	interval = 20
	for i in range(epocs):
		e = nn.mini_batch(X_train,Z_train)
		print "Epoc %d error: %f" % (i, e)
		if(testProportion and i%interval == 0):
			print("Train RMSE")
			Zhat = nn.predict(X_train)
			calcRmseEj2(Z_train, Zhat)
			print("Test RMSE")
			Zhat = nn.predict(X_test)
			Zhat = denormalize(Zhat, minTrain, maxTrain)
			calcRmseEj2(Z_test, Zhat)
	if (saveIn):
		saveAs(saveIn, (nn, minTrain, maxTrain))

# Lo usamos para probar distinas configuraciones
def experimentWithEj2():
	# lr = 0.0001
	# neurons = 1000
	for lr in [0.01]:
		for neurons in [10,50,100,500,2000]:
			print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>"
			print "lr: %f, neurons: %d" % (lr, neurons)
			print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>"
			trainEj2([neurons], lr, 2000, './data/tp1_ej2_training.csv')

###############################################################################
################################## TEST #######################################
###############################################################################

def testEj1(loadFrom, filenameInput):
	nn = load(loadFrom)
	dn = DatasetNormalizer(filenameInput, 'ej1', test=True)
	X  = dn.apply_discretization(dn.data, dn.EJ1_COLUMNS_NAME, loadFrom+'-normalizer.pkl')
	Zhat = nn.predict(X)
	print "Predictions:"
	for z in Zhat:
		print "%f" % z[0][0]

def testEj2(loadFrom, filenameInput):
	nn, minTrain, maxTrain = load(loadFrom)
	dn = DatasetNormalizer(filenameInput, 'ej2', test=True)
	X  = dn.apply_discretization(dn.data, dn.EJ2_COLUMNS_NAME, loadFrom+'-normalizer.pkl')
	Zhat = nn.predict(X)
	Zhat = denormalize(Zhat, minTrain, maxTrain)
	print "Predictions:"
	for z in Zhat:
		print "%f, %f" % (z[0][0],z[0][1])


if (args.mode == 'training'):
	if (args.problem == 1):
		trainEj1(args.layers, args.lr , args.input, args.epocs, saveIn=args.save)
	elif (args.problem == 2):
		trainEj2(args.layers, args.lr , args.input, args.epocs, saveIn=args.save, testProportion=args.testProportion)
elif (args.mode == 'test'):
	if (args.problem == 1):
		testEj1(args.load, args.input)
	elif (args.problem == 2):
		testEj2(args.load, args.input)

# trainEj1([200], 0.01, 5000, './data/tp1_ej1_training.csv')
# experimentWithEj2()
