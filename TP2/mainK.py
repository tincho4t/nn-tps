#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from NN_KOHONEM import NN_KOHONEM
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from random import randint
import cPickle
import argparse
import sklearn.metrics as metrics

# parser = argparse.ArgumentParser(description='TP 1 de Redes Neuronales')
# parser.add_argument('--problem', metavar='problem', type=int, help='Problema 1 o 2. Valores esperados: 1 o 2')
# parser.add_argument('--mode', metavar='mode', type=str, help='Modo en que se utilizara: "training"/"test"')
# parser.add_argument('--input', type=str, help='File .csv con la informacion que sera utilizada')

# # Parametros solo necesarios para el modo: test
# parser.add_argument('--load', type=str, help='Ruta del dump de la red.')

# # Parametros solo necesarios para el modo: train
# parser.add_argument('--save', type=str, help='Ruta donde se guardara la red entrenada.')
# parser.add_argument('--layers', metavar='layers', type=int, nargs='+', help='Solo requerido en modo: training. Lista de int con el tamanio de las capas intermedias de la red. Ej: "--layers 2 3" resulta en la red [_,2,3,1] para el problema 1.') # Esto se hace asi ya que al discretizar las variables generamos un numero mayor y no podemos permitir que se parametricen la entrada ni tampoco tiene sentido la salida.
# parser.add_argument('--lr', metavar='lr', type=float, help='Learning Rate')
# parser.add_argument('--epocs', type=int, help='Cantidad de epocs de entrenamiento.')
# parser.add_argument('--testProportion', type=float, help='Aplica a Problema 2 unicamente. Si se setea este parametro entonces se divide el train set en dos partes y se mide el Error Cuadratico Medio')

# args = parser.parse_args()

# print args

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

def getDataset(filename):
	csv_data = genfromtxt(filename, delimiter=',')
	return csv_data[:,0], csv_data[:,1:]

def getTrainingParamsEj1(filenameInput):
	Z, X = getDataset(filenameInput)
	return (X,Z,X.shape[1])

def getTrainingParamsEj2(inputLayers, filenameInput, saveIn):
	saveIn = saveIn if saveIn else './' # Si no viene definodo el parametro tomo ./ como default
	# dn = DatasetNormalizer(filenameInput, 'ej2')
	X = dn.discretization(dn.data[:,0:8], dn.EJ2_COLUMNS_NAME, save_path=saveIn+'-normalizer.pkl')
	Z = dn.data[:,8:]
	layers = [X.shape[1]] + inputLayers + [2]
	activationFunctions = getSigmoidFunctions(len(layers))
	return (X,Z,layers,activationFunctions)

def showMatrix(xPositions, matrixDimension):
	M = np.zeros(matrixDimension)
	for i in xPositions:
		M[i] += 1
	print M
	# plt.imshow(M/np.max(M), cmap='hot', interpolation='nearest')
	# plt.show()

def trainEj1(matrixDimension, filenameInput, epocs, lr, s0, sr, saveIn=None):
	X, Z, inputShape = getTrainingParamsEj1(filenameInput)
	
	nn = NN_KOHONEM(inputShape, matrixDimension, lr, s0, sr)
	for i in range(epocs):
		nn.mini_batch(X)
		print "Epoc %d" % i
		if (i % 25 == 0):
			xPositions = nn.predict(X)
			showMatrix(xPositions, matrixDimension)
	if (saveIn):
		saveAs(saveIn, nn)
	return nn


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


# if (args.mode == 'training'):
# 	if (args.problem == 1):
# 		trainEj1(args.layers, args.lr , args.input, args.epocs, saveIn=args.save)
# 	elif (args.problem == 2):
# 		trainEj2(args.layers, args.lr , args.input, args.epocs, saveIn=args.save, testProportion=args.testProportion)
# elif (args.mode == 'test'):
# 	if (args.problem == 1):
# 		testEj1(args.load, args.input)
# 	elif (args.problem == 2):
# 		testEj2(args.load, args.input)

# trainEj1([200], 0.01, 5000, './data/tp1_ej1_training.csv')
# experimentWithEj2()

def getMatrixForClass(matrixDimension, xPositions, category):
	M = np.zeros(matrixDimension)
	for i in range(len(xPositions)):
		p = xPositions[i]
		if((category == None ) or (category == Z[i])):
			M[p] += 1
	return M

def heatMap(nn, filename, matrixDimension, category = None):
	X, Z, _ = getTrainingParamsEj1(filename)
	xPositions = nn.predict(X)
	M = getMatrixForClass(matrixDimension, xPositions, category)
	print M
	plt.imshow(M/np.max(M), cmap='hot', interpolation='nearest')
	plt.show()

def createEvaluateVari2ables(matrixDimension, filenameInput, lr, s0, sr, saveIn=None):
	X, Z, inputShape = getTrainingParamsEj1(filenameInput)
	nn = NN_KOHONEM(inputShape, matrixDimension, lr, s0, sr)
	return nn,X,Z

# def evaluate(nn, X,Z):
# 	xPositions = nn.predict(X)
# 	return metrics.silhouette_score(xPositions, Z)

def trainAndEval(nn, X, Z, epocs, initialEpoc = 0):
	for i in range(epocs):
		nn.mini_batch(X)
	return evaluate(nn, X,Z)

############ Metrica Distancia a Centroides ############
import math

# Devuelve el indice del mayor indice
def getMaxIndex(M):
	maxIndex = (0,0)
	for i in range(M.shape[0]):
		for j in range(M.shape[1]):
			if(M[maxIndex] < M[(i,j)]):
				maxIndex = (i,j)
	return maxIndex

def calculateCentroid(xPositions, z, matrixDimension):
	M = getMatrixForClass(matrixDimension, xPositions, z)
	return getMaxIndex(M)

def calculateCentroids(xPositions, Z, matrixDimension):
	centroids = {}
	for z in Z:
		centroids[z] = calculateCentroid(xPositions, z, matrixDimension)
	return centroids

def distanceBetween(x,y):
	return math.hypot(x[0] - y[0], x[1] - y[1])

def getNearestCentroids(p,centroids):
	distancesToC = {}
	for z in centroids:
		c = centroids[z]
		distancesToC[z] = distanceBetween(c,p)
	cFirst = min(distancesToC, key=distancesToC.get)
	del distancesToC[cFirst] # Saco el maximo y busco el siguiente
	cSecond = min(distancesToC, key=distancesToC.get)
	return centroids[cFirst],centroids[cSecond]

# Return un mapa Z, ([Distancia al Centroide],[Distancia al proximo centroide mas cercano])
# solamente con las instancias que su centroide mas cercano es el correcto
def getDistancesOfCentroids(xPositions, centroids, Z):
	distances = {}
	for z in np.unique(Z):
		distances[z] = ([],[])
	for i in range(len(xPositions)):
		p = xPositions[i]
		z = Z[i]
		cFirst, cSecond = getNearestCentroids(p,centroids)
		if(cFirst == centroids[z]): # Si el punto tiene como centroide mas cercano el suyo
			distances[z][0].append(distanceBetween(p,cFirst)) # Agrego la distancia a su centroide
			distances[z][1].append(distanceBetween(p,cSecond)) # Agrego la distancia al proximo
	return distances

def sumInstancesOf(Z,z):
	return np.sum(Z == z)

# Mira la diferencia de distancia entre las intancias
def getError(nn, xPositions, Z, centroids):
	distances = getDistancesOfCentroids(xPositions, centroids, Z)
	errors = []
	for z in np.unique(Z):
		okClasificated = len(distances[z][0])
		total = sumInstancesOf(Z,z)
		print "Centroide %d -> Total %f --- ok: %d total: %d Diff: %f" % (z, (okClasificated/total) - (np.sum(distances[z][0])/ np.sum(distances[z][1])), okClasificated, total, (np.sum(distances[z][0])/ np.sum(distances[z][1])))
		error = (okClasificated/total) - (np.sum(distances[z][0])/ np.sum(distances[z][1]))
		errors.append(error)
	return errors

def getScore(nn, xPositions, Z, centroids):
	distances = getDistancesOfCentroids(xPositions, centroids, Z)
	scores = []
	acu = []
	d = []
	for z in np.unique(Z):
		okClasificated = len(distances[z][0])
		total = sumInstancesOf(Z,z)
		##########
		#Lo hago asi porque ya tengo codeadas las funciones
		_, nearestCentroid = getNearestCentroids(centroids[z],centroids) # Como el 1ro voy a ser yo mismo tomo el 2do
		print centroids[z], nearestCentroid, distanceBetween(centroids[z], nearestCentroid)
		distanceToNearestCentroid = distanceBetween(centroids[z], nearestCentroid) / math.hypot(nn.M1,nn.M2)
		#########
		score = (okClasificated/total) + distanceToNearestCentroid
		# print "Centroide %d -> Total %f --- Acu: %f(%d/%d) Distance: %f" % (z, score, okClasificated/total,okClasificated, total, distanceToNearestCentroid)
		scores.append(score)
		acu.append(okClasificated/total)
		d.append(distanceToNearestCentroid)
	zAmount = len(np.unique(Z))
	return (np.average(scores), np.average(acu), np.average(d))

def evaluate(nn, X, Z):
	xPositions = nn.predict(X)
	centroids = calculateCentroids(xPositions, Z, (nn.M1,nn.M2))
	errors = getError(nn, xPositions, Z, centroids)
	# return np.sum(errors)
	return getScore(nn, xPositions, Z, centroids), np.sum(errors)

#################################################################

filename = '../data/tp2_training_dataset.csv'
matrixDimension = (10,10)

nn,X,Z = createEvaluateVari2ables(matrixDimension, filename, 0.0001, 5, 20)

acum = 0
# for epocs in [50,50]:
# for i in range(100):
for i in range(200):
	epocs = 1
	score = trainAndEval(nn, X, Z, epocs, acum)
	acum += epocs
	print "Epocs: %d Score %f Acu: %f Distance: %f Error %f " %(acum, score[0][0],score[0][1],score[0][2],score[1])

print "(%d,%d) Epocs: %d LR: %f, s0: %f sr: %f" %(matrixDimension[0],matrixDimension[1],acum, nn.lr,nn.sigma0,nn.sigmar)
