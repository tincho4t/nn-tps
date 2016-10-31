#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from NN_KOHONEM import NN_KOHONEM
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import cPickle
import argparse
import math
# from random import randint

parser = argparse.ArgumentParser(description='TP 2 de Redes Neuronales Kohonem')
parser.add_argument('--mode', metavar='mode', type=str, help='Modo en que se utilizara: "training"/"test"')
parser.add_argument('--input', type=str, help='File .csv con la informacion que sera utilizada')

# Parametros solo necesarios para el modo: test
parser.add_argument('--load', type=str, help='Ruta del dump de la red.')

# Parametros solo necesarios para el modo: train
parser.add_argument('--save', type=str, help='Ruta donde se guardara la red entrenada.')
parser.add_argument('--lr', metavar='lr', type=float, help='Learning Rate')
parser.add_argument('--s0', metavar='s0', type=float, help='Sigma inicial')
parser.add_argument('--sr', metavar='sr', type=float, help='Sigma r')
parser.add_argument('--epocs', type=int, help='Cantidad de epocs de entrenamiento.')
parser.add_argument('--m1', type=int, help='Cantidad de filas de la matriz de salida.')
parser.add_argument('--m2', type=int, help='Cantidad de columnas de la matriz de salida.')
parser.add_argument('--testProportion', type=float, help='Aplica a Problema 2 unicamente. Si se setea este parametro entonces se divide el train set en dos partes y se mide el Error Cuadratico Medio')

args = parser.parse_args()

# print args

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

def showMatrix(xPositions, matrixDimension):
	M = np.zeros(matrixDimension)
	for i in xPositions:
		M[i] += 1
	print M
	# plt.imshow(M/np.max(M), cmap='hot', interpolation='nearest')
	# plt.show()

def train(matrixDimension, filenameInput, epocs, lr, s0, sr, saveIn=None):
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


###############################################################################
################################## TEST #######################################
###############################################################################

def test(loadFrom, filenameInput):
	nn = load(loadFrom)
	X  = genfromtxt(filename, delimiter=',')
	Zhat = nn.predict(X)
	print "Predictions:"
	for z in Zhat:
		print "%f" % z[0][0]

def getMatrixForClass(matrixDimension, xPositions, category, Z):
	M = np.zeros(matrixDimension)
	for i in range(len(xPositions)):
		p = xPositions[i]
		if((category == None ) or (category == Z[i])):
			M[p] += 1
	return M

def heatMap(nn, filename, matrixDimension, category = None):
	X, Z, _ = getTrainingParamsEj1(filename)
	xPositions = nn.predict(X)
	M = getMatrixForClass(matrixDimension, xPositions, category, Z)
	print M
	plt.imshow(M/np.max(M), cmap='hot', interpolation='nearest')
	plt.show()

def createEvaluateVariables(matrixDimension, filenameInput, lr, s0, sr, tp=0.2):
	X, Z, inputShape = getTrainingParamsEj1(filenameInput)
	nn = NN_KOHONEM(inputShape, matrixDimension, lr, s0, sr)
	X_train, Z_train, X_test, Z_test = splitSet(X, Z, testProportion=tp)
	return nn,X_train, Z_train, X_test, Z_test

def trainAndEval(nn, X_train, Z_train, X_test, Z_test, epocs, initialEpoc = 0):
	for i in range(epocs):
		nn.mini_batch(X_train)
	return evaluate(nn, X_train, Z_train, X_test, Z_test)

############ Metrica Distancia a Centroides ############

# Devuelve el indice del mayor indice
def getMaxIndex(M):
	maxIndex = (0,0)
	for i in range(M.shape[0]):
		for j in range(M.shape[1]):
			if(M[maxIndex] < M[(i,j)]):
				maxIndex = (i,j)
	return maxIndex

def calculateCentroid(xPositions, z, matrixDimension, Z):
	M = getMatrixForClass(matrixDimension, xPositions, z, Z)
	return getMaxIndex(M)

def calculateCentroids(xPositions, Z, matrixDimension):
	centroids = {}
	for z in Z:
		centroids[z] = calculateCentroid(xPositions, z, matrixDimension, Z)
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
	return cFirst,cSecond

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
		if(cFirst == z): # Si el punto tiene como centroide mas cercano el suyo
			distances[z][0].append(distanceBetween(p,centroids[cFirst])) # Agrego la distancia a su centroide
			distances[z][1].append(distanceBetween(p,centroids[cSecond])) # Agrego la distancia al proximo
	return distances

def sumInstancesOf(Z,z):
	return np.sum(Z == z)

def getAcu(distances, z, Z):
	okClasificated = len(distances[z][0])
	return okClasificated

def getDistanceToNearestCentroid(nn, centroids, z):
	#Lo hago asi porque ya tengo codeadas las funciones
	_, nearestCentroid = getNearestCentroids(centroids[z],centroids) # Como el 1ro voy a ser yo mismo tomo el 2do
	return distanceBetween(centroids[z], centroids[nearestCentroid]) / math.hypot(nn.M1,nn.M2)

def getScore(nn, x_trainPositions, Z_train, x_testPositions, Z_test, centroids):
	distances_train = getDistancesOfCentroids(x_trainPositions, centroids, Z_train)
	distances_test  = getDistancesOfCentroids(x_testPositions, centroids, Z_test)
	acus_train = []
	acus_test = []
	d = []
	for z in np.unique(Z_train):
		acu_train = getAcu(distances_train, z, Z_train)
		acu_test  = getAcu(distances_test, z, Z_test)
		distanceToNearestCentroid = getDistanceToNearestCentroid(nn, centroids, z)
		# print "Centroide %d -> Total %f --- Acu: %f(%d/%d) Distance: %f" % (z, score, okClasificated/total,okClasificated, total, distanceToNearestCentroid)
		acus_train.append(acu_train)
		acus_test.append(acu_test)
		d.append(distanceToNearestCentroid)
	return (np.sum(acus_train)/len(Z_train), np.sum(acus_test)/len(Z_test), np.average(d))

def evaluate(nn, X_train, Z_train, X_test, Z_test):
	x_trainPositions = nn.predict(X_train)
	x_testPositions = nn.predict(X_test)
	centroids = calculateCentroids(x_trainPositions, Z_train, (nn.M1,nn.M2))
	print centroids
	return getScore(nn, x_trainPositions, Z_train, x_testPositions, Z_test, centroids)

#################################################################


if (args.mode == 'training'):
	matrixDimension = (args.M1,args.M2)
	train(matrixDimension, args.input, args.epocs, args.lr, args.s0, args.sr, saveIn=args.save)
elif (args.mode == 'test'):
	test(args.load, args.input)


filename = '../data/tp2_training_dataset.csv'
matrixDimension = (10,10)

nn,X_train, Z_train, X_test, Z_test = createEvaluateVariables(matrixDimension, filename, 0.01, 1, 10, tp=0.2)

acum = 0
# for epocs in [50,50]:
# for i in range(100):
for i in range(35):
	epocs = 1
	acu_train, acu_test, distances = trainAndEval(nn, X_train, Z_train, X_test, Z_test, epocs, acum)
	acum += epocs
	print "Epocs: %d Score %f Acu Train: %f Acu Test: %f Distances %f " %(acum, acu_test + distances,acu_train, acu_test, distances)

print "(%d,%d) Epocs: %d LR: %f, s0: %f sr: %f" %(matrixDimension[0],matrixDimension[1],acum, nn.lr,nn.sigma0,nn.sigmar)

#heatMap(nn, filename, matrixDimension, category = None)