#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from NN_HEBBIANO import NN_HEBBIANO
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import genfromtxt
from random import randint
import cPickle
import argparse
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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

def trainEj1(outputShape, filenameInput, epocs, lr, method, saveIn=None):
	X, Z, inputShape = getTrainingParamsEj1(filenameInput)
	
	nn = NN_HEBBIANO(inputShape, outputShape, lr, method)
	for i in range(epocs):
		nn.mini_batch(X)
		print "Epoc %d, error: %f, silhouette score %f" % (i, nn.getError(), nn.getSilhouette(X, Z))
		# Y = nn.activation(X[0])
		# print Y
	if (saveIn):
		saveAs(saveIn, nn)
	return nn

def validationScorePerEpoc(filenameInput, lr, epocs, method):
	X, Z, inputShape = getTrainingParamsEj1(filenameInput)
	index = np.arange(len(Z))
	np.random.shuffle(index)
	train_to = int(0.7*len(index))
	train_index = index[0:train_to]
	test_index = index[train_to:]
	X_train = X[train_index,:]
	X_test = X[test_index,:]
	Z_test = Z[test_index]
	nn = NN_HEBBIANO(inputShape, 3, lr, method)
	for i in range(epocs):
		nn.mini_batch(X_train)
		print "Epoc %d, error: %f, silhouette score %f" % (i, nn.getError(), nn.getSilhouette(X_test, Z_test))

def keepTraining(nn, epocs):
	for i in range(epocs):
		nn.mini_batch(X)
		print "Epoc %d, error: %f" % (i, nn.getError())
	return nn

##################################### DRAWING ###############################
import matplotlib.cm as cm

def getColor(c):
	norm = matplotlib.colors.Normalize(vmin=1, vmax=9)
	cmap = cm.hot
	m = cm.ScalarMappable(norm=norm, cmap=cmap)
	return m.to_rgba(c)

def addPoints(ax,xs,ys,zs,c):
    xs = xs[0]
    ys = ys[0]
    zs = zs[0]
    print c
    for i in range(len(xs)):
        ax.scatter(xs[i], ys[i], zs[i], c=getColor(c), marker='o')
    return ax


def drawScatterPlot(matrix):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d')
    classes = np.unique(matrix[:,3])
    for c in classes:
        class_index = np.where(matrix[:,3]==c)
        xs = matrix[class_index,0]
        ys = matrix[class_index,1]
        zs = matrix[class_index,2]
        addPoints(ax,xs,ys,zs,c)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def buildPlotDataset(nn, filename):
	X, Z, inputShape = getTrainingParamsEj1(filename)
	Y = nn.predict(X)
	pd = list()
	for i in range(Y.shape[0]):
		y = Y[i]
		pd.append([y[0],y[1],y[2],Z[i]])
	return np.array(pd)

###############################################################################

filename = '../data/tp2_training_dataset.csv'
outputShape = 3
nn = trainEj1(outputShape, filename, 10, 0.001, 'sanger')

for lr in [0.01,0.001,0.0001]:
	print "DOING LR",lr
	validationScorePerEpoc(filename, lr, 100, 'sanger')