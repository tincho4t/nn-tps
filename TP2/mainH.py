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
		print "Epoc %d, error: %f" % (i, nn.getError())
		# Y = nn.activation(X[0])
		# print Y
	if (saveIn):
		saveAs(saveIn, nn)
	return nn

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

drawScatterPlot(dp)

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
nn = trainEj1(outputShape, filename, 3000, 0.001, 'sanger')

