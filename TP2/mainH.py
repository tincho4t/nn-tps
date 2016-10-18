#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from NN_HEBBIANO import NN_HEBBIANO
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from random import randint
import cPickle
import argparse


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
	if (saveIn):
		saveAs(saveIn, nn)
	return nn

filename = '../data/tp2_training_dataset.csv'
outputShape = 3
nn = trainEj1(outputShape, filename, 3000, 0.01, 'sanger')

