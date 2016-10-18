#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

###############################
##### #Back propagation #######
###############################
class NN_HEBBIANO(object):
    
    MINI_BATCH_SIZE = 15 # Cantidad de muestras que se toman por minibach
    INITIALIZATION_COEF = 0.01 # Coeficiente para disminuir la inicialización de los pesos
    
    # layers: Array con las dimensaiones de cada capa incluyendo inputs
    def __init__(self, inputSize, outputSize, lr, method):
        self.lr = lr # Learning Rate
        self.method = method
        self.N  = inputSize
        self.M  = outputSize
        self.W = np.random.rand(self.N, self.M) *  self.INITIALIZATION_COEF
        if(method == 'sanger'):
            self.U = np.triu(np.ones((self.M,self.M)))
        elif(method == 'oja_n'):
            self.U = np.ones((self.M,self.M))
        elif(method == 'oja_1'):
            self.U = np.zeros((self.M,self.M))
            np.fill_diagonal(self.U, 1)
        else:
            ValueError("Method %s no encontrado" % method)

    def activation(self, X):
        Y = np.dot(X,self.W)
        return Y

    def correction(self, X, Y):
        print X.shape
        print Y.shape
        print self.getXhat(Y).shape
        dW = self.lr * np.dot((X - self.getXhat(Y)).T, Y) # lr * (X - X^) * Y
        return dW

    def getXhat(self, Y):
        print ">>>>>>>>>>>>>>>>>>>>>>getXhat"
        print (Y.T * self.U).shape
        return np.dot(self.W, Y.T * self.U) # W * (Y^T * U)
    
    def adaptation(self, dW):
        self.W += dW
    
    def predict(self, X):
        return np.dot(X,self.W) # TODO: Validar que den el mismo resultado
        Y = list()
        for x in X:
            Y.append(self.activation(x))
        return Y

    def mini_batch(self, X):
        P = X.shape[0] # Cantidad de Instancias
        # Mezclo los indices
        random_index = list(range(P))
        np.random.shuffle(random_index)
        
        i = 0
        x = np.zeros((1,self.N))
        while i < P:
            index = random_index[i]
            dW = np.zeros_like(self.W)
            for j in range(self.MINI_BATCH_SIZE):
                if i < P: # Valido de no pasarme del limite
                    x[:] = X[index].T
                    Y = self.activation(x)
                    dW += self.correction(x,Y)
                i += 1
            self.adaptation(dW)

    # Definimos error como la suma de los productos internos dos a dos de W
    def getError(self):
        error = 0.0
        for i in range(self.M-1):
            v = self.W[:,i]
            w = self.W[:,i+1:]
            error += np.sum(np.dot(v.T,w))
        return error
