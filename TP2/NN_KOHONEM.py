#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

###############################
##### #Back propagation #######
###############################
class NN_KOHONEM(object):
    
    MINI_BATCH_SIZE = 15 # Cantidad de muestras que se toman por minibach
    INITIALIZATION_COEF = 0.01 # Coeficiente para disminuir la inicialización de los pesos
    
    # layers: Array con las dimensaiones de cada capa incluyendo inputs
    def __init__(self, inputSize, matrixDimensions, lr, sigma0, sigmar):
        self.lr = lr # Learning Rate
        self.sigma0 = sigma0
        self.sigmar = sigmar
        self.M1 = matrixDimensions[0]
        self.M2 = matrixDimensions[1]
        self.M  = self.M1 * self.M2
        self.W = np.random.rand(inputSize, self.M) *  self.INITIALIZATION_COEF
        self.t = 0
        self.sigma = self.getSigma()

    # Devuelve j*
    def activation(self, X):
        Y = self.inputMinusWieghts(X) #Distancia entre cada entrada y cada columna
        Y = np.power(np.linalg.norm(Y, axis=0), 2) # Norma 2 al cuadrado
        return np.argmin(Y)

    def correction(self, X, j_winner):
        D = self.distanceVector(j_winner)
        dW = self.lr * D * self.inputMinusWieghts(X)
        return dW
    
    def adaptation(self, dW):
        self.W += dW


    # Reliza la operacion X^T - W
    def inputMinusWieghts(self, X):
        return np.repeat(X, self.M).reshape(X.shape[0], self.M) - self.W

    # Genera un vector con las actualizaciones de distancia para cada posicion
    def distanceVector(self, j_winner):
        distances = np.zeros((1,self.M))
        for j in range(self.M):
            distances[0,j] = self.dDistance(j,j_winner)
        return distances

    # Delta de distancia
    def dDistance(self, j, j_winner):
        pNorm = self.norm2(np.subtract(self.p(j),self.p(j_winner)))
        exp = -pNorm / (2* self.sigma)
        return np.exp(exp)

    # Traduce posicion en array concatenado a posicion en matriz
    def p(self,j):
        return (j//self.M1,j%self.M1)

    # Norma 2 al cuadrado
    def norm2(self, E):
        return np.power(np.linalg.norm(E), 2)
    
    def newEpoc(self):
        self.t += 1
        self.sigma = self.getSigma()
        # print "new sigma %f" % self.sigma

    def getSigma(self):
        return self.sigma0 * np.exp(-self.t /self.sigmar)

    def predict(self, Xn):
        XPositions = list()
        for x in Xn:
            XPositions.append(self.p(self.activation(x)))
        return XPositions

    def mini_batch(self, X):
        P = X.shape[0] # Cantidad de Instancias
        # Mezclo los indices
        random_index = list(range(P))
        np.random.shuffle(random_index)
        
        i = 0
        while i < P:
            index = random_index[i]
            dW = np.zeros_like(self.W)
            for j in range(self.MINI_BATCH_SIZE):
                if i < P: # Valido de no pasarme del limite
                    x = X[index]
                    j_winner = self.activation(x)
                    dW += self.correction(x,j_winner)
                i += 1
            self.adaptation(dW)
        self.newEpoc()
