#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np


class NN_HEBBIANO(object):
    
    MINI_BATCH_SIZE = 15 # Cantidad de muestras que se toman por minibach
    INITIALIZATION_COEF = 0.001 # Coeficiente para disminuir la inicialización de los pesos
    
    def __init__(self, inputSize, outputSize, lr, method):
        self.lr = lr # Learning Rate
        self.method = method
        self.N  = inputSize
        self.M  = outputSize
        self.W = np.random.rand(self.N, self.M) *  self.INITIALIZATION_COEF
        if(method == 'sanger'):
            self.getTo = self.sangerTo
        elif(method == 'oja_n'):
            self.getTo = self.ojaNTo
        elif(method == 'oja_1'):
            self.getTo = self.oja1To
        else:
            ValueError("Method %s no encontrado" % method)

    def activation(self, X):
        Y = np.dot(X,self.W)
        return Y

    def sangerTo(self, j):
        return j+1

    def ojaNTo(self, j):
        return self.M

    def oja1To(self, j):
        return 1

    # def correction(self, X, Y):
    #     xhat = self.getXhat(Y)
    #     X_ = np.tile(X.T,(1,self.M))
    #     xDif = (X_ - xhat)
    #     dW = self.lr * np.dot(xDif, Y.T) # lr * (X - X^) * Y
    #     return dW

    def getXhat(self, Y):
        return np.dot(self.W, Y.T * self.U) # W * (Y^T * U)
    
    def correction(self, X, Y):
        dW = np.zeros_like(self.W)
        for j in range(self.M):
            for i in range(self.N):
                xhat = 0
                for k in range(self.getTo(j)):
                    xhat += Y[0,k] * self.W[i,k]
                dW[i,j] = self.lr * (X[0,i] - xhat) * Y[0,j]
        return dW

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
                    dW = self.correction(x,Y)
                    self.adaptation(dW)
                i += 1

    # Definimos error como la suma de los productos internos dos a dos de W
    def getError(self):
        error = 0.0
        for i in range(self.M-1):
            v = self.W[:,i]
            w = self.W[:,i+1:]
            error += np.sum(np.dot(v.T,w))
        return error

    def getSilhouette(self, X, Z):
        Y = self.predict(X)
        index_dict = {}
        classes = np.unique(Z)
        for c in classes:
            class_index = np.where(Z==c)
            index_dict[c] = class_index

        acum_silhouette = 0
        for i in range(len(Y)):
            current_class = Z[i]
            class_index = index_dict[current_class]
            i_point = Y[i,:]
            a = self.getAverageDistanceToPoint(i_point,Y[class_index,:])
            b = float("inf")
            for other_class in classes[classes!=current_class]:
                b_tmp = self.getAverageDistanceToPoint(i_point, Y[index_dict[other_class],:])
                b = (b if b < b_tmp else b_tmp)
            acum_silhouette += (b-a)/max(a,b)
        return acum_silhouette/len(Y)

    def getAverageDistanceToPoint(self, point, other_points):
        acum = 0
        for other_point in other_points:
            acum += np.linalg.norm(point-other_point)
        return acum/len(other_points)