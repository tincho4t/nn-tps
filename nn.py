#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

###############################
##### #Back propagation #######
###############################
class NN(object):
    
    INITIALIZATION_COEF = 0.01 # Coeficiente para disminuir la inicialización de los pesos
    
    # layers: Array con las dimensaiones de cada capa incluyendo inputs
    def __init__(self, layers, activationFunctions, lr):
        self.activationFunctions = activationFunctions # Funcion de activacion
        self.lr = lr # Learning Rate
        self.L = len(layers) # Layers
    
        ################### BIAS HACK ######################
        layers[0] += 1 # Agregamos el bias
        ####################### INITIALIZATION ######################
    
        self.Y = list() # Activación de las distintas capas
        for i in range(self.L): #TODO: Analizar si se puede volver local a activation 
            layerSize = layers[i] # Tamanio de la capa
            self.Y.append(np.zeros((1, layerSize))) # Y es vector Fila
        
        self.W = list() # Pesos de cada capa
        self.dW = list() # Adaptación de los Pesos de cada capa
        for i in range(1,self.L):
            originLayerSize = layers[i-1]
            destLayerSize = layers[i]
            weights = np.random.rand(originLayerSize, destLayerSize) *  self.INITIALIZATION_COEF
            self.W.append(weights)
            self.dW.append(np.zeros_like(weights))

    # Ejecuta la f[i]
    def activationFunction(self, i, x):
        return self.activationFunctions[i][0](x)

    # Ejecuta la  f'[i]
    def deltaF(self, i, x):
        return self.activationFunctions[i][1](x)
    
    def activation(self, Xn):
        self.Y[0] = Xn.reshape((1, -1))
        for j in range(1,self.L):
            self.Y[j] = self.activationFunction(j, np.dot(self.Y[j-1],self.W[j-1]))
        return self.Y[self.L-1] # Devuelvo el output
    
    def correction(self, Zh):
        # Error y delta de la última capa
        E = (Zh-self.Y[self.L -1])
        self.dW[-1] = self.dW[-1] + (self.lr * (np.multiply(self.Y[-2].T,E))) # dw = Learning Rate * ((Zh-Y[-1]) * Y[-2])
        
        e = self.norm2(E) # Calculo la Norma 2 al cuadrado del error para devolverla
        for j in range(self.L-2, -1, -1): # [L-1, 1]
            yDelta = self.deltaF(j+1, np.dot(self.Y[j], self.W[j])) # yj' = fj'(Yj-1 * Wj-1)
            D = np.multiply(E,yDelta) # D = (Dirección de correción * tamaño de paso) = (Dj * Wj) * y'j
            self.dW[j] = self.dW[j] + (self.lr * (np.dot(self.Y[j].T,D))) # dw = Learning Rate * (D * Yj)
            E = np.dot(D, self.W[j].T) # Error nuevo = D * Wj^Transpuesta 
        return e    

    def predict(self, Xn):
        Xn = self.addBias(Xn)
        Zhat = list()
        for x in Xn:
            Zhat.append(self.activation(x))
        return np.array(Zhat)
    
    # Norma 2 al cuadrado
    def norm2(self, E):
        # return E
        return np.power(np.linalg.norm(E), 2)
    
    def adaptation(self):
        for j in range(self.L-1):
            self.W[j] = self.W[j] + self.dW[j]
            self.dW[j] = 0
    
    
    ###############################
    ######### TRAINING ############
    ###############################
    
    def f_batch(self, X, Z):
        ones = np.atleast_2d(np.ones(X.shape[0])) #Agregamos el BIAS
        X = np.concatenate((ones.T, X), axis=1)
        e = 0
        P = X.shape[0]
        for h in range(P):
            self.activation(X[h])
            e += self.correction(Z[h])
        self.adaptation()
        return e

    def random_batch(self, X, Z):
        X = self.addBias(X)
        e = 0
        P = X.shape[0]
        random_index = list(range(P))
        np.random.shuffle(random_index)
        for h in random_index[0:15]:
            self.activation(X[h])
            e += self.correction(Z[h])
        self.adaptation()
        return e

    def addBias(self, X):
        ones = np.atleast_2d(np.ones(X.shape[0]))
        return np.concatenate((ones.T, X), axis=1)

# def incriemental(self, X,Z):
#     e = 0
#     for h in Permutacions(1..P):
#         activation(X[h])
#         e += correction(Z[h])
#         adaptation()
#     return e


###############################
######### VALIDATION ##########
###############################

# def none(epsilon, epocs):
#     e =  1
#     t = 0
#     while(epsilon < e and t < epocs):
#         e = training(X,Z)
#         t += 1
#     return (e,t)

# def holdout(epsilon, epocs):
#     e = 1
#     t = 0
#     V = [25% del dataset]
#     while(epsilon < e and t < epocs):
#         et = training(Xtrain,Ztrain)
#         ev = testing(Xval,Zval) # Lo calculo siempre para monitorearlo
#         t += 1
#     return (ev,t)

# SGD: Stocastic Gradient Decent
# Errores a monitorear en el TP
#Monitoreas ECM
#Error promedio
#Error máximo