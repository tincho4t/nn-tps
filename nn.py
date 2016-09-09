#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

###############################
##### #Back propagation #######
###############################
class NN(object):
    
    INITIALIZATION_COEF = 0.1 # Coeficiente para disminuir la inicialización de los pesos
    
    # layers: Array con las dimensaiones de cada capa incluyendo inputs
    def __init__(self, layers, activationFunction, deltaF, lr):
        self.activationFunction = activationFunction # Funcion de activacion
        self.deltaF = deltaF # Derivara de la funcion de acticacion
        self.lr = lr # Learning Rate
        self.L = len(layers) # Layers
    
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
    
    def activation(self, Xn):
        self.Y[0] = Xn.reshape((1, -1))
        for j in range(1,self.L):
            #print("---------------")
            #print("self.Y[j-1]", self.Y[j-1])
            #print("self.Y[j-1].shape", self.Y[j-1].shape)
            #print("self.W[j-1]", self.W[j-1])
            #print("self.W[j-1].shape", self.W[j-1].shape)
            #print(type(np.dot(self.Y[j-1],self.W[j-1])))
            #print(np.dot(self.Y[j-1],self.W[j-1]))
            #print(type(self.activationFunction(np.dot(self.Y[j-1],self.W[j-1]))))
            #print("activationFunction: ", self.activationFunction(np.dot(self.Y[j-1],self.W[j-1])))
            #print("END ---------------")
            self.Y[j] = self.activationFunction(np.dot(self.Y[j-1],self.W[j-1]))
        #return self.Y[L-1] # Devuelvo el output
    
    def correction(self, Zh):
        E = (Zh-self.Y[self.L -1])
        e = self.norm2(E) # Norma 2 al cuadrado del error
        self.dW[-1] = self.dW[-1] + (self.lr * (np.multiply(self.Y[-1].T,E))) # dw = Learning Rate * (D * Yj-1)
        for j in range(self.L-1, 0, -1): # [L-1, 1]
            #print "------- START CORRECTION j: %d -----" % j
            #print("E: ", E)
            yDelta = self.deltaF(np.dot(self.Y[j-1], self.W[j-1])) # y' = f'(Yj-1 * Wj)
            #print("yDelta: ", yDelta)
            D = np.multiply(E,yDelta) # D = (Dirección de correción * tamaño de paso) = (Dj * Wj) * y'j-1
            #print("D", D)
            #print("self.Y[j-1]", self.Y[j-1].T)
            #print("np.dot(D, self.Y[j-1])", np.dot(self.Y[j-1].T,D))
            self.dW[j-1] = self.dW[j-1] + (self.lr * (np.dot(self.Y[j-1].T,D))) # dw = Learning Rate * (D * Yj-1)
            E = np.dot(D, self.W[j-1].T) # Error nuevo = D * Wj^Transpuesta 
        return e    

    # def correction(self, Zh):
    #     E = (Zh-self.Y[self.L -1])
    #     e = self.norm2(E) # Norma 2 al cuadrado del error
    #     for j in range(self.L-1, 0, -1): # [L-1, 1]
    #         yDelta = self.deltaF(np.dot(self.Y[j-1], self.W[j-1])) # y' = f'(Yj-1 * Wj)
    #         D = np.multiply(E,yDelta) # D = (Dirección de correción * tamaño de paso) = (Dj * Wj) * y'j-1
    #         self.dW[j-1] = self.dW[j-1] + (self.lr * (np.dot(D, self.Y[j-1]))) # dw = Learning Rate * (D * Yj-1)
    #         E = np.dot(D, self.W[j-1].T) # Error nuevo = D * Wj^Transpuesta 
    #     return e
    
    # Norma 2 al cuadrado
    # TODO: Validar que esto sea efectivamente la norma 2 al cuadrado
    def norm2(self, E):
        return np.power(np.linalg.norm(E), 2)
    
    def adaptation(self):
        for j in range(self.L-1):
            self.W[j] = self.W[j] + self.dW[j]
            self.dW[j] = 0
    
    
    ###############################
    ######### TRAINING ############
    ###############################
    
    def f_batch(self, X, Z):
        e = 0
        P = X.shape[0]
        for h in range(P):
            self.activation(X[h])
            e += self.correction(Z[h])
        self.adaptation()
        return e

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