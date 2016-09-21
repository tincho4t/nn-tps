#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle
from scipy.stats import skew
import numpy as np
from numpy import genfromtxt

class DatasetNormalizer(object):
    # Training
    EJ_1_DTYPE="S5,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8"
    EJ_2_DTYPE="f8,f8,f8,f8,f8,f8,f8,f8,f8,f8"
    EJ1_COLUMNS_NAME=['radio', 'textura', 'perimetro', 'area', 'suavidad', 'compacidad', 'concavidad', 'puntos_concavos', 'simetria', 'desconocido']
    EJ2_COLUMNS_NAME=['Compacidad Relativa','Área de la Superficie Total','Área de las Paredes','Área del Techo','Altura Total','Orientación','Área de Reflejo Total','Distribución del Área de Reflejo'] #,'Carga de Calefacción',' Carga de Refrigeración']
    # Test
    EJ_1_DTYPE_TEST="f8,f8,f8,f8,f8,f8,f8,f8,f8,f8"
    EJ_2_DTYPE_TEST="f8,f8,f8,f8,f8,f8,f8,f8"

    def __init__(self, filename, ej=None, test=False):
        super(DatasetNormalizer, self).__init__()
        self.filename = filename
        if(ej == "ej1"):
            self.data = self.loadDatasetEj1(test)
        else:
            self.data = self.loadDatasetEj2(test)

    #TODO: Parece q toma la 1er linea como header
    def loadDatasetEj1(self, test):
        if (test):
            csv_data = genfromtxt(self.filename, delimiter=',', dtype=self.EJ_1_DTYPE_TEST)
        else:
            csv_data = genfromtxt(self.filename, delimiter=',', dtype=self.EJ_1_DTYPE)
        data = list()
        for row in csv_data:
            if(not test): # Si no es test discretizo el resultado
                row[0] = 1 if row[0] == 'M' else 0
            r = list()
            for element in row:
                r.append(float(element))
            data.append(np.array(r))
        return np.array(data)
    
    #TODO: Parece q toma la 1er linea como header
    def loadDatasetEj2(self, test):
        if (test):
            csv_data = genfromtxt(self.filename, delimiter=',', dtype=self.EJ_2_DTYPE_TEST)
        else:
            csv_data = genfromtxt(self.filename, delimiter=',', dtype=self.EJ_2_DTYPE)
        data = list()
        for row in csv_data:
            r = list()
            for element in row:
                r.append(float(element))
            data.append(np.array(r))
        return np.array(data)
    
    def discretization(self, data, colnames, save_path=None, intervals=[-11,-9,-7, -5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 6, 8]):
        intervals = list(np.arange(-15,15,0.5))
        save_dictionary = {}
        save_dictionary["intervals"] = intervals
        m_intervals = len(intervals)
        n = data.shape[0]
        m = data.shape[1]
        new_data_set = np.zeros((n, m*m_intervals))
        for col in range(m):        
            column_temp = data[:, col].copy()
            skewness = skew(column_temp)
            min_val = column_temp.min()
            if skewness > 2:
                column_temp = np.log(column_temp + abs(min_val) + 1)
            mean = np.mean(column_temp)
            std = np.std(column_temp)
            new_columns = np.zeros((column_temp.shape[0], len(intervals)))
            new_columns[column_temp > (mean + intervals[0] * std / 2.0), 0] = 1
            for i in range(1, len(intervals)):
                new_columns[column_temp >= (mean + intervals[i] * std / 2.0), i] = 1
            new_data_set[:, (col*m_intervals):((col+1)*m_intervals)] = new_columns
            #print("Doing column: "+str(colnames[col])+" to go "+str(data.shape[1]-col)+str(skewness)+" "+str(column_temp.min())+" "+str(mean)+" "+str(std))
            save_dictionary[colnames[col]] = {"skewness": skewness, "mean": mean, "std": std, "min": min_val}
        if save_path:
            f = file(save_path, 'wb')
            cPickle.dump(save_dictionary, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()
        return(new_data_set)


    def apply_discretization(self, data, colnames, load_path):
        f = file(load_path, 'rb')
        load_dictionary = cPickle.load(f)
        f.close()
        intervals = load_dictionary["intervals"]
        m_intervals = len(intervals)
        n = data.shape[0]
        m = data.shape[1]
        new_data_set = np.zeros((n, m*m_intervals))
        for col in range(m):
            column_temp = data[:, col].copy()
            skewness = load_dictionary[colnames[col]]["skewness"]
            if skewness > 2:
                column_temp = np.log(column_temp + abs(load_dictionary[colnames[col]]["min"]) + 1)
            mean = load_dictionary[colnames[col]]["mean"]
            std = load_dictionary[colnames[col]]["std"]
            new_columns = np.zeros((column_temp.shape[0], len(intervals)))
            new_columns[column_temp > (mean + intervals[0] * std / 2.0), 0] = 1
            for i in range(1, len(intervals)):
                new_columns[column_temp >= (mean + intervals[i] * std / 2.0), i] = 1
            new_data_set[:, (col*m_intervals):((col+1)*m_intervals)] = new_columns
            #print("Doing column: "+str(colnames[col])+" to go "+str(data.shape[1]-col)+str(skewness)+" "+str(load_dictionary[colnames[col]]["min"])+" "+str(mean)+" "+str(std))
        return(new_data_set)
