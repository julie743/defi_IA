#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 23:07:12 2022

@author: julie
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

def plot_pred_obs(Y_true,Y_pred): 
    plt.figure(figsize=(5,5))
    plt.plot(Y_true,Y_pred,"o",markersize = 0.4)
    plt.xlabel("prix prédit")
    plt.ylabel("prix observé")
    plt.show()

def scatterplot_residuals(Y_true,Y_pred):
    plt.figure(figsize=(5,5))
    plt.plot(Y_pred,Y_true-Y_pred,"o",markersize = 0.4)
    plt.xlabel(u"valeurs prédites")
    plt.ylabel(u"Résidus")
    plt.title("Residus pénalité L1 Lasso") 
    plt.hlines(0,0,3)
    plt.show()
    
def histogram_residuals(Y_true,Y_pred):
    plt.figure(figsize=(10,5))
    plt.hist(Y_true-Y_pred,bins=20)
    plt.title('histogramme des résidus')
    plt.xlabel('valeur des résidus')
    plt.ylabel('nombre de prédictions')
    plt.show()
    
def compute_scores(Y_true,Y_pred):
    dic = {'rmse': np.sqrt(mean_squared_error(Y_pred,Y_true)),
           'mae' : mean_absolute_error(Y_pred,Y_true),
           'r2'  : r2_score(Y_pred,Y_true) }
    print("RMSE = ", dic['rmse'])
    print("R2 = ", dic['r2'])
    print("MAE = ", dic['mae'])
    return dic


    