#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 01:54:25 2022

@author: julie
"""

import numpy as np
import pandas as pd
import time
import os

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

PATH_PROJECT = '/home/julie/Documents/cours/5A/IAF/defi_IA'

PATH_UTILITIES = os.path.join(PATH_PROJECT,'code/utilities')
os.chdir(PATH_UTILITIES)

import data_loading as DL
import data_preparation_for_models as DP
import predictions_analysis as PA
from download_prediction import download_pred_Xtest

# regression tree--------------------------------------------------------------
# 1. Optimisation
def Optimize_regTree(X, Y,list_max_depth) :
    tps0=time.perf_counter()
    param=[{"max_depth":list_max_depth}]
    tree= GridSearchCV(DecisionTreeRegressor(),param,cv=10,n_jobs=-1)
    treeOptr=tree.fit(X, Y)
    print("Meilleur score = %f, Meilleur paramètre = %s" % (1. - treeOptr.best_score_,treeOptr.best_params_))
    best_param = treeOptr.best_params_['max_depth']
    tps1=time.perf_counter()
    print("Temps execution en mn :",(tps1 - tps0))
    return best_param

# 2. Fit le modèle avec les meilleurs paramètres 
def Model_regTree(X,Y,param_opt):
    tps0=time.perf_counter()
    treeR=DecisionTreeRegressor(max_depth=param_opt) #on optimise ici selon la profondeur
    tps1=time.perf_counter()
    print("Temps execution en mn :",(tps1 - tps0))
    treeR.fit(X,Y)
    return treeR


# prediction échantillon de validation ----------------------------------------
def Predict_validation_set(X_vali,Y_vali,treeR,model_name):
    prev=treeR.predict(X_vali)
    prev_detransfo = np.exp(prev)
    Y_vali_detransfo = np.exp(Y_vali)
    scores = PA.compute_scores(Y_vali_detransfo,prev_detransfo)
    PA.plot_pred_obs(Y_vali_detransfo,prev_detransfo,model_name)
    PA.scatterplot_residuals(Y_vali_detransfo,prev_detransfo,model_name)
    PA.histogram_residuals(Y_vali_detransfo,prev_detransfo,model_name)
    return scores

# prediction échantillon de test
def Predict_test_set(X_test,treeR):
    prev_test = treeR.predict(X_test)
    prev_test = pd.DataFrame(np.exp(prev_test),columns=['price'])
    download_pred_Xtest(np.array(prev_test).flatten(),'prediction_regression_tree2')

def main_regression_tree(list_max_depth) :
    model_name = 'regression tree'
    data,Y,var_quant,var_quali = DL.main_load_data()
    X_train_renorm,Y_train,X_vali_renorm,Y_vali,X_test_renorm = DP.main_prepare_train_vali_data(data,Y,var_quant,var_quali)
    param_opt = Optimize_regTree(X_train_renorm, Y_train,list_max_depth)
    treeR = Model_regTree(X_train_renorm, Y_train, param_opt)
    Predict_validation_set(X_vali_renorm,Y_vali,treeR,model_name)
    Predict_test_set(X_test_renorm,treeR)
    

