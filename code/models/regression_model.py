#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 01:21:47 2022

@author: julie
"""

import os
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV, LassoLarsCV
from itertools import cycle
import pandas as pd

PATH_PROJECT = '/home/julie/Documents/cours/5A/IAF/defi_IA'
PATH_UTILITIES = os.path.join(PATH_PROJECT,'code/utilities')
os.chdir(PATH_UTILITIES)
import data_loading as DL
import data_preparation_for_models as DP
import predictions_analysis as PA
from download_prediction import download_pred_Xtest

# modele regression linéaire
def Model_reg(X_train,Y_train,alpha=0):
    regLin = linear_model.Lasso(alpha)
    regLin.fit(X_train,Y_train)
    return regLin

def Optimize_regLasso(X_train,Y_train,list_param):
    param=[{"alpha":list_param}]
    regLasso = GridSearchCV(linear_model.Lasso(), param,cv=5,n_jobs=-1)
    regLassOpt=regLasso.fit(X_train, Y_train)
    # paramètre optimal
    alpha_opt = regLassOpt.best_params_["alpha"]
    print("Meilleur R2 = %f, Meilleur paramètre = %s" % (regLassOpt.best_score_,regLassOpt.best_params_))
    return alpha_opt

# prediction échantillon de validation 
def Predict_validation_set(X_vali,Y_vali,model_opt,model_name):
    prev=model_opt.predict(X_vali)
    prev_detransfo = np.exp(prev)
    Y_vali_detransfo = np.exp(Y_vali)
    scores = PA.compute_scores(Y_vali_detransfo,prev_detransfo)
    PA.plot_pred_obs(Y_vali_detransfo,prev_detransfo,model_name)
    PA.scatterplot_residuals(Y_vali_detransfo,prev_detransfo,model_name)
    PA.histogram_residuals(Y_vali_detransfo,prev_detransfo,model_name)
    return scores

# prediction échantillon de test
def Predict_test_set(X_test,model_opt):
    prev_test = model_opt.predict(X_test)
    prev_test = pd.DataFrame(np.exp(prev_test),columns=['price'])
    download_pred_Xtest(np.array(prev_test).flatten(),'regression_model')

def main_Linear():
    data,Y,var_quant,var_quali = DL.main_load_data()
    X_train_renorm,Y_train,X_vali_renorm,Y_vali,X_test_renorm = DP.main_prepare_train_vali_data(data,Y,var_quant,var_quali)
    model_name = 'regression model'
    
    regLin = Model_reg(X_train_renorm,Y_train)
    scores = Predict_validation_set(X_vali_renorm,Y_vali,regLin,model_name)
    Predict_test_set(X_test_renorm,regLin)
    
def main_Linear():
    data,Y,var_quant,var_quali = DL.main_load_data()
    X_train_renorm,Y_train,X_vali_renorm,Y_vali,X_test_renorm = DP.main_prepare_train_vali_data(data,Y,var_quant,var_quali)
    model_name = 'Lasso regression model'
    
    alpha_opt = Optimize_regLasso(X_train_renorm,Y_train,[0.05,0.1,0.2,0.3,0.4])
    regLasso = Model_reg(X_train_renorm,Y_train,alpha_opt)
    scores = Predict_validation_set(X_vali_renorm,Y_vali,regLasso,model_name)
    Predict_test_set(X_test_renorm,regLasso)
    
    
