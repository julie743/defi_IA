#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 11:36:05 2022

@author: julie
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from math import log,sqrt,exp
import os
import time


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from itertools import cycle
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingRegressor

PATH_PROJECT = '/home/julie/Documents/cours/5A/IAF/defi_IA'
PATH_IMAGE = os.path.join(PATH_PROJECT,'images')

PATH_UTILITIES = os.path.join(PATH_PROJECT,'code/utilities')
os.chdir(PATH_UTILITIES)

import data_loading as DL
import data_preparation_for_models as DP
import predictions_analysis as PA
from download_prediction import download_pred_Xtest


def boosting(X_train,Y_train,X_vali,Y_vali,params) : 
    reg = GradientBoostingRegressor(**params)
    reg.fit(X_train, Y_train)
    test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
    
    for i, y_pred in enumerate(reg.staged_predict(X_vali)):
        test_score[i] = reg.loss_(Y_vali, y_pred)
    
    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("Deviance")
    plt.plot(
        np.arange(params["n_estimators"]) + 1,
        reg.train_score_,
        "b-",
        label="Training Set Deviance",
    )
    plt.plot(
        np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
    )
    
    #plt.axvline((np.arange(params["n_estimators"]) + 1)[np.argmin(test_score)])
    plt.axvline(200)
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Deviance/squared error loss")
    fig.tight_layout()
    plt.savefig(os.path.join(PATH_IMAGE,'boosting_optimization.png'))
    plt.show()
    
def Optimize_boosting(X_train, Y_train) :
    tps0=time.perf_counter()
    param=[{"learning_rate":[0.01,0.05,0.1,0.2,0.3,0.4,0.5], "max_depth": np.arange(2,20,3)}]#optimisation de m
    rf= GridSearchCV(GradientBoostingRegressor(n_estimators=500),param,cv=5,n_jobs=-1)
    boostOpt=rf.fit(X_train, Y_train)
    tps1=time.perf_counter()
    print("Temps execution en mn :",(tps1 - tps0))
    
    # paramètre optimal
    param_opt = boostOpt.best_params_
    print("Error la moins élevée = %f, Meilleur paramètre = %s" % (1. -boostOpt.best_score_,boostOpt.best_params_)) #1-R^2
    
    return param_opt

def Model_boosting(X_train,Y_train,param_opt):
    all_param = {
        "n_estimators": 500,
        "max_depth": param_opt["max_depth"],
        "min_samples_split": 5,
        "learning_rate":  param_opt["learning_rate"],
        "loss": "squared_error",
    }
    
    tps0=time.perf_counter()
    boosting_opt = GradientBoostingRegressor(**all_param)
    boosting_opt.fit(X_train, Y_train)
    tps1=time.perf_counter()
    print("Temps execution en sec :",(tps1 - tps0))
    return boosting_opt

def Predict_validation_set(X_vali,Y_vali,boosting_opt,model_name):
    prev=boosting_opt.predict(X_vali)
    prev_detransfo = np.exp(prev)
    Y_vali_detransfo = np.exp(Y_vali)
    scores = PA.compute_scores(Y_vali_detransfo,prev_detransfo)
    PA.plot_pred_obs(Y_vali_detransfo,prev_detransfo,model_name)
    PA.scatterplot_residuals(Y_vali_detransfo,prev_detransfo,model_name)
    PA.histogram_residuals(Y_vali_detransfo,prev_detransfo,model_name)
    return scores

def Predict_test_set(X_test,boosting_opt):
    prev_test = boosting_opt.predict(X_test)
    prev_test = pd.DataFrame(np.exp(prev_test),columns=['price'])
    download_pred_Xtest(np.array(prev_test).flatten(),'prediction_boosting')


def main_boosting(param_opt=0) :
    model_name = 'boosting'
    data,Y,var_quant,var_quali = DL.main_load_data()
    X_train_renorm,Y_train,X_vali_renorm,Y_vali,X_test_renorm = DP.main_prepare_train_vali_data(data,Y,var_quant,var_quali)
    if param_opt == 0 :
        param_opt = Optimize_boosting(X_train_renorm, Y_train)
    boost_opt = Model_boosting(X_train_renorm, Y_train, param_opt)
    Predict_validation_set(X_vali_renorm,Y_vali,boost_opt,model_name)
    Predict_test_set(X_test_renorm,boost_opt)

data,Y,var_quant,var_quali = DL.main_load_data()

params = {
    "n_estimators": 500,
    "max_depth": 10,
    "min_samples_split": 5,
    "learning_rate": 0.1,
    "loss": "squared_error",
}

main_boosting(param_opt=params)