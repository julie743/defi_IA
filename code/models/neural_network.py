#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 16:57:27 2022

@author: julie

Neural network
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time


from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score

PATH_PROJECT = '/home/julie/Documents/cours/5A/IAF/defi_IA'
PATH_IMAGE = os.path.join(PATH_PROJECT,'images')

PATH_UTILITIES = os.path.join(PATH_PROJECT,'code/utilities')
os.chdir(PATH_UTILITIES)

import data_loading as DL
import data_preparation_for_models as DP
import predictions_analysis as PA
from download_prediction import download_pred_Xtest



def Optimize_NN(X_train, Y_train) :
    '''
    Perfom GridSearchCV to find the best parameters of the neural network

    Parameters
    ----------
    X_train : pandas.dataframe
        training dataset input.
    Y_train : pandas.dataframe
        training dataset output.

    Returns
    -------
    param_opt : dic
        dictionary of optiaml aparameters for learning_rate and max_depth.

    '''
    tps0=time.perf_counter()
    param_grid=[{"hidden_layer_sizes":list([(i,) for i in range(30)]), "alpha":[2,20]}]
    nnet= GridSearchCV(MLPRegressor(max_iter=1500),param_grid,cv=10,n_jobs=-1)
    nnetOpt=nnet.fit(X_train, Y_train)
    tps1=time.perf_counter()
    print("Temps execution pour l'optimisation en secondes :",(tps1 - tps0))
    
    # paramètre optimal
    param_opt = nnetOpt.best_params_
    # paramètre optimal
    print("Erreur la plus basse = %f, Meilleur paramètre = %s" % (1. - nnetOpt.best_score_,param_opt))
    return param_opt

def Model_NN(X_train,Y_train,param_opt):
    '''
    Final model which takes as an imput the optimal parameters computed during
    the optimization

    Parameters
    ----------
    X_train : pandas.dataframe
        training dataset input.
    Y_train : pandas.dataframe
        training dataset output.
    param_opt : dic
        dictionary of optiaml aparameters for learning_rate and max_depth.

    Returns
    -------
    nnetOpt : optimal model fit on the training data

    '''
    alpha = param_opt['alpha']
    hidden_layer_sizes = param_opt["hidden_layer_sizes"]
    tps0=time.perf_counter()
    nnetOpt = MLPRegressor(random_state=1, max_iter=1500, alpha=alpha,hidden_layer_sizes=hidden_layer_sizes)
    history = nnetOpt.fit(X_train, Y_train)
    tps1=time.perf_counter()
    print("Temps execution pour l'entrainement en sec :",(tps1 - tps0))
    
    # plot loss
    plt.figure()
    plt.plot(history.loss_curve_, label="loss")
    plt.legend()
    plt.title("Loss Curve", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()
    
    return nnetOpt

def Predict_validation_set(X_vali,Y_vali,nnetOpt,model_name='neural network'):
    '''
    Predict the validation set using an optimal model. Plots and records
    the results

    Parameters
    ----------
    X_vali : pandas.dataframe
        validation dataset input.
    Y_vali : pandas.dataframe
        validation dataset output.
    nnetOpt : optimal model fit on the training data
    model_name : string
        Default = "neural network" => For plots and recording files. 

    Returns
    -------
    scores : dic
        dictionary of metrics computed on the validation data (MAE, RMSE, R2)

    '''
    
    prev=nnetOpt.predict(X_vali)
    prev_detransfo = np.exp(prev)
    Y_vali_detransfo = np.exp(Y_vali)
    scores = PA.compute_scores(Y_vali_detransfo,prev_detransfo)
    PA.plot_pred_obs(Y_vali_detransfo,prev_detransfo,model_name)
    PA.scatterplot_residuals(Y_vali_detransfo,prev_detransfo,model_name)
    PA.histogram_residuals(Y_vali_detransfo,prev_detransfo,model_name)
    return scores

def Predict_test_set(X_test,nnetOpt):
    '''
    Predict the test set and record the results 

    Parameters
    ----------
    X_test : pandas.dataframe
        test dataset input.
    nnetOpt : optimal model fit on the training data

    Returns
    -------
    None.

    '''
    prev_test = nnetOpt.predict(X_test)
    prev_test = pd.DataFrame(np.exp(prev_test),columns=['price'])
    download_pred_Xtest(np.array(prev_test).flatten(),'prediction_neural_network')


def main_NN(param_opt=0) :
    '''
    main function : calls the previous functions in the correct order to 
    perform all the computations for the neural network algorithm

    Parameters
    ----------
    param_opt : dic
        dictionary of optiaml aparameters for learning_rate and max_depth.

    Returns
    -------
    None.

    '''
    
    data,Y,var_quant,var_quali,var_quali_to_encode = DL.main_load_data()
    X_train_renorm,Y_train,X_vali_renorm,Y_vali,X_test_renorm = DP.main_prepare_train_vali_data(data,Y,var_quant,var_quali,var_quali_to_encode)
    model_name = 'neural network'
    if param_opt == 0 :
        param_opt = Optimize_NN(X_train_renorm, Y_train)
    boost_opt = Model_NN(X_train_renorm, Y_train, param_opt)
    Predict_validation_set(X_vali_renorm,Y_vali,boost_opt,model_name)
    Predict_test_set(X_test_renorm,boost_opt)

params = {
    "alpha": 2,
    "hidden_layer_sizes": (20,),
}

main_NN(param_opt=0)










