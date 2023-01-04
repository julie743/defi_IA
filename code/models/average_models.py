#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 19:07:14 2023

@author: julie
"""


import os
import time
import pickle

from sklearn.ensemble import VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor


PATH_PROJECT = open("set_path.txt",'r').readlines()[0]
PATH_IMAGE = os.path.join(PATH_PROJECT,'images')
PATH_WEIGTHS = os.path.join(PATH_PROJECT,'weigths')

PATH_UTILITIES = os.path.join(PATH_PROJECT,'code/utilities')
os.chdir(PATH_UTILITIES)
import data_loading as DL
import data_preparation_for_models as DP
from predict_validation_and_test import Predict_validation_set, Predict_test_set
os.chdir(PATH_PROJECT)

# load model's weigths 
def voting_model(X_train,Y_train, ranking = [1,1]) :
    '''
    model that combines the neural network model and the catboost model in 
    a weigthed average

    Parameters
    ----------
   X_train : pandas.dataframe
       training dataset input.
   Y_train : pandas.dataframe
       training dataset output.
    ranking : list 
        list of weigths. The default is [1,1].

    Returns
    -------
    ensemble : sklearn model
        voting regression model.

    '''
    nn = MLPRegressor(random_state=1, max_iter=1500, alpha=0.5,hidden_layer_sizes=18)
    
    cb = CatBoostRegressor(n_estimators=2478,
                          learning_rate=0.29014147234242005,
                          max_depth=10)
    
    models = [('nn',nn),('catboost',cb)]
    ensemble = VotingRegressor(estimators=models, weights=ranking)
    tps0=time.perf_counter()
    ensemble.fit(X_train,Y_train)
    tps1=time.perf_counter()
    print("Temps execution pour l'entrainement en sec :",(tps1 - tps0))
    return ensemble 


def main_average_models(ranking = [1,1]):
    '''
    main function : calls the previous functions in the correct order to 
    perform all the computations for the voting model

    Parameters
    ----------
    ranking : list 
        list of weigths. The default is [1,1].

    Returns
    -------
    None.

    '''
    data,Y,var_quant,var_quali,var_quali_to_encode = DL.main_load_data2()
    X_train,X_vali,X_train_renorm,Y_train,X_vali_renorm,Y_vali,X_test_renorm = DP.main_prepare_train_vali_data(data,Y,var_quant,var_quali,var_quali_to_encode)
   
    model_name = 'average models adversarial'
    ensemble = voting_model(X_train_renorm,Y_train,ranking)
    path_weigths = os.path.join(PATH_PROJECT,'weigths','average_3models_adverserial.sav')
    pickle.dump(ensemble, open(path_weigths, 'wb'))
    
    Predict_validation_set(X_vali,X_vali_renorm,Y_vali,ensemble,var_quant,var_quali,model_name)
    Predict_test_set(X_test_renorm,ensemble,model_name)

#main_average_models(ranking = [0.7,0.3])