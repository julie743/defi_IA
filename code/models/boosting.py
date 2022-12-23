import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

#Path Julie : '/home/julie/Documents/cours/5A/IAF/defi_IA'
#Path Eva : 'C:/Users/evaet/Documents/5A/defi_IA/' 
PATH_PROJECT = 'C:/Users/evaet/Documents/5A/defi_IA/'
PATH_IMAGE = os.path.join(PATH_PROJECT,'images')
PATH_UTILITIES = os.path.join(PATH_PROJECT,'code/utilities')


os.chdir(PATH_UTILITIES)

import data_loading as DL
import data_preparation_for_models as DP
import predictions_analysis as PA
from download_prediction import download_pred_Xtest
from predict_validation_and_test import Predict_validation_set, Predict_test_set


def boosting(X_train,Y_train,X_vali,Y_vali,params) : 
    '''
    Create boosting model, fit it on the training data, and measure the  
    accuracy on the validation data

    Parameters
    ----------
    X_train : pandas.dataframe
        training dataset input.
    Y_train : pandas.dataframe
        training dataset output.
    X_vali : pandas.dataframe
        validation dataset input.
    Y_vali : pandas.dataframe
        validation dataset output.
    params : dic
        dictionarry of parameters. For example : 
            params = {
                "n_estimators": 500,
                "max_depth": 10,
                "min_samples_split": 5,
                "learning_rate": 0.1,
                "loss": "squared_error",
            }
    Returns
    -------
    None.

    '''
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
    '''
    Perfom GridSearchCV to find the best learning rate and depth of the 
    Gradient Boosting regression

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
    param=[{"learning_rate":[0.01,0.05,0.1,0.2,0.4], "max_depth": np.arange(2,30,2)}] #optimisation de m
    rf= GridSearchCV(GradientBoostingRegressor(n_estimators=500),param,cv=5,n_jobs=1, verbose = 3) #Permet d'afficher les tests déjà réalisés
    #rf= GridSearchCV(GradientBoostingRegressor(n_estimators=500),param,cv=5,n_jobs=-1, verbose = 10)
    boostOpt=rf.fit(X_train, Y_train)
    tps1=time.perf_counter()
    print("Temps execution en sec :",(tps1 - tps0))
    
    # paramètre optimal
    param_opt = boostOpt.best_params_
    print("Erreur la moins élevée = %f, Meilleurs paramètres = %s" % (1. -boostOpt.best_score_,boostOpt.best_params_)) #1-R^2
    
    return param_opt

def Model_boosting(X_train,Y_train,param_opt):
    '''
    Final model which takes as an input the optimal parameters computed during
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
    boosting_opt : optimal model fit on the training data

    '''
    
    
    all_param = {
        "n_estimators": 500,
        "max_depth": param_opt["max_depth"],
        "min_samples_split": 5,
        "learning_rate":  param_opt["learning_rate"],
        #"loss": "squared_error",
        "loss": "ls",
    }
    
    tps0=time.perf_counter()
    boosting_opt = GradientBoostingRegressor(**all_param)
    boosting_opt.fit(X_train, Y_train)
    #boosting_opt.save(os.path.join(directory_weigths,file_name)) #### a modifier 
    tps1=time.perf_counter()
    print("Temps execution en sec :",(tps1 - tps0))
    return boosting_opt


def main_boosting(param_opt=0) :
    '''
    main function : calls the previous functions in the correct order to 
    perform all the computations for the boosting algorithm

    Parameters
    ----------
    param_opt : dic
        dictionary of optiaml aparameters for learning_rate and max_depth.

    Returns
    -------
    None.

    '''
    
    #data,Y,var_quant,var_quali,var_quali_to_encode = DL.main_load_data()
    data,Y,var_quant,var_quali,var_quali_to_encode = DL.main_load_data2()
    X_train,X_vali,X_train_renorm,Y_train,X_vali_renorm,Y_vali,X_test_renorm = DP.main_prepare_train_vali_data(data,Y,var_quant,var_quali,var_quali_to_encode)
    model_name = 'boosting_adversarial'
    if param_opt == 0 :
        param_opt = Optimize_boosting(X_train_renorm, Y_train)
    boost_opt = Model_boosting(X_train_renorm, Y_train, param_opt)
    Predict_validation_set(X_vali,X_vali_renorm,Y_vali,boost_opt,var_quant,var_quali,model_name)
    Predict_test_set(X_test_renorm,boost_opt,model_name)

params = {
    "n_estimators": 1000,
    "max_depth": 20,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}

#Avec param optimaux 
main_boosting(param_opt=params)

#Sans param optimaux
#main_boosting()

















