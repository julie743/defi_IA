import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor

PATH_PROJECT = open("set_path.txt",'r').readlines()[0]
PATH_IMAGE = os.path.join(PATH_PROJECT,'images')
PATH_UTILITIES = os.path.join(PATH_PROJECT,'code/utilities')


os.chdir(PATH_UTILITIES)
import data_loading as DL
import data_preparation_for_models as DP
import predictions_analysis as PA
from download_prediction import download_pred_Xtest
from predict_validation_and_test import Predict_validation_set, Predict_test_set
os.chdir(PATH_PROJECT)

def adaboost(X_train,Y_train,X_vali,Y_vali,params) : 
    reg = AdaBoostRegressor(**params)
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
    plt.xlabel("Adaboost Iterations")
    plt.ylabel("Deviance/squared error loss")
    fig.tight_layout()
    plt.savefig(os.path.join(PATH_IMAGE,'adaboost_optimization.png'))
    plt.show()
    
def Optimize_adaboost(X_train, Y_train) :
    tps0=time.perf_counter()
    param=[{"n_estimators": np.arange(2000,6000,1000)}]
    ada= GridSearchCV(AdaBoostRegressor(learning_rate=0.3),param,cv=5,n_jobs=1, verbose = 3) #Permet d'afficher les tests déjà réalisés
    adaOpt=ada.fit(X_train, Y_train)
    tps1=time.perf_counter()
    print("Temps execution en sec :",(tps1 - tps0))
    
    # paramètre optimal
    param_opt = adaOpt.best_params_
    print("Erreur la moins élevée = %f, Meilleurs paramètres = %s" % (1. -adaOpt.best_score_,adaOpt.best_params_)) #1-R^2
    
    return param_opt

def Model_adaboost(X_train,Y_train,param_opt):
    all_param = {
        "n_estimators": 500,
        "max_depth": param_opt["max_depth"],
        "min_samples_split": 5,
        "learning_rate":  param_opt["learning_rate"],
        #"loss": "squared_error",
        "loss": "ls",
    }
    
    tps0=time.perf_counter()
    adaboost_opt = AdaBoostRegressor(**all_param)
    adaboost_opt.fit(X_train, Y_train) 
    tps1=time.perf_counter()
    print("Temps execution en sec :",(tps1 - tps0))
    return adaboost_opt


def main_adaboost(param_opt=0) :
    #data,Y,var_quant,var_quali,var_quali_to_encode = DL.main_load_data()
    data,Y,var_quant,var_quali,var_quali_to_encode = DL.main_load_data2()
    X_train,X_vali,X_train_renorm,Y_train,X_vali_renorm,Y_vali,X_test_renorm = DP.main_prepare_train_vali_data(data,Y,var_quant,var_quali,var_quali_to_encode)
    model_name = 'adaboost_adversarial'
    if param_opt == 0 :
        param_opt = Optimize_adaboost(X_train_renorm, Y_train)
    #ada_opt = Model_adaboost(X_train_renorm, Y_train, param_opt)
    #Predict_validation_set(X_vali,X_vali_renorm,Y_vali,ada_opt,var_quant,var_quali,model_name)
    #Predict_test_set(X_test_renorm,ada_opt,model_name)

"""params = {
    "n_estimators": 1000,
    "max_depth": 20,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}"""

#Avec param optimaux 
#main_adaboost(param_opt=params)

#Sans param optimaux
#main_adaboost()

















