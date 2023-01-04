import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
import sys
import pickle

PATH_PROJECT = open("set_path.txt",'r').readlines()[0]
PATH_IMAGE = os.path.join(PATH_PROJECT,'images')
PATH_UTILITIES = os.path.join(PATH_PROJECT,'code/utilities')


os.chdir(PATH_UTILITIES)
#sys.path.insert(1, './')
import data_loading as DL
import data_preparation_for_models as DP
import predictions_analysis as PA
from download_prediction import download_pred_Xtest
from predict_validation_and_test import Predict_validation_set, Predict_test_set
os.chdir(PATH_PROJECT)

def catboost(X_train,Y_train,X_vali,Y_vali,params) : 
    reg = CatBoostRegressor(**params)
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
    plt.xlabel("Catboost Iterations")
    plt.ylabel("Deviance/squared error loss")
    fig.tight_layout()
    plt.savefig(os.path.join(PATH_IMAGE,'catboost_optimization.png'))
    plt.show()
    
def Optimize_catboost(X_train, Y_train) :
    tps0=time.perf_counter()
    param=[{"n_estimators": np.arange(1000,2500,500)}]
    cat= GridSearchCV(CatBoostRegressor(learning_rate=0.1),param,cv=5,n_jobs=1, verbose = 3) #Permet d'afficher les tests déjà réalisés
    catOpt=cat.fit(X_train, Y_train)
    tps1=time.perf_counter()
    print("Temps execution en sec :",(tps1 - tps0))
    
    # paramètre optimal
    param_opt = catOpt.best_params_
    print("Erreur la moins élevée = %f, Meilleurs paramètres = %s" % (1. -catOpt.best_score_,catOpt.best_params_)) #1-R^2
    
    return param_opt

def Model_catboost(X_train,Y_train,param_opt):
    all_param = {
        "n_estimators": param_opt["n_estimators"],
        "max_depth": param_opt["max_depth"],
        # "min_samples_split": 5,
        "learning_rate":  param_opt["learning_rate"],
        #"loss": "squared_error",
        # "loss": "ls",
    }
    
    tps0=time.perf_counter()
    catboost_opt = CatBoostRegressor(**all_param)
    catboost_opt.fit(X_train, Y_train) 
    tps1=time.perf_counter()
    print("Temps execution en sec :",(tps1 - tps0))
    return catboost_opt


def main_catboost(param_opt=0) :
    #data,Y,var_quant,var_quali,var_quali_to_encode = DL.main_load_data()
    data,Y,var_quant,var_quali,var_quali_to_encode = DL.main_load_data2()
    X_train,X_vali,X_train_renorm,Y_train,X_vali_renorm,Y_vali,X_test_renorm = DP.main_prepare_train_vali_data(data,Y,var_quant,var_quali,var_quali_to_encode)
    model_name = 'catboost_adversarial'
    if param_opt == 0 :
        param_opt = Optimize_catboost(X_train_renorm, Y_train)
    cat_opt = Model_catboost(X_train_renorm, Y_train, param_opt)
    
    # predict validation and test sets : 
    Predict_validation_set(X_vali,X_vali_renorm,Y_vali,cat_opt,var_quant,var_quali,model_name)
    Predict_test_set(X_test_renorm,cat_opt,model_name)
    
    # save the model
    path_weigths = os.path.join(PATH_PROJECT,'weigths','catboost_adversarial.sav')
    pickle.dump(cat_opt, open(path_weigths, 'wb'))
    
"""params = {
    "n_estimators": 1000,
    "max_depth": 20,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}"""

#Avec param optimaux 
#main_catboost(param_opt=params)

#Sans param optimaux
#main_catboost()

#----------------------------------- OPTUNA -----------------------------------
import optuna
from sklearn.metrics import mean_squared_error
SEED = 42
n_trials = 1000

#Function that we will use after to plot the best hyper parameters
def tune(objective) :
    study = optuna.create_study(direction = "minimize")
    study.optimize(objective, n_trials=n_trials)
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    best_params = study.best_trial.params
   
    return best_params

def cat_objective(trial) :
    #x_train, x_test, y_train, y_test = train_test_split(data[testons], target, test_size=0.2,random_state=42)    
    data,Y,var_quant,var_quali,var_quali_to_encode = DL.main_load_data2()
    X_train,X_vali,X_train_renorm,Y_train,X_vali_renorm,Y_vali,X_test_renorm = DP.main_prepare_train_vali_data(data,Y,var_quant,var_quali,var_quali_to_encode)
    _n_estimators = trial.suggest_int("n_estimators", 1000, 2500)
    _learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    _max_depth = trial.suggest_int("max_depth", 4, 10)
   
    cat = CatBoostRegressor(
        n_estimators = _n_estimators,  
        learning_rate = _learning_rate,
        max_depth = _max_depth,
    )
   
    #early_stopping_rounds avoids overfitting
    cat.fit(X_train_renorm,Y_train)
    preds = cat.predict(X_vali_renorm)
    rmse = mean_squared_error(Y_vali, preds, squared=False)
    
    return rmse

# cat_params = tune(cat_objective)
# cat_params = {'n_estimators': 2478, 'learning_rate': 0.29014147234242005, 'max_depth': 10}
# main_catboost(param_opt=cat_params)


















