import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from sklearn.model_selection import GridSearchCV

import lightgbm as lgbm

#Path Eva : 'C:/Users/evaet/Documents/5A/defi_IA/' 
#Path Julie : '/home/julie/Documents/cours/5A/IAF/defi_IA'
PATH_PROJECT = 'C:/Users/evaet/Documents/5A/defi_IA/' 
PATH_IMAGE = os.path.join(PATH_PROJECT,'images')
PATH_UTILITIES = os.path.join(PATH_PROJECT,'code/utilities')

#Store the weigths 
"""directory_weigths = os.path.join(PATH_PROJECT,'weigths')
file_name = "rf_weigths_opt"""

os.chdir(PATH_UTILITIES)

import data_loading as DL
import data_preparation_for_models as DP
import predictions_analysis as PA
from download_prediction import download_pred_Xtest


def LGBM_reg(X_train,Y_train,X_vali,Y_vali,params) : 
    LGBMreg = lgbm.LGBMRegressor(**params)
    LGBMreg.fit(X_train, Y_train)
    test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
    
    for i, y_pred in enumerate(LGBMreg.staged_predict(X_vali)):
        test_score[i] = LGBMreg.loss_(Y_vali, y_pred)
    
    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("Deviance")
    plt.plot(
        np.arange(params["n_estimators"]) + 1,
        LGBMreg.train_score_,
        "b-",
        label="Training Set Deviance",
    )
    plt.plot(
        np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
    )
    
    #plt.axvline((np.arange(params["n_estimators"]) + 1)[np.argmin(test_score)])
    plt.axvline(200)
    plt.legend(loc="upper right")
    plt.xlabel("LGBM Iterations")
    plt.ylabel("Deviance/squared error loss")
    fig.tight_layout()
    plt.savefig(os.path.join(PATH_IMAGE,'LGBM_optimization.png'))
    plt.show()
    
def Optimize_LGBM(X_train, Y_train) :
    tps0=time.perf_counter()
    param=[{"num_leaves":[30,40,50], "max_depth": np.arange(10,25,5), "learning_rate":[0.05,0.07,0.09], "n_estimators":np.arange(1000,1600,100)}] #optimisation de m
    LGBMreg= GridSearchCV(lgbm.LGBMRegressor(objective="regression", metric="ls"),param,cv=5,n_jobs=1, verbose = 3) #Permet d'afficher les tests déjà réalisés
    lgbmOpt=LGBMreg.fit(X_train, Y_train)
    tps1=time.perf_counter()
    print("Temps execution en sec :",(tps1 - tps0))
    
    # paramètre optimal
    param_opt = lgbmOpt.best_params_
    print("Erreur la moins élevée = %f, Meilleurs paramètres = %s" % (1. -lgbmOpt.best_score_,lgbmOpt.best_params_)) #1-R^2
    return param_opt

def Model_LGBM(X_train,Y_train,param_opt):
    """all_param = {
        "n_estimators": 500,
        "max_depth": param_opt["max_depth"],
        "min_samples_split": 5,
        "learning_rate":  param_opt["learning_rate"],
        #"loss": "squared_error",
        "loss": "ls",
    }"""
    all_param = param_opt
    
    tps0=time.perf_counter()
    lgbmOpt = lgbm.LGBMRegressor(**all_param)
    lgbmOpt.fit(X_train, Y_train)
    #rf_opt.save(os.path.join(directory_weigths,file_name)) #### a modifier 
    tps1=time.perf_counter()
    print("Temps execution en sec :",(tps1 - tps0))
    return lgbmOpt

def Predict_validation_set(X_vali,Y_vali,lgbmOpt,model_name='lgbm'):
    prev = lgbmOpt.predict(X_vali)
    prev_detransfo = np.exp(prev)
    Y_vali_detransfo = np.exp(Y_vali)
    scores = PA.compute_scores(Y_vali_detransfo,prev_detransfo)
    PA.plot_pred_obs(Y_vali_detransfo,prev_detransfo,model_name)
    PA.scatterplot_residuals(Y_vali_detransfo,prev_detransfo,model_name)
    PA.histogram_residuals(Y_vali_detransfo,prev_detransfo,model_name)
    return scores

def Predict_test_set(X_test,lgbmOpt):
    prev_test = lgbmOpt.predict(X_test)
    prev_test = pd.DataFrame(np.exp(prev_test),columns=['price'])
    download_pred_Xtest(np.array(prev_test).flatten(),'prediction_lgbm')


def main_lgbm(param_opt=0) :
    
    data,Y,var_quant,var_quali,var_quali_to_encode = DL.main_load_data()
    X_train_renorm,Y_train,X_vali_renorm,Y_vali,X_test_renorm = DP.main_prepare_train_vali_data(data,Y,var_quant,var_quali,var_quali_to_encode)
    model_name = 'lgbm'
    if param_opt == 0 :
        param_opt = Optimize_LGBM(X_train_renorm, Y_train)
    lgbmOpt = Model_LGBM(X_train_renorm, Y_train, param_opt)
    Predict_validation_set(X_vali_renorm,Y_vali,lgbmOpt,model_name)
    Predict_test_set(X_test_renorm,lgbmOpt)

#Sans optimisation 
"""params = {"objective":"regression",
    "num_leaves":32,
    "max_depth":10,
    "learning_rate":0.05, 
    "metric":'ls',
    "n_estimators":1500}
main_lgbm(param_opt=params)"""

#Avec optimisation 
main_lgbm()
















