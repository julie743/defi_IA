import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from sklearn.model_selection import GridSearchCV

from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet, RidgeCV, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.pipeline import make_pipeline
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
import xgboost as xgb

import xgboost as xgb 

#Path Eva : 'C:/Users/evaet/Documents/5A/defi_IA/' 
#Path Julie : '/home/julie/Documents/cours/5A/IAF/defi_IA'
PATH_PROJECT = 'C:/Users/evaet/Documents/5A/defi_IA/' 
PATH_IMAGE = os.path.join(PATH_PROJECT,'images')
PATH_UTILITIES = os.path.join(PATH_PROJECT,'code/utilities')

os.chdir(PATH_UTILITIES)

import data_loading as DL
import data_preparation_for_models as DP
import predictions_analysis as PA
from download_prediction import download_pred_Xtest


def xgboost_reg(X_train,Y_train,X_vali,Y_vali,params) : 
    xgboost = xgb.XGBRegressor(**params)
    xgboost.fit(X_train, Y_train)
    test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
    
    for i, y_pred in enumerate(xgboost.staged_predict(X_vali)):
        test_score[i] = xgboost.loss_(Y_vali, y_pred)
    
    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("Deviance")
    plt.plot(
        np.arange(params["n_estimators"]) + 1,
        xgboost.train_score_,
        "b-",
        label="Training Set Deviance",
    )
    plt.plot(
        np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
    )
    
    #plt.axvline((np.arange(params["n_estimators"]) + 1)[np.argmin(test_score)])
    plt.axvline(200)
    plt.legend(loc="upper right")
    plt.xlabel("xgboost Iterations")
    plt.ylabel("Deviance/squared error loss")
    fig.tight_layout()
    plt.savefig(os.path.join(PATH_IMAGE,'xgb_optimization.png'))
    plt.show()
    
def Optimize_xgb(X_train, Y_train) :
    tps0=time.perf_counter()
    param=[{"n_estimators":[500,700, 900, 1100], "max_depth": np.arange(5,25,5), "learning_rate":[0.01,0.05,0.1]}] #optimisation de m
    xgboost= GridSearchCV(xgb.XGBRegressor(n_estimators=500),param,cv=5,n_jobs=1, verbose = 3) #Permet d'afficher les tests déjà réalisés
    xgbOpt=xgboost.fit(X_train, Y_train)
    tps1=time.perf_counter()
    print("Temps execution en sec :",(tps1 - tps0))
    
    # paramètre optimal
    param_opt = xgbOpt.best_params_
    print("Erreur la moins élevée = %f, Meilleurs paramètres = %s" % (1. -xgbOpt.best_score_,xgbOpt.best_params_)) #1-R^2
    return param_opt


def Model_xgb(X_train,Y_train,param_opt):
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
    xgb_opt = xgb.XGBRegressor(**all_param)
    xgb_opt.fit(X_train, Y_train)
    #rf_opt.save(os.path.join(directory_weigths,file_name)) #### a modifier 
    tps1=time.perf_counter()
    print("Temps execution en sec :",(tps1 - tps0))
    return xgb_opt

def Predict_validation_set(X_vali,Y_vali,xgb_opt,model_name='xgboost'):
    prev = xgb_opt.predict(X_vali)
    prev_detransfo = np.exp(prev)
    Y_vali_detransfo = np.exp(Y_vali)
    scores = PA.compute_scores(Y_vali_detransfo,prev_detransfo)
    PA.plot_pred_obs(Y_vali_detransfo,prev_detransfo,model_name)
    PA.scatterplot_residuals(Y_vali_detransfo,prev_detransfo,model_name)
    PA.histogram_residuals(Y_vali_detransfo,prev_detransfo,model_name)
    return scores

def Predict_test_set(X_test,xgb_opt):
    prev_test = xgb_opt.predict(X_test)
    prev_test = pd.DataFrame(np.exp(prev_test),columns=['price'])
    download_pred_Xtest(np.array(prev_test).flatten(),'prediction_xgboost')


def main_xgboost(param_opt=0) :
    
    data,Y,var_quant,var_quali,var_quali_to_encode = DL.main_load_data()
    X_train_renorm,Y_train,X_vali_renorm,Y_vali,X_test_renorm = DP.main_prepare_train_vali_data(data,Y,var_quant,var_quali,var_quali_to_encode)
    model_name = 'xgboost'
    if param_opt == 0 :
        param_opt = Optimize_xgb(X_train_renorm, Y_train)
    xgb_opt = Model_xgb(X_train_renorm, Y_train, param_opt)
    Predict_validation_set(X_vali_renorm,Y_vali,xgb_opt,model_name)
    Predict_test_set(X_test_renorm,xgb_opt)


data,Y,var_quant,var_quali,var_quali_to_encode = DL.main_load_data()
x_train,y_train,x_vali,y_vali,x_test = DP.main_prepare_train_vali_data(data,Y,var_quant,var_quali,var_quali_to_encode)

x_train = x_train.values
y_train = y_train.values
x_test = x_test.values

SEED=0

rf=RandomForestRegressor(
    n_estimators=800,
    criterion='ls',
    max_depth=None,
    #min_samples_split=2,
    #min_samples_leaf=2,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=SEED,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None
)


ada = AdaBoostRegressor(
    n_estimators=900,
    learning_rate=0.05,
    loss='square',
    random_state=SEED
)

lgbm= LGBMRegressor(
    objective="regression",
    num_leaves=32,
    max_depth=5,
    random_state=SEED,
    learning_rate=0.12, 
    metric='mse',
    n_jobs=2,
    n_estimators=1500,
    colsample_bytree=0.7,
    subsample=0.7,
    verbose=-1
)

cb= CatBoostRegressor(
    iterations=1500,
    depth=7,
    learning_rate=0.3, 
    l2_leaf_reg=0.45,
    silent=True,
    random_seed=0 #42069
)

xgb= XGBRegressor(
    n_estimators=1100,
    max_depth=15,
    n_jobs=-2,
    #booster='gbtree',
    enable_categorical= False, #True,
    colsample_bytree=0.7,
    subsample=0.7,
    learning_rate=0.05,
    random_state=0 #42069
)

metamodel_level1=RidgeCV()

estimators_level0 =[
    #('xgb',xgb),
    ('random_forest', rf),
    ('lgbm',lgbm),
    ('catboost',cb),
]

reg=StackingRegressor(estimators=estimators_level0,n_jobs=-2,final_estimator=metamodel_level1,cv=5,verbose=-1)
#reg=StackingRegressor(estimators=estimators_level0,final_estimator=metamodel_level1,cv=5)

#names=[]
Valid_RMSE = []
predictions = []
kf=KFold(random_state=0,shuffle=True)

for train_index, test_index in kf.split(x_train):
    print("TRAIN : ",train_index, "TEST : ", test_index)
    x_train_fold, x_test_fold = x_train[train_index], x_train[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
    #model = reg.fit(x_train_fold,np.log(y_train_fold))
    reg.fit(x_train_fold,np.log(y_train_fold))
    #names.append(model)
    pred_valid =np.exp(reg.predict(x_test_fold))
    Valid_RMSE.append(root_mean_squared_log_error(y_test_fold,pred_valid))
    predictions.append(np.exp(reg.predict(data_test)))
    print("Valid_RMSE: ", Valid_RMSE)
    
print("Mean RMSE ", np.mean(Valid_RMSE))
















