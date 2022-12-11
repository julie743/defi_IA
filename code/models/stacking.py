import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet, RidgeCV, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

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

# Data loading  
data,Y,var_quant,var_quali,var_quali_to_encode = DL.main_load_data()
X_train,X_vali,X_train_renorm,Y_train,X_vali_renorm,Y_vali,X_test_renorm =  DP.main_prepare_train_vali_data(data,Y,var_quant,var_quali,var_quali_to_encode)
X_train_renorm = X_train_renorm.values
X_test_renorm = X_test_renorm.values

# STACKING  
#The parameters must be tuned for each algorithm before running this code 

SEED=0

rf=RandomForestRegressor(
    n_estimators=900,
    #criterion='ls',
    max_depth=None,
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
    colsample_bytree=0.7,
    subsample=0.7,
    learning_rate=0.05,
    random_state=0 
)

# 
metamodel_level1=RidgeCV()

#
estimators_level0 =[
    ('xgb',xgb),
    #('random_forest', rf),
    ('lgbm',lgbm),
    #('catboost',cb),
]

reg=StackingRegressor(estimators=estimators_level0,n_jobs=-2,final_estimator=metamodel_level1,cv=5,verbose=-1)
Valid_RMSE = []
predictions = []
kf=KFold(random_state=0,shuffle=True)

for train_index, test_index in kf.split(X_train_renorm):
    print("TRAIN : ",train_index, "TEST : ", test_index)
    x_train_fold, x_test_fold = X_train_renorm[train_index], X_train_renorm[test_index]
    y_train_fold, y_test_fold = Y_train[train_index], Y_train[test_index]
    reg.fit(x_train_fold,np.log(y_train_fold))
    pred_valid =np.exp(reg.predict(x_test_fold))
    Valid_RMSE.append(np.sqrt(mean_squared_error(y_test_fold,pred_valid)))
    predictions.append(np.exp(reg.predict(X_test_renorm)))
    print("Valid_RMSE: ", Valid_RMSE)
    
print("Mean RMSE ", np.mean(Valid_RMSE))

#print(predictions)
#final_predictions=np.mean(predictions,axis=0)
#print(final_predictions)















