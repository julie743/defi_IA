import os
import time
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV

from catboost import CatBoostRegressor
import pickle

PATH_PROJECT = open("set_path.txt",'r').readlines()[0]
PATH_IMAGE = os.path.join(PATH_PROJECT,'images')
PATH_UTILITIES = os.path.join(PATH_PROJECT,'code/utilities')

os.chdir(PATH_UTILITIES)
import data_loading as DL
import data_preparation_for_models as DP
from predict_validation_and_test import Predict_validation_set, Predict_test_set
os.chdir(PATH_PROJECT)

# models : -------------------------------------------------------------------
nn = MLPRegressor(random_state=1, max_iter=1500, alpha=0.5,hidden_layer_sizes=18)

cb= CatBoostRegressor(n_estimators=2478,
                      learning_rate=0.29014147234242005,
                      max_depth=10)

metamodel_level1=RidgeCV()
#metamodel_level1=RandomForestRegressor()

estimators_level0 =[
    ('neural_network',nn),
    ('catboost',cb),
]
# ----------------------------------------------------------------------------

def model_stacking(X_train, Y_train) :
    '''
    model that combines the neural network model and the catboost model in 
    a stacking
    
    Parameters
    ----------
    X_train : pandas.dataframe
       training dataset input.
    Y_train : pandas.dataframe
       training dataset output.
    
    Returns
    -------
    ensemble : sklearn model
        voting regression model.
    '''
    
    tps0=time.perf_counter()
    reg=StackingRegressor(estimators=estimators_level0,n_jobs=-1,final_estimator=metamodel_level1,cv=5,verbose=-1)
    reg.fit(X_train, Y_train)
    tps1=time.perf_counter()
    print("Temps execution pour l'entrainement en sec :",(tps1 - tps0))
    return reg

def main_stacking() :
    '''
    main function : calls the previous functions in the correct order to 
    perform all the computations for the neural network algorithm

    Parameters
    ----------
    None.

    Returns
    -------
    None.

    '''
    
    #data,Y,var_quant,var_quali,var_quali_to_encode = DL.main_load_data()
    data,Y,var_quant,var_quali,var_quali_to_encode = DL.main_load_data2()
    X_train,X_vali,X_train_renorm,Y_train,X_vali_renorm,Y_vali,X_test_renorm = DP.main_prepare_train_vali_data(data,Y,var_quant,var_quali,var_quali_to_encode)
    model_name = 'stacking adversarial'
    
    # create model
    reg = model_stacking(X_train_renorm, Y_train)
    
    #download model's weigths
    path_weigths = os.path.join(PATH_PROJECT,'weigths','stacking_adversarial.sav')
    pickle.dump(reg, open(path_weigths, 'wb'))
    
    # predict validation and test sets
    Predict_validation_set(X_vali,X_vali_renorm,Y_vali,reg,var_quant,var_quali,model_name)
    Predict_test_set(X_test_renorm,reg,model_name)

#main_stacking()






