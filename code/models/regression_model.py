import os
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV, LassoLarsCV
from itertools import cycle
import pandas as pd
import time
import matplotlib.pyplot as plt

#Path Eva : 'C:/Users/evaet/Documents/5A/defi_IA/'
#Path Julie : '/home/julie/Documents/cours/5A/IAF/defi_IA'
#PATH_PROJECT = '/home/julie/Documents/cours/5A/IAF/defi_IA'
PATH_PROJECT = "../.."
PATH_IMAGE = os.path.join(PATH_PROJECT,'images')
PATH_UTILITIES = os.path.join(PATH_PROJECT,'code/utilities')
os.chdir(PATH_UTILITIES)
import data_loading as DL
import data_preparation_for_models as DP
import predictions_analysis as PA
from download_prediction import download_pred_Xtest
from predict_validation_and_test import Predict_validation_set, Predict_test_set

# modele regression linéaire
def Model_reg(X_train,Y_train,alpha=0):
    '''
    Create regression model with given alpha and fit it on the training data

    Parameters
    ----------
    X_train : pandas.dataframe
        training dataset input.
    Y_train : pandas.dataframe
        training dataset output.
    alpha : float
        Parameter of Lasso penalization. default = 0 for simple linear regression

    Returns
    -------
    regLin : regression model fit on the data

    '''
    
    tps0=time.perf_counter()
    regLin = linear_model.Lasso(alpha)
    regLin.fit(X_train,Y_train)
    tps1=time.perf_counter()
    print("Temps execution en sec pour l'entrainement :",(tps1 - tps0))
    return regLin

def Optimize_regLasso(X_train,Y_train,list_param):
    '''
    Perfom GridSearchCV to find the best alpha (Lasso penalization term)


    Parameters
    ----------
    X_train : pandas.dataframe
        training dataset input.
    Y_train : pandas.dataframe
        training dataset output.
    list_param : list of floats
        possible values of alpha.

    Returns
    -------
    alpha_opt : float
        optimal value of alpha.

    '''
    
    tps0=time.perf_counter()
    param=[{"alpha":list_param}]
    regLasso = GridSearchCV(linear_model.Lasso(), param,cv=5,n_jobs=-1)
    regLassOpt=regLasso.fit(X_train, Y_train)
    # paramètre optimal
    alpha_opt = regLassOpt.best_params_["alpha"]
    print("Meilleur R2 = %f, Meilleur paramètre = %s" % (regLassOpt.best_score_,regLassOpt.best_params_))
    tps1=time.perf_counter()
    print("Temps execution en sec pour l'optimisation ':",(tps1 - tps0))
    return alpha_opt


def plot_lasso_coeff(regLasso,X_train_renorm,model_name):
    '''
    Plot Lasso coefficient by order of importance

    Parameters
    ----------
    regLasso : Lasso model fit on the training data
    X_train_renorm : pandas.dataframe
        training dataset input.
    model_name : string
        => For plots and recording files.

    Returns
    -------
    None.

    '''
    coef = pd.Series(regLasso.coef_, index = X_train_renorm.columns)
    print("Lasso conserve " + str(sum(coef != 0)) + 
      " variables et en supprime " +  str(sum(coef == 0)))
    imp_coef = coef.sort_values()
    plt.figure()
    plt.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind = "barh")
    plt.title(u"Coefficients du modèle lasso")
    sub_directory = os.path.join(PATH_IMAGE,model_name.replace(' ', '_'))
    PA.mkdir(sub_directory)
    file_name = 'prediction_'+model_name.replace(' ', '_')+'.png'
    plt.savefig(os.path.join(sub_directory,file_name))
    plt.show()

def main_Linear():
    '''
    main function : calls the previous functions in the correct order to 
    perform all the computations for the linear regression

    Parameters
    ----------
    None

    Returns
    -------
    None.

    '''
    
    #data,Y,var_quant,var_quali,var_quali_to_encode = DL.main_load_data()
    data,Y,var_quant,var_quali,var_quali_to_encode = DL.main_load_data2()
    X_train,X_vali,X_train_renorm,Y_train,X_vali_renorm,Y_vali,X_test_renorm = DP.main_prepare_train_vali_data(data,Y,var_quant,var_quali,var_quali_to_encode)
    #model_name = 'linear regression'
    model_name = 'linear regression adversarial'
    
    regLin = Model_reg(X_train_renorm,Y_train)
    scores = Predict_validation_set(X_vali,X_vali_renorm,Y_vali,regLin,var_quant,var_quali,model_name)
    Predict_test_set(X_test_renorm,regLin,model_name)
    
def main_Lasso():
    '''
    main function : calls the previous functions in the correct order to 
    perform all the computations for the Lasso regression

    Parameters
    ----------
    None

    Returns
    -------
    None.

    '''
    
    #data,Y,var_quant,var_quali,var_quali_to_encode = DL.main_load_data()
    data,Y,var_quant,var_quali,var_quali_to_encode = DL.main_load_data2()
    X_train,X_vali,X_train_renorm,Y_train,X_vali_renorm,Y_vali,X_test_renorm = DP.main_prepare_train_vali_data(data,Y,var_quant,var_quali,var_quali_to_encode)
    #model_name = 'Lasso regression model'
    model_name = 'Lasso regression model adversarial'
    
    alpha_opt = Optimize_regLasso(X_train_renorm,Y_train,[0.01,0.02,0.04,0.06,0.08,0.1,0.2])
    regLasso = Model_reg(X_train_renorm,Y_train,alpha_opt)
    
    plot_lasso_coeff(regLasso,X_train_renorm,model_name) # plot coeff 
    
    scores = Predict_validation_set(X_vali,X_vali_renorm,Y_vali,regLasso,var_quant,var_quali,model_name)
    Predict_test_set(X_test_renorm,regLasso,model_name)
    
#main_Lasso()    
