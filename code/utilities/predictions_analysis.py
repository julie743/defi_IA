#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 23:07:12 2022

@author: julie
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from data_loading import PATH_PROJECT
import os

PATH_IMAGE = os.path.join(PATH_PROJECT,'images')

'''
mkdir : check if a directory exists and create it if it does not
Input : 
  - directory : name of the directory to create
'''
def mkdir(directory) : 
    '''
    check if a directory exists, create it otherwise

    Parameters
    ----------
    directory : string
        path of the directory.

    Returns
    -------
    None.

    '''
    
    
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_pred_obs(Y_true,Y_pred,model_name): 
    '''
    plot the graph of predictions against true values

    Parameters
    ----------
    Y_true : np.darray
        real ouput data (price) of the validation set.
    Y_pred : np.darray
        predicted ouput data (price) of the validation set..
    model_name : string
        => For plots and recording files.

    Returns
    -------
    None.

    '''
    
    plt.figure(figsize=(5,5))
    plt.plot(Y_true,Y_pred,"o",markersize = 0.4)
    plt.xlabel("predicted price")
    plt.ylabel("observed price")
    plt.title("prediction " + model_name)
    file_name = 'prediction_'+model_name.replace(' ', '_')+'.png'
    sub_directory = os.path.join(PATH_IMAGE,model_name.replace(' ', '_'))
    mkdir(sub_directory)
    plt.savefig(os.path.join(sub_directory,file_name))
    plt.show()

def scatterplot_residuals(Y_true,Y_pred,model_name):
    '''
    plot the graph of residuals against predicted values

    Parameters
    ----------
   Y_true : np.darray
       real ouput data (price) of the validation set.
   Y_pred : np.darray
       predicted ouput data (price) of the validation set..
   model_name : string
       => For plots and recording files.

    Returns
    -------
    None.

    '''
    
    plt.figure(figsize=(5,5))
    plt.plot(Y_pred,Y_true-Y_pred,"o",markersize = 0.4)
    plt.xlabel(u"predicted values")
    plt.ylabel(u"residuals")
    plt.title("Residuals " + model_name) 
    plt.hlines(0,0,3)
    file_name = 'residuals_'+model_name.replace(' ', '_')+'.png'
    sub_directory = os.path.join(PATH_IMAGE,model_name.replace(' ', '_'))
    mkdir(sub_directory)
    plt.savefig(os.path.join(sub_directory,file_name))
    plt.show()
    
def histogram_residuals(Y_true,Y_pred, model_name):
    '''
    plot the histogram of residuals 

    Parameters
    ----------
   Y_true : np.darray
       real ouput data (price) of the validation set.
   Y_pred : np.darray
       predicted ouput data (price) of the validation set.
   model_name : string
       => For plots and recording files.

    Returns
    -------
    None.

    '''
    
    
    plt.figure(figsize=(10,5))
    plt.hist(Y_true-Y_pred,bins=20)
    plt.title('Histogram of residuals ' + model_name)
    plt.xlabel('residuals values')
    plt.ylabel('number of predictions')
    file_name = 'histogram_residuals_'+model_name.replace(' ', '_')+'.png'
    sub_directory = os.path.join(PATH_IMAGE,model_name.replace(' ', '_'))
    mkdir(sub_directory)
    plt.savefig(os.path.join(sub_directory,file_name))
    plt.show()
    
    
def outliers_prediction(Y_true,Y_pred) :
    '''
    get the indices of the residual outliers in order to analyze later the data
    that we have for these points that we did not manage to predict well.

    Parameters
    ----------
    Y_true : np.darray
        real ouput data (price) of the validation set.
    Y_pred : np.darray
        predicted ouput data (price) of the validation set.

    Returns
    -------
    outliers_inf : np.darray
        indices of negative outliers in the prediction.
    outliers_sup : np.darray
        indices of positive outliers in the prediction.

    '''
    res = Y_true-Y_pred
    q025 = np.quantile(res,0.25)
    q075 = np.quantile(res,0.75)
    lower_bound = q025 - 1.5*(q075-q025)
    upper_bound = q075 + 1.5*(q075-q025)
    outliers_inf = np.where(res<lower_bound)[0]
    outliers_sup = np.where(res>upper_bound)[0]
    return outliers_inf, outliers_sup

def histogram_outliers(Y_true,Y_pred,outliers_inf, outliers_sup, model_name) :
    '''
    plot the histogram of negative and positive outliers to see their repartition
    

    Parameters
    ----------
    Y_true : np.darray
        real ouput data (price) of the validation set.
    Y_pred : np.darray
        predicted ouput data (price) of the validation set.
    outliers_inf : np.darray
        indices of negative outliers in the prediction.
    outliers_sup : np.darray
        indices of positive outliers in the prediction.
    model_name : string
        => For plots and recording files.

    Returns
    -------
    None.

    '''
    plt.figure(figsize=(10,5))
    plt.hist(Y_true[outliers_inf]-Y_pred[outliers_inf],bins=20)
    plt.title('Histogram of negative outlier residuals ' + model_name)
    plt.xlabel('residuals values')
    plt.ylabel('number of predictions')
    file_name = 'histogram_residuals_outliersInf_'+model_name.replace(' ', '_')+'.png'
    sub_directory = os.path.join(PATH_IMAGE,model_name.replace(' ', '_'))
    mkdir(sub_directory)
    plt.savefig(os.path.join(sub_directory,file_name))
    plt.show()
    
    plt.figure(figsize=(10,5))
    plt.hist(Y_true[outliers_sup]-Y_pred[outliers_sup],bins=20)
    plt.title('Histogram of positive outlier residuals ' + model_name)
    plt.xlabel('residuals values')
    plt.ylabel('number of predictions')
    file_name = 'histogram_residuals_outliersSup_'+model_name.replace(' ', '_')+'.png'
    sub_directory = os.path.join(PATH_IMAGE,model_name.replace(' ', '_'))
    mkdir(sub_directory)
    plt.savefig(os.path.join(sub_directory,file_name))
    plt.show()

def analysis_var_quali_outliers(data,outliers_inf,outliers_sup,var_quali, model_name):
    '''

    Parameters
    ----------
    data : pd.DataFrame
        dataframe of validation dataset (X_vali).
    outliers_inf : np.darray
        indices of negative outliers in the prediction.
    outliers_sup : np.darray
        indices of positive outliers in the prediction.
    var_quant : list of string
        qualitative variables 
    model_name : string
        => For plots and recording files.

    Returns
    -------
    None.

    '''
    fig = plt.figure(figsize=(15, 10))
    i = 1
    for var in var_quali : 
        plt.subplot(2, 4, i)
        data.loc[outliers_inf,var].value_counts().plot.pie(subplots=True, title="pie chart of negative outliers (prediction)", autopct='%1.1f%%')
        i+=1
    file_name = 'piechart_residuals_outliersInf_'+model_name.replace(' ', '_')+'.png'
    sub_directory = os.path.join(PATH_IMAGE,model_name.replace(' ', '_'))
    mkdir(sub_directory)
    plt.savefig(os.path.join(sub_directory,file_name))
    plt.show()
        
    fig = plt.figure(figsize=(15, 10))
    i = 1
    for var in var_quali : 
        plt.subplot(2, 4, i)
        data.loc[outliers_sup,var].value_counts().plot.pie(subplots=True, title="pie chart of positive outliers (prediction)", autopct='%1.1f%%')
        i+=1
    file_name = 'piechart_residuals_outliersSup_'+model_name.replace(' ', '_')+'.png'
    sub_directory = os.path.join(PATH_IMAGE,model_name.replace(' ', '_'))
    mkdir(sub_directory)
    plt.savefig(os.path.join(sub_directory,file_name))
    plt.show()
    
def analysis_var_quanti_outliers(data,Y_true,Y_pred,outliers_inf,outliers_sup,var_quant, model_name):
    '''
    

    Parameters
    ----------
    data : pd.DataFrame
        dataframe of validation dataset (X_vali).
    Y_true : np.darray
        real ouput data (price) of the validation set.
    Y_pred : np.darray
        predicted ouput data (price) of the validation set.
    outliers_inf : np.darray
        indices of negative outliers in the prediction.
    outliers_sup : np.darray
        indices of positive outliers in the prediction.
    var_quant : list of string
        quantitative variables 
    model_name : string
       => For plots and recording files.

    Returns
    -------
    None.

    '''
    res_inf = Y_true[outliers_inf]-Y_pred[outliers_inf]
    ref_sup =  Y_true[outliers_sup]-Y_pred[outliers_sup]
    plt.figure(figsize=(15, 10))
    i = 1
    for var in var_quant : 
        plt.subplot(1, len(var_quant), i)
        plt.plot(data.loc[outliers_inf,var], res_inf, "o", markersize = 0.5, color = 'firebrick', label = 'negative outliers')
        plt.plot(data.loc[outliers_sup,var], ref_sup, "o", markersize = 0.5, color = 'royalblue', label = 'positive outliers')
        plt.xlabel(var)
        plt.ylabel('prediction residuals')
        plt.legend()
        i += 1
    plt.suptitle('Analysis of quantitative variables values of outlier residuals ' + model_name)
    file_name = 'scatterplot_var_quanti_residual_outliers_'+model_name.replace(' ', '_')+'.png'
    sub_directory = os.path.join(PATH_IMAGE,model_name.replace(' ', '_'))
    mkdir(sub_directory)
    plt.savefig(os.path.join(sub_directory,file_name))
    plt.show()
    
    
def compute_scores(Y_true,Y_pred):
    '''
    compute and printmetrics on the prediction of the validation set
    Parameters
    ----------
   Y_true : np.darray
       real ouput data (price) of the validation set.
   Y_pred : np.darray
       predicted ouput data (price) of the validation set..
   model_name : string
       => For plots and recording files.

    Returns
    -------
    dic : dictionary of the obtained scores on the prediction 
    
    '''
    
    
    dic = {'rmse': np.sqrt(mean_squared_error(Y_pred,Y_true)),
           'mae' : mean_absolute_error(Y_pred,Y_true),
           'r2'  : r2_score(Y_pred,Y_true) }
    print("RMSE = ", dic['rmse'])
    print("R2 = ", dic['r2'])
    print("MAE = ", dic['mae'])
    return dic





    