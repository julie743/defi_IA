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
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_pred_obs(Y_true,Y_pred,model_name): 
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
    
def compute_scores(Y_true,Y_pred):
    dic = {'rmse': np.sqrt(mean_squared_error(Y_pred,Y_true)),
           'mae' : mean_absolute_error(Y_pred,Y_true),
           'r2'  : r2_score(Y_pred,Y_true) }
    print("RMSE = ", dic['rmse'])
    print("R2 = ", dic['r2'])
    print("MAE = ", dic['mae'])
    return dic



    