#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 18:58:04 2022

@author: julie
"""
import numpy as np
import pandas as pd
import os
from math import log,sqrt,exp

import random
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler 

def transform_Y(Y) :
    # Transform the data according to the distirbution identified when making
    # data analysis of the requests
    Y_mod = pd.Series(Y).map(lambda x: log(x)).to_numpy()
    return Y_mod

def prepare_input_data(data,var_quant,var_quali): 
    # 1. quantitative variables : 
    # Transform the data according to the distirbution identified when making
    # data analysis of the requests
    X_quant = data[var_quant]
    X_quant["stock_mod"]=X_quant["stock"].map(lambda x: sqrt(x))
    X_quant.drop("stock",axis=1,inplace=True)
    var_quant_new = ["date","stock_mod"]
    
    # 2. qualitative variables : 
    for var in var_quali :
        data[var]=pd.Categorical(data[var],ordered=False)
    X_dum = pd.get_dummies(data[var_quali],drop_first=True)
    var_dum = X_dum.columns
    
    X = pd.concat([data[['order_requests','avatar_id','hotel_id']],X_quant,X_dum],axis=1)
    return X,var_quant_new,var_dum

def renorm_var_quant(X,var_quant,var_dum,scaler=0) :
    # Renormalization of quantitative variables : 
    if scaler == 0:
        scaler = StandardScaler()  
        scaler.fit(X[var_quant])
    X_quant = pd.DataFrame(scaler.transform(X[var_quant]),columns=var_quant)
    X = pd.concat([X[['order_requests','avatar_id','hotel_id']], X_quant, X[var_dum]],axis=1)
    return X,scaler

def define_order_requests(data) :
    order = np.zeros(len(data.index),dtype=int)
    # reference for the beginning = the first line of the dataframe
    ref_line = 0
    order[ref_line] = 1
    for line in range(0,len(data.index)):
        request_line = data.loc[line,['avatar_id','city','date']].to_numpy()
        request_ref_line = data.loc[ref_line,['avatar_id','city','date']].to_numpy() 
        if (request_line == request_ref_line).all() :
            order[line] = order[ref_line]
        else : 
            order[line] = order[ref_line]+1
            ref_line = line
    data['order_requests'] = order
    return data

def split_train_vali(X,Y):
    # 20% des avatar_ID seront dans le test set et 80% dans le train set
    random.seed(0)
    alpha = 0.8
    ind_user = np.unique(X['avatar_id'])
    n = len(ind_user)
    random.shuffle(ind_user)
    ind_train = ind_user[:int(alpha*n)]
    ind_vali = ind_user[int(alpha*n):]
    
    # sélection des lignes correspondantes dans le dataframe 
    X_train = X.loc[X['avatar_id'].isin(ind_train)].reset_index()
    X_vali = X.loc[X['avatar_id'].isin(ind_vali)].reset_index()
    
    # séparation de la colonne des outputs en training et validation : 
    indX_train = X.index[X['avatar_id'].isin(ind_train)]
    Y_train = Y[indX_train]
    indX_vali = X.index[X['avatar_id'].isin(ind_vali)]
    Y_vali = Y[indX_vali]
    return X_train,X_vali,Y_train,Y_vali
    

def main_prepare_train_vali_data(data,Y,var_quant,var_quali) :
    # split train/validation :
    Y_mod = transform_Y(Y)
    X_train,X_vali,Y_train,Y_vali = split_train_vali(data,Y_mod)
    
    # define order request : 
    X_train = define_order_requests(X_train)
    X_vali = define_order_requests(X_vali)
    
    # Prepare input and output data : 
    X_train,var_quant_new,var_dum =  prepare_input_data(X_train,var_quant,var_quali)
    X_vali,_,_ =  prepare_input_data(X_vali,var_quant,var_quali)
    
    # renormalize
    X_train_renorm, scalerX = renorm_var_quant(X_train,var_quant_new,var_dum)
    X_vali_renorm, _ = renorm_var_quant(X_vali,var_quant_new,var_dum,scalerX)
      
    X_train_renorm.drop('avatar_id',axis=1,inplace=True)
    X_vali_renorm.drop('avatar_id',axis=1,inplace=True)
    return X_train_renorm,Y_train,X_vali_renorm,Y_vali


def main_prepare_train_test_data(data,Y,var_quant,var_quali) :
    # Prepare X_train : -------------------------------------------------------
    Y_train = transform_Y(Y)
    data = define_order_requests(data) # define order request
    X_train,var_quant_new,var_dum =  prepare_input_data(data,var_quant,var_quali)
    X_train_renorm, scalerX = renorm_var_quant(X_train,var_quant_new,var_dum) # renormalize
    
    # Prepare X_test : --------------------------------------------------------
    data_test = pd.read_csv('../Test_set_analysis/test_set_complet.csv')
    X_test,_,_ = prepare_input_data(data_test,var_quant,var_quali)
    X_test_renorm, _ = renorm_var_quant(X_test,var_quant_new,var_dum)
    
    X_train_renorm.drop('avatar_id',axis=1,inplace=True)
    X_test_renorm.drop('avatar_id',axis=1,inplace=True)
    return X_train,Y_train,X_test
    
