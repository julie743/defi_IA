#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 18:58:04 2022

@author: julie
"""
import numpy as np
import pandas as pd
from math import log,sqrt
import copy
import os

import random 
from sklearn.preprocessing import StandardScaler 
from data_loading import PATH_DATA

def transform_Y(Y) :
    # Transform the data according to the distribution identified when making
    # data analysis of the requests
    Y_mod = pd.Series(Y).map(lambda x: log(x)).to_numpy()
    return Y_mod

def define_order_requests(data,var_quant) :
    consecutive_identical = (data[['avatar_id','city','date']] != data[['avatar_id','city','date']].shift(1)).cumsum()
    order = np.max(consecutive_identical,axis=1)
    data['order_requests'] =  order.values.squeeze()
    var_quant_new = copy.deepcopy(var_quant)
    var_quant_new.append('order_requests')
    return data, var_quant_new

def prepare_input_data(data,var_quant,var_quali): 
    # 1. quantitative variables : 
    # Transform the data according to the distirbution identified when making
    # data analysis of the requests
    X_quant = data[var_quant]
    X_quant["stock_mod"]=X_quant["stock"].map(lambda x: sqrt(x))
    X_quant.drop("stock",axis=1,inplace=True)
    var_quant_new = ["date","stock_mod",'order_requests']
    
    # 2. qualitative variables : 
    for var in var_quali :
        data[var]=pd.Categorical(data[var],ordered=False)
    X_dum = pd.get_dummies(data[var_quali],drop_first=True)
    var_dum = X_dum.columns
    
    #X = pd.concat([data[['order_requests','avatar_id','hotel_id']],X_quant,X_dum],axis=1)
    X = pd.concat([data[['avatar_id']],X_quant,X_dum],axis=1)
    return X,var_quant_new,var_dum

def renorm_var_quant(X,var_quant,var_dum,scaler=0) :
    # Renormalization of quantitative variables : 
    if scaler == 0:
        scaler = StandardScaler()  
        scaler.fit(X[var_quant])
    X_quant = pd.DataFrame(scaler.transform(X[var_quant]),columns=var_quant)
    #X = pd.concat([X[['order_requests','avatar_id','hotel_id']], X_quant, X[var_dum]],axis=1)
    X = pd.concat([X[['avatar_id']], X_quant, X[var_dum]],axis=1)
    return X,scaler

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
    X_train = X.loc[X['avatar_id'].isin(ind_train)].reset_index(drop=True)
    X_vali = X.loc[X['avatar_id'].isin(ind_vali)].reset_index(drop=True)
    
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
    X_train,var_quant_new = define_order_requests(X_train,var_quant)
    X_vali,_ = define_order_requests(X_vali,var_quant)
    
    # Prepare input and output data : 
    X_train,var_quant_last,var_dum =  prepare_input_data(X_train,var_quant_new,var_quali)
    X_vali,_,_ =  prepare_input_data(X_vali,var_quant_new,var_quali)
    
    # renormalize
    X_train_renorm, scalerX = renorm_var_quant(X_train,var_quant_last,var_dum)
    X_vali_renorm, _ = renorm_var_quant(X_vali,var_quant_last,var_dum,scalerX)
      
    X_train_renorm.drop('avatar_id',axis=1,inplace=True)
    X_vali_renorm.drop('avatar_id',axis=1,inplace=True)
    
    # charge test set 
    data_test = pd.read_csv(os.path.join(PATH_DATA,'all_data','test_set_complet.csv'))
    X_test,_,_ = prepare_input_data(data_test,var_quant_new,var_quali)
    # some columns that are in the training set are not in the test set because
    #  the test set does not contain all the hotel_id for example => we had 
    # them all set up to 0
    missing_col = [c for c in X_train_renorm.columns if not c in X_test.columns] 
    X_test[missing_col] = 0
    X_test_renorm, _ = renorm_var_quant(X_test,var_quant_last,var_dum)
    X_test_renorm.drop('avatar_id',axis=1,inplace=True)
    return X_train_renorm,Y_train,X_vali_renorm,Y_vali,X_test_renorm


def main_prepare_train_test_data(data,Y,var_quant,var_quali) :
    # Prepare X_train : -------------------------------------------------------
    Y_train = transform_Y(Y)
    data,var_quant = define_order_requests(data,var_quant) # define order request
    X_train,var_quant_new,var_dum =  prepare_input_data(data,var_quant,var_quali)
    X_train_renorm, scalerX = renorm_var_quant(X_train,var_quant_new,var_dum) # renormalize
    
    # Prepare X_test : --------------------------------------------------------
    data_test = pd.read_csv(os.path.join(PATH_DATA,'all_data','test_set_complet.csv'))
    X_test,_,_ = prepare_input_data(data_test,var_quant,var_quali)
    # some columns that are in the training set are not in the test set because
    #  the test set does not contain all the hotel_id for example => we had 
    # them all set up to 0
    missing_col = [c for c in X_train.columns if not c in X_test.columns] 
    X_test[missing_col] = 0
    X_test_renorm, _ = renorm_var_quant(X_test,var_quant_new,var_dum)
    
    X_train_renorm.drop('avatar_id',axis=1,inplace=True)
    X_test_renorm.drop('avatar_id',axis=1,inplace=True)
    return X_train_renorm,Y_train,X_test_renorm
    