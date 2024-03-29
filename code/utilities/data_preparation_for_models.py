import numpy as np
import pandas as pd
from math import log,sqrt
import copy
import os

import random 
from sklearn.preprocessing import StandardScaler 
from data_loading import PATH_DATA
from category_encoders import TargetEncoder

from create_new_features import add_cost_living

def transform_Y(Y) :
    
    '''
    Transform the data according to the distribution identified when making
    ata analysis of the requests
    
    Parameters
    ----------
    Y : np.darray
        output (price) of the request data

    Returns
    -------
    Y_mod : np.darray
        modified output (price) with a log transformation

    '''
    Y_mod = pd.Series(Y).map(lambda x: log(x)).to_numpy()
    return Y_mod

def define_order_requests(data,var_quant) :
    '''
    define the order in which the requests have been made on the website

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe of the request data
    var_quant : list of string
        quantitative variables.

    Returns
    -------
    data : pandas.DataFrame
        dataframe of the requests data with the new column 'order_requests'
    var_quant_new : list of string
        quantitative variables with the new 'order_requests' feature

    '''
    
    consecutive_identical = (data[['avatar_id','city','date']] != data[['avatar_id','city','date']].shift(1)).cumsum()
    order = np.max(consecutive_identical,axis=1)
    data['order_requests'] =  order.values.squeeze()
    var_quant_new = copy.deepcopy(var_quant)
    var_quant_new.append('order_requests')
    return data, var_quant_new

def target_encoding_all(data, quantitative_vars, encoder_list = {}, Y=0) : 
    
    '''
    ==> Performs the target encoding on several variables using the Python library
    
    Parameters
    ----------
    data : pandas.DataFrame
        dataframe of the request data
    quantitative_vars : list of string
        quantitative variables.
    encoder_list : dic of fitted encoder on the training datset
        Default = {} : in this case no encoder has been defined yet, so we have
        to fit one on the training data
    Y : np.darray
        Output 

    Returns
    -------
    data : pandas.DataFrame
        dataframe of the requests data with new encoded variables
    var_quant_new : list of string
        quantitative variables with the new 'order_requests' feature
    Y : np.darray
        output (price) of the request data

    '''
    new_var_quali_to_encode  = []
    if len(encoder_list) == 0 : # case when we treat the training data
        for var in quantitative_vars : 
            encoder = TargetEncoder()
            new_var = var + "_target"
            new_var_quali_to_encode.append(var + "_target")
            
            if data[var].dtype != 'int64' : 
                encoder.fit(data[var], pd.DataFrame(Y)) 
            else : 
                encoder.fit((data[var]).astype(str), pd.DataFrame(Y)) 
            data[new_var] = encoder.transform(data[var])
            encoder_list[new_var] = encoder
    
    else : # case when we treat the test data : re-use the encoder of the training dataset
        for var in quantitative_vars : 
            new_var = var + "_target"
            new_var_quali_to_encode.append(new_var)
            encoder = encoder_list[new_var]
            data[new_var] = encoder.transform(data[var]) 
        
    return new_var_quali_to_encode, encoder_list

def renorm_var_quant(X,var_quant,var_dum,scaler=0) :
    '''
    renormalize quantitative variables

    Parameters
    ----------
    X : pandas.DataFrame
        dataframe of the input data 
    var_quant : list of string
        quantitative variables 
    var_dum : list of string
        dummy variables 
    scaler : scaler fitted on the training data
        DESCRIPTION. The default is 0. In this case, no scaler has been fitted
        yet => we have to define one

    Returns
    -------
    X : pandas.DataFrame
        dataframe of the input rescaled data 
    scaler : scaler fitted on the training data

    '''
    # Renormalization of quantitative variables : 
    if scaler == 0:
        scaler = StandardScaler()  
        scaler.fit(X[var_quant])
    X_quant = pd.DataFrame(scaler.transform(X[var_quant]),columns=var_quant)
    #X = pd.concat([X[['order_requests','avatar_id','hotel_id']], X_quant, X[var_dum]],axis=1)
    X = pd.concat([X[['avatar_id']], X_quant, X[var_dum]],axis=1)
    return X,scaler

def prepare_input_data(data,var_quant,var_quali,var_quali_to_encode, encoder_list = {}, Y=0): 
    '''
    Prepare the input data : change types, apply transformations

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe of the input data 
    var_quant : list of string
        quantitative variables.
    var_quali : list of string
        qualitative variables that will be encoded with dummies.
    var_quali_to_encode : list of string
        qualitative variables that will be encoded with target encoding.
    encoder_list : dic of fitted encoder on the training datset
        Default = {} : in this case no encoder has been defined yet, so we have
        to fit one on the training data
    Y : np.darray
        Output data (price)

    Returns
    -------
    X : pandas.DataFrame
        dataframe of the modified input data 
    var_quant_new : list of string
        quantitative variables.
    var_dum : list of string
        dummy variables.
    encoder_list : dic of fitted encoder on the training datset

    '''
    data_mod = copy.deepcopy(data)
    # 1. qualitative variables : 
    for var in var_quali :
        data_mod[var]=pd.Categorical(data_mod[var],ordered=False)
    X_dum = pd.get_dummies(data_mod[var_quali],drop_first=True) # A CHANGER AVEC LE TARGET ENCODEUR
    var_dum = X_dum.columns
    
    # 2. qualitative variables that must be encoded with the target-encoding
    new_var_quali_to_encode, encoder_list = target_encoding_all(data_mod, var_quali_to_encode, encoder_list, Y)
    var_quant_new = copy.deepcopy(var_quant)
    var_quant_new.extend(new_var_quali_to_encode)

    # 3. quantitative variables : 
    # Transform the data according to the distribution identified when making
    # data analysis of the requests
    X_quant = data_mod[var_quant_new]
    X_quant["stock_mod"]=X_quant["stock"].map(lambda x: sqrt(x))
    X_quant.drop("stock",axis=1,inplace=True)
    var_quant_new.remove("stock")
    var_quant_new.extend(["stock_mod"])
    
    #X = pd.concat([data_mod[['order_requests','avatar_id','hotel_id']],X_quant,X_dum],axis=1)
    X = pd.concat([data_mod[['avatar_id']],X_quant,X_dum],axis=1)
    
    return X,var_quant_new,var_dum, encoder_list

def split_train_vali(X,Y):
    '''
    split the request data into training and validation set. The spli is made 
    such that 80% of the avatar_id are in the training set and the 20% left
    are in the test set. One avatar can belong to one datasets only (the lines
    corresponding to his requests cannot be spread accross two different 
    datasets)

    Parameters
    ----------
    X : pandas.DataFrame
        dataframe of the input data 
    Y : np.darray
        Output data (price)

    Returns
    -------
    X_train : pandas.dataframe
        training dataset input.
    Y_train : np.darray
        training utput data (price)
    X_vali : pandas.dataframe
        validation dataset input.
    Y_vali : np.darray
        validation utput data (price)

    '''
    
    
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
    

def main_prepare_train_vali_data(data,Y,var_quant,var_quali,var_quali_to_encode) :
    '''
    main function : calls the previous functions in the correct order to 
    perform all the computations for the linear regression


    Parameters
    ----------
    data : pandas.DataFrame
        dataframe of the input data 
    Y : np.darray
        Output data (price)
    var_quant : list of string
        quantitative variables.
    var_quali : list of string
        qualitative variables that will be encoded with dummies.
    var_quali_to_encode : list of string
        qualitative variables that will be encoded with target encoding.

    Returns
    -------
    X_train_renorm : pandas.dataframe
        training dataset input.
    Y_train : np.darray
        training output data (price)
    X_vali_renorm : pandas.dataframe
        validation dataset input.
    Y_vali : np.darray
        validation utput data (price)
    X_test_renorm : np.darray
        test utput data (price)

    '''

    # split train/validation :
    Y_mod = transform_Y(Y)
    X_train,X_vali,Y_train,Y_vali = split_train_vali(data,Y_mod)
    
    # add cost of living : 
    X_train = add_cost_living(X_train)
    X_vali = add_cost_living(X_vali)
    var_quant.append("cost_life")

    # define order request : 
    X_train,var_quant_new = define_order_requests(X_train,var_quant)
    X_vali,_ = define_order_requests(X_vali,var_quant)
    
    # Prepare input and output data : 
    X_train_mod_type,var_quant_last,var_dum,encoder_list =  prepare_input_data(X_train,var_quant_new,var_quali,var_quali_to_encode, encoder_list = {}, Y=Y_train)
    X_vali_mod_type,_,_,_ =  prepare_input_data(X_vali,var_quant_new,var_quali,var_quali_to_encode, encoder_list = encoder_list,Y = Y_vali)
   
    # renormalize
    X_train_renorm, scalerX = renorm_var_quant(X_train_mod_type,var_quant_last,var_dum) ##PB    
    X_vali_renorm, _ = renorm_var_quant(X_vali_mod_type,var_quant_last,var_dum,scalerX)
  
    X_train_renorm.drop('avatar_id',axis=1,inplace=True)
    X_vali_renorm.drop('avatar_id',axis=1,inplace=True)

    # charge test set 
    data_test = pd.read_csv(os.path.join(PATH_DATA,'all_data','test_set_complet.csv'))
    data_test = add_cost_living(data_test)
    var_quant.append('order_requests')
    X_test,_,_,_ = prepare_input_data(data_test,var_quant_new,var_quali,var_quali_to_encode, encoder_list = encoder_list, Y =0)

    X_test_renorm, _ = renorm_var_quant(X_test,var_quant_last,var_dum)
    X_test_renorm.drop('avatar_id',axis=1,inplace=True)
    
    return X_train,X_vali,X_train_renorm,Y_train,X_vali_renorm,Y_vali,X_test_renorm

#X_train,X_vali,X_train_renorm,Y_train,X_vali_renorm,Y_vali,X_test_renorm = main_prepare_train_vali_data(data,Y,var_quant,var_quali,var_quali_to_encode)

'''
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
'''    
