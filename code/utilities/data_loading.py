#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 17:44:25 2022

@author: julie
"""
import numpy as np
import pandas as pd
import os

PATH_PROJECT = '/home/julie/Documents/cours/5A/IAF/defi_IA'
PATH_DATA = os.path.join(PATH_PROJECT,'data')

def load_data() : 
    path = os.path.join(PATH_DATA,'results_requests')
    directories = os.listdir(path)
    data = pd.DataFrame()
    for d in directories :
        files = os.listdir(os.path.join(path,d))
        for f in files :
            path_file = os.path.join(path,d,f)
            data = pd.concat([data,pd.read_csv(path_file)],ignore_index=True)
    Y = data['price'].to_numpy()
    data.drop('price',axis=1,inplace=True)
    return data, Y

def add_hotel_features(data) :
    path = os.path.join(PATH_DATA,'all_data')
    features_hotels = pd.read_csv(os.path.join(path,'features_hotels.csv'))
    
    # take hotels in the order in which they are found in the request dataset and concatenate the two dataframes
    hotels_in_order = features_hotels.loc[data['hotel_id']]
    
    # check that the city between the two dataframes are matching : 
    cities_in_order = np.array(hotels_in_order['hotel_id'])
    cities_request = np.array(data['hotel_id'])
    diff_cities = len(np.where(cities_in_order!=cities_request)[0])
    
    # if cities are the same we can delete one of the columns :
    if diff_cities == 0 : 
        hotels_in_order.drop('city',axis=1,inplace=True)
    else :
        raise Exception("some hotels are not in the requested city")
    
    # finally we concatenate the two dataframes:
    hotels_in_order.drop('hotel_id',axis=1,inplace=True)
    hotels_in_order.reset_index(inplace=True)
    data = pd.concat([data,hotels_in_order],axis=1)
    data.drop('index',axis=1,inplace=True)
    data.to_csv(os.path.join(path,'requetes_total.csv'),index=False)
    return data

def var_types(data): 
    var_quant = ["date","stock"]
    #var_quali = ["city","language", "mobile","group","brand","parking","pool","children_policy"]
    #var_quali = ["city","language", "mobile",'hotel_id',"group","brand","parking","pool","children_policy"]
    var_quali = ["mobile","group","brand","parking","pool","children_policy"]
    var_quali_to_encode = ["city","hotel_id","language"]
    return var_quant,var_quali, var_quali_to_encode

def main_load_data():
    data, Y = load_data() 
    data = add_hotel_features(data)
    var_quant,var_quali,var_quali_to_encode = var_types(data)
    return data,Y,var_quant,var_quali,var_quali_to_encode

data,Y,var_quant,var_quali,var_quali_to_encode = main_load_data()
    
    
    
    
    