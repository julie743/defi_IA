#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 12:46:14 2022

@author: julie
"""
import pandas as pd

def add_cost_living(data) :
    '''
    add the cost of living to the features. The information was found on the 
    following link :
    https://fr.numbeo.com/co%C3%BBt-de-la-vie/classements-par-pays

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe of the input data

    Returns
    -------
    data : TYPE
        DESCRIPTION.

    '''
    cost = {"amsterdam": 67.71,
            "copenhagen": 73.09,
            "madrid": 47.51,
            "paris" : 65.55,
            "rome": 58.47,
            "sofia": 37.58,
            "valletta": 57.71,
            "vienna" : 64.11,
            "vilnius" : 43.1}
    
    df_cost = pd.DataFrame.from_dict(cost,orient='index',columns=['cost_life'])
    
    cities_in_order = df_cost.loc[data['city']]
    cities_in_order.reset_index(inplace=True)
    cities_in_order.drop('index',axis=1,inplace=True)
    data = pd.concat([data,cities_in_order],axis=1)
    
    return data

   
    
    