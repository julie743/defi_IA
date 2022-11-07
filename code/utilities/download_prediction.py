#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 23:24:05 2022

@author: julie
"""
import pandas as pd
import numpy as np

def download_pred_Xtest(Y_pred,name_model) :
    df_pred = pd.DataFrame({'price' : Y_pred})
    df_pred['index'] = df_pred.index
    df_pred = df_pred[['index','price']]
    df_pred.to_csv('../../predictions/'+name_model+'.csv',index=False)