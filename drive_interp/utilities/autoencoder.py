#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 00:15:10 2022

@author: julie
"""

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import os
tensorflow.__version__

PATH_PROJECT = '/home/julie/Documents/cours/5A/IAF/defi_IA'
PATH_IMAGE = os.path.join(PATH_PROJECT,'images')

PATH_UTILITIES = os.path.join(PATH_PROJECT,'code/utilities')
os.chdir(PATH_UTILITIES)

import data_loading as DL
import data_preparation_for_models as DP
import predictions_analysis as PA
from download_prediction import download_pred_Xtest

data,Y,var_quant,var_quali,var_quali_to_encode = DL.main_load_data()
X_train_renorm,Y_train,X_vali_renorm,Y_vali,X_test_renorm = DP.main_prepare_train_vali_data(data,Y,var_quant,var_quali,var_quali_to_encode)
n_var = len(X_train_renorm.columns)


encoder = km.Sequential(name='encoder')
encoder.add(kl.Conv1D(8,3, activation='relu', padding='same',dilation_rate=2, input_shape=(n_var,1,1)))
encoder.add(kl.MaxPooling1D(2))
encoder.add(kl.Conv1D(4,3, activation='relu', padding='same',dilation_rate=2))
encoder.add(kl.MaxPooling1D(2))
encoder.add(kl.AveragePooling1D())
encoder.add(kl.Flatten())
encoder.add(kl.Dense(2))

decoder = km.Sequential(name='decoder')
decoder.add(kl.Dense(64))
decoder.add(kl.Reshape((16,4)))
decoder.add(kl.Conv1D(4,1,strides=1, activation='relu', padding='same'))
decoder.add(kl.UpSampling1D(2))
decoder.add(kl.Conv1D(8,1,strides=1, activation='relu', padding='same'))
decoder.add(kl.UpSampling1D(2))
decoder.add(kl.UpSampling1D(2))
decoder.add(kl.Conv1D(1,1,strides=1, activation='sigmoid', padding='same'))
