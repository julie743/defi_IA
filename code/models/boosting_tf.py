import numpy as np
import pandas as pd
import time
import os

import tensorflow_decision_forests as tfdf
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

PATH_PROJECT = '/home/julie/Documents/cours/5A/IAF/defi_IA'
PATH_IMAGE = os.path.join(PATH_PROJECT,'images')
PATH_UTILITIES = os.path.join(PATH_PROJECT,'code/utilities')
os.chdir(PATH_UTILITIES)

import data_loading as DL
import data_preparation_for_models as DP
import predictions_analysis as PA
from download_prediction import download_pred_Xtest

model_name = 'gradient boosted trees model'
data,Y,var_quant,var_quali,var_quali_to_encode = DL.main_load_data()
X_train_renorm,Y_train,X_vali_renorm,Y_vali,X_test_renorm = DP.main_prepare_train_vali_data(data,Y,var_quant,var_quali,var_quali_to_encode)

data_labelled = pd.concat([data,pd.DataFrame(Y,columns=["price_mod"])],axis=1)

model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
data_tf = tfdf.keras.pd_dataframe_to_tf_dataset(data_labelled,task=tfdf.keras.Task.REGRESSION, label="price_mod")
model.fit(data_tf)
print(model.summary())

'''

tuner = tfdf.tuner.RandomSearch(num_trials=20)

# Hyper-parameters to optimize.
tuner.discret("max_depth", [5,10,15,20])

model = tfdf.keras.GradientBoostedTreesModel(tuner=tuner)
model.fit(data_tf)

'''

















