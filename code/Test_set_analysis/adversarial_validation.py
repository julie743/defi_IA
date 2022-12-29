from xgboost import XGBClassifier
import pandas as pd
import os
import sys
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

#PATH_PROJECT = 'C:\\Users\\alexa\\OneDrive\\Documents\\INSA\\5A\\Defi_IA\\defi_IA\\code\\'
#PATH_PROJECT = 'C:/Users/evaet/Documents/5A/defi_IA/code'
PATH_PROJECT = '../..'
PATH_UTILITIES = os.path.join(PATH_PROJECT, 'utilities')
sys.path.append(PATH_UTILITIES)
os.chdir(PATH_UTILITIES)

import data_loading as DL
import data_preparation_for_models as DP
import predictions_analysis as PA
from download_prediction import download_pred_Xtest

datatest = pd.read_csv("../../data/all_data/test_set_complet.csv")
datatest = datatest.drop(['order_requests', 'avatar_id'], axis=1)
datatest["is_test"] = 1

# init dataset and batches
datatrain, Y, var_quant, var_quali, var_quali_to_encode = DL.main_load_data()
Ybis = pd.DataFrame(dict({'price':Y}))
#datatrain = datatrain.drop(['avatar_id'], axis=1)
datatrain["is_test"] = 0
datatrain = pd.concat([datatrain, Ybis], axis=1)

datatrain_batch = np.array_split(datatrain, 100)
fd_dataframe = pd.DataFrame([])
estimated_accuracy = 0

stock_prices = pd.DataFrame([])

# to iterate
for i in range(100):
    # get batch and fit classifier model
    train_sample = datatrain_batch[i].drop(['price','avatar_id'], axis=1)
    frames = [train_sample, datatest]
    data = pd.concat(frames,axis=0)
    data_X = data.drop(["is_test"], axis=1)
    data_Y = pd.DataFrame(data.is_test)
    data_X = data_X.drop(["city", "language", "hotel_id", "group", "brand"], axis=1)
    model = XGBClassifier()
    model.fit(data_X, data_Y)

    # make predictions
    y_pred = model.predict(data_X)
    predictions = [round(value) for value in y_pred]
    predVSreal = pd.DataFrame({'predictions': y_pred,
                              'real': data_Y.to_numpy().reshape(-1)})
    
    # update accuracy
    fpr, tpr, thresholds = roc_curve(data_Y, predictions)
    accuracy = auc(fpr, tpr)
    estimated_accuracy += accuracy
    # get False Discoveries (predicted test, were train)
    idxs_of_fd = predVSreal.index[(predVSreal.predictions==1) & (predVSreal.real==0)].to_list()
    fd_list = train_sample.iloc[idxs_of_fd,]
    stock_prices = pd.concat((stock_prices, datatrain_batch[i].iloc[idxs_of_fd,]))
    fd_dataframe = pd.concat((fd_dataframe, fd_list), axis=0)
    
print("estimated accuracy of train/test classification (Monte Carlo): ", estimated_accuracy/100)

# Save selected data to csv
stock_prices.to_csv("../../data/adversarial_validation_data/selected_train_data_with_price.csv", index=False)