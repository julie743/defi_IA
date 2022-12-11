import pandas as pd
import numpy as np 

# NAIVE STACKING  

# Load the predictions made on different maching learning algorithms 
pred_xgb = pd.read_csv('predictions/prediction_xgboost.csv')
pred_lgbm = pd.read_csv('predictions/prediction_lgbm.csv')
pred_boosting = pd.read_csv('predictions/prediction_boosting.csv')
pred_lineareg = pd.read_csv('predictions/prediction_regLineaire.csv')
pred_regression_tree = pd.read_csv('predictions/prediction_random_forest.csv')

# Choose the ones we want to keep for stacking 
predictions = [pred_lgbm["price"],pred_boosting["price"], pred_regression_tree["price"]] 

# Formatting of new predictions for kaggle 
price = pd.DataFrame(np.mean(predictions,axis=0))
index = pd.DataFrame(pred_xgb["index"])
final_predictions = pd.DataFrame(pd.concat([index, price], join = 'outer', axis = 1))
final_predictions.columns = ['index', 'price']
final_predictions.to_csv('predictions/naive_stacking.csv', index=False)