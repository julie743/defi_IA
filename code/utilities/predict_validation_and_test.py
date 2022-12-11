import predictions_analysis as PA
from download_prediction import download_pred_Xtest
import numpy as np
import pandas as pd

# prediction échantillon de validation ----------------------------------------
def Predict_validation_set(X_vali,X_vali_mod,Y_vali,model,var_quant,var_quali,model_name):
    ''''
    Predict the validation set using an optimal model. Plots and records
    the results

    Parameters
    ----------
    X_vali : pandas.dataframe
        validation dataset input.
    Y_vali : pandas.dataframe
        validation dataset output.
    model : optimal model fit on the training data
    model_name : string
        => For plots and recording files.
        
    Returns
    -------
    scores : dic
        dictionary of metrics computed on the validation data (MAE, RMSE, R2)

    '''
    prev=model.predict(X_vali_mod)
    prev_detransfo = np.exp(prev)
    Y_vali_detransfo = np.exp(Y_vali)
    scores = PA.compute_scores(Y_vali_detransfo,prev_detransfo)
    PA.plot_pred_obs(Y_vali_detransfo,prev_detransfo,model_name)
    PA.scatterplot_residuals(Y_vali_detransfo,prev_detransfo,model_name)
    PA.histogram_residuals(Y_vali_detransfo,prev_detransfo,model_name)
    outliers_inf, outliers_sup = PA.outliers_prediction(Y_vali_detransfo,prev_detransfo)
    PA.histogram_outliers(Y_vali_detransfo,prev_detransfo,outliers_inf, outliers_sup, model_name) 
    PA.analysis_var_quali_outliers(X_vali,outliers_inf,outliers_sup,var_quali, model_name)
    PA.analysis_var_quanti_outliers(X_vali,Y_vali_detransfo,prev_detransfo,outliers_inf,outliers_sup,var_quant, model_name)
    return scores

# prediction échantillon de test ----------------------------------------------
def Predict_test_set(X_test,model,model_name):
    '''
    Predict the test set and record the results 

    Parameters
    ----------
    X_test : pandas.dataframe
        test dataset input.
    model : optimal model fit on the training data

    Returns
    -------
    None.

    '''
    prev_test = model.predict(X_test)
    prev_test = pd.DataFrame(np.exp(prev_test),columns=['price'])
    download_pred_Xtest(np.array(prev_test).flatten(),'prediction_'+model_name.replace(' ', '_')+'.png')