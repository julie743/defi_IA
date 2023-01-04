import pandas as pd
import numpy as np
import os 
PATH_PROJECT = open("set_path.txt",'r').readlines()[0]

def download_pred_Xtest(Y_pred,name_model) :
    '''
    download the prediction on the test set under the correct format to be 
    received on the kaggle website

    Parameters
    ----------
    Y_pred : np.darray
        predict output data (price)
    name_model : string
        name of the model that we used on the data => for files name

    Returns
    -------
    None.

    '''
    
    df_pred = pd.DataFrame({'price' : Y_pred})
    df_pred['index'] = df_pred.index
    df_pred = df_pred[['index','price']]
    path_PRED = os.path.join(PATH_PROJECT,'predictions/')
    file_name = name_model + '.csv'
    df_pred.to_csv(os.path.join(path_PRED,file_name),index=False)