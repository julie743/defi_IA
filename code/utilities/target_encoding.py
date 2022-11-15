import numpy as np
import pandas as pd

def target_encoding1(data, Y, quantitative_var) : 
    
    '''
    input : takes a quantitative variable
   
    output : returns the table of correspondences between a quantitative 
             variable to encode and its code according to the target encoding method
    
    Also, it modifies the data by adding a new column with the target code for 
    each modality of the quantitative variable
    '''
    
    table_corresp = pd.DataFrame(columns = [quantitative_var, 'Code'])
    mod = np.unique(data[quantitative_var])
    # Corresponds to the codes given to each modality of a quantitative variable
    table_corresp[quantitative_var] = mod 
    new_var = quantitative_var + '_new'
    
    for modality in mod : 
        ind = np.where(data[quantitative_var]==modality)
        prices = Y[ind]
        target_code = np.mean(prices)
        table_corresp.loc[table_corresp[quantitative_var] == modality, 'Code'] = target_code
        data.loc[data[quantitative_var] == modality, new_var] = target_code 
        
    return table_corresp

def target_encoding_all(data, Y, quantitative_vars) : 
    
    '''
    ==> Performs the target encoding on several variables 
    
    input : takes a list of all the quantitative variables we wish to encode
    
    output : returns the table of correspondences of all the quantitative variables
    
    Also, it modifies the data by adding a new column with the target code for 
    each modality of the quantitative variable
    '''
    
    table_corresp = pd.DataFrame()
    for var in quantitative_vars : 
        tmp = target_encoding1(data, Y, var)
        table_corresp = pd.concat([tmp,table_corresp], axis=1)
    return table_corresp

# Example of use
#quantitative_vars = ['city','language','hotel_id']
#table_corresp = target_encoding_all(data, Y, quantitative_vars)



    
    
