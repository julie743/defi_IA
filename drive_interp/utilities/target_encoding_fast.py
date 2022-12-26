import pandas as pd
from category_encoders import TargetEncoder

def target_encoding_all(data, Y, quantitative_vars) : 
    
    '''
    ==> Performs the target encoding on several variables using the Python library
    
    input : takes a list of all the quantitative variables we wish to encode ; data : Y (the label)
    
    Mdifies the data by adding a new column with the target code for 
    each modality of the quantitative variable
    '''

    for var in quantitative_vars : 
        encoder = TargetEncoder()
        new_var = var + "_new"
        
        if data[var].dtype != 'int64' : 
            data[new_var] = encoder.fit_transform(data[var], pd.DataFrame(Y)) 
        else : 
            data[new_var] = encoder.fit_transform((data[var]).astype(str), pd.DataFrame(Y)) 

# Example of use
#quantitative_vars = ['city','language','hotel_id']
#target_encoding_all(data, Y, quantitative_vars)


