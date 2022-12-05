import numpy as np
import pandas as pd
import os

#Path Eva : 'C:/Users/evaet/Documents/5A/defi_IA/' 
#Path Julie : '/home/julie/Documents/cours/5A/IAF/defi_IA'
PATH_PROJECT = 'C:/Users/evaet/Documents/5A/defi_IA/' 
PATH_DATA = os.path.join(PATH_PROJECT,'data/')
PATH_UTILITIES = os.path.join(PATH_PROJECT,'code/utilities')
PATH_STOCK_REQUESTS = os.path.join(PATH_DATA,'stock_requetes/')

#fonctionne que pour nb_requests = 1 pour l'instant
def remove_duplicates(new_requests, old_requests, criteria, nb_requests) :
    '''
    Removes the requests that have already been done the previous weeks 

    Parameters
    ----------
    new_requests : pandas.dataframe
        new requests in which we want to remove the requests we already made 
    old_requests : pandas.dataframe
        requests done so far 
    criteria : list 
        column names we base on to remove the duplicated requests
    nb_requests : integer 
        scale at which we look at the duplicates
        
    Returns
    -------
    final_requests : pandas.dataframe
        final requests we are allowed to use

    '''
    colnames = ['avatar_name', 'language', 'city', 'date', 'mobile','count_request']
    #Assigns the number of requests made by each avatar 
    tmp_count = old_requests.groupby('avatar_name')['avatar_name'].transform('count')
    old_requests = pd.concat([old_requests,tmp_count], axis=1)
    old_requests.columns = colnames
    
    #Same manipulation for the new requests
    tmp_count = new_requests.groupby('avatar_name')['avatar_name'].transform('count')
    new_requests = pd.concat([new_requests,tmp_count], axis=1)
    new_requests.columns = colnames
    
    old_requests_nb = old_requests.iloc[np.where(old_requests['count_request']==nb_requests)]
    new_requests_nb = new_requests.iloc[np.where(new_requests['count_request']==1)]
    
    #We remove all the elements from old_requests_nb in to_remove_nb
    final_1 = new_requests_nb.merge(old_requests_nb, on=criteria, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    final_1 = final_1.drop(['avatar_name_y'], axis=1)
    final_1.columns = colnames 
    
    final = new_requests.iloc[np.where(new_requests['count_request']!=nb_requests)]
    final = pd.concat([final,final_1], axis=0)
    
    return final.drop(['count_request'], axis=1) 
 
#TEST
"""requests_1 = pd.read_csv(PATH_STOCK_REQUESTS+'tmp/data_requests_total_1.csv') 
requests_1_4 = pd.read_csv(PATH_STOCK_REQUESTS+'tmp/data_requests_total_1_4.csv') 
old_requests = requests_1[0:1000]
new_requests = requests_1[1000:2000]
criteria = ['language', 'city', 'date', 'mobile','count_request']
remove_duplicates(new_requests, old_requests, criteria, 1)"""

def load_previous_requests() : 
    path = os.path.join(PATH_DATA,'stock_requetes/')
    files = os.listdir(path) 
    requests_all = pd.DataFrame()
    for f in files :
        path_file = os.path.join(path,f)
        requests_all = pd.concat([requests_all,pd.read_csv(path_file)],ignore_index=True)
    return requests_all

#Requets of the week :
tmp_requests = pd.read_csv(os.path.join(PATH_DATA,'tmp/data_requests6.csv'))

#Load the requests from the previous weeks 
requests_all = load_previous_requests()

#Apply the function 
criteria = ['language', 'city', 'date', 'mobile','count_request']
nb_requests = 1
final_requests = remove_duplicates(tmp_requests, requests_all, criteria, nb_requests)
final_requests.to_csv(os.path.join(PATH_DATA,'stock_requetes/data_requests6.csv'),index=False)


