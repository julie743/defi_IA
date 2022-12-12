import numpy as np
import pandas as pd
import os

#Path Julie : '/home/julie/Documents/cours/5A/IAF/defi_IA'
#Path Eva : 'C:/Users/evaet/Documents/5A/defi_IA/' 
PATH_PROJECT = '/home/julie/Documents/cours/5A/IAF/defi_IA'
PATH_DATA = os.path.join(PATH_PROJECT,'data/')

def load_data() : 
    
    '''
    load the request data
    Parameters
    ----------
    None

    Returns
    -------
    data : pandas.DataFrame
        dataframe of the requests data

    '''
    path = os.path.join(PATH_DATA,'results_requests/')
    directories = os.listdir(path)
    data = pd.DataFrame()
    for d in directories :
        files = os.listdir(os.path.join(path,d))
        for f in files :
            path_file = os.path.join(path,d,f)
            data = pd.concat([data,pd.read_csv(path_file)],ignore_index=True)
    Y = data['price'].to_numpy()
    data.drop('price',axis=1,inplace=True)
    return data, Y

def add_hotel_features(data) :
    '''
    Add hotel features to every line of the dataset

    Parameters
    ----------
    data : pandas.DataFrame
        raw dataframe of the requests data

    Raises
    ------
    Exception
        If the city given by the result of the request is not the same as the 
        city requested by the user, it raises an exception 
        "some hotels are not in the requested city".

    Returns
    -------
    data : pandas.DataFrame
        dataframe of the requests data with hotels features.

    '''
    
    path = os.path.join(PATH_DATA,'all_data')
    features_hotels = pd.read_csv(os.path.join(path,'features_hotels.csv'))
    
    # take hotels in the order in which they are found in the request dataset and concatenate the two dataframes
    hotels_in_order = features_hotels.loc[data['hotel_id']]
    
    # check that the city between the two dataframes are matching : 
    cities_in_order = np.array(hotels_in_order['hotel_id'])
    cities_request = np.array(data['hotel_id'])
    diff_cities = len(np.where(cities_in_order!=cities_request)[0])
    
    # if cities are the same we can delete one of the columns :
    if diff_cities == 0 : 
        hotels_in_order.drop('city',axis=1,inplace=True)
    else :
        raise Exception("some hotels are not in the requested city")
    
    # finally we concatenate the two dataframes:
    hotels_in_order.drop('hotel_id',axis=1,inplace=True)
    hotels_in_order.reset_index(inplace=True)
    data = pd.concat([data,hotels_in_order],axis=1)
    data.drop('index',axis=1,inplace=True)
    data.to_csv(os.path.join(path,'requetes_total.csv'),index=False)
    return data

def var_types(data): 
    '''
    Define the type of variables in the dataset

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe of the requests data with hotels features.

    Returns
    -------
    var_quant : list of string
        quantitative variables.
    var_quali : list of string
        qualitative variables that will be encoded with dummies.
    var_quali_to_encode : list of string
        qualitative variables that will be encoded with target encoding.

    '''
    var_quant = ["date","stock"]
    var_quali = ["city","language", "mobile","group","brand","parking","pool","children_policy"]
    #var_quali = ["city","language", "mobile",'hotel_id',"group","brand","parking","pool","children_policy"]
    #var_quali = ["mobile","parking","pool","children_policy"]
    #var_quali_to_encode = ["city","hotel_id","language","group","brand"]
    var_quali_to_encode = ["hotel_id"]
    return var_quant,var_quali, var_quali_to_encode

def main_load_data():
    '''
    main function : calls the previous functions in the correct order to 
    perform the complete loading of the data

    Parameters
    ----------
    None

    Returns
    -------
    data : pandas.DataFrame
        dataframe of the requests data with hotels features.
    Y : np.darray
        output (price) of the request data
    var_quant : list of string
        quantitative variables.
    var_quali : list of string
        qualitative variables that will be encoded with dummies.
    var_quali_to_encode : list of string
        qualitative variables that will be encoded with target encoding.

    '''
    data, Y = load_data() 
    data = add_hotel_features(data)
    var_quant,var_quali,var_quali_to_encode = var_types(data)
    return data,Y,var_quant,var_quali,var_quali_to_encode

#data,Y,var_quant,var_quali,var_quali_to_encode = main_load_data()
    


    
    
    