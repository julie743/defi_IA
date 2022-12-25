import gradio as gr
import pandas as pd
import numpy as np 
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
import os

#Path Eva : 'C:/Users/evaet/Documents/5A/defi_IA/' 
#Path Julie : '/home/julie/Documents/cours/5A/IAF/defi_IA'
PATH_PROJECT = 'C:/Users/evaet/Documents/5A/defi_IA/' 

#Weigths 
PATH_WEIGTHS = os.path.join(PATH_PROJECT,'weigths')
filename = 'finalized_model.sav'

PATH_UTILITIES = os.path.join(PATH_PROJECT,'code/utilities')
os.chdir(PATH_UTILITIES)

import data_loading as DL
import data_preparation_for_models as DP
import predictions_analysis as PA
from download_prediction import download_pred_Xtest

#--------------------- Information regarding the request ----------------------
cities = ["amsterdam","copenhagen","madrid","paris","rome","sofia","valletta","vienna","vilnius"]
languages = ["austrian", "belgian", "bulgarian", "croatian", "cypriot", "czech", "danish", "dutch", "estonian", "finnish", "french", "german", "greek", "hungarian", "irish", "italian", "latvian", "lithuanian", "luxembourgish", "maltese", "polish", "portuguese", "romanian", "slovakian", "slovene", "spanish", "swedish"]


city = gr.inputs.Radio(choices=cities, label="Select a city")
language =  gr.inputs.Radio(choices=languages, label="Select a language")
mobile =  gr.inputs.Slider(minimum=0, maximum=1, step=1, label="Select a device - 0 (computer) & 1 (mobile phone)")
date =  gr.inputs.Slider(minimum=0, maximum=44, step=1, label="Select a date")
hotel_id =  gr.inputs.Slider(minimum=0, maximum=998, step=1, label="Select the hotel_id")
stock = gr.inputs.Slider(minimum=0, maximum=200, step=1, label="Select the stock") #to change 
parking = gr.inputs.Slider(minimum=0, maximum=1, step=1, label="Parking available") 
pool = gr.inputs.Slider(minimum=0, maximum=1, step=1, label="Swimming pool available")
children_policy = gr.inputs.Slider(minimum=0, maximum=2, step=1, label="Select the children policy of the hotel")

#-------------------------------- Prediction ----------------------------------
def predict(city,language,mobile,date,hotel_id,stock,group,brand,parking,pool,children_policy):
    #Download of the best model weights 
    os.chdir(PATH_WEIGTHS)
    model = pickle.load(open(filename, 'rb'))
    
    #Formatting the data into a dataframe 
    avatar_id = 5555555555555
    new = dict(avatar_id=avatar_id, city=city, date=date, language=language, mobile=mobile, hotel_id=hotel_id, stock=stock, group=group, brand=brand, parking=parking, pool=pool, children_policy=children_policy)
    #new = dict(avatar_id=avatar_id, city="paris", date=40, language="dutch", mobile=0, hotel_id=853, stock=110, group="Chillton Worldwide", brand="Tripletree", parking=1, pool=0, children_policy=0)
    new = pd.DataFrame(new, index=[0])
    data,Y,var_quant,var_quali,var_quali_to_encode = DL.main_load_data()
    frames = [data, new]
    result = pd.concat(frames)
    X_train,X_vali,X_train_renorm,Y_train,X_vali_renorm,Y_vali,X_test_renorm = DP.main_prepare_train_vali_data(result,Y,var_quant,var_quali,var_quali_to_encode)
    index = X_train[X_train["avatar_id"]==avatar_id].index

    #Making a prediction 
    prediction = model.predict(X_train_renorm.iloc[index])
    
    return prediction 

#------------------------------- Lauch Gradio ---------------------------------
if __name__=='__main__':
    interface = gr.Interface(
    fn=predict,
    inputs=[city,language,mobile,date,hotel_id,stock,"text","text",parking,pool,children_policy],
    outputs=["text"],   
    title="Hotel price prediction - defi IA 2023",
    description="To predict the price of a hotel, please enter the necessary information"
).launch(debug=True, share=True)


"""A finaliser : 
- Format écriture brand et group ==> majuscule pour la premiere lettre puis minuscule 
- Valeur max du stock 
- Interpretabilité 
"""















