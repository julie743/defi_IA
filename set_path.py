#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 12:48:46 2023

@author: julie
"""

import os
PATH_PROJECT =  os.getcwd()

print('Get current working directory : ', PATH_PROJECT)
PATH_CODE = os.path.join(PATH_PROJECT,'code/')

PATH_MODELS = os.path.join(PATH_CODE,'models/')
PATH_UTILITIES = os.path.join(PATH_CODE,'utilities/')
PATH_REQUEST_ANALYSIS = os.path.join(PATH_CODE,'request_data_analysis/')
PATH_TEST_SET_ANALYSIS = os.path.join(PATH_CODE,'Test_set_analysis/')
PATH_GRADIO = os.path.join(PATH_CODE,'gradio/')
ALL_PATHS = [PATH_MODELS,PATH_UTILITIES,PATH_REQUEST_ANALYSIS, PATH_TEST_SET_ANALYSIS, PATH_GRADIO]


file_name = "set_path.txt"

for folder in ALL_PATHS : 
    file_path  = open(os.path.join(folder,file_name), "w+")
    file_path.write(PATH_PROJECT)
    file_path.close()
