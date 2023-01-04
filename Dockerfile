# Use an official Python runtime as a parent image
FROM ubuntu:latest

# Set up the local zone and UTC info
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ/etc/localtime && echo $TZ > /etc/timezone

# build instructions 
RUN apt-get update && apt-get install -y python3-pip

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install numpy==1.23.4
RUN pip install tensorflow==2.11.0
RUN pip install pandas==1.5.1
RUN pip install scikit-learn==1.1.3
RUN pip install category_encoders==2.5.1.post0
RUN pip install catboost==1.1.1
RUN pip install gradio==3.15.0
RUN pip install matplotlib==3.6.2
