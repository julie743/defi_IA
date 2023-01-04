# Use an official Python runtime as a parent image
FROM ubuntu:latest

# Set up the local zone and UTC info
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ/etc/localtime && echo $TZ > /etc/timezone

# build instructions 
RUN apt-get update && apt-get install -y \
python3-pip

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt


