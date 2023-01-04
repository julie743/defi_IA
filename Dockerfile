# Use an official Python runtime as a parent image
FROM python:3.8

RUN pwd
RUN ls
# Set the working directory to /app
WORKDIR ./

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
#RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["python", "./app/code/gradio/gradio_main.py"]
