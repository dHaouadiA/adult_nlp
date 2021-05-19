# FROM python:3.7-slim
# ENV APP_HOME /app
# WORKDIR $APP_HOME
# COPY . ./
# RUN pip install -r requirements.txt
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 app.app:app

#Using the base image with python 3.7
 FROM python:3.7
 
 #Set our working directory as app
 WORKDIR /app  

 ADD ./requirements.txt /app/requirements.txt
 # Copy the models directory and server.py files
 ADD ./app ./app

 COPY . .
 #Exposing the port 5000 from the container
 EXPOSE 5000

 #RUN pip install --default-timeout=1000 tensorflow

  #Installing python packages pandas, scikit-learn and gunicorn
 RUN pip3 install -r requirements.txt

 RUN python -m spacy download en_core_web_sm

 #Starting the python application
 CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server:app"]
