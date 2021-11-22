# Disaster_Response_ML_Pipeline

## Installation
This project uses the following libraries and should run with python3 interpreter\
* Sys
* Pandas
* sqlalchemy
* NumPy
* nltk
* re
* pickle
* Sckit-learn

## Introduction and Motivation
In the time of emergemcy like natrual disasters, people send information online to disaster response agencies or to social media and ask for help. Different angencies are in charge of different resources. For example, one angency is in charge of water supply while the other is in charge of medical supplies. In this project, data is taken from real world messages that people send in time of natural disasters. It will preprocess all the text data using an ETL pieline and build ML pipeline to classify what kind of this message is. For example, some messages are asking for water but other messages are asking for medical supplies. Then we know where to forward those messages in the time of disasters.

## File Introduction
data/process_data.py: This is the ETL pipeline file. It contains how original raw text data is transformed into data that is ready to go into ML models. It contains operations such as normaliztion, tokenization and lemmatization.\
data/disaster_messages.csv and data/disaster_categories.csv: These two files contain the data we use in this project.\
app/templates/go.html and app/templates/master.html: These are two html files that will build the web app.\
app/run.py: This contains python code that uses Falsk to run the web app. The web app contains visualization of the data and an interface to classify different types of messages.\

## How to run the program
- To run ETL pipeline that cleans data and stores in database\
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`\
- To run ML pipeline that trains classifier and saves\
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`\
- To run the web app, run following command in the app's directory\
    `python run.py`\
     Then go to http://0.0.0.0:3001/\
     
