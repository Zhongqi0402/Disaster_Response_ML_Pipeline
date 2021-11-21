# Disaster_Response_ML_Pipeline

## Introduction
In the time of emergemcy like natrual disasters, people send information online to disaster response agencies or to social media and ask for help. In this project, data is taken from real world messages that people send in time of natural disasters. It will preprocess all the text data using an ETL pieline and build ML pipeline to classify what kind of this message is. For example, some messages are asking for water but other messages are asking for medical supplies.

## File Introduction
data/process_data.py: This is the ETL pipeline file. It contains how original raw text data is transformed into data that is ready to go into ML models. It contains operations such as normaliztion, tokenization and lemmatization.\
data/disaster_messages.csv and data/disaster_categories.csv: These two files contain the data we use in this project.
