# DSND-Project-2
Udacity Data Scientist Nanodegree Project - Disaster Response Pipeline

Create a webapp and built Natural Language Processing pipeline that categorize messages from real-life emergencies.

## Overview
The project is divided in the following steps:

1.ETL Pipeline: data prepocessing, data cleaning, data storing in a database structure

2.ML Pipeline: train a NLP model to classify text message in categories

3.WebApp Development: plots dashboard, model results

## Dependencies

-Python 3

-sklearn.__version__=0.21.3

-SQLalchemy

-Flask

## Executing Program
1.Run the following commands in the project's root directory to set up your database and model

  - ETL pipeline: python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db
  - ML pipeline:  python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
  
2.Run the following command in the app's directory to run your web app. python run.py

3.Go to http://0.0.0.0:3001/

## Acknowledgements

The data was provided by [Figure Eight](https://appen.com/) 

## Screenshots

Exploratory Data Analysis

![EDA](/screenshots/EDA.png)
![EDA](/screenshots/distribution_classes.png)
![EDA](/screenshots/common_words.png)

Model results

![EDA](/screenshots/model.png)




