# Disaster Response Pipeline Project

### Table of Contents
1. [Description](#description)
1. [Deployment](#deployment)
2. [Dataset](#dataset)
3. [Licensing](#licensing)

## Description <a name="description"></a>
In this project, disaster data from [Figure Eight](https://www.figure-eight.com/) is analyzed and used to build a model for an API that classifies disaster messages. The project includes a web application deployed on Heroku where an emergency worker can input a new message and get classification results in several categories in real time. The project challenges my knowledge and software engineering skills as well as ability to develop data pipelines and deply eb applciations that use data scand write clean, organized code!

## Deployment <a name="deployment"></a>

The live application can be viewed on [heroku](http://heroku.com)

### Instructions for deploying locally
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves model weights
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ to view the live app


## Licensing <a name="licensing"></a>

Standard MIT License.


