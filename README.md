# Movies IR 

Trenton Yuntian Wang, Lynn Sihao Yin

## Overview
In this project, we focus on movie and movie reviews data and perform different info retrieval tasks using these
 collected data. We build a movie genre classifier for classifying genre of a movie based on its plot, a sentiment
  analyzer for determining sentiment of movie reviews, and a movie preference predictor for predicting a user's
   preferred movies based on relevant feedback from the user.
   
Data files are uploaded to here: https://drive.google.com/drive/folders/1iz2Wz2i9tyrManCKxWcNOQHASFtvOxy7?usp=sharing
This includes raw data crawled from IMDB, term vectors data, trained models, experiment results etc. 

Project github: https://github.com/trenton-jhu/InfoRetrievalProject

Google Colab Notebook: https://colab.research.google.com/drive/1yJ9kQ609gHBgBvy1P7DmG-iSEWgq0cFw?usp=sharing

Video Demo: 

## Depedencies
This project requires many different packages ranging from data analysis, storage, ML classifiers, serialization
, flask servers, etc. To ensure that all the components can run smoothly, please make sure all the dependencies are
 installed. `pip` is the recommended way to install. 
 
```
Package             Version
------------------- ----------
beautifulsoup4      4.9.3
Flask               1.1.2
Flask-Cors          3.0.10
more-itertools      8.6.0
nltk                3.5
numpy               1.19.1
pandas              1.2.4
pandocfilters       1.4.3
pickleshare         0.7.5
pip                 20.2.3
py                  1.10.0
scikit-learn        0.24.1
urllib3             1.26.4
```
Alternatively, some experiments can be run using a Google Colab python notebook remotely which has many pre-installed
 packages available, eliminating the need to install all the dependencies. We provide a copy of the Colab notebook
  here: https://colab.research.google.com/drive/1yJ9kQ609gHBgBvy1P7DmG-iSEWgq0cFw?usp=sharing
 

## Crawler


## Genre Classifier


## Sentiment Analysis

## Feedback Loop

## Future Steps