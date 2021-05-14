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
The crawl class has three variables: movie_id, genre, and max_links. This class is used for crawling movies’ information, and we store the information in a queue: we push the url of films (e.g. https://www.imdb.com/title/tt1345836/) into the queue. Then, we iterate through the queue until the queue is empty, or until the max_link number is reached. In one iteration, we first receive one URL, use requests to request the source code, and use Beautiful Soup to understand the information. We first evaluate if “genre” matches the given one. (If not, we iterate the next url.) And then evaluate other information, including name, image, summary, director, storyline, cast, runtime, rating, count of rating, and store the information in a dictionary. In addition, we also need to understand the href in rec_item, which are the URLs to other films. We push these links to the queue for later iterations. Lastly, we store the information of the film in a list called result, and then store the URL of the current film in the list called visited, as well as in “crawled” to prevent analyzing the same film multiple times. 

When the iteration is accomplished, we store the result and visited in json files: visited.json and movie.json.

Since we initially gave 3 movie_id, and set max_links to be 1000, we received 3000 films’ information.


The other piece of information that we need to craw is the movie reviews. Since visited.json contains the URLs of all films that we crawled, we iterate through all URLs in that file. To get the review information, we use each film’s review URL (e.g. //m.imdb.com/title/ tt1345836/reviews) to get the review information.

Then, we want to understand if a review is good one or not. When the corresponding rating is 0 or 1, we define it to be bad review, and when the rating is 9 or 10, we define it to be a good review. Use filters, we can get the url that contains all reviews with a rating of 1 (e.g.https://www.imdb.com/title/tt1345836//reviews?spoiler=hide&sort=helpfulnessScore&dir=desc&ratingFilter=1). Simiarily, we use requests to get the source code, Beautiful Soup to extract the information, and we can get information such as name of the file, review title, rating, and review content. And we label the review as a bad_review or a good_review.
Lastly, we store the name, good_review and bad_review as the three keys of a film. The bad_review and good_review each contains multiple reviews. We print such information to review.json for later analysis.




## Genre Classifier


## Sentiment Analysis

## Feedback Loop

## Future Steps
