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
scikit-image        0.18.1
imageio             2.9.0
urllib3             1.26.4
```
Alternatively, some experiments can be run using a Google Colab python notebook remotely which has many pre-installed
 packages available, eliminating the need to install all the dependencies. We provide a copy of the Colab notebook
  here: https://colab.research.google.com/drive/1yJ9kQ609gHBgBvy1P7DmG-iSEWgq0cFw?usp=sharing
 
## How to Run
To run crawler for collecting raw data, run `python crawl.py` which will store the movies and reviews data collected
 as json files.

To run all experiments, use `python experiment.py`. We strongly suggest running all the experiments on the
 (Google Colab Notebook)[https://colab.research.google.com/drive/1yJ9kQ609gHBgBvy1P7DmG-iSEWgq0cFw?usp=sharing
 ] rather than running this locally because all the experiments together take considerable amount of time and compute
  resource.
  
To run the web app for testing trained models and feedback loop, first start the server by running `python server.py
` which will spin up a Flask server at localhost port `5000`. After the server is up, open `index.html` in a web
 browser and user the web app on the page.

## Crawler
The crawl class has three variables: movie_id, genre, and max_links. This class is used for crawling movies’ information, and we store the information in a queue: we push the url of films (e.g. https://www.imdb.com/title/tt1345836/) into the queue. Then, we iterate through the queue until the queue is empty, or until the max_link number is reached. In one iteration, we first receive one URL, use requests to request the source code, and use Beautiful Soup to understand the information. We first evaluate if “genre” matches the given one. (If not, we iterate the next url.) And then evaluate other information, including name, image, summary, director, storyline, cast, runtime, rating, count of rating, and store the information in a dictionary. In addition, we also need to understand the href in rec_item, which are the URLs to other films. We push these links to the queue for later iterations. Lastly, we store the information of the film in a list called result, and then store the URL of the current film in the list called visited, as well as in “crawled” to prevent analyzing the same film multiple times. 

When the iteration is accomplished, we store the result and visited in json files: visited.json and movie.json.

Since we initially gave 3 movie_id, and set max_links to be 1000, we received 3000 films’ information.


The other piece of information that we need to craw is the movie reviews. Since visited.json contains the URLs of all films that we crawled, we iterate through all URLs in that file. To get the review information, we use each film’s review URL (e.g. //m.imdb.com/title/ tt1345836/reviews) to get the review information.

Then, we want to understand if a review is good one or not. When the corresponding rating is 0 or 1, we define it to be bad review, and when the rating is 9 or 10, we define it to be a good review. Use filters, we can get the url that contains all reviews with a rating of 1 (e.g.https://www.imdb.com/title/tt1345836//reviews?spoiler=hide&sort=helpfulnessScore&dir=desc&ratingFilter=1). Simiarily, we use requests to get the source code, Beautiful Soup to extract the information, and we can get information such as name of the file, review title, rating, and review content. And we label the review as a bad_review or a good_review.
Lastly, we store the name, good_review and bad_review as the three keys of a film. The bad_review and good_review each contains multiple reviews. We print such information to review.json for later analysis.




## Genre Classifier
For classifying genre, we use both single-label and multi-label classifiers. For single-label classifier, we convert
 the data such that each movie is only labeled with one genre (Action, Comedy, or Romance). We crawled data
  specifically for movies belonging to these genres. For multi-label classifier, each movie can be labeled with
   multiple different genres and the classifier can also predict many genres.
   
We experimented with different feature vector extraction techniques, including basic TF, TF-IDF and different n-gram
 models. To build the classifiers, we used many different IR and ML models including NearestCentroid, KNN
 , DecisionTree, RandomForest, etc. For performance metric, we held out 20% of available data as validation set and
  train the model on the development set then evaluate the model on the held out set to report its validation
accuracy. The results are recorded in `results.tsv`. We find that linear classifiers like Ridge and Logistic
classifiers seem to perform pretty well for single-label classifier and neural networks work a lot better for
multi-label classifier. We also note here that for multi-label classifier, the accuracy measure is more relaxed
in the sense that the prediction will be correct when at least one of the predicted genres is a correct genre
for the movie.
      
We also performed Image IR on the movie poster and extract feature vectors for the movie poster as image. We then
 tried our classifiers on this but the performance is not great. This is probably because movie poster is too complex
  for simple models and classifiers like this. In particular, we only consider the image as a 1D feature vectors of
   pixels and in doing so may neglect any spatial dependencies.
   
      
## Sentiment Analysis
To classify movie review as positive or negative, we follow the same training and evaluation process as in the single
-label genre classifier experiments. We find that classifiers for the sentiment analysis tend to perform better than
 genre classification. This is as expected since term vectors extracted from text would probably be more
  representative of reviews (which are just plain text) than movies.


## Feedback Loop
We build a movie preference predictor based on techniques similar to relevant feedback loop. We use either
 NearestCentroid or KNN model based on TF-IDF term vectors for all movies. To do this, we first pre-compute and store
  the term vectors from all movie. For each iteration of the feedback loop, we randomly select a movie and ask the user
   for his preference on it. We then record this input, which along with all previous user feedback, can be used to
 predict whether the user prefers the movie we sample in the next iteration. This means that as we get more feedback
  from the user, the predictor is getting more training data and so will make better prediction in the future.
  

## Future Steps
Here are some future prospects that can improve the project:
* For movie genre classification, incorporate other movie attributes like runtime, directors, cast into the feature
 vectors. This may potentially increase performance as we can making different aspects of the movie help make the
  genre classification judgement
  
* For Image IR, we probably should try using classifier that respects spatial dependencies of pixel values, a
 Convolutional Neural Network (CNN) could be better suited for this task.
 
* We can experiment with different models besides NearestCentroid and KNN for building the movie preference predictor
 using the feedback loop.
 
* Try to perform regression instead of classification for movie review sentiment analysis. We can predict a user's
 rating out of 10, which is a continuous rather than discrete label, based on the content of their review.