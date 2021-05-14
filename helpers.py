from typing import List

import nltk
import pickle
import numpy as np
import pandas as pd
import skimage.io as io
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, LabelEncoder, MultiLabelBinarizer


class TermVectorizer:
    """
    Class for extracting tf or tf-idf feature vectors from text data
    """

    def __init__(self, data_column, tf_idf=True, remove_stopwords=False):
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=nltk.tokenize.word_tokenize,
            use_idf=tf_idf,
            stop_words=nltk.corpus.stopwords.words('english') if remove_stopwords else None
        )
        self.col = data_column

    def fit_transform(self, X: pd.DataFrame):
        return self.tfidf_vectorizer.fit_transform(X[self.col])

    def transform(self, X: pd.DataFrame):
        return self.tfidf_vectorizer.transform(X[self.col])


class NGramVectorizer:
    """
    Class for extracting n-gram feature vectors from text data
    """

    def __init__(self, data_column, ngram=(1, 2), normal=True):
        self.vectorizer = CountVectorizer(
            tokenizer=nltk.tokenize.word_tokenize,
            ngram_range=ngram
        )
        self.normal = normal
        self.col = data_column

    def fit_transform(self, X):
        vec = self.vectorizer.fit_transform(X[self.col])
        return normalize(vec) if self.normal else vec

    def transform(self, X):
        vec = self.vectorizer.transform(X[self.col])
        return normalize(vec) if self.normal else vec


class ImageVectorizer:
    """
    Class for extracting feature vectors from image
    """

    def __init__(self):
        """
        Cache transformed image vectors for future use
        """
        self.Xtrain = None
        self.Xval = None

    def fit_transform(self, X):
        if self.Xtrain is not None:
            return self.Xtrain
        pixels = X['image'].apply(lambda x: io.imread(x, as_gray=True).flatten())
        self.Xtrain = np.array(pixels.values.tolist())
        return self.Xtrain

    def transform(self, X):
        if self.Xval is not None:
            return self.Xtrain
        pixels = X['image'].apply(lambda x: io.imread(x, as_gray=True).flatten())
        self.Xval = np.array(pixels.values.tolist())
        return self.Xval


class SingleLabelClassifier:
    """
    Helper class for training and evaluating single-label classifiers
    """

    def __init__(self, vectorizer=None, classifier=None):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.encoder = LabelEncoder()
        self.trained = False

    def save_clf(self, file_name):
        if not self.trained:
            raise Exception("Classifier needs to be trained first")
        with open(file_name, 'wb') as output:
            pickle.dump(self.vectorizer, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.classifier, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.encoder, output, pickle.HIGHEST_PROTOCOL)

    def load_clf(self, file_name):
        with open(file_name, 'rb') as input:
            self.vectorizer = pickle.load(input)
            self.classifier = pickle.load(input)
            self.encoder = pickle.load(input)
            self.trained = True

    def prepare_data(self, label_col, df: pd.DataFrame):
        y = self.encoder.fit_transform(df[label_col])
        return train_test_split(df.drop(columns=[label_col]), y, test_size=0.2, random_state=9)

    def train(self, X, Y):
        Xtrain = self.vectorizer.fit_transform(X)
        self.classifier.fit(Xtrain, Y)
        self.trained = True

    def predict_one(self, x) -> str:
        """
        Given one data point as a single DataFrame, predict its label
        """
        if not self.trained:
            raise Exception("Classifier needs to be trained first")
        xtest = self.vectorizer.transform(x)
        ytest = self.classifier.predict(xtest)
        return self.encoder.inverse_transform(ytest)[0]

    def evaluate(self, X, Y):
        if not self.trained:
            raise Exception("Classifier needs to be trained first")
        Xval = self.vectorizer.transform(X)
        Ypredict = self.classifier.predict(Xval)
        correct = np.sum(Ypredict == Y)
        incorrect = np.sum(Ypredict != Y)
        return correct, incorrect


class MultiLabelClassifier:
    """
    Helper class for training and evaluating multi-label classifiers on movie genres
    Classifier can predict multiple genres for a given movie
    """

    def __init__(self, vectorizer=None, classifier=None):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.encoder = MultiLabelBinarizer()
        self.trained = False

    def save_clf(self, file_name):
        if not self.trained:
            raise Exception("Classifier needs to be trained first")
        with open(file_name, 'wb') as output:
            pickle.dump(self.vectorizer, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.classifier, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.encoder, output, pickle.HIGHEST_PROTOCOL)

    def load_clf(self, file_name):
        with open(file_name, 'rb') as input:
            self.vectorizer = pickle.load(input)
            self.classifier = pickle.load(input)
            self.encoder = pickle.load(input)
            self.trained = True

    def prepare_data(self, label_col, df: pd.DataFrame):
        y = self.encoder.fit_transform(df[label_col])
        return train_test_split(df.drop(columns=[label_col]), y, test_size=0.2, random_state=9)

    def train(self, X, Y):
        Xtrain = self.vectorizer.fit_transform(X)
        self.classifier.fit(Xtrain, Y)
        self.trained = True

    def predict_one(self, x) -> List[str]:
        """
        Given one movie as a single DataFrame, predict possible genres
        """
        if not self.trained:
            raise Exception("Classifier needs to be trained first")
        xtest = self.vectorizer.transform(x)
        ytest = self.classifier.predict(xtest)
        return list(self.encoder.inverse_transform(ytest)[0])

    def evaluate(self, X, Y):
        if not self.trained:
            raise Exception("Classifier needs to be trained first")
        Xval = self.vectorizer.transform(X)
        Ypredict = self.classifier.predict(Xval)
        overlap = np.count_nonzero(Ypredict + Y == 2, axis=1)
        total = overlap.shape[0]
        correct = np.count_nonzero(overlap > 0)
        incorrect = total - correct
        return correct, incorrect
