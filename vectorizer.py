import nltk
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize


class MultiHotEncoder(BaseEstimator, TransformerMixin):
    """
    Mutli-class version of sklearn OneHotEncoder for supporting ColumnTransformer
    """

    def __init__(self):
        self.mlbs = list()
        self.n_columns = 0
        self.categories_ = self.classes_ = list()

    def fit(self, X: pd.DataFrame):
        for i in range(X.shape[1]):
            mlb = MultiLabelBinarizer()
            mlb.fit(X.iloc[:, i])
            self.mlbs.append(mlb)
            self.classes_.append(mlb.classes_)
            self.n_columns += 1
        return self

    def transform(self, X: pd.DataFrame):
        if self.n_columns == 0:
            raise ValueError("Please fit the vectorizer first by calling fit()")
        if self.n_columns != X.shape[1]:
            raise ValueError("Shape does not match")
        result = list()
        for i in range(self.n_columns):
            result.append(self.mlbs[i].transform(X.iloc[:, i]))

        result = np.concatenate(result, axis=1)
        return result


class MovieVectorizer:
    """
    Class for extracting feature vectors from movie home page data
    """

    def __init__(self, tf_idf=True):
        self.tfidf_vectorizer = TfidfVectorizer(
            preprocessor=lambda x: x.lower(),
            tokenizer=nltk.tokenize.word_tokenize,
            use_idf=tf_idf
        )
        self.transformers = [('storyline', self.tfidf_vectorizer, 'storyline')]
        self.trans = None
        self.tf_idf = tf_idf

    def extract_director(self):
        self.transformers.append(('director', MultiHotEncoder(), ['director']))

    def extract_cast(self):
        self.transformers.append(('cast', MultiHotEncoder(), ['cast']))

    def extract_runtime(self):
        self.transformers.append(('runtime', 'passthrough', ['runtime']))

    def fit_transform(self, X: pd.DataFrame):
        self.trans = ColumnTransformer(transformers=self.transformers)
        return self.trans.fit_transform(X)

    def transform(self, X: pd.DataFrame):
        if self.trans is None:
            raise ValueError("Please fit the vectorizer first by calling fit()")
        else:
            return self.trans.transform(X)


class MovieNGramVectorizer:
    """
    Class for extracting n-gram feature vectors from movie text data
    """

    def __init__(self, ngram=(1, 2), normal=True):
        self.vectorizer = CountVectorizer(
            tokenizer=nltk.tokenize.word_tokenize,
            ngram_range=ngram
        )
        self.normal = normal

    def fit_transform(self, X):
        vec = self.vectorizer.fit_transform(X['storyline'])
        return normalize(vec) if self.normal else vec

    def transform(self, X):
        vec = self.vectorizer.transform(X['storyline'])
        return normalize(vec) if self.normal else vec

