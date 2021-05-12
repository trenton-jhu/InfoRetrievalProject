import numpy as np
import pandas as pd
import math
from collections import Counter, defaultdict
from typing import Dict


# Baseline text vectorization
def compute_doc_freqs(data: pd.DataFrame):
    """
    Computes document frequency given all docs stored in a single-column DataFrame
    """
    freq = Counter()
    for doc in data:
        words = set()
        for word in doc:
            words.add(word)
        for word in words:
            freq[word] += 1
    return freq


def compute_tf(text):
    vec = defaultdict(float)
    for word in text:
        vec[word] += 1
    return dict(vec)


def compute_tfidf(text, num_docs, doc_freqs):
    tf = compute_tf(text)
    vec = dict()
    for word in tf:
        if doc_freqs[word] == 0:
            continue
        vec[word] = tf[word] * math.log(num_docs / doc_freqs[word])
    return dict(vec)


# Term vectors computation
def dict_add(x: Dict[str, float], y: Dict[str, float]):
    result = defaultdict(float)
    for word in x:
        result[word] += x[word]
    for word in y:
        result[word] += y[word]
    return dict(result)


def dict_divide(x: Dict[str, float], a: int):
    result = defaultdict(float)
    for word in x:
        result[word] = x[word] / a
    return dict(result)


def dict_dot(x: Dict[str, float], y: Dict[str, float]):
    keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
    return sum(x.get(key, 0) * y.get(key, 0) for key in keys)


def cosine_sim(x, y):
    num = dict_dot(x, y)
    if num == 0:
        return 0
    return num / (np.linalg.norm(list(x.values())) * np.linalg.norm(list(y.values())))


class KNNRelevanceFeedback:
    """
    Relevance feedback loop using K Nearest Neighbor
    """

    def __init__(self, k=5):
        self.vectors = []
        self.k = k

    def add_label(self, term_vector: Dict[str, float], label: int):
        """
        Add user feedback for a term vector. Label 0 is preferred, Label 1 is not preferred
        """
        self.vectors.append((term_vector, label))

    def predict_label(self, term_vector: Dict[str, float]) -> int:
        """
        Predict label for a given term vector
        """
        sims = [(cosine_sim(term_vector, d[0]), d[1]) for d in self.vectors]
        sims.sort(reverse=True)
        c0 = c1 = 0
        for x in sims[:self.k]:
            if x[1] == 0:
                c0 += 1
            else:
                c1 += 1
        return 0 if c0 >= c1 else 1


class NearestCentroidRelevanceFeedback:
    """
    Relevance feedback loop using Nearest Centroid
    """

    def __init__(self):
        self.sum0 = dict()
        self.sum1 = dict()
        self.n0 = 0
        self.n1 = 0

    def add_label(self, term_vector: Dict[str, float], label: int):
        if label == 0:
            self.sum0 = dict_add(self.sum0, term_vector)
            self.n0 += 1
        else:
            self.sum1 = dict_add(self.sum1, term_vector)
            self.n1 += 1

    def predict_label(self, term_vector: Dict[str, float]) -> int:
        if self.n0 == 0:
            return 1
        elif self.n1 == 0:
            return 0
        avg0 = dict_divide(self.sum0, self.n0)
        avg1 = dict_divide(self.sum1, self.n1)
        if cosine_sim(term_vector, avg0) >= cosine_sim(term_vector, avg1):
            return 0
        else:
            return 1


def test():
    df = pd.read_csv('term_vectors.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    feedback = NearestCentroidRelevanceFeedback()
    for i in range(20):
        if i < 10:
            feedback.add_label(eval(df.iloc[i]['tfidf']), 0)
        else:
            feedback.add_label(eval(df.iloc[i]['tfidf']), 1)

    print(feedback.predict_label(eval(df.iloc[300]['tfidf'])))


if __name__ == '__main__':
    test()
