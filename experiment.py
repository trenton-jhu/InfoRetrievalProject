from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from vectorizer import *


def to_single_label(x):
    """
    Convert to a single movie genre for single-label classifier
    """
    if 'Action' in x:
        return 'Action'
    if 'Romance' in x:
        return 'Romance'
    if 'Comedy' in x:
        return 'Comedy'
    raise ValueError("unrecognized genre")


class SingleLabelGenreClassifier:
    """
    Helper class for training and evaluating single-label classifiers on movie genres
    """

    def __init__(self, vectorizer=MovieVectorizer(), classifier=NearestCentroid()):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.encoder = LabelEncoder()
        self.trained = False

    def prepare_data(self, df: pd.DataFrame):
        genre = df['genre'].apply(to_single_label)
        y = self.encoder.fit_transform(genre)
        return train_test_split(df.drop(columns=['genre']), y, test_size=0.2, random_state=9)

    def train(self, X, Y):
        Xtrain = self.vectorizer.fit_transform(X)
        self.classifier.fit(Xtrain, Y)
        self.trained = True

    def predict_one(self, x) -> str:
        """
        Given one movie as a single DataFrame, predict its genre
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


class MultiLabelGenreClassifier:
    """
    Helper class for training and evaluating multi-label classifiers on movie genres
    Classifier can predict multiple genres for a given movie
    """

    def __init__(self, vectorizer=MovieVectorizer(), classifier=NearestCentroid()):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.encoder = MultiLabelBinarizer()
        self.trained = False

    def prepare_data(self, df: pd.DataFrame):
        y = self.encoder.fit_transform(df['genre'])
        return train_test_split(df.drop(columns=['genre']), y, test_size=0.2, random_state=9)

    def train(self, X, Y):
        Xtrain = self.vectorizer.fit_transform(X)
        self.classifier.fit(Xtrain, Y)
        self.trained = True

    def predict_one(self, x) -> List[str]:
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


class ReviewClassifier:
    """
    Helper class for training and evaluating classifiers on movie reviews sentiment analysis
    """
    # TODO: Implement



def main():
    df = pd.read_json("movies.json")
    df['storyline'] = df['storyline'] + df['summary']
    df = df[['storyline', 'genre']]

    single_label_classifiers = {
        'NearestCentroid': NearestCentroid(),
        'KNN': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'Ridge': RidgeClassifier(),
    }

    multi_label_classifiers = {
        'KNN': KNeighborsClassifier(),
        'RandomForest': RandomForestClassifier(),
        'Neural': MLPClassifier(hidden_layer_sizes=(500, 500), max_iter=10000)
    }

    print('Experiment with single label classifier:')
    print('classifier', 'correct', 'incorrect', 'accuracy', sep='\t')
    for name in single_label_classifiers:
        clf = SingleLabelGenreClassifier(classifier=single_label_classifiers[name])
        Xtrain, Xval, Ytrain, Yval = clf.prepare_data(df)
        clf.train(Xtrain, Ytrain)
        correct, incorrect = clf.evaluate(Xval, Yval)
        accuracy = correct / (correct + incorrect)
        print(name, correct, incorrect, accuracy, sep='\t')

    print('Experiment with multi label classifier (classifier can predict more than one genres):')
    print('classifier', 'correct', 'incorrect', 'accuracy', sep='\t')
    for name in multi_label_classifiers:
        clf = MultiLabelGenreClassifier(classifier=multi_label_classifiers[name])
        Xtrain, Xval, Ytrain, Yval = clf.prepare_data(df)
        clf.train(Xtrain, Ytrain)
        correct, incorrect = clf.evaluate(Xval, Yval)
        accuracy = correct / (correct + incorrect)
        print(name, correct, incorrect, accuracy, sep='\t')


if __name__ == '__main__':
    main()
