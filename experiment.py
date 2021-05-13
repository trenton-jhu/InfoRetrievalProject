import itertools

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from helpers import *


nltk.download('stopwords')

def to_single_genre(x):
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


def genre_classification(file_name="movies.json"):
    """
    Run genre classification experiments based on movie data
    """
    df = pd.read_json(file_name)
    df['storyline'] = df['storyline'] + df['summary']
    df['single_genre'] = df['genre'].apply(to_single_genre)
    df = df[['storyline', 'genre', 'single_genre']]
    data_col = 'storyline'

    vectorizers = {
        'TF': TermVectorizer(data_col, tf_idf=False),
        'TF-IDF': TermVectorizer(data_col),
        'TF-IDF NoStopwords': TermVectorizer(data_col, remove_stopwords=True),
        '2-gram': NGramVectorizer(data_col, ngram=(2, 2)),
        '3-gram': NGramVectorizer(data_col, ngram=(3, 3)),
        '1,2-gram': NGramVectorizer(data_col, ngram=(1, 2)),
    }

    single_label_classifiers = {
        'NearestCentroid': NearestCentroid(),
        'KNN': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'Ridge': RidgeClassifier(),
        'Bernoulli NB': BernoulliNB(),
        'Logistic': LogisticRegression(),
    }

    multi_label_classifiers = {
        'KNN': KNeighborsClassifier(),
        'RandomForest': RandomForestClassifier(),
        'Neural': MLPClassifier(hidden_layer_sizes=(500, 500), max_iter=10000),
    }

    best_acc = 0
    best_single_label_clf = None
    best_vec, best_clf = "", ""

    print('Experiment with single label classifier for classifying movie genre:')
    print('vectorizer', 'classifier', 'correct', 'incorrect', 'accuracy', sep='\t')
    for vectorizer, classifier in itertools.product(*[vectorizers, single_label_classifiers]):
        clf = SingleLabelClassifier(
            vectorizer=vectorizers[vectorizer],
            classifier=single_label_classifiers[classifier]
        )
        Xtrain, Xval, Ytrain, Yval = clf.prepare_data('single_genre', df)
        clf.train(Xtrain, Ytrain)
        correct, incorrect = clf.evaluate(Xval, Yval)
        accuracy = correct / (correct + incorrect)
        if accuracy > best_acc:
            best_acc = accuracy
            best_vec = vectorizer
            best_clf = classifier
            best_single_label_clf = clf
        print(vectorizer, classifier, correct, incorrect, accuracy, sep='\t')

    print(f"Best accuracy achieved: {best_acc} with {best_vec} and {best_clf}")
    print(f"Saving model to file")
    best_single_label_clf.save_clf("best_single_label_clf.pkl")

    best_multi_label_clf = None

    print('Experiment with multi label classifier (classifier can predict more than one genres):')
    print('vectorizer', 'classifier', 'correct', 'incorrect', 'accuracy', sep='\t')
    for vectorizer, classifier in itertools.product(*[vectorizers, multi_label_classifiers]):
        clf = MultiLabelClassifier(
            vectorizer=vectorizers[vectorizer],
            classifier=multi_label_classifiers[classifier]
        )
        Xtrain, Xval, Ytrain, Yval = clf.prepare_data('genre', df)
        clf.train(Xtrain, Ytrain)
        correct, incorrect = clf.evaluate(Xval, Yval)
        accuracy = correct / (correct + incorrect)
        if accuracy > best_acc:
            best_acc = accuracy
            best_vec = vectorizer
            best_clf = classifier
            best_multi_label_clf = clf
        print(vectorizer, classifier, correct, incorrect, accuracy, sep='\t')

    print(f"Best accuracy achieved: {best_acc} with {best_vec} and {best_clf}")
    print(f"Saving model to file")
    best_multi_label_clf.save_clf("best_multi_label_clf.pkl")


def sentiment_analysis(file_name="reviews.json"):
    """
    Run sentiment analysis experiments to classify movie reviews as positive or negative
    """
    df = pd.read_json(file_name)
    df['text'] = df['title'] + df['text']
    df = df[['text', 'label']]
    data_col, label_col = 'text', 'label'

    vectorizers = {
        'TF': TermVectorizer(data_col, tf_idf=False),
        'TF-IDF': TermVectorizer(data_col),
        'TF-IDF NoStopwords': TermVectorizer(data_col, remove_stopwords=True),
        '2-gram': NGramVectorizer(data_col, ngram=(2, 2)),
        '3-gram': NGramVectorizer(data_col, ngram=(3, 3)),
        '1,2-gram': NGramVectorizer(data_col, ngram=(1, 2)),
    }

    single_label_classifiers = {
        'NearestCentroid': NearestCentroid(),
        'KNN': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'Ridge': RidgeClassifier(),
        'Bernoulli NB': BernoulliNB(),
        'Logistic': LogisticRegression(),
    }

    best_acc = 0
    best_sentiment_analysis_clf = None
    best_vec, best_clf = "", ""

    print('Experiment with sentiment analysis for movie reviews:')
    print('vectorizer', 'classifier', 'correct', 'incorrect', 'accuracy', sep='\t')
    for vectorizer, classifier in itertools.product(*[vectorizers, single_label_classifiers]):
        clf = SingleLabelClassifier(
            vectorizer=vectorizers[vectorizer],
            classifier=single_label_classifiers[classifier]
        )
        Xtrain, Xval, Ytrain, Yval = clf.prepare_data(label_col, df)
        clf.train(Xtrain, Ytrain)
        correct, incorrect = clf.evaluate(Xval, Yval)
        accuracy = correct / (correct + incorrect)
        if accuracy > best_acc:
            best_acc = accuracy
            best_vec = vectorizer
            best_clf = classifier
            best_sentiment_analysis_clf = clf
        print(vectorizer, classifier, correct, incorrect, accuracy, sep='\t')

    print(f"Best accuracy achieved: {best_acc} with {best_vec} and {best_clf}")
    print(f"Saving model to file")
    best_sentiment_analysis_clf.save_clf("best_sentiment_analysis_clf.pkl")


def main():
    genre_classification()
    sentiment_analysis()


if __name__ == '__main__':
    main()
