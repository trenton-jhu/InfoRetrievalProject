import flask
from flask import request, jsonify
from flask_cors import CORS

from feedback import *
from helpers import *

# Flask server init
app = flask.Flask(__name__)
CORS(app)
app.config["DEBUG"] = False

# Load trained classifiers
sclf = SingleLabelClassifier()
sclf.load_clf("best_single_label_clf.pkl")
mclf = MultiLabelClassifier()
mclf.load_clf("best_multi_label_clf.pkl")
saclf = SingleLabelClassifier()
saclf.load_clf("best_sentiment_analysis_clf.pkl")

# Load movie data and term vectors
df = pd.read_csv('term_vectors.csv')
df = df.sample(frac=1).reset_index(drop=True)
state = {}


def get_movie(index):
    term_vector = eval(df.iloc[index]['tfidf'])
    movie = {
        'title': df.iloc[index]['name'],
        'url': df.iloc[index]['url'],
        'image': df.iloc[index]['image'],
        'genre': df.iloc[index]['genre'],
        'director': df.iloc[index]['director'],
        'cast': df.iloc[index]['cast'],
        'summary': df.iloc[index]['storyline'],
        'runtime': int(df.iloc[index]['runtime']),
        'prediction': state['feedback'].predict_label(term_vector),
    }
    return jsonify(movie)


@app.route('/genre', methods=['GET'])
def predict_genre():
    data = request.args.get("data")
    genre = sclf.predict_one(pd.DataFrame({
        'storyline': data
    }, index=[0]))
    return jsonify({'genre': genre})


@app.route('/genres', methods=['GET'])
def predict_genres():
    data = request.args.get("data")
    genres = mclf.predict_one(pd.DataFrame({
        'storyline': data
    }, index=[0]))
    return jsonify({'genres': genres})


@app.route('/sent', methods=['GET'])
def predict_sentiment():
    data = request.args.get("data")
    sent = saclf.predict_one(pd.DataFrame({
        'text': data
    }, index=[0]))
    return jsonify({'label': sent})


@app.route('/init', methods=['GET'])
def init_feedback():
    data = request.args.get("mode")
    if data == 'nearest':
        state['feedback'] = NearestCentroidRelevanceFeedback()
    else:
        state['feedback'] = NearestCentroidRelevanceFeedback()
    state['index'] = 0
    return get_movie(state['index'])


@app.route('/feedback', methods=['GET'])
def step_feedback():
    data = request.args.get("data")
    term_vector = eval(df.iloc[state['index']]['tfidf'])
    state['feedback'].add_label(term_vector, int(data))
    state['index'] += 1
    return get_movie(state['index'])


if __name__ == '__main__':
    app.run()

