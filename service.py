#!flask/bin/python
"""
Flask application to serve Machine Learning models
"""
import os
import flask
import json
import numpy as np

from time import time
from model import Model


class NumpyEncoder(flask.json.JSONEncoder):
    primitives = (np.ndarray, np.integer, np.inexact)

    def default(self, obj):
        if isinstance(obj, np.flexible):
            return None if isinstance(obj, np.void) else obj.tolist()
        elif isinstance(obj, self.primitives):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Create Flask Application
app = flask.Flask(__name__)

# Customize Flask Application
app.json_encoder = NumpyEncoder

# Read env variables
DEBUG = os.environ.get('DEBUG', True)
MODEL_NAME = os.environ.get('MODEL_NAME', 'model.joblib')
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'local')

# Create Model instance
base_dir = os.getcwd()

if os.path.basename(base_dir) == 'docs':
    base_dir = os.path.dirname(base_dir)

model_path = os.path.join(base_dir, MODEL_NAME)
if not os.path.exists(model_path):
    raise RuntimeError("Model {} not found".format(model_path))
else:
    model = Model(model_path)

# laod saved model
app.logger.info('ENVIRONMENT: {}'.format(ENVIRONMENT))
app.logger.info('Loading model...')
model.load()


@app.route('/predict', methods=['POST'])
def predict():
    input = json.loads(flask.request.data or '{}')
    # Parameters
    do_proba = int(flask.request.args.get('proba', 0))
    do_explain = int(flask.request.args.get('explain', 0))
    # Predict
    before_time = time()
    try:
        predict_function = 'predict_proba' if do_proba else 'predict'
        prediction = getattr(model, pred_function)(input)
    except Exception as err:
        return flask.Response(str(err), status=500)
    result = {'prediction': prediction}
    # Eplain
    if do_explain:
        try:
            explanation = model.explain(input)
        except Exception as err:
            return flask.Response(str(err), status=500)
        else:
            result['explanation'] = explanation
    after_time = time()
    # log
    to_be_logged = {
        'input': flask.request.data,
        'params': flask.request.args,
        'request_id': flask.request.headers.get('X-Correlation-ID'),
        'result': result,
        'model': model.metadata,
        'elapsed_time': after_time - before_time
    }
    app.logger.info(to_be_logged)
    return flask.jsonify(result)


@app.route('/predict_proba', methods=['POST'])
def predict_proba():
    return flask.redirect(flask.url_for('predict', proba=1))


@app.route('/explain', methods=['POST'])
def explain():
    return flask.redirect(flask.url_for('predict', proba=1, explain=1))


@app.route('/info',  methods=['GET'])
def info():
    try:
        data = model.info
    except Exception as err:
        return flask.Response(str(err), status=500)
    return flask.jsonify(data)


@app.route('/features',  methods=['GET'])
def features():
    try:
        features = model.features()
    except Exception as err:
        return flask.Response(str(err), status=500)

    return flask.jsonify(features)


@app.route('/preprocess',  methods=['POST'])
def preprocess():
    input = json.loads(flask.request.data or '{}')
    try:
        features = model.preprocess(input)
    except Exception as err:
        return flask.Response(str(err), status=500)

    return flask.jsonify(features)


@app.route('/health')
def health_check():
    return flask.Response("up", status=200)


@app.route('/ready')
def readiness_check():
    if model.is_ready():
        return flask.Response("ready", status=200)
    else:
        return flask.Response("not ready", status=503)


if __name__ == '__main__':
    app.run(
        debug=DEBUG,
        host=os.environ.get('HOST', 'localhost'),
        port=os.environ.get('PORT', '5000'))
