#!flask/bin/python
"""
Flask application to serve Machine Learning models
"""
import os
import flask
import json
import argparse
import logging
import numpy as np

from time import time
from model import Model
from functools import wraps


# Version of this APP template
__version__ = '1.2.0'
# Read env variables
DEBUG = os.environ.get('DEBUG', True)
MODEL_NAME = os.environ.get('MODEL_NAME', 'model.joblib')
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'local')
SERVICE_START_TIMESTAMP = time()


class NumpyEncoder(flask.json.JSONEncoder):
    """Encoder of numpy primitives into JSON strings"""
    primitives = (np.ndarray, np.integer, np.inexact)

    def default(self, obj):
        if isinstance(obj, np.flexible):
            return None if isinstance(obj, np.void) else obj.tolist()
        elif isinstance(obj, self.primitives):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def returns_json(f):
    """Wraps a function to transform the output into a JSON string with a
    specific encoder"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        r = f(*args, **kwargs)
        if isinstance(r, flask.Response):
            return r
        else:
            return flask.Response(json.dumps(r, cls=NumpyEncoder), status=200,
                                  mimetype='application/json; charset=utf-8')
    return decorated_function


def create_model_instance():
    global MODEL_NAME
    # Get current directory
    base_dir = os.getcwd()
    # Fix for documentation compilation
    if os.path.basename(base_dir) == 'docsrc':
        base_dir = os.path.dirname(base_dir)
    # Check if there is a model in the directory with the expected name
    model_path = os.path.join(base_dir, MODEL_NAME)
    if not os.path.exists(model_path):
        raise RuntimeError("Model {} not found".format(model_path))
    else:
        # Model found! now create an instance
        return Model(model_path)


# Create Flask Application
app = flask.Flask(__name__)
# Customize Flask Application
app.logger.setLevel(logging.DEBUG if DEBUG else logging.ERROR)
app.json_encoder = NumpyEncoder
# Create Model instance
model = create_model_instance()
# laod saved model
app.logger.info('ENVIRONMENT: {}'.format(ENVIRONMENT))
app.logger.info('Using template version: {}'.format(__version__))
app.logger.info('Loading model...')
model.load()


@app.route('/predict', methods=['POST'])
@returns_json
def predict():
    """Make preditcions and explain them

    Model inference using input data. This is the main function.

    URL Params:
        proba (int):
            1 in order to compute probabilities for classification models or 0
            to return predicted class (classification) or value (regression).
            Default 0.
        explain (int):
            1 in order to compute moeldel explanations for the predicted value.
            This will return a status 500 when the model does not support
            explanations. Default 0.

    Payload:
        JSON string that can take two forms:

        The first, the payload is a record or a list of records with one value
        per feature. This will be directly interpreted as the input for the
        model.

        The second, the payload is a dictionary with 1 or 2 elements. The key
        "_data" is mandatory because this will be the input for the model and
        its format is expected to be a record or a list of records. On the
        other hand the key "_samples" (optional) will be used to obtain
        different explanations (see :func:`~model.Model.explain`)
    """
    # Parameters
    do_proba = int(flask.request.args.get('proba', 0))
    do_explain = int(flask.request.args.get('explain', 0))
    input = json.loads(flask.request.data or '{}')
    if isinstance(input, dict):
        samples = input.get('_samples', None)
        input = input.get('_data', {})
    else:
        samples = None
    # Predict
    before_time = time()
    try:
        predict_function = 'predict_proba' if do_proba else 'predict'
        prediction = getattr(model, predict_function)(input)
    except Exception as err:
        return flask.Response(str(err), status=500)
    result = {'prediction': prediction}
    # Explain
    if do_explain:
        try:
            explanation = model.explain(input, samples=samples)
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
        'model_info': model.info,
        'elapsed_time': after_time - before_time
    }
    app.logger.debug(to_be_logged)
    return result


@app.route('/info',  methods=['GET'])
@returns_json
def info():
    """Model information

    Get the model information: metadata, type, classifier, etc.

    """
    try:
        info = model.info
    except Exception as err:
        return flask.Response(str(err), status=500)
    else:
        return info


@app.route('/features',  methods=['GET'])
@returns_json
def features():
    """Model features

    Get the model accepted features. This includes feature inportance if the
    model allows it.

    """
    try:
        features = model.features()
    except Exception as err:
        return flask.Response(str(err), status=500)
    else:
        return features


@app.route('/preprocess',  methods=['POST'])
@returns_json
def preprocess():
    """Preporcess input data

    Get the preprocessed version of the input data. If the model does not
    include preprocessing steps, this method will return the same data as the
    input.

    """
    input = json.loads(flask.request.data or '{}')
    try:
        data = model.preprocess(input)
    except Exception as err:
        return flask.Response(str(err), status=500)
    else:
        return data


@app.route('/health')
def health_check():
    return flask.Response("up", status=200)


@app.route('/ready')
def readiness_check():
    if model.is_ready():
        return flask.Response("ready", status=200)
    else:
        return flask.Response("not ready", status=503)


@app.route('/service-info')
@returns_json
def service_info():
    """Service information

    Get information about the service: up-time, varsion of the template, name
    of the served model, etc.

    """
    info =  {
        'version-template': __version__,
        'running-since': SERVICE_START_TIMESTAMP,
        'serving-model': MODEL_NAME,
        'debug': DEBUG}
    return info


if __name__ == '__main__':
    app.run(
        debug=DEBUG,
        host=os.environ.get('HOST', 'localhost'),
        port=os.environ.get('PORT', '5000'))
