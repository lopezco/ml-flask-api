#!flask/bin/src
"""
Flask application to serve Machine Learning models
"""
import os
import flask
import json
import logging

from time import time

from src.utils.encoder import ExtendedEncoder, returns_json
from src.factory import ModelFactory

# Version of this APP template
__version__ = '2.2.0'
# Read env variables
DEBUG = os.environ.get('DEBUG', False)
MODEL_NAME = os.environ.get('MODEL_NAME', 'model.joblib')
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'local')
MODEL_TYPE = os.environ.get('MODEL_TYPE', 'SKLEARN_MODEL')
SERVICE_START_TIMESTAMP = time()
# Create Flask Application
application = flask.Flask(__name__)
# Customize Flask Application
application.logger.setLevel(logging.DEBUG if DEBUG else logging.ERROR)
application.json_encoder = ExtendedEncoder
# Create Model instance
model = ModelFactory.create_model(MODEL_NAME, MODEL_TYPE)
# load saved model
application.logger.info('ENVIRONMENT: {}'.format(ENVIRONMENT))
application.logger.info('Using template version: {}'.format(__version__))
application.logger.info('Loading model...')
model.load()


@application.route('/predict', methods=['POST'])
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
        data = input.get('_data', {})
        if len(data.items()):
            input = data
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
    application.logger.debug(to_be_logged)
    return result


@application.route('/info', methods=['GET'])
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


@application.route('/features', methods=['GET'])
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


@application.route('/preprocess', methods=['POST'])
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


@application.route('/health')
def health_check():
    return flask.Response("up", status=200)


@application.route('/ready')
def readiness_check():
    if model.is_ready():
        return flask.Response("ready", status=200)
    else:
        return flask.Response("not ready", status=503)


@application.route('/service-info')
@returns_json
def service_info():
    """Service information

    Get information about the service: up-time, varsion of the template, name
    of the served model, etc.

    """
    info = {
        'version-template': __version__,
        'running-since': SERVICE_START_TIMESTAMP,
        'serving-model-file': MODEL_NAME,
        'serving-model-family': model.family,
        'debug': DEBUG
    }
    return info


if __name__ == '__main__':
    application.run(
        debug=DEBUG,
        host=os.environ.get('HOST', 'localhost'),
        port=os.environ.get('PORT', '5000'))
