#!flask/bin/python
import os
from flask import Flask, Response, jsonify, request, redirect, url_for
from model import Model
from time import time
import json

app = Flask(__name__)

DEBUG = os.environ.get('DEBUG', True)

# Load model
base_dir = os.getcwd()
model_path = os.path.join(base_dir, os.environ.get('MODEL_NAME', 'model.joblib'))
if not os.path.exists(model_path):
    raise RuntimeError("Model {} not found".format(model_path))
else:
    model = Model(model_path)

app.logger.info('Loading model...')
model.load_model()


@app.route('/predict', methods=['POST'])
def predict():
    input = json.loads(request.data or '{}')
    try:
        input = model.validate(input)
    except ValueError as err:
        return Response(str(err), status=400)
    # Parameters
    output_proba = int(request.args.get('proba', 0))
    output_explanation = int(request.args.get('explain', 0))
    # Predict
    before_time = time()
    try:
        prediction = model.predict_proba(input) if output_proba else model.predict(input)
    except Exception as err:
        return Response(str(err), status=500)
    result = {'prediction': prediction}
    # Eplain
    if output_explanation:
        try:
            explanation = model.explain(input)
        except Exception as err:
            return Response(str(err), status=500)
        else:
            result['explanation'] = explanation
    after_time = time()
    # log
    to_be_logged = {
        'input': request.data,
        'params': request.args,
        'request_id': request.headers.get('X-Correlation-ID'),
        'result': result,
        'model': model.metadata,
        'elapsed_time': after_time - before_time
    }
    app.logger.info(to_be_logged)
    return jsonify(result)


@app.route('/predict_proba', methods=['POST'])
def predict_proba():
    return redirect(url_for('predict', proba=1))


@app.route('/explain', methods=['POST'])
def explain():
    return redirect(url_for('predict', proba=1, explain=1))


@app.route('/info',  methods=['GET'])
def info():
    try:
        data = model.info
    except Exception as err:
        return Response(str(err), status=500)
    return jsonify(data)


@app.route('/features',  methods=['GET'])
def features():
    try:
        features = model.features()
    except Exception as err:
        return Response(str(err), status=500)

    return jsonify(features)


@app.route('/health')
def health_check():
    return Response("up", status=200)


@app.route('/ready')
def readiness_check():
    if model.is_ready():
        return Response("ready", status=200)
    else:
        return Response("not ready", status=503)


if __name__ == '__main__':
    app.run(
        debug=DEBUG,
        host=os.environ.get('HOST', 'localhost'),
        port=os.environ.get('PORT', '5000'))
