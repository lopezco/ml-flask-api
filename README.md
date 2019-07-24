# Flask template for Machine Learning model deployment

A simple example of a Python web service for real time machine learning model deployment.
It is based on [this post](https://mikulskibartosz.name/a-comprehensive-guide-to-putting-a-machine-learning-model-in-production-using-flask-docker-and-e3176aa8d1ce)

This includes Docker integration and SHAP explanations for the deployed model.

[Website](https://lopezco.github.io/python-flask-sklearn-docker-template) | [Source](https://github.com/lopezco/python-flask-sklearn-docker-template/) | [Issues](https://github.com/lopezco/python-flask-sklearn-docker-template/issues)

## Installation

### Requirements  

* [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
* [docker-compose](https://docs.docker.com/compose/install/) (Recommended)

### Before using

Make sure that you have a model in the main directory.
You can launch the example using the following line in order to create a quick
classification model.
```bash
$ python example/build_linear_binary.py
```
or one of the scripts in the `example` folder

### Configuration

* ```variables.env```: Controls API parameters via environment variables
* ```requirements.txt```: Controls Python packages installed inside the container
* ```model.joblib```: Model saved inside a dictionary with this format

    ```javascript
    {
        "model": trained_model,
        "metadata": {"features": [{"name": "feature1", "type": "numeric"},
                                  {"name": "feature2", "type": "numeric", "default": -1},
                                  {"name": "feature3", "type": "numeric"}]}
    }
    ```

## Run the service

### On Docker

#### Development

Build the image (this has to be done every time the code or the model change)
```bash
$ docker-compose build
```
Create and run the container
```bash
$ docker-compose up
```

#### Production

Using uWSGI and nginx for production.

Build the image (this has to be done every time the code or the model change)
```bash
$ docker-compose -f docker-compose-production.yml build
```
Create and run the container
```bash
$ docker-compose -f docker-compose-production.yml up
```

### On local Python environment

Create the environment
```bash
$ conda create -n flask_ml_template python=3
$ conda activate flask_ml_template
```
Install requirements
```bash
$ pip install -r ./requirements-service.txt  
$ pip install -r ./requirements.txt  
```
Run the API service
```bash
$ python service.py  
```

## Usage of the API  

This example considers that the API was launched with the default parameters
(`localhost` at port `5000`) and its calling the example model.

For `/predict` endpoint the JSON string in the payload of hte request can take
two forms:

1. The first, the payload is a record or a list of records with one value
per feature. This will be directly interpreted as the input for the
model.

2. The second, the payload is a dictionary with 1 or 2 elements. The key
`_data` is mandatory because this will be the input for the model and
its format is expected to be a record or a list of records. On the
other hand, the key `_samples` (optional) will be used to obtain
different explanations.

If `_samples` is not given, then the explanations returned are the raw output of
the trees, which varies by model (for binary classification in XGBoost
this is the log odds ratio). On the contrary, if `_samples` is given,
then the explanations are the output of the model transformed into
probability space (note that this means the SHAP values now sum to the
probability output of the model).
See the [SHAP documentation](https://shap.readthedocs.io/en/latest/#shap.TreeExplainer)
for details.

### Check the API's health status

Endpoint: `/health`

```bash
$ curl -X GET http://localhost:5000/health
up
```

### Is model ready?

Endpoint: `/ready`

```bash
$ curl -X GET http://localhost:5000/ready
ready
```

### Get information about service

Endpoint: `/service-info`

```bash
$ curl -X GET http://localhost:5000/service-info
{
  "debug": true,
  "running-since": 1563355369.6482198,
  "serving-model": "model.joblib",
  "version-template": "1.2.0"
}
```

### Get information about the model

Endpoint: `/info`

```bash
$ curl -X GET http://localhost:5000/info
{
  "metadata": {
    "features": [
      {
        "default": -1,
        "importance": 0.2,
        "name": "feature1",
        "type": "numeric"
      },
      {
        "default": -1,
        "importance": 0.1,
        "name": "feature2",
        "type": "numeric"
      },
      {
        "default": -1,
        "importance": 0.3,
        "name": "feature3",
        "type": "numeric"
      }
    ]
  },
  "model": {
    "type": "<class 'sklearn.ensemble.forest.RandomForestClassifier'>",
    "predictor_type": "<class 'sklearn.ensemble.forest.RandomForestClassifier'>",
    "is_explainable": false,
    "task": "BINARY_CLASSIFICATION",
    "class_names": ["0", "1"]
  }
}
```

### Compute predictions

Endpoint: `/predict`

```bash
$ curl -d '[{"feature1": 1, "feature2": 1, "feature3": 2}, {"feature1": 1, "feature2": 1, "feature3": 2}]' -H "Content-Type: application/json" -X POST http://localhost:5000/predict
{
  "prediction": [0, 0]
}
```

### Predict probabilities

Endpoint: `/predict?proba=1`

```bash
$ curl -d '{"feature1": 1, "feature2": 1, "feature3": 2}' -H "Content-Type: application/json" -X POST "http://localhost:5000/predict?proba=1"
{
  "prediction": [{
    "0": 0.8,
    "1": 0.2
  }]
}
```


### Get features of the Model with features importances

Endpoint: `/features`
```bash
$ curl -X GET "http://localhost:5000/features"
[
  {
    "default": -1,
    "importance": 0.2,
    "name": "feature1",
    "type": "numeric"
  },
  {
    "default": -1,
    "importance": 0.1,
    "name": "feature2",
    "type": "numeric"
  },
  {
    "default": -1,
    "importance": 0.3,
    "name": "feature3",
    "type": "numeric"
  }
]
```

### Get SHAP explanations

Endpoint: `/predict?proba=1&explain=1`

```bash
$curl -d '{"feature1": 1, "feature2": 1, "feature3": 2}' -H "Content-Type: application/json" -X POST "http://localhost:5000/predict?proba=1&explain=1"
{
  "explanation": {
    "feature1": 0.10000000149011613,
    "feature2": 0.03333333383003871,
    "feature3": -0.1666666691501935
  },
  "prediction": [{
    "0": 0.7,
    "1": 0.3
  }]
}
```
