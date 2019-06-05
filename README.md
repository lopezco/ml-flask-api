# Python-Flask Docker template for Machine Learning model deployment including SHAP explanations
A simple example of a Python web service for real time machine learning model deployment.
It is based on [this post](https://mikulskibartosz.name/a-comprehensive-guide-to-putting-a-machine-learning-model-in-production-using-flask-docker-and-e3176aa8d1ce)

## Requirements  
* [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
* [docker-compose](https://docs.docker.com/compose/install/) (Recommended)

## Before using
Make sure that you have a model in the main directory.
You can launch the example using the following line in order to create a quick classification model.
```bash
$ python example/build_linear_model.py
```
or
```bash
$ python example/build_rf_model.py
```

## Run on docker
Build the image (this has to be done every time the code or the model change)
```bash
$ docker-compose build
```
Create and run the container
```bash
$ docker-compose build up
```

## Run on local Python environment
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
This example considers that the API was launched with the default parameters (localhost at port 5000) and its calling
the example model.

* Health (`/health`)
```bash
$ curl -X GET http://localhost:5000/health
up
```

* Is model ready? (`/ready`)
```bash
$ curl -X GET http://localhost:5000/ready
ready
```

* Get information about the model (`/info`)
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
    "class": "<class 'sklearn.ensemble.forest.RandomForestClassifier'>",
    "cls_name": "RandomForestClassifier",
    "cls_type": "<class 'sklearn.ensemble.forest.RandomForestClassifier'>",
    "is_explainable": false,
    "preprocessing_script": false
  }
}
```

* Prediction (`/predict`)
```bash
$ curl -d '{"feature1": 1, "feature2": 1, "feature3": 2}' -H "Content-Type: application/json" -X POST http://localhost:5000/predict
{
  "prediction": 0
}
```

* Predict probabilities (`/predict?proba=1` or `/predict_proba`)
```bash
$ curl -d '{"feature1": 1, "feature2": 1, "feature3": 2}' -H "Content-Type: application/json" -X POST "http://localhost:5000/predict?proba=1"
{
  "prediction": [
    [
      0.6606847344865265,
      0.3393152655134735
    ]
  ]
}
```


* Get features of the Model with features importances (`/features`)
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

* Get SHAP explanations (`/predict?proba=1&explain=1` or `/explain`)
```bash
$curl -d '{"feature1": 1, "feature2": 1, "feature3": 2}' -H "Content-Type: application/json" -X POST "http://localhost:5000/predict?proba=1&explain=1"
{
  "explanation": {
    "feature1": 0.10000000149011613,
    "feature2": 0.03333333383003871,
    "feature3": -0.1666666691501935
  },
  "prediction": {
    "0": 0.7,
    "1": 0.3
  }
}
```

## Files that can be configured
* ```variables.env```: Controls API parameters via environment variables
* ```requirements.txt```: Controls Python packages installed inside the container
* ```model.joblib```: Model saved inside a dictionary with this format
```json
{
    "model": trained_model,
    "metadata": {"features": [{"name": "feature1", "type": "numeric"},
                              {"name": "feature2", "type": "numeric", "default": -1},
                              {"name": "feature3", "type": "numeric"}]}
}
```
