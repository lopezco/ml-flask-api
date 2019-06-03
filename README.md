# Python-Flask Docker template for Machine Learning model deployment
A simple example of python api for real time machine learning.
It is base on [this post](https://mikulskibartosz.name/a-comprehensive-guide-to-putting-a-machine-learning-model-in-production-using-flask-docker-and-e3176aa8d1ce)

## Requirements  
* [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
* [docker-compose](https://docs.docker.com/compose/install/) (Recommended)

## Before using
Make sure that you have a model in the main directory.
You can launch the example using the following line in order to create a quick model
```bash
$ python example/build_model.py
```

## Run on docker
Build th eimage (this has to be done every time the code or the model change)
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
This example considers that the API was launched with the default parameters (localhost at port 5000) and its calling the example model.

* Health
```bash
$ curl -X GET http://localhost:5000/health
up
```

* Is model ready?
```bash
$ curl -X GET http://localhost:5000/ready
ready
```

* Prediction
```bash
$ curl -d '{"feature1": 1, "feature2": 1, "feature3": 2}' -H "Content-Type: application/json" -X POST http://localhost:5000/predict
{
  "prediction": 0
}
```

* Predict probabilities
```bash
$ curl -d '{"feature1": 1, "feature2": 1, "feature3": 2}' -H "Content-Type: application/json" -X POST "http://localhost:5000/predict?output_proba=1"
{
  "prediction": [
    [
      0.6606847344865265,
      0.3393152655134735
    ]
  ]
}
```
 
## Files that can be configured
* ```variables.env```: Controls API parameters via environment variables
* ```requirements.txt```: Controls Python packages installed inside the container
* ```model.joblib```: Model saved inside a dictionary with this fomat
```python
{
    'model': model,
    'metadata': {'features': [{'name': 'feature1', 'type': 'numeric'},
                              {'name': 'feature2', 'type': 'numeric', 'default': -1},
                              {'name': 'feature3', 'type': 'numeric'}]}
}
```
