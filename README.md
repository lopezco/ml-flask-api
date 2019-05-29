# Python-Flask Docker template for MAchine Learning model deployment
A simple example of python api for real time machine learning.
It is base on [this post](https://mikulskibartosz.name/a-comprehensive-guide-to-putting-a-machine-learning-model-in-production-using-flask-docker-and-e3176aa8d1ce)

## Requirements  
docker installed
docker-compose installed

## Before using
Pleas put a model in the main directory.
For this yoi can launch the example using
```bash
python example/build_model.py
```

## Run on docker
```bash
docker-compose build
docker-compose build up
```

## Run on local computer
```bash
conda create -n flask_ml_template python=3
conda activate flask_ml_template
pip install -r ./requirements-service.txt  
pip install -r ./requirements.txt  
python service.py  
```

## Use sample api  
127.0.0.1:5000/health
127.0.0.1:5000/predict
127.0.0.1:5000/predict?output_proba=1

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
