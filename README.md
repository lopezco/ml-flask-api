# Python-Flask Docker template for MAchine Learning model deployment
A simple example of python api for real time machine learning.
On init, a model inside `./data` folder is loaded (if any) else a simple linear regression model is created.
For more information read [this post](https://mikulskibartosz.name/a-comprehensive-guide-to-putting-a-machine-learning-model-in-production-using-flask-docker-and-e3176aa8d1ce)

# requirements  
docker installed

# Run on docker
docker build . -t {some tag name}  -f ./Dockerfile  
detached : docker run -p 5000:5000 -d {some tag name}  
interactive (recommended for debug): docker run -p 3000:5000 -it {some tag name}  

# Run on local computer
conda create -n flask_ml_template python=3
conda activate flask_ml_template
pip install -r ./requirements.txt  
python service.py  

# Use sample api  
127.0.0.1:3000/isAlive  
127.0.0.1:3000/prediction/api/v1.0/some_prediction?f1=4&f2=4&f3=4  
