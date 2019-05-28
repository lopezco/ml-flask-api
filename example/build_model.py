from sklearn import linear_model
from sklearn.externals import joblib

model_path = './model.joblib'

model = linear_model.LinearRegression()
model.fit([[1.,1.,5.], [2.,2.,5.], [3.,3.,1.]], [0.,0.,1.])

with open(model_path, 'wb') as fo:
    joblib.dump(model, fo)
