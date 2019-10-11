from lightgbm import LGBMClassifier
import pandas as pd
import joblib


df = pd.DataFrame({"feature1": [1.,1.,5.], "feature2": [2.,2.,5.], "feature3": ["B","B","A"]})
df['feature3'] = df['feature3'].astype('category')

model_path = './model.joblib'

model = LGBMClassifier()
model.fit(df, [0, 0, 1])
to_save = dict(model=model,
               metadata={"features": [
                   {"name": "feature1", "type": "numeric"},
                   {"name": "feature2", "type": "numeric", "default": -1},
                   {"name": "feature3", "type": "category", "categories": ["A", "B"]}]})

with open(model_path, 'wb') as fo:
    joblib.dump(to_save, fo)
