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
                   {"name": "feature1", "type": "numeric", "accepts_missing": False},
                   {"name": "feature2", "type": "numeric", "default": -1, "accepts_missing": True},
                   {"name": "feature3", "type": "category", "accepts_missing": True, "categories": ["A", "B"]}]})

with open(model_path, 'wb') as fo:
    joblib.dump(to_save, fo)
