import lightgbm as lgb
import pandas as pd
import joblib


df = pd.DataFrame({"feature1": [1.,1.,5.], "feature2": [2.,2.,5.], "feature3": ["B","B","A"]})
df['feature3'] = df['feature3'].astype('category')

model_path = './model.joblib'

model = lgb.train(train_set=lgb.Dataset(df, label=[0, 0, 1]), params={'objective': 'binary'})
to_save = dict(model=model.model_to_string(),
               metadata={
                   'target_mapping': {0: 'Good', 1: 'Bad'},
                   "features": [
                       {"name": "feature1", "type": "numeric", "accepts_missing": False},
                       {"name": "feature2", "type": "numeric", "default": -1, "accepts_missing": True},
                       {"name": "feature3", "type": "category", "accepts_missing": True, "categories": ["A", "B"]}]})

with open(model_path, 'wb') as fo:
    joblib.dump(to_save, fo)
