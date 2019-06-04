import joblib
import numpy as np
import sys
from threading import Thread
from copy import deepcopy


def _check_if_model_is_ready(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        if self.is_ready():
            return func(*args, **kwargs)
        else:
            raise RuntimeError('Model is not ready yet.')
    return wrapper


class Model:
    def __init__(self, file_name):
        def get_last_column(X):
            return X[:, -1].reshape(-1, 1)

        setattr(sys.modules['__main__'], 'get_last_column', get_last_column)
        self._file_name = file_name
        self._is_ready = False
        self._model = None
        self._metadata = None
        self._preprocessing = None

    # Private
    def _load_model(self):
        loaded = joblib.load(self._file_name)
        self._model = loaded['model']
        self._metadata = loaded['metadata']
        self._preprocessing = loaded.get('preprocessing', None)
        if hasattr(self._model, 'feature_importances_'):
            importance = self._model.feature_importances_
            for imp, feat in zip(importance, loaded['metadata']['features']):
                feat['importance'] = imp
        self._is_ready = True

    # Public
    def load_model(self):
        Thread(target=self._load_model).start()

    def is_ready(self):
        return self._is_ready

    @property
    @_check_if_model_is_ready
    def metadata(self):
        return self._metadata

    @_check_if_model_is_ready
    def preprocess(self, input):
        return input if self._preprocessing is None else self._preprocessin(input)

    @_check_if_model_is_ready
    def predict(self, features):
        input = np.asarray(list(features.values())).reshape(1, -1)
        input = self.preprocess(input)
        result = self._model.predict(input)
        return int(result[0])

    @_check_if_model_is_ready
    def predict_proba(self, features):
        input = np.asarray(list(features.values())).reshape(1, -1)
        input = self.preprocess(input)
        result = self._model.predict_proba(input)
        return result.tolist()

    @_check_if_model_is_ready
    def features(self):
        return deepcopy(self.metadata.get('features', []))

    @_check_if_model_is_ready
    def validate(self, input):
        output = {}
        for feature in self.metadata['features']:
            name, var_type, default = feature['name'], feature['type'], feature.get('default', np.nan)
            value = input.get(name)
            if var_type == 'numeric':
                output[name] =  float(value) if value is not None else default
            elif var_type == 'string':
                output[name] = value or default
            else:
                raise ValueError('Unknown variable type in metadata: {}'.format(var_type))
            # TO DO: add validation logic
        return output
