from sklearn.externals import joblib
import numpy as np
import sys
from threading import Thread


class Model:
    def __init__(self, file_name):
        def get_last_column(X):
            return X[:, -1].reshape(-1, 1)

        setattr(sys.modules['__main__'], 'get_last_column', get_last_column)
        self._file_name = file_name
        self._is_ready = False
        self._model = None
        self._meta_data = None

    # Private
    def _load_model(self):
        loaded = joblib.load(self._file_name)
        self._model = loaded['model']
        self._meta_data = loaded['metadata']
        self._is_ready = True

    # Public
    @property
    def meta_data(self):
        return self._meta_data

    def load_model(self):
        Thread(target=self._load_model).start()

    def is_ready(self):
        return self._is_ready

    def predict(self, features):
        if not self.is_ready():
            raise RuntimeError('Model is not ready yet.')

        input = np.asarray(features).reshape(1, -1)
        result = self._model.predict(input)
        return int(result[0])

    def predict_proba(self, features):
        if not self.is_ready():
            raise RuntimeError('Model is not ready yet.')

        input = np.asarray(features).reshape(1, -1)
        result = self._model.predict_proba(input)
        return result
