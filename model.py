import joblib
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
        self._metadata = None

    # Private
    def _load_model(self):
        loaded = joblib.load(self._file_name)
        self._model = loaded['model']
        self._metadata = loaded['metadata']
        self._is_ready = True

    # Public
    @property
    def metadata(self):
        if not self.is_ready():
            raise RuntimeError('Model is not ready yet.')

        return self._metadata

    def load_model(self):
        Thread(target=self._load_model).start()

    def is_ready(self):
        return self._is_ready

    def predict(self, features):
        if not self.is_ready():
            raise RuntimeError('Model is not ready yet.')

        input = np.asarray(list(features.values())).reshape(1, -1)
        print(input)
        result = self._model.predict(input)
        return int(result[0])

    def predict_proba(self, features):
        if not self.is_ready():
            raise RuntimeError('Model is not ready yet.')

        input = np.asarray(list(features.values())).reshape(1, -1)
        result = self._model.predict_proba(input)
        return result.tolist()

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
