import joblib
import numpy as np
import sys
from threading import Thread
from copy import deepcopy

try:
    import shap
except ImportError:
    SHAP_AVAILABLE = False
else:
    SHAP_AVAILABLE = True


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
        # Explainability
        self._shap_models = ['Booster', 'Booster']
        for m in ('RandomForest', 'XGB', 'CatBoost', 'LGBM', 'DecisionTree'):
            self._shap_models.extend([m + 'Classifier', m + 'Regressor'])

    # Private
    def _load_model(self):
        loaded = joblib.load(self._file_name)
        self._model = loaded['model']
        self._metadata = loaded['metadata']
        self._class_names = loaded['metadata'].get('class_names', None)
        self._is_ready = True

        self._preprocessing = loaded.get('preprocessing', None)
        if hasattr(self.get_classifier(), 'feature_importances_'):
            importance = self.get_classifier().feature_importances_
            for imp, feat in zip(importance, loaded['metadata']['features']):
                feat['importance'] = imp
        is_shap_model = type(self.get_classifier()).__name__ in self._shap_models
        self._is_explainable = SHAP_AVAILABLE and is_shap_model

    # Public
    def load_model(self):
        Thread(target=self._load_model).start()

    def is_ready(self):
        return self._is_ready

    @property
    @_check_if_model_is_ready
    def metadata(self):
        return self._metadata

    @property
    @_check_if_model_is_ready
    def info(self):
        result = {}
        # Metadata
        result['metadata'] = self._metadata
        # Info Form model
        classifier_type = type(self.get_classifier())
        result['model'] = {
            'class': str(type(self._model)),
            'cls_type': str(classifier_type),
            'cls_name': classifier_type.__name__,
            'is_explainable': self._is_explainable,
            'preprocessing_script': self._preprocessing is not None,
            'class_names': self._class_names
        }
        return result

    @_check_if_model_is_ready
    def preprocess(self, input):
        return input if self._preprocessing is None else self._preprocessing(input)

    @_check_if_model_is_ready
    def predict(self, features):
        input = np.asarray(list(features.values())).reshape(1, -1)
        input = self.preprocess(input)
        result = int(self._model.predict(input)[0])
        if self._class_names is not None:
            result =  self._class_names[result]
        return result

    @_check_if_model_is_ready
    def predict_proba(self, features):
        input = np.asarray(list(features.values())).reshape(1, -1)
        input = self.preprocess(input)
        prediction = self._model.predict_proba(input)[0].tolist()
        if self._class_names is None:
            result = {c: v for c, v in enumerate(prediction)}
        else:
            result = {c: v for c, v in zip(self._class_names, prediction)}
        return result

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

    @_check_if_model_is_ready
    def get_classifier(self):
        model_name = type(self._model).__name__
        if model_name == 'Pipeline':
            return self._model.steps[-1][1]
        else:
            return self._model

    @_check_if_model_is_ready
    def explain(self, features):
        if not self._is_explainable:
            raise RuntimeError('Model {} is not supported for explanations'.format(type(self._model).__name__))
        input = np.asarray(list(features.values())).reshape(1, -1)
        # Apply pre-processing
        preprocessed = self.preprocess(input)
        if hasattr(self._model, 'transform'):
            preprocessed = self._model.transform(preprocessed)
        explainer = shap.TreeExplainer(self.get_classifier())
        shap_values = explainer.shap_values(preprocessed)[0][0].tolist()
        features = [x['name'] for x in self.features()]
        explanations = {f: v for f, v in zip(features, shap_values)}
        return explanations
