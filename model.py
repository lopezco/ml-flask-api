import joblib
import sys
import numpy as np
import pandas as pd

from pandas.api.types import CategoricalDtype
from threading import Thread
from copy import deepcopy
from functools import wraps
from enum import Enum


try:
    import shap
except ImportError:
    SHAP_AVAILABLE = False
else:
    SHAP_AVAILABLE = True


def _check_if_model_is_ready(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        if self.is_ready():
            return func(*args, **kwargs)
        else:
            raise RuntimeError('Model is not ready yet.')
    return wrapper


class Task(int):
    _REGRESSION, _CLASSIFICATION = 0, 1
    _BINARY_CLASSIFICATION, _MULTILABEL_CLASSIFICATION = 2, 3

    def __new__(cls, name):
        assert(isinstance(name, str))
        try:
            val = getattr(cls, '_{}'.format(name.upper()))
        except AttributeError:
            raise AttributeError('Unknown task-name: {}'.format(name))
        else:
            return  super(Task, cls).__new__(cls, val)

    def __init__(self, name):
        self.name = name.upper()
        self._id = int(self)

    def __repr__(self):
        return "Task('{}')".format(self._name)


class Model(object):
    """Class that handles the loaded model.

    This class can handle models that respect the scikit-learn API. This
    includes :class:`sklearn.pipeline.Pipeline`.

    The data coming from a request if validated using the metadata setored with
    the model. The data fed to the `predict`, `predict_proba`, `explain` handle
    `preprocess` should be a dictionary that object must contain one key per
    feature or a list of such dictionaries (recors).
    Example: `{'feature1': 5, 'feature2': 'A', 'feature3': 10}`

    Args:
        file_name (str):
            File path of the serialized model. It must be a file that can be
            loaded using :mod:`joblib`
    """

    def __init__(self, file_name):
        def get_last_column(X):
            return X[:, -1].reshape(-1, 1)

        setattr(sys.modules['__main__'], 'get_last_column', get_last_column)
        self._file_name = file_name
        self._is_ready = False
        self._model = None
        self._metadata = None
        self._task_type = None
        # Explainability
        self._shap_models = ['Booster', 'Booster']
        for m in ('RandomForest', 'XGB', 'CatBoost', 'LGBM', 'DecisionTree'):
            self._shap_models.extend([m + 'Classifier', m + 'Regressor'])

    # Private
    def _load(self):
        # Load serialized model (dict expected)
        loaded = joblib.load(self._file_name)
        self._model = loaded['model']
        self._metadata = loaded['metadata']
        self._is_ready = True
        # Hydrate class
        cls = self._get_classifier()
        # SHAP
        is_shap_model = type(cls).__name__ in self._shap_models
        self._is_explainable = SHAP_AVAILABLE and is_shap_model
        # Feature importances
        if hasattr(cls, 'feature_importances_'):
            importance = cls.feature_importances_
            for imp, feat in zip(importance, loaded['metadata']['features']):
                feat['importance'] = imp
        # Set model types
        if not hasattr(self._get_classifier(), 'classes_'):
            self._task_type = Task('REGRESSION')
        elif len(self._get_classifier().classes_) <= 2:
            self._task_type = Task('BINARY_CLASSIFICATION')
        elif len(self._get_classifier().classes_) > 2:
            self._task_type = Task('MULTILABEL_CLASSIFICATION')

    @_check_if_model_is_ready
    def _get_classifier(self):
        model_name = type(self._model).__name__
        if model_name == 'Pipeline':
            return self._model.steps[-1][1]
        else:
            return self._model

    @_check_if_model_is_ready
    def _get_class_names(self):
        return np.array(self._get_classifier().classes_, str)

    @_check_if_model_is_ready
    def _feature_names(self):
        return [variable['name'] for variable in self.features()]

    @_check_if_model_is_ready
    def _validate(self, input):
        if self.metadata.get('features') is None:
            raise AttributeError("Missing key 'features' in model's metadata")

        # Ensure input is lislike shaped
        _, input = self._is_listlike(input)
        # Get feature names in order
        feature_names = [f['name'] for f in self.metadata['features']]
        # Create an index to handle multiple samples input
        index = list(range(len(input)))
        # Create DataFrame
        df = pd.DataFrame(input, index=index, columns=feature_names)
        # Convert features to expected types
        for feature in self.metadata['features']:
            name, var_type = feature['name'], feature['type']
            default = feature.get('default', np.nan)
            categories = feature.get('categories', None)
            if name not in df.columns:
                df[name] = default
            else:
                if var_type == 'numeric':
                    df[name] =  df[name].astype(float)
                elif var_type == 'string':
                    df[name] =  df[name].astype(str)
                elif var_type == 'category':
                    if categories is not None:
                        var_type = CategoricalDtype(categories=categories,
                                                    ordered=True)
                    df[name] =  df[name].astype(var_type)
                else:
                    msg = 'Unknown variable type: {}'.format(var_type)
                    raise ValueError(msg)
            # TO DO: add validation logic
        return df

    @property
    @_check_if_model_is_ready
    def _is_classification(self):
        return self._task_type >= Task('CLASSIFICATION')

    @property
    @_check_if_model_is_ready
    def _is_binary_classification(self):
        return self._task_type == Task('BINARY_CLASSIFICATION')

    @property
    @_check_if_model_is_ready
    def _is_multilabel_classification(self):
        return self._task_type == Task('MULTILABEL_CLASSIFICATION')

    @property
    @_check_if_model_is_ready
    def _is_regression(self):
        return self._task_type == Task('REGRESSION')

    # Private (static)
    @staticmethod
    def _is_listlike(data):
        _is_listlike = pd.api.types.is_list_like(data)
        is_dict = isinstance(data, dict)
        is_input_listlike = _is_listlike and not is_dict
        if not is_input_listlike:
            data = [data]
        return is_input_listlike, data

    # Public
    def load(self):
        """Launch model loading in a separated thread

        Once it finishes, the instance `_is_ready` parameter is set to `True`.

        The loaded object is expected to be a :class:`dict` containing the
        following keys: `model` (model object) and `metadata` (:class:`dict`).
        The later contains one or two elements: `features`
        (:class:`list` of :class:`dict`) with at least the `name` and `type` of
        the variables and optional `class_names` (:class:`list` of :class:`str`)
        with the list of class-names in order (for classification).
        """
        Thread(target=self._load).start()

    def is_ready(self):
        """Check if model is already loaded.

        Returns:
            bool:
                Is the model already loaded and ready for predictions?
        """
        return self._is_ready

    @property
    @_check_if_model_is_ready
    def metadata(self):
        """Get metadata of the model_name.

        Returns:
            dict:
                Metadata of the model containing information about the features
                and classes (optional)

        Raises:
            RuntimeError: If the model is not ready.
        """
        return self._metadata

    @_check_if_model_is_ready
    def task_type(self, as_text=False):
        """Get task type of the model

        Either 'REGRESSION', 'CLASSIFICATION', 'BINARY_CLASSIFICATION' or
        'MULTILABEL_CLASSIFICATION'.

        Returns:
            :class:`Task` or :class:`str`:
                If `as_text=False`, returns the task of the model
                (classification, regression, etc.) as a :class:`Task` class
                instance. If `as_text=True`, returns the task of the model as
                text.

        Raises:
            RuntimeError: If the model is not ready.
        """
        return self._task_type.name if as_text else self._task_type

    @_check_if_model_is_ready
    def features(self):
        """Get the features of the model

        The returned list contains records. Each record contais (at least)
        the `name` and `type` of the variable. If the model supports
        feature importances calculation (if the clasifier has
        `feature_importances_` atribute), they will also be present.

        Returns:
            list[dict]:
                Model features.

        Raises:
            RuntimeError: If the model is not ready.
        """
        return deepcopy(self.metadata['features'])

    @property
    @_check_if_model_is_ready
    def info(self):
        """Get model information.

        This function gives complete description of the model.
        The returned ibject contais the following keys:

            metadata (:class:`dict`): Model metadata (see :func:`~model.Model.metadata`).

            model (:class:`dict`): Context information of the learnt model.
                class (:class:`str`):
                    Type of the underlying model object.
                cls_type (:class:`str`):
                    Classifier type. It could be the same as 'class'. However, for
                    :class:`sklearn.pipeline.Pipeline` it will output the class of
                    the classifier inside it.
                cls_name (:class:`str`):
                    Classifier name.
                is_explainable (:class:`bool`):
                    `True` if the model class allows SHAP explanations to be
                    computed.
                task (:class:`str`):
                    Task type. Either 'BINARY_CLASSIFICATION',
                    'MULTILABEL_CLASSIFICATION' or 'REGRESSION'
                class_names (:class:`list` or :class:`None`):
                    Class names if defined.

        Returns:
            dict:
                Information about the model.

        Raises:
            RuntimeError: If the model is not ready.
        """
        result = {}
        # Metadata
        result['metadata'] = self._metadata
        # Info from model
        classifier_type = type(self._get_classifier())
        result['model'] = {
            'class': str(type(self._model)),
            'cls_type': str(classifier_type),
            'cls_name': classifier_type.__name__,
            'is_explainable': self._is_explainable,
            'task': self.task_type(as_text=True)
        }
        if self._is_classification:
            result['model']['class_names'] = self._get_class_names()
        return result

    @_check_if_model_is_ready
    def preprocess(self, input):
        """Preprocess data

        This function is used before prediction or interpretation.

        Args:
            input (dict):
                The expected object must contain one key per feature.
                Example: `{'feature1': 5, 'feature2': 'A', 'feature3': 10}`

        Returns:
            dict:
                Processed data if a preprocessing function was definded in the
                model's metadata. The format must be the same as the input.

        Raises:
            RuntimeError: If the model is not ready.
        """
        if hasattr(self._model, 'transform'):
            return self._model.transform(input)
        else:
            return input

    @_check_if_model_is_ready
    def predict(self, features):
        """Make a prediciton

        Prediction function that returns the predicted class. The returned value
        is an integer when the class names are not expecified in the model's
        metadata.

        Args:
            features (dict):
                Record to be used as input data to make predictions. The
                expected object must contain one key per feature.
                Example: `{'feature1': 5, 'feature2': 'A', 'feature3': 10}`

        Returns:
            int or str:
                Predicted class.

        Raises:
            RuntimeError: If the model is not ready.
        """
        input = self._validate(features)
        result = self._model.predict(input)
        return result

    @_check_if_model_is_ready
    def predict_proba(self, features):
        """Make a prediciton

        Prediction function that returns the probability of the predicted
        classes. The returned object contais one value per class. The keys of
        the dictionary are the classes of the model.

        Args:
            features (dict): Record to be used as input data to make
                predictions. The expected object must contain one key per
                feature. Example:
                {'feature1': 5, 'feature2': 'A', 'feature3': 10}

        Returns:
            dict: Predicted class probabilities.

        Raises:
            RuntimeError: If the model is not ready.
        """
        # Test for model task
        if self._is_regression:
            raise ValueError("Can't predict probabilities of regression model")
        input = self._validate(features)
        prediction = self._model.predict_proba(input)
        colnames = self._get_class_names()
        df = pd.DataFrame(prediction, columns=colnames)
        return df.to_dict(orient='records')

    @_check_if_model_is_ready
    def explain(self, features):
        """Explain the prediction of a model.

        Explanation function that returns the SHAP value for each feture.
        The returned object contais one value per feature of the model.

        Args:
            features (dict): Record to be used as input data to explain the
                model. The expected object must contain one key per
                feature. Example:
                {'feature1': 5, 'feature2': 'A', 'feature3': 10}

        Returns:
            dict: Explanations.

        Raises:
            RuntimeError: If the model is not ready.
            ValueError: If the model classifier doesn't support SHAP
                explanations or the model is not ready.
                Or if the explainer outputs an unknown object
        """
        if not self._is_explainable:
            model_name = type(self._model).__name__
            msg = 'Model not supported for explanations: {}'.format(model_name)
            raise ValueError(msg)
        input = self._validate(features)
        # Apply pre-processing
        preprocessed = self.preprocess(input)
        # Explainer
        explainer = shap.TreeExplainer(self._get_classifier())
        colnames = self._feature_names()
        shap_values = explainer.shap_values(preprocessed[colnames].values)

        # Create an index to handle multiple samples input
        index = preprocessed.index
        result = {}
        if self._is_classification:
            class_names = self._get_class_names()
            if isinstance(shap_values, list):
                # The result is one set of explanations per target class
                process_shap_values = False
            elif isinstance(shap_values, np.ndarray) and self._is_binary_classification:
                # The result is one ndarray set of explanations for one class
                # Expected only for binary classification for some models.
                # Ex: LGBMClassifier
                process_shap_values = True
            else:
                raise ValueError('Unknown objet class for shap_values variable')
            # Format output
            for i, c in enumerate(class_names):
                if process_shap_values:
                    _values = shap_values * (-1 if i == 0 else 1)
                else:
                    _values = shap_values[i]
                result[c] = pd.DataFrame(_values,
                                         index=index,
                                         columns=colnames).to_dict(orient='records')
        else:  # self._is_regression
            result = pd.DataFrame(shap_values, index=index,
                                  columns=colnames).to_dict(orient='records')
        return result
