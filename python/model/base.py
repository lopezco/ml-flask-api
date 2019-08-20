import joblib
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


def _check(ready=True, explainable=False, task=None):
    def actual_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            # Check rediness
            if ready and not self.is_ready():
                raise RuntimeError('Model is not ready yet.')
            # Check explainable
            if explainable and not self._is_explainable:
                model_name = type(self._model).__name__
                raise ValueError('Model not supported for explanations: {}'.format(model_name))
            # Check for task
            if task is not None:
                self_task = self.task_type()
                if not getattr(self_task, '__ge__' if task.upper() == 'CLASSIFICATION' else '__eq__')(Task(task)):
                    raise RuntimeError('This method is not available for {} tasks'.format(self_task.name.lower()))
            # Execute function
            return func(*args, **kwargs)
        return wrapper
    return actual_decorator


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


class BaseModel(object):
    """Abstract class that handles the loaded model."""
    family = ''
    # Explainable models
    _explainable_models = tuple()

    def __init__(self, file_name):
        self._file_name = file_name
        self._is_ready = False
        self._model = None
        self._metadata = None
        self._task_type = None
        self._is_explainable = False

    # Abstract
    def _load(self):
        """Abstract method"""
        raise NotImplementedError()

    @_check()
    def _get_predictor(self):
        """Abstract method"""
        raise NotImplementedError()

    @_check(task='classification')
    def _get_class_names(self):
        """Abstract method"""
        raise NotImplementedError()

    @_check()
    def preprocess(self, features):
        """Abstract method"""
        raise NotImplementedError()

    @_check()
    def predict(self, features):
        """Abstract method"""
        raise NotImplementedError()

    @_check(task='classification')
    def predict_proba(self, features):
        """Abstract method"""
        raise NotImplementedError()

    @_check(explainable=True)
    def explain(self, features, samples=None):
        """Abstract method"""
        raise NotImplementedError()

    # Private
    def _hydrate(self, model, metadata):
        # Fill attributes
        self._model = model
        self._metadata = metadata
        self._is_ready = True
        # Hydrate class
        clf = self._get_predictor()
        # SHAP
        model_name = type(clf).__name__
        self._is_explainable = SHAP_AVAILABLE and (model_name in self._explainable_models)
        # Feature importances
        if hasattr(clf, 'feature_importances_'):
            importance = clf.feature_importances_
            for imp, feat in zip(importance, metadata['features']):
                feat['importance'] = imp
        # Set model task type
        if not hasattr(clf, 'classes_'):
            self._task_type = Task('REGRESSION')
        elif len(clf.classes_) <= 2:
            self._task_type = Task('BINARY_CLASSIFICATION')
        elif len(clf.classes_) > 2:
            self._task_type = Task('MULTILABEL_CLASSIFICATION')

    @_check()
    def _feature_names(self):
        return [variable['name'] for variable in self.features()]

    @_check()
    def _validate(self, input):
        if self.metadata.get('features') is None:
            raise AttributeError("Missing key 'features' in model's metadata")

        # Ensure input is lislike shaped
        input = self._get_list_from(input)
        # Get feature names in order
        feature_names = [f['name'] for f in self.metadata['features']]
        # Create an index to handle multiple samples input
        index = list(range(len(input)))
        # Create DataFrame
        df = pd.DataFrame(input, index=index, columns=feature_names)
        # Convert features to expected types
        for feature in self.metadata['features']:
            name, var_type = feature['name'], feature['type']
            default = feature.get('default', None)
            categories = feature.get('categories', None)
            if name not in df.columns:
                df[name] = default or np.nan
            else:
                if var_type == 'numeric':
                    var_type = float
                elif var_type == 'string':
                    var_type = str
                elif var_type == 'category':
                    if categories is not None:
                        var_type = CategoricalDtype(categories=categories, ordered=True)
                    else:
                        msg = 'Missing "categories" for "{}" in metadata'.format(name)
                        raise ValueError(msg)
                else:
                    msg = 'Unknown variable type: {}'.format(var_type)
                    raise ValueError(msg)

                if default is None:
                    df[name] =  df[name].astype(var_type)
                else:
                    df[name] =  df[name].fillna(default).astype(var_type)
            # TO DO: add more validation logic
        return df

    @property
    @_check()
    def _is_classification(self):
        return self._task_type >= Task('CLASSIFICATION')

    @property
    @_check()
    def _is_binary_classification(self):
        return self._task_type == Task('BINARY_CLASSIFICATION')

    @property
    @_check()
    def _is_multilabel_classification(self):
        return self._task_type == Task('MULTILABEL_CLASSIFICATION')

    @property
    @_check()
    def _is_regression(self):
        return self._task_type == Task('REGRESSION')

    # Private (static)
    @staticmethod
    def _get_list_from(data):
        if isinstance(data, dict):
            return [data]
        elif pd.api.types.is_list_like(data):
            return data
        else:
            return [data]

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
    @_check()
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

    @_check()
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

    @_check()
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
    @_check()
    def info(self):
        """Get model information.

        This function gives complete description of the model.
        The returned ibject contais the following keys:

            metadata (:class:`dict`): Model metadata (see :func:`~python.model.base.BaseModel.metadata`).

            model (:class:`dict`): Context information of the learnt model.
                type (:class:`str`):
                    Type of the underlying model object.
                predictor_type (:class:`str`):
                    It could be the same as 'type'. However, for
                    `sklearn.pipeline.Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_
                    it will output the class of the predictor inside it.
                is_explainable (:class:`bool`):
                    `True` if the model class allows SHAP explanations to be
                    computed.
                task (:class:`str`):
                    Task type. Either 'BINARY_CLASSIFICATION',
                    'MULTILABEL_CLASSIFICATION' or 'REGRESSION'
                class_names (:class:`list` or :class:`None`):
                    Class names if defined (for classification only).

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
        result['model'] = {
            'type': str(type(self._model)),
            'predictor_type': str(type(self._get_predictor())),
            'is_explainable': self._is_explainable,
            'task': self.task_type(as_text=True),
            'family': self.family
        }
        if self._is_classification:
            result['model']['class_names'] = self._get_class_names()
        return result
