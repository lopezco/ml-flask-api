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


def _check_readiness(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        if self.is_ready():
            return func(*args, **kwargs)
        else:
            raise RuntimeError('Model is not ready yet.')
    return wrapper


def _check_task(task):
    def actual_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self_task = args[0].task_type()
            strict = task.upper() != 'CLASSIFICATION'
            target_task = Task(task)
            if (strict and (self_task == target_task)) or \
                (not strict and (self_task >= target_task)):
                return func(*args, **kwargs)
            else:
                raise RuntimeError('This method is not available for {} tasks'.format(self_task.name.lower()))
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


class Model(object):
    """Class that handles the loaded model.

    This class can handle models that respect the scikit-learn API. This
    includes `sklearn.pipeline.Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_.

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
    # Explainable models
    _explainable_models = (
        # Sklearn
        'DecisionTreeClassifier', 'DecisionTreeRegressor',
        'RandomForestClassifier', 'RandomForestRegressor',
        # XGBoost
        'XGBClassifier', 'XGBRegressor', 'Booster',
        # CatBoost
        'CatBoostClassifier', 'CatBoostRegressor',
        # LightGBM
        'LGBMClassifier', 'LGBMRegressor')

    def __init__(self, file_name):
        self._file_name = file_name
        self._is_ready = False
        self._model = None
        self._metadata = None
        self._task_type = None

    # Private
    def _load(self):
        # Load serialized model (dict expected)
        loaded = joblib.load(self._file_name)
        self._model = loaded['model']
        self._metadata = loaded['metadata']
        self._is_ready = True
        # Hydrate class
        clf = self._get_predictor()
        # SHAP
        model_name = type(clf).__name__
        self._is_explainable = SHAP_AVAILABLE and (model_name in self._explainable_models)
        # Feature importances
        if hasattr(clf, 'feature_importances_'):
            importance = clf.feature_importances_
            for imp, feat in zip(importance, loaded['metadata']['features']):
                feat['importance'] = imp
        # Set model task type
        if not hasattr(clf, 'classes_'):
            self._task_type = Task('REGRESSION')
        elif len(clf.classes_) <= 2:
            self._task_type = Task('BINARY_CLASSIFICATION')
        elif len(clf.classes_) > 2:
            self._task_type = Task('MULTILABEL_CLASSIFICATION')

    @_check_readiness
    def _get_predictor(self):
        return Model._extract_base_predictor(self._model)

    @_check_readiness
    @_check_task('classification')
    def _get_class_names(self):
        return np.array(self._get_predictor().classes_, str)

    @_check_readiness
    def _feature_names(self):
        return [variable['name'] for variable in self.features()]

    @_check_readiness
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
            default = feature.get('default', None)
            categories = feature.get('categories', None)
            if name not in df.columns:
                df[name] = default or np.nan
            else:
                if var_type == 'numeric':
                    var_type = float
                elif var_type == 'string':
                    var_type = str
                elif (var_type == 'category') and (categories is not None):
                    var_type = CategoricalDtype(categories=categories,
                                                ordered=True)
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
    @_check_readiness
    def _is_classification(self):
        return self._task_type >= Task('CLASSIFICATION')

    @property
    @_check_readiness
    def _is_binary_classification(self):
        return self._task_type == Task('BINARY_CLASSIFICATION')

    @property
    @_check_readiness
    def _is_multilabel_classification(self):
        return self._task_type == Task('MULTILABEL_CLASSIFICATION')

    @property
    @_check_readiness
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

    @staticmethod
    def _extract_base_predictor(model):
        model_name = type(model).__name__
        if model_name == 'Pipeline':
            return Model._extract_base_predictor(model.steps[-1][1])
        elif 'CalibratedClassifier' in model_name:
            return Model._extract_base_predictor(model.base_estimator)
        else:
            return model

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
    @_check_readiness
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

    @_check_readiness
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

    @_check_readiness
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
    @_check_readiness
    def info(self):
        """Get model information.

        This function gives complete description of the model.
        The returned ibject contais the following keys:

            metadata (:class:`dict`): Model metadata (see :func:`~model.Model.metadata`).

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
            'task': self.task_type(as_text=True)
        }
        if self._is_classification:
            result['model']['class_names'] = self._get_class_names()
        return result

    @_check_readiness
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

    @_check_readiness
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

    @_check_readiness
    @_check_task('classification')
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
            RuntimeError: If the model isn't ready or the task isn't classification.
        """
        input = self._validate(features)
        prediction = self._model.predict_proba(input)
        df = pd.DataFrame(prediction, columns=self._get_class_names())
        return df.to_dict(orient='records')

    @_check_readiness
    def explain(self, features, samples=None):
        """Explain the prediction of a model.

        Explanation function that returns the SHAP value for each feture.
        The returned object contais one value per feature of the model.

        If `samples` is not given, then the explanations are the raw output of
        the trees, which varies by model (for binary classification in XGBoost
        this is the log odds ratio). On the contrary, if `sample` is given,
        then the explanations are the output of the model transformed into
        probability space (note that this means the SHAP values now sum to the
        probability output of the model).
        See the `SHAP documentation <https://shap.readthedocs.io/en/latest/#shap.TreeExplainer>`_ for details.

        Args:
            features (dict): Record to be used as input data to explain the
                model. The expected object must contain one key per
                feature. Example:
                {'feature1': 5, 'feature2': 'A', 'feature3': 10}
            samples (dict): Records to be used as a sample pool for the
                explanations. It must have the same structure as `features`
                parameter. According to SHAP documentation, anywhere from 100
                to 1000 random background samples are good sizes to use.
        Returns:
            dict: Explanations.

        Raises:
            RuntimeError: If the model is not ready.
            ValueError: If the model' predictor doesn't support SHAP
                explanations or the model is not already loaded.
                Or if the explainer outputs an unknown object
        """
        if not self._is_explainable:
            model_name = type(self._model).__name__
            msg = 'Model not supported for explanations: {}'.format(model_name)
            raise ValueError(msg)
        # Process input
        input = self._validate(features)
        preprocessed = self.preprocess(input)
        # Define parameters
        if samples is None:
            params = {
                'feature_dependence': 'tree_path_dependent',
                'model_output': 'margin'}
        else:
            params = {
                'data': self.preprocess(self._validate(samples)),
                'feature_dependence': 'independent',
                'model_output': 'probability'}
        # Explainer
        explainer = shap.TreeExplainer(self._get_predictor(), **params)
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
                result[c] = pd.DataFrame(_values, index=index,
                                         columns=colnames).to_dict(orient='records')
        else:  # self._is_regression
            result = pd.DataFrame(shap_values, index=index,
                                  columns=colnames).to_dict(orient='records')
        return result
