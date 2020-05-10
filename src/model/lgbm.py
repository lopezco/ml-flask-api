import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from .base import BaseModel, _check

try:
    import shap
except ImportError:
    pass

# https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py#L82-L84


class LGBMModel(BaseModel):
    """Class that handles the loaded model.

    This class can handle models that respect the train API of LightGBM.

    The data coming from a request if validated using the metadata stored with
    the model. The data fed to the `predict`, `predict_proba`, `explain` handle
    `preprocess` should be a dictionary that object must contain one key per
    feature or a list of such dictionaries (records).
    Example: `{'feature1': 5, 'feature2': 'A', 'feature3': 10}`

    Args:
        file_name (str):
            File path of the serialized model. It must be a file that can be
            loaded using :mod:`joblib`
    """
    family = 'LGBM_MODEL'

    # Explainable models
    _explainable_models = ('Booster', )

    # Private
    def _load(self):
        loaded = joblib.load(self._file_name)
        # load model to predict
        model = lgb.Booster(model_str=loaded['model'])
        self._hydrate(model, loaded['metadata'])

    @_check()
    def _get_predictor(self):
        return self._model

    @_check(task='classification')
    def _get_class_names(self):
        return np.array(self._metadata.get('target_mapping').values(), str)

    # Public
    @_check()
    def preprocess(self, features):
        """Preprocess data

        This function is used before prediction or interpretation.

        Args:
            features (dict):
                The expected object must contain one key per feature.
                Example: `{'feature1': 5, 'feature2': 'A', 'feature3': 10}`

        Returns:
            dict:
                Processed data if a preprocessing function was defined in the
                model's metadata. The format must be the same as the input.

        Raises:
            RuntimeError: If the model is not ready.
        """
        return self._validate(features)

    @_check()
    def predict(self, features):
        """Make a prediction

        Prediction function that returns the predicted class. The returned value
        is an integer when the class names are not specified in the model's
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
        # TODO : set threshold
        return result

    @_check(task='classification')
    def predict_proba(self, features):
        """Make a prediction

        Prediction function that returns the probability of the predicted
        classes. The returned object contains one value per class. The keys of
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
        prediction = self._model.predict(input)
        df = pd.DataFrame(prediction, columns=self._get_class_names())
        return df.to_dict(orient='records')

    @_check(explainable=True)
    def explain(self, features, samples=None):
        """Explain the prediction of a model.

        Explanation function that returns the SHAP value for each feature.
        The returned object contains one value per feature of the model.

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
        # Process input
        preprocessed = self.preprocess(features)
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
        # This patch will ensure that the data will be fed as a pandas DataFrame
        # instead of as a numpy array to some models. Ex: LightGBM
        input_data = preprocessed[colnames]
        predictor_type = self._get_predictor_type()
        use_pandas = any(c in predictor_type for c in ('LGBMClassifier', 'LGBMRegressor'))
        shap_values = explainer.shap_values(input_data if use_pandas else input_data.values)

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
                raise ValueError('Unknown object class for shap_values variable')
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
