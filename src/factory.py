import os
from .model import SklearnModel


class ModelFactory(object):
    available_models = (SklearnModel, )

    @classmethod
    def create_model(cls, model_name, model_type='SKLEARN_MODEL'):
        # Get current directory
        base_dir = os.getcwd()
        # Fix for documentation compilation
        if os.path.basename(base_dir) == 'docsrc':
            base_dir = os.path.dirname(base_dir)
        # Check if there is a model in the directory with the expected name
        model_path = os.path.join(base_dir, model_name)
        if not os.path.exists(model_path):
            raise RuntimeError("Model {} not found".format(model_path))
        else:
            # Model found! now create an instance
            for model_class in cls.available_models:
                if model_class.family == model_type:
                    return model_class(model_path)
