import flask
import json
import numpy as np
import pandas as pd

from functools import wraps


class ExtendedEncoder(flask.json.JSONEncoder):
    """Encoder of numpy primitives and Pandas objects into JSON strings"""
    primitives = (np.ndarray, np.integer, np.inexact)

    def default(self, obj):
        if isinstance(obj, np.flexible):
            return None if isinstance(obj, np.void) else obj.tolist()
        elif isinstance(obj, self.primitives):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return json.JSONEncoder.default(self, obj.to_frame())
        return json.JSONEncoder.default(self, obj)


def returns_json(f):
    """Wraps a function to transform the output into a JSON string with a
    specific encoder"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        r = f(*args, **kwargs)
        if isinstance(r, flask.Response):
            return r
        else:
            return flask.Response(json.dumps(r, cls=ExtendedEncoder), status=200,
                                  mimetype='application/json; charset=utf-8')
    return decorated_function
