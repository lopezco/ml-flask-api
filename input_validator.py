from flask import Response


def validate_features(f):
    parameter_names = ['feature1', 'feature2', 'feature3']
    @wraps(f)
    def wrapper(*args, **kw):
        for parameter in parameter_names:
            to_be_validated = request.args.get(parameter)
            try:
                number_to_validate = int(to_be_validated)
                if number_to_validate < 0 or number_to_validate > 1:
                    raise ValueError('Value must be 0 or 1.')
            except ValueError as err:
                return Response(str(err), status = 400)
         return f(*args, **kw)
    return wrapper
