from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype


def metadata_from_dataframe(df):
    """Build metadata from a dataset

    Args:
        df (:class:`pandas.DataFrame`):
            File path of the serialized model. It must be a file that can be
            loaded using :mod:`joblib`

    Returns:
        dict:
            Information about the variables of the dataset.

    Raises:
        ValueError: If a type of variable is not handled by this function
    """
    metadata = []
    for c in df.columns:
        if is_categorical_dtype(df[c]):
            tmp = {'name': c, 'type': 'category',
                'categories': sorted(df[c].dtype.categories.values.tolist())})
        elif is_numeric_dtype(df[c]):
            tmp = {'name': c, 'type': 'numeric'}
        elif is_string_dtype(df[c]):
            tmp = {'name': c, 'type': 'string'}
        else:
            raise ValueError('Unknown type for {}'.format(c))
        metadata.append(tmp)
    return metadata
