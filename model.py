from sklearn.base import TransformerMixin


class Feature(TransformerMixin):
    """Feature Interface."""
    def fit(self, X, y=None, **fit_params):
        return self


class ToMatrix(Feature):
    """Transforms the features dict to a matrix"""
    def __init__(self, features):
        self.features = features

    def transform(self, X):
        final_X = []

        for instance in X:
            sent_vector = []
            for feat in self.features:
                if isinstance(instance['features'][feat], list):
                    sent_vector += instance['features'][feat]
                else:
                    sent_vector.append(instance['features'][feat])
            final_X.append(sent_vector)
        return final_X