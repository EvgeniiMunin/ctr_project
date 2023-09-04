import sys
import pandas as pd
import logging

from sklearn.base import BaseEstimator, TransformerMixin


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)


class CtrTransformer(BaseEstimator, TransformerMixin):
    """
    Pipeline feature transformer for CTR computation
    """

    def __init__(self, features: list = None):
        self.ctr_df = None
        self.mean_ctr = dict()
        self.vocab = dict()
        self.features = features

    def _response_fit(self, data, feature_name):
        # group data on each cat feature
        df_vocab = data.groupby([feature_name, "click"]).size().unstack()
        df_vocab["ctr"] = df_vocab[1] / (df_vocab[0] + df_vocab[1])

        # drop nans and compute avg CTR
        df_vocab.dropna(inplace=True)
        mean_ctr = df_vocab["ctr"].mean()

        # prepare dictionary for further transform
        keys = list(df_vocab.index)
        values = list(df_vocab["ctr"].values)
        vocab = {keys[i]: values[i] for i in range(len(keys))}

        return vocab, mean_ctr

    def _response_transform(self, X: pd.DataFrame, name: str):
        vector = []
        for row in X:
            vector.append(self.vocab[name].get(row, self.mean_ctr[name]))
        return vector

    def fit(self, X: pd.DataFrame, y=None):
        for column_name in self.features:
            vocab_feat, mean_ctr_feat = self._response_fit(X, column_name)
            self.vocab[column_name] = vocab_feat
            self.mean_ctr[column_name] = mean_ctr_feat
        return self

    def transform(self, X: pd.DataFrame):
        self.ctr_df = pd.DataFrame()
        for column_name in self.features:
            self.ctr_df[column_name] = self._response_transform(
                X[column_name], column_name
            )
        return self.ctr_df
