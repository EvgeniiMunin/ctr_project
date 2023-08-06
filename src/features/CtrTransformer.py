import sys
import pandas as pd
import logging

from sklearn.base import BaseEstimator, TransformerMixin


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)


class CtrTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feats: list = None):
        self.mean_ctr = dict()
        self.vocab = dict()
        self.feats = feats

    def _response_fit(self, data, feature_name):
        df_vocab = data.groupby([feature_name, "click"]).size().unstack()
        df_vocab["ctr"] = df_vocab[1] / (df_vocab[0] + df_vocab[1])

        df_vocab.dropna(inplace=True)
        mean_ctr = df_vocab["ctr"].mean()

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
        for name in self.feats:
            vocab_feat, mean_ctr_feat = self._response_fit(X, name)
            self.vocab[name] = vocab_feat
            self.mean_ctr[name] = mean_ctr_feat
        return self

    def transform(self, X: pd.DataFrame):
        self.xpctr = pd.DataFrame()
        for name in self.feats:
            self.xpctr[name] = self._response_transform(X[name], name)
        return self.xpctr
