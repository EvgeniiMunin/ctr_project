import sys
import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.entities.feature_params import FeatureParams
from tqdm.notebook import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)


class CtrCategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feats: list = None):
        self.mean_ctr = dict()
        self.vocab = dict()
        self.xpctr = pd.DataFrame()
        self.feats = feats

    def _response_fit(self, data, feature_name):
        # print("data: ", data.shape, feature_name, "\n", data[feature_name], "\n", data.head())

        df_vocab = data.groupby([feature_name, 'click']).size().unstack()
        df_vocab['ctr'] = df_vocab[1] / (df_vocab[0] + df_vocab[1])
        # print("df_vocab: \n", df_vocab, "\n", df_vocab.info(), type(df_vocab))
        # print("df_vocab[0]: \n", df_vocab.iloc[0, :])

        # compute the mean CTR to substitute CTR for those feature values which will not be found in the train data
        df_vocab.dropna(inplace=True)
        mean_ctr = df_vocab['ctr'].mean()
        # print("mean_ctr: \n", mean_ctr)

        # create a dictionary with keys= feature category names and values = respective CTR values
        keys = list(df_vocab.index)
        values = list(df_vocab['ctr'].values)
        vocab = {keys[i]: values[i] for i in range(len(keys))}
        # print("vocab: \n", vocab)

        return vocab, mean_ctr

    def _response_transform(self, X: pd.DataFrame, name: str):
        vector = []
        for row in X:
            vector.append(self.vocab[name].get(row, self.mean_ctr[name]))
        return vector

    def fit(self, X: pd.DataFrame, y=None):
        for name in tqdm(self.feats):
            # print('fit: ', name)
            vocab_feat, mean_ctr_feat = self._response_fit(X, name)
            self.vocab[name] = vocab_feat
            self.mean_ctr[name] = mean_ctr_feat

            # print('self.vocab: ', self.vocab)
            # print('self.mean_ctr: ', self.mean_ctr)
        return self

    def transform(self, X: pd.DataFrame):
        for name in tqdm(self.feats):
            # print('transform: ', name)
            self.xpctr[name] = self._response_transform(X[name], name)
        return self.xpctr


if __name__ == "__main__":
    df = pd.read_csv("data/raw/sampled_preprocessed_train_50k.csv")
    print("dfsample: ", df.shape, "\n", df.info())
    print("resdf: \n", df, type(df))
    print("df.nunique: \n", df.nunique())

    feature_names = [
        'site_id', 'site_domain', 'site_category', 'app_id', 'app_category', 'app_domain',
        'device_model', 'device_type', 'device_conn_type', 'device_id_counts', 'device_ip_counts',
        'banner_pos', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
        'hour_of_day', 'day_of_week', 'hourly_user_count'
    ]

    #feature_names = [
    #    'hourly_user_count'
    #]

    ctr_transformer = CtrCategoricalTransformer(feature_names)
    resdf = ctr_transformer.fit_transform(df)
    print("resdf: \n", resdf, type(resdf))
    print("resdf.nunique: \n", resdf.nunique())


