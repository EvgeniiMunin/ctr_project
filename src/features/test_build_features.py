import sys
import pandas as pd
import logging
import numpy as np
from tqdm.notebook import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from src.entities.feature_params import FeatureParams

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


class CtrCategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feats: list = None):
        self.mean_ctr = dict()
        self.vocab = dict()
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

    def fit(self, X: pd.DataFrame, y=None):
        for name in tqdm(self.feats):
            # print('fit: ', name)
            vocab_feat, mean_ctr_feat = self._response_fit(X, name)
            self.vocab[name] = vocab_feat
            self.mean_ctr[name] = mean_ctr_feat

            # print('self.vocab: ', self.vocab)
            # print('self.mean_ctr: ', self.mean_ctr)
        return self

    def _response_transform(self, X: pd.DataFrame, name: str):
        vector = []
        for row in X:
            vector.append(self.vocab[name].get(row, self.mean_ctr[name]))
        return vector

    def transform(self, X: pd.DataFrame):
        self.xpctr = pd.DataFrame()
        for name in tqdm(self.feats):
            # print('transform: ', name)
            self.xpctr[name] = self._response_transform(X[name], name)
        return self.xpctr


class DeviceCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str):
        self.device_count_feature = []
        self.column_name = column_name
        self.data_group = None

    def fit(self, X: pd.DataFrame, y=None):
        # count the number of ads per unique user ( device ip )
        data_group = X[[self.column_name, 'id']].groupby([self.column_name]).count()

        # make a column with values with device id counts
        for index in tqdm(X[self.column_name]):
            self.device_count_feature.append(data_group['id'][index])

        return self

    def transform(self, X: pd.DataFrame):
        tempdf = pd.DataFrame(self.device_count_feature, columns=[f"{self.column_name}_count"])
        #return pd.concat([X, tempdf], axis=1)
        return pd.DataFrame(self.device_count_feature, columns=[f"{self.column_name}_count"])


class UserCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.user_count_feature = []

    def fit(self, X, y=None):
        print("UserCountTransformer X: \n", X, type(X))

        # Extract the required columns from the input NumPy array
        hour_of_day = X[:, 2]
        device_ip = X[:, 4]

        print("hour_of_day: ", hour_of_day)
        print("device_ip: ", device_ip)
        print("np.sum(device_ip[hour_of_day == hour]): ", device_ip[hour_of_day == 12])

        # Groupby and count the number of users per unique hour of the day
        data_group = {}
        for hour in np.unique(hour_of_day):
            data_group[hour] = (device_ip[hour_of_day == hour]).shape[0]

        for hour in hour_of_day:
            self.user_count_feature.append(data_group[hour])

        print("data_group: ", data_group)
        print("user_count_feature: ", self.user_count_feature)

        return self

    def transform(self, X):
        # return pd.DataFrame(self.user_count_feature, columns=["hourly_user_count"])
        # return pd.DataFrame(res_arr, columns=["hourly_user_count"])
        res_arr = np.array(self.user_count_feature).reshape(-1, 1)
        leftdf = pd.DataFrame(X, columns=[
            "device_ip_count", "device_id_count", "hour_of_day", "day_of_week", "device_ip", "device_id"
        ])
        leftdf["hourly_user_count"] = res_arr
        return leftdf


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def process_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(transformer.transform(df))


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    return df[params.target_col]


if __name__ == "__main__":
    df = pd.DataFrame({
        "id": ["1111", "22222", "33333", "44444", "55555"],
        'device_ip': ['ip1', 'ip2', 'ip1', 'ip3', 'ip2'],
        'device_id': ['id1', 'id2', 'id1', 'id3', 'id2'],
        'hour': pd.to_datetime(
            ['2023-07-20 12:30:00', '2023-07-20 13:45:00', '2023-07-20 14:15:00', '2023-07-21 09:00:00',
             '2023-07-21 10:30:00']),
        'click': [1, 0, 1, 1, 0]
    })
    print("df: \n", df.head(), type(df))

    time_transformer = FunctionTransformer(
        lambda df: pd.DataFrame({
            "hour_of_day": df["hour"].dt.hour,
            "day_of_week": df["hour"].dt.dayofweek,
            "device_ip": df["device_ip"],
            "device_id": df["device_id"]
        })
    )

    device_transformer = DeviceCountTransformer("device_ip")

    transformer = ColumnTransformer(
        transformers=[
            ("device_ip_count", DeviceCountTransformer("device_ip"), ["id", "hour", "device_ip", "device_id"]), # OK
            ("device_id_count", DeviceCountTransformer("device_id"), ["id", "hour", "device_ip", "device_id"]), # OK
            ("time_transformer", time_transformer, ["hour", "device_ip", "device_id"]), # OK
        ],
        #remainder="passthrough"
    )
    #print("transformer: \n", transformer)

    user_count_transformer = UserCountTransformer()

    pipeline = Pipeline(
        steps=[
            ("transformer", transformer),
            ("user_count_transformer", user_count_transformer),
        ]
    )
    print("pipeline: \n", pipeline)

    #timedf = time_transformer.fit_transform(df) # OK
    #print("timedf: \n", timedf, type(timedf))

    #devicedf = device_transformer.fit_transform(df)
    #print("devicedf: \n", devicedf, type(devicedf))

    transdf = transformer.fit_transform(df)
    print("transdf: \n", transdf, type(transdf)) # OK output np.array

    resdf = pipeline.fit_transform(df)
    print("resdf: \n", resdf, type(resdf), resdf.info()) # OK


