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
logger.addHandler(handler)


def device_counts(data: pd.DataFrame, column_name: str):
    # count the number of ads per unique user ( device ip )
    data_group = data[[column_name, 'id']].groupby([column_name]).count()

    # make a column with values with device id counts
    ip_count_feature = []
    for index in tqdm(data[column_name]):
        ip_count_feature.append(data_group['id'][index])

    # finally add the column to the dataframe
    data[f'{column_name}_counts'] = ip_count_feature

    return data


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
        return pd.DataFrame(self.device_count_feature)


class UserCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.user_count_feature = []
        self.data_group = None

    def fit(self, X: pd.DataFrame, y=None):
        print("UserCountTransformer X: \n", X, type(X))

        #data_group = X[['hour_of_day', 'device_ip']].groupby(['hour_of_day']).count()
        data_group = X.groupby(['hour_of_day']).count()
        for index in tqdm(X['hour_of_day']):
            self.user_count_feature.append(data_group['device_ip'][index])
        return self

    def transform(self, X: pd.DataFrame):
        return pd.DataFrame(self.user_count_feature)


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


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
        })
    )

    transformer = ColumnTransformer(
        transformers=[
            ("device_ip_count", DeviceCountTransformer("device_ip"), ["id", "device_ip"]), # OK
            ("device_id_count", DeviceCountTransformer("device_id"), ["id", "device_id"]), # OK
            ("time_transformer", time_transformer, ["hour"]), # OK
        ],
        #remainder=DataFrameSelector()
    )

    user_count_transformer = UserCountTransformer()

    pipeline = Pipeline(
        steps=[
            ("transformer", transformer),
            ("user_count_transformer", user_count_transformer),
        ]
    )
    print("pipeline: \n", pipeline)

    timedf = time_transformer.fit_transform(df) # OK
    print("resdf: \n", timedf, type(timedf))

    transdf = transformer.fit_transform(df)
    print("transdf: \n", transdf, type(transdf))

    resdf = pd.DataFrame(pipeline.fit_transform(df)) # columns=["device_ip_count", "device_id_count"]
    print("resdf: \n", resdf, type(resdf))

