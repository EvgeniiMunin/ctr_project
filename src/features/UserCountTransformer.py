import sys
import logging
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


class UserCountTransformer(BaseEstimator, TransformerMixin):
    """
    Pipeline feature transformer to count users per hour
    Should be applied after computation of hour features
    """

    def __init__(self):
        self.user_count_feature = []

    def fit(self, X, y=None):
        hour_of_day = X[:, 2]
        device_ip = X[:, 4]

        # groupby and count the number of users per unique hour of the day
        data_group = {}
        for hour in np.unique(hour_of_day):
            data_group[hour] = (device_ip[hour_of_day == hour]).shape[0]

        for hour in hour_of_day:
            self.user_count_feature.append(data_group[hour])

        return self

    def transform(self, X):
        res_arr = np.array(self.user_count_feature).reshape(-1, 1)
        count_df = pd.DataFrame(
            X,
            columns=[
                "device_ip_count",
                "device_id_count",
                "hour_of_day",
                "day_of_week",
                "device_ip",
                "device_id",
            ],
        )
        count_df["hourly_user_count"] = res_arr
        return count_df
