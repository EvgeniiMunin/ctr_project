import sys
import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

from src.entities.feature_params import FeatureParams
from src.features.UserCountTransformer import UserCountTransformer
from src.features.DeviceCountTransformer import DeviceCountTransformer
from src.features.CtrTransformer import CtrTransformer

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def build_transformer(params: FeatureParams) -> Pipeline:
    time_transformer = FunctionTransformer(
        lambda df: pd.DataFrame(
            {
                "hour_of_day": df["hour"].dt.hour,
                "day_of_week": df["hour"].dt.dayofweek,
                "device_ip": df["device_ip"],
                "device_id": df["device_id"],
            }
        )
    )

    device_time_transformer = ColumnTransformer(
        transformers=[
            (
                "device_ip_count",
                DeviceCountTransformer("device_ip"),
                ["id", "hour", "device_ip", "device_id"],
            ),
            (
                "device_id_count",
                DeviceCountTransformer("device_id"),
                ["id", "hour", "device_ip", "device_id"],
            ),
            ("time_transformer", time_transformer, ["hour", "device_ip", "device_id"]),
        ],
    )

    user_count_transformer = UserCountTransformer()

    pipeline: Pipeline = Pipeline(
        steps=[
            ("device_time_transformer", device_time_transformer),
            ("user_count_transformer", user_count_transformer),
        ]
    )
    logger.info(f"build_time_device_transformer: \n {pipeline}")

    return pipeline


def build_ctr_transformer(params: FeatureParams) -> CtrTransformer:
    feature_names = [
        "site_id",
        "site_domain",
        "site_category",
        "app_id",
        "app_category",
        "app_domain",
        "device_model",
        "device_type",
        "device_conn_type",
        "device_id_count",
        "device_ip_count",
        "banner_pos",
        "C1",
        "C14",
        "C15",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
        "C21",
        "hour_of_day",
        "day_of_week",
        "hourly_user_count",
    ]
    ctr_transformer = CtrTransformer(feature_names)
    logger.info(f"ctr_transformer: \n {ctr_transformer}")

    return ctr_transformer


def process_count_features(
    transformer: Pipeline, df: pd.DataFrame, params: FeatureParams = None,
) -> pd.DataFrame:
    count_features = [
        "device_ip_count",
        "device_id_count",
        "hour_of_day",
        "day_of_week",
        "hourly_user_count",
    ]
    transdf = transformer.fit_transform(df)
    return pd.concat([df, transdf[count_features]], axis=1)


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    return df[params.target_col]
