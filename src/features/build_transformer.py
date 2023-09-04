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


def build_transformer() -> Pipeline:
    """
    Build pipeline feature transformer consisting of
    time_transformer
    device_trnasformer
    user_count_transformer

    param:
    return:
        pipeline: Pipeline
    """

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

    original_features = ["id", "hour", "device_ip", "device_id"]
    device_time_transformer = ColumnTransformer(
        transformers=[
            (
                "device_ip_count",
                DeviceCountTransformer("device_ip"),
                original_features,
            ),
            (
                "device_id_count",
                DeviceCountTransformer("device_id"),
                original_features,
            ),
            ("time_transformer", time_transformer, original_features[1:]),
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
    """
    Build pipeline feature transformer for CTR computation
    param:
        params: FeatureParams: list of features to process
    return:
        ctr_transformer: CtrTransformer
    """

    feature_names = params.ctr_features
    ctr_transformer = CtrTransformer(feature_names)
    logger.info(f"ctr_transformer: \n {ctr_transformer}")

    return ctr_transformer


def process_count_features(
    transformer: Pipeline, df: pd.DataFrame, params: FeatureParams = None,
) -> pd.DataFrame:
    """
    Apply the count_transformer on dataframe
    param:
        transformer: Pipeline: count_transformer
        df: pd.DataFrame: input df
        params: FeatureParams: list of features to process
    return:
        pd.DataFrame: dataframe with processed features
    """

    counts_df = transformer.fit_transform(df)
    return pd.concat([df, counts_df[params.count_features]], axis=1)


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    return df[params.target_col]
