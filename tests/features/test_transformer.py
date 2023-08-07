import pytest
import logging
import sys
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.features.DeviceCountTransformer import DeviceCountTransformer
from src.features.UserCountTransformer import UserCountTransformer

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@pytest.fixture(scope="function")
def synthetic_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["1111", "22222", "33333", "44444", "55555"],
            "device_ip": ["ip1", "ip2", "ip1", "ip3", "ip2"],
            "device_id": ["id1", "id2", "id1", "id3", "id2"],
            "hour": pd.to_datetime(
                [
                    "2023-07-20 12:30:00",
                    "2023-07-20 13:45:00",
                    "2023-07-20 14:15:00",
                    "2023-07-21 09:00:00",
                    "2023-07-21 10:30:00",
                ]
            ),
            "click": [1, 0, 1, 1, 0],
        }
    )


@pytest.fixture(scope="function")
def time_transformer() -> FunctionTransformer:
    return FunctionTransformer(
        lambda df: pd.DataFrame(
            {
                "hour_of_day": df["hour"].dt.hour,
                "day_of_week": df["hour"].dt.dayofweek,
                "device_ip": df["device_ip"],
                "device_id": df["device_id"],
            }
        )
    )


@pytest.fixture(scope="function")
def device_transformer() -> ColumnTransformer:
    return ColumnTransformer(
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
        ]
    )


@pytest.fixture(scope="function")
def user_device_transformer(time_transformer: FunctionTransformer) -> Pipeline:
    user_count_transformer = UserCountTransformer()
    transformer = ColumnTransformer(
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
            ("time_transformer", time_transformer, ["hour", "device_ip", "device_id"],),
        ],
    )
    return Pipeline(
        steps=[
            ("transformer", transformer),
            ("user_count_transformer", user_count_transformer),
        ]
    )


def test_time_transformer(
    synthetic_dataset: pd.DataFrame, time_transformer: FunctionTransformer
):
    expected_hour = [12, 13, 14, 9, 10]
    expected_day = [3, 3, 3, 4, 4]
    timedf = time_transformer.fit_transform(synthetic_dataset)
    logger.info(f"timedf: \n{timedf}")

    assert timedf["hour_of_day"].values.tolist() == expected_hour
    assert timedf["day_of_week"].values.tolist() == expected_day


def test_device_transformer(
    synthetic_dataset: pd.DataFrame, device_transformer: DeviceCountTransformer
):
    expected_device_ip_count = [2, 2, 2, 1, 2]
    expected_device_id_count = [2, 2, 2, 1, 2]
    devicedf = device_transformer.fit_transform(synthetic_dataset)
    logger.info(f"devicedf: \n{devicedf}")

    assert devicedf[:, 0].tolist() == expected_device_ip_count
    assert devicedf[:, 1].tolist() == expected_device_id_count


def test_user_trnasformer(
    synthetic_dataset: pd.DataFrame, user_device_transformer: Pipeline
):
    expected_hourly_user_count = [1, 1, 1, 1, 1]
    userdf = user_device_transformer.fit_transform(synthetic_dataset)
    logger.info(f"userdf: \n{userdf}")

    assert userdf["hourly_user_count"].values.tolist() == expected_hourly_user_count
