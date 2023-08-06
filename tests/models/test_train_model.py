import os
import joblib
import logging
import sys
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from datetime import datetime

from src.data.make_dataset import read_data
from src.entities.feature_params import FeatureParams
from src.entities.train_params import TrainingParams
from src.features.build_transformer import (
    build_transformer,
    process_count_features,
    build_ctr_transformer,
    extract_target,
)
from src.models.model_fit_predict import train_model, serialize_model

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@pytest.fixture()
def dataset() -> pd.DataFrame:
    return read_data("sampled_train_50k.csv")


@pytest.fixture()
def target_col():
    return "click"


def test_train_model(dataset: pd.DataFrame):
    dataset["hour"] = dataset.hour.apply(
        lambda val: datetime.strptime(str(val), "%y%m%d%H")
    )
    feature_params = FeatureParams(["1", "2"], "click")

    transformer = build_transformer(feature_params)
    ctr_transformer = build_ctr_transformer(feature_params)

    processed_data = process_count_features(transformer, dataset)
    logger.info(
        f"processed_data:  {processed_data.shape} \n {processed_data.info()} \n {processed_data.nunique()}"
    )

    train_features = ctr_transformer.fit_transform(processed_data)
    train_target = extract_target(processed_data, feature_params)

    model = train_model(train_features, train_target, TrainingParams())

    assert isinstance(model, CatBoostClassifier)
    assert model.predict(train_features).shape[0] == train_target.shape[0]


def test_serialization_model(tmpdir):
    expected_output = tmpdir.join("model.pkl")
    model = CatBoostClassifier()
    real_output = serialize_model(model, expected_output)
    assert real_output == expected_output
    assert os.path.exists
    with open(real_output, "rb") as f:
        model = joblib.load(f)
    assert isinstance(model, CatBoostClassifier)
