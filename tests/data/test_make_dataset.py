import numpy as np
import logging
import sys

from src.data.make_dataset import read_data, split_train_val_data
from src.entities.split_params import SplittingParams


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def test_load_dataset(dataset_path: str, target_col: str):
    print("dataset_path: ", dataset_path)
    data = read_data(dataset_path)
    assert data.shape[0] > 10
    assert target_col in data.columns


def test_split_dataset(dataset_path: str):
    splitting_params = SplittingParams(random_state=42, val_size=0.2)
    data = read_data(dataset_path)
    train, val = split_train_val_data(data, splitting_params)
    assert train.shape[0] > 10
    assert val.shape[0] > 10
    logger.info(
        f"check split: {val.shape[0] / train.shape[0]}, {np.allclose(val.shape[0] / train.shape[0], 0.2, atol=0.06)}"
    )
    assert np.allclose(val.shape[0] / train.shape[0], 0.2, atol=0.06)
