# -*- coding: utf-8 -*-
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from src.entities.split_params import SplittingParams


def read_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_train_val_data(
    data: pd.DataFrame, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, val_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )

    return train_data, val_data


if __name__ == "__main__":
    df = pd.read_csv("data/sampled_preprocessed_train_5m.csv")
    dfsample = df.sample(frac=0.01)
    print("dfsample: ", dfsample.shape, "\n", dfsample.info())

    dfsample.to_csv("data/sampled_preprocessed_train_50k.csv")