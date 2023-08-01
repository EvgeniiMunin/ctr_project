import json
import logging
import sys
import argparse

import pandas as pd

from src.data.make_dataset import read_data, split_train_val_data

from src.entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(config_path: str):
    training_pipeline_params: TrainingPipelineParams = read_training_pipeline_params(config_path)

    data: pd.DataFrame = read_data(training_pipeline_params.input_data_path)
    logger.info(f"Start train pipeline with params {training_pipeline_params}")
    logger.info(f"Data.shape is  {data.shape}")

    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )
    logger.info(f"train_df.shape is  {train_df.shape}")
    logger.info(f"val_df.shape is  {val_df.shape}")

    # prepare train features
    transformer = build_transformer(training_pipeline_params.feature_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, default="configs/train_config_lr.yaml"
    )
    args = parser.parse_args()
    train_pipeline(args.config)