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
from src.features.build_features import extract_target, process_features
from src.models.model_fit_predict import train_model, predict_model, evaluate_model, serialize_model

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


    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_df)

    # prepare train features
    train_features = process_features(transformer, train_df)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)
    logger.info(f"train_features.shape is  {train_features.shape}")

    # prepare val features
    val_features = process_features(transformer, val_df)
    val_target = extract_target(val_df, training_pipeline_params.feature_params)
    logger.info(f"val_features.shape is  {val_features.shape}")

    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )

    predicted_proba, preds = predict_model(model, val_features)
    metrics = evaluate_model(predicted_proba, preds, val_target)
    logger.debug(f"preds/ targets shapes:  {(preds.shape, val_target.shape)}")

    # dump metrics to json
    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"Metric is {metrics}")

    # serialize model
    path_to_model = serialize_model(model, training_pipeline_params.output_model_path)

    return path_to_model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, default="configs/train_config_lr.yaml"
    )
    args = parser.parse_args()
    train_pipeline(args.config)