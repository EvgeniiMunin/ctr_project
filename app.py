import sys
import logging
import os
from typing import List
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from catboost import CatBoostClassifier

from src.entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from src.features.CtrTransformer import CtrTransformer
from src.models.model_fit_predict import predict_model

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


class AdOpportunity(BaseModel):
    data: list
    features: list


class ClickResponse(BaseModel):
    device_ip: str
    click_proba: float


app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_models(training_pipeline_params: TrainingPipelineParams):
    model = joblib.load(training_pipeline_params.output_model_path)
    ctr_transformer = joblib.load(training_pipeline_params.output_ctr_transformer_path)
    return model, ctr_transformer


@app.get("/health")
def check_models(training_pipeline_params: TrainingPipelineParams):
    model, ctr_transformer = load_models(training_pipeline_params)
    if model is None or ctr_transformer is None:
        logger.error("app/check_models models are None")
        raise HTTPException(status_code=400, detail="Models are unavailable")


@app.get("/check_schema")
def check_schema(features: list, training_pipeline_params: TrainingPipelineParams):
    if not set(training_pipeline_params.feature_params.ctr_features).issubset(
        set(features)
    ):
        logger.error("app/check_schema missing columns")
        raise HTTPException(
            status_code=400, detail=f"Missing features in schema {features}"
        )


def make_predict(
    data: list,
    features: list,
    model: CatBoostClassifier,
    ctr_transformer: CtrTransformer,
    training_pipeline_params: TrainingPipelineParams,
) -> List[ClickResponse]:
    check_schema(features, training_pipeline_params)

    df = pd.DataFrame(data, columns=features)

    features = ctr_transformer.transform(df)
    predicted_proba, _ = predict_model(model, features)

    logger.debug("df.device_ip: ", df["device_ip"].values[0])
    logger.debug("predicted_proba", predicted_proba, predicted_proba[0, 1])

    return [
        ClickResponse(
            device_ip=df["device_ip"].values[0],
            click_proba=round(predicted_proba[0, 1], 4),
        )
    ]


@app.post("/predict/", response_model=List[ClickResponse])
def predict(request: AdOpportunity):
    logger.debug("app/predict run")

    config_path = "configs/train_config.yaml"
    training_pipeline_params: TrainingPipelineParams = read_training_pipeline_params(
        config_path
    )
    logger.debug(f"app/predict training_pipeline_params: {training_pipeline_params}")

    check_models(training_pipeline_params)
    logger.debug("app/predict check_models passed")

    model, ctr_transformer = load_models(training_pipeline_params)

    return make_predict(
        request.data, request.features, model, ctr_transformer, training_pipeline_params
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
