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
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, Counter, Histogram

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
Instrumentator().instrument(app).expose(app)
proba_gauge = Gauge("predicted_proba", "Predicted price")
predict_request_counter = Counter(
    "http_predict_request_total", "Total HTTP Predict Requests"
)
proba_hist = Histogram(
    "predicted_proba_hist",
    "Histogram of predicted click probas",
    buckets=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
)


@app.get("/")
def main():
    return "it is entry point of our predictor"


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

    # set metrics for prometheus
    proba_gauge.set(round(predicted_proba[0, 1], 4))
    predict_request_counter.inc()
    proba_hist.observe(round(predicted_proba[0, 1], 4))

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
