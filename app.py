import sys
import logging
import os
from typing import List
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from catboost import CatBoostClassifier

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


# create object fastapi
app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_models():
    #model = CatBoostClassifier()
    #model.load_model("models/catclf.pkl")
    model = joblib.load("models/catclf.pkl")
    ctr_transformer = joblib.load("models/ctr_transformer.pkl")
    return model, ctr_transformer


@app.get("/healz")
def health() -> bool:
    model, ctr_transformer = load_models()
    return not (model is None) and not (ctr_transformer is None)


def make_predict(
    data: list,
    features: list,
    model: CatBoostClassifier,
    ctr_transformer: CtrTransformer,
) -> List[ClickResponse]:
    df = pd.DataFrame(data, columns=features)
    features = ctr_transformer.transform(df)
    predicted_proba, _ = predict_model(model, features)

    logger.debug("df.device_ip: ", df["device_ip"].values[0])
    logger.debug("predicted_proba", predicted_proba, predicted_proba[0, 1])

    return [
        ClickResponse(
            device_ip=df["device_ip"].values[0],
            click_proba=round(predicted_proba[0, 1], 4)
        )
    ]


@app.get("/predict/", response_model=List[ClickResponse])
def predict(request: AdOpportunity):
    model, ctr_transformer = load_models()
    return make_predict(request.data, request.features, model, ctr_transformer)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
