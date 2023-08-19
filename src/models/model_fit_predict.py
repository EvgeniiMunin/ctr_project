import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from typing import Dict, Union, Tuple
import joblib

from src.entities.train_params import TrainingParams

Classifier = Union[CatBoostClassifier]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> Classifier:
    """Train and save model from configs"""

    model = CatBoostClassifier(
        n_estimators=train_params.n_estimators,
        learning_rate=train_params.learning_rate,
        depth=train_params.depth,
        random_seed=train_params.random_state,
        bagging_temperature=train_params.bagging_temperature,
        verbose=True,
    )
    model.fit(features, target)
    return model


def predict_model(
    model: CatBoostClassifier, features: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict model from configs"""
    predicted_proba = model.predict_proba(features)
    preds = np.argmax(predicted_proba, axis=1)
    return predicted_proba, preds


def evaluate_model(
    predicted_proba: np.ndarray, predicts: np.ndarray, target: pd.Series
) -> Dict[str, float]:
    """Evaluate model from configs"""
    return {
        "f1_score": f1_score(target, predicts, average="weighted"),
        "log_loss": log_loss(target, predicted_proba),
        "precision": precision_score(target, predicts),
        "recall": recall_score(target, predicts),
        "roc_auc_score": roc_auc_score(target, predicted_proba[:, 1]),
    }


def serialize_model(model, output: str) -> str:
    """Serialize model from configs"""
    with open(output, "wb") as file:
        joblib.dump(model, file)
    return output
