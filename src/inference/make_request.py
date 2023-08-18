import logging
import sys
from time import sleep

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


dataset_path = "data/raw/sampled_preprocessed_train_50k.csv"


def read_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


if __name__ == "__main__":
    data = read_data(dataset_path)

    for i in range(10):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]
        logger.info(f"check request_data: {request_data}")
        logger.info(f"check data.columns: {list(data.columns)}")

        response = requests.post(
            "http://0.0.0.0:8000/predict/",
            json={"data": [request_data], "features": list(data.columns)},
        )

        # server is working, can see output
        logger.info(f"check response.status_code: {response.status_code}")
        logger.info(f"check response.json(): {response.json()}\n")

        sleep(1)
