import pytest
import logging
import sys
import pandas as pd
from typing import Tuple

from src.features.CtrTransformer import CtrTransformer

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@pytest.fixture(scope="function")
def synthetic_dataset() -> Tuple[pd.DataFrame, list]:
    data = pd.DataFrame(
        {"feature": ["A", "B", "C", "C", "B", "B"], "click": [1, 0, 0, 1, 1, 1]}
    )
    logger.info(f"data: \n{data}")

    return data, ["feature"]


@pytest.fixture(scope="function")
def ctr_transformer(synthetic_dataset: Tuple[pd.DataFrame, list]) -> CtrTransformer:
    return CtrTransformer(synthetic_dataset[1])


def test_time_transformer(
    synthetic_dataset: Tuple[pd.DataFrame, list], ctr_transformer: CtrTransformer
):
    expected_processed_ctr = [0.58, 0.67, 0.5, 0.5, 0.67, 0.67]

    ctrdf = ctr_transformer.fit_transform(synthetic_dataset[0])
    logger.info(f"ctrdf: \n{ctrdf}")

    assert (
        ctrdf["feature"].apply(lambda x: round(x, 2)).values.tolist()
        == expected_processed_ctr
    )
