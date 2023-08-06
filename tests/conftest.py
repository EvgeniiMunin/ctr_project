import os
import logging
import sys
import pytest


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@pytest.fixture()
def dataset_path():
    curdir = os.path.dirname(__file__)
    logger.info(curdir)
    return os.path.join(curdir, "sampled_data_unit_test.csv")


@pytest.fixture()
def target_col():
    return "click"
