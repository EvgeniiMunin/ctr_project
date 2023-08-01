from dataclasses import dataclass, field
from typing import List


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    target_col: str = field(default="target")
