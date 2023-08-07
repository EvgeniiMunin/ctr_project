from dataclasses import dataclass, field
from typing import List


@dataclass()
class FeatureParams:
    count_features: List[str]
    ctr_features: List[str]
    target_col: str = field(default="target")
