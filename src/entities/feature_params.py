from dataclasses import dataclass, field
from typing import List


@dataclass()
class FeatureParams:
    features: List[str]
    target_col: str = field(default="target")
