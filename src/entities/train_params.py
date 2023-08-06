from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    model_type: str = field(default="CatBoostClassifier")
    random_state: int = field(default=42)
    n_estimators: int = field(default=100)
    learning_rate: float = field(default=0.05)
    depth: int = field(default=5)
    bagging_temperature: float = field(default=0.2)
