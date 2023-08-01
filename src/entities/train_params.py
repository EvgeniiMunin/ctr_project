from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    model_type: str = field(default="RandomForestClassifier")
    random_state: int = field(default=42)

    # RF params
    max_depth: int = field(default=5)

    # LR params
    solver: str = field(default="lbfgs")
    reg_coeff: float = field(default=1.0)
