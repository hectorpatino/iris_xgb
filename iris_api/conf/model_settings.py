from pathlib import Path
from typing import List
from pydantic import BaseSettings


class ModelTrainingSettings(BaseSettings):
    target = "Species"
    test_size = 0.2
    random_state = 42
    pipeline_name = "pipeline_iris"
    pipeline_save_file: str = "iris_pipeline_v_"
    format: str = ".pkl"
    features_to_combine: List[str] = [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm",
    ]
    shuffle: bool = True
    full_combination_step: str = "full_combination"
    xgb_step: str = "xgb_step"

    n_estimators: int = 16
    learning_rate: float = 5.869262439392996
    max_depth: int = 4
    gamma: float = 1.3381078668366866
    subsample: float = 0.38674136078825716
    objective: str = "multi:softprob"
