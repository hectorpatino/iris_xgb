from typing import List
from pydantic import BaseModel
from .prediction_input import MultiplePredictionInputs
from iris_api.conf import app_settings


class PredictionOutput(BaseModel):
    predictions: List[float]
    version: str = app_settings.api_version
