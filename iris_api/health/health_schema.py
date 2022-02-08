from typing import List
from pydantic import BaseModel
from iris_api.conf import app_settings, model_training_settings


class HealthSchema(BaseModel):
    api_version: str = app_settings.api_version
    model_version: str = model_training_settings.version
    name: str = app_settings.name
    notes: List[str]
