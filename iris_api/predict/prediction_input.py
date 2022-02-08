from pydantic import BaseModel
from typing import List


class PredictionInput(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float


class MultiplePredictionInputs(BaseModel):
    inputs: List[PredictionInput]

    class Config:
        arbytrary_types_allowed = True
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "SepalLengthCm": 5.1,
                        "SepalWidthCm": 5.2,
                        "PetalLengthCm": 5.3,
                        "PetalWidthCm": 5.4
                    }
                ]
            }
        }
