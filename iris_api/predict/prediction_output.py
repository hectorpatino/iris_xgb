from pydantic import BaseModel


class PredictionOutput(BaseModel):
    specie: float
