from fastapi import APIRouter, Depends
from .iris_model import IrisModel
from .prediction_input import PredictionInput
from .prediction_output import PredictionOutput


prediction_router = APIRouter()

prediction_model = IrisModel()


@prediction_router.post("/predictions")
def prediction(output: PredictionOutput = Depends(prediction_model.predict)) -> PredictionOutput:
    return output



