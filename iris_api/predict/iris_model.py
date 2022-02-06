from typing import Optional

import joblib
import os
import pandas as pd

from sklearn.pipeline import Pipeline

from iris_api.conf import model_training_settings, TRAINED_MODELS_DIR, full_pipe_line
from iris_api.predict.prediction_input import PredictionInput
from iris_api.predict.prediction_output import PredictionOutput


class IrisModel:
    model: Optional[Pipeline]

    def load_model(self):
        pipeline_location = TRAINED_MODELS_DIR / full_pipe_line
        self.model = joblib.load(str(pipeline_location))

    async def predict(self, input_data: PredictionInput) -> PredictionOutput:
        if not self.model:
            raise RuntimeError("Model not loaded")
        data = pd.DataFrame.from_records([input_data.dict()])
        prediction = self.model.predict(data)
        print(prediction[0])
        return PredictionOutput(specie=prediction[0])
