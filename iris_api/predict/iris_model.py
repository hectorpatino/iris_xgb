from typing import Optional

import joblib
import os
import pandas as pd

from sklearn.pipeline import Pipeline

from iris_api.conf import TRAINED_MODELS_DIR, full_pipe_line
from iris_api.predict.prediction_input import PredictionInput, MultiplePredictionInputs
from iris_api.predict.prediction_output import PredictionOutput


class IrisModel:
    model: Optional[Pipeline]

    def load_model(self):
        pipeline_location = TRAINED_MODELS_DIR / full_pipe_line
        self.model = joblib.load(str(pipeline_location))

    async def predict(self,
                      input_data: MultiplePredictionInputs
                      ) -> PredictionOutput:
        if not self.model:
            raise RuntimeError("Model not loaded")
        data = pd.DataFrame.from_dict(input_data.dict()['inputs'])
        predictions = self.model.predict(data)
        print("##############################################################")
        print(predictions)
        print(type(predictions))
        print("##############################################################")
        return PredictionOutput(predictions=predictions.tolist())
