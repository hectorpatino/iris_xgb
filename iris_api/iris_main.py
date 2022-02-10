import iris_api

from fastapi import FastAPI
from iris_api.datasets.generate_test_train import router as generate_test_train_router
from iris_api.health.health_router import health_router
from iris_api.iris_pipeline.iris_pipeline import train_router, train_model
from iris_api.predict.predict import prediction_router, prediction_model

description = """
Iris Prediction using Xgboost.

Developed by: [Hector Patiño](https://github.com/hectorpatino).

# What you can do?
* Train the model.
* Predict different inputs.
* Generate test and train data from sklearn.
* Initial training is generated at ``@app.on_event('startup')``.

# TODOs
* Read a CSV file -> make predictions -> download the csv file with predictions.
"""
app = FastAPI(docs_url="/", description=description)

app.include_router(generate_test_train_router, prefix="/generate_test_train")
app.include_router(train_router, prefix="/train_pipeline")
app.include_router(prediction_router, prefix="/predict")
app.include_router(health_router, prefix="/health")


@app.on_event("startup")
def startup():
    train_model()
    prediction_model.load_model()
