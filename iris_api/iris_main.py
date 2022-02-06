import iris_api

from fastapi import FastAPI
from iris_api.datasets.generate_test_train import router as generate_test_train_router
from iris_api.iris_pipeline.iris_pipeline import train_router, train_model
from iris_api.predict.predict import prediction_router, prediction_model

app = FastAPI()

app.include_router(generate_test_train_router, prefix="/generate_test_train")
app.include_router(train_router, prefix="/train_pipeline")
app.include_router(prediction_router, prefix="/predict")


@app.on_event("startup")
def startup():
    train_model()
    prediction_model.load_model()
