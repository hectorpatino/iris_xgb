import iris_api

from fastapi import FastAPI
from iris_api.datasets.generate_test_train import router as generate_test_train_router
from iris_api.iris_pipeline.iris_pipeline import train_router

app = FastAPI()

app.include_router(generate_test_train_router, prefix="/generate_test_train")
app.include_router(train_router, prefix="/train_pipeline")