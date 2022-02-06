import xgboost as xgb
import pandas as pd
import joblib

from fastapi import APIRouter
from feature_engine.creation import MathematicalCombination
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from iris_api.conf import model_training_settings, app_settings, DATASETS_DIR, TRAINED_MODELS_DIR
from pathlib import Path


train_router = APIRouter()


def load_dataset(file_name: str) -> pd.DataFrame:
    """Load dataset from `file_name`. The function handles the directory

    Example:
        >>> load_dataset("iris.csv")

    Args:
        file_name (str): The name of the file to load. It must exist on the config.DATASET_DIR directory.

    Returns:
        pd.DataFrame: The loaded dataset
    """
    dataframe = pd.read_csv(Path(f"{DATASETS_DIR}/{file_name}"))
    dataframe.rename(columns={'sepal length (cm)': model_training_settings.features_to_combine[0],
                              'sepal width (cm)': model_training_settings.features_to_combine[1],
                              'petal length (cm)': model_training_settings.features_to_combine[2],
                              'petal width (cm)': model_training_settings.features_to_combine[3]},
                     inplace=True)
    return dataframe


@train_router.get("/train")
async def train_model():
    data = load_dataset(app_settings.training_data_file)
    iris_pipeline = Pipeline(
        [
            (
                model_training_settings.full_combination_step,
                MathematicalCombination(
                    variables_to_combine=model_training_settings.features_to_combine
                ),
            ),
            (
                model_training_settings.xgb_step,
                xgb.XGBClassifier(
                    n_estimators=model_training_settings.n_estimators,
                    learning_rate=model_training_settings.learning_rate,
                    max_depth=model_training_settings.max_depth,
                    gamma=model_training_settings.gamma,
                    subsample=model_training_settings.subsample,
                    objective=model_training_settings.objective,
                    eval_metric=accuracy_score,
                ),
            ),
        ]
    )
    iris_pipeline.fit(data.drop(model_training_settings.target, axis=1), data[model_training_settings.target])
    version = 1
    full_version = f"{model_training_settings.pipeline_save_file}{version}"
    pipeline_location = TRAINED_MODELS_DIR / full_version
    joblib.dump(pipeline_location, pipeline_location)
    return {"message": f"Model trained and saved to {pipeline_location}"}


