from pathlib import Path

from pydantic import BaseSettings
import iris_api


API_ROOT = Path(iris_api.__file__).resolve().parent
ROOT = API_ROOT.parent
DATASETS_DIR = API_ROOT / 'datasets'
TRAINED_MODELS_DIR = API_ROOT / 'trained_models'


class AppSettings(BaseSettings):
    training_data_file: str = "train.csv"
    test_data_path: str = "test.csv"
    api_version = "v1"
    name = "iris_xgboost_api"
