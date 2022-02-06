"""
Generate train and test datasets from kaggle.
"""
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from iris_api.conf import model_training_settings, DATASETS_DIR, app_settings
from fastapi import APIRouter


router = APIRouter()


@router.get("/generate")
async def generate_test_train():
    """
    Generate train and test datasets from kaggle.
    """
    # Load data from kaggle
    iris = datasets.load_iris()
    data1 = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                         columns=iris['feature_names'] + ['target'])

    X_train, X_test, y_train, y_test = train_test_split(data1.drop('target', axis=1).copy(),
                                                        data1['target'].copy(),
                                                        test_size=model_training_settings.test_size,
                                                        random_state=model_training_settings.random_state,
                                                        stratify=data1['target'].copy(),
                                                        shuffle=model_training_settings.shuffle)

    X_train[model_training_settings.target] = y_train
    X_train.to_csv(f'{DATASETS_DIR}/{app_settings.training_data_file}', index=False)
    X_test.to_csv(f'{DATASETS_DIR}/{app_settings.test_data_path}', index=False)

    return {
        'message': 'Generated train and test datasets from kaggle.'
    }
