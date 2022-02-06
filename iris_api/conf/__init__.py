from .model_settings import ModelTrainingSettings
from .app_settings import AppSettings, API_ROOT, ROOT, DATASETS_DIR, TRAINED_MODELS_DIR

model_training_settings = ModelTrainingSettings()
app_settings = AppSettings()

full_pipe_line = f"{model_training_settings.pipeline_save_file}"\
                 f"{model_training_settings.version}" \
                 f"{model_training_settings.format}"
