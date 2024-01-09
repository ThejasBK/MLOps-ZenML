from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluate import evaluate_model

@pipeline(enable_cache = False)
def training_pipeline(data_path: str):
    data = ingest_data(data_path)
    clean_data(data)
    train_model(data)
    evaluate_model(data)