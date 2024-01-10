from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluate import evaluate_model

@pipeline(enable_cache = False)
def training_pipeline(data_path: str):
    data = ingest_data(data_path)
    X_train, X_test, y_trainm, y_test = clean_data(data)
    model = train_model(X_train, X_test, y_trainm, y_test)
    r2, rmse, mae = evaluate_model(model, X_test, y_test)