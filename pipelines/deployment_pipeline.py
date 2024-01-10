import pandas as pd
import numpy as np
from zenml import pipeline, step
from zenml.config import DockerSettings
from materializer.custom_materializer import cs_materializer
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.constants import MLFLOW
from zenml.steps import BaseParameters, Output

from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluate import evaluate_model
from steps.ingest_data import ingest_data

docker_settings = DockerSettings(required_integrations = [MLFLOW])

@pipeline(enable_cache = True, 
          settings = {'docker_settings': docker_settings})
def continuous_deployment_pipeline(min_accuracy: float = 0.92, workers: int = 1, 
                                   timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT):
    data = ingest_data()
    X_train, X_test, y_trainm, y_test = clean_data(data)
    model = train_model(X_train, X_test, y_trainm, y_test)
    r2, rmse, mae = evaluate_model(model, X_test, y_test)


