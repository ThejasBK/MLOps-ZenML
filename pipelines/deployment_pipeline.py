import pandas as pd
import numpy as np
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.constants import MLFLOW
from zenml.steps import BaseParameters, Output
import json

from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluate import evaluate_model
from steps.ingest_data import ingest_data
from .utils import get_data_for_test

docker_settings = DockerSettings(required_integrations = [MLFLOW])

class DeployTriggerConfig(BaseParameters):
    min_accuracy: float = 0

class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    pipeline_name: str
    step_name: str
    running: bool = True

@step(enable_cache = False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data

@step(enable_cache = False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = 'model',
) -> MLFlowDeploymentService:
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name = pipeline_name,
        pipeline_step_name = pipeline_step_name,
        model_name = model_name,
        running = running,
    )

    if not existing_services:
        raise RuntimeError('No MLFlow deployment service found')    
    
    return existing_services[0]

@step
def predictor(service: MLFlowDeploymentService, data: str) -> np.ndarray:
    service.start(timeout = 15)
    data = json.loads(data)
    # data.pop('columns')
    # data.pop('index')
    columns_for_df = []
    df = pd.DataFrame(data) # 2:51:31
    # json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(df)
    prediction = service.predict(data)
    return prediction


@step
def deploy_trigger(deploy_trigger_config: DeployTriggerConfig, accuracy: float):
    if accuracy >= deploy_trigger_config.min_accuracy:
        return True
    return True

@pipeline(enable_cache = False, 
          settings = {'docker': docker_settings})
def continuous_deployment_pipeline(min_accuracy: float = 0, workers: int = 1, 
                                   timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT, 
                                   data_path: str = r'/Users/thejas/Library/CloudStorage/OneDrive-UCB-O365/MLOps project/zenml-projects/customer-satisfaction/data/olist_customers_dataset.csv'):
    data = ingest_data(data_path)
    X_train, X_test, y_trainm, y_test = clean_data(data)
    model = train_model(X_train, X_test, y_trainm, y_test)
    r2, rmse, mae = evaluate_model(model, X_test, y_test)
    deployment_decision = deploy_trigger(accuracy = r2)
    mlflow_model_deployer_step(
        model = model,
        workers = workers,
        timeout = timeout,
        deploy_decision = deployment_decision,
    )

@pipeline(enable_cache = False, 
          settings = {'docker': docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    data = dynamic_importer()
    service = prediction_service_loader(pipeline_name = pipeline_name, pipeline_step_name = pipeline_step_name, 
                                        running = False)
    prediction = predictor(service = service, data = data)
    return prediction