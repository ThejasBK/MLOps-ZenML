from pipelines.deployment_pipeline import inference_pipeline, continuous_deployment_pipeline
import click 
from rich import print
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from typing import cast

DEPLOY = 'deploy'
PREDICT = 'predict'
DEPLOY_AND_PREDICT = 'deploy_and_predict'

@click.command()
@click.option('--config', '-c', type = click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]), default = DEPLOY_AND_PREDICT, 
              help='What do you wanna do?')
@click.option('--min-accuracy', default = 0, help = 'Minimum accuracy for model to be deployed')

def run_deployment(config: str, min_accuracy: float):
    mlflow_model_deployer_componenet = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT
    if deploy:
        continuous_deployment_pipeline(min_accuracy = min_accuracy, workers = 3, timeout = 60)
    if predict:
        inference_pipeline(
            pipeline_name = "continuous_deployment_pipeline",
            pipeline_step_name = "mlflow_model_deployer_step",
        )

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri {get_tracking_uri()}\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )

    existing_services = mlflow_model_deployer_componenet.find_model_server(
        pipeline_name = 'continuous_deployment_pipeline',
        pipeline_step_name = 'mlflow_model_deployer_step',
        model_name = 'model',
    )

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print('Service is running')
            print(f'Process service and accepts inference requests at:{service.prediction_url}')
            print(f'To stop the service, run: `zenml model-deployer models delete {str(service.uuid)}')
        elif service.is_failed:
            print(f'Service has failed \n last state: {service.status.state.value} \n Last error: {service.status.last_error}')
    else:
        print('No service found')

if __name__ == '__main__':
    run_deployment()