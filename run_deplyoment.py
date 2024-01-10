from pipelines.deployment_pipeline import deployment_pipeline, inference_pipeline
import click 

@click.command()
@click.option('--config', '-c', type = click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]), default = DEPLOY_AND_PREDICT, 
              help='Path to config file')