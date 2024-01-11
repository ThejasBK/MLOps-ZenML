# from zenml.client import Client

# run = Client().active_stack.experiment_tracker.get_tracking_uri()

# print(run)

# # mlflow ui --backend-store-uri "file:/Users/thejas/Library/Application Support/zenml/local_stores/8f5375d4-e381-453a-840b-330d73200410/mlruns"

from zenml.repository import Repository

repo = Repository()
pipeline = repo.get_pipeline(pipeline_name="continuous_deployment_pipeline")
# run = pipeline.get_run("custom_pipeline_run_name")
print(pipeline)