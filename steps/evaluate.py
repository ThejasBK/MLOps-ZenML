import logging
import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
import mlflow

from zenml.client import Client

from zenml import step
from src.evaluatiion import RMSE, MAE, R2

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker = experiment_tracker.name)
def evaluate_model(model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[
    Annotated[float, 'r2'], 
    Annotated[float, 'rmse'], 
    Annotated[float, 'mae']]:
    try:
        prediction = model.predict(X_test)
        rmse = RMSE().evaluate(y_test, prediction)
        mae = MAE().evaluate(y_test, prediction)
        r2 = R2().evaluate(y_test, prediction)

        mlflow.log_metric('r2', r2)
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mae', mae)
        logging.info('Model evaluation complete')
        return r2, rmse, mae
    except Exception as e:
        logging.error('Error evaluating model: ' + str(e))
        raise e