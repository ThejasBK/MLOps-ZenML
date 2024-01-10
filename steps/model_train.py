import logging
import pandas as pd
import mlflow

from zenml import step
from zenml.client import Client
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin 
from .config import ModelNameConfig

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker = experiment_tracker.name)
def train_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, 
                config: ModelNameConfig) -> RegressorMixin:
    try:
        if config.model_name == 'LinearRegression':
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            train_model = model.train(X_train, y_train)
            return train_model
        else:
            raise NotImplementedError('Model not implemented')
    except Exception as e:
        logging.error('Error training model: ' + str(e))
        raise e