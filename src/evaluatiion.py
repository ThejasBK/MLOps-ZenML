import logging
from abc import ABC, abstractmethod
import numpy as np

class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_preds: np.ndarray):
        pass

class RMSE(Evaluator):
    def evaluate(self, y_true: np.ndarray, y_preds: np.ndarray):
        try:
            rmse = np.sqrt(np.mean((y_true - y_preds) ** 2))
            logging.info('RMSE: ' + str(rmse))
            return rmse
        except Exception as e:
            logging.error('Error evaluating model: ' + str(e))
            raise e
        
class MAE(Evaluator):
    def evaluate(self, y_true: np.ndarray, y_preds: np.ndarray):
        try:
            mae = np.mean(np.abs(y_true - y_preds))
            logging.info('MAE: ' + str(mae))
            return mae
        except Exception as e:
            logging.error('Error evaluating model: ' + str(e))
            raise e
        
class R2(Evaluator):
    def evaluate(self, y_true: np.ndarray, y_preds: np.ndarray):
        try:
            r2 = 1 - np.sum((y_true - y_preds) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
            logging.info('R2: ' + str(r2))
            return r2[0]
        except Exception as e:
            logging.error('Error evaluating model: ' + str(e))
            raise e