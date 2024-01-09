import logging
import pandas as pd

from zenml import step

@step
def evaluate_model(data: pd.DataFrame) -> None:
    pass