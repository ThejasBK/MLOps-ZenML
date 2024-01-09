import logging
import pandas as pd
from typing_extensions import Annotated
from typing import Tuple

from zenml import step

from src.data_cleaning import DataCleanStrategy, DataPreProcessStrategy, DataSplitStrategy

@step
def clean_data(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_test']
]:
    try:
        preprocess_strategy = DataPreProcessStrategy()
        split_strategy = DataSplitStrategy()

        data_cleaning = DataCleanStrategy(data, preprocess_strategy)
        processed_data = data_cleaning.handle_data()

        data_spliting = DataCleanStrategy(processed_data, split_strategy)
        X_train, X_test, y_train, y_test = data_spliting.handle_data()
        logging.info('Data cleaning completed')
        
        # Not performed this step
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error('Error cleaning data: ' + str(e))
        raise e