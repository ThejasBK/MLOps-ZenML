import logging
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    # TODO: proper preprocessing
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        logging.info('Preprocessing data')
        try:
            df.drop(['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 
                          'order_estimated_delivery_date', 'order_purchase_timestamp'], axis = 1, inplace = True)
            df['product_weight_g'].fillna(df['product_weight_g'].mean(), inplace = True)
            df['product_height_cm'].fillna(df['product_height_cm'].mean(), inplace = True)
            df['product_length_cm'].fillna(df['product_length_cm'].mean(), inplace = True)
            df['product_width_cm'].fillna(df['product_width_cm'].mean(), inplace = True)
            df['review_comment_message'].fillna('No review', inplace = True)
            df = df.select_dtypes(include=[np.number])
            df.drop(['customer_zip_code_prefix', 'order_item_id'], axis = 1, inplace = True)
            return df
        except Exception as e:
            logging.error('Error preprocessing data: ' + str(e))
            raise e
        
class DataSplitStrategy(DataStrategy):
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        logging.info('Splitting data')
        try:
            X = df.drop(['review_score'], axis = 1)
            y = df['review_score']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error('Error splitting data: ' + str(e))
            raise e
        
class DataCleanStrategy(DataStrategy):
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.strategy = strategy
        self.data = data

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error('Error handling data: ' + str(e))
            raise e