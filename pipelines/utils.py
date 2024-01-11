import logging
import pandas as pd
from src.data_cleaning import DataCleanStrategy, DataPreProcessStrategy

def get_data_for_test():
    try:
        data = pd.read_csv(r'/Users/thejas/Library/CloudStorage/OneDrive-UCB-O365/MLOps project/zenml-projects/customer-satisfaction/data/olist_customers_dataset.csv')
        data = data.sample(n = 100)
        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleanStrategy(data, preprocess_strategy)
        processed_data = data_cleaning.handle_data()
        logging.info('Data cleaning completed')
        processed_data.drop(['review_score'], axis = 1, inplace = True)
        return processed_data.to_json()    # orient = 'split'
    except Exception as e:
        logging.error('Error cleaning data: ' + str(e))
        raise e