import logging
import pandas as pd

from zenml import step

class IngestData:
    def __init__(self, path: str):
        self.path = path

    def get_data(self):
        logging.info(f'Ingesting data from {self.path}')
        return pd.read_csv(self.path)
    
@step
def ingest_data(path: str) -> pd.DataFrame:
    try:
        data = IngestData(path).get_data()
        return data
    except Exception as e:
        logging.error(f'Error ingesting data: {e}')
        raise e