import glob
import os

import pandas as pd
from .local_data_source_options import LocalDataSourceOptions

__all__ = ['LocalDataSource']

import logging

logger = logging.getLogger('data_source')

class LocalDataSource(object):

    def __init__(self, config):
        self.dir = os.path.join(os.getcwd(), config[LocalDataSourceOptions.dir])
        self.current_data_file = []

    def data_change(self):
        now_data_file = glob.glob(os.path.join(self.dir, '*'))
        return set(self.current_data_file) != set(now_data_file)

    def get_last_data_ident(self):
        return self.current_data_file

    def update_data_files(self):
        self.current_data_file = glob.glob(os.path.join(self.dir, '*'))

    def load_data(self):
        df = pd.DataFrame()
        for file in glob.glob(os.path.join(self.dir, '*')):
            df = pd.concat([df, pd.read_csv(f'{file}', encoding="utf-8")])
        return df
