from .data_source_types import DataSourceTypes
from module.datasource.local_dir import LocalDataSource

__all__ = ['DataSource']

class DataSource(object):

    def __init__(self, config):
        if config['type'] == DataSourceTypes.local_dir:
            self.impl = LocalDataSource(config)

    def data_change(self):
        return self.impl.data_change()

    def get_last_data_ident(self):
        return self.impl.get_last_data_ident()

    def update_data_files(self):
        return self.impl.update_data_files()

    def data_dir(self):
        return self.impl.dir

    def load_data(self):
        return self.impl.load_data()
