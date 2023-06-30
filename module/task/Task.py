from ..autofe.autofe_factory import AutofeFactory
from ..datasource import DataSource

__all__ = ['Task']

class Task:

    def __init__(self, config):
        self.config: dict = config
        self.data_source: DataSource = DataSource(config['DataSource'])
        self.autofe: AutofeFactory = AutofeFactory(config['AutoFE'])

    def continue_incrml(self):
        self.autofe.start(self.data_source.data_dir())
        self.autofe.save_results()
