
from .autofe_types import AutoFETypes
from .dfs import DFS
from .autofeat.autofeat_tuner import autofeatTuner

class AutofeFactory(object):

    def __init__(self, config):

        if config["Method"] == AutoFETypes.dfs:
            self.impl = DFS(config)
        elif config["Method"] == AutoFETypes.autofeat:
            self.impl = autofeatTuner(config)


    def start(self, data_dir):
        self.impl.start(data_dir)

    def save_results(self):
        self.impl.save_results()

    def load_results(self):
        self.impl.load_results()

    def feature_engineering(self, ori_data):
        return self.impl.feature_engineering(ori_data)