import gc
import os
import sys

from nni import Experiment
from nni.experiment import ExperimentConfig
import json
from nni.experiment.config.algorithm import CustomAlgorithmConfig
from module.util.result_saver import ResultSaver
from ..abs_autofe import ABSAutoFE
from ..fe_util import name2feature

from module.common.config.task_config import TaskConfig



class DFS(ABSAutoFE):

    def __init__(self, config):
        super().__init__()
        self.experiment = None
        self.config = config
        self.seq = 0
        self.combination_features = []
        self.target_name = TaskConfig().cfg['Feature']['TargetName']
        if 'id_index' in TaskConfig().cfg['Feature'].keys():
            self.id_index = TaskConfig().cfg['Feature']['id_index']
        else:
            self.id_index = None
        self.all_features = TaskConfig().cfg['Feature']['FeatureName']
        self.cat_features = TaskConfig().cfg['Feature']['CategoricalFeature']
        self.metric = TaskConfig().cfg['Feature']['Metric']

    def start(self, data_dir):
        self.seq += 1
        self.experiment = Experiment(TaskConfig().cfg['Resource']['trainingServicePlatform'])
        self.experiment.config = ExperimentConfig(TaskConfig().cfg['Resource']['trainingServicePlatform'])
        self.experiment.config.search_space = self.generate_search_space()
        self.experiment.config.tuner = CustomAlgorithmConfig()
        self.experiment.config.tuner.class_name = "autofe_tuner.AutoFETuner"
        self.experiment.config.tuner.code_directory = "module/autofe/dfs/"
        # self.experiment.config.tuner.code_directory = "."
        self.experiment.config.tuner.class_args = {"optimize_mode": "maximize"}

        # self.experiment.config.max_trial_number = 2
        self.experiment.config.max_experiment_duration = '10h'
        self.experiment.config.trial_concurrency = 1
        self.experiment.config.max_trial_number = int(TaskConfig().cfg['AutoFE']['maxTrialNum'])
        self.port = int(TaskConfig().cfg['AutoFE']['port'])
        # self.experiment.config.max_experiment_duration = f"{TaskConfig().cfg['Resource']['maxExecDuration']}"
        # self.experiment.config.trial_concurrency = int(TaskConfig().cfg['Resource']['trialConcurrency'])

        self.experiment.config.experiment_name = f"{TaskConfig().cfg['TaskName']}_{self.seq}"
        self.experiment.config.trial_code_directory = "module/autofe/dfs"
        self.experiment.config.experiment_working_directory = ResultSaver.generate_res_path(self.stage_name, str(self.seq), 'logs')
        if self.id_index is None:
            self.experiment.config.trial_command = f"{sys.executable} main.py {os.getcwd()} {TaskConfig().cfg_path} {self.target_name}"
        else:
            self.experiment.config.trial_command = f"{sys.executable} main.py {os.getcwd()} {TaskConfig().cfg_path} {self.target_name} {self.id_index}"

        self.experiment.run(self.port)

    def save_results(self):
        xxx = self.experiment.export_data()
        ori_precision = xxx[0].value["default"]
        print(f"original precision:{ori_precision}")
        ori_feature_names = set(xxx[0].value["feature_importance"]["feature_name"])
        final_features = ori_feature_names
        for i in range(1, len(xxx)):
            if self.better_than(xxx[i].value["default"], ori_precision):
                effect_feature_names = set(xxx[i].value["feature_importance"]["feature_name"][0:len(ori_feature_names)])
                final_features = final_features | effect_feature_names

        self.combination_features = list(final_features - ori_feature_names)

        print(f'autofe experiment {self.seq} is over , get advanced features: {self.combination_features}')
        autofe_res_dict = {
            'id': self.seq,
            'combination_features': self.combination_features
        }
        result_json_str = json.dumps(autofe_res_dict, sort_keys=False, indent=4, separators=(',', ': '))

        ResultSaver.save(self.stage_name, self.seq, result_json_str)
        self.experiment.stop()

    def load_results(self):
        res_file = 'simpleTask/autofe/1'
        self.combination_features = json.load(open(res_file))['combination_features']


    def feature_engineering(self, ori_data):
        return name2feature(ori_data, self.combination_features)


    def generate_search_space(self):
        num_features = set(self.all_features) - set(self.cat_features)
        res_dict = {}
        if len(self.cat_features) > 0:
            res_dict['count'] = self.cat_features
            res_dict['crosscount'] = [self.cat_features, self.cat_features]
            res_dict['nunique'] = [self.cat_features, self.cat_features]
        if len(num_features) > 0:
            res_dict['aggregate'] = [list(num_features), self.cat_features]
        return res_dict


    def better_than(self, exp, ori):
        if self.metric in ['auc']:
            return exp >= ori
        else:
            return exp <= ori



