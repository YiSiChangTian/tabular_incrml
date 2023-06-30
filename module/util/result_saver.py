import os.path
import pickle

from module.common.config.task_config import TaskConfig

from module.common.config.get_global_config import global_config

class ResultSaver(object):

    @staticmethod
    def save(stage_name, exp_seq, content: str):
        if global_config['SaveResult']["type"] == 'local_dir':
            file_name = ResultSaver.generate_res_path(stage_name, exp_seq, 'result')
            ResultSaver.to_local_json(file_name, content)

    @staticmethod
    def generate_res_path(stage_name, exp_seq, process):
        file_name = os.path.join(os.getcwd(), global_config['SaveResult']["prefix"], TaskConfig().cfg["TaskName"], str(exp_seq),
                                 stage_name, process)
        return file_name

    @staticmethod
    def mkdir(file_name):
        dir_name = os.path.dirname(file_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    @staticmethod
    def to_local_json(file_name, content:str):
        dir_name = os.path.dirname(file_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(file_name, 'w') as f:
            f.write(content)
            f.close()

    @staticmethod
    def save_model(stage_name, exp_seq, model):
        file_name = ResultSaver.generate_res_path(stage_name, exp_seq, "result")
        dir_name = os.path.dirname(file_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(file_name, "wb") as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
            f.close()

