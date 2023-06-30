from schedule import every, repeat, run_pending
import os
import sys
from module.task.Task import Task
from module.datasource import c_dataset

os.environ['PYTHONPATH'] = os.path.dirname(__file__)

experimenting = False
current_data_file = []
task = None
import warnings
warnings.filterwarnings('ignore')
from module.common.config.task_config import TaskConfig

@repeat(every(5).seconds)
def scan_file_change():
    global experimenting
    global current_data_file
    global task
    if experimenting:
        print("incremental learning is running, new data change will be added in next experiment")
        return
    if task.data_source.data_change():
        print("File change detected, start incremental learning experiment")
        task.data_source.update_data_files()
        experimenting = True
        task.continue_incrml()
        experimenting = False
        
if __name__ == '__main__':
    # config_file = 'benchmark/criteo/config.yaml'

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        label_name = sys.argv[2]
        task_name = data_path.split('/')[-1].split('.')[0]
    config_file = c_dataset(path=data_path, task_name=task_name, y_name=label_name)

    task_config = TaskConfig(config_file)
    task = Task(task_config.cfg)
    print("increamental learning started")
    while True:
        run_pending()