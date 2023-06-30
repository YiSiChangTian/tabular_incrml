import os
import yaml
import pandas as pd

def default_yaml():
    return {
        'TaskName': 'cancer',
        'AuthorName': 'futureGuo',
        'DataSource':
            {'type': 'local_dir',
            'dir': 'benchmark/cancer/data'},
        'Feature':
            {'TaskType': 'classification',
            'Metric': 'roc_auc',
            'TargetName': 'Class',
            'FeatureName': ["age","menopause","tumorsize","invnodes","nodecaps","degmalig","breast","breastquad","irradiat"],
            'CategoricalFeature': ["age","menopause","tumorsize","invnodes","nodecaps","degmalig","breast","breastquad","irradiat"]},
        'AutoFE':
            {'Method': 'DFS',
            'maxTrialNum': 5,
            'port': 8088,
            'SaveFeatures': 'OnlyGreater',
            },
        'AutoML':
            {'Method': 'flaml',
            'TimeBudget': 5},
        'IncrML':
            {'Method': 'iCaRL',
            'Trigger': 'OnDataFileIncrease'},
        'Resource':
            {'trainingServicePlatform': 'local'}
    }


def c_dataset(path, task_name, y_name, id_name=None, is_classification=True):

    df = pd.read_csv(path)
    new_columns = [col.strip().replace(" ", "_") for col in df.columns]
    df.columns = new_columns

    if any(df[y_name].isnull()):
        df = df.dropna(subset=[y_name])
    
    config = default_yaml()

    task_path = f'benchmark/{task_name}'
    yaml_path = f'benchmark/{task_name}/config.yaml'
    data_path = f'benchmark/{task_name}/data'

    if not (os.path.exists(task_path) and os.path.exists(yaml_path) and os.path.exists(data_path)):
        os.makedirs(f'{task_path}/data', mode=0o777, exist_ok=True)
    
        df.to_csv(f"{task_path}/data/{task_name}.csv", index=False)
        
        config['TaskName'] = task_name
        config['DataSource']['dir'] = "benchmark/" + task_name + "/data"

        unique_values_count = df[y_name].nunique()
        if unique_values_count >= len(df) // 10 or unique_values_count > 1000:
            config['Feature']['TaskType'] = "regression"
            config['Feature']['Metric'] = "rmse"
        config['Feature']['TargetName'] = y_name
        if id_name:
            config['Feature']['id_index'] = id_name
            feature_name = df.columns.to_series().drop([y_name, id_name]).index
            df_feature = df.drop(columns=[y_name, id_name])
        else:
            feature_name = df.columns.to_series().drop([y_name]).index
            df_feature = df.drop(columns=[y_name])
        config['Feature']['FeatureName'] = feature_name.to_list()
        config['Feature']['CategoricalFeature'] = get_categorical_features(df_feature)
        with open(f"{task_path}/config.yaml", 'w') as file:
            file.write(yaml.dump(config, allow_unicode=True))
        
    return f"{task_path}/config.yaml"


def get_categorical_features(dataframe):
    """
    获取数据帧中的所有类别特征
    """
    # 获取数据帧的所有列
    columns = dataframe.columns

    # 用于存储所有类别特征的列表
    categorical_features = []

    # 遍历数据帧的每一列
    for column in columns:
        # 如果数据类型是 object 或 category，则将其添加到类别特征列表中
        if dataframe[column].dtype == 'object' or dataframe[column].dtype.name == 'category':
            categorical_features.append(column)
        elif len(dataframe[column].unique()) <= (len(dataframe) // 2):
            categorical_features.append(column)

    return categorical_features


# if __name__ == "__main__":
#     c_dataset(path="cancer/data/breast-cancer.data", # csv数据路径
#               task_name="cancer",  # 任务的名称
#               y_name="Class", # 因变量的名称
#               id_name=None
#               ) # 训练任务是否为分类，如果是为True，反之False