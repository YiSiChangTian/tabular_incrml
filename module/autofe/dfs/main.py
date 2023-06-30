# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import os

import nni
from module.autofe.fe_util import *
from model import *
import sys
from module.common.config.task_config import TaskConfig
from module.datasource import DataSource

logger = logging.getLogger('auto-fe-examples')

if __name__ == '__main__':

    pwd = sys.argv[1]

    os.chdir(pwd)

    config_path = sys.argv[2]

    target_name = sys.argv[3]

    id_index = None
    if len(sys.argv) > 4:
        id_index = sys.argv[4]

    # get parameters from tuner
    RECEIVED_PARAMS = nni.get_next_parameter()
    logger.info("Received params:\n", RECEIVED_PARAMS)

    ds = DataSource(TaskConfig(config_path).cfg['DataSource'])
    # list is a column_name generate from tuner
    df = ds.load_data()
    print(df.head())
    if 'sample_feature' in RECEIVED_PARAMS.keys():
        sample_col = RECEIVED_PARAMS['sample_feature']
    else:
        sample_col = []
    
    # raw feaure + sample_feature
    df = name2feature(df, sample_col)
    
    feature_imp, val_score = lgb_model_train(df,  _epoch = 1000, target_name = target_name, id_index = id_index)
    nni.report_final_result({
        "default":val_score, 
        "feature_importance":feature_imp
    })
