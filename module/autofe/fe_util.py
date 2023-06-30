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

import pandas as pd
import numpy as np 
from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD
import logging
from .feature_predefines import FeatureType, AGGREGATE_TYPE
from ..util.const import Const

logger = logging.getLogger('autofe-tuner')

def left_merge(data1, data2, on):
    """
    merge util for dataframe
    """
    if type(on) != list:
        on = [on]
    if (set(on) & set(data2.columns)) != set(on):
        data2_temp = data2.reset_index()
    else:
        data2_temp = data2.copy()
    columns = [f for f in data2.columns if f not in on]
    result = data1.merge(data2_temp, on = on, how='left')
    result = result[columns]
    return result


def concat(L):
    """
    tools for concat some dataframes into a new dataframe.
    """
    result = None
    for l in L:
        if l is None:
            continue
        if result is None:
            result = l
        else:
            result[l.columns.tolist()] = l
    return result


def name2feature(df, feature_space):
    assert isinstance(feature_space, list)

    for key in feature_space:
        temp = key.split(Const.feature_separator)
        assert len(temp) > 1

        op_name = temp[0]
        if op_name in [FeatureType.COUNT]:
            i = temp[1]
            command = op_name + '(df, i)'
        elif op_name in [FeatureType.CROSSCOUNT]:
            i, j = temp[1], temp[2]
            command = op_name + '(df, [i, j])'
        elif op_name in [FeatureType.NUNIQUE, FeatureType.HISTSTAT]:
            i, j = temp[1], temp[2]
            command = op_name + '(df, i, j)'
        elif op_name in [FeatureType.AGGREGATE]:
            stat, i, j = temp[1], temp[2], temp[3]
            command = op_name + '(df, i, j, [stat])'
        else:
            raise RuntimeError('Do not support this OP: ' + str(key))

        df = eval(command)
    
    return df


def count(df, col):
    """
    tools for count encode
    """
    df[f'count{Const.feature_separator}{col}'] = df.groupby(col)[col].transform('count')
    return df


def crosscount(df, col_list):
    """
    tools for multy thread bi_count
    """
    assert isinstance(col_list, list)
    assert len(col_list) >= 2
    name = "crosscount" + Const.feature_separator + Const.feature_separator.join(col_list)
    df[name] = df.groupby(col_list)[col_list[0]].transform('count')
    return df


def aggregate(df, num_col, group_col, stat_list = AGGREGATE_TYPE):
    agg_dict = {}
    for i in stat_list:
        dict_value = Const.feature_separator.join([FeatureType.AGGREGATE, i, num_col, group_col])
        agg_dict[i] = dict_value
    agg_result = df.groupby([group_col])[num_col].agg(list(agg_dict.keys())).rename(columns = agg_dict)
    r = left_merge(df, agg_result, on = [group_col])
    df = concat([df, r])
    return df


def nunique(df, id_col, group_col):
    """
    get id group_by(id) nunique
    """
    nunique_col_name = Const.feature_separator.join([FeatureType.NUNIQUE, id_col, group_col])
    
    agg_result = df.groupby([group_col])[id_col].nunique()
    agg_result_df = pd.DataFrame(agg_result).rename(columns={id_col: nunique_col_name}).reset_index()
    r = left_merge(df, agg_result_df, on = [group_col])
    df = concat([df, r])
    return df


def histstat(df, id_col, group_col, stat_list = AGGREGATE_TYPE):
    """
    get id group_by(id) histgram statitics
    """
    agg_dict = {}
    for i in stat_list:
        dict_value = Const.feature_separator.join([FeatureType.HISTSTAT, i, id_col, group_col])
        agg_dict[i] = dict_value
    df['temp_count'] = df.groupby(id_col)[id_col].transform('count')
    agg_result = df.groupby([group_col])['temp_count'].agg(list(agg_dict.keys())).rename(columns = agg_dict)
    r = left_merge(df, agg_result, on = [group_col])
    df = concat([df, r])
    del df['temp_count']
    return df
