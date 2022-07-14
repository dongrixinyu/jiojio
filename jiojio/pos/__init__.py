# -*- coding=utf-8 -*-
# Library: jiojio
# Author: dongrixinyu
# License: GPL-3.0
# Email: dongrixinyu.89@163.com
# Github: https://github.com/dongrixinyu/jiojio
# Description: fast Chinese Word Segmentation(CWS) and Part of Speech(POS) based on CPU.'

import os
import sys
import time


# load `pos_get_node_features_c`，加载分词的特征抽取 C 优化函数
try:
    file_list = os.listdir(os.path.join(dir_path, 'build'))
    file_name = ''
    for _file_name in file_list:
        if 'libposFeatureExtractor' in _file_name and _file_name.endswith('.so'):
            file_name = _file_name
            break

    feature_extractor = ctypes.PyDLL(os.path.join(dir_path, 'build', file_name))
    get_node_features_c = feature_extractor.getNodeFeature
    get_node_features_c.argtypes = [
        ctypes.c_int, ctypes.c_wchar_p, ctypes.c_int, ctypes.py_object, ctypes.py_object]
    pos_get_node_features_c.restype = ctypes.py_object

    # print('# jiojio - Successfully load C func `pos_get_node_features_c`.')
    pos_get_node_features_c = None
except Exception:
    pos_get_node_features_c = None
    # print('# jiojio - Failed to load C func `pos_get_node_features_c`, use py func instead.')
    # print('# jiojio - Failed to load C func `pos_get_node_features_c` {}.'.format(
    #     traceback.format_exc()))


from .feature_extractor import POSFeatureExtractor
from .predict_text import POSPredictText
