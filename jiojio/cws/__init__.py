# -*- coding=utf-8 -*-
# Library: jiojio
# Author: dongrixinyu
# License: GPL-3.0
# Email: dongrixinyu.89@163.com
# Github: https://github.com/dongrixinyu/jiojio
# Description: fast Chinese Word Segmentation(CWS) and Part of Speech(POS) based on CPU.'


import os
import pdb
import ctypes
import traceback


dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'jiojio_cpp')

# load `cws_get_node_features_c`，加载分词的特征抽取 C 优化函数
try:
    file_list = os.listdir(os.path.join(dir_path, 'build'))
    file_name = ''
    for _file_name in file_list:
        if 'libcwsFeatureExtractor' in _file_name and _file_name.endswith('.so'):
            file_name = _file_name
            break

    feature_extractor = ctypes.PyDLL(os.path.join(dir_path, 'build', file_name))
    cws_get_node_features_c = feature_extractor.getCwsNodeFeature
    cws_get_node_features_c.argtypes = [
        ctypes.c_int, ctypes.c_wchar_p, ctypes.c_int, ctypes.py_object, ctypes.py_object]
    cws_get_node_features_c.restype = ctypes.py_object

    # print('# Successfully load C func `cws_get_node_features_c`.')

except Exception:
    cws_get_node_features_c = None
    # print('# Failed to load C func `cws_get_node_features_c`, use py func instead.')


# load `tag_words_converter_c`，加载分词的标签词汇转换 C 优化函数
try:
    file_list = os.listdir(os.path.join(dir_path, 'build'))
    file_name = ''
    for _file_name in file_list:
        if 'libtagWordsConverter' in _file_name and _file_name.endswith('.so'):
            file_name = _file_name
            break

    tag_words_converter = ctypes.PyDLL(os.path.join(dir_path, 'build', file_name))
    cws_tag2word_c = tag_words_converter.tagWordsConverter
    # tag2word_c.argtypes = [ctypes.c_wchar_p, ctypes.py_object]
    cws_tag2word_c.restype = ctypes.py_object

    # print('# Successfully load C func `cws_tag2word_c`.')

except Exception:
    cws_tag2word_c = None
    # print('# Failed to load C func `cws_tag2word_c`, use py func instead.')


# load `cws_feature2idx_c`，加载分词的标签词汇转换 C 优化函数
try:
    file_list = os.listdir(os.path.join(dir_path, 'build'))
    file_name = ''
    for _file_name in file_list:
        if 'libcwsFeatureToIndex' in _file_name and _file_name.endswith('.so'):
            file_name = _file_name
            break

    cws_feature_to_idx = ctypes.PyDLL(os.path.join(dir_path, 'build', file_name))
    cws_feature2idx_c = cws_feature_to_idx.getFeatureIndex
    cws_feature2idx_c.argtypes = [ctypes.py_object, ctypes.py_object]
    cws_feature2idx_c.restype = ctypes.py_object

    # print('# Successfully load C func `cws_feature2idx_c`.')

except:
    cws_feature2idx_c = None
    # print('# Failed to load C func `cws_feature2idx_c`, use py func instead.')

if cws_get_node_features_c is not None and cws_tag2word_c is not None and cws_feature2idx_c is not None:
    print('# jiojio - Successfully load C funcs for acceleration.')
elif cws_get_node_features_c is None and cws_tag2word_c is None and cws_feature2idx_c is None:
    print('# jiojio - Failed to load C funcs, use py func instead.')
else:
    print('# jiojio - Loaded several C funcs, not all.')

from .feature_extractor import CWSFeatureExtractor
from .add_dict_to_model import CWSAddDict2Model
from .scorer import F1_score
from .predict_text import CWSPredictText
# from .tag_words_converter import word2tag, tag2word
