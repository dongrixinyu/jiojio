# -*- coding=utf-8 -*-

import os
import pdb
import ctypes
import traceback

from .time_it import TimeIt
from .zip_file import unzip_file, zip_file
from .file_io import read_file_by_iter, write_file_by_line
from .trie_tree import TrieTree


dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'jiojio_cpp')

# load `get_node_features_c`
try:
    file_list = os.listdir(os.path.join(dir_path, 'build'))
    file_name = ''
    for _file_name in file_list:
        if 'libfeatureExtractor' in _file_name and _file_name.endswith('.so'):
            file_name = _file_name
            break

    feature_extractor = ctypes.PyDLL(
        os.path.join(dir_path, 'build', file_name))
    get_node_features_c = feature_extractor.getNodeFeature
    get_node_features_c.argtypes = [
        ctypes.c_int, ctypes.c_wchar_p, ctypes.c_int, ctypes.py_object, ctypes.py_object]
    get_node_features_c.restype = ctypes.py_object

    print('# Successfully load C func `get_node_features_c`.')

except Exception:
    get_node_features_c = None
    print('# Failed to load C func `get_node_features_c`, use py func instead.')
    # print('# Failed to load C func `get_node_features_c` {}.'.format(
    #     traceback.format_exc()))

# load `tag_words_converter_c`
try:
    file_list = os.listdir(os.path.join(dir_path, 'build'))
    file_name = ''
    for _file_name in file_list:
        if 'libtagWordsConverter' in _file_name and _file_name.endswith('.so'):
            file_name = _file_name
            break

    tag_words_converter = ctypes.PyDLL(
        os.path.join(dir_path, 'build', file_name))
    tag2word_c = tag_words_converter.tagWordsConverter
    # tag2word_c.argtypes = [ctypes.c_wchar_p, ctypes.py_object]
    tag2word_c.restype = ctypes.py_object

    print('# Successfully load C func `tag2word_c`.')

except Exception:
    tag2word_c = None
    print('# Failed to load C func `tag2word_c`, use py func instead.')
    # print('# Failed to load C func `tag2word_c` {}.'.format(
    #     traceback.format_exc()))
