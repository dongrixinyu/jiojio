# -*- coding=utf-8 -*-

import os
import pdb
import ctypes

from .time_it import TimeIt
from .zip_file import unzip_file, zip_file
from .file_io import read_file_by_iter, write_file_by_line  # , read_file_by_line,
from .trie_tree import TrieTree


dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'jiojio_cpp')

try:
    feature_extractor = ctypes.cdll.LoadLibrary(
        os.path.join(dir_path, 'build', 'libfeatureExtractor.so'))
    get_node_features_c = feature_extractor.getNodeFeature
    get_node_features_c.argtypes = [
        ctypes.c_int, ctypes.c_wchar_p, ctypes.c_int, ctypes.py_object, ctypes.py_object]
    get_node_features_c.restype = ctypes.py_object

    print('# Successfully load C func `get_node_features_c`.')
except Exception:
    get_node_features_c = None
    print('# Failed to load C func `get_node_features_c`.')
