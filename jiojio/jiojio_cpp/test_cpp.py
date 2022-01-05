# -*- coding=utf-8 -*-

import os
import numpy as np
import numpy.ctypeslib as npct
import ctypes
import pdb
import jionlp as jio
from jiojio.tag_words_converter import tag2word


dir_path = '/home/ubuntu/github/jiojio/jiojio/jiojio_cpp'
feature_extractor = ctypes.cdll.LoadLibrary(
    os.path.join(dir_path, 'build', 'libfeatureExtractor.so'))
get_node_feature_c = feature_extractor.getNodeFeature
get_node_feature_c.restype = ctypes.py_object

text = "今天是个好天气！"
unigram = set(["天气", "今天", "中国"])
res = get_node_feature_c(4, text, len(text))
print(res)
print(type(res[0]))

'''
dir_path = '/home/ubuntu/github/jiojio/jiojio/jiojio_cpp'
tag_words_converter = ctypes.cdll.LoadLibrary(
    os.path.join(dir_path, 'build', 'libtagWordsConverter.so'))
# tag_words_converter = npct.load_library(
#     'libtagWordsConverter',
#     os.path.join(dir_path, 'build'))
tag2word_c = tag_words_converter.tagWordsConverter
tag2word_c.restype = ctypes.py_object
# tag2word.restype = ctypes.py_object
# tag2word.argtypes = [ctypes.POINTER(ctypes.c_wchar), ctypes.py_object]  # ctypes.data_as(c_void_p)]
array_1d_int8 = npct.ndpointer(dtype=np.int8, ndim=1, flags='CONTIGUOUS')
# tag2word.argtypes = [ctypes.POINTER(ctypes.c_wchar)] # , array_1d_int8, ctypes.c_long]

text = "今天是个好天气！"
text = text * 10
states = np.empty(len(text), dtype=np.int8)  # 1 个字节，np.int32 为 4个字节
for i in range(len(text)):
    states[i] = np.random.randint(2)

# states = np.array(
#     [1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
#      1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0,
#      0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0,
#      0, 0, 0], dtype=np.int8)
# text = '今天是个好天气！今天是个好天气！今天是个好天气！今天是个好天气！今天是个好天气！今天是个好天气！今天是个好天气！今天是' \
#        '个好天气！今天是个好天气！今天是个好天'
print(states, type(states))
print(text)

# pdb.set_trace()

# 100000 次计算耗时对比:
# c: 0.920   py: 15.572
# c: 0.950   py: 15.786
# c: 0.909   py: 15.557
# 修改一条 else 逻辑部分
# c: 0.787   py: 11.118
# c: 0.773   py: 11.722
# c: 0.788   py: 10.676


times = 10000
with jio.TimeIt('c '):
    for i in range(times):
        res = tag2word_c(text, states.ctypes.data_as(ctypes.c_void_p), len(states))
# res = tag2word(text, states, len(states))
# pdb.set_trace()
with jio.TimeIt('py '):
    for i in range(times):
        res1 = tag2word(text, states)

print("if the same: ", res == res1)
'''
