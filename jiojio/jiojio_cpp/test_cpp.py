# -*- coding=utf-8 -*-

import os
import numpy as np
import numpy.ctypeslib as npct
import ctypes
import pdb


dir_path = '/home/cuichengyu/github/jiojio/jiojio/jiojio_cpp'
tag_words_converter = ctypes.cdll.LoadLibrary(
    os.path.join(dir_path, 'build', 'libtagWordsConverter.so'))
# tag_words_converter = npct.load_library(
#     'libtagWordsConverter',
#     os.path.join(dir_path, 'build'))
tag2word = tag_words_converter.tagWordsConverter
tag2word.restype = None
# tag2word.restype = ctypes.py_object
# tag2word.argtypes = [ctypes.POINTER(ctypes.c_wchar), ctypes.py_object]  # ctypes.data_as(c_void_p)]
array_1d_int8 = npct.ndpointer(dtype=np.int8, ndim=1, flags='CONTIGUOUS')
# tag2word.argtypes = [ctypes.POINTER(ctypes.c_wchar)] # , array_1d_int8, ctypes.c_long]

text = "今天是个好天气！"
states = np.empty(len(text), dtype=np.int8)  # 1 个字节，np.int32 为 4个字节
for i in range(len(text)):
    states[i] = np.random.randint(2)

print(states, type(states))
pdb.set_trace()

# res = tag2word(text, states.ctypes.data_as(ctypes.c_void_p))
res = tag2word(ctypes.c_int(2)) #, states, len(states))
print(res)
