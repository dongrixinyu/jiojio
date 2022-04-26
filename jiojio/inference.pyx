# distutils: language = c++
# cython: infer_types=True
# cython: language_level=3

cimport cython
import numpy as np
cimport numpy as np

from libcpp.vector cimport vector

# -*- coding=utf-8 -*-
# Library: jiojio
# Author: dongrixinyu
# License: GPL-3.0
# Email: dongrixinyu.89@163.com
# Github: https://github.com/dongrixinyu/jiojio
# Description: fast Chinese Word Segmentation(CWS) and Part of Speech(POS) based on CPU.'


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_log_Y_YY(vector[vector[int]] sequence_feature_list,
                   np.ndarray[double, ndim=2] node_weight,
                   np.float16 dtype):

    cdef:
        int node_num = len(sequence_feature_list)
        int tag_num = node_weight.shape[1]
        # 每个节点的得分
        np.ndarray[double, ndim=2] node_score = np.empty((node_num, tag_num), dtype=dtype)

    for i in range(node_num):
        node_score[i] = np.sum(node_weight[sequence_feature_list[i]], axis=0)

    return node_score
