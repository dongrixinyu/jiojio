# -*- coding=utf-8 -*-
# Library: jiojio
# Author: dongrixinyu
# License: GPL-3.0
# Email: dongrixinyu.89@163.com
# Github: https://github.com/dongrixinyu/jiojio
# Description: fast Chinese Word Segmentation(CWS) and Part of Speech(POS) based on CPU.'

import os
import pdb
import sys
import numpy as np


class Model(object):
    def __init__(self, config, n_feature, n_tag):
        self.config = config
        self.n_tag = n_tag
        self.n_feature = n_feature

        if self.config.random_init:
            # 均值为 0 的均匀分布
            # self.w = np.random.random(
            #     size=(self.n_transition_feature,)) * 2 - 1

            # 正太分布更加符合真实情况
            self.node_weight = np.random.normal(
                loc=0.0, scale=0.4, size=(self.n_feature, self.n_tag))
            self.edge_weight = np.random.normal(
                loc=0.0, scale=0.4, size=(self.n_tag, self.n_tag))

            self.node_weight = self.node_weight.astype(np.float32)
            self.edge_weight = self.edge_weight.astype(np.float32)
            self.bi_ratio = np.array(np.random.random(), dtype=np.float32)  # 介于 0 ~ 1 的 float
            # self.bi_ratio = 0.5
        else:
            self.node_weight = np.zeros(self.n_feature, self.n_tag, dtype=np.float32)
            self.edge_weight = np.zeros(self.n_tag, self.n_tag, dtype=np.float32)
            self.bi_ratio = np.array(1., dtype=np.float32)

    @classmethod
    def load(cls, model_dir, dtype=np.float32):

        model_path = os.path.join(model_dir, 'weights.npz')
        if os.path.exists(model_path):
            npz = np.load(model_path)
            sizes = npz['sizes']
            bi_ratio = np.array(npz['bi_ratio'], dtype=dtype)

            node_weight = npz['node_weight'].astype(dtype)  # 强制转换 数据类型
            edge_weight = npz['edge_weight'].astype(dtype)  # 强制转换 数据类型

            model = cls.__new__(cls)
            model.n_tag = int(sizes[0])
            model.n_feature = int(sizes[1])
            model.bi_ratio = bi_ratio
            model.node_weight = node_weight
            model.edge_weight = edge_weight

            assert model.node_weight.shape[0] == model.n_feature
            assert model.node_weight.shape[1] == model.n_tag
            assert model.edge_weight.shape[0] == model.n_tag
            assert model.edge_weight.shape[1] == model.n_tag

            return model

        raise FileNotFoundError('the model file `{}` does not exist.'.format(model_path))

    def save(self):
        sizes = np.array([self.n_tag, self.n_feature])
        np.savez(os.path.join(self.config.model_dir, 'weights.npz'),
            sizes=sizes, bi_ratio=self.bi_ratio,
            node_weight=self.node_weight, edge_weight=self.edge_weight)
