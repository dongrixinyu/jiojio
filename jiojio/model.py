import os
import pdb
import sys
import numpy as np

from .config import config


class Model(object):
    def __init__(self, n_feature, n_tag):
        self.n_tag = n_tag
        self.n_feature = n_feature

        if config.random_init:
            # 均值为 0 的均匀分布
            # self.w = np.random.random(
            #     size=(self.n_transition_feature,)) * 2 - 1

            # 正太分布更加符合真实情况
            self.node_weight = np.random.normal(
                loc=0.0, scale=0.4, size=(self.n_feature, self.n_tag))
            self.edge_weight = np.random.normal(
                loc=0.0, scale=0.4, size=(self.n_tag, self.n_tag))
        else:
            self.node_weight = np.zeros(self.n_feature, self.n_tag)
            self.edge_weight = np.zeros(self.n_tag, self.n_tag)

    @classmethod
    def load(cls, model_dir=None):
        if model_dir is None:
            model_dir = config.model_dir

        model_path = os.path.join(model_dir, "weights.npz")
        if os.path.exists(model_path):
            npz = np.load(model_path)
            sizes = npz["sizes"]
            node_weight = npz["node_weight"].astype(np.float32)  # 强制转换 数据类型
            edge_weight = npz["edge_weight"].astype(np.float32)  # 强制转换 数据类型

            model = cls.__new__(cls)
            model.n_tag = int(sizes[0])
            model.n_feature = int(sizes[1])
            model.node_weight = node_weight
            model.edge_weight = edge_weight

            assert model.node_weight.shape[0] == model.n_feature
            assert model.node_weight.shape[1] == model.n_tag
            assert model.edge_weight.shape[0] == model.n_tag
            assert model.edge_weight.shape[1] == model.n_tag

            return model

        raise FileNotFoundError('the file `{}` does not exist.'.format(model_path))

    def save(self, model_dir=None):
        if model_dir is None:
            model_dir = config.model_dir

        sizes = np.array([self.n_tag, self.n_feature])
        np.savez(os.path.join(model_dir, "weights.npz"),
            sizes=sizes, node_weight=self.node_weight, edge_weight=self.edge_weight)
