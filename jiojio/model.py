import os
import sys
import numpy as np

from .config import config


class Model(object):
    def __init__(self, n_feature, n_tag):
        self.n_tag = n_tag
        self.n_feature = n_feature
        self.offset = self.n_tag * self.n_feature
        self.n_transition_feature = n_tag * (n_feature + n_tag)
        if config.random:
            # 均值为 0 的均匀分布
            # self.w = np.random.random(
            #     size=(self.n_transition_feature,)) * 2 - 1

            # 正太分布更加符合真实情况
            self.w = np.random.normal(
                loc=0.0, scale=0.4, size=(self.n_transition_feature,))
        else:
            self.w = np.zeros(self.n_transition_feature)

    def expand(self, n_feature, n_tag):
        # 在原有模型特征基础上，扩展特征和标签
        new_transition_feature = n_tag * (n_feature + n_tag)
        if config.random:
            new_w = np.random.random(size=(new_transition_feature,)) * 2 - 1
        else:
            new_w = np.zeros(new_transition_feature)

        n_node = self.n_tag * self.n_feature
        n_edge = self.n_tag * self.n_tag
        new_w[:n_node] = self.w[:n_node]
        new_w[-n_edge:] = self.w[-n_edge:]
        self.n_tag = n_tag
        self.n_feature = n_feature
        self.offset = self.n_tag * self.n_feature
        self.n_transition_feature = new_transition_feature
        self.w = new_w

    @classmethod
    def load(cls, model_dir=None):
        if model_dir is None:
            model_dir = config.model_dir

        model_path = os.path.join(model_dir, "weights.npz")
        if os.path.exists(model_path):
            npz = np.load(model_path)
            sizes = npz["sizes"]
            w = npz["w"]

            model = cls.__new__(cls)
            model.n_tag = int(sizes[0])
            model.n_feature = int(sizes[1])
            model.offset = model.n_tag * model.n_feature
            model.n_transition_feature = model.n_tag * \
                (model.n_feature + model.n_tag)
            model.w = w

            assert model.w.shape[0] == model.n_transition_feature
            return model

        raise FileNotFoundError('the file {} does not exist.'.format(model_path))

    @classmethod
    def new(cls, model, copy_weight=True):
        new_model = cls.__new__(cls)
        new_model.n_tag = model.n_tag
        if copy_weight:
            new_model.w = model.w.copy()
        else:
            new_model.w = np.zeros_like(model.w)

        new_model.n_feature = (
            new_model.w.shape[0] // new_model.n_tag - new_model.n_tag)
        new_model.offset = new_model.n_tag * new_model.n_feature
        new_model.n_transition_feature = new_model.w.shape[0]

        return new_model

    def save(self, model_dir=None):
        if model_dir is None:
            model_dir = config.model_dir

        sizes = np.array([self.n_tag, self.n_feature])
        np.savez(os.path.join(model_dir, "weights.npz"), sizes=sizes, w=self.w)
