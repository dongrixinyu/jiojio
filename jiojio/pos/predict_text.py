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
import json
import ctypes
import numpy as np

from jiojio.pre_processor import PreProcessor
from jiojio.inference import get_log_Y_YY, viterbi
from jiojio.model import Model
from jiojio import download_model

from . import pos_get_node_features_c
from .config import Config
from .feature_extractor import POSFeatureExtractor
from .add_dict_to_model import POSAddDict2Model


class POSPredictText(object):
    """ 预测文本，用于对外暴露接口 """
    def __init__(self, model_dir=None, user_dict=None, with_viterbi=True,
                 pos_rule_types=True):
        """初始化函数，加载模型及用户词典"""
        default_model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

        # 指定 POS 模型目录
        if model_dir is None:
            model_dir = os.path.join(default_model_dir, 'default_pos_model')

            if not os.path.exists(os.path.join(model_dir, 'weights.npz')):  # 下载模型
                default_url = 'https://github.com/dongrixinyu/jiojio/releases/download/v1.1.4/default_pos_model.zip'
                # 备用链接，也可用
                # default_url = 'http://182.92.160.94:17777/jio_api/default_pos_model.zip'
                download_model(default_url, default_model_dir)

        else:
            if os.path.isabs(model_dir):
                pass
            else:
                model_dir = os.path.join(default_model_dir, model_dir)

        # 加载推理参数
        with open(os.path.join(model_dir, 'params.json'), 'r', encoding='utf-8') as fr:
            config = json.load(fr)

        pos_config = Config()
        for key, val in config.items():
            if hasattr(pos_config, key):
                setattr(pos_config, key, val)

        if user_dict is not None:
            self.user_dict = POSAddDict2Model(user_dict)
        else:
            self.user_dict = None

        self.with_viterbi = with_viterbi
        self.pos_rule_types = pos_rule_types

        self.feature_extractor = POSFeatureExtractor.load(
            config=pos_config, model_dir=model_dir)

        self.dtype = np.float16  # 原因在于模型不需要太高精度就可以做区分，此值不需要变
        self.model = Model.load(model_dir, dtype=self.dtype)

        self.idx_to_tag = {
            idx: tag for tag, idx in self.feature_extractor.tag_to_idx.items()}
        self.tag_num = len(self.idx_to_tag)

        self.pre_processor = PreProcessor(
            convert_num_letter=pos_config.convert_num_letter,
            normalize_num_letter=pos_config.normalize_num_letter,
            convert_exception=pos_config.convert_exception)

        # C 方式调用
        self.get_node_features_c = pos_get_node_features_c

    def _cut(self, words):

        length = len(words)
        all_features = list()
        all_node_features = list()
        for idx in range(length):

            # if self.get_node_features_c is None:
            #     # 以 python 方式计算，效率较低
            node_features = self.feature_extractor.get_node_features(idx, words)
            # else:
            #     # 以 C 方式计算，效率高
            #     node_features = self.get_node_features_c(
            #         idx, words, len(words), self.feature_extractor.unigram,
            #         self.feature_extractor.bigram)

            # if node_features != self.feature_extractor.get_node_features(idx, words):
            #     print(node_features)
            #     print(self.feature_extractor.get_node_features(idx, words))
            all_node_features.append(node_features)

            node_feature_idx = [
                self.feature_extractor.feature_to_idx[node_feature]
                for node_feature in node_features
                if node_feature in self.feature_extractor.feature_to_idx]

            # pdb.set_trace()
            # node_feature_idx = map(lambda i:self.feature_extractor.feature_to_idx.get(i, 0),
            #                        node_features)
            all_features.append(node_feature_idx)

        # pdb.set_trace()
        Y = get_log_Y_YY(all_features, self.model.node_weight, dtype=self.dtype)

        # 添加词典
        if self.user_dict is not None:
            self.user_dict(words, Y)

        if self.with_viterbi:
            tags_idx = viterbi(Y, self.model.edge_weight, bi_ratio=self.model.bi_ratio)
        else:
            tags_idx = Y.argmax(axis=1)

        return tags_idx

    def cut(self, word_list, word_pos_map=None):

        tags_idx = self._cut(word_list)
        tags_list = [self.idx_to_tag[idx] for idx in tags_idx]
        if word_pos_map is None:
            return tags_list

        else:
            for idx in range(len(word_list)):
                if word_list[idx] in word_pos_map:
                    tags_list[idx] = word_pos_map[word_list[idx]]

            # pdb.set_trace()
            return tags_list
