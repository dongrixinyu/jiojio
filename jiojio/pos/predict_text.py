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

from jiojio.pre_processor import PreProcessor
from jiojio.inference import get_log_Y_YY, viterbi
from jiojio.model import Model

from . import pos_get_node_features_c
from .config import Config
from .feature_extractor import POSFeatureExtractor
from .add_dict_to_model import POSAddDict2Model


class POSPredictText(object):
    """ 预测文本，用于对外暴露接口 """
    def __init__(self, model_dir=None, user_dict=None):
        """初始化函数，加载模型及用户词典"""
        default_model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

        # 指定分词模型目录
        if model_dir is None:
            model_dir = os.path.join(default_model_dir, 'default_pos_model')
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

        self.feature_extractor = POSFeatureExtractor.load(
            config=pos_config, model_dir=model_dir)
        self.model = Model.load(model_dir)

        self.idx_to_tag = {
            idx: tag for tag, idx in self.feature_extractor.tag_to_idx.items()}

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

            if self.get_node_features_c is None:
                # 以 python 方式计算，效率较低
                node_features = self.feature_extractor.get_node_features(idx, words)
            else:
                # 以 C 方式计算，效率高
                node_features = self.get_node_features_c(
                    idx, words, len(words), self.feature_extractor.unigram,
                    self.feature_extractor.bigram)

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

        Y = get_log_Y_YY(all_features, self.model.node_weight)
        '''
        for idx in range(length):
            print(words[idx])
            print(all_node_features[idx])
            print(all_features[idx])
            print(Y[idx])
            print(self.idx_to_tag[Y[idx].argmax()])
            pdb.set_trace()
        '''
        # 添加词典
        if self.user_dict is not None:
            self.user_dict(words, Y)
        # pure_tags_idx = Y.argmax(axis=1)
        tags_idx = viterbi(Y, self.model.edge_weight, bi_ratio=0.2)
        # pdb.set_trace()
        return tags_idx

    def cut(self, word_list):

        tags_idx = self._cut(word_list)
        tags_list = [self.idx_to_tag[idx] for idx in tags_idx]
        # pdb.set_trace()
        return tags_list
