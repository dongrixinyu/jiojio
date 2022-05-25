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
from jiojio import Extractor

from . import cws_get_node_features_c, cws_tag2word_c, cws_feature2idx_c
from .config import Config
from .tag_words_converter import tag2word
from .feature_extractor import CWSFeatureExtractor
from .add_dict_to_model import CWSAddDict2Model


class CWSPredictText(object):
    """ 预测文本，用于对外暴露接口 """
    def __init__(self, model_dir=None, user_dict=None, with_viterbi=False,
                 rule_extractor=False):
        """初始化函数，加载模型及用户词典"""
        default_model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

        self.with_viterbi = with_viterbi

        # 采用规则将某些内容抽取出
        if rule_extractor:
            self.rule_extractor = Extractor()
        else:
            self.rule_extractor = None

        # 指定分词模型目录
        if model_dir is None:
            model_dir = os.path.join(default_model_dir, 'default_cws_model')
        else:
            if os.path.isabs(model_dir):
                pass
            else:
                model_dir = os.path.join(default_model_dir, model_dir)

        self.user_dict = CWSAddDict2Model(user_dict)

        # 加载推理参数
        with open(os.path.join(model_dir, 'params.json'), 'r', encoding='utf-8') as fr:
            config = json.load(fr)

        cws_config = Config()
        for key, val in config.items():
            if hasattr(cws_config, key):
                setattr(cws_config, key, val)

        self.feature_extractor = CWSFeatureExtractor.load(
            config=cws_config, model_dir=model_dir)
        self.model = Model.load(model_dir, task='cws')
        # self.model = Model.load(model_dir, task=None)  # 与 cws 指定有区别

        self.idx_to_tag = {
            idx: tag for tag, idx in self.feature_extractor.tag_to_idx.items()}
        self.tag_num = len(self.idx_to_tag)

        self.pre_processor = PreProcessor(
            convert_num_letter=cws_config.convert_num_letter,
            normalize_num_letter=cws_config.normalize_num_letter,
            convert_exception=cws_config.convert_exception)

        # C 方式调用
        self.get_node_features_c = cws_get_node_features_c
        self.tag2word_c = cws_tag2word_c
        self.cws_feature2idx_c = cws_feature2idx_c

    def _cut(self, text):
        length = len(text)
        all_features = list()

        # 每个节点的得分
        # Y = np.empty((length, 2), dtype=np.float16)
        for idx in range(length):

            if self.get_node_features_c is None:
                # 以 python 方式计算，效率较低
                node_features = self.feature_extractor.get_node_features(idx, text)
            else:
                # 以 C 方式计算，效率高
                node_features = self.get_node_features_c(
                    idx, text, length, self.feature_extractor.unigram,
                    self.feature_extractor.bigram)

            # 测试:
            # if node_features != self.feature_extractor.get_node_features(idx, text):
            #     print(node_features)
            #     print(self.feature_extractor.get_node_features(idx, text))
            #     pdb.set_trace()

            if self.cws_feature2idx_c is None:
                # 此处考虑，通用未匹配特征 “/”，即索引为 0 的特征
                node_feature_idx = [
                    self.feature_extractor.feature_to_idx[node_feature]
                    for node_feature in node_features
                    if node_feature in self.feature_extractor.feature_to_idx]

                if len(node_feature_idx) != len(node_features):
                    node_feature_idx.append(0)

            else:
                node_feature_idx = self.cws_feature2idx_c(
                    node_features, self.feature_extractor.feature_to_idx)

            all_features.append(node_feature_idx)
            # Y[idx] = np.sum(node_weight[node_feature_idx], axis=0)

        Y = get_log_Y_YY(all_features, self.model.node_weight, dtype=np.float16)

        # 添加词典
        if self.user_dict.trie_tree_obj is not None:
            self.user_dict(text, Y)

        if self.with_viterbi:
            tags_idx = viterbi(
                Y, self.model.edge_weight, bi_ratio=self.model.bi_ratio,
                dtype=np.float16)
        else:
            tags_idx = Y.argmax(axis=1).astype(np.int8)

        # print(tags_idx)
        # pdb.set_trace()
        return tags_idx

    def _cut_with_rule(self, text):
        """采用规则抽取出某些词汇，如ip地址，email，身份证号，url，电话号码等等"""
        rule_res_list = self.rule_extractor.extract_info(text)

        # pdb.set_trace()
        if len(rule_res_list) == 0:
            # 空匹配
            return [text], [], False, False

        start_flag = False
        end_flag = False

        if len(rule_res_list) == 1:
            rule_res_length = 1
        else:
            _rule_res_list = sorted(rule_res_list, key=lambda item: item['o'][0])

            # 将错误的信息进行过滤
            rule_res_list = [_rule_res_list[0]]
            for item in _rule_res_list[1:]:
                if item['o'][0] < rule_res_list[-1]['o'][1]:
                    continue
                rule_res_list.append(item)

            rule_res_length = len(rule_res_list)

        seg_list = list()
        for idx in range(rule_res_length):
            item = rule_res_list[idx]['o']
            if idx == 0:
                if item[0] != 0:
                    seg_list.append(text[: item[0]])
                else:
                    start_flag = True

                if rule_res_length == 1:
                    seg_list.append(text[item[1]:])
                else:
                    next_item = rule_res_list[idx + 1]['o']
                    seg_list.append(text[item[1]: next_item[0]])

            elif idx == len(rule_res_list) - 1:
                if item[1] != len(text):
                    seg_list.append(text[item[1]:])
                else:
                    end_flag = True

            else:
                next_item = rule_res_list[idx + 1]['o']
                seg_list.append(text[item[1]: next_item[0]])

        return seg_list, rule_res_list, start_flag, end_flag

    def cut(self, text):

        if not text:
            return list()

        if self.rule_extractor:
            seg_list, rule_res_list, start_flag, end_flag = self._cut_with_rule(text)

            seg_res_list = list()
            for segment in seg_list:

                norm_segment = self.pre_processor(segment)
                tags = self._cut(norm_segment)

                if self.tag2word_c is None:
                    # 以 python 方式进行计算
                    words_list = tag2word(segment, tags)
                else:
                    # 以 C 方式进行计算
                    words_list = self.tag2word_c(
                        segment, tags.ctypes.data_as(ctypes.c_void_p), len(tags))

                seg_res_list.append(words_list)

            words_list = list()

            if start_flag:
                for idx in range(len(seg_res_list)):
                    words_list.append(rule_res_list[idx]['s'])
                    words_list.extend(seg_res_list[idx])
                if end_flag:
                    words_list.append(rule_res_list[-1])

            else:
                # pdb.set_trace()
                for idx in range(len(rule_res_list)):
                    words_list.extend(seg_res_list[idx])
                    words_list.append(rule_res_list[idx]['s'])
                if not end_flag:
                    words_list.extend(seg_res_list[-1])

            return words_list

        else:
            # 不进行分片切分的结果
            norm_text = self.pre_processor(text)
            tags = self._cut(norm_text)

            if self.tag2word_c is None:
                # 以 python 方式进行计算
                words_list = tag2word(text, tags)
            else:
                # 以 C 方式进行计算
                words_list = self.tag2word_c(
                    text, tags.ctypes.data_as(ctypes.c_void_p), len(tags))
                # print(words_list)

            return words_list

    def cut_with_pos(self, text):
        """ 当后续要接 POS 任务时，需要返回两个 words_list """
        if not text:
            return list()

        if self.rule_extractor:
            seg_list, rule_res_list, start_flag, end_flag = self._cut_with_rule(text)
            seg_res_list = list()
            norm_seg_res_list = list()
            for segment in seg_list:

                norm_segment = self.pre_processor(segment)
                tags = self._cut(norm_segment)

                if self.tag2word_c is None:
                    # 以 python 方式进行计算
                    words_list = tag2word(segment, tags)
                    norm_words_list = tag2word(norm_segment, tags)
                else:
                    # 以 C 方式进行计算
                    words_list = self.tag2word_c(
                        segment, tags.ctypes.data_as(ctypes.c_void_p), len(tags))
                    norm_words_list = self.tag2word_c(
                        norm_segment, tags.ctypes.data_as(ctypes.c_void_p), len(tags))

                seg_res_list.append(words_list)
                norm_seg_res_list.append(norm_words_list)

            words_list = list()
            norm_words_list = list()

            if start_flag:
                for idx in range(len(seg_res_list)):
                    words_list.append(rule_res_list[idx]['s'])
                    words_list.extend(seg_res_list[idx])

                    norm_words_list.append(rule_res_list[idx]['s'])
                    norm_words_list.extend(norm_seg_res_list[idx])
                if end_flag:
                    words_list.append(rule_res_list[-1])
                    norm_words_list.append(rule_res_list[-1])

            else:
                # pdb.set_trace()
                for idx in range(len(rule_res_list)):
                    words_list.extend(seg_res_list[idx])
                    words_list.append(rule_res_list[idx]['s'])

                    norm_words_list.extend(norm_seg_res_list[idx])
                    norm_words_list.append(rule_res_list[idx]['s'])
                if not end_flag:
                    words_list.extend(seg_res_list[-1])
                    norm_words_list.extend(norm_seg_res_list[-1])

            if len(rule_res_list) == 0:  # 为了避免一遍循环
                word_pos_map = None

            else:
                word_pos_map = dict()
                for item in rule_res_list:
                    word_pos_map.update({item['s']: item['t']})

            return words_list, norm_words_list, word_pos_map

        else:
            norm_text = self.pre_processor(text)
            tags = self._cut(norm_text)

            if self.tag2word_c is None:
                # 以 python 方式进行计算
                words_list = tag2word(text, tags)
                norm_words_list = tag2word(norm_text, tags)
            else:
                # 以 C 方式进行计算
                words_list = self.tag2word_c(
                    text, tags.ctypes.data_as(ctypes.c_void_p), len(tags))
                norm_words_list = self.tag2word_c(
                    norm_text, tags.ctypes.data_as(ctypes.c_void_p), len(tags))
                # pdb.set_trace()

            return words_list, norm_words_list, None
