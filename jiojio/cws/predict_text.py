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
from jiojio import Extractor

from . import cws_get_node_features_c, cws_tag2word_c
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
        self.model = Model.load(model_dir)

        self.idx_to_tag = {
            idx: tag for tag, idx in self.feature_extractor.tag_to_idx.items()}

        self.pre_processor = PreProcessor(
            convert_num_letter=cws_config.convert_num_letter,
            normalize_num_letter=cws_config.normalize_num_letter,
            convert_exception=cws_config.convert_exception)

        # C 方式调用
        self.get_node_features_c = cws_get_node_features_c
        self.tag2word_c = cws_tag2word_c

    def _cut(self, text):

        length = len(text)
        all_features = list()
        for idx in range(length):

            if self.get_node_features_c is None:
                # 以 python 方式计算，效率较低
                node_features = self.feature_extractor.get_node_features(idx, text)
            else:
                # 以 C 方式计算，效率高
                node_features = self.get_node_features_c(
                    idx, text, len(text), self.feature_extractor.unigram,
                    self.feature_extractor.bigram)
                # pdb.set_trace()

            # if node_features != self.feature_extractor.get_node_features(idx, text):
            #     print(node_features)
            #     print(self.feature_extractor.get_node_features(idx, text))
            #     pdb.set_trace()
            # 此处考虑，通用未匹配特征 “/”，即索引为 0 的特征
            node_feature_idx = [
                self.feature_extractor.feature_to_idx[node_feature]
                for node_feature in node_features
                if node_feature in self.feature_extractor.feature_to_idx]

            if len(node_feature_idx) != len(node_features):
                node_feature_idx.append(0)

            # node_feature_idx = map(lambda i:self.feature_extractor.feature_to_idx.get(i, 0),
            #                        node_features)
            all_features.append(node_feature_idx)

        Y = get_log_Y_YY(all_features, self.model.node_weight)

        # 添加词典
        if self.user_dict.trie_tree_obj is not None:
            self.user_dict(text, Y)

        if self.with_viterbi:
            tags_idx = viterbi(Y, self.model.edge_weight, bi_ratio=self.model.bi_ratio)
        else:
            tags_idx = tags_idx = Y.argmax(axis=1)

        return tags_idx

    def _cut_with_rule(self, text):

        rule_res_list = list()
        rule_res_list.extend(self.rule_extractor.extract_ip_address(text, detail=True))
        rule_res_list.extend(self.rule_extractor.extract_email(text, detail=True))
        rule_res_list.extend(self.rule_extractor.extract_id_card(text, detail=True))
        rule_res_list.extend(self.rule_extractor.extract_url(text, detail=True))
        rule_res_list.extend(self.rule_extractor.extract_phone_number(text, detail=True))

        start_flag = False
        end_flag = False

        if rule_res_list == []:
            # 空匹配
            return [text], [], False, False

        rule_res_list = sorted(rule_res_list, key=lambda item: item['offset'][0])

        seg_list = list()
        for idx in range(len(rule_res_list)):
            item = rule_res_list[idx]['offset']
            if idx == 0:
                if item[0] != 0:
                    seg_list.append(text[: item[0]])
                else:
                    start_flag = True

                next_item = rule_res_list[idx + 1]['offset']
                seg_list.append(text[item[1]: next_item[0]])

            elif idx == len(rule_res_list) - 1:
                if item[1] != len(text):
                    seg_list.append(text[item[1]:])
                else:
                    end_flag = True

            else:
                next_item = rule_res_list[idx + 1]['offset']
                seg_list.append(text[item[1]: next_item[0]])
        # pdb.set_trace()
        return seg_list, rule_res_list, start_flag, end_flag

    def cut(self, text):

        if not text:
            return list()

        norm_text = self.pre_processor(text)

        if self.rule_extractor:
            seg_list, rule_res_list, start_flag, end_flag = self._cut_with_rule(norm_text)
            seg_res_list = list()
            for segment in seg_list:

                tags = self._cut(segment)
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
                    words_list.append(rule_res_list[idx]['text'])
                    words_list.extend(seg_res_list[idx])
                if end_flag:
                    words_list.append(rule_res_list[-1])

            else:
                # pdb.set_trace()
                for idx in range(len(rule_res_list)):
                    words_list.extend(seg_res_list[idx])
                    words_list.append(rule_res_list[idx]['text'])
                if not end_flag:
                    words_list.extend(seg_res_list[-1])

            return words_list

        else:
            # 不进行分片切分的结果
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

        return words_list, norm_words_list