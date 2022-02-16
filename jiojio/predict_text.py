# -*- coding=utf-8 -*-

import os
import pdb
import sys
import ctypes

from jiojio.util import get_node_features_c, tag2word_c
from jiojio.tag_words_converter import tag2word
from jiojio.pre_processor import PreProcessor
from jiojio.feature_extractor import FeatureExtractor
from jiojio.inference import decodeViterbi_fast, get_log_Y_YY, run_viterbi
from jiojio.add_dict_to_model import AddDict2Model
from jiojio.model import Model


class PredictText(object):
    """ 预测文本，用于对外暴露接口 """
    def __init__(self, config, model_name=None, user_dict=None, pos=False):
        """初始化函数，加载模型及用户词典"""
        if model_name is not None:
            config.model_dir = model_name

        self.user_dict = AddDict2Model(user_dict)

        self.feature_extractor = FeatureExtractor.load(
            config, model_dir=config.model_dir)
        self.model = Model.load()

        self.idx_to_tag = {
            idx: tag for tag, idx in self.feature_extractor.tag_to_idx.items()}

        self.pre_processor = PreProcessor(
            convert_num_letter=config.convert_num_letter,
            normalize_num_letter=config.normalize_num_letter,
            convert_exception=config.convert_exception)

        self.pos = pos
        # if pos:
        #     download_model(config.model_urls["postag"], config.jiojio_home)
        #     postag_dir = os.path.join(config.jiojio_home, "postag")
        #     self.pos = Postag(postag_dir)

        # C 方式调用
        self.get_node_features_c = get_node_features_c
        self.tag2word_c = tag2word_c

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
                # print(node_features)
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

        tags_idx = run_viterbi(Y, self.model.edge_weight)

        return tags_idx

    def cut(self, text):

        if not text:
            return list()

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
            # pdb.set_trace()
        return words_list
