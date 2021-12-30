# -*- coding=utf-8 -*-

import os
import pdb

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

        self.idx_to_tag = {idx: tag
            for tag, idx in self.feature_extractor.tag_to_idx.items()}

        self.pre_processor = PreProcessor(
            convert_num_letter=config.convert_num_letter,
            normalize_num_letter=config.normalize_num_letter,
            convert_exception=config.convert_exception)

        self.pos = pos
        if pos:
            download_model(config.model_urls["postag"], config.jiojio_home)
            postag_dir = os.path.join(config.jiojio_home, "postag")
            self.pos = Postag(postag_dir)

    def _cut(self, text):

        length = len(text)
        all_features = list()
        for idx in range(length):
            node_features = self.feature_extractor.get_node_features(idx, text)

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
        # tags = [self.idx_to_tag[tag_idx] for tag_idx in tags_idx]
        # return tags

    def cut(self, text):

        if not text:
            return list()

        norm_text = self.pre_processor(text)

        tags = self._cut(norm_text)

        words_list = tag2word(text, tags)
        if self.pos:
            tags = self.pos.tag(ret.copy())

        return words_list
