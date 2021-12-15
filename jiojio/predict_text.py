# -*- coding=utf-8 -*-

import os

from jiojio.tag_words_converter import tag2word
from jiojio.lexicon_cut import LexiconCut
from jiojio.pre_processor import PreProcessor
from jiojio.feature_extractor import FeatureExtractor
from jiojio.inference import decodeViterbi_fast

from jiojio.model import Model


class PredictText(object):
    def __init__(self, model_name="default_model", user_dict="default", postag=False):
        """初始化函数，加载模型及用户词典"""
        self.postag = postag

        if model_name in ["default_model"]:
            config.model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                "models", model_name)
        else:
            config.model_dir = model_name
        # pdb.set_trace()
        self.feature_extractor = FeatureExtractor.load(model_dir=config.model_dir)
        self.model = Model.load()

        self.idx_to_tag = {
            idx: tag for tag, idx in self.feature_extractor.tag_to_idx.items()}

        self.pre_processor = PreProcessor()


        if postag:
            download_model(config.model_urls["postag"], config.jiojio_home)
            postag_dir = os.path.join(config.jiojio_home, "postag")
            self.tagger = Postag(postag_dir)

    def _cut(self, text):
        examples = self.pre_processor(text)
        length = len(examples)

        all_features = list()
        for idx in range(length):
            node_features = self.feature_extractor.get_node_features(idx, examples)

            # 此处考虑，通用未匹配特征 “/”，即索引未 0 的特征
            node_feature_idx = [self.feature_extractor.feature_to_idx[node_feature]
                                for node_feature in node_features
                                if node_feature in self.feature_extractor.feature_to_idx]

            if len(node_feature_idx) != len(node_features):
                node_feature_idx.append(0)

            # node_feature_idx = map(lambda i:self.feature_extractor.feature_to_idx.get(i, 0),
            #                        node_features)

            all_features.append(node_feature_idx)

        tags_idx = decodeViterbi_fast(all_features, self.model)
        return tags_idx
        # tags = [self.idx_to_tag[tag_idx] for tag_idx in tags_idx]
        # tags = map(lambda i:self.idx_to_tag[i], tags_idx)  # for speeding up

        # return tags

    def cut(self, text, convert_num_letter=True, normalize_num_letter=True,
            convert_exception=True):

        if not text:
            return list()

        norm_text = self.pre_processor(
            text, convert_num_letter=convert_num_letter,
            normalize_num_letter=normalize_num_letter,
            convert_exception=convert_exception)

        tags = self._cut(norm_text)

        words_list = tag2word(text, tags)
        if self.postag:
            tags = self.tagger.tag(ret.copy())
            for i, user_tag in enumerate(user_tags):
                if user_tag:
                    tags[i] = user_tag
            ret = list(zip(ret, tags))

        return words_list
