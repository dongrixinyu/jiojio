# distutils: language = c++
# cython: infer_types=True
# cython: language_level=3

import json
import os
import pdb
import sys
from collections import Counter

import jionlp as jio

from jiojio import logging
from jiojio.tag_words_converter import word2tag
from jiojio.util import unzip_file
from jiojio.config import config
from jiojio.tag_words_converter import tag2word
from jiojio.pre_processor import PreProcessor


def get_slice_str(iterator_obj, start, length, all_len):
    # 截取字符串，其中，iterable 为字、词列表

    if start < 0 or start >= all_len:
        return ""
    if start + length > all_len:
        return ""

    if type(iterator_obj) is str:
        return iterator_obj[start: start + length]
    else:
        return "".join(iterator_obj[start: start + length])


class FeatureExtractor(object):

    def __init__(self):
        self.unigram = set()
        self.bigram = set()
        self.feature_to_idx = dict()
        self.tag_to_idx = dict()
        self.pre_processor = PreProcessor()

        self._create_features()

    def _create_features(self):
        self.start_feature = '[START]'
        self.end_feature = '[END]'
        self.norm_num = '7'  # 与 pre_processor.py文件中定义一致
        self.norm_letter = 'Z'  # 与 pre_processor.py文件中定义一致

        self.delim = '.'
        self.empty_feature = '/'
        self.default_feature = '$$'

        self.char_current = 'c.'
        self.char_before = 'c-1.'
        self.char_next = 'c1.'
        self.char_before_2 = 'c-2.'
        self.char_next_2 = 'c2.'
        self.char_before_current = 'c-1c.'
        self.char_current_next = 'cc1.'
        self.char_before_2_1 = "c-2c-1."
        self.char_next_1_2 = "c1c2."

        self.word_before = "w-1."
        self.word_next = "w1."
        self.word_2_left = "ww.l."
        self.word_2_right = "ww.r."

        self.no_word = "**noWord"

    def build(self, train_file):
        word_length_info = Counter()
        # specials = set()
        feature_freq = Counter()  # 计算各个特征的出现次数，减少罕见 token 计数

        unigrams = Counter()  # 计算各个 unigrams 出现次数，避免罕见 unigram 进入计数
        bigrams = Counter()  # 计算各个 bigrams 出现次数，避免罕见 bigram 进入计数
        for sample_idx, words in enumerate(jio.read_file_by_iter(train_file)):

            if sample_idx % 100000 == 0:
                print(sample_idx)
            # first pass to collect unigram and bigram and tag info
            word_length_info.update(map(len, words))
            # specials.update(word for word in words if len(word) >= 10)

            # 对文本进行归一化和整理
            if config.norm_text:
                words = [self.pre_processor(word, convert_exception=True,
                            convert_num_letter=True, normalize_num_letter=False)
                            for word in words]

            example = ''.join(words)
            unigrams.update(words)

            for pre, suf in zip(words[:-1], words[1:]):
                # self.bigram.add("{}*{}".format(pre, suf))
                bigrams.update("{}*{}".format(pre, suf))

            # second pass to get features
            for idx in range(len(example)):
                node_features = self.get_node_features(idx, example)
                feature_freq.update(feature for feature in node_features)

        # 对 self.unigram 的清理和整理
        # 若 self.unigram 中的频次都不足 feature_trim 则，在后续特征删除时必然频次不足
        self.unigram = set([unigram for unigram, freq in unigrams.most_common()
                            if freq > config.feature_trim])
        if '' in self.unigram:  # 删除错误 token
            self.unigram.remove('')

        # 对 self.bigram 的清理和整理
        self.bigram = set([bigram for bigram, freq in bigrams.most_common()
                           if freq > config.feature_trim])

        # print token length counter
        short_token_total_length = 0
        total_length = sum(list(word_length_info.values()))
        print("token length\ttoken num\tratio:")
        for length in range(1, 12):
            if length <= 10:
                short_token_total_length += word_length_info[length]
                print("\t{} : {} : {:.2%}".format(
                    length, word_length_info[length],
                    word_length_info[length] / total_length))
            else:
                print("\t{}+: {} : {:.2%}".format(
                    length, total_length - short_token_total_length,
                    (total_length - short_token_total_length) / total_length))
        # print('special words num: {}\n'.format(specials))
        # print(json.dumps(list(specials), ensure_ascii=False))

        print('# orig feature_num: {}'.format(len(feature_freq)))
        print('# {:.2%} features are saved.'.format(
            sum([freq for _, freq in feature_freq.most_common()
                 if freq > config.feature_trim]) / sum(list(feature_freq.values()))))

        feature_set = (feature for feature, freq in feature_freq.most_common()
                       if freq > config.feature_trim)

        self.feature_to_idx = {feature: idx for idx, feature in enumerate(feature_set, 3)}
        # 特殊 token 须加入，如 空特征，起始特征，结束特征等
        self.feature_to_idx.update({self.empty_feature: 0})  # 空特征更新为第一个
        self.feature_to_idx.update({self.start_feature: 1})
        self.feature_to_idx.update({self.end_feature: 2})
        print('# true feature_num: {}'.format(len(self.feature_to_idx)))

        # create tag map
        B, B_single, I_first, I, I_end = FeatureExtractor._create_label()
        tag_set = {B, B_single, I_first, I, I_end}
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(sorted(tag_set))}
        assert self.tag_to_idx == {'B': 0, 'I': 1}, \
            'tag map must be like this for speeding up inferencing.'
        # self.idx_to_tag = FeatureExtractor._reverse_dict(self.tag_to_idx)

    def get_node_features(self, idx, token_list):
        # 给定一个  token_list，找出其中 token_list[idx] 匹配到的所有特征
        length = len(token_list)
        w = token_list[idx]
        feature_list = list()

        # 1 start feature
        feature_list.append(self.default_feature)

        # 8 unigram/bgiram feature
        # 当前字特征
        feature_list.append(self.char_current + w)

        # 前一个字特征
        if idx > 0:
            feature_list.append(self.char_before + token_list[idx - 1])
        else:
            # 字符为起始位特征
            feature_list.append(self.start_feature)

        # 后一个字特征
        if idx < len(token_list) - 1:
            feature_list.append(self.char_next + token_list[idx + 1])
        else:
            # 字符为终止位特征
            feature_list.append(self.end_feature)

        # 前第二字特征
        if idx > 1:
            feature_list.append(self.char_before_2 + token_list[idx - 2])

        # 后第二字特征
        if idx < len(token_list) - 2:
            feature_list.append(self.char_next_2 + token_list[idx + 2])

        # 当前字和前一字组合特征
        if idx > 0:
            feature_list.append(self.char_before_current + token_list[idx - 1] + self.delim + w)

        # 当前字和后一字组合特征
        if idx < len(token_list) - 1:
            feature_list.append(self.char_current_next + w + self.delim + token_list[idx + 1])

        # 前一字和前第二字组合
        if idx > 1:
            feature_list.append(self.char_before_2_1 + token_list[idx - 2] + self.delim + token_list[idx - 1])

        # 后一字和后第二字组合
        if idx < len(token_list) - 2:
            feature_list.append(self.char_next_1_2 + token_list[idx + 1] + self.delim + token_list[idx + 2])

        # no num/letter based features
        if not config.word_feature:
            return feature_list

        # 2 * (wordMax-wordMin+1) word features (default: 2*(6-2+1)=10 )
        # the character starts or ends a word

        # 寻找该字前一个词汇特征(包含该字)
        pre_list_in = list()
        for l in range(config.word_max, config.word_min - 1, -1):
            # length 6 ... 2 (default)
            # "prefix including current c" token_list[n-l+1, n]
            # current character ends word
            tmp = get_slice_str(token_list, idx - l + 1, l, length)
            if tmp in self.unigram:
                feature_list.append(self.word_before + tmp)
                pre_list_in.append(tmp)
                break  # 此 break 的取舍很重要
            # else:
            #     pre_list_in.append(self.no_word)

        # 寻找该字后一个词汇特征(包含该字)
        post_list_in = list()
        for l in range(config.word_max, config.word_min - 1, -1):
            # "suffix" token_list[n, n+l-1]
            # current character starts word
            tmp = get_slice_str(token_list, idx, l, length)
            if tmp in self.unigram:
                feature_list.append(self.word_next + tmp)
                post_list_in.append(tmp)
                break  # 此 break 的取舍很重要
            # else:
            #     post_list_in.append(self.no_word)

        # 寻找该字前一个词汇特征(不包含该字)
        pre_list_ex = list()
        for l in range(config.word_max, config.word_min - 1, -1):
            # "prefix excluding current c" token_list[n-l, n-1]
            tmp = get_slice_str(token_list, idx - l, l, length)
            if tmp in self.unigram:
                pre_list_ex.append(tmp)
                break  # 此 break 的取舍很重要
            # else:
            #     pre_list_ex.append(self.no_word)

        # 寻找该字后一个词汇特征(不包含该字)
        post_list_ex = list()  # 其中个数由 word_max 决定
        for l in range(config.word_max, config.word_min - 1, -1):
            # "suffix excluding current c" token_list[n+1, n+l]
            tmp = get_slice_str(token_list, idx + 1, l, length)
            if tmp in self.unigram:
                post_list_ex.append(tmp)
                break  # 此 break 的取舍很重要
            # else:
            #     post_list_ex.append(self.no_word)

        # this character is in the middle of a word
        # 2 * (wordMax - wordMin + 1) ^ 2 (default: 2 * (6 -2 + 1) ^ 2 = 50)
        # 寻找连续两个词汇特征(该字在右侧词汇中)
        for pre in pre_list_ex:
            for post in post_list_in:
                bigram = pre + "*" + post
                if bigram in self.bigram:
                    feature_list.append(self.word_2_left + bigram)

        # 寻找连续两个词汇特征(该字在左侧词汇中)
        for pre in pre_list_in:
            for post in post_list_ex:
                bigram = pre + "*" + post
                if bigram in self.bigram:
                    feature_list.append(self.word_2_right + bigram)

        return feature_list

    def convert_feature_file_to_idx_file(self, feature_file,
                                         feature_idx_file, tag_idx_file):

        with open(feature_file, "r", encoding="utf8") as reader:
            with open(feature_idx_file, "w", encoding="utf8") as f_writer, \
                    open(tag_idx_file, "w", encoding="utf8") as t_writer:

                f_writer.write("{}\n\n".format(len(self.feature_to_idx)))
                t_writer.write("{}\n\n".format(len(self.tag_to_idx)))

                tags_idx = list()
                features_idx = list()
                while True:
                    line = reader.readline()
                    if line == '':
                        break
                    line = line.strip()
                    if not line:
                        # sentence finish
                        for feature_idx in features_idx:
                            if not feature_idx:
                                f_writer.write("0\n")
                            else:
                                f_writer.write(",".join(map(str, feature_idx)))
                                f_writer.write("\n")
                        f_writer.write("\n")

                        t_writer.write(",".join(map(str, tags_idx)))
                        t_writer.write("\n\n")

                        tags_idx = list()
                        features_idx = list()
                        continue

                    splits = json.loads(line)
                    feature_idx = [self.feature_to_idx[feat] for feat in splits[:-1]
                                   if feat in self.feature_to_idx]
                    features_idx.append(feature_idx)
                    tags_idx.append(self.tag_to_idx[splits[-1]])

    @staticmethod
    def _create_label():
        if config.label_num == 2:
            B = B_single = "B"
            I_first = I = I_end = "I"
        elif config.label_num == 3:
            B = B_single = "B"
            I_first = I = "I"
            I_end = "I_end"
        elif config.label_num == 4:
            B = "B"
            B_single = "B_single"
            I_first = I = "I"
            I_end = "I_end"
        elif config.label_num == 5:
            B = "B"
            B_single = "B_single"
            I_first = "I_first"
            I = "I"
            I_end = "I_end"

        return B, B_single, I_first, I, I_end

    @staticmethod
    def _reverse_dict(dict_obj):
        return dict([(v, k) for k, v in dict_obj.items()])

    def convert_text_file_to_feature_file(
            self, text_file, conll_file, feature_file):
        # 从文本中，构建所有的特征和训练标注数据
        B, B_single, I_first, I, I_end = FeatureExtractor._create_label()

        with open(conll_file, "w", encoding="utf8") as c_writer, \
                open(feature_file, "w", encoding="utf8") as f_writer:
            for words in jio.read_file_by_iter(text_file):
                # 对文本进行归一化和整理
                if config.norm_text:
                    words = [self.pre_processor(word, convert_exception=True,
                                convert_num_letter=True, normalize_num_letter=False)
                                for word in words]

                example, tags = word2tag(words)

                c_writer.write(json.dumps([example, tags], ensure_ascii=False) + "\n")

                for idx, tag in enumerate(tags):
                    features = self.get_node_features(idx, example)
                    # 某些特征不存在，则将其转换为 `/` 特征
                    norm_features = [feature for feature in features if feature in self.feature_to_idx]
                    if len(norm_features) < len(features):
                        norm_features.append(self.empty_feature)

                    norm_features.append(tag)
                    f_writer.write(json.dumps(norm_features, ensure_ascii=False) + '\n')
                f_writer.write("\n")

    def save(self, model_dir=None):
        if model_dir is None:
            model_dir = config.model_dir

        data = dict()
        data["unigram"] = sorted(list(self.unigram))
        data["bigram"] = sorted(list(self.bigram))
        data["feature_to_idx"] = self.feature_to_idx
        data["tag_to_idx"] = self.tag_to_idx

        with open(os.path.join(model_dir, "features.json"), "w", encoding="utf8") as f_w:
            json.dump(data, f_w, ensure_ascii=False, indent=4, separators=(',', ':'))

    @classmethod
    def load(cls, model_dir=None):
        if model_dir is None:
            model_dir = config.model_dir

        extractor = cls.__new__(cls)
        extractor._create_features()

        feature_path = os.path.join(model_dir, "features.pkl")
        if os.path.exists(feature_path):
            with open(feature_path, "rb") as reader:
                data = pickle.load(reader)

            extractor.unigram = set(data["unigram"])
            extractor.bigram = set(data["bigram"])
            extractor.feature_to_idx = data["feature_to_idx"]
            extractor.tag_to_idx = data["tag_to_idx"]
            # extractor.idx_to_tag = extractor._reverse_dict(extractor.tag_to_idx)

            return extractor

        print("WARNING: features.pkl does not exist, try loading features.json", file=sys.stderr)
        feature_path = os.path.join(model_dir, "features.json")
        zip_feature_path = os.path.join(model_dir, "features.zip")

        if (not os.path.exists(feature_path)) and os.path.exists(zip_feature_path):
            logging.info('unzipping `{}` to `{}`.'.format(
                zip_feature_path, feature_path))
            unzip_file([zip_feature_path])

        if os.path.exists(feature_path):
            with open(feature_path, "r", encoding="utf8") as reader:
                data = json.load(reader)

            extractor.unigram = set(data["unigram"])
            extractor.bigram = set(data["bigram"])
            extractor.feature_to_idx = data["feature_to_idx"]
            extractor.tag_to_idx = data["tag_to_idx"]
            # extractor.idx_to_tag = extractor._reverse_dict(extractor.tag_to_idx)
            return extractor

        print("WARNING: features.json does not exist, try loading using old format", file=sys.stderr)
        with open(os.path.join(model_dir, "unigram_word.txt"), "r", encoding="utf8") as reader:
            extractor.unigram = set([line.strip() for line in reader])

        with open(os.path.join(model_dir, "bigram_word.txt"), "r", encoding="utf8") as reader:
            extractor.bigram = set(line.strip() for line in reader)

        extractor.feature_to_idx = dict()
        feature_base_name = os.path.join(model_dir, "featureIndex.txt")
        for i in range(10):
            with open("{}_{}".format(feature_base_name, i), "r", encoding="utf8") as reader:
                for line in reader:
                    feature, index = line.split(" ")
                    feature = ".".join(feature.split(".")[1:])
                    extractor.feature_to_idx[feature] = int(index)

        extractor.tag_to_idx = dict()
        with open(os.path.join(model_dir, "tagIndex.txt"), "r", encoding="utf8") as reader:
            for line in reader:
                tag, index = line.split(" ")
                extractor.tag_to_idx[tag] = int(index)

        return extractor
