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
import yaml
from collections import Counter

from jiojio.util import unzip_file
from jiojio import logging, TimeIt, zip_file, \
    read_file_by_iter, write_file_by_line

from jiojio.pre_processor import PreProcessor
from . import pos_get_node_features_c


class POSFeatureExtractor(object):

    def __init__(self, config):

        self.unigram = set()
        # self.bigram = set()
        self.single_pos_word = set()
        self.feature_to_idx = dict()
        self.tag_to_idx = dict()
        self.pre_processor = PreProcessor(
            convert_num_letter=config.convert_num_letter,
            normalize_num_letter=config.normalize_num_letter,
            convert_exception=config.convert_exception)

        self.config = config
        self._create_features()

        if pos_get_node_features_c is None:
            self.get_node_features_c = None
        else:
            self.get_node_features_c = True

        with open(config.pos_types_file, 'r', encoding='utf-8') as f:
            self.pos_types = yaml.load(f, Loader=yaml.SafeLoader)

    def _create_features(self):
        self.start_feature = '[START]'
        self.end_feature = '[END]'

        self.delim = '.'
        self.mark = '*'
        self.empty_feature = '/'
        self.default_feature = '$$'

        # 为了减少字符串个数，缩短匹配时间，根据前后字符的位置，制定规则进行缩短匹配，规则如下：
        # c 代表 char，后一个位置 char 用 d 表示，前一个用 b 表示，按字母表顺序完成。
        # w 代表 word，后一个位置 word 用 x 表示，前一个用 v 表示，按字母表顺序完成。
        # k 代表 unknown，
        seg = ''

        self.part_before_2_left = 'al' + seg  # 部分词特征
        self.part_before_2_right = 'ar' + seg  # 部分词特征

        self.part_before_left = 'bl' + seg  # 部分词特征
        self.part_before_right = 'br' + seg  # 部分词特征

        self.char_current_1 = 'c1' + seg  # c.1.
        self.char_current_2 = 'c2' + seg  # c.2.
        self.char_current_3 = 'c3' + seg  # c.3.
        self.char_current_4 = 'c4' + seg  # c.4.
        self.char_current_5 = 'c5' + seg  # c.-2.
        self.char_current_6 = 'c6' + seg  # c.-1.
        self.char_current_unk = 'ck'  # 词汇不足 4 字的补齐特征
        self.part_current_left = 'cl' + seg  # 部分词特征
        self.part_current_right = 'cr' + seg  # 部分词特征

        self.part_next_left = 'dl' + seg  # 部分词特征
        self.part_next_right = 'dr' + seg  # 部分词特征

        self.part_next_2_left = 'el' + seg  # 部分词特征
        self.part_next_2_right = 'er' + seg  # 部分词特征

        self.word_current_unknown = 'wk'
        self.word_before_unknown = 'vk'
        self.word_before_2_unknown = 'uk'
        self.word_next_unknown = 'xk'
        self.word_next_2_unknown = 'yk'

        self.word_before_2 = 'u' + seg  # w-2
        self.word_before = 'v' + seg  # w-1.
        self.word_current = 'w' + seg
        self.word_next = 'x' + seg  # w1.
        self.word_next_2 = 'y' + seg  # w2.

        # cur word, other word
        self.bi_word_before_2_word_current = 'uw' + seg  # w-2.w.
        self.bi_word_before_word_current = 'vw' + seg  # w-1.w.
        self.bi_word_current_word_next = 'wx' + seg  # w.w1.
        self.bi_word_current_word_next_2 = 'wy' + seg  # w.w2.

        # cur word, other left part
        self.bi_part_before_2_left_word_current = 'alw' + seg  # w-2.w.
        self.bi_part_before_left_word_current = 'blw' + seg  # w-1.w.
        self.bi_word_current_part_next_left = 'wdl' + seg  # w.w1.
        self.bi_word_current_part_next_2_left = 'wel' + seg  # w.w2.

        # cur word other right part
        self.bi_part_before_2_right_word_current = 'arw' + seg  # w-2.w.
        self.bi_part_before_right_word_current = 'brw' + seg  # w-1.w.
        self.bi_word_current_part_next_right = 'wdr' + seg  # w.w1.
        self.bi_word_current_part_next_2_right = 'wer' + seg  # w.w2.

        # cur left part, other word
        self.bi_word_before_2_part_current_left = 'ucl' + seg  # w-2.p.
        self.bi_word_before_part_current_left = 'vcl' + seg  # w-1.p.
        self.bi_part_current_left_word_next = 'clx' + seg  # p.w1.
        self.bi_part_current_left_word_next_2 = 'cly' + seg  # p.w2.

        # cur left part, other left part
        self.bi_part_before_2_left_part_current_left = 'alcl' + seg  # w-2.w.
        self.bi_part_before_left_part_current_left = 'blcl' + seg  # w-1.w.
        self.bi_part_current_left_part_next_left = 'cldl' + seg  # w.w1.
        self.bi_part_current_left_part_next_2_left = 'clel' + seg  # w.w2.

        # cur left part, other right part
        self.bi_part_before_2_right_part_current_left = 'arcl' + seg  # w-2.w.
        self.bi_part_before_right_part_current_left = 'brcl' + seg  # w-1.w.
        self.bi_part_current_left_part_next_right = 'cldr' + seg  # w.w1.
        self.bi_part_current_left_part_next_2_right = 'cler' + seg  # w.w2.

        # cur right part, other word
        self.bi_word_before_2_part_current_right = 'ucr' + seg  # w-2.p.
        self.bi_word_before_part_current_right = 'vcr' + seg  # w-1.p.
        self.bi_part_current_right_word_next = 'crx' + seg  # p.w1.
        self.bi_part_current_right_word_next_2 = 'cry' + seg  # p.w2.

        # cur right part, other left part
        self.bi_part_before_2_left_part_current_right = 'alcr' + seg  # w-2.w.
        self.bi_part_before_left_part_current_right = 'blcr' + seg  # w-1.w.
        self.bi_part_current_right_part_next_left = 'crdl' + seg  # w.w1.
        self.bi_part_current_right_part_next_2_left = 'crel' + seg  # w.w2.

        # cur right part, other right part
        self.bi_part_before_2_right_part_current_right = 'arcr' + seg  # w-2.w.
        self.bi_part_before_right_part_current_right = 'brcr' + seg  # w-1.w.
        self.bi_part_current_right_part_next_right = 'crdr' + seg  # w.w1.
        self.bi_part_current_right_part_next_2_right = 'crer' + seg  # w.w2.

        self.word_current_length = 'wl' + seg  # w.l.

    def _print_pos_tag_freq_info(self, pos_tag_num_info):
        """ 打印所有词性数量 """
        total_num = sum(list(pos_tag_num_info.values()))
        logging.info('\tpos\tname\ttoken num\tratio:')

        for pos_tag, pos_num in list(pos_tag_num_info.items()):
            logging.info('\t{}\t{:\u3000<8s} {: <10d} \t {:.2%}'.format(
                pos_tag, self.pos_types['model_type'][pos_tag], pos_num,
                pos_num / total_num))

        return total_num

    def cleansing_unigrams(self, unigrams):
        # 针对词性标注，unigram 的清洗策略有不同
        clean_unigrams = dict()
        total_freq = 0
        clean_freq = 0

        for word, freq in unigrams.most_common():
            total_freq += freq

            if freq < self.config.unigram_feature_trim:
                continue
            if len(word) > self.config.word_max:  # 超过 4 个字，往往已经脱离了词汇，到了句式层面
                continue
            if self.pre_processor.num_pattern.search(word):
                # if len(word) == 1:
                # print(word)
                # pdb.set_trace()
                continue
            if self.pre_processor.percent_num_pattern.search(word):
                continue
            if self.pre_processor.time_pattern.search(word):
                continue
            if ('@' in word or '，' in word or '·' in word or 'ん' in word or word == '') and len(word) != 1:
                # @ 往往为网名微博起始：@小狗狗爱吃糖
                # ， 往往为数字错误
                # · 往往为外文名：奥拉·新德丝
                continue

            if freq < 500:  # 低频中有大量不常见人名，须删除
                if self.pre_processor.check_chinese_name(word):
                    # print(word)
                    # pdb.set_trace()
                    continue

            clean_unigrams.update({word: freq})
            clean_freq += freq

        return clean_unigrams, clean_freq / total_freq

    def build(self, train_file):

        pos_tag_num_info = Counter()  # 计算所有词性的出现频次
        unigrams = Counter()  # 计算所有词汇的频次
        chars = Counter()  # 计算所有字的频次
        parts = Counter()  # 计算词汇的部分特征，包括前半部分，后半部分
        word_pos_freq = dict()  # 计算所有词汇的各个词性与频次
        first_char_pos_freq = dict()  # 计算所有词汇的首字符的各个词性与频次

        # 第一次遍历统计获取 self.unigram 与所有词长 word_length_info
        for sample_idx, samples in enumerate(read_file_by_iter(train_file)):

            if sample_idx % 1e6 == 0:
                logging.info(sample_idx)

            words, pos_tags = samples
            pos_tag_num_info.update(pos_tags)

            # 对文本进行归一化和整理
            if self.config.norm_text:
                words = [self.pre_processor(word) for word in words]

            unigrams.update(words)
            for word in words:
                chars.update(word)
                for i in range(1, len(word)):
                    parts.update([word[:i], word[i:]])

            # 统计词的各个词性及频次
            for word, pos_tag in zip(words, pos_tags):
                if word in word_pos_freq:
                    if pos_tag in word_pos_freq[word]:
                        word_pos_freq[word][pos_tag] += 1
                    else:
                        word_pos_freq[word].update({pos_tag: 1})
                else:
                    word_pos_freq.update({word: {pos_tag: 1}})

        # 打印全语料不同长度词 token 的计数和比例
        total_tag_num = self._print_pos_tag_freq_info(pos_tag_num_info)

        # 对 self.unigram 的清理和整理
        logging.info('orig unigram num: {}'.format(len(unigrams)))
        clean_unigrams, unigram_ratio = self.cleansing_unigrams(unigrams)
        logging.info('{:.2%} unigram are saved.'.format(unigram_ratio))

        # 若 self.unigram 中的频次都不足 unigram_feature_trim 则，在后续特征删除时必然频次不足
        self.unigram = set([unigram for unigram, freq in clean_unigrams.items()])
        logging.info('true unigram num: {}'.format(len(self.unigram)))
        # 再获取脏 unigram 并存入文件中
        all_unigram = set([unigram for unigram, freq in unigrams.items()])
        dirty_unigram = list(all_unigram - self.unigram)
        write_file_by_line(dirty_unigram, os.path.join(self.config.train_dir, 'dirty_unigram.json'))

        logging.info('orig chars num: {}'.format(len(chars)))
        total_num = sum([freq for _, freq in chars.items()])
        saved_num = sum([freq for _, freq in chars.items() if freq >= self.config.char_feature_trim])
        logging.info('{:.2%} chars are saved.'.format(saved_num / total_num))
        # 对字符特征的构建
        self.char = set([char for char, freq in chars.items()
                         if freq >= self.config.char_feature_trim])
        logging.info('true chars num: {}'.format(len(self.char)))

        # 对仅具有唯一词性的词汇的特征构建
        single_pos_word = list()
        single_tag_num = 0
        for word, pos_dict in word_pos_freq.items():
            pos_val_list = list(pos_dict.values())
            # 近似 100% 的单词性词汇
            if (max(pos_val_list) / sum(pos_val_list) > 0.995) and sum(pos_dict.values()) >= self.config.unigram_feature_trim:
            # 绝对 100% 的单词性词汇
            # if len(pos_dict) == 1 and sum(pos_dict.values()) >= self.config.unigram_feature_trim:
                if word in self.unigram and 'nr' not in pos_dict:
                    # 刨除一些人名等信息进入词典以及一些特殊的词性，如人名
                    single_pos_word.append(word)
                    # print(word, pos_dict.keys())
                    single_tag_num += sum(pos_dict.values())

        self.single_pos_word = set(single_pos_word)
        logging.info('single pos word num: {}, ratio: {:.2%}.'.format(
            len(self.single_pos_word), single_tag_num / total_tag_num))

        # 统计词缀的特征
        # 词缀主要用于处理人名、地名、机构名等信息，例如“霍元甲”、“山岗村”、“组织部” 等
        # 对 parts 的清洗，其中有大量的数字特征和重复特征需要进行清除
        # 数字特征：
        #    “34.4”、“9.02” 等等。这种特征无意义
        # 重复特征：
        #    “条不紊”、“不紊”，可以观察，“有条不紊” 是无效的
        # 过长的特征：
        #    “study_few” 等，该种特征无效
        parts = [(part, freq) for part, freq in parts.items()
                 if len(part) <= self.config.part_length_trim
                 and not self.pre_processor.pure_num_pattern.search(part)]
                 # 去除纯数字特征，留下字母特征，但对字母特征的频次有要求

        parts = sorted(parts, key=lambda item: len(item[0]), reverse=True)

        chinese_parts = [item[0] for item in parts if self.pre_processor.check_chinese_char(item[0])
                         and item[1] >= self.config.part_feature_trim]
        non_chinese_parts = [item[0] for item in parts if not self.pre_processor.check_chinese_char(item[0])
                             and item[1] >= self.config.part_feature_non_chinese_trim]
        parts = chinese_parts + non_chinese_parts
        logging.info('parts num: {}'.format(len(parts)))
        # pdb.set_trace()
        self.part = set(parts)

        # 统计有多少词不在 unigram 中，且无 parts 特征
        unk_num = 0
        char_num = 0
        unk_words_list = list()
        for sample_idx, samples in enumerate(read_file_by_iter(train_file)):

            if sample_idx % 1e6 == 0:
                logging.info(sample_idx)

            words, pos_tags = samples
            for word in words:
                if word in self.unigram:
                    continue

                has_parts = False
                for i in range(1, len(word)):
                    if word[:i] in self.part or word[i:] in self.part:
                        has_parts = True
                        break
                if has_parts:
                    continue

                if word in self.char:
                    char_num += 1  # 这些 char 将会反补至 self.unigram 此处仅作统计

                unk_words_list.append(word)
                unk_num += 1

        unk_words = set(unk_words_list)
        # 大约 1% ~ 2% 的词不具有任何特征，这就导致了必须采用字符特征来完善其特征定义。
        # 此时其中绝大部分都为数字，则可以进行字符的处理。
        logging.info('unk_word num: {}, ratio: {:.2%}'.format(
            unk_num - char_num, (unk_num - char_num) / total_tag_num))
        # 此类 unknown 的词汇，其中有一半都属于单字特征，即，应当将该字添加入 self.unigram 中
        # 此前原因是其总词频导致了其不在 self.unigram 其中
        # 经过从 char 向 self.unigram 的反添加，剩余的 unknown 词汇大约仅占 0.6% ~ 0.7%
        # 此步即不再管理剩余的异常词汇特征，如若再有常见的词汇位于此类 unk_words 中，说明参数
        # trim 存在失误，或语料实在缺乏某些常见词汇。
        common_chars = unk_words & self.char
        self.unigram = self.unigram | common_chars  # 反哺一些 chars 进入 self.unigram 中
        unk_words = unk_words - self.char
        unk_words_path = os.path.join(self.config.model_dir, 'unk_words.json')

        with open(unk_words_path, 'w', encoding='utf-8') as fw:
            json.dump(list(unk_words), fw, ensure_ascii=False, indent=4, separators=(',', ':'))

        # 对词长特征的构建 - 此种特征实际上对词性没有独特性区分性，会将大量的2字词错误化为 名词 或 动词

        # 3、统计具有歧义词性的词汇，即某个词汇具有两种或多种词性
        #   1、“信息通信”：“通信”为名词
        #   2、“信息通信时，传输的是比特”：“通信”为动词
        #   3、“在学习信息通信时，要结合实验才能融会贯通”：“通信”为名词
        #   4、“信息通信技术”：“通信”为名动词
        # 从该例可以看出，仅仅抓取两个连续词汇作为 bigram 特征稍显不足，因为词性的表达非常复杂。
        # 此种类型根据后续进度进行调整

        # 第三次循环获取样本所有特征
        feature_freq = Counter()  # 计算各个特征的出现次数，减少罕见特征计数
        for sample_idx, samples in enumerate(read_file_by_iter(train_file)):

            words, pos_tags = samples

            if sample_idx % 1e6 == 0:
                logging.info(sample_idx)

            # 对文本进行归一化和整理
            if self.config.norm_text:
                words = [self.pre_processor(word) for word in words]

            # second pass to get features
            if self.get_node_features_c is None:
                for idx in range(len(words)):
                    node_features = self.get_node_features(idx, words)
                    feature_freq.update(feature for feature in node_features)
                    # tmp = [item for item in node_features if 'c1' in item]
                    # if len(tmp) > 0:
                    #     print('    '.join(words[max(0, idx - 2): min(idx + 3, len(words))]))
                    #     print(node_features)
                    #     pdb.set_trace()
            else:
                words_length = len(words)
                # print(example)
                for idx in range(words_length):
                    node_features = get_node_features_c(
                        idx, words, words_length, self.unigram, self.char)
                    feature_freq.update(feature for feature in node_features)

                print(example)
                print(sample_idx, '\n')

        logging.info('# orig feature num: {}'.format(len(feature_freq)))
        # pdb.set_trace()
        feature_set = list()
        feature_count_sum = 0
        for feature, freq in feature_freq.most_common():

            if freq >= self.config.feature_trim:
                feature_set.append(feature)
                feature_count_sum += freq

        logging.info('# {:.2%} features are saved.'.format(
            feature_count_sum / sum(list(feature_freq.values()))))

        self.feature_to_idx = {feature: idx for idx, feature in enumerate(feature_set, 0)}
        logging.info('# true feature_num: {}'.format(len(self.feature_to_idx)))

        # create tag map
        # pos_tag_set = self._create_label()
        pos_tag_dict = dict(pos_tag_num_info)
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(sorted(pos_tag_dict))}
        print(self.tag_to_idx)
        self.idx_to_tag = dict([(v, k) for k, v in self.tag_to_idx.items()])
        # pdb.set_trace()

    def get_node_features(self, idx, token_list):
        # 给定一个 token_list，找出其中 token_list[idx] 匹配到的所有特征
        cur_w = token_list[idx]
        feature_list = list()

        # 某些特殊的词汇，仅仅具有唯一一种词性，此时仅仅添加 “w我” 即可。不需要其它特征。
        # 此时，我们需要将此类词汇统计出来。词频大于10，且具有唯一的词性的词汇
        # self.single_pos
        # 此外，某些字也代表着唯一词性，例如 “崔” 几乎仅仅作为中文姓氏出现，再无别义，也可以采取此方式处理。
        if cur_w in self.single_pos_word:
            feature_list.append(self.word_current + cur_w)
            return feature_list

        length = len(token_list)

        # before_2_words = None
        # before_2_lefts = list()
        # before_2_rights = list()

        before_word = None
        before_lefts = list()
        before_rights = list()

        current_word = None
        current_lefts = list()
        current_rights = list()

        next_word = None
        next_lefts = list()
        next_rights = list()

        # next_2_words = None
        # next_2_lefts = list()
        # next_2_rights = list()

        # 一、添加词长特征，例如，中文人名常为 2、3 个字，日本人名常为 4 个字，动词常为 1、2 字
        # 俗语长度常为 4 个字等，此种特征被淘汰，不适用于词性标注，会将大量的二字词分为名词或动词
        cur_w_length = len(cur_w)

        '''
        if idx > 1:
            before_w2 = token_list[idx - 2]
            if before_w2 in self.unigram:
                # 前第二词特征
                feature_list.append(self.word_before_2 + before_w2)
            else:
                # 前第二词 parts 特征
                # before_w2_length = len(before_w2)
                has_part = False
                for i in range(1, len(before_w2)):
                    if before_w2[:i] in self.part:
                        feature_list.append(self.part_before_2_left + before_w2[:i])
                        has_part = True

                    if before_w2[i:] in self.part:
                        feature_list.append(self.part_before_2_right + before_w2[i:])
                        has_part = True

                if not has_part:
                    feature_list.append(self.word_before_2_unknown)
        '''
        if idx > 0:
            before_w = token_list[idx - 1]
            # 前一个词 特征
            if before_w in self.unigram:
                feature_list.append(self.word_before + before_w)
                before_word = before_w
            else:
                # 前第二词 parts 特征
                has_part = False
                for i in range(1, len(before_w)):
                    left_tmp = before_w[:i]
                    if left_tmp in self.part:
                        feature_list.append(self.part_before_left + left_tmp)
                        before_lefts.append(left_tmp)
                        has_part = True

                    right_tmp = before_w[i:]
                    if right_tmp in self.part:
                        feature_list.append(self.part_before_right + right_tmp)
                        before_rights.append(right_tmp)
                        has_part = True

                if not has_part:
                    feature_list.append(self.word_before_unknown)

        else:
            # 词汇为起始位特征
            feature_list.append(self.start_feature)

        # 由于词汇量相当大，因此必须考虑是否在 unigram 中
        if cur_w in self.unigram:
            # 当前词特征
            feature_list.append(self.word_current + cur_w)
            current_word = cur_w
        else:
            # 当前词 parts 特征
            has_part = False
            for i in range(1, len(cur_w)):
                left_tmp = cur_w[:i]
                if left_tmp in self.part:
                    feature_list.append(self.part_current_left + left_tmp)
                    current_lefts.append(left_tmp)
                    has_part = True

                right_tmp = cur_w[i:]
                if right_tmp in self.part:
                    feature_list.append(self.part_current_right + right_tmp)
                    current_rights.append(right_tmp)
                    has_part = True

            if not has_part:
                # 当前词 unk 特征
                feature_list.append(self.word_current_unknown)
                # 为该词添加字特征

                if cur_w_length < 4:
                    # 添加正向字符特征
                    for i in range(cur_w_length):
                        if i == 0:
                            feature_list.append(self.char_current_1 + cur_w[0])
                        elif i == 1:
                            feature_list.append(self.char_current_2 + cur_w[1])
                        elif i == 2:
                            feature_list.append(self.char_current_3 + cur_w[2])
                    for i in range(4 - cur_w_length):
                        feature_list.append(self.char_current_unk)

                    # 添加反向字符特征
                    if cur_w_length == 1:
                        feature_list.append(self.char_current_unk)
                        feature_list.append(self.char_current_6 + cur_w[-1])
                    else:
                        feature_list.append(self.char_current_5 + cur_w[-2])
                        feature_list.append(self.char_current_6 + cur_w[-1])
                else:
                    feature_list.append(self.char_current_1 + cur_w[0])
                    feature_list.append(self.char_current_2 + cur_w[1])
                    feature_list.append(self.char_current_3 + cur_w[2])
                    feature_list.append(self.char_current_4 + cur_w[3])
                    feature_list.append(self.char_current_5 + cur_w[-2])
                    feature_list.append(self.char_current_6 + cur_w[-1])

        if idx < length - 1:
            next_w = token_list[idx + 1]
            # 后一个词特征
            if next_w in self.unigram:
                feature_list.append(self.word_next + next_w)
                next_word = next_w
            else:
                # 后第一词 parts 特征
                has_part = False
                for i in range(1, len(next_w)):
                    left_tmp = next_w[:i]
                    if left_tmp in self.part:
                        feature_list.append(self.part_next_left + left_tmp)
                        next_lefts.append(left_tmp)
                        has_part = True

                    right_tmp = next_w[i:]
                    if right_tmp in self.part:
                        feature_list.append(self.part_next_right + right_tmp)
                        next_rights.append(right_tmp)
                        has_part = True

                if not has_part:
                    feature_list.append(self.word_next_unknown)
        else:
            # 字符为终止位特征
            feature_list.append(self.end_feature)

        '''
        if idx < length - 2:
            next_w2 = token_list[idx + 2]

            if next_w2 in self.unigram:
                # 后第二词特征
                feature_list.append(self.word_next_2 + next_w2)
            else:
                # 前二词 unk 特征
                has_part = False
                for i in range(1, len(next_w2)):
                    if next_w2[:i] in self.part:
                        feature_list.append(self.part_next_2_left + next_w2[:i])
                        has_part = True

                    if next_w2[i:] in self.part:
                        feature_list.append(self.part_next_2_right + next_w2[i:])
                        has_part = True

                if not has_part:
                    feature_list.append(self.word_next_2_unknown)
        '''

        # bigram 特征
        if current_word is not None:
            if before_word is not None:
                feature_list.append(
                    self.bi_word_before_word_current + before_word + self.mark + current_word)
            else:
                for before_left in before_lefts:
                    feature_list.append(
                        self.bi_part_before_left_word_current + before_left + self.mark + current_word)

                for before_right in before_rights:
                    feature_list.append(
                        self.bi_part_before_right_word_current + before_right + self.mark + current_word)

            if next_word is not None:
                feature_list.append(
                    self.bi_word_current_word_next + current_word + self.mark + next_word)
            else:
                for next_left in next_lefts:
                    feature_list.append(
                        self.bi_word_current_part_next_left + current_word + self.mark + next_left)

                for next_right in next_rights:
                    feature_list.append(
                        self.bi_word_current_part_next_right + current_word + self.mark + next_right)

        else:
            # 有 current parts left and right 特征
            if before_word is not None:
                for current_left in current_lefts:
                    feature_list.append(
                        self.bi_word_before_part_current_left + before_word + self.mark + current_left)

                for current_right in current_rights:
                    feature_list.append(
                        self.bi_word_before_part_current_right + before_word + self.mark + current_right)
            else:
                for before_left in before_lefts:
                    for current_left in current_lefts:
                        feature_list.append(
                            self.bi_part_before_left_part_current_left + before_left + self.mark + current_left)
                    for current_right in current_rights:
                        feature_list.append(
                            self.bi_part_before_left_part_current_right + before_left + self.mark + current_right)

                for before_right in before_rights:
                    for current_left in current_lefts:
                        feature_list.append(
                            self.bi_part_before_right_part_current_left + before_right + self.mark + current_left)
                    for current_right in current_rights:
                        feature_list.append(
                            self.bi_part_before_right_part_current_right + before_right + self.mark + current_right)

            if next_word is not None:
                for current_left in current_lefts:
                    feature_list.append(
                        self.bi_part_current_left_word_next + current_left + self.mark + next_word)

                for current_right in current_rights:
                    feature_list.append(
                        self.bi_part_current_right_word_next + current_right + self.mark + next_word)
            else:
                for next_left in next_lefts:
                    for current_left in current_lefts:
                        feature_list.append(
                            self.bi_part_current_left_part_next_left + current_left + self.mark + next_left)
                    for current_right in current_rights:
                        feature_list.append(
                            self.bi_part_current_right_part_next_left + current_right + self.mark + next_left)

                for next_right in next_rights:
                    for current_left in current_lefts:
                        feature_list.append(
                            self.bi_part_current_left_part_next_right + current_left + self.mark + next_right)
                    for current_right in current_rights:
                        feature_list.append(
                            self.bi_part_current_right_part_next_right + current_right + self.mark + next_right)

        return feature_list

    def _create_label(self):
        tag_list = list(self.pos_types['model_type'].keys())

        return list(set(tag_list))

    def convert_text_file_to_feature_idx_file(
            self, text_file, feature_idx_file, gold_idx_file):
        # 从文本中，构建所有的特征和训练标注数据

        with open(feature_idx_file, 'w', encoding='utf-8') as f_writer, \
                open(gold_idx_file, 'w', encoding='utf-8') as g_writer:

            f_writer.write('{}\n\n'.format(len(self.feature_to_idx)))
            g_writer.write('{}\n\n'.format(len(self.tag_to_idx)))

            for samples in read_file_by_iter(text_file):

                words, pos_tags = samples

                # 对文本进行归一化和整理
                if self.config.norm_text:
                    words = [self.pre_processor(word) for word in words]

                tags_idx = list()

                for idx, tag in enumerate(pos_tags):
                    features = self.get_node_features(idx, words)
                    feature_idx = [self.feature_to_idx[feature] for feature in features
                                   if feature in self.feature_to_idx]

                    tags_idx.append(self.tag_to_idx[tag])

                    f_writer.write(",".join(map(str, feature_idx)))
                    f_writer.write("\n")

                f_writer.write("\n")
                g_writer.write(",".join(map(str, tags_idx)))
                g_writer.write('\n')

    def save(self, model_dir=None):
        if model_dir is None:
            model_dir = self.config.model_dir

        data = dict()
        data['unigram'] = sorted(list(self.unigram))
        data['char'] = sorted(list(self.char))
        data['part'] = sorted(list(self.part))
        data['single_pos_word'] = sorted(list(self.single_pos_word))
        data['feature_to_idx'] = list(self.feature_to_idx.keys())
        data['tag_to_idx'] = self.tag_to_idx

        feature_path = os.path.join(model_dir, 'features.json')

        with open(feature_path, 'w', encoding='utf-8') as f_w:
            json.dump(data, f_w, ensure_ascii=False, indent=4, separators=(',', ':'))

        zip_file(feature_path)

    @classmethod
    def load(cls, config, model_dir=None):
        extractor = cls.__new__(cls)
        extractor.config = config
        extractor._create_features()

        feature_path = os.path.join(model_dir, 'features.json')
        zip_feature_path = os.path.join(model_dir, 'features.zip')

        if (not os.path.exists(feature_path)) and os.path.exists(zip_feature_path):
            logging.info('\n\tunzip `{}`\n\tto `{}`.'.format(zip_feature_path, feature_path))
            unzip_file(zip_feature_path)

        if os.path.exists(feature_path):
            with open(feature_path, 'r', encoding='utf8') as reader:
                data = json.load(reader)

            extractor.unigram = set(data['unigram'])
            extractor.char = set(data['char'])
            extractor.part = set(data['part'])
            extractor.single_pos_word = set(data['single_pos_word'])
            feature_list = data['feature_to_idx']
            extractor.feature_to_idx = dict(
                [(feature, idx) for idx, feature in enumerate(feature_list)])
            extractor.tag_to_idx = data['tag_to_idx']

            # extractor.idx_to_tag = extractor._reverse_dict(extractor.tag_to_idx)
            return extractor

        raise FileNotFoundError('`features.json` does not exist.')
