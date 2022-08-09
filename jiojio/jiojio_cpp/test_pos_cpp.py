# -*- coding=utf-8 -*-

import os
import pdb
import sys
import json
import time
import ctypes

import jiojio
import jionlp as jio

from jiojio.pos import ReadPOSDictionary

word_pos_default_dict = ReadPOSDictionary(jiojio.pos_config).word_pos_dict

class POSFeatureExtractor(object):

    def __init__(self, char, part, unigram):

        self.unigram = unigram
        # self.bigram = set()
        self.char = char
        self.part = part
        self.feature_to_idx = dict()

        self._create_features()

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

    def get_node_features(self, idx, token_list):
        # 给定一个 token_list，找出其中 token_list[idx] 匹配到的所有特征
        cur_w = token_list[idx]
        feature_list = []

        # flag = False
        length = len(token_list)

        before_word = None
        before_lefts = None
        before_rights = None

        current_word = None
        current_lefts = None
        current_rights = None

        next_word = None
        next_lefts = None
        next_rights = None

        cur_w_length = len(cur_w)

        if idx > 0:
            before_w = token_list[idx - 1]
            # 前一个词 特征
            if before_w in self.unigram:
                feature_list.append(self.word_before + before_w)
                before_word = before_w
            else:
                # 前第二词 parts 特征
                before_w_pure_length = len(before_w)
                before_w_length = min(before_w_pure_length, 5)
                has_left_part = False
                for i in range(before_w_length-1, 0, -1):
                    left_tmp = before_w[:i]
                    # print(i, left_tmp)
                    if left_tmp in self.part:
                        feature_list.append(self.part_before_left + left_tmp)
                        before_lefts = left_tmp
                        has_left_part = True
                        break

                has_right_part = False
                for i in range(before_w_pure_length - before_w_length + 1, before_w_pure_length):
                # for i in range(1, before_w_length):
                    right_tmp = before_w[i:]
                    if right_tmp in self.part:
                        feature_list.append(self.part_before_right + right_tmp)
                        before_rights = right_tmp
                        has_right_part = True
                        break

                if (not has_left_part) and (not has_right_part):
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
            has_left_part = False
            cur_w_trim_length = min(cur_w_length, 5)
            for i in range(cur_w_trim_length - 1, 0, -1):
                left_tmp = cur_w[:i]
                if left_tmp in self.part:
                    feature_list.append(self.part_current_left + left_tmp)
                    current_lefts = left_tmp
                    has_left_part = True
                    break

            has_right_part = False
            for i in range(cur_w_length - cur_w_trim_length + 1, cur_w_length):
                right_tmp = cur_w[i:]

                if right_tmp in self.part:
                    feature_list.append(self.part_current_right + right_tmp)
                    current_rights = right_tmp
                    has_right_part = True
                    break

            if (not has_left_part) and (not has_right_part):
                # 当前词 unk 特征
                feature_list.append(self.word_current_unknown)
                # 为该词添加字特征

                if cur_w_length == 1:
                    feature_list.append(self.char_current_unk)
                    if cur_w in self.char:
                        feature_list.append(self.char_current_1 + cur_w)

                elif cur_w_length == 2:
                    feature_list.append(self.char_current_unk)
                    if cur_w[0] in self.char:
                        feature_list.append(self.char_current_1 + cur_w[0])
                    if cur_w[1] in self.char:
                        feature_list.append(self.char_current_2 + cur_w[1])

                elif cur_w_length == 3:
                    feature_list.append(self.char_current_unk)
                    if cur_w[0] in self.char:
                        feature_list.append(self.char_current_1 + cur_w[0])
                    if cur_w[1] in self.char:
                        feature_list.append(self.char_current_2 + cur_w[1])
                    if cur_w[2] in self.char:
                        feature_list.append(self.char_current_3 + cur_w[2])

                elif cur_w_length >= 4:
                    if cur_w[0] in self.char:
                        feature_list.append(self.char_current_1 + cur_w[0])
                    if cur_w[1] in self.char:
                        feature_list.append(self.char_current_2 + cur_w[1])
                    if cur_w[2] in self.char:
                        feature_list.append(self.char_current_3 + cur_w[2])
                    if cur_w[3] in self.char:
                        feature_list.append(self.char_current_4 + cur_w[3])

                    if cur_w_length == 5:
                        if cur_w[-1] in self.char:
                            feature_list.append(self.char_current_6 + cur_w[-1])
                    elif cur_w_length > 5:
                        if cur_w[-1] in self.char:
                            feature_list.append(self.char_current_6 + cur_w[-1])
                        if cur_w[-2] in self.char:
                            feature_list.append(self.char_current_5 + cur_w[-2])

        if idx < length - 1:
            next_w = token_list[idx + 1]
            # 后一个词特征
            if next_w in self.unigram:
                feature_list.append(self.word_next + next_w)
                next_word = next_w
            else:
                # 后第一词 parts 特征
                has_left_part = False
                next_w_pure_length = len(next_w)
                next_w_length = min(next_w_pure_length, 5)
                for i in range(next_w_length-1, 0, -1):
                    left_tmp = next_w[:i]
                    if left_tmp in self.part:
                        feature_list.append(self.part_next_left + left_tmp)
                        next_lefts = left_tmp
                        has_left_part = True
                        break

                has_right_part = False
                for i in range(next_w_pure_length - next_w_length + 1, next_w_pure_length):
                    right_tmp = next_w[i:]
                    # print(i, right_tmp, next_w, right_tmp in self.part)
                    if right_tmp in self.part:

                        feature_list.append(self.part_next_right + right_tmp)
                        next_rights = right_tmp
                        has_right_part = True
                        break

                if (not has_left_part) and (not has_right_part):
                    feature_list.append(self.word_next_unknown)
        else:
            # 字符为终止位特征
            feature_list.append(self.end_feature)

        # bigram 特征
        if current_word is not None:
            if before_word is not None:
                feature_list.append(
                    self.bi_word_before_word_current + before_word + self.mark + current_word)
            else:
                if before_lefts is not None:
                    feature_list.append(
                        self.bi_part_before_left_word_current + before_lefts + self.mark + current_word)

                if before_rights is not None:
                    feature_list.append(
                        self.bi_part_before_right_word_current + before_rights + self.mark + current_word)

            if next_word is not None:
                feature_list.append(
                    self.bi_word_current_word_next + current_word + self.mark + next_word)
            else:
                if next_lefts is not None:
                    feature_list.append(
                        self.bi_word_current_part_next_left + current_word + self.mark + next_lefts)

                if next_rights is not None:
                    feature_list.append(
                        self.bi_word_current_part_next_right + current_word + self.mark + next_rights)

        else:
            # 有 current parts left and right 特征
            if before_word is not None:
                if current_lefts is not None:
                    feature_list.append(
                        self.bi_word_before_part_current_left + before_word + self.mark + current_lefts)

                if current_rights is not None:
                    feature_list.append(
                        self.bi_word_before_part_current_right + before_word + self.mark + current_rights)
            else:
                if before_lefts is not None:
                    if current_lefts is not None:
                        feature_list.append(
                            self.bi_part_before_left_part_current_left + before_lefts + self.mark + current_lefts)
                    if current_rights is not None:
                        feature_list.append(
                            self.bi_part_before_left_part_current_right + before_lefts + self.mark + current_rights)

                if before_rights is not None:
                    if current_lefts is not None:
                        feature_list.append(
                            self.bi_part_before_right_part_current_left + before_rights + self.mark + current_lefts)
                    if current_rights is not None:
                        feature_list.append(
                            self.bi_part_before_right_part_current_right + before_rights + self.mark + current_rights)

            if next_word is not None:
                if current_lefts is not None:
                    feature_list.append(
                        self.bi_part_current_left_word_next + current_lefts + self.mark + next_word)

                if current_rights is not None:
                    feature_list.append(
                        self.bi_part_current_right_word_next + current_rights + self.mark + next_word)
            else:
                if next_lefts is not None:
                    if current_lefts is not None:
                        feature_list.append(
                            self.bi_part_current_left_part_next_left + current_lefts + self.mark + next_lefts)
                    if current_rights is not None:
                        feature_list.append(
                            self.bi_part_current_right_part_next_left + current_rights + self.mark + next_lefts)

                if next_rights is not None:
                    if current_lefts is not None:
                        feature_list.append(
                            self.bi_part_current_left_part_next_right + current_lefts + self.mark + next_rights)
                    if current_rights is not None:
                        feature_list.append(
                            self.bi_part_current_right_part_next_right + current_rights + self.mark + next_rights)

        return feature_list


dir_path = '/home/ubuntu/github/jiojio/jiojio/jiojio_cpp'
feature_extractor = ctypes.PyDLL(
    os.path.join(dir_path, 'build', 'libposFeatureExtractor.so'))
get_pos_node_feature_c = feature_extractor.getPosNodeFeature
get_pos_node_feature_c.argtypes = [
    ctypes.c_int, ctypes.py_object, # ctypes.c_int,
    ctypes.py_object,
    ctypes.py_object, ctypes.py_object]
get_pos_node_feature_c.restype = ctypes.py_object


word_list = ['今天', '圣诞节', '中华人民共和国', '的', '天气', '真的', '挺好', '。']
unigram = set(["天气", "今天", "中国", "美国", "总统", "总统府"])
char = set("abcdefghijklmnopqrstuvwxyz0123456789.:-_")
part = set(["节", "共和国", "型"])


feature_path = '/home/ubuntu/github/jiojio/jiojio/models/default_pos_model/features.json'
with open(feature_path, 'r', encoding='utf8') as reader:
    data = json.load(reader)

unigram = set(data['unigram'])
char = set(data['char'])
part = set(data['part'])

# pdb.set_trace()
# word_list = ['重大', '革命', '历史', '题材', '是', '指', '一八四０', '年', '鸦片战争', '以来', '，', '特别', '是', '一九二一', '年', '中国共产党', '成立', '以来', '的', '重大', '\\', 'par', '革命', '斗争', '的', '内容', '；', '重大', '现实', '题材', '是', '指', '新中国', '成立', '后', '社会主义革命', '、', '建设', '和', '改革', '中的', '重大', '事件', '。']
word_list = ['代表', '们', '对', '去年', '春天', '，', '他', '在', '广交会', '期间', '，', '湖南华湘公司', '在', '广州东方宾馆', '请客', '，', '他', '和', '女儿', '、', '女婿', '等', '赴宴', '，', '一', '席', '吃掉', '人民币', '４４３７．１', '元', '的', '问题', '；', '对', '他', '作为', '全省', '清理', '整顿', '公司', '领导', '小组', '组长', '，', '对', '公司', '的', '清理', '没有', '什么', '显著', '成效', '的', '问题', '，', '代表', '们', '也', '提出', '了', '尖锐', '批评', '。']
print(word_list)
index = 28

times = 100000

word_lists = jio.read_file_by_line('/home/ubuntu/datasets/train_cws.txt1')
# pdb.set_trace()

with jio.TimeIt('empty:', ) as ti:
    word_lists_length = len(word_lists)
    for i in range(word_lists_length):
        _word_list = word_lists[i]
        word_list_length = len(_word_list)
        for idx in range(word_list_length):
            if _word_list[idx] not in word_pos_default_dict:
                pass
    pure_cost_time = ti.break_point()

fe_obj = POSFeatureExtractor(char, part, unigram)
res = fe_obj.get_node_features(28, word_list)

# print('start: ', sys.gettotalrefcount())
pdb.set_trace()

with jio.TimeIt('py cost:', ) as ti:
    word_lists_length = len(word_lists)
    kk = 0
    while kk < 50:
        for i in range(word_lists_length):
            _word_list = word_lists[i]
            word_list_length = len(_word_list)
            # index = 1
            # pdb.set_trace()
            # print(i)
            for idx in range(word_list_length):
                # print(idx)
                if _word_list[idx] not in word_pos_default_dict:
                    res_py = fe_obj.get_node_features(idx, _word_list)
        kk += 1
    py_cost_time = ti.break_point()


with jio.TimeIt('C  cost:', ) as ti:
    word_lists_length = len(word_lists)
    kk = 0
    while kk < 50:
        for i in range(word_lists_length):
            _word_list = word_lists[i]
            word_list_length = len(_word_list)
            for idx in range(word_list_length):
                # res_py = fe_obj.get_node_features(idx, _word_list)
                if _word_list[idx] not in word_pos_default_dict:
                    res_c = get_pos_node_feature_c(
                        idx, _word_list, part, unigram, char)
                    # print(res_c)
                # if res_c != res_py:
                #     print(idx)
                #     print('res_C:  ', res_c)
                #     print('res_py: ', res_py)
                #     pdb.set_trace()
        kk += 1
    C_cost_time = ti.break_point()

print('C  cost: ', C_cost_time - pure_cost_time)
print('py cost: ', py_cost_time - pure_cost_time)
print('ratio: ', (C_cost_time - pure_cost_time) / (py_cost_time - pure_cost_time))
pdb.set_trace()


sys.exit()
# pdb.set_trace()
index = 1
# print('before:', sys.gettotalrefcount())
# with jio.TimeIt('c ', no_print=True) as ti:

for i in range(times):
    # index = i % len(word_list)

    res = get_pos_node_feature_c(
        index, word_list, single_pos_word, part, unigram, char)
    print(res)
    pdb.set_trace()
# print('after: ', sys.gettotalrefcount())
# print(res)
pdb.set_trace()
#     c_cost_time = ti.break_point()

'''
163M 47868 18356
164M 48464 18824
596 字节

163M 47900 18392
164M 48468 18860

163M 47736 18228
163M 47736 18228

163M 47572 18060
164M 48132 18520


'''

print(res)
pdb.set_trace()
fe_obj = POSFeatureExtractor(single_pos_word, part, unigram)
with jio.TimeIt('py ', no_print=True) as ti:
    for i in range(times):
        index = i % len(word_list)
        res1 = fe_obj.get_node_features(index, word_list)
    py_cost_time = ti.break_point()


print('c _time: ', c_cost_time - pure_cost_time)
print('py_time: ', py_cost_time - pure_cost_time)
print('faster: ', (py_cost_time - c_cost_time) / (c_cost_time - pure_cost_time))
print("if the same: ", res == res1)
print(index)
print(res)
print(res1)



# 100000 次计算耗时对比:
# c: 0.920   py: 15.572
# c: 0.950   py: 15.786
# c: 0.909   py: 15.557
# 修改一条 else 逻辑部分
# c: 0.787   py: 11.118
# c: 0.773   py: 11.722
# c: 0.788   py: 10.676
