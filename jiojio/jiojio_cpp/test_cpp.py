# -*- coding=utf-8 -*-

import os
import numpy as np
import numpy.ctypeslib as npct
import ctypes
import pdb
import json
import jionlp as jio
# from jiojio.tag_words_converter import tag2word


def get_slice_str(iterator_obj, start, length, all_len):
    return iterator_obj[start: start + length]


class FeatureExtractor(object):

    def __init__(self):
        self.unigram = set(["天气", "今天", "中国", "美国", "总统", "总统府"])
        self.bigram = set(["美国.总统", "天气.晴朗", "美国.总统府"])
        self._create_features()
        self.word_max = 4
        self.word_min = 2

    def _create_features(self):
        self.start_feature = '[START]'
        self.end_feature = '[END]'

        self.delim = '.'
        self.empty_feature = '/'
        self.default_feature = '$$'

        seg = ''
        # 为了减少字符串个数，缩短匹配时间，根据前后字符的位置，制定规则进行缩短匹配，规则如下：
        # c 代表 char，后一个位置 char 用 d 表示，前一个用 b 表示，按字母表顺序完成。
        # w 代表 word，后一个位置 word 用 x 表示，双词用 w 表示，
        self.char_current = 'c' + seg
        self.char_before = 'b' + seg  # 'c-1.'
        self.char_next = 'd' + seg  # 'c1.'
        self.char_before_2 = 'a' + seg  # 'c-2.'
        self.char_next_2 = 'e' + seg  # 'c2.'
        self.char_before_3 = 'z' + seg
        self.char_next_3 = 'f' + seg
        self.char_before_current = 'bc' + seg  # 'c-1c.'
        self.char_before_2_current = 'ac' + seg  # 'c-2c'
        self.char_before_3_current = 'zc' + seg  # 'c-3c'
        self.char_current_next = 'cd' + seg  # 'cc1.'
        self.char_current_next_2 = 'ce' + seg  # 'cc2.'
        self.char_current_next_3 = 'cf' + seg  # 'cc3.'
        self.char_before_2_1 = 'ab' + seg  # 'c-2c-1.'
        self.char_next_1_2 = 'de' + seg  # 'c1c2.'

        self.word_before = 'v' + seg  # 'w-1.'
        self.word_next = 'x' + seg  # 'w1.'
        self.word_2_left = 'wl' + seg  # 'ww.l.'
        self.word_2_right = 'wr' + seg  # 'ww.r.'

        self.no_word = "nw" + seg
        self.length_feature_pattern = '{}{}'

    def get_node_features(self, idx, token_list):
        # 给定一个  token_list，找出其中 token_list[idx] 匹配到的所有特征
        length = len(token_list)
        cur_c = token_list[idx]
        feature_list = list()

        # 1 start feature
        # feature_list.append(self.default_feature)  # 取消默认特征，仅有极微小影响

        # 8 unigram/bgiram feature
        # 当前字特征
        feature_list.append(self.char_current + cur_c)

        if idx > 0:
            before_c = token_list[idx - 1]
            # 前一个字 特征
            feature_list.append(self.char_before + before_c)
            # 当前字和前一字组合特征
            feature_list.append(self.char_before_current + before_c + self.delim + cur_c)
        else:
            # 字符为起始位特征
            feature_list.append(self.start_feature)

        if idx < len(token_list) - 1:
            next_c = token_list[idx + 1]
            # 后一个字特征
            feature_list.append(self.char_next + next_c)
            # 当前字和后一字组合特征
            feature_list.append(self.char_current_next + cur_c + self.delim + next_c)
        else:
            # 字符为终止位特征
            feature_list.append(self.end_feature)

        if idx > 1:
            before_c2 = token_list[idx - 2]
            # 前第二字特征
            feature_list.append(self.char_before_2 + before_c2)
            # 前一字和前第二字组合
            # feature_list.append(self.char_before_2_1 + before_c2 + self.delim + before_c)
            feature_list.append(self.char_before_2_current + before_c2 + self.delim + cur_c)


        if idx < len(token_list) - 2:
            next_c2 = token_list[idx + 2]
            # 后第二字特征
            feature_list.append(self.char_next_2 + next_c2)
            # 后一字和后第二字组合
            # feature_list.append(self.char_next_1_2 + next_c + self.delim + next_c2)
            feature_list.append(self.char_current_next_2 + cur_c + self.delim + next_c2)

        if idx > 2:
            before_c3 = token_list[idx - 3]
            feature_list.append(self.char_before_3 + before_c3)
            # 前三字和当前字组合
            feature_list.append(self.char_before_3_current + before_c3 + self.delim + cur_c)

        if idx < len(token_list) - 3:
            next_c3 = token_list[idx + 3]
            feature_list.append(self.char_next_3 + next_c3)
            # 后一字和后第二字组合
            feature_list.append(self.char_current_next_3 + cur_c + self.delim + next_c3)

        # 2 * (wordMax-wordMin+1) word features (default: 2*(6-2+1)=10 )
        # the character starts or ends a word
        # 寻找该字前一个词汇特征(包含该字)
        pre_list_in = None
        # 寻找该字后一个词汇特征(包含该字)
        post_list_in = None
        # 寻找该字前一个词汇特征(不包含该字)
        pre_list_ex = None
        # 寻找该字后一个词汇特征(不包含该字)
        post_list_ex = None
        for l in range(self.word_max, self.word_min - 1, -1):
            # "suffix" token_list[n, n+l-1]
            # current character starts word
            if pre_list_in is None:
                pre_in_tmp = get_slice_str(token_list, idx - l + 1, l, length)
                if pre_in_tmp in self.unigram:
                    feature_list.append(self.word_before + pre_in_tmp)
                    pre_list_in = pre_in_tmp  # 列表或字符串，关系计算速度

            if post_list_in is None:
                post_in_tmp = get_slice_str(token_list, idx, l, length)
                if post_in_tmp in self.unigram:
                    feature_list.append(self.word_next + post_in_tmp)
                    post_list_in = post_in_tmp

            if pre_list_ex is None:
                pre_ex_tmp = get_slice_str(token_list, idx - l, l, length)
                if pre_ex_tmp in self.unigram:
                    pre_list_ex = pre_ex_tmp

            if post_list_ex is None:
                post_ex_tmp = get_slice_str(token_list, idx + 1, l, length)
                if post_ex_tmp in self.unigram:
                    post_list_ex = post_ex_tmp

        # 寻找连续两个词汇特征(该字在右侧词汇中)
        if pre_list_ex and post_list_in:  # 加速处理
            bigram = pre_list_ex + self.delim + post_list_in
            if bigram in self.bigram:
                feature_list.append(self.word_2_left + bigram)
            feature_list.append(
                self.word_2_left + self.length_feature_pattern.format(
                    len(pre_list_ex), len(post_list_in)))

        # 寻找连续两个词汇特征(该字在左侧词汇中)
        if pre_list_in and post_list_ex:  # 加速处理
            bigram = pre_list_in + self.delim + post_list_ex
            if bigram in self.bigram:
                feature_list.append(self.word_2_right + bigram)
            feature_list.append(
                self.word_2_right + self.length_feature_pattern.format(
                    len(pre_list_in), len(post_list_ex)))

        return feature_list


dir_path = '/home/ubuntu/github/jiojio/jiojio/jiojio_cpp'
feature_extractor = ctypes.PyDLL(
    os.path.join(dir_path, 'build', 'libfeatureExtractor.so'))
get_node_feature_c = feature_extractor.getNodeFeature
get_node_feature_c.argtypes = [
    ctypes.c_int, ctypes.c_wchar_p, ctypes.c_int, ctypes.py_object, ctypes.py_object]
get_node_feature_c.restype = ctypes.py_object

text = "一切存在之物，各自有其存在之本质，所谓本质（essence），即物之为物，所必具之固有性，缺此要素，则不成其为同类之物也，故本质常含有普遍性，必然性，而为某事某物共通之特质。"
unigram = set(["天气", "今天", "中国", "美国", "总统", "总统府"])
bigram = set(["美国.总统", "天气.晴朗", "美国.总统府"])
# unigram = list(["天气", "今天", "中国"])

with open('/home/ubuntu/datasets/unigram.json', 'r', encoding='utf-8') as fr:
    unigram = set(json.load(fr))
with open('/home/ubuntu/datasets/bigram.json', 'r', encoding='utf-8') as fr:
    bigram = set(json.load(fr))
print(text[25])
print(len(text))
unigram.add('nc')
unigram.add('nc\x00')
unigram.add('kl')
unigram.remove('据')
# unigram.add('kl\x00')
# for i in range(len(text)):
res = get_node_feature_c(5, text, len(text), unigram, bigram)
#    print(i, text[i], res)
print(text)
print(res)
print(type(res[0]))
sys.exit()
times = 100000
idx = 10
with jio.TimeIt('c ', no_print=True) as ti:
    for i in range(times):
        pass
    pure_cost_time = ti.break_point()

with jio.TimeIt('c ', no_print=True) as ti:
    for i in range(times):
        res = get_node_feature_c(idx, text, len(text), unigram, bigram)
    c_cost_time = ti.break_point()
# res = tag2word(text, states, len(states))
# pdb.set_trace()
fe_obj = FeatureExtractor()
with jio.TimeIt('py ', no_print=True) as ti:
    for i in range(times):
        res1 = fe_obj.get_node_features(idx, text)
    py_cost_time = ti.break_point()

print('c__time: ', c_cost_time - pure_cost_time)
print('py_time: ', py_cost_time - pure_cost_time)
print('faster: ', (py_cost_time - c_cost_time) / (c_cost_time - pure_cost_time))
print("if the same: ", res == res1)
print(res)
print(res1)
'''
dir_path = '/home/ubuntu/github/jiojio/jiojio/jiojio_cpp'
tag_words_converter = ctypes.cdll.LoadLibrary(
    os.path.join(dir_path, 'build', 'libtagWordsConverter.so'))
# tag_words_converter = npct.load_library(
#     'libtagWordsConverter',
#     os.path.join(dir_path, 'build'))
tag2word_c = tag_words_converter.tagWordsConverter
tag2word_c.restype = ctypes.py_object
# tag2word.restype = ctypes.py_object
# tag2word.argtypes = [ctypes.POINTER(ctypes.c_wchar), ctypes.py_object]  # ctypes.data_as(c_void_p)]
array_1d_int8 = npct.ndpointer(dtype=np.int8, ndim=1, flags='CONTIGUOUS')
# tag2word.argtypes = [ctypes.POINTER(ctypes.c_wchar)] # , array_1d_int8, ctypes.c_long]

text = "今天是个好天气！"
text = text * 10
states = np.empty(len(text), dtype=np.int8)  # 1 个字节，np.int32 为 4个字节
for i in range(len(text)):
    states[i] = np.random.randint(2)

# states = np.array(
#     [1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
#      1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0,
#      0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0,
#      0, 0, 0], dtype=np.int8)
# text = '今天是个好天气！今天是个好天气！今天是个好天气！今天是个好天气！今天是个好天气！今天是个好天气！今天是个好天气！今天是' \
#        '个好天气！今天是个好天气！今天是个好天'
print(states, type(states))
print(text)

# pdb.set_trace()

# 100000 次计算耗时对比:
# c: 0.920   py: 15.572
# c: 0.950   py: 15.786
# c: 0.909   py: 15.557
# 修改一条 else 逻辑部分
# c: 0.787   py: 11.118
# c: 0.773   py: 11.722
# c: 0.788   py: 10.676


times = 10000
with jio.TimeIt('c '):
    for i in range(times):
        res = tag2word_c(text, states.ctypes.data_as(ctypes.c_void_p), len(states))
# res = tag2word(text, states, len(states))
# pdb.set_trace()
with jio.TimeIt('py '):
    for i in range(times):
        res1 = tag2word(text, states)

print("if the same: ", res == res1)
# '''
