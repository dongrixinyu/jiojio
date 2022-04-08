# -*- coding=utf-8 -*-

import os
import pdb
import sys
import time
import ctypes
# import jionlp as jio


class POSFeatureExtractor(object):

    def __init__(self, single_pos_word, part, unigram):

        self.unigram = unigram
        # self.bigram = set()
        self.single_pos_word = single_pos_word
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
        feature_list = list()

        # 某些特殊的词汇，仅仅具有唯一一种词性，此时仅仅添加 “w我” 即可。不需要其它特征。
        # 此时，我们需要将此类词汇统计出来。词频大于10，且具有唯一的词性的词汇
        # self.single_pos
        # 此外，某些字也代表着唯一词性，例如 “崔” 几乎仅仅作为中文姓氏出现，再无别义，也可以采取此方式处理。
        if cur_w in self.single_pos_word:
            feature_list.append(self.word_current + cur_w)
            return feature_list

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
        # 俗语长度常为 4 个字等
        cur_w_length = len(cur_w)
        if cur_w_length <= 9:
            feature_list.append(self.word_current_length + str(cur_w_length))
        else:
            feature_list.append(self.word_current_length + '0')

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
        return feature_list
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

        if idx < len(token_list) - 1:
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
        if idx < len(token_list) - 2:
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


dir_path = '/home/ubuntu/github/jiojio/jiojio/jiojio_cpp'
feature_extractor = ctypes.PyDLL(
    os.path.join(dir_path, 'build', 'libposFeatureExtractor.so'))
get_pos_node_feature_c = feature_extractor.getPosNodeFeature
get_pos_node_feature_c.argtypes = [
    ctypes.c_int, ctypes.py_object, ctypes.py_object,
    ctypes.py_object, ctypes.py_object, ctypes.py_object]
get_pos_node_feature_c.restype = ctypes.py_object


word_list = ['今天', '圣诞节', '中华人民共和国', '的', '天气', '真的', '挺好', '。']
single_pos_word = set(['天气', '今天'])
unigram = set(["天气", "今天", "中国", "美国", "总统", "总统府"])
bigram = set(["美国.总统", "天气.晴朗", "美国.总统府"])
part = set(["节", "共和国"])
print(word_list)
# with open('/home/ubuntu/datasets/unigram.json', 'r', encoding='utf-8') as fr:
#     unigram = set(json.load(fr))
# with open('/home/ubuntu/datasets/bigram.json', 'r', encoding='utf-8') as fr:
#     bigram = set(json.load(fr))

index = 3
# res = get_pos_node_feature_c(
#     2, word_list, single_pos_word, unigram, bigram)

# sys.exit()
times = 100


# with jio.TimeIt('c ', no_print=True) as ti:
#     for i in range(times):
#         index = i % len(word_list)
#     pure_cost_time = ti.break_point()
'''
fe_obj = POSFeatureExtractor(single_pos_word, part, unigram)
for i in range(times):
    index = i % len(word_list)
    res1 = fe_obj.get_node_features(index, word_list)
    print(sys.gettotalrefcount())
    print(res1)
    pdb.set_trace()
'''

pdb.set_trace()
print('before:', sys.gettotalrefcount())
# with jio.TimeIt('c ', no_print=True) as ti:
for i in range(times):
    index = i % len(word_list)
    # index = 0
    res = get_pos_node_feature_c(
        index, word_list, single_pos_word, part, unigram, bigram)
print('after: ', sys.gettotalrefcount())
print(res)
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
