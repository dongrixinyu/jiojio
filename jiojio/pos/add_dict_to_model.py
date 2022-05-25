# -*- coding=utf-8 -*-
# Library: jiojio
# Author: dongrixinyu
# License: GPL-3.0
# Email: dongrixinyu.89@163.com
# Github: https://github.com/dongrixinyu/jiojio
# Description: fast Chinese Word Segmentation(CWS) and Part of Speech(POS) based on CPU.'

import os
import pdb
import yaml
import numpy as np

from jiojio import logging, read_file_by_iter


class POSAddDict2Model(object):
    """ 向模型中添加词典，提高模型泛化
    """
    def __init__(self, user_dict_path=None, pos_types_file=None):
        if user_dict_path is None:
            self.word_pos_obj = None
        else:
            assert type(user_dict_path) is str
            self._add_dict(user_dict_path)

            if pos_types_file is None:
                with open(os.path.join(os.path.dirname(
                              os.path.abspath(__file__)), 'pos_types.yml'),
                          'r', encoding='utf-8') as f:
                    self.pos_types = yaml.load(f, Loader=yaml.SafeLoader)
            self.tag_to_idx = dict(
                [(t, i) for i, t in enumerate(sorted(list(self.pos_types['model_type'].keys())))])

    def _add_dict(self, user_dict_path):

        self.word_pos_obj = dict()
        for idx, line in enumerate(read_file_by_iter(user_dict_path)):
            if line.count('\t') == 2:
                word, pos, weight = line.strip().split('\t')
                weight = float(weight)
                self.word_pos_obj.update({word: [pos, weight]})

            elif line.count('\t') == 1:
                word, pos = line.strip().split('\t')
                weight = 1.
                self.word_pos_obj.update({word: [pos, weight]})
            else:
                logging.warn('`{}` is illegal.'.format(line))

        logging.info('add {} words to pos_user_dict.'.format(idx + 1))

    def __call__(self, word_list, node_states):
        """为节点状态添加词汇权重，软性增强词汇被识别的能力

        Args:
            word_list(list[str]): 待处理文本，词 list 格式
            node_states: 根据模型得到的numpy 格式的节点状态矩阵

        Returns:
            (numpy.Array): 根据模型增强后的矩阵，该类型无需返回值

        """
        for idx, word in enumerate(word_list):
            if word in self.word_pos_obj:

                pos_type, weight = self.word_pos_obj[word]
                node_states[idx][self.tag_to_idx[pos_type]] += weight
