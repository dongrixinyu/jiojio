# -*- coding=utf-8 -*-

import os
import pdb
import numpy as np

from jiojio import logging, read_file_by_iter, TrieTree


class AddDict2Model(object):
    """ 向模型中添加词典，提高模型泛化
    """
    def __init__(self, user_dict_path=None):
        if user_dict_path is None:
            self.trie_tree_obj = None
        else:
            assert type(user_dict_path) is str
            self._add_dict(user_dict_path)

    def _add_dict(self, user_dict_path):

        self.trie_tree_obj = TrieTree()
        for idx, line in enumerate(read_file_by_iter(user_dict_path)):
            if line.count('\t') == 1:
                word, weight = line.strip().split('\t')
                weight = float(weight)
            elif line.count('\t') == 0:
                word = line.strip()
                weight = 1
            else:
                logging.warn('`{}` is illegal.'.format(line))

            self.trie_tree_obj.add_node(word.lower(), weight)  # 要先预处理 TODO
        logging.info('add {} words to user_dict.'.format(idx + 1))

    def __call__(self, text, node_states):
        """为节点状态添加词汇权重，软性增强词汇被识别的能力

        Args:
            text(str): 待处理文本
            node_states: 根据模型得到的numpy 格式的节点状态矩阵

        Returns:
            (numpy.Array): 根据模型增强后的矩阵，该类型无需返回值

        """
        text_length = len(text)
        i = 0

        while i < text_length:
            pointer = text[i: self.trie_tree_obj.depth + i]
            # pointer = pointer_orig.lower()  # 不需要，因预处理已处理过 TODO
            step, typing = self.trie_tree_obj.search(pointer)
            if typing is not None:
                node_states[i][0] += typing
                for j in range(i + 1, i + step):
                    node_states[j][1] += typing
                if i + step < text_length:
                    node_states[i + step][0] += typing

            i += step

        # return node_states
