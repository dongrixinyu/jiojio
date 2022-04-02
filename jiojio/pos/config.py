# -*- coding=utf-8 -*-
# Library: jiojio
# Author: dongrixinyu
# License: GPL-3.0
# Email: dongrixinyu.89@163.com
# Github: https://github.com/dongrixinyu/jiojio
# Description: fast Chinese Word Segmentation(CWS) and Part of Speech(POS) based on CPU.'

import os
import pdb
import json


class Config(object):
    # model_urls = {
    #     "pos": "https://github.com/dongrixinyu/jiojio/releases/download/v1.0.1/pos.zip",
    # }

    def __init__(self):
        # 若自行训练模型，请务必保留该页参数，尤其是特征参数 features params 选取部分，将直接影响结果

        # dirs
        self.jiojio_home = os.path.expanduser('~/.jiojio')
        self.train_dir = os.path.join(self.jiojio_home, 'pos_train_dir')
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                      'models/default_pos_model')

        # POS 类型文件
        self.pos_types_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'pos_types.yml')

        # training params
        self.initial_learning_rate = 0.004  # 梯度初始值
        self.dropping_rate = 0.6  # 维持学习速率，越大则下降越快(0~1) 推荐(0.7~0.999)
        self.random_init = True  # False for 0-init of model weights, True for random init of model weights
        self.train_epoch = 6  # 训练迭代次数
        self.mini_batch = 3000  # mini-batch in stochastic training
        self.nThread = 10  # number of processes in testing and training
        self.regularization = True  # 建议保持
        self.sample_ratio = 0.03  # 抽取总数据集中做训练中途验证的数据集比例
        self.interval = 50  # 按多少间隔 batch 打印日志

        self.feature_train_file = os.path.join(self.train_dir, 'feature_train.txt')
        self.gold_train_file = os.path.join(self.train_dir, 'gold_train.txt')
        self.feature_test_file = os.path.join(self.train_dir, 'feature_test.txt')
        self.gold_test_file = os.path.join(self.train_dir, 'gold_test.txt')

        # features params
        self.norm_text = True  # 将所有的 数字、字母正规化，但该参数基本上是必选参数，否则造成特征稀疏与 F1 值降低
        self.convert_num_letter = True  # 将所有数字、字母、空格等转换为 ascii 形式
        self.normalize_num_letter = False  # 将所有数字正规化为 7，字母正规化为 Z
        self.convert_exception = True  # 将所有异常字符全部转为固定常见字符 ん

        self.char_feature_trim = 20  # 普通字符的频次删减数值 (20/5)
        self.part_length_trim = 3  # part 词缀长度的最大值，超过此值删除，一般固定不变
        self.part_feature_trim = 30  # 词缀特征出现次数，该特征数应当小于 unigram_feature_trim 以便把低频的特征抽取出
        self.part_feature_non_chinese_trim = 50  # 词缀特征出现次数，低于该值的频次被删除，此为应对英文等字符情况
        self.feature_trim = 14  # 普通特征的删减数值 (20/3)
        self.unigram_feature_trim = 40  # 单词特征的数量 (40/4)
        self.word_max = 4  # 越大，则计算耗时越长，因此不建议超过 6，过短如 3 则会造成模型效果下降

        if True:
            self.interval = 5  # 按多少间隔打印日志
            self.initial_learning_rate = 0.5  # 梯度初始值
            # self.bi_ratio_times = int(1 / self.initial_learning_rate)
            self.dropping_rate = 0.3  # 维持学习速率，越大则下降越快(0~1) 推荐(0.7~0.999)
            self.train_epoch = 4  # 训练迭代次数

            self.char_feature_trim = 5
            self.part_length_trim = 3  # part 词缀长度的最大值，超过此值删除，一般固定不变
            self.part_feature_trim = 3  # 词缀特征出现次数，该特征数应当小于 unigram_feature_trim 以便把低频的特征抽取出
            self.part_feature_non_chinese_trim = 5  # 词缀特征出现次数，低于该值的频次被删除，此为应对英文等字符情况
            self.feature_trim = 3
            self.unigram_feature_trim = 5
            self.sample_ratio = 0.3  # 抽取总数据集中做训练中途验证的数据集比例

    def params_check(self):
        assert self.initial_learning_rate > 0
        assert self.train_epoch > 0
        assert self.mini_batch > 0
        assert self.part_feature_trim < self.part_feature_non_chinese_trim

    def to_json(self):
        # 将关键推理参数，导出为 json 数据，存入模型目录下
        inference_dict = {
            'word_max': self.word_max,
            'norm_text': self.norm_text,
            # 'pos_types_file' : self.pos_types_file,
            'convert_num_letter': self.convert_num_letter,
            'normalize_num_letter': self.normalize_num_letter,
            'convert_exception': self.convert_exception,
        }
        with open(os.path.join(self.model_dir, 'params.json'), 'w', encoding='utf-8') as fw:
            json.dump(inference_dict, fw, ensure_ascii=False, indent=4, separators=(',', ':'))


pos_config = Config()
pos_config.params_check()
