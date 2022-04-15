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

    def __init__(self):
        # 若自行训练模型，请务必保留该页参数，尤其是特征参数 features params 选取部分，将直接影响结果

        # dirs
        self.jiojio_home = os.path.expanduser('~/.jiojio')
        self.train_dir = os.path.join(self.jiojio_home, 'cws_train_dir')
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                      'models/default_cws_model')

        # training params
        self.initial_learning_rate = 0.015  # 梯度初始值
        self.dropping_rate = 0.12  # 维持学习速率，越大则下降越快(0~1) 推荐(0.7~0.999)
        self.random_init = True  # False for 0-init of model weights, True for random init of model weights
        self.train_epoch = 7  # 训练迭代次数
        self.mini_batch = 4000  # mini-batch in stochastic training
        self.nThread = 20  # number of processes in testing and training
        self.regularization = True  # 建议保持
        self.sample_ratio = 0.02  # 抽取总数据集中做训练中途验证的数据集比例
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

        self.feature_trim = 20  # 普通特征的删减数值 (20/3)
        self.gap_1_feature_trim = 35  # 间隔为1的删减阈值(30/4)
        self.gap_2_feature_trim = 65  # 带有 2 个字的跨度的特征删除量 (60/7)
        self.gap_3_feature_trim = 80  # 带有 3 个字的跨度的特征删除量 (80/10)
        self.unigram_feature_trim = 40  # 单词特征的数量 (80/4)
        self.bigram_feature_trim = 5  # 单词特征的数量 (6/2)
        # self.word_feature = True  # 需要返回 词汇 特征，若丢弃词汇特征，则计算耗时减少 30~40%
        self.word_max = 4  # 越大，则计算耗时越长，因此不建议超过 6，过短如 3 则会造成模型效果下降
        self.word_min = 2  # 此值基本固定不变
        self.label_num = 2  # 标签数量，即 B,I 两个，有助于模型计算加速

        if False:  # 针对小数据集的参数
            self.interval = 2  # 按多少间隔打印日志
            self.train_epoch = 3  # 训练轮数
            self.initial_learning_rate = 0.2  # 梯度初始值
            self.dropping_rate = 0.4  # 梯度下降率
            self.mini_batch = 1000

            self.feature_trim = 2  # 普通特征的删减数值 (20/3)
            self.gap_1_feature_trim = 2  # 间隔为1的删减阈值(30/4)
            self.gap_2_feature_trim = 3  # 带有 2 个字的跨度的特征删除量 (60/7)
            self.gap_3_feature_trim = 4  # 带有 3 个字的跨度的特征删除量 (80/10)
            self.unigram_feature_trim = 2  # 单词特征的数量 (80/4)
            self.bigram_feature_trim = 1  # 单词特征的数量 (7/2)

            self.sample_ratio = 0.3

    def params_check(self):
        assert self.initial_learning_rate > 0
        assert self.train_epoch > 0 and type(self.train_epoch) is int
        assert self.mini_batch > 0 and type(self.train_epoch) is int

    def to_json(self):
        # 将关键推理参数，导出为 json 数据，存入模型目录下
        inference_dict = {
            'word_max': self.word_max,
            'word_min': self.word_min,
            'norm_text': self.norm_text,
            'convert_num_letter': self.convert_num_letter,
            'normalize_num_letter': self.normalize_num_letter,
            'convert_exception': self.convert_exception,
        }
        with open(os.path.join(self.model_dir, 'params.json'), 'w', encoding='utf-8') as fw:
            json.dump(inference_dict, fw, ensure_ascii=False, indent=4, separators=(',', ':'))


cws_config = Config()
cws_config.params_check()
