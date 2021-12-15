# -*- coding=utf-8 -*-

import os


class Config:
    lineEnd = "\n"
    biLineEnd = "\n\n"
    triLineEnd = "\n\n\n"
    undrln = "_"
    blank = " "
    tab = "\t"
    star = "*"
    slash = "/"
    comma = ","
    B = "B"
    model_urls = {
        "postag": "https://github.com/lancopku/pkuseg-python/releases/download/v0.0.16/postag.zip",
    }

    def __init__(self):
        # dirs
        self.train_dir = '/home/cuichengyu/github/jiojio/train_dir'
        self.jiojio_home = os.path.expanduser('~/.jiojio')
        self.model_dir = os.path.join(os.path.dirname(self.train_dir),
                                      "models/default_model")

        # training params
        self.initial_learning_rate = 0.05  # 梯度初始值
        self.dropping_rate = 0.8  # 维持学习速率，越大则下降越快(0~1) 推荐(0.7~0.999)
        self.random = 1  # 0 for 0-init of model weights, 1 for random init of model weights
        self.train_epoch = 20  # training iterations
        self.mini_batch = 2000  # mini-batch in stochastic training
        self.nThread = 10  # number of processes
        self.regularization = True  # 建议保持

        self.c_train = os.path.join(self.train_dir, "train.conll.txt")
        self.f_train = os.path.join(self.train_dir, "train.feat.txt")

        self.c_test = os.path.join(self.train_dir, "test.conll.txt")
        self.f_test = os.path.join(self.train_dir, "test.feat.txt")

        self.feature_train_file = os.path.join(self.train_dir, "feature_train.txt")
        self.gold_train_file = os.path.join(self.train_dir, "gold_train.txt")
        self.feature_test_file = os.path.join(self.train_dir, "feature_test.txt")
        self.gold_test_file = os.path.join(self.train_dir, "gold_test.txt")

        # features params
        self.norm_text = True  # 将所有的 数字、字母，正规化
        self.feature_trim = 5  # 特征出现频次过低则丢弃，当数据量超大时使用
        self.word_feature = True  # 需要返回 词汇 特征
        self.word_max = 5
        self.word_min = 2
        self.label_num = 2
        self.start_token = '[START]'  # start and end token
        self.end_token = '[END]'

    def globalCheck(self):
        assert self.initial_learning_rate > 0
        assert self.train_epoch > 0
        assert self.mini_batch > 0


config = Config()
config.globalCheck()
