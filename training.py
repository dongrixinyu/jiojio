# -*- coding=utf-8 -*-

import os
import cProfile
import jiojio


def train_cws():
    dataset_dir = r'/home/cuichengyu/dataset/cws'
    jiojio.train(os.path.join(dataset_dir, 'train_cws.txt1'),
                 os.path.join(dataset_dir, 'test_cws.txt1'),
                 # model_dir='/home/cuichengyu/github/jiojio/jiojio/models/pos_model',
                 model_dir='test_cws_model',
                 task='cws')


def train_pos():
    dataset_dir = r'/home/cuichengyu/dataset/bosson'
    jiojio.train(os.path.join(dataset_dir, 'train_pos.txt'),
                 os.path.join(dataset_dir, 'test_pos.txt'),
                 # model_dir='test_pos_model',
                 task='pos')


def test():
    dataset_dir = r'/home/cuichengyu/dataset/cws/'
    jiojio.test(os.path.join(dataset_dir, 'test_cws.txt'), nthread=1)


def predict():
    text = '625年，春天陈中和去了上海。你喜欢学政治经济学吗。十道海量数据处理面试题与十个方法大总结 - CSDN博客，郎平和秦钟和教练激动地握了握手'
    jiojio.init()  # 初始化
    # jiojio.init(user_dict='/home/cuichengyu/github/jiojio/user_dict.txt')
    res = jiojio.cut(text)
    assert len(text) == len(''.join(res))
    print(res)


if __name__ == '__main__':
    # predict()
    # cProfile.run('train_cws()', sort='cumtime')
    # train_cws()
    train_pos()
    # cProfile.run('test()', sort='cumtime')
    # test()
