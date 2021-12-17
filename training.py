# -*- coding=utf-8 -*-

import os
import cProfile
import jiojio


def train():
    dataset_dir = r'/home/cuichengyu/dataset'
    jiojio.train(os.path.join(dataset_dir, 'train_cws.txt'),
                 os.path.join(dataset_dir, 'test_cws.txt'),
                 train_epoch=7)


def test():
    dataset_dir = r'/home/cuichengyu/dataset'
    jiojio.test(os.path.join(dataset_dir, 'test_cws.txt'),
                nthread=10)


def predict():
    text = '十道海量数据处理面试题与十个方法大总结 - CSDN博客'
    jiojio.init()  # 初始化
    res = jiojio.cut(text)
    assert len(text) == len(''.join(res))
    print(res)


if __name__ == '__main__':
    # predict()
    # cProfile.run('train()', sort='cumtime')
    # train()
    # cProfile.run('test()', sort='cumtime')
    test()
