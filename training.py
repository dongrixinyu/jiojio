# -*- coding=utf-8 -*-
# this script can NOT be run directly!! need to be checked.
import os
import json
import cProfile
import jiojio


def train_cws():
    dataset_dir = r'/home/cuichengyu/dataset/cws'
    jiojio.train(os.path.join(dataset_dir, 'train_cws.txt'),
                 os.path.join(dataset_dir, 'test_cws.txt'),
                 # model_dir='/home/cuichengyu/github/jiojio/jiojio/models/pos_model',
                 model_dir='test_cws_model',
                 # train_dir='test_cws_model1',
                 task='cws')


def train_pos():
    dataset_dir = r'/home/cuichengyu/dataset/bosson'
    jiojio.train(os.path.join(dataset_dir, 'train_pos.txt1'),
                 os.path.join(dataset_dir, 'test_pos.txt1'),
                 model_dir='test_pos_model',
                 task='pos')


def test():
    dataset_dir = r'/home/cuichengyu/dataset/cws/'
    jiojio.test(os.path.join(dataset_dir, 'test_cws.txt'), nthread=1)


def predict():
    text = '使用标准词典对分词标注数据进行矫正和修复'
    # text = '紫光流媒体服务'

    jiojio.init(pos=True)  # 初始化
    # jiojio.init(user_dict='/home/cuichengyu/github/jiojio/user_dict.txt')
    res = jiojio.cut(text)
    assert len(text) == len(''.join(res[0]))
    # print(res)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == '__main__':
    # predict()
    # cProfile.run('train_cws()', sort='cumtime')
    train_cws()
    # train_pos()
    # cProfile.run('test()', sort='cumtime')
    # test()
