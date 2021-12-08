# -*- coding=utf-8 -*-

import os
import cProfile
import jiojio

print(jiojio.__doc__)


def train():
    dataset_dir = r'/home/cuichengyu/dataset'
    jiojio.train(os.path.join(dataset_dir, 'train_cws.txt'),
                 os.path.join(dataset_dir, 'test_cws.txt'),
                 '/home/cuichengyu/github/jiojio/train_dir',
                 train_epoch=5)


def test():
    dataset_dir = r'/home/cuichengyu/dataset'


if __name__ == '__main__':
    # cProfile.run('train()', sort='cumtime')
    train()
