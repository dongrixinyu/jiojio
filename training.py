# -*- coding=utf-8 -*-

import os
import cProfile
import jiojio

print(jiojio.__doc__)


def test():
    dataset_dir = r'/home/ubuntu/datasets/word_segmentation'
    jiojio.train(os.path.join(dataset_dir, 'train_cws.txt'),
                 os.path.join(dataset_dir, 'test_cws.txt'),
                 '/home/ubuntu/github/jiojio/train_dir',
                 train_iter=1)


if __name__ == '__main__':
    cProfile.run('test()', sort='cumtime')
