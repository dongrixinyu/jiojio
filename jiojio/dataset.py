# -*- coding=utf-8 -*-

import pdb
import random


class DataSet(object):
    def __init__(self, n_tag=0, n_feature=0):
        self.samples = list()  # List[Sample]
        self.n_tag = n_tag
        self.n_feature = n_feature

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return self.iterator()

    def __getitem__(self, x):
        return self.samples[x]

    def iterator(self):
        for i in self.samples:
            yield i

    def append(self, x):
        self.samples.append(x)

    def clear(self):
        self.samples = list()

    def _resize(self, scale):
        # 将原有的 old_size 的数据集，重复复制 scale 遍
        dataset = DataSet(self.n_tag, self.n_feature)
        new_size = int(len(self) * scale)
        old_size = len(self)
        for i in range(new_size):
            if i >= old_size:
                i %= old_size
            dataset.append(self[i])
        return dataset

    @classmethod
    def load(cls, feature_idx_file, tag_idx_file, sample_ratio=1.0):
        """ 从文件加载数据集.

        Args:
            feature_idx_file(str): 特征文件，即 X
            tag_idx_file(str): 标签文件，即 Y
            sample_ratio(float): 欲从文件中取出的样本数量占全量比例，主要用于在训练
                过程中仅对部分数据集作验证，若为 1.0，则全部取出。

        """
        dataset = cls.__new__(cls)
        assert 0.0 < sample_ratio <= 1.0

        with open(feature_idx_file, encoding="utf-8") as f_reader:
            sample_strs = f_reader.read().split("\n\n")[:-1]

        with open(tag_idx_file, encoding="utf-8") as t_reader:
            tags_strs = t_reader.read().split("\n\n")[:-1]

        assert len(sample_strs) == len(tags_strs), \
            "lengths do not match:\t{}\n{}\n".format(sample_strs, tags_strs)

        # pdb.set_trace()
        n_feature = int(sample_strs[0])
        n_tag = int(tags_strs[0])

        dataset.n_feature = n_feature
        dataset.n_tag = n_tag
        dataset.samples = list()

        for sample_str, tags_str in zip(sample_strs[1:], tags_strs[1:]):
            if sample_ratio != 1.0:
                if random.random() > sample_ratio:
                    continue

            features = [list(map(int, feature_line.split(",")))
                        for feature_line in sample_str.split("\n")]
            tags = tags_str.split(",")
            sample = Sample(features, tags)
            dataset.samples.append(sample)

        return dataset


class Sample(object):
    def __init__(self, features, tags):
        self.features = features  # List[List[int]]
        self.tags = list(map(int, tags))  # List[int]
        self.predicted_tags = None

    def __len__(self):
        return len(self.features)
