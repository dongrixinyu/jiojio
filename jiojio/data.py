# -*- coding=utf-8 -*-

import pdb
import copy
import random


class DataSet(object):
    def __init__(self, n_tag=0, n_feature=0):
        self.samples = list()  # List[Example]
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
    def load(cls, feature_idx_file, tag_idx_file):
        dataset = cls.__new__(cls)

        with open(feature_idx_file, encoding="utf-8") as f_reader:
            example_strs = f_reader.read().split("\n\n")[:-1]

        with open(tag_idx_file, encoding="utf-8") as t_reader:
            tags_strs = t_reader.read().split("\n\n")[:-1]

        assert len(example_strs) == len(tags_strs), \
            "lengths do not match:\t{}\n{}\n".format(example_strs, tags_strs)

        # pdb.set_trace()
        n_feature = int(example_strs[0])
        n_tag = int(tags_strs[0])

        dataset.n_feature = n_feature
        dataset.n_tag = n_tag
        dataset.samples = list()

        for example_str, tags_str in zip(example_strs[1:], tags_strs[1:]):
            features = [list(map(int, feature_line.split(",")))
                        for feature_line in example_str.split("\n")]
            tags = tags_str.split(",")
            example = Example(features, tags)
            dataset.samples.append(example)

        return dataset


class Example:
    def __init__(self, features, tags):
        self.features = features  # List[List[int]]
        self.tags = list(map(int, tags))  # List[int]
        self.predicted_tags = None

    def __len__(self):
        return len(self.features)
