# -*- coding=utf-8 -*-

import os
import pdb
import time
import numpy as np
from multiprocessing import Process, Queue

from jiojio import TimeIt, config
from jiojio.data import DataSet
from jiojio.feature_extractor import FeatureExtractor

from jiojio.model import Model
from jiojio.inference import decodeViterbi_fast
from jiojio.optimizer import SGD
from jiojio.scorer import F1_score


def train(config):

    feature_extractor = FeatureExtractor()

    # ''' # 构建 特征数据集
    with TimeIt('# build datasets'):
        feature_extractor.build(config.train_file)
        feature_extractor.save()

    with TimeIt('# make feature files'):
        feature_extractor.convert_text_file_to_feature_file(
            config.train_file, config.c_train, config.f_train)
        feature_extractor.convert_text_file_to_feature_file(
            config.test_file, config.c_test, config.f_test)

        feature_extractor.convert_feature_file_to_idx_file(
            config.f_train, config.feature_train_file, config.gold_train_file)
        feature_extractor.convert_feature_file_to_idx_file(
            config.f_test, config.feature_test_file, config.gold_test_file)
    # '''
    # feature_extractor.load(config.train_dir)

    print("\nstart training ...")

    with TimeIt('loading dataset'):
        train_set = DataSet.load(config.feature_train_file, config.gold_train_file)
        test_set = DataSet.load(config.feature_test_file, config.gold_test_file)

    print("train_set size: {}, test_set size: {}\n".format(len(train_set), len(test_set)))

    trainer = Trainer(config, train_set, feature_extractor)

    for i in range(config.train_epoch):
        print('- epoch {}:'.format(i))
        with TimeIt('training epoch {}'.format(i)):
            err, diff = trainer.train_epoch()

        # 测试
        with TimeIt('loading valid dataset'):
            if i != config.train_epoch - 1:  # 最后一个 epoch 用全量
                sample_ratio = 1
            else:
                sample_ratio = 0.1  # 仅用全数据量的 5% 做训练中验证

            train_valid_set = DataSet.load(
                config.feature_train_file, config.gold_train_file, sample_ratio=sample_ratio)
            test_valid_set = DataSet.load(
                config.feature_test_file, config.gold_test_file, sample_ratio=sample_ratio)

        with TimeIt('Testing'):
            print('# train_set:')
            train_score_list = trainer.test(train_valid_set)
            print('# test_set:')
            test_score_list = trainer.test(test_valid_set)

            # 计算所有参数的最大值，平均值，中位值，确保模型的参数稳定
            max_weight = np.max(trainer.model.w)
            min_weight = np.min(trainer.model.w)
            average_weight = np.sum(trainer.model.w) / len(trainer.model.w)
            average_abs_weight = np.sum(np.abs(trainer.model.w)) / len(trainer.model.w)

        print("- epoch {}: \n"
              "\t- diff={:.4f}  error={:.4f}\n"
              "\t- max-weight={:.4f}  min-weight={:.4f}\n"
              "\t  average-weight={:.4f}  average-abs-weight={:.4f}".format(
                  i, diff, err, max_weight, min_weight,
                  average_weight, average_abs_weight))
        print("-" * 50 + "\n")

    trainer.model.save()

    print("finished.")


class Trainer(object):

    def __init__(self, config, dataset, feature_extractor):
        self.config = config
        self.X = dataset
        self.n_feature = dataset.n_feature
        self.n_tag = dataset.n_tag

        self.model = Model(self.n_feature, self.n_tag)

        self.optim = self._get_optimizer(dataset, self.model)

        self.feature_extractor = feature_extractor
        self.idx_to_chunk_tag = dict()
        for tag, idx in feature_extractor.tag_to_idx.items():
            # if tag.startswith("I"):
            #     tag = "I"
            self.idx_to_chunk_tag[idx] = tag

    def _get_optimizer(self, dataset, model):
        config = self.config
        return SGD(config, dataset, model)

    def train_epoch(self):
        return self.optim.optimize()

    def _decode(self, test_set: DataSet, model: Model):
        if config.nThread == 1:
            self._decode_single(test_set, model)
        else:
            self._decode_multi_proc(test_set, model)

    def _decode_single(self, test_set: DataSet, model: Model):
        for example in test_set:
            tags = decodeViterbi_fast(example.features, model)
            example.predicted_tags = tags

    @staticmethod
    def _decode_proc(model, in_queue, out_queue):
        while True:
            item = in_queue.get()
            if item is None:
                return

            idx, features = item
            tags = decodeViterbi_fast(features, model)
            out_queue.put((idx, tags))

    def _decode_multi_proc(self, test_set: DataSet, model: Model):
        in_queue = Queue()
        out_queue = Queue()
        procs = list()
        nthread = self.config.nThread
        for i in range(nthread):
            p = Process(
                target=self._decode_proc, args=(model, in_queue, out_queue))
            procs.append(p)

        for idx, example in enumerate(test_set):
            in_queue.put((idx, example.features))

        for proc in procs:
            in_queue.put(None)
            proc.start()

        for _ in range(len(test_set)):
            idx, tags = out_queue.get()
            test_set[idx].predicted_tags = tags

        for p in procs:
            p.join()

    def test(self, dataset):
        self._decode(dataset, self.model)

        gold_tags = list()
        pred_tags = list()

        sample_wrong = 0
        token_wrong = 0
        token_total = 0
        for example in dataset:
            pred_tags.append(example.predicted_tags)
            gold_tags.append(example.tags)

            flag = False
            for pred_tag, gold_tag in zip(example.predicted_tags, example.tags):
                if pred_tag != gold_tag:
                    flag = True
                    token_wrong += 1

            token_total += len(example.tags)
            if flag:
                sample_wrong += 1  # 样本错误

        score_list, info_list = F1_score(gold_tags, pred_tags, self.idx_to_chunk_tag)
        print("\t- test-sample-num={}\n"
              "\t- gold-num={}  output-num={}  correct-num={}\n"
              "\t- precision={:.2%}  recall={:.2%}  f-score={:.2%}\n"
              "\t- token_acc={:.2%}  sample_acc={:.2%}\n".format(
                  len(dataset),
                  info_list[0], info_list[1], info_list[2],
                  score_list[1], score_list[2], score_list[0],
                  (token_total - token_wrong) / token_total,
                  (len(dataset) - sample_wrong) / len(dataset)))

        return score_list
