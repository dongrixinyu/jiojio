# from .config import config
# from .feature import *
# from .data_format import *
# from .toolbox import *
import os
import pdb
import time
from multiprocessing import Process, Queue

import jionlp as jio

from jiojio.config import Config, config
from jiojio.data import DataSet
from jiojio.feature_extractor import FeatureExtractor

# from .feature_generator import *
from jiojio.model import Model
from jiojio.inference import decodeViterbi_fast
from jiojio.optimizer import ADF
from jiojio.scorer import F1_score


def train(config=None):
    if config is None:
        config = Config()

    feature_extractor = FeatureExtractor()

    # ''' # 构建 特征数据集
    with jio.TimeIt('# build datasets'):
        feature_extractor.build(config.train_file)
        feature_extractor.save()

    with jio.TimeIt('# make feature files'):
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
    print("\nreading training & test data ...")

    with jio.TimeIt('loading dataset'):
        train_set = DataSet.load(config.feature_train_file, config.gold_train_file)
        test_set = DataSet.load(config.feature_test_file, config.gold_test_file)

    print("train_set size: {}, test_set size: {}\n".format(len(train_set), len(test_set)))

    trainer = Trainer(config, train_set, feature_extractor)

    time_list = list()

    for i in range(config.train_epoch):
        print('- epoch {}:'.format(i))
        with jio.TimeIt('training epoch {}'.format(i), no_print=True) as ti:
            err, diff = trainer.train_epoch()
            time_list.append(ti.break_point())

        # 测试
        print('# train_set:')
        train_score_list = trainer.test(train_set, i)
        print('# test_set:')
        test_score_list = trainer.test(test_set, i)

        print("- epoch {}  diff={:.4f}  error={:.4f} train-time(s)={:.2f} \n" \
              "\ttrain {}={:.2f}%  test {}={:.2f}%".format(
                  i, diff, err, time_list[-1], config.metric,
                  train_score_list[0], config.metric, test_score_list[0]))
        print("-" * 50 + "\n")

    if config.save:
        trainer.model.save()

    print("finished.")


class Trainer:
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
        return ADF(config, dataset, model)

        raise ValueError("Invalid Optimizer")

    def train_epoch(self):
        return self.optim.optimize()

    def test(self, test_set, iteration):
        score_list = self._decode_fscore(test_set, self.model)

        return score_list

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

    # token accuracy
    def _decode_tokAcc(self, dataset, model):

        self._decode(dataset, model)
        n_tag = model.n_tag
        all_correct = [0] * n_tag
        all_pred = [0] * n_tag
        all_gold = [0] * n_tag

        for example in dataset:
            pred = example.predicted_tags
            gold = example.tags

            print(",".join(map(str, pred)))
            print()

            for pred_tag, gold_tag in zip(pred, gold):
                all_pred[pred_tag] += 1
                all_gold[gold_tag] += 1
                if pred_tag == gold_tag:
                    all_correct[gold_tag] += 1

        print("% tag-type  #gold  #output  #correct-output  token-precision  "
              "token-recall  token-f-score\n")
        sumGold = 0
        sumOutput = 0
        sumCorrOutput = 0

        for i, (correct, gold, pred) in enumerate(
                zip(all_correct, all_gold, all_pred)):
            sumGold += gold
            sumOutput += pred
            sumCorrOutput += correct

            if gold == 0:
                rec = 0
            else:
                rec = correct * 100.0 / gold

            if pred == 0:
                prec = 0
            else:
                prec = correct * 100.0 / pred

            print("% {}:  {}  {}  {}  {:.2f}  {:.2f}  {:.2f}\n".format(
                i, gold, pred, correct, prec, rec,
                (2 * prec * rec / (prec + rec))))

        if sumGold == 0:
            rec = 0
        else:
            rec = sumCorrOutput * 100.0 / sumGold
        if sumOutput == 0:
            prec = 0
        else:
            prec = sumCorrOutput * 100.0 / sumOutput

        if prec == 0 and rec == 0:
            fscore = 0
        else:
            fscore = 2 * prec * rec / (prec + rec)

        print("% overall-tags:  {}  {}  {}  {:.2f}  {:.2f}  {:.2f}\n".format(
            sumGold, sumOutput, sumCorrOutput, prec, rec, fscore))
        return [fscore]

    def _decode_strAcc(self, dataset, model):

        self._decode(dataset, model)

        correct = 0
        total = len(dataset)

        for example in dataset:
            pred = example.predicted_tags
            gold = example.tags

            print(",".join(map(str, pred)))
            print()

            for pred_tag, gold_tag in zip(pred, gold):
                if pred_tag != gold_tag:
                    break
            else:
                correct += 1

        acc = correct / total * 100.0
        print("total-tag-strings={}  correct-tag-strings={}  string-accuracy={}%".format(
            total, correct, acc))
        return [acc]

    def _decode_fscore(self, dataset, model):
        self._decode(dataset, model)

        gold_tags = list()
        pred_tags = list()

        for example in dataset:
            pred_tags.append(example.predicted_tags)
            gold_tags.append(example.tags)

        score_list, info_list = F1_score(gold_tags, pred_tags, self.idx_to_chunk_tag)
        print("\t# gold-num={}  output-num={}  correct-num={}\n"
              "\t# precision={:.2f}%  recall={:.2f}%  f-score={:.2f}%\n".format(
                  info_list[0], info_list[1], info_list[2],
                  score_list[1], score_list[2], score_list[0]))

        return score_list
