import pdb
import random

import numpy as np
from jiojio.gradient import get_grad_SGD_minibatch


class Optimizer(object):
    def __init__(self):
        self._pre_values = list()

    def converge_test(self, err):
        val = err
        if len(self._pre_values) > 0:
            if len(self._pre_values) == 10:  # 超出长度，弹出第一个值
                self._pre_values.pop(0)

            average_improvement = (self._pre_values[0] - err) / \
                len(self._pre_values)  # 训练提升量
            val = average_improvement / abs(err)  # 训练提升比率

        self._pre_values.append(err)
        return val

    def optimize(self):
        raise NotImplementedError()


class ADF(Optimizer):
    def __init__(self, config, dataset, model):
        super().__init__()

        self.config = config

        self._model = model
        self.dataset = dataset
        self.decay_rate = config.initial_learning_rate
        # self.training_batch_num = 0  # 计算训练 batch 数
        self.training_epoch_num = 0  # 计算训练轮数

    def _tune_reshuffle_samples(self, config):
        # 调整样本，打乱样本
        sample_num = len(self.dataset)

        random_index = list(range(sample_num))
        random.shuffle(random_index)  # 样本打乱
        remainder = sample_num % config.miniBatch
        if remainder != 0:
            sample_num += (config.miniBatch - remainder)
            random_index.extend(random_index[:(config.miniBatch - remainder)])

        return sample_num, random_index

    def optimize(self):
        config = self.config
        print('model w init: ', self._model.w[:3], '...', self._model.w[-2:])
        feature_num = self._model.w.shape[0]  # 特征参数量
        grad = np.zeros(feature_num)  # 梯度值
        error_list = list()

        sample_num, random_index = self._tune_reshuffle_samples(config)

        for t in range(0, sample_num, config.miniBatch):
            XX = list()
            for i in random_index[t: t + config.miniBatch]:
                XX.append(self.dataset[i])  # 小 batch 样本

            error, feature_set = get_grad_SGD_minibatch(grad, self._model, XX)
            error_list.append(error)

            # update decay rates
            self.decay_rate = config.initial_learning_rate * \
                np.exp((- self.training_epoch_num - t / sample_num) * config.dropping_rate)
            if t / config.miniBatch == 50:
                print('\tlr: {:.5f}, grad {}: '.format(self.decay_rate, t),
                      grad[:3], '...', grad[-1])
            # update weights
            self._model.w -= self.decay_rate * grad

            # regularization
            if config.regularization != 0:
                # 参数正则化，该公式，对大参数值的惩罚越大
                r2 = self.decay_rate * self._model.w
                # print('\tsum(abs(regular)): {:.4f}'.format(abs(r2).sum()))
                # print('\tsum(abs(weight)):  {:.4f}'.format(abs(self._model.w).sum()))
                self._model.w -= r2

        diff = self.converge_test(sum(error_list) / len(error_list))

        print('err/diff: {:.4f}/{:.4f}'.format(sum(error_list) / len(error_list), diff))
        print('model w change: ', self._model.w[:3], '...', self._model.w[-2:])

        self.training_epoch_num += 1
        return error, diff
