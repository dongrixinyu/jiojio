import random

import numpy as np
from jiojio.gradient import get_grad_SGD_minibatch


class Optimizer(object):
    def __init__(self):
        self._pre_values = list()

    def converge_test(self, err):
        val = 1e100
        if len(self._pre_values) > 1:
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
        # self.decay_list = np.ones_like(self._model.w) * config.rate0  # 梯度下降步长
        self.decay_rate = config.rate0
        self.epoch_num = 0  # 计算训练第几轮

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
        w = self._model.w
        feature_num = w.shape[0]  # 特征参数量
        grad = np.zeros(feature_num)  # 梯度值
        error = 0

        sample_num, random_index = self._tune_reshuffle_samples(config)

        for t in range(0, sample_num, config.miniBatch):
            XX = list()
            for i in random_index[t: t + config.miniBatch]:
                XX.append(self.dataset[i])  # 小 batch 样本

            err, feature_set = get_grad_SGD_minibatch(grad, self._model, XX)
            error += err

            feature_set = list(feature_set)

            # update decay rates
            # self.decay_rate *= (config.upper - (config.upper -
            #                     config.lower) / config.miniBatch)
            self.decay_rate = config.rate0 * \
                np.exp(- self.epoch_num * config.rate1)
            self.epoch_num += 1

            # update weights
            w[feature_set] -= self.decay_rate * \
                grad[feature_set] / config.miniBatch

            # regularization
            if config.regularization != 0:
                # 参数正则化，该公式，对大参数值的惩罚越大
                w -= self.decay_rate * (
                    w / (config.regularization ^ 2) * config.miniBatch / sample_num)

            if config.regularization != 0:
                s = (w * w).sum()
                error += s / (2.0 * (config.regularization ^ 2))  # 对误差加正则参数

        diff = self.converge_test(error)

        return error, diff
