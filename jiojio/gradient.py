import pdb
import numpy as np
from typing import List

from jiojio.dataset import Sample
from jiojio.model import Model
from jiojio.inference import get_Y_YY, get_beliefs, \
    get_masked_beliefs, Belief, MaskedBelief


def get_grad_SGD_minibatch(node_grad: np.ndarray, edge_grad: np.ndarray,
                           model: Model, X: List[Sample]):
    feature_id_dict = dict()
    errors = 0
    for x in X:
        error, feature_id_set = get_grad_CRF(node_grad, edge_grad, model, x)
        errors += error

        for i in feature_id_set:
            if i in feature_id_dict:
                feature_id_dict[i] += 1
            else:
                feature_id_dict.update({i: 1})

    for feature_id, num in feature_id_dict.items():
        node_grad[feature_id] /= num
    edge_grad /= len(X)

    return errors / len(X), feature_id_dict


def get_grad_CRF(node_grad: np.ndarray, edge_grad: np.ndarray,
                 model: Model, x: Sample):
    feature_id_set = set()

    n_tag = model.n_tag
    belief = Belief(len(x), n_tag)
    belief_masked = MaskedBelief(len(x), n_tag)

    # 为节省内存，每次均须将字符串解为 list 处理，增加了耗时
    orig_features = x.features
    orig_tags = x.tags
    x.features = [list(map(int, feature_line.split(",")))
                  for feature_line in orig_features.split("\n")]
    x.tags = list(map(int, orig_tags.split(',')))

    Y, YY, masked_Y = get_Y_YY(model, x)

    Z, sum_edge = get_beliefs(belief, Y, YY)
    sum_edge_masked = get_masked_beliefs(belief_masked, masked_Y)

    for i, node_feature_list in enumerate(x.features):
        diff = belief.node_states[i] - belief_masked.node_states[i]
        for feature_id in node_feature_list:

            feature_id_set.add(feature_id)  # 需要更新梯度值的特征值索引
            node_grad[feature_id] += diff  # 计算梯度

    #还原数据，节省内存
    x.features = orig_features
    x.tags = orig_tags

    edge_grad += np.reshape(sum_edge - sum_edge_masked, (n_tag, n_tag))  # 更新转移概率梯度

    return Z, feature_id_set
