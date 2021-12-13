import pdb
import jiojio.model
from typing import List

from jiojio.inference import get_Y_YY, get_beliefs, \
    get_masked_beliefs, Belief, MaskedBelief
import jiojio.data


def get_grad_SGD_minibatch(grad: List[float], model: jiojio.model.Model,
                           X: List[jiojio.data.Example]):
    feature_id_dict = dict()
    errors = 0
    for x in X:
        error, feature_id_set = get_grad_CRF(grad, model, x)
        errors += error

        for i in feature_id_set:
            if i in feature_id_dict:
                feature_id_dict[i] += 1
            else:
                feature_id_dict.update({i: 1})

    for feature_id, num in feature_id_dict.items():
        grad[feature_id] /= num

    return errors / len(X), feature_id_dict


def get_grad_CRF(grad: List[float], model: jiojio.model.Model,
                 x: jiojio.data.Example):
    feature_id_set = set()

    n_tag = model.n_tag
    belief = Belief(len(x), n_tag)
    belief_masked = MaskedBelief(len(x), n_tag)

    Y, YY, masked_Y = get_Y_YY(model, x)

    Z, sum_edge = get_beliefs(belief, Y, YY)
    sum_edge_masked = get_masked_beliefs(belief_masked, masked_Y)
    # pdb.set_trace()
    for i, node_feature_list in enumerate(x.features):
        diff = belief.node_states[i] - belief_masked.node_states[i]
        for feature_id in node_feature_list:
            trans_id = feature_id * n_tag

            feature_id_set.update(range(trans_id, trans_id + n_tag))  # 需要更新梯度值的特征值索引
            grad[trans_id: trans_id + n_tag] += diff  # 计算梯度

    grad[model.offset:] += sum_edge - sum_edge_masked  # 更新转移概率梯度
    feature_id_set.update(range(model.offset, model.offset + n_tag * n_tag))

    return Z, feature_id_set
