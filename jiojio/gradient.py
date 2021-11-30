import jiojio.model
from typing import List

from jiojio.inference import get_Y_YY, get_beliefs, Belief
import jiojio.data


def get_grad_SGD_minibatch(grad: List[float], model: jiojio.model.Model,
                           X: List[jiojio.data.Example]):
    all_id_set = set()
    errors = 0
    for x in X:
        error, id_set = get_grad_CRF(grad, model, x)
        errors += error
        all_id_set.update(id_set)

    return errors, all_id_set


def get_grad_CRF(grad: List[float], model: jiojio.model.Model,
                 x: jiojio.data.Example):
    id_set = set()

    n_tag = model.n_tag
    belief = Belief(len(x), n_tag)
    belief_masked = Belief(len(x), n_tag)

    Y, YY, masked_Y, masked_YY = get_Y_YY(model, x)

    Z, sum_edge = get_beliefs(belief, Y, YY)
    ZGold, sum_edge_masked = get_beliefs(belief_masked, masked_Y, masked_YY)

    for i, node_feature_list in enumerate(x.features):
        for feature_id in node_feature_list:
            trans_id = model._get_node_tag_feature_id(feature_id, 0)
            id_set.update(range(trans_id, trans_id + n_tag))  # 需要更新梯度值的特征值索引
            grad[trans_id: trans_id + n_tag] += belief.node_states[i] - \
                belief_masked.node_states[i]  # 概率应为 0 或 1，根据得到的概率值进行更改

    backoff = model.n_feature * n_tag
    grad[backoff:] += sum_edge - sum_edge_masked  # 更新转移概率梯度
    id_set.update(range(backoff, backoff + n_tag * n_tag))

    return Z - ZGold, id_set
