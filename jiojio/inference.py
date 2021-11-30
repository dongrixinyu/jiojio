# -*- coding=utf-8 -*-

"""
参数与公式表征：
    - 模型特征计数索引：k
    - 模型特征参数：w = [w_1, w_2 ... w_k ... w_K]
    - 序列计数索引：t
    - 输入序列：x = [x_1, x_2 ... x_t ... x_T]
    - 输出序列：y = [y_1, y_2 ... y_t ... y_T]
    - 输出序列状态计数索引：i
    - 输出序列状态参数：s = [s_1, s_2 ... s_i ... s_I]

"""


import numpy as np


class Belief:
    def __init__(self, nNodes, nStates):
        self.node_states = np.zeros((nNodes, nStates))
        self.transition_states = np.zeros((nNodes, nStates * nStates))
        self.Z = 0


def get_beliefs(bel, Y, YY):
    """求解 Z 值。
    由于搜索空间巨大，将全局概率转换为 HMM 的转移概率进行计算。

    Args:
        bel: 条件转移概率的计算状态
        Y: 根据特征权重计算得到的 节点特征值
        YY: 根据特征权重计算得到的 转移特征值

    公式:
        alpha_t(i) = p(x_1, x_2 ... x_t, y_t=i) 表示当 t 时刻 y_t 为 i 的概率

        alpha_t+1(j) = SIGMA_s_1_I(alpha_t(i) * phi(y_t-1=i, y_t=j, x))

        其中，phi 函数为势函数，表示从 t-1 时刻到 t 时刻，状态从 i 转移到 j 的非规范化概率，即其不介于 0~1 之间
        phi(y_t-1=i, y_t=j, x) = exp(SIGMA_k_1_K(w_k * f_k(x, y_t-1, y_t)))

        将其 log 化，即演变成例子中的 YY 变量，这个特征函数 f_k 由单独的少量参数决定，但公式中该值还受到 x 控制，此处实践受到了简化。

        alpha_t+1(j) = SIGMA_s_1_I(phi(y_t=s) * alpha_t(i))
                     = SIGMA_s_1_I(exp(SIGMA_k_1_K(w_k * f_k(x, y_t=s, y_t+1=j))) * alpha_t(i))    (F1)

        同理，有 beta 公式。

        有前后向的计算公式，计算节点概率，节点 t 时，y_t=i 的概率：

        p(y_t=i|x) = (alpha_t(y_t=i) * beta(y_t=i)) / Z(x)
                   = (alpha_t(y_t=i) * beta(y_t=i)) / (SIGMA_s_1_I(alpha_T(y_T=s)))            (F2)

    """
    node_states = bel.node_states
    transition_states = bel.transition_states
    node_num = Y.shape[0]  # 序列节点个数
    tag_num = YY.shape[0]

    Z = 0
    alpha_Y = np.zeros(tag_num)  # 表示当前节点 t 处，状态 s 为 i 的概率，此处为各个状态的矩阵形式
    new_alpha_Y = np.zeros(tag_num)
    # tmp_Y = np.zeros(tag_num)
    YY_trans = YY.transpose()
    YY_t_r = YY_trans.reshape(-1)
    sum_edge = np.zeros(tag_num * tag_num)

    for i in range(node_num - 1, 0, -1):
        # 此时为逆序，即计算 beta_t(i) 的状态值，方便后续计算
        # 对当前的初始状态 bel_state，加上当前状态 Y，乘以转移概率 YY，形成前一个标签的状态
        tmp_Y = node_states[i] + Y[i]  # 转移得到的特征值与当前特征值相加得到的结果
        node_states[i-1] = log_multiply(YY, tmp_Y)  # 公式 (F1)

    for i in range(node_num):  # 正序计算
        if i > 0:
            tmp_Y = alpha_Y.copy()
            new_alpha_Y = log_multiply(YY_trans, tmp_Y) + Y[i]  # 公式 (F1)
        else:
            new_alpha_Y = Y[i].copy()

        if i > 0:
            tmp_Y = Y[i] + node_states[i]  # tmp_Y 为 beta 概率值
            transition_states[i] = YY_t_r
            for yPre in range(tag_num):
                for y in range(tag_num):
                    # 将 beta 值和 alpha 值相加，实为相乘
                    transition_states[i, y * tag_num + yPre] += \
                        tmp_Y[y] + alpha_Y[yPre]

        node_states[i] = node_states[i] + new_alpha_Y
        alpha_Y = new_alpha_Y

    Z = log_sum_exp(alpha_Y)

    for i in range(node_num):
        node_states[i] = np.exp(node_states[i] - Z)  # 转换为真实概率值
    for i in range(1, node_num):
        sum_edge += np.exp(transition_states[i] - Z)  # 转移概率转换为真实概率值，加和

    return Z, sum_edge


def run_viterbi(node_score, edge_score):
    # i, y, y_pre, i_pre, tag = 0, 0, 0, 0, 0
    w = node_score.shape[0]
    h = node_score.shape[1]
    max_score = np.zeros((w, h), dtype=np.float64)
    pre_tag = np.zeros((w, h), dtype=np.int8)
    init_check = np.zeros((w, h), dtype=np.uint8)
    states = np.zeros(w, dtype=np.int32)

    max_score[w-1] = node_score[w-1]

    for i in range(w - 2, -1, -1):
        for y in range(h):
            for y_pre in range(h):
                i_pre = i + 1
                sc = max_score[i_pre, y_pre] + \
                    node_score[i, y] + edge_score[y, y_pre]
                if not init_check[i, y]:
                    init_check[i, y] = 1
                    max_score[i, y] = sc
                    pre_tag[i, y] = y_pre
                elif sc >= max_score[i, y]:
                    max_score[i, y] = sc
                    pre_tag[i, y] = y_pre

    tag = np.argmax(max_score[0])
    ma = np.max(max_score[0])

    states[0] = tag
    for i in range(1, w):
        states[i] = pre_tag[i-1, tag]

    return states
    # if ma > 100:
    #     ma = 100
    # return np.exp(ma), states


def get_log_Y_YY(sequence_feature_list, tag_num, offset, w, scalar):
    """根据模型的参数，以及 x 序列，匹配得到特征值，计算得到两个矩阵
    node_score 和 transition_score，即每个节点，对应各个标签的得分。

    即公式：
        SIGMA_k_1_K(w_k * f_k(x, y))

    其中，w 表示特征参数，k 表示特征个数，SIGMA 表示连加符号，f 表示特征函数，
    x、y 分别表示输入序列和输出序列。

    此处为矩阵形式，即计算出所有可能的 y 序列的基础特征加权值，方便后续计算。

    Args:
        sequence_feature_list: 一个句子的特征值索引号，如 [[0, 12, 1903], [0, 8, 10293, 29801], [0, 2]]
        num_tag: 标签个数，比如，B、I，其标签数为 2
        offset: 特征数与标签数的乘积，为寻找转移特征的偏移量
        w: 模型参数
        scalar: 尺度标量

    """
    node_num = len(sequence_feature_list)
    # 每个节点的得分，base 为 0
    node_score = np.zeros((node_num, tag_num), dtype=np.float64)
    # 转移矩阵的得分，base 为 1
    edge_score = np.ones((tag_num, tag_num), dtype=np.float64)

    for i in range(node_num):
        node_feature_list = sequence_feature_list[i]
        for s in range(tag_num):
            for node_feature in node_feature_list:
                f = node_feature * tag_num + s  # 找到特征在 w 中的位置
                node_score[i, s] += w[f] * scalar

    for s in range(tag_num):
        for s_pre in range(tag_num):
            f = offset + s * tag_num + s_pre  # 找到转移特征在 w 中的位置
            edge_score[s_pre, s] += w[f] * scalar  # 由于特征参数只有一个，所以只累加一次

    return node_score, edge_score


def mask_Y(tags, node_num, tag_num, Y):
    """将前向计算得到的 Y 序列，根据标注的标签序列，做掩码，并返回。

    Args:
        tags: 样本的标签列表
        node_num: 样本的节点长度、节点个数
        tag_num: 标签的个数，如 B、I 标签规则，个数为 2。
        Y: 前向计算得到的特征势函数 矩阵

    """
    mask_Yi = Y.copy()  # numpy.Array 类型
    # maskValue = -1e100  # 负无穷
    mask_value = -np.inf
    # tagList = tags
    # i = 0
    for i in range(node_num):
        for s in range(tag_num):
            if tags[i] != s:  # 将不符合标注序列的部分置为负无穷
                mask_Yi[i, s] = mask_value

    return mask_Yi


def log_multiply(A, B):
    """ A 为状态转移矩阵，B 为初始化的节点特征，计算下一个节点的状态特征。A 和 B 均为 log 化的矩阵。

    公式为：
    针对某时刻 t 的节点特征状态，计算其 t+1 时刻的节点特征状态：
    alpha_t+1(j) = SIGMA_s_1_I(exp(phi(y_t=s)) * alpha_t(i))

    即公式：
    np.log(np.sum(np.exp(A) * np.exp(B), axis=1))
    其中：
        A: [state * state]
        B: [state]
        ret: [state]
        np.log 是为了便于将乘法转为加法的折中。

    """
    return np.log(np.sum(np.exp(A) * np.exp(B), axis=1))


def log_sum_exp(a):
    # a = [1, 0.5, -1, 10, 9, -4, 0, 1]
    # log(sum(exp(a_1), exp(a_i), ..., exp(a_n)))
    # np.log(np.sum(np.exp(a)))
    # 该函数偏向计算整个列表中最大的值的结果
    return np.log(np.sum(np.exp(a)))


def decodeViterbi_fast(feature_temp, model):
    Y, YY = get_log_Y_YY(feature_temp, model.n_tag,
                         model.n_feature*model.n_tag, model.w, 1.0)
    tags = run_viterbi(Y, YY)
    tags = list(tags)
    return tags


def get_Y_YY(model, example):
    """
    Args:
        model: 模型参数 object
        example: 一条样本，包含正确的标签，所有的特征索引。

    YY: Y 标签的转移势函数
    Y: Y 标签的节点势函数

    """
    Y, YY = get_log_Y_YY(example.features, model.n_tag,
                         model.n_feature * model.n_tag, model.w, 1.0)
    masked_Y = mask_Y(example.tags, len(example), model.n_tag, Y)
    masked_YY = YY  # 对某些标注系统，某个标签转移到另一个标签的概率为 0，此时需要做掩码。但一般情况下，不需要做，模型自己可以学习。
    return Y, YY, masked_Y, masked_YY
