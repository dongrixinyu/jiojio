# -*- coding=utf-8 -*-
# Library: jiojio
# Author: dongrixinyu
# License: GPL-3.0
# Email: dongrixinyu.89@163.com
# Github: https://github.com/dongrixinyu/jiojio
# Description: fast Chinese Word Segmentation(CWS) and Part of Speech(POS) based on CPU.'
# Website: http://www.jionlp.com/


__doc__ = 'jiojio: for fast Chinese Word Segmentation(CWS) and Part of Speech(POS) based on CPU.'
__version__ = '1.1.8'


import os
import pdb
import yaml

from multiprocessing import Process, Queue, get_start_method

from .util import TimeIt, zip_file, unzip_file, read_file_by_iter, \
    write_file_by_line, TrieTree, set_logger, download_model

logging = set_logger(level='INFO', log_dir_name='.jiojio/jiojio_logs')

from .model import Model
from .parse_rule_type import Extractor

from .cws import CWSPredictText  # , F1_score, word2tag
from .pos import POSPredictText
from jiojio.cws.config import cws_config
from jiojio.pos.config import pos_config

import jiojio.cws.trainer as cws_trainer
import jiojio.pos.trainer as pos_trainer


__all__ = ['init', 'cut', 'train', 'test', 'help']

global jiojio_cws_obj, jiojio_pos_obj, jiojio_pos_flag


def help():
    print('    `jiojio` is an efficient Chinese Word Segmenter and Part of Speech tool.\n' \
          'It uses Python to wrap C for accelerating processing speed in CPU machines.\n' \
          'Here is an example for a quick start:\n' \
          '    >>> import jiojio\n' \
          '    >>> jiojio.init()\n' \
          '    >>> jiojio.cut("这是一个测试用例。")\n')

    print('    If you use computers with x86 structrue and linux OS, then when you execute\n' \
          '`import jiojio` and get the words `# Successfully load C funcs for acceleration`,\n' \
          'it means that you can get fast processing speed with C code.\n' \
          '    Otherwise you would get `# Failed to load C funcs, use py func instead`.\n' \
          'This does NOT mean `jiojio` is not being imported correctly. It means you could\n' \
          'only run `jiojio` relatively slow in Python code.\n')

    print('    If you have any questions, Github(https://github.com/dongrixinyu/jiojio) is\n' \
          'available to raise an issue. http://www.jionlp.com/ is used to try it online.')


def init(cws_model_dir=None, cws_user_dict=None, pos=False,
         pos_model_dir=None, pos_user_dict=None,
         cws_with_viterbi=False, pos_with_viterbi=False,
         cws_rule=False, pos_rule=False):
    """ 初始化模型，包括分别初始化分词模型与词性标注模型。

    Args:
        cws_model_dir(str): 分词模型名称，若为 None，则加载默认模型 default_cws_model；
            若自训练模型，则建议填写为模型目录的绝对路径，若仅仅为目录名（相对路径），如
            “self_train_cws_model” ，则加载时默认该模型目录存储在 “jiojio/models” 目录下。
        cws_user_dict(str): 指定加载分词的用户自定义词典文件的绝对路径，若不指定则不加载，
            默认不加载。
        pos(bool): 是否加载词性标注，默认为 False。
        pos_model_dir(str): 词性标注模型名称，若为 None，则加载默认模型 default_pos_model；
            若自训练模型，则建议填写为模型目录的绝对路径，若仅仅为目录名（相对路径），如
            “self_train_pos_model” ，则加载时默认该模型目录存储在 “jiojio/models” 目录下。
            在 pos 为 True 时生效。
        pos_user_dict(str): 指定加载词性标注用户自定义词典文件的绝对路径，若不指定则不加载，
            默认不加载。在 pos 为 True 时生效。
        cws_with_viterbi(bool): 是否在推断时采用 viterbi 解码。默认为 False。该参数原因在于
            分词模型由于 node_weight 参数强力的表达能力，以及转移参数过少导致的不可泛化，
            bi_ratio 参数值往往极小，此时可以考虑不经过 viterbi 转码，以节省推断时间。
        pos_with_viterbi(bool): 是否在推断时采用 viterbi 解码。默认为 False。仅在 pos 为
            True 时有效。该参数原因在于词性标注模型由于 node_weight 参数强力的表达能力，以及
            转移参数过少导致的不可泛化，bi_ratio 参数值往往极小，此时可以考虑不经过 viterbi
            转码，以节省推断时间。
        cws_rule(bool): 是否返回由规则切分词汇，默认为 False。规则词性类型包括：email、
            身份证号(id)、ip地址(ip)、日文(jp)、俄文(ru)、韩文(ko)、url。这些类型绝大多数
            并非由 CWS 模型返回，而是在模型返回结果基础上，再次套用规则计算得到。
        pos_rule(bool): 是否返回规则词性类型，默认为 False。规则词性类型包括：email、
            身份证号(id)、ip地址(ip)、日文(jp)、俄文(ru)、韩文(ko)、url。这些类型并非由 POS
            模型返回，而是在模型返回结果基础上，再次套用规则计算得到。

    Returns:
        None

    """
    global jiojio_cws_obj, jiojio_pos_obj, jiojio_pos_flag

    if pos_rule and pos:
        cws_rule = True

    jiojio_cws_obj = CWSPredictText(
        model_dir=cws_model_dir, user_dict=cws_user_dict,
        with_viterbi=cws_with_viterbi, rule_extractor=cws_rule)

    if pos:
        jiojio_pos_flag = True
        jiojio_pos_obj = POSPredictText(
            model_dir=pos_model_dir, user_dict=pos_user_dict,
            with_viterbi=pos_with_viterbi, pos_rule_types=pos_rule)
    else:
        jiojio_pos_flag = False
        jiojio_pos_obj = None


def cut(text):
    """ 对文本进行分词和词性标注 """
    if jiojio_pos_flag:
        words, norm_words, word_pos_map = jiojio_cws_obj.cut_with_pos(text)
        tags = jiojio_pos_obj.cut(norm_words, word_pos_map=word_pos_map)

        return [(w, t) for w, t in zip(words, tags)]

    else:
        words = jiojio_cws_obj.cut(text)

        return words


def pos_types():
    """ 打印并返回 POS 的词性类型 """
    pos_types_path = os.path.join(os.path.dirname(__file__), 'pos/pos_types.yml')

    with open(pos_types_path, 'r', encoding='utf-8') as f:
        pos_types = yaml.load(f, Loader=yaml.SafeLoader)

    return pos_types


def train(train_file, test_file, train_dir=None,
          model_dir=None, train_epoch=0, task='cws'):
    """用于训练模型，分析、确定参数

    Args:
        train_file: 训练数据文件，建议指定绝对路径。
        test_file: 测试数据文件，建议指定绝对路径。
        train_dir: 训练数据保存路径，主要为训练临时文件，例如，标注文本进行特征抽取后的中间文件等，
            用于分析数据，不用于推理，训练完后可删除。
        model_dir: 模型文件保存路径，训练模型文件、特征索引文件、以及推理参数文件，用于推理加载。
        train_epoch: 训练轮数。
        task: 训练任务，只支持 cws（分词）和 pos（词性标注）两种。

    """

    if task == 'cws':
        config = cws_config
    elif task == 'pos':
        config = pos_config
    else:
        raise ValueError('the param `task` must be `cws` or `pos`.')

    if (train_dir is None) or (type(train_dir) is not str):
        logging.info('using the default `train_dir` in `./jiojio/{}/config.py`.'.format(task))
    else:
        default_train_dir = config.jiojio_home
        if os.path.isabs(train_dir):
            config.train_dir = train_dir
        else:
            config.train_dir = os.path.join(default_train_dir, train_dir)

    if not os.path.exists(config.train_dir):
        os.makedirs(config.train_dir)

    if (model_dir is None) or (type(model_dir) is not str):
        logging.info('using the default `model_dir` in `./jiojio/{}/config.py`.'.format(task))
    else:
        default_model_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'models')
        if os.path.isabs(model_dir):
            config.model_dir = model_dir
        else:
            config.model_dir = os.path.join(default_model_dir, model_dir)

    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

    if not os.path.exists(train_file):
        raise ValueError('train_file does not exist.')
    if not os.path.exists(test_file):
        raise ValueError('test_file does not exist.')
    config.train_file = train_file
    config.test_file = test_file

    # config.nThread = 1
    if train_epoch > 0 and (type(train_epoch) is int):
        config.train_epoch = train_epoch

    with TimeIt('total training time'):
        if task == 'cws':
            cws_trainer.train(config)
        elif task == 'pos':
            # pdb.set_trace()
            pos_trainer.train(config)


def _test_single_proc(input_file, model_name=None, user_dict=None, pos=False):

    with TimeIt('# loading model'):
        seg = CWSPredictText(model_name, user_dict)

    if not os.path.exists(input_file):
        raise Exception('input_file {} does not exist.'.format(input_file))

    total_token_num = 0
    total_sample_num = 0
    orig_tag_list = list()
    pred_tag_list = list()
    with TimeIt('# cutting text') as ti:
        diff_results = list()
        for idx, line in enumerate(read_file_by_iter(input_file, line_num=100000)):
            text = ''.join(line)
            if idx % 1e5 == 0:
                print(idx)
            # print(text)
            total_sample_num += 1
            total_token_num += len(text)

            words_list = seg.cut(text)
            '''
            if words_list != line:
                # diff_results.append(line)
                min_len = min(len(words_list), len(line))
                for idx, (c, p) in enumerate(zip(words_list[:min_len], line[:min_len])):
                    if c != p:
                        print(line[max(0, idx - 5):idx + 7])
                        diff_results.append(words_list[max(0, idx - 5):idx + 7] + ['⚫'] + line[max(0, idx - 5):idx + 7])
                        # pdb.set_trace()
                        break
            '''

            # assert len(''.join(words_list)) == len(text)
            # print('true: ' , line)
            # print('pred: ', words_list)
            # print('correct num: ', total_sample_num - len(diff_results))
            # pdb.set_trace()
        cost_time = ti.break_point()

    logging.info("# total_time:\t{:.3f}".format(cost_time))
    logging.info("# total sample:{:},\t average {:} chars per sample ".format(
        total_sample_num, total_token_num / total_sample_num))
    logging.info("# {} chars per second".format(total_token_num / (cost_time)))

    # write_file_by_line(diff_results,
    #                    '/home/cuichengyu/dataset/diff_pred_cws_samples.txt')


def _proc(seg, in_queue, out_queue):
    # TODO: load seg (json or pickle serialization) in sub_process
    #       to avoid pickle seg online when using start method other
    #       than fork
    count = 0
    while True:
        item = in_queue.get()
        if item is None:
            logging.info('finished! Processed {} samples. '.format(count))
            return

        idx, line = item
        output_list = seg.cut(line)
        out_queue.put((idx, output_list))
        count += 1


def _proc_alt(model_name, user_dict, pos, in_queue, out_queue):
    seg = jiojio(model_name, user_dict, pos=pos)  # 独立进程，从 0 启动一个节点

    while True:
        item = in_queue.get()
        if item is None:
            return

        idx, line = item
        output_list = seg.cut(line)
        out_queue.put((idx, output_list))


def _test_multi_proc(input_file, nthread, model_name="default_model",
                     user_dict=None, pos=False):

    with TimeIt('# select method of starting subprocess'):
        alt = get_start_method() == 'spawn'
        # spawn 启动方式是 windows 的默认方式，fork 为 unix 的默认启动方式

        if alt:
            seg = None
        else:
            seg = PredictText(config, model_name, user_dict, pos)

    if not os.path.exists(input_file):
        raise Exception('input_file {} does not exist.'.format(input_file))

    in_queue = Queue()
    out_queue = Queue()
    procs = list()
    for _ in range(nthread):
        if alt:
            p = Process(target=_proc_alt,
                        args=(model_name, user_dict, pos, in_queue, out_queue))
        else:
            p = Process(target=_proc, args=(seg, in_queue, out_queue))
        procs.append(p)

    total_token_num = 0
    total_sample_num = 0
    for idx, line in enumerate(read_file_by_iter(input_file, line_num=100000)):
        text = ''.join(line)
        in_queue.put((idx, text))
        total_token_num += len(text)
        total_sample_num += 1

    for proc in procs:
        in_queue.put(None)  # 为每一个节点提供一个终止标识符，否则获取队列资源将进入等待状态，除非使用 get_nowait
        proc.start()  # 子进程启动

    # 调整结果的顺序，因处理并非同步，此步为真实计算过程，耗时较长
    result = [None] * total_sample_num
    with TimeIt('test computing...') as ti:
        for _ in result:
            idx, line = out_queue.get()  # 阻塞式等待处理结果
            result[idx] = line

        cost_time = ti.break_point()

    for p in procs:
        p.join()

    logging.info('# total sample:{:},\t average {:} chars per sample '.format(
        total_sample_num, total_token_num / total_sample_num))
    logging.info('# {} chars per second'.format(total_token_num / (cost_time)))


def test(input_file, model_name=None, user_dict=None, nthread=1, pos=False):
    """ 测试数据集预测效果，其中 nthread 可以使用 os.cpu_count() - 1 指定，
    来进行最大限度资源利用，主要用于测试处理速度 """

    if nthread > 1:
        _test_multi_proc(input_file, nthread, model_name, user_dict, pos)
    else:
        _test_single_proc(input_file, model_name, user_dict, pos)
