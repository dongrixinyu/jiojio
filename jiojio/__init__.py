# -*- coding=utf-8 -*-

__doc__ = 'for fast Chinese Word Segmentation(CWS) and Part of Speech(POS).'

import sys
import os
import pdb
import time

from multiprocessing import Process, Queue, get_start_method

from .util.logger import set_logger

logging = set_logger(level='INFO', log_dir_name='.jiojio_logs')

from .util import TimeIt, unzip_file, read_file_by_iter, write_file_by_line
from .config import config
import jiojio.trainer as trainer
from jiojio.trie_tree import TrieTree
from jiojio.predict_text import PredictText

__all__ = ['init', 'cut', 'train', 'test']

global jiojio_obj


def init(model_name=None, pos=False, user_dict=None):
    """ 初始化模型

    Args:
        model_name(str): 模型名称，若为 None，则加载默认模型 default_model
        pos(bool): 是否加载词性标注，默认为 False
        user_dict(str): 指定加载用户自定义词典的路径，若不指定则不加载

    Returns:
        None

    """
    global jiojio_obj
    jiojio_obj = PredictText(config,
                             model_name=model_name,
                             user_dict=user_dict)


def cut(text):
    words = jiojio_obj.cut(text)
    return words


def train(train_file,
          test_file,
          train_dir=None,
          model_dir=None,
          train_epoch=20):
    """用于训练模型，分析、确定参数"""

    if not os.path.exists(train_file):
        raise Exception("train_file does not exist.")
    if not os.path.exists(test_file):
        raise Exception("test_file does not exist.")

    if (train_dir is None) or (type(train_dir) is not str):
        print('using the default `train_dir` in `./jiojio/config.py`.')
    else:
        config.train_dir = train_dir
    if not os.path.exists(config.train_dir):
        os.makedirs(config.train_dir)

    if (model_dir is None) or (type(model_dir) is not str):
        print('using the default `model_dir` in `./jiojio/config.py`.')
    else:
        config.model_dir = model_dir
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

    config.train_file = train_file
    config.test_file = test_file

    config.nThread = 1
    config.train_epoch = train_epoch

    with TimeIt('total training time'):
        trainer.train(config)


def _test_single_proc(input_file, model_name=None, user_dict=None, pos=False):

    with TimeIt('# loading model'):
        seg = PredictText(config, model_name, user_dict, pos=pos)

    if not os.path.exists(input_file):
        raise Exception("input_file {} does not exist.".format(input_file))

    total_token_num = 0
    total_sample_num = 0
    with TimeIt('# cutting text') as ti:
        diff_results = list()
        for line in read_file_by_iter(input_file, line_num=100000):
            text = ''.join(line)

            # print(text)
            total_sample_num += 1
            total_token_num += len(text)

            words_list = seg.cut(text, normalize_num_letter=False)
            # if words_list != line:
            #     diff_results.append(line)

            # print(line)
            # print(words_list)
            # pdb.set_trace()
        cost_time = ti.break_point()

    print("# total_time:\t{:.3f}".format(cost_time))
    print("# total sample:{:},\t average {:} chars per sample ".format(
        total_sample_num, total_token_num / total_sample_num))
    print("# {} chars per second".format(total_token_num / (cost_time)))

    write_file_by_line(diff_results,
                       '/home/cuichengyu/dataset/diff_cws_samples.txt')


def _proc(seg, in_queue, out_queue):
    # TODO: load seg (json or pickle serialization) in sub_process
    #       to avoid pickle seg online when using start method other
    #       than fork
    count = 0
    while True:
        item = in_queue.get()
        if item is None:
            print('finished! Processed {} samples. '.format(count))
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


def _test_multi_proc(input_file,
                     nthread,
                     model_name="default_model",
                     user_dict=None,
                     pos=False):

    with TimeIt('# select method of starting subprocess'):
        alt = get_start_method() == "spawn"
        # spawn 启动方式是 windows 的默认方式，fork 为 unix 的默认启动方式

        if alt:
            seg = None
        else:
            seg = PredictText(config, model_name, user_dict, pos)

    if not os.path.exists(input_file):
        raise Exception("input_file {} does not exist.".format(input_file))

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

    print("# total sample:{:},\t average {:} chars per sample ".format(
        total_sample_num, total_token_num / total_sample_num))
    print("# {} chars per second".format(total_token_num / (cost_time)))


def test(input_file, model_name=None, user_dict=None, nthread=1, pos=False):
    """ 测试数据集预测效果，其中 nthread 可以使用 os.cpu_count() - 1 指定，来进行最大限度资源利用，
    主要用于测试处理速度 """

    if nthread > 1:
        _test_multi_proc(input_file, nthread, model_name, user_dict, pos)
    else:
        _test_single_proc(input_file, model_name, user_dict, pos)
