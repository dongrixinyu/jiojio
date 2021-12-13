from __future__ import print_function
import sys

__doc__ = 'this is valid.'
import os
import pdb
import time
import pickle as pkl
import multiprocessing

from multiprocessing import Process, Queue

import jionlp as jio

from .logger import set_logger


logging = set_logger(level='INFO', log_dir_name='.jiojio_logs')


from .zip_file import unzip_file
from jiojio.tag_words_converter import tag2word
from jiojio.lexicon_cut import LexiconCut
from jiojio.pre_processor import PreProcessor
import jiojio.trainer as trainer
from jiojio.inference import decodeViterbi_fast

from jiojio.config import config
from jiojio.feature_extractor import FeatureExtractor
from jiojio.model import Model
from jiojio.postag import Postag


class TrieNode:
    """建立词典的 Trie 树节点"""
    def __init__(self, is_word):
        self.is_word = is_word
        self.user_tag = ''
        self.children = dict()


class jiojio(object):
    def __init__(self, model_name="default_model", user_dict="default", postag=False):
        """初始化函数，加载模型及用户词典"""
        self.postag = postag

        if model_name in ["default_model"]:
            config.model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                "models", model_name)
        else:
            config.model_dir = model_name
        # pdb.set_trace()
        self.feature_extractor = FeatureExtractor.load(model_dir=config.model_dir)
        self.model = Model.load()

        self.idx_to_tag = {
            idx: tag for tag, idx in self.feature_extractor.tag_to_idx.items()}

        self.pre_processor = PreProcessor()


        if postag:
            download_model(config.model_urls["postag"], config.jiojio_home)
            postag_dir = os.path.join(config.jiojio_home, "postag")
            self.tagger = Postag(postag_dir)

    def _cut(self, text):
        examples = self.pre_processor(text)
        length = len(examples)

        all_features = list()
        for idx in range(length):
            node_features = self.feature_extractor.get_node_features(idx, examples)

            # 此处考虑，通用未匹配特征 “/”，即索引未 0 的特征
            node_feature_idx = [self.feature_extractor.feature_to_idx[node_feature]
                                for node_feature in node_features
                                if node_feature in self.feature_extractor.feature_to_idx]

            if len(node_feature_idx) != len(node_features):
                node_feature_idx.append(0)

            # node_feature_idx = map(lambda i:self.feature_extractor.feature_to_idx.get(i, 0),
            #                        node_features)

            all_features.append(node_feature_idx)

        tags_idx = decodeViterbi_fast(all_features, self.model)

        # tags = [self.idx_to_tag[tag_idx] for tag_idx in tags_idx]
        tags = map(lambda i:self.idx_to_tag[i], tags_idx)

        return tags

    def cut(self, text, convert_num_letter=True, normalize_num_letter=True,
            convert_exception=True):

        if not text:
            return list()

        norm_text = self.pre_processor(
            text, convert_num_letter=convert_num_letter,
            normalize_num_letter=normalize_num_letter,
            convert_exception=convert_exception)

        tags = self._cut(norm_text)

        words_list = tag2word(text, tags)
        if self.postag:
            tags = self.tagger.tag(ret.copy())
            for i, user_tag in enumerate(user_tags):
                if user_tag:
                    tags[i] = user_tag
            ret = list(zip(ret, tags))

        return words_list


def train(train_file, test_file, save_dir, train_epoch=20, init_model=None):
    """用于训练模型"""
    # config = Config()
    starttime = time.time()
    if not os.path.exists(train_file):
        raise Exception("train_file does not exist.")
    if not os.path.exists(test_file):
        raise Exception("test_file does not exist.")
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

    config.train_file = train_file
    config.test_file = test_file
    config.train_dir = save_dir

    config.nThread = 1
    config.train_epoch = train_epoch
    config.init_model = init_model

    os.makedirs(config.train_dir, exist_ok=True)

    trainer.train(config)

    print("Total time: " + str(time.time() - starttime))


def _test_single_proc(input_file, output_file, model_name="default_model",
                      user_dict="default_dict", postag=False, verbose=False):

    start_time = time.time()
    with jio.TimeIt('# loading model'):
        seg = jiojio(model_name, user_dict, postag=postag)

    if not os.path.exists(input_file):
        raise Exception("input_file {} does not exist.".format(input_file))

    total_token_num = 0
    total_sample_num = 0
    with jio.TimeIt('# cutting text'):
        diff_results = list()
        for line in jio.read_file_by_iter(input_file, line_num=100000):
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
    print("# total_time:\t{:.3f}".format(time.time() - start_time))
    print("# total sample:{:},\t average {:} chars per sample ".format(
        total_sample_num, total_token_num / total_sample_num))
    print("# {} chars per second".format(total_token_num / (time.time() - start_time)))

    jio.write_file_by_line(diff_results, '/home/cuichengyu/dataset/diff_cws_samples.txt')


def _proc(seg, in_queue, out_queue):
    # TODO: load seg (json or pickle serialization) in sub_process
    #       to avoid pickle seg online when using start method other
    #       than fork
    while True:
        item = in_queue.get()
        if item is None:
            return
        idx, line = item
        if not seg.postag:
            output_str = " ".join(seg.cut(line))
        else:
            output_str = " ".join(map(lambda x: "/".join(x), seg.cut(line)))
        out_queue.put((idx, output_str))


def _proc_alt(model_name, user_dict, postag, in_queue, out_queue):
    seg = jiojio(model_name, user_dict, postag=postag)
    while True:
        item = in_queue.get()
        if item is None:
            return
        idx, line = item
        if not postag:
            output_str = " ".join(seg.cut(line))
        else:
            output_str = " ".join(map(lambda x: "/".join(x), seg.cut(line)))
        out_queue.put((idx, output_str))


def _test_multi_proc(input_file, nthread, model_name="default_model",
                     user_dict="default_dict", postag=False, verbose=False):

    alt = multiprocessing.get_start_method() == "spawn"

    times = []
    times.append(time.time())

    if alt:
        seg = None
    else:
        seg = jiojio(model_name, user_dict, postag)

    times.append(time.time())
    if not os.path.exists(input_file):
        raise Exception("input_file {} does not exist.".format(input_file))
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    times.append(time.time())
    in_queue = Queue()
    out_queue = Queue()
    procs = []
    for _ in range(nthread):
        if alt:
            p = Process(
                target=_proc_alt,
                args=(model_name, user_dict, postag, in_queue, out_queue),
            )
        else:
            p = Process(target=_proc, args=(seg, in_queue, out_queue))
        procs.append(p)

    for idx, line in enumerate(lines):
        in_queue.put((idx, line))

    for proc in procs:
        in_queue.put(None)
        proc.start()

    times.append(time.time())
    result = [None] * len(lines)
    for _ in result:
        idx, line = out_queue.get()
        result[idx] = line

    times.append(time.time())
    for p in procs:
        p.join()

    times.append(time.time())
    print("， ".join(result))
    times.append(time.time())

    print("total_time:\t{:.3f}".format(times[-1] - times[0]))

    if verbose:
        time_strs = [
            "load_model", "read_file", "start_proc", "word_seg", "join_proc", "write_file"]

        if alt:
            times = times[1:]
            time_strs = time_strs[1:]
            time_strs[2] = "load_modal & word_seg"

        for key, value in zip(
            time_strs,
            [end - start for start, end in zip(times[:-1], times[1:])],
        ):
            print("{}:\t{:.3f}".format(key, value))


def test(input_file, output_file, model_name="default_model", user_dict="default_dict",
         nthread=10, postag=False, verbose=False):
    nthread = 1
    if nthread > 1:
        _test_multi_proc(input_file, output_file, nthread,
                         model_name, user_dict, postag, verbose)
    else:
        _test_single_proc(input_file, output_file, model_name,
                          user_dict, postag, verbose)
