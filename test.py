# -*- coding=utf-8 -*-
# this file can NOT be excuted directly! need to be checked!
import os
import pdb
import sys
import zipfile
import jionlp as jio
import json
import jieba
import spacy_pkuseg
import requests
import numpy as np
import cProfile
import jiojio
from LAC import LAC

from scipy.special import comb, perm


# sys.exit()

def test():
    times = 1000000

    text = '电话010-23437524，http://baidu.com/f43rhi然后这邮箱dongrixin.89@163.com 124335432@qq.com。'
    text = '举一个我面试中遇到过的目前最牛的例子，简称M。我一般负责和人吃饭闲聊。这位先生以前在Intel，只大概看过机器学习的一些东西'

    text = '平台曝光平台名称:      握握金服平台网址:       http://www.wowodai.cn曝光原因:  标的无协议投资 握握金服 标的无协议，问客服说老的协议下线了，新的协议审核中，不知什么时候上线， 借款 方信息非常模糊，无处可查。'
    # jieba.lcut(text)
    pkuseg_obj = spacy_pkuseg.pkuseg(postag=True)
    # lac_obj = LAC(mode='seg')

    file_path = '/home/cyy/datasets/cws/test_cws.txt'
    with jio.TimeIt('empty') as ti:
        for idx, line in enumerate(jio.read_file_by_iter(file_path, line_num=times)):
            text = ''.join(line)
        empty_time = ti.break_point()

    jiojio.init(cws_rule=True, pos_rule=True, pos=True,
                cws_user_dict='/home/cyy/github/jiojio/example/cws_user_dict.txt',
                pos_user_dict='/home/cyy/github/jiojio/example/pos_user_dict.txt')#, pos_model_dir='test_pos_model')
    with jio.TimeIt('with rule') as ti:
        for idx, line in enumerate(jio.read_file_by_iter(file_path, line_num=times)):
            text = ''.join(line)
            # text = '平台曝光平台名称:      握握金服平台网址:       http://www.wowodai.cn曝光原因:  标的无协议投资 握握金服 标的无协议，问客服说老的协议下线了，新的协议审核中，不知什么时候上线， 借款 方信息非常模糊，无处可查。'

            # print('#', text)
            jio_res = jiojio.cut(text)

            assert len(''.join(jio_res[0])) == len(text)
            assert '' not in jio_res[0]

            print('jiojio: ', '\t'.join([i1 + '/' + i2 for i1, i2 in zip(jio_res[0], jio_res[1])]))

            # res = jieba.lcut(text, HMM=True)
            # print(''.join(jio_res))
            pku_res = pkuseg_obj.cut(text)
            print('pkuseg: ', '\t'.join([item[0] + '/' + item[1] for item in pku_res]))
            # res = lac_obj.run(text)
            pdb.set_trace()
        with_rule_time = ti.break_point()

    jiojio.init(cws_rule=False)
    with jio.TimeIt('without rule') as ti:
        for idx, line in enumerate(jio.read_file_by_iter(file_path, line_num=times)):
            text = ''.join(line)
            res = jiojio.cut(text)
            # res = jieba.lcut(text, HMM=False)
            # res = pkuseg_obj.cut(text)
            # res = lac_obj.run(text)
            assert len(''.join(res)) == len(text)
            assert '' not in res

        without_rule_time = ti.break_point()
    print(res)

    print('speed without rule:', len(text) * times / (without_rule_time - empty_time))
    print('speed with rule:', len(text) * times / (with_rule_time - empty_time))
    print('ratio: ', (without_rule_time - empty_time) / (with_rule_time - empty_time))

    print(text)
    # print(''.join(res))
    assert ''.join(res) == text

test()
# cProfile.run('test()', sort='cumtime')
