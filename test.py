# -*- coding=utf-8 -*-
# Library: jiojio
# Author: dongrixinyu
# License: GPL-3.0
# Email: dongrixinyu.89@163.com
# Github: https://github.com/dongrixinyu/jiojio
# Description: fast Chinese Word Segmentation(CWS) and Part of Speech(POS) based on CPU.'
# Website: http://www.jionlp.com

# this file is provided for testing the processing speed of jiojio
# compared with several other CWS and POS tools. And also with
# jiojio itself of different parameters.

import os
import pdb
import sys
import jionlp as jio
import json
import argparse
import jieba
import cProfile
import jiojio
# from LAC import LAC


def test(args):

    # text = '电话010-23437524，http://baidu.com/f43rhi然后这邮箱dongrixin.89@163.com 124335432@qq.com。'
    # text = '举一个我面试中遇到过的目前最牛的例子，简称M。我一般负责和人吃饭闲聊。这位先生以前在Intel，只大概看过机器学习的一些东西'

    # text = '平台曝光平台名称:    握握金服平台网址:   http://www.wowodai.cn曝光原因:  标的无协议投资 握握金服 标的无协议，问客服说老的协议下线了，新的协议审核中，不知什么时候上线， 借款 方信息非常模糊，无处可查。'
    # # jieba.lcut(text)
    # pkuseg_obj = spacy_pkuseg.pkuseg(postag=True)
    # lac_obj = LAC(mode='seg')

    if args.line_num == 0:
        line_num = None
    else:
        line_num = args.line_num

    # file_path = '/home/ubuntu/datasets/test_cctv.txt'
    with jio.TimeIt('empty', no_print=True) as ti:
        char_num = 0

        for _ in range(args.times):
            for idx, text in enumerate(jio.read_file_by_iter(args.file_path, line_num=line_num)):
                if type(text) is not str:
                    continue
                char_num += len(text)

                if args.check:
                    pass
                if args.print:
                    pass

        empty_time = ti.break_point()

    cws_rule = args.rule
    pos = args.pos
    if cws_rule:
        if pos:
            pos_rule = True
        else:
            pos_rule = False
    else:
        pos_rule = False

    if args.dict:
        cws_user_dict = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'example/cws_user_dict.txt')
        pos_user_dict = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'example/pos_user_dict.txt')
    else:
        cws_user_dict = None
        pos_user_dict = None
    print('args: ', args)

    jiojio.init(cws_rule=cws_rule, pos_rule=pos_rule, pos=pos,
                cws_user_dict=cws_user_dict,
                pos_user_dict=pos_user_dict)
    with jio.TimeIt('with params for jiojio') as ti:
        char_num = 0
        for _ in range(args.times):
            for idx, text in enumerate(jio.read_file_by_iter(args.file_path, line_num=line_num)):
                if type(text) is not str:
                    continue

                char_num += len(text)
                jio_res = jiojio.cut(text)

                if args.print:
                    print(jio_res)
                if args.check:
                    if pos:
                        words_res = [i[0] for i in jio_res]
                        assert len(''.join(words_res)) == len(text)
                        assert '' not in words_res
                    else:
                        assert len(''.join(jio_res)) == len(text)
                        assert '' not in jio_res

            # res = jieba.lcut(text, HMM=True)
            # print(''.join(jio_res))
            # pku_res = pkuseg_obj.cut(text)
            # print('pkuseg: ', '\t'.join([item[0] + '/' + item[1] for item in pku_res]))
            # res = lac_obj.run(text)
            # pdb.set_trace()
        with_params_time = ti.break_point()

    jiojio.init()
    with jio.TimeIt('base for jiojio') as ti:
        char_num = 0

        for _ in range(args.times):
            for idx, text in enumerate(jio.read_file_by_iter(args.file_path, line_num=line_num)):
                if type(text) is not str:
                    continue

                char_num += len(text)
                res = jiojio.cut(text)
                # res = jieba.lcut(text, HMM=False)
                # res = pkuseg_obj.cut(text)
                # res = lac_obj.run(text)
                if args.print:
                    print(res)
                if args.check:
                    assert len(''.join(res)) == len(text)
                    assert '' not in res

        base_version_time = ti.break_point()

    print('# speed of base  jiojio.init(): {:.2f} chars/second.'.format(char_num / (base_version_time - empty_time)))
    print('# speed of param jiojio.init(\n\tpos={}, cws_rule={}, pos_rule={}, \n\t'
          'cws_user_dict={}, \n\tpos_user_dict={}): \n\t{:.2f} chars/second.'.format(
              pos, cws_rule, pos_rule, cws_user_dict, pos_user_dict,
              char_num / (with_params_time - empty_time)))
    print('# ratio: {:.2%}'.format((base_version_time - empty_time) / (with_params_time - empty_time)))

    if args.pkuseg:
        import pkuseg

        pkuseg_obj = pkuseg.pkuseg(postag=pos)
        with jio.TimeIt('pkuseg: ', no_print=True) as ti:
            char_num = 0

            for _ in range(args.times):
                for idx, text in enumerate(jio.read_file_by_iter(args.file_path, line_num=line_num)):
                    if type(text) is not str:
                        continue

                    char_num += len(text)
                    # res = jiojio.cut(text)
                    # res = jieba.lcut(text, HMM=False)
                    res = pkuseg_obj.cut(text)
                    # res = lac_obj.run(text)
                    if args.print:
                        print(res)
                    if args.check:
                        assert len(''.join(res)) == len(text)
                        assert '' not in res

            pkuseg_time = ti.break_point()

        print('# speed of param spacy_pkuseg.pkuseg(postag={}): \n\t{:.2f} chars/second.'.format(
                  pos, char_num / (pkuseg_time - empty_time)))

    if args.jieba:
        import jieba

        jieba.lcut(text, HMM=True)
        with jio.TimeIt('jieba: ', no_print=True) as ti:
            char_num = 0

            for _ in range(args.times):
                for idx, text in enumerate(jio.read_file_by_iter(args.file_path, line_num=line_num)):
                    if type(text) is not str:
                        continue

                    char_num += len(text)
                    # res = jiojio.cut(text)
                    # res = jieba.lcut(text, HMM=False)
                    res = jieba.lcut(text, HMM=True)
                    # res = lac_obj.run(text)
                    if args.print:
                        print(res)
                    if args.check:
                        assert len(''.join(res)) == len(text)
                        assert '' not in res

            jieba_time = ti.break_point()

        print('# speed of param jieba.lcut(text, HMM=True): \n\t{:.2f} chars/second.'.format(
                  char_num / (jieba_time - empty_time)))


if __name__ == '__main__':
    # import spacy_pkuseg

    # pkuseg_obj = spacy_pkuseg.pkuseg(postag=True)
    jiojio_parser = argparse.ArgumentParser(description='test speed of jiojio.')
    jiojio_parser.add_argument(
        '-t', '--times', type=int, default=1, help='how many times to process the given dataset.')
    jiojio_parser.add_argument(
        '-f', '--file_path', type=str, required=True, help='the text file to process.')
    jiojio_parser.add_argument(
        '-p', '--pos', action='store_true', help='the text file to process.')
    jiojio_parser.add_argument(
        '-l', '--line_num', type=int, default=0, help='how many lines to read from text file to process.')
    jiojio_parser.add_argument(
        '-r', '--rule', action='store_true', help='whether to use cws rule and pos_rule.')
    jiojio_parser.add_argument(
        '-c', '--check', action='store_true', help='whether to check if the results are correct.')
    jiojio_parser.add_argument(
        '--print', '--print', action='store_true', help='whether to print the results.')
    jiojio_parser.add_argument(
        '-d', '--dict', action='store_true', help='predict with designated dictionary.')
    jiojio_parser.add_argument(
        '--pkuseg', '--pkuseg', action='store_true', help='test the speed of pkuseg.')
    jiojio_parser.add_argument(
        '--jieba', '--jieba', action='store_true', help='test the speed of jieba.')
    jiojio_parser.add_argument(
        '--lac', '--lac', action='store_true', help='test the speed of LAC.')

    jiojio_args = jiojio_parser.parse_args()

    test(jiojio_args)

# cProfile.run('test()', sort='cumtime')
