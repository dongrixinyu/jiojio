# -*- coding=utf-8 -*-

import pdb
import unittest

import jiojio


class TestPosPure(unittest.TestCase):
    """ 测试词性标注 POS 纯模型功能 """
    def test_pos_pure(self):

        text_list = [
            '西湖区蒋村花园小区管局农贸市场',
            '在这个课程里，在展示一个基础的 MPI Hello World 程序的同时我会介绍一下该如何运行 MPI 程序。'
        ]
        jiojio.init(pos=True)

        for text in text_list:
            words_tags = jiojio.cut(text)
            words = [item[0] for item in words_tags]
            tags = [item[1] for item in words_tags]
            print(text)
            self.assertEqual(text, ''.join(words))
            self.assertFalse('' in words)


if __name__ == '__main__':

    suite = unittest.TestSuite()
    test_pos_pure = [TestPosPure('test_pos_pure')]
    suite.addTests(test_pos_pure)

    runner = unittest.TextTestRunner(verbosity=1)
    runner.run(suite)
