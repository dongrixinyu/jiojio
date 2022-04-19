# -*- coding=utf-8 -*-

import unittest

import jiojio


class TestCwsPure(unittest.TestCase):
    """ 测试分词 CWS 纯模型功能 """
    def test_cws_pure(self):

        text_list = [
            '西湖区蒋村花园小区管局农贸市场',
            '在这个课程里，在展示一个基础的 MPI Hello World 程序的同时我会介绍一下该如何运行 MPI 程序。'
        ]
        jiojio.init()

        for text in text_list:
            words = jiojio.cut(text)

            self.assertEqual(text, ''.join(words))
            self.assertFalse('' in words)


if __name__ == '__main__':

    suite = unittest.TestSuite()
    test_cws_pure = [TestCwsPure('test_cws_pure')]
    suite.addTests(test_cws_pure)

    runner = unittest.TextTestRunner(verbosity=1)
    runner.run(suite)
