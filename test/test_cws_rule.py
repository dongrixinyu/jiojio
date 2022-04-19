# -*- coding=utf-8 -*-

import unittest

import jiojio


class TestCwsRule(unittest.TestCase):
    """ 测试分词 CWS 模型加规则抽取功能 """
    def test_cws_rule(self):

        text_list = [
            ['电话010-23437524，http://baidu.com/f43rhi然后这邮箱dongrixin.89@163.com 124335432@qq.com。',
             ['010-23437524', 'http://baidu.com/f43rhi', 'dongrixin.89@163.com', '124335432@qq.com']]

        ]
        jiojio.init(cws_rule=True)

        for text, items in text_list:
            words = jiojio.cut(text)
            self.assertEqual(text, ''.join(words))
            self.assertFalse('' in words)

            for item in items:
                self.assertTrue(item in words)


if __name__ == '__main__':

    suite = unittest.TestSuite()
    test_cws_rule = [TestCwsRule('test_cws_rule')]
    suite.addTests(test_cws_rule)

    runner = unittest.TextTestRunner(verbosity=1)
    runner.run(suite)
