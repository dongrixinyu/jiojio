# -*- coding=utf-8 -*-

import os
import unittest

import jiojio


JIOJIO_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestCwsAddWord(unittest.TestCase):
    """ 测试分词 CWS 动态加载词典功能
    即指定 cws_user_dict 参数，得到正确的响应，有三种可选值
    """
    def test_cws_add_word(self):

        text = '西湖区蒋村花园小区管局农贸市场'

        custom_word = '西湖区蒋村'
        # 测试 1：cws_user_dict = None，即参数不指定
        jiojio.init()
        words = jiojio.cut(text)
        self.assertFalse(custom_word in words)

        # 测试 2：cws_user_dict = True， 即参数仅指定为 True 且 trie_tree_dict 为空
        jiojio.init(cws_user_dict=True)

        words = jiojio.cut(text)
        self.assertFalse(custom_word in words)

        jiojio.add_word(custom_word, 5)

        words = jiojio.cut(text)
        self.assertTrue(custom_word in words)

        # 测试 3：cws_user_dict = '/path/to/word_dict.txt'，词典被指定
        jiojio.init(cws_user_dict=os.path.join(JIOJIO_DIR_PATH, 'example/cws_user_dict.txt'))

        words = jiojio.cut(text)
        self.assertFalse(custom_word in words)

        jiojio.add_word(custom_word, 5)

        words = jiojio.cut(text)
        self.assertTrue(custom_word in words)


if __name__ == '__main__':

    suite = unittest.TestSuite()
    test_cws_add_word = [TestCwsAddWord('test_cws_add_word')]
    suite.addTests(test_cws_add_word)

    runner = unittest.TextTestRunner(verbosity=1)
    runner.run(suite)
