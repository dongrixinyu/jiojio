# -*- coding=utf-8 -*-

import os
import unittest

import jiojio


JIOJIO_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestPOSAddWordPos(unittest.TestCase):
    """ 测试分词 POS 动态加载词典功能
    即指定 pos_user_dict 参数，得到正确的响应，有两种可选值
    """
    def test_pos_add_word_pos(self):

        text = '西湖区蒋村花园小区管局农贸市场'

        custom_word = '西湖区'
        custom_pos_type_default = 'ns'
        custom_pos_type_changed = 'v'

        # 测试 1：pos_user_dict = None，即参数不指定
        jiojio.init(pos=True)
        words_pos = jiojio.cut(text)
        self.assertTrue(words_pos[0] == (custom_word, custom_pos_type_default))

        # 测试 2：pos_user_dict = '/path/to/word_dict.txt'，词典被指定
        jiojio.init(
            pos=True,
            pos_user_dict=os.path.join(JIOJIO_DIR_PATH, 'example/pos_user_dict.txt'))

        words_pos = jiojio.cut(text)
        self.assertTrue(words_pos[0] == (custom_word, custom_pos_type_default))

        jiojio.add_word_pos(custom_word, custom_pos_type_changed)

        words_pos = jiojio.cut(text)
        self.assertTrue(words_pos[0] == (custom_word, custom_pos_type_changed))


if __name__ == '__main__':

    suite = unittest.TestSuite()
    test_pos_add_word_pos = [TestPOSAddWordPos('test_pos_add_word_pos')]
    suite.addTests(test_pos_add_word_pos)

    runner = unittest.TextTestRunner(verbosity=1)
    runner.run(suite)
