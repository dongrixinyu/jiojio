# -*- coding=utf-8 -*-

import unittest

import jiojio


class TestPosRule(unittest.TestCase):
    """ 测试词性标注 POS 模型加规则抽取功能 """
    def test_pos_rule(self):

        text_list = [
            ['电话010-23437524，http://baidu.com/f43rhi然后这邮箱dongrixin.89@163.com 124335432@qq.com。',
             ['010-23437524', 'http://baidu.com/f43rhi', 'dongrixin.89@163.com', '124335432@qq.com']],
            ['你的邮箱多少？是电话号码？17358560923@gmail.com。记住了吗？',
             ['17358560923@gmail.com']],  # 电话号码与邮箱的混合
        ]
        jiojio.init(pos_rule=True, pos=True)

        for text, items in text_list:
            words_tags = jiojio.cut(text)
            words = [item[0] for item in words_tags]
            tags = [item[1] for item in words_tags]
            print(text)
            self.assertEqual(text, ''.join(words))
            self.assertFalse('' in words)

            for item in items:
                self.assertTrue(item in words)


if __name__ == '__main__':

    suite = unittest.TestSuite()
    test_pos_rule = [TestPosRule('test_pos_rule')]
    suite.addTests(test_pos_rule)

    runner = unittest.TextTestRunner(verbosity=1)
    runner.run(suite)
