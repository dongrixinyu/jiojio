# -*- coding=utf-8 -*-

import unittest

from test_cws_pure import TestCwsPure
from test_cws_rule import TestCwsRule
from test_pos_pure import TestPosPure
from test_pos_rule import TestPosRule
from test_cws_add_word import TestCwsAddWord
from test_pos_add_word_pos import TestPOSAddWordPos


if __name__ == '__main__':

    suite = unittest.TestSuite()

    tests = [
        TestCwsPure('test_cws_pure'),
        TestCwsRule('test_cws_rule'),
        TestPosPure('test_pos_pure'),
        TestPosRule('test_pos_rule'),
        TestCwsAddWord('test_cws_add_word'),
        TestPOSAddWordPos('test_pos_add_word_pos'),
    ]

    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=1)
    runner.run(suite)
