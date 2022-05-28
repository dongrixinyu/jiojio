# -*- coding=utf-8 -*-

import unittest

from test_cws_pure import TestCwsPure
from test_cws_rule import TestCwsRule
from test_pos_pure import TestPosPure
from test_pos_rule import TestPosRule


if __name__ == '__main__':

    suite = unittest.TestSuite()

    tests = [
        TestCwsPure('test_cws_pure'),
        TestCwsRule('test_cws_rule'),
        TestPosPure('test_pos_pure'),
        TestPosRule('test_pos_rule'),

    ]

    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=1)
    runner.run(suite)
