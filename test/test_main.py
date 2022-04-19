
import unittest

from test_cws_pure import TestCwsPure
from test_cws_rule import TestCwsRule


if __name__ == '__main__':

    suite = unittest.TestSuite()

    tests = [
        TestCwsPure('test_cws_pure'),
        TestCwsRule('test_cws_rule'),

    ]

    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=1)
    runner.run(suite)
