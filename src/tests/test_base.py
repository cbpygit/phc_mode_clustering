"""Basic unit tests for phc_mode_clustering.

Authors : Carlo Barth

"""

# coding: utf-8

import logging
import os
import sys

from collections import defaultdict
from copy import deepcopy
from datetime import date

# Manage path
PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                            os.pardir,
                                            os.pardir))
sys.path.insert(0, PROJECT_DIR)
from src.tools import tools
from src.in_out import in_out

MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
from shutil import rmtree
import unittest


# ==============================================================================
class TestDataLoading(unittest.TestCase):

    def setUp(self):
        tools.set_dummy_mode(True, False)

    def tearDown(self):
        tools.set_dummy_mode(False, False)

    def test_load_results(self):
        data = tools.get_results()
        self.assertItemsEqual(data.shape, (882, 67))

    def test_load_metadata(self):
        tools.get_metadata()

    def test_load_field_data(self):
        sim_data = tools.get_results()
        data = tools.load_field_data_for_sims_(sim_data.index, 0, 'electric')
        self.assertItemsEqual(data.shape, (882, 603))


if __name__ == '__main__':
    this_test = os.path.splitext(os.path.basename(__file__))[0]
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info('This is {}'.format(this_test))

    # list of all test suites
    suites = [
        unittest.TestLoader().loadTestsFromTestCase(TestDataLoading)]

    # Get a log file for the test output
    log_dir = os.path.abspath('logs')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    today_fmt = date.today().strftime("%y%m%d")
    test_log_file = os.path.join(log_dir, '{}_{}.log'.format(today_fmt,
                                                             this_test))
    logger.info('Writing test logs to: {}'.format(test_log_file))
    with open(test_log_file, 'w') as f:
        for suite in suites:
            unittest.TextTestRunner(f, verbosity=10).run(suite)
    with open(test_log_file, 'r') as f:
        content = f.read()
    logger.info('\n\nTest results:\n' + 80 * '=' + '\n' + content)
