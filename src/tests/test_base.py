"""Basic unit tests for phc_mode_clustering.

Authors : Carlo Barth

"""

# coding: utf-8

import logging
import os
import sys

from copy import deepcopy
from datetime import date

# Manage path
PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                            os.pardir,
                                            os.pardir))
sys.path.insert(0, PROJECT_DIR)
from src.tools import tools

MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
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
        for ipol in [0, 1]:
            for ftype in ['electric', 'magnetic']:
                data = tools.load_field_data_for_sims(sim_data.index, ipol,
                                                      ftype)
                self.assertItemsEqual(data.shape, (882, 603))

    def test_cache_usage(self):
        sim_data = tools.get_results()
        tools.set_cache(True)
        tools.load_field_data_for_sims(sim_data.index, 0, 'electric')
        tools.load_field_data_for_sims(sim_data.index, 0, 'electric')
        tools.set_cache(False)


# ==============================================================================
class TestClustering(unittest.TestCase):

    def setUp(self):
        tools.set_dummy_mode(True, False)

    def tearDown(self):
        tools.set_dummy_mode(False, False)

    def test_MiniBatchKMeans(self):
        sim_data_init = tools.get_results()

        # Define init parameters
        ddict = tools.DEFAULT_SIM_DDICT
        pdict = tools.DEFAULT_SIM_PDICT
        field_type = 'electric'
        cluster_type = 'MiniBatchKMeans'

        # Type `tools.get_clustering_input_data?` for info on these parameters
        common_clkws = dict(treat_complex=None,
                            preprocess='normalize',
                            random_state=0)  # <- for reproducibility

        # Set these defaults for all combinations of direction and polarization
        clkw_dicts = tools.defaultdict(dict)
        for direc_ in ddict:
            for pol_ in pdict:
                clkw_dicts[direc_][pol_] = deepcopy(common_clkws)

        # Set individual parameters, in this case for the number of clusters
        clkw_dicts['Gamma-K']['TE']['n_clusters'] = 8
        clkw_dicts['Gamma-K']['TM']['n_clusters'] = 8
        clkw_dicts['Gamma-M']['TE']['n_clusters'] = 7
        clkw_dicts['Gamma-M']['TM']['n_clusters'] = 7
        clkw_dicts = dict(clkw_dicts)

        tools.cluster_all_modes(sim_data_init, cluster_type=cluster_type,
                                cluster_kwargs_dicts=clkw_dicts,
                                field_type=field_type)

    def test_GMM(self):
        sim_data_init = tools.get_results()

        # Define init parameters
        ddict = tools.DEFAULT_SIM_DDICT
        pdict = tools.DEFAULT_SIM_PDICT
        field_type = 'electric'
        cluster_type = 'GaussianMixture'

        # Type `tools.get_clustering_input_data?` for info on these parameters
        common_clkws = dict(covariance_type='tied',
                            max_iter=20,
                            treat_complex=None,
                            preprocess='normalize',
                            random_state=0)  # <- for reproducibility

        # Set these defaults for all combinations of direction and polarization
        clkw_dicts = tools.defaultdict(dict)
        for direc_ in ddict:
            for pol_ in pdict:
                clkw_dicts[direc_][pol_] = deepcopy(common_clkws)

        # Set individual parameters, in this case for the number of clusters
        clkw_dicts['Gamma-K']['TE']['n_components'] = 8
        clkw_dicts['Gamma-K']['TM']['n_components'] = 8  # maybe 8, before: 7
        clkw_dicts['Gamma-M']['TE']['n_components'] = 7  # maybe 8, before: 7
        clkw_dicts['Gamma-M']['TM']['n_components'] = 7
        clkw_dicts = dict(clkw_dicts)

        tools.cluster_all_modes(sim_data_init, cluster_type=cluster_type,
                                cluster_kwargs_dicts=clkw_dicts,
                                field_type=field_type)


if __name__ == '__main__':
    this_test = os.path.splitext(os.path.basename(__file__))[0]
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info('This is {}'.format(this_test))

    # list of all test suites
    suites = [
        unittest.TestLoader().loadTestsFromTestCase(TestDataLoading),
        unittest.TestLoader().loadTestsFromTestCase(TestClustering)]

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
