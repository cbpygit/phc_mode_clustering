# coding: utf-8

import logging
import os

from collections import defaultdict
from copy import deepcopy

from tools import tools
from in_out import in_out

MODEL_DIR = os.path.join(tools.PROJECT_DIR, 'models')

def test_GMM(every):
    logger = logging.getLogger(__name__)
    logger.info('Loading simulation data, using every {}'.format(every) +
                ' angles and wavelength.')
    sim_data_init = tools.get_results(every, every)
    logger.info('Total number of rows is {}'.format(len(sim_data_init)))

    # Define init parameters
    ddict = tools.DEFAULT_SIM_DDICT
    pdict = tools.DEFAULT_SIM_PDICT
    direc = 'Gamma-M'
    pol = 'TE'
    pol_suf = pdict[pol]
    theta_split = None
    field_type = 'electric'
    cluster_type = 'GaussianMixture'

    # Gaussian Mixture settings -> every = 5 works
    common_clkws = dict(covariance_type='tied',
                        max_iter=20,
                        treat_complex=None,
                        preprocess='normalize',
                        random_state=0)  # <- for reproducibility

    clkw_dicts = defaultdict(dict)
    for direc_ in ddict:
        for pol_ in pdict:
            clkw_dicts[direc_][pol_] = deepcopy(common_clkws)

    # Individual settings
    clkw_dicts['Gamma-K']['TE']['n_components'] = 8
    clkw_dicts['Gamma-K']['TM']['n_components'] = 8  # maybe 8, before: 7
    clkw_dicts['Gamma-M']['TE']['n_components'] = 7  # maybe 8, before: 7
    clkw_dicts['Gamma-M']['TM']['n_components'] = 7
    clkw_dicts = dict(clkw_dicts)

    # Clustering
    sim_data, model_data = tools.cluster_all_modes(sim_data_init,
                                             theta_split=theta_split,
                                             cluster_type=cluster_type,
                                             cluster_kwargs_dicts=clkw_dicts,
                                             # pols=[pol],
                                             # direcs=[direc],
                                             field_type=field_type)

    lengths, pointlist, domain_ids = tools.get_metadata()
    metadata = dict(lengths=lengths, pointlist=pointlist, domain_ids=domain_ids)
    in_out.save_plots_models(MODEL_DIR, sim_data,
                             model_data, cluster_type, metadata)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)
    test_GMM(20)