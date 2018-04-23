# coding: utf-8

import logging
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.externals import joblib

# Manage path
PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                            os.pardir,
                                            os.pardir))
sys.path.insert(0, PROJECT_DIR)
from src.visualization import visualize as vis
from src.tools import tools

# Module constants
STD_METADATA_FILENAME = 'field_hex_metadata.npz'


def pickle_model(model, filepath):
    joblib.dump(model, filepath, compress=9)


def load_model(filepath):
    return joblib.load(filepath)


def makedirs(name, mode=511):
    """Calls os.makedirs in the same way, but only if the path
    does not already exist.

    """
    if not os.path.isdir(name):
        os.makedirs(name, mode=mode)


def save_plots_models(directory, sim_data_, model_data,
                      cluster_type, metadata, theta_split=None,
                      only_plots=False, overlay_kwargs=None,
                      field_plane_indices=None):
    logger = logging.getLogger(__name__)
    if field_plane_indices is None:
        field_plane_indices = [0, 3, 4]
    pointlist = metadata['pointlist']
    lengths = metadata['lengths']
    domain_ids = metadata['domain_ids']

    if overlay_kwargs is None:
        overlay_kwargs = dict(aspect='auto')

    # Create directories
    logger.info('Saving model data and plots...')
    if theta_split is None:
        subdir = os.path.join(directory, cluster_type)
    else:
        subdir = os.path.join(directory,
                              '{}_theta_split_{}'.format(cluster_type,
                                                         int(theta_split)))
    makedirs(subdir)

    # Iterate over direction, polarization
    n_clusters_list = 4 * [None]
    _sorting = {'Gamma-K': {'TE': 0, 'TM': 1}, 'Gamma-M': {'TE': 2, 'TM': 3}}
    for direc, poldict in model_data.iteritems():
        for pol, internals in poldict.iteritems():
            logger.info('Treating {} {}'.format(direc, pol))
            model = internals['model']
            n_clusters = len(model.cluster_centers_)
            n_clusters_list[_sorting[direc][pol]] = n_clusters
            sim_nums = internals['sim_nums']
            pol_suf = tools.DEFAULT_SIM_PDICT[pol]
            dpdir = os.path.join(subdir, '{}_{}_ncluster_{}'.
                                 format(direc, pol, n_clusters))
            makedirs(dpdir)

            # Pickle
            if not only_plots:
                logger.info('Saving pickles')
                pickle_model(model, os.path.join(dpdir, 'model.pkl'))
                pickle_model(sim_nums, os.path.join(dpdir, 'sim_nums.pkl'))

            # Plots ---------------------------------------------------------
            logger.info('Plotting...')
            data = sim_data_.loc[sim_nums]

            # Silhouettes
            fig = plt.figure(figsize=(4, 4))
            ax = plt.gca()
            vis.plot_silhouettes(sim_data_, pol_suf, sim_nums, ax=ax,
                                 n_clusters=n_clusters)
            comp_s = os.path.join(dpdir, 'silhouettes.png')
            fig.savefig(comp_s, bbox_inches='tight', dpi=300)
            del fig
            plt.clf()
            plt.close("all")

            # Comparison
            fig = plt.figure(1, figsize=(8, 3))
            vis.compare_values_and_classification(data, 'E', pol_suf, fig,
                                                  overlay_kwargs=overlay_kwargs,
                                                  interpolation='spline16')
            comp_f = os.path.join(dpdir, 'comparison.png')
            fig.savefig(comp_f, bbox_inches='tight', dpi=300)
            del fig
            plt.clf()
            plt.close("all")

            # Clustered mode profiles
            for plane_idx in field_plane_indices:
                fig = plt.figure(figsize=(2. * n_clusters, 4.))
                grid = ImageGrid(fig, 111,  # similar to subplot(111)
                                 nrows_ncols=(3, n_clusters),
                                 # creates 2x2 grid of axes
                                 axes_pad=0.05,  # pad between axes in inch.
                                 label_mode="L",
                                 cbar_mode='single',
                                 cbar_pad=0.1,
                                 cbar_size='5%',
                                 aspect='equal')

                for icl, mcc in enumerate(model.cluster_centers_):
                    dups = vis.DataOnPlanes(mcc, pointlist, lengths, domain_ids,
                                            name='H')
                    plane = dups.planes[plane_idx]
                    vmin, vmax = (plane.field.min(), plane.field.max())
                    for i, comp in enumerate(['x', 'y', 'z']):
                        _ax = grid[icl + n_clusters * i]
                        _, im = plane.imshow(comp, ax=_ax, vmin=vmin, vmax=vmax)
                        _ax.set_xticks(())
                        _ax.set_yticks(())
                        if icl == 0:
                            _ax.set_ylabel('$E_{}$'.format(['x', 'y', 'z'][i]))
                        if i == 0:
                            _ax.set_title('CC {}'.format(icl + 1))
                cb = plt.colorbar(im, cax=grid.cbar_axes[0])
                cb.set_ticks(())
                grid.cbar_axes[0].set_ylabel(
                    r'$|E|$-field strength (no absolute values)')
                plt.suptitle('Plane index {}'.format(plane_idx), y=1.05)
                cc_f = os.path.join(dpdir, 'centroids_on_plane_{}.png'.
                                    format(plane_idx))
                fig.savefig(cc_f, bbox_inches='tight', dpi=300)
                del fig
                plt.clf(), plt.cla()
                plt.close("all")

    if not only_plots:
        # Save classified simulation data
        # Excel
        fname_fmt = 'results_n_clusters_comb_' + \
                    '_'.join([str(i) for i in n_clusters_list])

        logger.info('Writing data to Excel')
        file_path = os.path.join(subdir, fname_fmt + '.xlsx')
        try:
            writer = pd.ExcelWriter(file_path)
            sim_data_.to_excel(writer, 'data')
            writer.save()
        except Exception as e:
            logger.warn('Unable to write Excel file. The error was: {}'.
                        format(str(e)))

        # Pickle
        logger.info('Writing data to pickle')
        sim_data_.to_pickle(os.path.join(subdir, fname_fmt + '.pkl'))

        # Copy metadata-file
        logger.info('Writing metadata to compressed numpy file.')
        metaf = os.path.join(subdir, STD_METADATA_FILENAME)
        np.savez_compressed(metaf, pointlist=pointlist, lengths=lengths,
                            domain_ids=domain_ids)
    logger.info('Finished')


def load_data(directory, cluster_type, n_clusters_dicts, suffix='',
              theta_split=None):
    logger = logging.getLogger(__name__)
    if theta_split is None:
        subdir = os.path.join(directory, cluster_type) + suffix
    else:
        raise NotImplementedError('Sorry :)')

    # Load metadata
    logger.info('Loading metadata...')
    _metadata = np.load(os.path.join(subdir, STD_METADATA_FILENAME))

    # Load pickled models
    logger.info('Loading model and sim_num data...')
    n_clusters_list = 4 * [None]
    _sorting = {'Gamma-K': {'TE': 0, 'TM': 1}, 'Gamma-M': {'TE': 2, 'TM': 3}}
    _model_data = defaultdict(dict)
    for direc, poldict in n_clusters_dicts.iteritems():
        for pol, n_clusters in poldict.iteritems():
            n_clusters_list[_sorting[direc][pol]] = n_clusters
            dpdir = os.path.join(subdir, '{}_{}_ncluster_{}'.
                                 format(direc, pol, n_clusters))
            _model_data[direc][pol] = {}
            _model_data[direc][pol]['model'] = load_model(
                os.path.join(dpdir, 'model.pkl'))
            _model_data[direc][pol]['sim_nums'] = load_model(
                os.path.join(dpdir, 'sim_nums.pkl'))

    # Load result file
    logger.info('Loading result data...')
    fname_fmt = 'results_n_clusters_comb_' + \
                '_'.join([str(i) for i in n_clusters_list])
    _sim_data = pd.read_pickle(os.path.join(subdir, fname_fmt + '.pkl'))
    logger.info('Finished')

    return _sim_data, _model_data, _metadata


if __name__ == '__main__':
    pass
