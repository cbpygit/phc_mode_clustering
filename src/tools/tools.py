# coding: utf-8

import logging
import os
import time
import warnings

from collections import defaultdict
from copy import deepcopy
from datetime import timedelta

import numpy as np
import pandas as pd

from scipy.linalg import expm, norm
from typing import Any, Union

from silhouette_implementations import silhouette_samples_block
from sklearn import cluster, mixture, preprocessing
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.externals import joblib

# Global constants
DEFAULT_SIM_DDICT = {'Gamma-K': 0., 'Gamma-M': 90.}
DEFAULT_SIM_PDICT = {'TM': '_2', 'TE': '_1'}
POLS = ['TE', 'TM']
FIELDS = dict(electric='E', magnetic='H')

# Paths and file names
PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                            os.pardir,
                                            os.pardir))
RAW_DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'raw')
RAW_DATA_DIR_DUMMY = os.path.join(PROJECT_DIR, 'data', 'raw_dummy')
DATA_FILE_NAMES = dict(results='parameters_and_results.h5',
                       fields='field_data_{}_{}.h5')

# Helper to switch between real and dummy data
_USE_DUMMIES = {'do': False}

# A global joblib cache to persist output of functions.
MEMORY = joblib.Memory(cachedir=os.path.abspath('cache'), verbose=0)


def format_time(t):
    """Returns a well formatted time string."""
    return str(timedelta(seconds=t))


def set_dummy_mode(use_dummies, verbose=True):
    """
    Convenience function to toggle dummy mode.

    Parameters
    ----------
    use_dummies : bool
        Whether to use dummy data.

    """
    logger = logging.getLogger(__name__)
    if verbose:
        if use_dummies:
            logger.info('Switched to dummy data')
        else:
            logger.info('Switched to real mode')
    _USE_DUMMIES['do'] = use_dummies


def get_data_file_path(kind='fields', field_comp='E', polarization='TE'):
    """ Returns the full path to the data file of `kind`, as well as field
    component `field_comp` and `polarization` in case of field data.

    Parameters
    ----------
    kind : {'fields' | 'results'}
        Short-hand key for the data file, i.e. 'fields':'field_data.h5' and
        'results':'parameters_and_results.h5'.
    field_type : {'E' | 'H'}, ignored if kind=='results'
        Field component, 'E' for electric or 'H' for magnetic field.
    polarization : {'TE' | 'TM'}, ignored if kind=='results'
        Polarization, 'TE' for transversal electric or 'TM' for transversal
        magnetic.
    """
    fname = DATA_FILE_NAMES[kind]
    if kind == 'fields':
        fname = fname.format(field_comp[0], polarization)

    data_folder = RAW_DATA_DIR_DUMMY if _USE_DUMMIES['do'] else RAW_DATA_DIR
    path = os.path.join(data_folder, fname)
    if not os.path.exists(path):
        raise EnvironmentError('The database file "{}" '.
                               format(fname) +
                               'is missing in your data/raw folder. You may' +
                               ' want to run the "src/data/make_dataset.py" ' +
                               'script to download and verify the necessary' +
                               ' raw data. Leaving.')
    return path


def get_results(theta_every=1, wvl_every=1):
    """Returns the complete result data frame. An incident angle correction
    which is needed for this particular data set is carried out automatically
    and a column 'wavelength' is added that holds the vacuum wavelength in
    nanometers. Use `theta_every` and `wvl_every` to get reduced data sets.

    Parameters
    ----------
    theta_every : int, default 1
        Reduction factor for the incident angle theta. If >1, only every
        `theta_every`th angle is used.
    wvl_every : int, default 1
        Reduction factor for the wavelengths. If >1, only every
        `wvl_every`th wavelength is used.
    Returns
    -------
    df : pandas.DataFrame

    """
    logger = logging.getLogger(__name__)
    logger.info('Loading parameter and result data frame')
    df = pd.read_hdf(get_data_file_path('results'), 'data')
    df = fix_angle_in_scattering_dset(df)

    theta_unique = sorted(df.theta.unique())
    wvl_unique = sorted(df.vacuum_wavelength.unique())

    theta_reduced = theta_unique[::theta_every]
    wvl_reduced = wvl_unique[::wvl_every]

    df = df[df['theta'].isin(theta_reduced)]
    df = df[df['vacuum_wavelength'].isin(wvl_reduced)]

    # Add a wavelength in nm column
    df['wavelength'] = df['vacuum_wavelength'] * 1.e9
    return df


def get_metadata():
    """Reads the metadata needed to restructure the flat field data from the
    field HDF5 database.

    Returns
    -------
    lengths, point_list, domain_ids : tuple of numpy.ndarrays

    """
    logger = logging.getLogger(__name__)
    logger.info('Reading metadata')
    h5 = get_data_file_path()
    df_points = pd.read_hdf(h5, 'points')
    df_lengths = pd.read_hdf(h5, 'lengths')
    return (df_lengths['length'].values,
            df_points.loc[:, ('x', 'y', 'z')].values,
            df_points['domain_ids'].values)


def refracted_angles(angles, refractive_index, reverse=False):
    """Returns angles after refraction on an air-medium interface, where the
    medium has a refractive index of `refractive_index`."""
    if reverse:
        refractive_index = 1. / refractive_index
    return np.rad2deg(np.arcsin(np.sin(np.deg2rad(angles)) / refractive_index))


def fix_angle_in_scattering_dset(dset, angle_col_name='theta'):
    """Undos the refraction correction for the angle column in a
    pandas.DataFrame."""
    d_ = dset.copy(deep=True)
    n_ = d_.mat_sup_n.unique()[0]
    d_.loc[:, angle_col_name] = refracted_angles(d_[angle_col_name], n_,
                                                 reverse=True)
    return d_.dropna(subset=[angle_col_name])


def rot_mat(axis, theta):
    """Returns a matrix that rotates by angle `theta` around `axis`."""
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def rotate_vector(v, axis, theta):
    """Rotates a vector `v` by angle `theta` around `axis`."""
    matrix = rot_mat(axis, theta)
    return np.tensordot(matrix, v, axes=([0], [1])).T  # np.dot(M, v)


def rotate_around_z(v, theta):
    """Rotates a vector `v` by angle `theta` around the z axis."""
    return rotate_vector(v, np.array([0., 0., 1.]), theta)


def is_odd(num):
    """Returns whether `num` is odd."""
    return num & 0x1


def is_inside_hexagon(x, y, d=None, x0=0., y0=0.):
    p_eps = 10. * np.finfo(float).eps
    if d is None:
        d = y.max() - y.min() + p_eps
    dx = np.abs(x - x0) / d
    dy = np.abs(y - y0) / d
    a = 0.25 * np.sqrt(3.0)
    return np.logical_and(dx <= a, a * dy + 0.25 * dx <= 0.5 * a)


def get_hex_plane(plane_idx, inradius, z_height, z_center, np_xy,
                  np_z):
    # We use 10* float machine precision to correct the ccordinates
    # to avoid leaving the computational domain due to precision
    # problems
    p_eps = 10. * np.finfo(float).eps

    ri = inradius  # short for inradius
    rc = inradius / np.sqrt(3.) * 2.  # short for circumradius

    if np_z == 'auto':
        np_z = int(np.round(float(np_xy) / 2. / rc * z_height))

    # x_y-plane (no hexagonal shape!)
    if plane_idx == 6:
        x = np.linspace(-ri + p_eps, ri - p_eps, np_xy)
        y = np.linspace(-rc + p_eps, rc - p_eps, np_xy)
        x_y = np.meshgrid(x, y)
        x_y_rs = np.concatenate((x_y[0][..., np.newaxis],
                                 x_y[1][..., np.newaxis]),
                                axis=2)
        # x_y_rs = np.concatenate((np.expand_dims(x_y[0], axis=-1),
        #                          np.expand_dims(x_y[1], axis=-1)),
        #                         axis=2)
        z = np.ones((np_xy, np_xy, 1)) * z_center
        pl = np.concatenate((x_y_rs, z), axis=2)
        pl = pl.reshape(-1, pl.shape[-1])

        # Restrict to hexagon
        idx_hex = is_inside_hexagon(pl[:, 0], pl[:, 1])
        return pl[idx_hex]

    # Vertical planes
    elif plane_idx < 6:
        r = rc if is_odd(plane_idx) else ri
        r = r - p_eps
        xy_line = np.empty((np_xy, 2))
        xy_line[:, 0] = np.linspace(-r, r, np_xy)
        xy_line[:, 1] = 0.
        z_points = np.linspace(0. + p_eps, z_height - p_eps, np_z)

        # Construct the plane
        plane = np.empty((np_xy * np_z, 3))
        for i, xy in enumerate(xy_line):
            for j, z in enumerate(z_points):
                idx = i * np_z + j
                plane[idx, :2] = xy
                plane[idx, 2] = z

        # Rotate the plane
        return rotate_around_z(plane, plane_idx * np.pi / 6.)
    else:
        raise ValueError('`plane_idx` must be in [0...6].')


def get_hex_planes_point_list(inradius, z_height, z_center, np_xy, np_z,
                              plane_indices=None):
    # Construct the desired planes
    if plane_indices is None:
        plane_indices = [0, 1, 2, 3, 6]
    planes = []
    for i in plane_indices:
        planes.append(get_hex_plane(i, inradius, z_height, z_center,
                                    np_xy, np_z))

    # Flatten and save lengths
    lengths = [len(p) for p in planes]
    return np.vstack(planes), np.array(lengths)


def hex_planes_point_list_for_keys(keys, plane_indices=None):
    if plane_indices is None:
        plane_indices = [0, 1, 2, 3, 6]
    if 'uol' not in keys:
        keys['uol'] = 1.e-9
    inradius = keys['p'] * keys['uol'] / 2.
    z_height = (keys['h'] + keys['h_sub'] + keys['h_sup']) * keys['uol']
    z_center = (keys['h_sub'] + keys['h'] / 2.) * keys['uol']
    np_xy = keys['hex_np_xy']
    if 'hex_np_z' in keys:
        np_z = keys['hex_np_z']
    else:
        np_z = 'auto'
    return get_hex_planes_point_list(inradius, z_height, z_center, np_xy, np_z,
                                     plane_indices=plane_indices)


def plane_idx_iter(lengths_):
    """Yields the plane index plus lower index `idx_i` and upper index
    `idx_f` of the point list representing this plane
    (i.e. pointlist[idx_i:idx_f]).

    """
    i = 0
    while i < len(lengths_):
        yield i, lengths_[:i].sum(), lengths_[:(i + 1)].sum()
        i += 1


def select_from_field_data_store(sim_numbers, ipol, field_type='electric'):
    """
    Queries the field HDF5 database and returns the rows that match
    the given criteria.

    Parameters
    ----------
    sim_numbers : list-like
        List of simulation numbers, i.e. indices of the
        `parameters_and_results.h5` database.
    ipol : int
        Polarization index {0: 'TE', 1: 'TM'}.
    field_type: {'electric' | 'magnetic'}

    Returns
    -------
    df : pandas.DataFrame
        Multiindexed dataframe with the appropriate rows and flat field values
        as columns.
    """
    logger = logging.getLogger(__name__)
    logger.info('Collecting field data')
    logger.debug('Length of `sim_numbers`: {}'.format(len(sim_numbers)))

    # Open the appropriate HDF5 store and verify the index
    pol = POLS[ipol]
    comp = FIELDS[field_type]
    h5 = get_data_file_path(field_comp=comp, polarization=pol)
    logger.debug('Opening HDF5store: {}'.format(h5))
    h5store = pd.HDFStore(h5)

    # Check proper index to avoid performance issues
    i = h5store.root.data.table.cols.index.index
    if i.optlevel != 9 or i.kind != 'full':
        logger.warn('The HDF5 database for the field data does not seem to' +
                    'be properly indexed, which may cause critical ' +
                    'performance issues! Please open the store and run ' +
                    'create_table_index("data", optlevel=9, kind="full")' +
                    'to recreate the full index, then retry.')

    # Query the database
    coords = h5store.select_as_coordinates('data', 'index=sim_numbers')
    df = h5store.select('data', where=coords)
    h5store.close()
    return df


def load_field_data_for_sims_(sim_numbers, ipol, field_type='electric'):
    """Calls `select_from_field_data_store` but returns the data as
    numpy.ndarray. See the doc-string of `select_from_field_data_store`
    for details."""
    return select_from_field_data_store(sim_numbers, ipol,
                                        field_type=field_type).values


# The cached version of the `test_data_for_sims_` function
load_field_data_for_sims = MEMORY.cache(load_field_data_for_sims_)


def get_clustering_input_data(data, ipol, treat_complex, preprocess,
                              field_type='electric', use_cache=False):
    """
    Loads the field data for the given result dataframe `data` and the given
    field type and polarization and preprocesses the data for clustering.

    Parameters
    ----------
    data : pandas.DataFrame
        Data as queried from the `parameters_and_results.h5` database for which
        the field data should be loaded.
    ipol : int
        Polarization index {0: 'TE', 1: 'TM'}.
    treat_complex : None or str (numpy function name or 'concat')
        If None, complex numbers are preserved. If concat, real and imaginary
        parts are concatenated using `numpy.hstack`. Other str-values are
        treated as numpy function names, e.g. `'abs'`, which may raise an
        Exception if numpy does not have this attribute.
    preprocess : None or str (function name of the sklearn.preprocessing module)
        Name of a sklearn.preprocessing function to be used to preprocess
        the data (e.g. `'scale'`).
    field_type: {'electric' | 'magnetic'}
    use_cache : bool
        Whether to use the joblib cache for the `load_field_data_for_sims_`
        function. This will cause a small overhead on the first call but
        will give a large speed-up on following calls with the same
        keyword values.

    Returns
    -------
    test_data : numpy.ndarray
        Preprocessed input matrix to be used in the clustering process.
    """
    logger = logging.getLogger(__name__)
    t0 = time.time()
    sim_nums = data.index.tolist()
    if use_cache:
        test_data = load_field_data_for_sims(sim_nums, ipol,
                                             field_type=field_type)
    else:
        test_data = load_field_data_for_sims_(sim_nums, ipol,
                                              field_type=field_type)
    if treat_complex is not None:
        if treat_complex == 'concat':
            test_data = np.hstack((test_data.real, test_data.imag))
        else:
            test_data = getattr(np, treat_complex)(test_data)
    if preprocess is not None:
        test_data = getattr(preprocessing, preprocess)(test_data, axis=1)

    td = time.time() - t0
    logger.info('Loading time: {}.'.format(format_time(td)))
    return test_data


def get_single_sample(sim_num, ipol, treat_complex='abs', preprocess='scale',
                      field_type='electric', use_cache=False):
    if use_cache:
        test_data = load_field_data_for_sims([sim_num], ipol,
                                             field_type=field_type)
    else:
        test_data = load_field_data_for_sims_([sim_num], ipol,
                                              field_type=field_type)
    if treat_complex is not None:
        if treat_complex == 'concat':
            test_data = np.hstack((test_data.real, test_data.imag))
        else:
            test_data = getattr(np, treat_complex)(test_data)
    if preprocess is not None:
        test_data = getattr(preprocessing, preprocess)(test_data, axis=1)
    return test_data


def cluster_fit_data(cluster_type, samples, **cluster_kwargs):
    logger = logging.getLogger(__name__)
    t0 = time.time()
    if hasattr(cluster, cluster_type):
        family = cluster
    elif hasattr(mixture, cluster_type):
        family = mixture
    else:
        raise ValueError('Cannot find cluster_type {}'.format(cluster_type) +
                         ' in cluster or mixture modules.')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if cluster_type == 'MeanShift':
            bandwidth = family.estimate_bandwidth(samples, quantile=0.2,
                                                  n_samples=500)
            cluster_kwargs['bandwidth'] = bandwidth
            if 'n_clusters' in cluster_kwargs:
                del cluster_kwargs['n_clusters']
                logger.debug('Ignoring `n_clusters` as {} autodetects it.'.
                             format(cluster_type))

        model = getattr(family, cluster_type)(**cluster_kwargs)
        model.fit(samples)

        # DBSCAN supports outliers, so that the number of clusters must
        # be computed differently
        if cluster_type == 'DBSCAN':
            labels = model.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            clusters = pd.Series(
                [samples[labels == i] for i in range(n_clusters)])
            means_ = np.vstack(
                clusters.apply(lambda clust: np.mean(clust, axis=0)).tolist())
            logger.info('DBSCAN has found {} clusters'.format(n_clusters))

            # Add
            model.n_clusters = n_clusters
            model.cluster_centers_ = means_

        # Mixture methods do not return labels, so we need to predict them
        if family == mixture:
            model.labels_ = model.predict(samples)
            # The same is true for the cluster centers
            model.cluster_centers_ = model.means_
            model.n_clusters = model.means_.shape[0]
            # Add the probability matrix too the cluster too
            model.cluster_probas_per_sample = model.predict_proba(samples)

        td = time.time() - t0
        logger.info('Time for clustering: {}.'.format(format_time(td)))
        return model


def get_silhouette(samples, labels):
    logger = logging.getLogger(__name__)
    try:
        s_samples = silhouette_samples(samples, labels)
        s_avg = np.average(s_samples)
        return s_avg, s_samples
    except:
        try:
            s_samples = silhouette_samples_block(samples, labels)
            s_avg = np.average(s_samples)
            return s_avg, s_samples
        except Exception as e:
            logger.warn('Unable to calculate the silhouette coefficients.' +
                        'The error was: {}'.format(e))
            return None, None


def get_only_sample_data(sim_data, pol='TE', direc='Gamma-K',
                         treat_complex='abs', preprocess='normalize',
                         field_type='electric'):
    logger = logging.getLogger(__name__)
    # Get polarization and direction data properties
    pol_suf = {'TE': '_1', 'TM': '_2'}[pol]
    ipol = int(pol_suf[1:]) - 1
    phi = {'Gamma-K': 0.0, 'Gamma-M': 90.0}[direc]

    # Find the relevant sim-numbers and load test data
    logger.info('Reducing data set for pol={} and direction {}'.
                format(pol, direc))
    data = sim_data[sim_data.phi == phi]

    logger.info('Loading sample data')
    return get_clustering_input_data(data, ipol, treat_complex, preprocess,
                                     field_type=field_type)


def cluster_modes(sim_data, pol='TE', direc='Gamma-K',
                  theta_split=None, treat_complex='abs', preprocess='scale',
                  cluster_type='MiniBatchKMeans',
                  field_type='electric', **cluster_kwargs):
    logger = logging.getLogger(__name__)
    # Get polarization and direction data properties
    pol_suf = {'TE': '_1', 'TM': '_2'}[pol]
    ipol = int(pol_suf[1:]) - 1
    phi = {'Gamma-K': 0.0, 'Gamma-M': 90.0}[direc]

    # Find the relevant sim-numbers and load test data
    logger.info('Reducing data set for pol={} and direction {}'.
                format(pol, direc))
    data = sim_data[sim_data.phi == phi]

    # Split data to fit and prediction set if theta_split is given
    if theta_split is not None:
        logger.info('Splitting data in test set for theta<=' +
                    '{0} and prediction set for theta>{0}'.format(theta_split))
        data_fit = data[data.theta <= theta_split]
        data_predict = data[data.theta > theta_split]
        logger.info('Fit data size: {}'.format(data_fit.shape))
        logger.info('Prediction data size:', data_predict.shape)

        # Load and prepare data for the clustering algorithm
        logger.info('Loading sample data for fit')
        samples_fit = get_clustering_input_data(data_fit, ipol, treat_complex,
                                                preprocess,
                                                field_type=field_type)
        logger.info('Loading sample data for prediction')
        samples_pred = get_clustering_input_data(data_predict, ipol,
                                                 treat_complex, preprocess,
                                                 field_type=field_type)
    else:
        # Load and prepare data for the clustering algorithm
        logger.info('Loading sample data')
        samples_fit = get_clustering_input_data(data, ipol, treat_complex,
                                                preprocess,
                                                field_type=field_type)

    # Run the clustering algorithm
    logger.info('Running {} ...'.format(cluster_type))
    model = cluster_fit_data(cluster_type, samples_fit, **cluster_kwargs)
    sim_numbers = data.index.tolist()
    score_fit, silhouettes_fit = get_silhouette(samples_fit, model.labels_)
    logger.info('Silhouette score fit: {}'.format(score_fit))
    if theta_split is not None:
        # Get fitted and predicted labels
        labs_f = model.labels_
        labs_p = model.predict(samples_pred)
        score_pred, silhouettes_pred = get_silhouette(samples_pred, labs_p)
        logger.info('Silhouette score predict: {}'.format(score_pred))
        # Get simulations numbers for each case
        sn_f = data_fit.index.tolist()
        sn_p = data_predict.index.tolist()
        # Build dictionary mapping simulation number to label
        ldict = {}
        # Holds Euclidian distances from the assigned cluster centers
        distances = {}
        sil_dict = {}
        logger.info('Calculating Euclidian distances')
        for irow, (sn_i, l_i, sl_i) in enumerate(
                zip(sn_f, labs_f, silhouettes_fit)):
            ldict[sn_i] = l_i
            sil_dict[sn_i] = sl_i
            ccen = model.cluster_centers_[l_i].reshape(1, -1)
            samp = samples_fit[irow].reshape(1, -1)
            distances[sn_i] = euclidean_distances(ccen, samp)[0][0]
        for irow, (sn_i, l_i, sl_i) in enumerate(
                zip(sn_p, labs_p, silhouettes_pred)):
            assert sn_i not in ldict
            ldict[sn_i] = l_i
            sil_dict[sn_i] = sl_i
            ccen = model.cluster_centers_[l_i].reshape(1, -1)
            samp = samples_pred[irow].reshape(1, -1)
            distances[sn_i] = euclidean_distances(ccen, samp)[0][0]
        labels = [ldict[sn_i] for sn_i in sim_numbers]
        silhouettes = [sil_dict[sn_i] for sn_i in sim_numbers]
        euclidian_distances = [distances[sn_i] for sn_i in sim_numbers]

        # We calculate the silhouettes a second time using the complete
        # set of results
        silhouettes = dict(silhouettes_partial=silhouettes)
        samples_all = np.vstack((samples_fit, samples_pred))
        sn_all = sn_f + sn_p
        labs_all = np.hstack((labs_f, labs_p))
        score_all, sils_all = get_silhouette(samples_all, labs_all)
        logger.info('Silhouette score union: {}'.format(score_all))
        sils_union = [None] * len(sn_all)
        for i, sn_i in enumerate(sn_all):
            sils_union[sim_numbers.index(sn_i)] = sils_all[i]
        silhouettes['silhouettes_union'] = sils_union
    else:
        labels = model.labels_
        silhouettes = silhouettes_fit
        euclidian_distances = []
        logger.info('Calculating Euclidian distances')
        for irow, l_i in enumerate(labels):
            ccen = model.cluster_centers_[l_i].reshape(1, -1)
            samp = samples_fit[irow].reshape(1, -1)
            euclidian_distances.append(euclidean_distances(ccen, samp)[0][0])

    return model, sim_numbers, labels, euclidian_distances, silhouettes


def predict_modes(model, sim_data, pol, direc, treat_complex='abs',
                  preprocess='scale', field_type='electric'):
    logger = logging.getLogger(__name__)
    # Get polarization and direction data properties
    pol_suf = {'TE': '_1', 'TM': '_2'}[pol]
    ipol = int(pol_suf[1:]) - 1
    phi = {'Gamma-K': 0.0, 'Gamma-M': 90.0}[direc]

    # Find the relevant sim-numbers and load test data
    logger.info('Reducing data set for pol={} and direction {}'.
                format(pol, direc))
    data = sim_data[sim_data.phi == phi]

    logger.info('Loading prediction data')
    samples_pred = get_clustering_input_data(data, ipol, treat_complex,
                                             preprocess, field_type=field_type)
    sim_numbers = data.index.tolist()
    labels = model.predict(samples_pred)
    score_pred, silhouettes_pred = get_silhouette(samples_pred, labels)
    logger.info('Silhouette score predict: {}'.format(score_pred))

    # Euclidian distances from the assigned cluster centers
    euclidian_distances = []
    logger.info('Calculating Euclidian distances')
    for irow, l_i in enumerate(labels):
        ccen = model.cluster_centers_[l_i].reshape(1, -1)
        samp = samples_pred[irow].reshape(1, -1)
        euclidian_distances.append(euclidean_distances(ccen, samp)[0][0])

    return sim_numbers, labels, euclidian_distances, silhouettes_pred


def cluster_all_modes(sim_data_, theta_split=None,
                      cluster_type='MiniBatchKMeans', cluster_kwargs_dicts=None,
                      pols=None, direcs=None, field_type='electric'):
    logger = logging.getLogger(__name__)
    # Copy input data
    sim_data = deepcopy(sim_data_)

    # Get direction and polarization data
    ddict = DEFAULT_SIM_DDICT
    pdict = DEFAULT_SIM_PDICT
    if pols is None:
        pols = pdict.keys()
    if direcs is None:
        direcs = ddict.keys()

    # Init nested dicts
    model_data = defaultdict(dict)

    for direc in direcs:
        for pol in pols:
            logger.info('Clustering for {} {}'.format(direc, pol))
            model_data[direc][pol] = {}
            pol_suf = pdict[pol]
            if cluster_kwargs_dicts is None:
                cluster_kwargs = {}
            else:
                cluster_kwargs = cluster_kwargs_dicts[direc][pol]
            if 'n_clusters' in cluster_kwargs:
                if isinstance(cluster_kwargs['n_clusters'],
                              (tuple, list, np.ndarray)):
                    n_cluster_list = cluster_kwargs['n_clusters']
                else:
                    n_cluster_list = [cluster_kwargs['n_clusters']]
                # Find optimum for the given n_clusters list based on
                # silhouette score
                _models = []
                _sim_numss = []
                _labelss = []
                _distancess = []
                _silhouettes = []
                kwa = deepcopy(cluster_kwargs)
                if len(n_cluster_list) > 1:
                    logger.info('Searching optimum in n_cluster list {}'.
                                format(n_cluster_list))
                for n_clusters in n_cluster_list:
                    kwa['n_clusters'] = n_clusters
                    model, sim_nums, labels, distances, silhouettes = \
                        cluster_modes(sim_data,
                                      pol=pol,
                                      direc=direc,
                                      theta_split=theta_split,
                                      cluster_type=cluster_type,
                                      field_type=field_type,
                                      **kwa)
                    _models.append(model)
                    _sim_numss.append(sim_nums)
                    _labelss.append(labels)
                    _distancess.append(distances)
                    _silhouettes.append(silhouettes)

                # Set optimum
                if isinstance(silhouettes, dict):
                    _sils_union = [_sildict['silhouettes_union'] for _sildict in
                                   _silhouettes]
                    _sil_avgs = [np.average(_sil) for _sil in _sils_union]
                else:
                    _sil_avgs = [np.average(_sil) for _sil in _silhouettes]
                _opt_ind = _sil_avgs.index(max(_sil_avgs))
                if len(n_cluster_list) > 1:
                    logger.info('Found optimum for n_clusters = {}'.
                                format(n_cluster_list[_opt_ind],) +
                                'with score = {}'.format(_sil_avgs[_opt_ind]))
                model = _models[_opt_ind]
                sim_nums = _sim_numss[_opt_ind]
                labels = _labelss[_opt_ind]
                distances = _distancess[_opt_ind]
                silhouettes = _silhouettes[_opt_ind]
            else:
                model, sim_nums, labels, distances, silhouettes = \
                    cluster_modes(sim_data,
                                  pol=pol,
                                  direc=direc,
                                  theta_split=theta_split,
                                  cluster_type=cluster_type,
                                  field_type=field_type,
                                  **cluster_kwargs)

            model_data[direc][pol]['model'] = model
            model_data[direc][pol]['sim_nums'] = sim_nums

            # Write labels, distances and silhouettes to the data frame
            logger.info('Updating simulation data set.')

            if isinstance(silhouettes, dict):
                _cols = ['Classification' + pol_suf,
                         'Euclidian_Distances' + pol_suf,
                         'Silhouettes' + pol_suf,
                         'Silhouettes_partial' + pol_suf]
                for _col, _cdata in zip(_cols, [labels, distances,
                                                silhouettes[
                                                    'silhouettes_union'],
                                                silhouettes[
                                                    'silhouettes_partial']]):
                    if _col not in sim_data:
                        sim_data[_col] = np.NaN
                    sim_data.loc[sim_nums, _col] = _cdata
            else:
                _cols = ['Classification' + pol_suf,
                         'Euclidian_Distances' + pol_suf,
                         'Silhouettes' + pol_suf]
                for _col, _cdata in zip(_cols,
                                        [labels, distances, silhouettes]):
                    if _col not in sim_data:
                        sim_data[_col] = np.NaN
                    sim_data.loc[sim_nums, _col] = _cdata

            # Write probabilities for each cluster to the data frame
            # (only for probabilistic cluster_types)
            if hasattr(model, 'cluster_probas_per_sample'):
                for ic in range(model.n_clusters):
                    _col = 'Probability_pol{}_label_{}'.format(pol_suf, ic)
                    if _col not in sim_data:
                        sim_data[_col] = np.NaN
                    sim_data.loc[
                        sim_nums, _col] = model.cluster_probas_per_sample[:, ic]

                # Single column for the probability to be in its own cluster
                _col = 'Probability' + pol_suf
                if _col not in sim_data:
                    sim_data[_col] = np.NaN
                for ic in range(model.n_clusters):
                    _idx = sim_data['Classification' + pol_suf] == ic
                    _snums = sim_data[_idx].index
                    _col_own = 'Probability_pol{}_label_{}'.format(pol_suf, ic)
                    _ser = sim_data.loc[_snums, _col_own]
                    sim_data.loc[_snums, _col] = _ser.values

            logger.info('Finished')
    return sim_data, dict(model_data)


def classify_with_model(sim_data_, model, pols=None, direcs=None,
                        treat_complex=None, preprocess='scale',
                        field_type='electric'):
    logger = logging.getLogger(__name__)
    # Copy input data
    sim_data = deepcopy(sim_data_)

    # Get direction and polarization data
    ddict = DEFAULT_SIM_DDICT
    pdict = DEFAULT_SIM_PDICT
    if pols is None:
        pols = pdict.keys()
    if direcs is None:
        direcs = ddict.keys()

    # Init nested dicts
    sim_num_data = defaultdict(dict)

    for direc in direcs:
        for pol in pols:
            logger.info('Classifying for {} {}'.format(direc, pol))
            sim_num_data[direc][pol] = {}
            pol_suf = pdict[pol]

            sim_nums, labels, distances, silhouettes = \
                predict_modes(model, sim_data, pol, direc,
                              treat_complex=treat_complex,
                              preprocess=preprocess, field_type=field_type)
            sim_num_data[direc][pol]['sim_nums'] = sim_nums

            # Write labels, distances and silhouettes to the data frame
            logger.info('Updating simulation data set.')
            _cols = ['Classification' + pol_suf,
                     'Euclidian_Distances' + pol_suf,
                     'Silhouettes' + pol_suf]
            for _col, _cdata in zip(_cols, [labels, distances, silhouettes]):
                if _col not in sim_data:
                    sim_data[_col] = np.NaN
                sim_data.loc[sim_nums, _col] = _cdata
                logger.info('Finished')

    return sim_data, dict(sim_num_data)


def test(every):
    logger = logging.getLogger(__name__)
    df = get_results(every, every)
    lengths, pointlist, domain_ids = get_metadata()
    data = load_field_data_for_sims_([0, 1, 2, 3], 0, 'magnetic')
    logger.info(data.shape)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)
    #set_dummy_mode(True)
    test(2)
