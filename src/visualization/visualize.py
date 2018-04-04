# coding: utf-8

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from copy import deepcopy


CUSTOM_COLORS = [(0.8901960784313725, 0.03529411764705882, 0.09411764705882353),
                 (0.21602460800432691, 0.49487120380588606, 0.71987698697576341),
                 (0.30426760128900115, 0.68329106055054012, 0.29293349969620797),
                 (1.0, 0.50591311045721465, 0.0031372549487095253),
                 (0.99315647868549117, 0.9870049982678657, 0.19915417450315812),
                 (0.95850826852461868, 0.50846600392285513, 0.74492888871361229),
                 (0.33999999999999997, 0.82879999999999987, 0.86),
                 (0.60000002384185791, 0.60000002384185791, 0.60000002384185791),
                 (0.0, 0.6196078431372549, 0.8784313725490196),
                 (0.37119999999999997, 0.33999999999999997, 0.86),
                 (0.60083047361934883, 0.30814303335021526, 0.63169552298153153)]


def plot_silhouettes(_data, pol_suf, indices=None, n_clusters=None,
                     partial_silhouettes=False, ax=None, xlim=None,
                     cmap='custom',
                     xlabel='default', ylabel='default', legend=False,
                     data_every=1,
                     label_text_kwargs=None):
    import matplotlib.cm as cm
    if ax is None:
        plt.figure(figsize=(5, 3))
        ax = plt.gca()

    if not indices is None:
        _data = _data.loc[indices]
    if partial_silhouettes:
        sample_silhouette_values = _data['Silhouettes_partial' + pol_suf].values
    else:
        sample_silhouette_values = _data['Silhouettes' + pol_suf].values
    silhouette_avg = np.average(sample_silhouette_values)
    cluster_labels = _data['Classification' + pol_suf].values
    n_samples = len(sample_silhouette_values)
    if n_clusters is None:
        n_clusters = len(np.unique(cluster_labels))

    if cmap == 'custom':
        cmap = sns.color_palette(CUSTOM_COLORS, n_colors=n_clusters)
    else:
        cmap = sns.color_palette(cmap, n_colors=n_clusters)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    if xlim is None:
        xlim = [sample_silhouette_values.min(), 1]
    ax.set_xlim(xlim)
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    yoff = int(n_samples / 50.)
    ax.set_ylim([0, n_samples + (n_clusters + 3) * yoff])

    y_lower = yoff * 2.
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cmap[i]
        ax.fill_betweenx(np.arange(y_lower, y_upper)[::data_every],
                         0, ith_cluster_silhouette_values[::data_every],
                         facecolor=color, edgecolor=color)

        # Label the silhouette plots with their cluster numbers at the middle
        if label_text_kwargs is None:
            label_text_kwargs = {}
        if not 'verticalalignment' in label_text_kwargs:
            label_text_kwargs['verticalalignment'] = 'center'
        if not 'horizontalalignment' in label_text_kwargs:
            label_text_kwargs['horizontalalignment'] = 'right'
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i + 1),
                **label_text_kwargs)

        # Compute the new y_lower for next plot
        y_lower = y_upper + yoff

    if xlabel == 'default':
        xlabel = 'Silhouette coefficient'
    if ylabel == 'default':
        ylabel = 'Cluster label'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--",
               label='Silhouette\naverage')
    if legend:
        ax.legend(loc='best', fontsize=7)
    ax.set_yticks([])  # Clear the yaxis labels / ticks


def get_pivot_data(_data, cq, xq='theta', yq='wavelength'):
    return _data.pivot(yq, xq, cq).astype('float')


def imshow_data(ax, _data, cq, xq='theta', yq='wavelength',
                cmap='coolwarm', **kwargs):
    _data = get_pivot_data(_data, cq, xq, yq)
    image = _data.values

    if cq.startswith('Classification'):
        # make a color map of fixed colors
        from matplotlib import colors
        n_colors = int(image.max()) + 1
        cmap = colors.ListedColormap(sns.color_palette(cmap,
                                                       n_colors=n_colors))
        bounds = np.linspace(0., float(n_colors), n_colors + 1)
        norm = colors.BoundaryNorm(bounds, cmap.N)
    else:
        norm = None

    kwdefaults = dict(origin='lower', aspect='auto',
                      interpolation='none')
    for dkey, dval in kwdefaults.iteritems():
        if not dkey in kwargs:
            kwargs[dkey] = dval

    extent = (_data.columns.min(), _data.columns.max(),
              _data.index.min(), _data.index.max())
    im = ax.imshow(image, extent=extent, cmap=cmap,
                   norm=norm, **kwargs)
    ax.set_xlabel(xq)
    ax.set_ylabel(yq)

    if hasattr(ax, 'cax'):
        cax = ax.cax
    else:
        cax = None

    if cq.startswith('Classification'):
        cb = plt.colorbar(im, cmap=cmap, norm=norm, boundaries=bounds,
                          ticks=bounds[1:] - 0.5, cax=cax)
        cb.set_ticklabels([str(int(it)) for it in bounds[1:]])
        if hasattr(ax, 'cax'):
            ax.cax.set_ylabel('Mode labels')
        ax.set_title('Clustering of $E$-field data')
    else:
        plt.colorbar(im, cax=cax)
        if hasattr(ax, 'cax'):
            ax.cax.set_ylabel(cq)
        ax.set_title('Quantity from post-process')
    return im


def get_discrete_cmap(n_colors, cmap='custom'):
    from matplotlib import colors
    # make a color map of fixed colors
    if cmap == 'custom':
        cmap = sns.color_palette(CUSTOM_COLORS, n_colors=n_colors)
    else:
        cmap = sns.color_palette(cmap, n_colors=n_colors)

    cmap = colors.ListedColormap(cmap)
    bounds = np.linspace(0., float(n_colors), n_colors + 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    return cmap, bounds, norm


def imshow_classification_overlay(ax, _data, pol_suf, xq='theta',
                                  yq='wavelength', cmap='custom',
                                  repr_measure='silhouettes',
                                  make_comparable_to=None,
                                  bkg_black=True, exponent=1., clip_zero=0.0,
                                  clip_softly=True, overdrive=0.,
                                  plot_cbar=True, **kwargs):
    from matplotlib.image import AxesImage
    from sklearn.preprocessing import minmax_scale
    data = deepcopy(_data)

    # The image data based on the cluster labels
    data_labels = get_pivot_data(data, 'Classification' + pol_suf, xq, yq)
    image = data_labels.values

    if repr_measure == 'distances':
        if not make_comparable_to is None:
            raise NotImplementedError('`make_comparable_to` ia only ' +
                                      'implemented for `repr_measure="silhouettes"`')
        # Alpha data based on the Euclidian distances
        distances = data['Euclidian_Distances' + pol_suf]
        data['_pl_alpha'] = minmax_scale(1. / distances)
        data_alpha = get_pivot_data(data, '_pl_alpha', xq, yq)
        alpha = data_alpha.values
    elif repr_measure == 'silhouettes':
        silhouettes = get_pivot_data(data, 'Silhouettes' + pol_suf, xq,
                                     yq).values

        if make_comparable_to is None:
            _gmin = silhouettes.min()
            _gmax = silhouettes.max()
            _ptp = silhouettes.ptp()
        else:
            sil_comp = make_comparable_to['Silhouettes' + pol_suf]

            _min1, _max1 = silhouettes.min(), silhouettes.max()
            _min2, _max2 = sil_comp.min(), sil_comp.max()
            _gmax = max([_max1, _max2])
            _gmin = min([_min1, _min2])
            _ptp = _gmax - _gmin
        sil_range = (_gmin, _gmax)
        alpha = (silhouettes - _gmin) / _ptp
    elif repr_measure == 'probability':
        alpha = get_pivot_data(data, 'Probability' + pol_suf, xq, yq).values
        sil_range = (0., 1.)

    # Manipulate the alphas based on clipping and n_sqrts settings
    if overdrive > 0.:
        alpha = np.clip(alpha * (1. + overdrive), 0., 1.)
    if exponent != 1.:
        alpha = np.power(alpha, exponent)
    if clip_zero > 0. and clip_zero < 1.:
        if clip_softly:
            alpha = minmax_scale(alpha.ravel(), (clip_zero, 1.)). \
                reshape(alpha.shape)
        else:
            alpha = np.clip(alpha, clip_zero, 1.)

    #    # Set new maximum
    #    if max_final < 1.:
    #        alpha = minmax_scale(alpha.ravel(), (clip_zero, max_final)).\
    #                                                        reshape(alpha.shape)
    if clip_zero >= 1.:
        alpha = np.ones_like(alpha)
    extent = (data_labels.columns.min(), data_labels.columns.max(),
              data_labels.index.min(), data_labels.index.max())

    # make a color map of fixed colors
    n_colors = int(image.max()) + 1
    cmap, bounds, norm = get_discrete_cmap(n_colors, cmap)

    # Set default kwargs
    kwdefaults = dict(origin='lower', aspect='auto',
                      interpolation='none')
    for dkey, dval in kwdefaults.iteritems():
        if not dkey in kwargs:
            kwargs[dkey] = dval

    # get dummy_image for colorbar
    im_dummy = ax.imshow(image, extent=extent, cmap=cmap,
                         norm=norm, **kwargs)
    ax.images.pop()  # remove it afterwards

    # Get an AxesImage to convert to RGBA
    im = AxesImage(ax, cmap, norm, kwdefaults['interpolation'],
                   kwdefaults['origin'], extent)
    rgba = im.to_rgba(image)

    # Set the alpha column to the alpha values based on the distance measure
    rgba[:, :, 3] = alpha

    # Create a black background image if bkg_black is True
    if bkg_black:
        im_black = deepcopy(rgba)
        im_black[:, :, :3] = 0.
        im_black[:, :, 3] = 1.
        im = ax.imshow(im_black, extent=extent, **kwargs)
    im = ax.imshow(rgba, extent=extent, **kwargs)

    # Draw the colorbar
    if plot_cbar:
        if hasattr(ax, 'cax'):
            cax = ax.cax
        else:
            cax = None
        cb = plt.colorbar(im_dummy, cmap=cmap, norm=norm, boundaries=bounds,
                          ticks=bounds[1:] - 0.5, cax=cax)
        cb.set_ticklabels([str(int(it)) for it in bounds[1:]])

    # Set labels
    if hasattr(ax, 'cax') and plot_cbar:
        ax.cax.set_ylabel('Labels')
    ax.set_xlabel(xq)
    ax.set_ylabel(yq)
    ax.set_title('Clustering of $E$-field data')
    if repr_measure == 'distances':
        return im, im_dummy, bounds
    return im, im_dummy, bounds, sil_range


def compare_values_and_classification(data_, cq_quant, pol_suf, fig=None,
                                      axes=None, cmap_values='viridis',
                                      xq='theta', yq='wavelength',
                                      theta_split=None, overlay_kwargs=None,
                                      return_axes=False,
                                      **kwargs):
    from mpl_toolkits.axes_grid1 import ImageGrid
    if axes is None:
        if fig is None:
            fig = plt.figure(1, (9., 4.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(1, 2),  # creates 2x2 grid of axes
                         axes_pad=0.8,  # pad between axes in inch.
                         label_mode="L",
                         cbar_mode='each',
                         aspect=False,
                         cbar_pad=0.1)
        ax1, ax2 = (grid[0], grid[1])
    else:
        ax1, ax2 = axes

    im1 = imshow_data(ax1, data_, cq_quant + pol_suf, cmap=cmap_values,
                      **kwargs)
    if not theta_split is None:
        ax1.axvline(theta_split, lw=1., color='white', ls='--')
    kwargs['interpolation'] = 'none'
    if overlay_kwargs is None:
        overlay_kwargs = {}
    if not 'cmap' in overlay_kwargs:
        overlay_kwargs['cmap'] = 'custom'
    ico_ret = imshow_classification_overlay(
        ax2, data_, pol_suf, **overlay_kwargs)
    if len(ico_ret) == 3:
        im2, im_dummy, bnds = ico_ret
        if return_axes:
            return (im1, im2, im_dummy), bnds, ax1, ax2
        else:
            return (im1, im2, im_dummy), bnds
    else:
        im2, im_dummy, bnds, sil_range = ico_ret
        if return_axes:
            return (im1, im2, im_dummy), bnds, sil_range, ax1, ax2
        else:
            return (im1, im2, im_dummy), bnds, sil_range