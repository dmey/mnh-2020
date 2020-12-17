from typing import Tuple, NamedTuple, List, Optional, Union
import time
from functools import reduce
import xarray as xr
import numpy as np
import tempfile
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

import string


from collections import defaultdict

import pickle

import time

USE_CLOUD_OPTICAL_DEPTH = True


class StackInfoVar(NamedTuple):
    name: str
    dims: Tuple[str]
    shape: Tuple[int]


StackInfo = List[StackInfoVar]

# Specialized stacking/unstacking functions (as opposed to
# using xarray's to_stacked_array/to_unstacked_dataset).
# This allows to have control over the exact stacking behaviour
# which in turn allows to store compact stacking metadata and use it
# to unstack arbitrary arrays not directly related to the input dataset object.


def to_stacked_array(ds: xr.Dataset, var_names=None, new_dim='stacked', name=None) -> Tuple[xr.DataArray, StackInfo]:
    # Sample dimension must be the first dimension in all variables.
    if not var_names:
        var_names = sorted(ds.data_vars)
    stack_info = []
    var_stacked = []
    for var_name in var_names:
        v = ds.data_vars[var_name]
        if len(v.dims) > 1:
            stacked = v.stack({new_dim: v.dims[1:]})
            stacked = stacked.drop(list(stacked.coords.keys()))
        else:
            stacked = v.expand_dims(new_dim, axis=-1)
        stack_info.append(StackInfoVar(var_name, v.dims, v.shape[1:]))
        var_stacked.append(stacked)
    arr = xr.concat(var_stacked, new_dim)
    if name:
        arr = arr.rename(name)
    return arr, stack_info


def to_unstacked_dataset(arr: np.ndarray, stack_info: StackInfo) -> xr.Dataset:
    if type(arr) == xr.DataArray:
        arr = arr.values
    elif type(arr) == np.ndarray:
        pass
    else:
        raise RuntimeError('Passed array must be of type DataArray or ndarray')

    unstacked = {}
    curr_i = 0
    for var in stack_info:
        feature_len = 1
        unstacked_shape = [arr.shape[0], ]
        for dim_len in var.shape:
            feature_len *= dim_len
            unstacked_shape.append(dim_len)
        var_slice = arr[:, curr_i:curr_i+feature_len]
        var_unstacked = var_slice.reshape(unstacked_shape)
        unstacked[var.name] = xr.DataArray(var_unstacked, dims=var.dims)
        curr_i += feature_len
    ds = xr.Dataset(unstacked)
    return ds


def shuffle_dataset(ds: xr.Dataset, dim: str, seed=None) -> xr.Dataset:
    idx = np.arange(ds.dims[dim])
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(idx)
    return ds.isel({dim: idx})


def train_test_split_dataset(ds: xr.Dataset, dim: str,
                             train_size: Optional[Union[float, int]] = None,
                             test_size: Optional[Union[float, int]] = None,
                             shuffle=True, seed=None) -> tuple:
    if shuffle:
        ds = shuffle_dataset(ds, dim, seed=seed)
    count = ds.dims[dim]
    if train_size is None:
        assert test_size is not None
        if test_size > 1:
            test_count = int(test_size)
            assert test_count < count
        else:
            test_count = int(count * test_size)
            assert test_count >= 0
        train_count = count - test_count
    else:
        assert test_size is None
        if train_size > 1:
            train_count = int(train_size)
            assert train_count <= count
        else:
            train_count = int(count * train_size)
            assert train_count > 0
        test_count = count - train_count
    train = ds.isel({dim: slice(0, train_count)})
    test = ds.isel({dim: slice(train_count, None)})
    return train, test


def to_normalized_dataset(ds: xr.Dataset, stats_info: Optional[dict] = None) -> Union[xr.Dataset, dict]:
    """ Normalize quantities in a dataset by their mean and standard deviation.
    """
    stats_info = {}
    ds_normalized = xr.zeros_like(ds)
    for name in list(ds):
        stats_info[name] = {
            'mean': ds[name].mean(),
            'std': ds[name].std()
        }
        ds_normalized[name] = (ds[name] - stats_info[name]['mean']) \
            / stats_info[name]['std']
    return ds_normalized, stats_info


def to_unnormalized_dataset(ds: xr.Dataset, stats_info: dict) -> xr.Dataset:
    """ Recover a dataset of previously normalized quantities by their mean and standard deviation.
    """
    ds_unnormalized = xr.zeros_like(ds)
    for name in list(ds):
        ds_unnormalized[name] = ds[name] * stats_info[name]['std'] \
            + stats_info[name]['mean']
    return ds_unnormalized


def compute_stats(true, pred):
    assert true.shape == pred.shape
    bias = (true - pred).mean().values
    mae = np.abs(true - pred).mean().values
    mse = ((true - pred)**2).mean().values
    rmse = np.sqrt(mse)
    # TODO: add 1st and second derivative
    df_stats = pd.DataFrame([bias, mae, mse, rmse]).T
    df_stats.columns = ['bias', 'mae', 'mse', 'rmse']
    df_stats[['bias', 'mae', 'mse', 'rmse']] = df_stats[[
        'bias', 'mae', 'mse', 'rmse']].astype(float)
    return df_stats


def load_ds_inputs(proj_path, columns=slice(0, None)):
    """ Load and subset the input data used throughout the experiments
    """
    ds_inputs = xr.open_dataset(proj_path / 'data' / 'nwp_saf_profiles_in.nc')
    ds_inputs = compute_layer_cloud_optical_depth(ds_inputs)
    inputs_relevant = ['temperature_fl', 'pressure_hl']
    if USE_CLOUD_OPTICAL_DEPTH:
        inputs_relevant += ['layer_cloud_optical_depth']
    ds_inputs = ds_inputs[inputs_relevant].sel(column=columns)
    return ds_inputs


def compute_lw_dn_mlp(train, validation, test, X_vars, y_var, epochs, iterations, verbose=0, tmpdir=None):
    """ Fits and run inference, then return statistics on the test
    """
    X_train, _ = to_stacked_array(train[X_vars])
    X_validation, _ = to_stacked_array(validation[X_vars])
    X_test, _ = to_stacked_array(test[X_vars])
    y_train, _ = to_stacked_array(train[y_var])
    y_validation, _ = to_stacked_array(validation[y_var])
    y_test, _ = to_stacked_array(test[y_var])

    validation_data = (X_validation.values, y_validation.values)

    df_stats = pd.DataFrame()
    ds_y_pred_tests = []

    if verbose:
        print('\nBegin ML iteration loop')

    for iter_count in range(iterations):
        t0 = time.time()
        # ML Model
        model = keras.Sequential([
            layers.Input(X_train.shape[1]),
            layers.Dense(512, activation="elu"),
            layers.Dense(512, activation="elu"),
            layers.Dense(512, activation="elu"),
            layers.Dense(y_train.shape[1], activation="linear")
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.Huber(),
            metrics=[keras.metrics.mean_squared_error,
                     keras.metrics.mean_absolute_error]
        )
        try:
            handle, tmp = tempfile.mkstemp(suffix='.hdf5', dir=tmpdir)
            os.close(handle)
            mcp_save = keras.callbacks.ModelCheckpoint(tmp, save_best_only=True,
                                                       monitor='val_loss', mode='min')
            callback = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=25)

            model.fit(X_train.values, y_train.values,
                      batch_size=64, epochs=epochs, validation_data=validation_data, verbose=verbose, shuffle=False,
                      callbacks=[callback, mcp_save])
            model.load_weights(tmp)
        finally:
            os.remove(tmp)
        da_y_pred_test = model.predict(X_test.values)
        ds_y_pred_tests.append(to_unstacked_dataset(da_y_pred_test, _))
        df_stats = df_stats.append(compute_stats(
            y_test, da_y_pred_test), ignore_index=True)

        print(f"ML for iter_count # {iter_count} took {time.time() - t0} s")
        if verbose:
            print(f'\nML count: {iter_count+1}/{iterations}')

    # Always default to sort values by mse -- we do not care about the order ot the index.
    df_stats = df_stats.sort_values('mse', ascending=False).reset_index()
    sorted_index_order = df_stats['index'].values
    df_stats.index = df_stats.index.rename('ml_iter')
    df_stats = df_stats.drop('index', axis=1)

    ds_y_pred_test_best = ds_y_pred_tests[sorted_index_order[-1]]

    return df_stats, ds_y_pred_test_best


def create_synthetic(ds_data, n_samples, n_quantiles,
                     uniformization_ratio, stretch_factor, pyvinecopulib_ctrl=None, verbose=0, generator=None,
                     copula_type=None, has_targets=None, data_fraction=None, seed=None):
    """
    """
    if not generator:
        dir_path = Path('../results/fitting')
        fname = f"copula_type={copula_type}-has_targets={has_targets}-is_normalised=1-data_fraction={data_fraction}.pkl"
        f_path = dir_path / fname
        with open(f_path, 'rb') as f:
            generator = pickle.load(f)

    kws = {}
    if copula_type != 'gaussian':
        kws['num_threads'] = get_num_threads()

    ds_synthetic = generator.generate(n_samples=n_samples, uniformization_ratio=uniformization_ratio,
                                      stretch_factor=stretch_factor, qrng=True, seed=seed, **kws)
    return ds_synthetic, generator


def compute_lw_dn_synth_mlp(
        ds_train, ds_train_X_y, ds_validation_X_y, ds_test_X_y, X_vars, y_var, stats_info,  # Data
        factor_synthetic, n_quantiles, uniformization_ratio, stretch_factor, pyvinecopulib_ctrl,  # Copula
        epochs,  # ML
        n_iter_synthetic, n_iter_ml,  # iteration loop
        verbose=0, tmpdir=None, copula_type=None, has_targets=None, data_fraction=None,
        copula_seed=None):
    """
    """
    df_stats = pd.DataFrame()
    n_samples = ds_train.dims['column'] * factor_synthetic
    multi_index = []
    ds_y_pred_test_bests = []

    generator = None

    if verbose:
        print('\nBegin data generation iteration loop\n')
    t0 = time.time()

    if copula_seed is not None:
        assert n_iter_synthetic == 1

    for iter_count in tqdm(range(n_iter_synthetic), disable=verbose):
        ds_synthetic, generator = create_synthetic(ds_train, n_samples, n_quantiles,
                                                   uniformization_ratio, stretch_factor,
                                                   pyvinecopulib_ctrl, verbose, generator, copula_type, has_targets, data_fraction,
                                                   seed=copula_seed)

        if has_targets:
            # We do not run the physical models as the targets are already presnet. I.e. obs-based pred case
            ds_synthetic_X_y = ds_synthetic
            ds_train_X_y = ds_train_X_y
        else:
            column_gas_optical_depth = 1.7
            ds_synthetic_y = compute_layer_longwave_downwelling(
                to_unnormalized_dataset(ds_synthetic, stats_info), column_gas_optical_depth)
            ds_synthetic_X_y = xr.merge([ds_synthetic, ds_synthetic_y])
            ds_train_y = compute_layer_longwave_downwelling(
                to_unnormalized_dataset(ds_train, stats_info), column_gas_optical_depth)
            ds_train_X_y = xr.merge([ds_train, ds_train_y])

        ds_train_X_y_synthetic = xr.concat(
            [ds_synthetic_X_y, ds_train_X_y], dim='column')

        stats_ml, ds_y_pred_test_best = compute_lw_dn_mlp(
            ds_train_X_y_synthetic, ds_validation_X_y, ds_test_X_y, X_vars, y_var, epochs, n_iter_ml, verbose=verbose, tmpdir=tmpdir)
        # Create a multiindex
        for value in stats_ml.index.values:
            multi_index.append((iter_count, value))
        df_stats = df_stats.append(stats_ml)

        ds_y_pred_test_bests.append(ds_y_pred_test_best)

        if verbose:
            print(
                f'\nData generation count: {iter_count+1}/{n_iter_synthetic}')

    print(f"Generation took {time.time() - t0} s")

    multi_index = pd.MultiIndex.from_tuples(
        multi_index, names=['synth_iter', 'ml_iter'])
    df_stats = pd.DataFrame(
        df_stats.values, index=multi_index, columns=df_stats.columns)

    if verbose:
        print(f'\n===END====')
    return df_stats, ds_y_pred_test_bests


def compute_layer_cloud_optical_depth(ds: xr.Dataset) -> xr.Dataset:
    """ Compute per-layer profiles of cloud optical depth using SAF profile data.
    """

    # Constants
    g = 9.80665  # m s^{-2}
    rho_liquid = 1000  # kg m^{-3}
    rho_ice = 917  # kg m^{-3}
    d_pressure = ds['pressure_hl'].diff(
        'half_level').rename({'half_level': 'level'})

    optical_depth = (ds['q_liquid'] / (rho_liquid * ds['re_liquid']) +
                     ds['q_ice'] / (rho_ice * ds['re_ice'])) * d_pressure / g
    optical_depth = optical_depth.rename('layer_cloud_optical_depth')
    optical_depth.attrs = {
        'long_name': 'Layer cloud optical depth', 'units': '1'}
    return xr.merge([ds, optical_depth])


def compute_layer_longwave_downwelling(ds: xr.Dataset, column_gas_optical_depth: float) -> xr.Dataset:
    """ Compute per-layer profiles of downwelling longwave radiation.
    """

    def compute_layer_emissivity(ds: xr.Dataset, column_gas_optical_depth: float) -> xr.DataArray:
        """ Compute per-layer profiles of emissivities.
        """

        # Pressure expressed in sigma coordinate system.
        pressure_sigma_hl = ds['pressure_hl'] / \
            ds['pressure_hl'].sel(half_level=-1)
        d_pressure_sigma_hl = pressure_sigma_hl.diff(
            'half_level').rename({'half_level': 'level'})
        # Compute gas optical depth.
        layer_gas_optical_depth = column_gas_optical_depth * d_pressure_sigma_hl

        if USE_CLOUD_OPTICAL_DEPTH:
            # Combine gas and cloud optical depths as they are additive.
            layer_optical_depths = layer_gas_optical_depth + \
                ds['layer_cloud_optical_depth']
        else:
            layer_optical_depths = layer_gas_optical_depth

        # Use common diffusivity factor.
        diffusivity_factor = 1/np.cos(np.radians(53))
        # Compute the emissivy at each layer (full level)
        return 1 - np.exp(-diffusivity_factor * layer_optical_depths)

    # Create dummy DataArray
    flux_dn_hl = xr.zeros_like(ds['pressure_hl']).rename('flux_dn_lw')
    flux_dn_hl.attrs = {
        'long_name': 'Downwelling longwave flux', 'units': 'W m⁻²'}

    # Constants and initilization
    Stefan_Boltzmann = 5.670374419e-08  # W m^{-2} K^{-4}
    plank_fl = Stefan_Boltzmann * ds['temperature_fl']**4
    emissivity_fl = compute_layer_emissivity(ds, column_gas_optical_depth)

    # No downwelling flux at the top of the atmosphere.
    flux_dn_hl[:, 0] = 0.

    for hl in range(flux_dn_hl.shape[1] - 1):
        flux_dn_hl[:, hl+1] = flux_dn_hl[:, hl] * \
            (1 - emissivity_fl[:, hl]) + plank_fl[:, hl] * emissivity_fl[:, hl]
    return flux_dn_hl


item_params = {
    'control':
        {
            'color': 'black',
            'position': None,
            'name': 'Baseline'
        },
    'gaussian':
        {
            'color': '#648FFF',
            'position': -1,
            'name': 'Gaussian'
        },
    'parametric':
        {
            'color': '#FFB000',
            'position': 0,
            'name': 'Vine-parametric'
        },
    'tll':
        {
            'color': '#DC267F',
            'position': 1,
            'name': 'Vine-nonparametric'
        }
}


data_fraction_to_row_idx = {
    1: 0
}

marker_styles = {
    'has_targets': {
        '0': 'o',
        '1': '^'
    }
}


metric_labels = {
    'bias': 'Mean bias error in W m⁻²',
    'mae': 'Mean absolute error in W m⁻²',
    'mse': 'Mean square error in (W m⁻²)²',
    'rmse': 'Root mean sqaure error in W m⁻²'
}


def plot_boxplots(item, ax, col_idx, row_idx, metric, y_lim, x_min_max=(-1, 12)):
    name = item['name']
    color = item_params[name]['color']
    if item['is_control']:
        control_y_min = item['stats'][metric].min()
        control_y_max = item['stats'][metric].max()
        control_y_mean = item['stats'][metric].median()
        control_y_q_25 = np.quantile(item['stats'][metric], 0.25)
        control_y_q_75 = np.quantile(item['stats'][metric], 0.75)
        x_min, x_max = x_min_max
        ax[row_idx, col_idx].plot(
            [x_min, x_max], [control_y_mean, control_y_mean], color=color)
        ax[row_idx, col_idx].fill_between(
            [x_min, x_max], control_y_min, control_y_max, alpha=0.1, color='grey')
        ax[row_idx, col_idx].fill_between(
            [x_min, x_max], control_y_q_25, control_y_q_75, alpha=0.4, color='grey')
    else:
        bp = ax[row_idx, col_idx].boxplot(item['stats'][metric],
                                          positions=[item['factor_synthetic'] + item_params[name]['position']], widths=0.7,
                                          patch_artist=True)
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='k')
        for patch in bp['boxes']:
            patch.set(facecolor=color)

    ax[row_idx, col_idx].set_xlim(xmin=x_min_max[0], xmax=x_min_max[-1])
    ax[row_idx, col_idx].set_ylim(y_lim)
    yscale = 'log' if metric == 'rmse' else 'linear'
    ax[row_idx, col_idx].set_yscale(yscale)
    ax[row_idx, col_idx].tick_params(axis='both', which='major', labelsize=16)
    ax[row_idx, col_idx].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax[row_idx, 0].set_ylabel(metric_labels[metric], fontsize=16)
    ax[row_idx, col_idx].set_xlabel(
        'Copula type grouped by augmentation factor', fontsize=16)


def make_legend(ax, row_idx, col_idx, type):
    patches = []
    patches.append(Line2D([0], [0], color='k', lw=1, label='Baseline'))
    if type == 'boxplot':
        for item in item_params.keys():
            if item != 'control':
                patches.append(mpatches.Patch(
                    color=item_params[item]['color'], label=item_params[item]['name']))
    else:
        raise RuntimeError('Plot type not supported in legend generator.')

    ax[row_idx, col_idx].legend(
        handles=patches, loc='upper right', fontsize=16)


def plot_boxplot_multi(stats_objs_obs, stats_objs_emu, metric, f_path=None, plot_title=True, skip_x_label=True, subplot_ids=['a', 'b']):
    nrows = 1
    ncols = 2
    _, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(
        10 * ncols, 6 * nrows), squeeze=False)

    min_per_row = defaultdict(list)  # mapping from row idx to list of mins
    max_per_row = defaultdict(list)  # mapping from row idx to list of maxs
    for item in stats_objs_obs + stats_objs_emu:
        row_idx = data_fraction_to_row_idx[item['data_fraction']]

        # Calculate min and max for all items
        y_min, y_max = item['stats'][metric].min(), item['stats'][metric].max()

        if metric in ['mae', 'mse', 'rmse']:
            y_min = 0.
        min_per_row[row_idx].append(y_min)
        max_per_row[row_idx].append(y_max)

    min_max_per_row = {}  # mapping from row idx to (y_min, y_max)
    for row_idx in min_per_row.keys():
        min_max_per_row[row_idx] = (
            np.min(min_per_row[row_idx]), np.max(max_per_row[row_idx]))

    for item_obs, item_emu in zip(stats_objs_obs, stats_objs_emu):
        row_idx_obs = data_fraction_to_row_idx[item_obs['data_fraction']]
        row_idx_emu = data_fraction_to_row_idx[item_emu['data_fraction']]

        plot_boxplots(item_obs, axs, 0, row_idx_obs, metric,
                      y_lim=min_max_per_row[row_idx_obs])
        plot_boxplots(item_emu, axs, 1, row_idx_emu, metric,
                      y_lim=min_max_per_row[row_idx_emu])

    if plot_title:
        axs[nrows - 1, 0].set_title("Observation-based training (OBT)",
                                    fontsize=18, fontweight='bold')
        axs[nrows - 1, 1].set_title("Emulation-based training (EBT)",
                                    fontsize=18, fontweight='bold')
        # Legends on top plots only.
        make_legend(axs, nrows - 1, 1, 'boxplot')

    for i, ax in enumerate(axs.flatten()):
        ax.text(-0.05, 1.06, '[' + subplot_ids[i] + ']', transform=ax.transAxes,
                fontsize=16, va='top', ha='right')
        # Show ticks and x labels only on the bottom plot.
        if skip_x_label:
            ax.tick_params(labelbottom=False)
            ax.set_xlabel('')

    xticks_vals = [1, 5, 10]
    plt.setp(axs, xticks=[1, 5, 10], xticklabels=[
             f'{x}x' for x in xticks_vals])

    if f_path:
        plt.tight_layout(pad=3.)
        plt.savefig(f_path)


def load_stats(path_to_data):

    # (copula_name, has_targets) -> stat_name -> inner_name
    tmp = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for i, path in enumerate(path_to_data.glob('*.pkl')):
        copula_name = path.stem.split('-')[1].split('=')[1].capitalize()
        has_targets = path.stem.split('-')[2].split('=')[1]
        with open(path, 'rb') as f:
            stats = pickle.load(f)
        for stat_name in stats:
            for inner in stats[stat_name]:
                tmp[(copula_name, has_targets)][stat_name][inner].append(
                    stats[stat_name][inner])

    copulas = []
    for (copula_name, has_target), stats in tmp.items():
        # Convert list to numpy arrays
        for stat_name, inner_dict in stats.items():
            for inner_name, arr in inner_dict.items():
                inner_dict[inner_name] = np.array(arr)

        copulas.append({
            'name': copula_name,
            'has_targets': has_target,
            'stats': stats
        })

    return copulas


def plot_error_proj(copulas, fig_path):
    stat_names = ['Mean', 'Variance', 'Standard deviation',
                  '0.1-quantile', '0.5-quantile', '0.9-quantile']
    labels = ['Mean', 'Variance', 'Standard deviation',
              '10 % quantile', '50 % quantile', '90 % quantile']
    _, ax = plt.subplots(2, 3, figsize=(16, 12))
    ax = ax.flatten()
    for idx, stat_name in enumerate(stat_names):
        for copula in copulas:
            label = f"{item_params[copula['name'].lower()]['name']} {'($X$, $Y$)' if copula['has_targets'] == '1' else '($X$)'}"
            stat = copula['stats'][stat_name]
            ax[idx].scatter(stat['true'], stat['pred'],
                            marker=marker_styles['has_targets'][copula['has_targets']], s=50, facecolors='none',
                            edgecolors=item_params[copula['name'].lower()]['color'], linewidth=1, label=label)

        ax[idx].set_ylabel(f'Copula generated data {labels[idx].lower()}')
        ax[idx].set_title(f'Errors in {labels[idx].lower()}')
        ax[idx].set_xlabel(f'Real data {labels[idx].lower()}')
        ax[idx].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
        ax[idx].set_box_aspect(1)

    # Manually set limits as we can better control appearance
    lims = [[1.77e6, 2.75e6], [1.2e10, 3.6e10], [1.1e5, 1.9e5],
            [1.65e6, 2.55e6], [1.8e6, 2.85e6], [1.8e6, 2.85e6]]
    for i, lim in enumerate(lims):
        ax[i].set_ylim(lim)
        ax[i].set_xlim(lim)

    ax[0].legend(loc='upper left')

    # Labels
    subplot_ids = list(string.ascii_lowercase)
    for i, ax in enumerate(ax.flatten()):
        ax.text(-0.05, 1.06, '[' + subplot_ids[i] + ']', transform=ax.transAxes,
                fontsize=12, va='top', ha='right')

    if fig_path:
        plt.tight_layout(pad=3.)
        plt.savefig(fig_path)


def get_num_threads():
    num_threads = int(os.environ['num_threads'])
    return num_threads


def plot_random_columns(ds_true, ds_synth, n_columns=10):
    for name in ds_true:
        if ds_true[name].ndim != 2:
            continue
        level = ds_true[name].dims[1]
        _, ax = plt.subplots(1, 2, figsize=(15, 5))
        rnd = np.random.choice(ds_true['column'], n_columns)
        ds_true[name][rnd, :].plot.line(x=level, ax=ax[0])
        ax[0].legend('')
        ax[0].title.set_text('True')
        # Only if the two datasets have not the same size
        # we need to generate new random indices.
        if len(ds_true.column) != len(ds_synth.column):
            rnd = np.random.choice(ds_synth['column'], n_columns)
        ds_synth[name][rnd, :].plot.line(x=level, ax=ax[1])
        ax[1].legend('')
        ax[1].title.set_text('Pred')
        plt.show()
        d1_synthetic = np.mean(np.abs(np.diff(ds_synth[name])))
        d1_true = np.mean(np.abs(np.diff(ds_true[name])))
        d2_synthetic = np.mean(np.abs(np.diff(ds_synth[name], n=2)))
        d2_true = np.mean(np.abs(np.diff(ds_true[name], n=2)))
        print(
            f'First derivatives for synthetic: {d1_synthetic}, true: {d1_true}')
        print(
            f'Second derivatives for synthetic: {d2_synthetic}, true: {d2_true}')
