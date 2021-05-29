from pathlib import Path
import pickle

import numpy as np
import xarray as xr
import pandas as pd
import synthia as syn

import re
from collections import defaultdict

from util import load_ds_inputs
from util import plot_error_proj
from util import plot_boxplot_multi
from util import compute_layer_longwave_downwelling
from util import load_stats

from util import train_test_split_dataset

import statsmodels.api as sm

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import string

import time


def plot_copula_fits(proj_path, plots_path):
    """ Generate scatterplot to evaluate the different copula type fits
    """
    print("Starting")
    t0 = time.time()
    copulas = load_stats(proj_path / 'results' / 'stats')
    print(f"Loading took {time.time() - t0} s")
    plot_error_proj(copulas, plots_path / 'copula_fittings.svg')
    plot_error_proj(copulas, plots_path / 'copula_fittings.pdf')


def plot_copula_ml_fits(proj_path, plots_path, metric, data_fraction, plot_title, skip_x_label, subplot_ids):
    """ Generate boxplots to evaluate the general method to improve ml models
    """
    data_path = proj_path / 'results' / 'ml'
    stats_objs_obs = []
    stats_objs_emu = []

    def load_pkls(fname_filter):
        # load in groups
        # group name -> list of pkl objects
        stats_objs_grouped = defaultdict(list)
        for f_path in data_path.glob(f"*data_fraction={data_fraction}*.pkl"):
            with open(f_path, 'rb') as f:
                if fname_filter not in f_path.name:
                    group_name = re.sub(
                        r'idx_iter_synthetic=\d+', '', f_path.name)
                    stats_objs_grouped[group_name].append(pickle.load(f))
        # merge stats of each group
        stats_objs = []
        for group_name, objs in stats_objs_grouped.items():
            dfs = [obj['stats'] for obj in objs]
            df = pd.concat(dfs)
            obj = objs[0]
            obj['stats'] = df
            stats_objs.append(obj)

        return stats_objs

    stats_objs_obs = load_pkls('has_targets=0')
    stats_objs_emu = load_pkls('has_targets=1')

    plot_boxplot_multi(stats_objs_obs, stats_objs_emu, metric,
                       plots_path / f'copula_ml_fits_{metric}.svg',
                       plot_title=plot_title, skip_x_label=skip_x_label,
                       subplot_ids=subplot_ids)

    plot_boxplot_multi(stats_objs_obs, stats_objs_emu, metric,
                       plots_path / f'copula_ml_fits_{metric}.pdf',
                       plot_title=plot_title, skip_x_label=skip_x_label,
                       subplot_ids=subplot_ids)

def load_best_mse_pkl(fname_filter, proj_path):
    data_path = proj_path / 'results' / 'ml'
    best_pkl = None
    for f_path in data_path.glob(fname_filter):
        with open(f_path, 'rb') as f:
            if best_pkl is None:
                best_pkl = pickle.load(f)
            else:
                pkl = pickle.load(f)
                best_mse = best_pkl['stats']['mse'].iloc[[-1]].values[0]
                mse = pkl['stats']['mse'].iloc[[-1]].values[0]
                if mse < best_mse:
                    best_pkl = pkl
    return best_pkl


def plot_diff_profile(proj_path, plot_path, data_fraction=1.0, batch_size=None, with_banddepth=True):

    ds_true_in = load_ds_inputs(proj_path)
    _, ds_test = train_test_split_dataset(
        ds_true_in, test_size=0.6, dim='column', shuffle=True, seed=42)
    ds_test, _ = train_test_split_dataset(
        ds_test, test_size=0.33334, dim='column', shuffle=True, seed=42)

    if batch_size == None:
        batch_size = len(ds_test.column)
    print(f"Plotting {batch_size} profiles")

    column_gas_optical_depth = 1.7
    y_true = compute_layer_longwave_downwelling(
        ds_test, column_gas_optical_depth)

    fname_filter = f'ml-synth-copula_type=gaussian-has_targets=0-data_fraction={data_fraction}-factor_synthetic=10-idx_iter_synthetic*.pkl'
    ml_copula = load_best_mse_pkl(fname_filter, proj_path)

    fname_filter = f'ml-control-data_fraction={data_fraction}.pkl'
    ml_control = load_best_mse_pkl(fname_filter, proj_path)

    diff_ml_control = y_true - ml_control['y_best']
    diff_ml_copula = y_true - ml_copula['y_best'][0]

    name = 'flux_dn_lw'
    diff_ml_control = diff_ml_control[name]
    diff_ml_copula = diff_ml_copula[name]

    y_lim = find_ylim([diff_ml_control, diff_ml_copula])

    n_rows = 1
    n_cols = 2
    figsize = (n_rows*12, n_cols*2.5)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)
    var_name = ['baseline', 'Gaussian copula']
    for row, item in enumerate([diff_ml_control, diff_ml_copula]):
        print(f'Plotting {var_name[row]}')
        y_label = 'Error in downwelling longwave radiation in W m⁻²' if row == 0 else ''
        plot_lines(item, y_label, ax[row], y_lim, batch_size=batch_size,
                   with_banddepth=with_banddepth, alpha=1)
        ax[row].set_xlabel('Atmospheric level')
        ax[row].set_title(
            f"Predictions from {var_name[row]} case with lowest MAE")

    # Legend
    patches = []
    if with_banddepth:
        patches.append(Line2D([0], [0], color='k', lw=1, label='Median'))
        patches.append(
            Line2D([0], [0], color='#785EF0', lw=1, label=' 0 - 25 %'))
        patches.append(
            Line2D([0], [0], color='#A091E4', lw=1, label='25 - 50 %'))
        patches.append(Line2D([0], [0], color='#E7E2FB',
                              lw=1, label='50 - 100 %'))
    else:
        patches.append(Line2D([0], [0], color='#785EF0',
                              lw=1, label='$i$ᵗʰ profile'))
    ax[1].legend(handles=patches, loc='best')

    # Labels [a,b,c,d,...]
    subplot_ids = list(string.ascii_lowercase)
    for i, ax in enumerate(ax.flat):
        ax.text(0, 1.08, '[' + subplot_ids[i] + ']', transform=ax.transAxes,
                fontsize=12, va='top', ha='right')

    plt.tight_layout()
    fig.savefig(plot_path / f"diff_profiles-batch_size={batch_size}"
                f"-with_badnddepth={with_banddepth}-data_fraction={data_fraction}.png", dpi=300)


def diff_stats(proj_path, data_path, data_fraction=1.0, levels=slice(None, None)):

    ds_true_in = load_ds_inputs(proj_path)
    _, ds_test = train_test_split_dataset(
        ds_true_in, test_size=0.6, dim='column', shuffle=True, seed=42)
    ds_test, _ = train_test_split_dataset(
        ds_test, test_size=0.33334, dim='column', shuffle=True, seed=42)

    column_gas_optical_depth = 1.7
    y_true = compute_layer_longwave_downwelling(
        ds_test, column_gas_optical_depth)

    fname_filter = f'ml-synth-copula_type=gaussian-has_targets=0-data_fraction={data_fraction}-factor_synthetic=10-idx_iter_synthetic*.pkl'
    ml_copula_gaussian = load_best_mse_pkl(fname_filter, proj_path)

    fname_filter = f'ml-synth-copula_type=parametric-has_targets=0-data_fraction={data_fraction}-factor_synthetic=10-idx_iter_synthetic*.pkl'
    ml_copula_parametric = load_best_mse_pkl(fname_filter, proj_path)

    fname_filter = f'ml-synth-copula_type=tll-has_targets=0-data_fraction={data_fraction}-factor_synthetic=10-idx_iter_synthetic*.pkl'
    ml_copula_tll = load_best_mse_pkl(fname_filter, proj_path)

    fname_filter = f'ml-control-data_fraction={data_fraction}.pkl'
    ml_control = load_best_mse_pkl(fname_filter, proj_path)

    var_name = 'flux_dn_lw'
    diff_ml_control = (y_true - ml_control['y_best']).sel(half_level=levels)
    diff_ml_copula_gaussian = (
        y_true - ml_copula_gaussian['y_best'][0]).sel(half_level=levels)
    diff_ml_copula_parametric = (
        y_true - ml_copula_parametric['y_best'][0]).sel(half_level=levels)
    diff_ml_copula_tll = (
        y_true - ml_copula_tll['y_best'][0]).sel(half_level=levels)
    d = []
    for diff, name in zip([diff_ml_control[var_name],
                           diff_ml_copula_gaussian[var_name],
                           diff_ml_copula_parametric[var_name],
                           diff_ml_copula_tll[var_name],
                           ], ['Baseline', 'Gaussian', 'Vine-parametric', 'Vine-nonparametric']):
        d.append({
            'Name': name,
            'MBE': diff.mean().values,
            'MAE': np.abs(diff).mean().values,
            'RMSE': np.sqrt((diff**2).mean()).values
        })

    df = pd.DataFrame(data=d)
    df.to_csv(
        data_path / f'diff_stats_data_fraction={data_fraction}_levels={levels.start}-{levels.stop}.csv')
    return df


def plot_lines(ds, y_label, ax, y_lim, hide_y_label=False, batch_size=None, with_banddepth=False, alpha=0.01):

    def compute_idx_range(idx_arr, min, max, batch_size=None):
        idx_range = idx_arr[int(len(idx_arr) * min): int(len(idx_arr) * max)]
        if batch_size:
            np.random.seed(1)
            idx_range = np.random.choice(idx_range, batch_size)
        return idx_range

    def compute_banddepth_idx(ds, ax):
        res = sm.graphics.fboxplot(ds, ax=ax)
        return res[2]

    if with_banddepth:
        depth_ixs = compute_banddepth_idx(ds, ax=ax)
        ax.clear()
        for i in compute_idx_range(depth_ixs, 0.50, 1, batch_size // 3):
            ds.sel(column=i).plot(c='#E7E2FB', ax=ax, alpha=alpha)
        for i in compute_idx_range(depth_ixs, 0.25, 0.50, batch_size // 3):
            ds.sel(column=i).plot(c='#A091E4', ax=ax, alpha=alpha)
        for i in compute_idx_range(depth_ixs, 0.0, 0.25, batch_size // 3):
            ds.sel(column=i).plot(c='#785EF0', ax=ax, alpha=alpha)
        ds.sel(column=depth_ixs[0]).plot(c='k', ax=ax)
    else:
        for i in np.random.choice(ds.column, batch_size):
            ds.sel(column=i).plot(c='#785EF0', alpha=alpha, ax=ax)

    ax.set_xlabel('')
    ax.set_ylabel(y_label)
    if hide_y_label:
        ax.set_yticklabels([])
        ax.set_ylabel('')
    ax.set_ylim(y_lim)


def find_ylim(ds_list):
    bounds = []
    for ds in ds_list:
        bounds.append([ds.min(), ds.max()])
    return (np.min(bounds), np.max(bounds))


def plot_profiles(proj_path, plot_path, plot_copula, with_banddepth, batch_size, alpha):

    d_names = {
        'temperature_fl': 'Dry-bulb air temperature in K',
        'pressure_hl': 'Atmospheric pressure in hPa',
        'layer_cloud_optical_depth': 'Cloud optical depth',
        'flux_dn_lw': 'Downwelling longwave \n radiation in W m⁻²'
    }

    ds_inputs = load_ds_inputs(proj_path)
    if batch_size == None:
        batch_size = len(ds_inputs.column)

    if plot_copula:
        ds_inputs['pressure_hl'] /= 100  # Convert Pa to hPa for plotting
        # Generate synthetic data using gaussian copula
        generator = syn.CopulaDataGenerator(verbose=True)
        generator.fit(ds_inputs, copula=syn.GaussianCopula(),
                      parameterize_by=None)
        ds_synth = generator.generate(n_samples=len(
            ds_inputs.column), qrng=True, seed=42)
        n_rows = 3
        n_cols = 2
        figsize=(9,8)
    else:
        # Compute fluxes with physical model
        column_gas_optical_depth = 1.7
        flux_dn_hl_train = compute_layer_longwave_downwelling(
            ds_inputs, column_gas_optical_depth)
        ds_inputs = xr.merge([ds_inputs, flux_dn_hl_train])

        ds_inputs['pressure_hl'] /= 100  # Convert Pa to hPa for plotting
        n_rows = 2
        n_cols = 2
        figsize=(9,5.5)

    
    _, ax = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, sharex=True)
    for idx, name in enumerate(ds_inputs):
        print(f'Plotting {name}')
        if plot_copula:
            y_lim = find_ylim([ds_inputs[name], ds_synth[name]])
            print('Plotting source data')
            plot_lines(ds_inputs[name], d_names[name], ax[idx, 0], y_lim,
                       batch_size=batch_size, with_banddepth=with_banddepth, alpha=alpha)
            print('Plotting synthetic data')
            plot_lines(ds_synth[name], d_names[name], ax[idx, 1], y_lim, hide_y_label=True,
                       batch_size=batch_size, with_banddepth=with_banddepth, alpha=alpha)
        else:
            y_lim = find_ylim([ds_inputs[name]])
            print('Plotting source data')
            plot_lines(ds_inputs[name], d_names[name], ax.flat[idx], y_lim,
                       batch_size=batch_size, with_banddepth=with_banddepth, alpha=alpha)

    ax[-1, 0].set_xlabel('Atmospheric level')
    ax[-1, 1].set_xlabel('Atmospheric level')

    # Legend
    patches = []
    if with_banddepth:
        patches.append(Line2D([0], [0], color='k', lw=1, label='Median'))
        patches.append(
            Line2D([0], [0], color='#785EF0', lw=1, label=' 0 - 25 %'))
        patches.append(
            Line2D([0], [0], color='#A091E4', lw=1, label='25 - 50 %'))
        patches.append(Line2D([0], [0], color='#E7E2FB',
                              lw=1, label='50 - 100 %'))
    else:
        patches.append(Line2D([0], [0], color='#785EF0',
                              lw=1, label='$i$ᵗʰ profile'))
    ax[0, 1].legend(handles=patches, loc='best')

    # Labels [a,b,c,d,...]
    subplot_ids = list(string.ascii_lowercase)
    for i, ax in enumerate(ax.flat):
        ax.text(0, 1.15, '[' + subplot_ids[i] + ']', transform=ax.transAxes,
                fontsize=12, va='top', ha='right')

    plt.tight_layout()
    plt.savefig(plot_path / f"plot_profile-plot_copula={plot_copula}"
                f"-with_banddepth={with_banddepth}-batch_size={batch_size}.png", dpi=300)


if __name__ == "__main__":
    PROJ_PATH = Path(__file__).parents[1]
    PLOTS_PATH = PROJ_PATH / 'results' / 'plots'
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    DATA_PATH = PROJ_PATH / 'results' / 'tabular'
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    plot_profiles(PROJ_PATH, PLOTS_PATH,
        plot_copula=False, with_banddepth=True, batch_size=None, alpha=1)
    plot_copula_fits(PROJ_PATH, PLOTS_PATH)
    plot_profiles(PROJ_PATH, PLOTS_PATH,
        plot_copula=True, with_banddepth=True, batch_size=90, alpha=1)
    plot_copula_ml_fits(PROJ_PATH, PLOTS_PATH, 'bias',
        data_fraction=1, plot_title=True, skip_x_label=False, subplot_ids=['a', 'b'])
    plot_diff_profile(PROJ_PATH, PLOTS_PATH,
                      data_fraction=1.0, batch_size=None, with_banddepth=True)
    diff_stats(PROJ_PATH, DATA_PATH,
               data_fraction=1.0, levels=slice(None, None))
